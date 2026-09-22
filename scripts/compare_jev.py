"""Compare TypeSafe's Jev (real API) with our model on scenarios never used in training.

    export JEV_MODEL_API_KEY=...        # your TypeSafe API key (read from the environment, never stored)
    .venv/bin/python scripts/compare_jev.py --limit 5            # smoke test: 5 scenarios
    .venv/bin/python scripts/compare_jev.py                      # full test split (150 scenarios)

Each scenario goes to both models as ONE request: the same `state` and the same `questions` map, exactly the
`POST /v1/systemone` shape from https://docs.typesafe.ai/api.md. Both answers are scored against our held-out
soft targets with the same metrics as scripts/evaluate.py.

Outputs:
    results/compare_jev.json           summary, per-type / per-domain / per-difficulty tables, head-to-head,
                                       and every question with both models' distributions
    results/compare_jev.md             human-readable summary
    results/jev_cache_<split>.jsonl    raw Jev responses. A re-run reuses them, so you never pay twice.

The test split is strictly unseen: it was never used for training, checkpoint selection or calibration. The
validation split (`--split val`) was used to pick the best epoch and fit calibration temperatures.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import ssl
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import metrics_by_type, question_metrics  # noqa: E402
from jevlite.schema import option_keys, target_distribution  # noqa: E402

JEV_URL = "https://api.typesafe.ai/v1/systemone"   # https://docs.typesafe.ai/api.md
JEV_PRICE_PER_MTOK = 0.042                          # USD per million input tokens; https://docs.typesafe.ai/models.md
KEY_ENV = "JEV_MODEL_API_KEY"
RETRYABLE = {429, 500, 502, 503, 504, 529}


# ---------------------------------------------------------------------------------------------- Jev client

class JevError(Exception):
    pass


def tls_context() -> ssl.SSLContext:
    """Verified TLS using certifi's CA bundle. The python.org macOS build doesn't read the system keychain, so
    the default context fails with CERTIFICATE_VERIFY_FAILED. Verification is never disabled."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def call_jev(key: str, model: str, state, questions: dict, timeout: float, max_attempts: int = 6) -> tuple[dict, float]:
    """One POST /v1/systemone. Retries 429/529/5xx and network errors with exponential backoff."""
    body = json.dumps({"state": state, "model": model, "questions": questions}).encode()
    context = tls_context()
    delay = 1.0
    for attempt in range(1, max_attempts + 1):
        req = urllib.request.Request(JEV_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": "jevlite-compare/1.0"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
                return json.loads(resp.read()), time.perf_counter() - started
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            if e.code == 401:
                raise SystemExit(f"Jev returned 401 Unauthorized: check {KEY_ENV}.")
            if e.code in RETRYABLE and attempt < max_attempts:
                wait = float(e.headers.get("retry-after") or delay)
                print(f"    Jev HTTP {e.code}, retrying in {wait:.0f}s ({attempt}/{max_attempts})", flush=True)
                time.sleep(wait)
                delay = min(delay * 2, 30)
                continue
            raise JevError(f"HTTP {e.code}: {detail}")
        except (urllib.error.URLError, TimeoutError) as e:
            if isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError):
                raise SystemExit(f"TLS certificate verification failed ({e.reason}). Install the CA bundle with "
                                 "`.venv/bin/pip install certifi` and run again with .venv/bin/python.")
            if attempt < max_attempts:
                print(f"    Jev network error ({e}), retrying in {delay:.0f}s", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise JevError(f"network error: {e}")
    raise JevError("retries exhausted")


def load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["id"]: r for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())}


# ---------------------------------------------------------------------------------------------- scoring

def to_distribution(question: dict, answer: dict | None) -> list[float] | None:
    """A Jev-shaped answer as a probability vector aligned with option_keys(question); None if unusable."""
    if not answer or answer.get("type") != question["type"]:
        return None
    if question["type"] == "noul":
        p = answer.get("noul")
        return None if not isinstance(p, (int, float)) else [float(p), 1.0 - float(p)]
    probs = answer.get("probabilities") or {}
    vec = [float(probs.get(k, 0.0)) for k in option_keys(question)]
    total = sum(vec)
    return None if total <= 0 else [v / total for v in vec]


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))] if s else float("nan")


def paired_bootstrap(diffs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval for the mean of paired differences."""
    if not diffs:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def grouped(rows: list[dict], key: str) -> dict:
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    for g, rs in sorted(groups.items()):
        out[g] = {"n": len(rs),
                  "jev_acc": statistics.fmean(r["jev"]["correct"] for r in rs),
                  "ours_acc": statistics.fmean(r["ours"]["correct"] for r in rs),
                  "jev_logloss": statistics.fmean(r["jev"]["logloss"] for r in rs),
                  "ours_logloss": statistics.fmean(r["ours"]["logloss"] for r in rs)}
    return out


def clean(x):
    """JSON-safe: NaN/inf -> None, rounded floats."""
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else round(x, 5)
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


# ---------------------------------------------------------------------------------------------- report

def to_markdown(rep: dict) -> str:
    m, s, h = rep["meta"], rep["summary"], rep["head_to_head"]
    f = lambda v, d=3: "–" if v is None else f"{v:.{d}f}"  # noqa: E731
    lines = [
        "# Jev vs our model on unseen scenarios", "",
        f"- Split: **{m['split']}**, {m['scenarios_scored']} scenarios, {m['questions_scored']} questions scored by both "
        f"({m['jev_failed_scenarios']} Jev failures, {m['ours_failed_scenarios']} of ours)",
        f"- Jev model: `{m['jev_model_reported']}` · our model: `{m['our_model']}` on `{m['our_device']}`",
        f"- Jev input tokens: {m['jev_input_tokens']:,} (estimated ${m['jev_estimated_cost_usd']:.4f})",
        "- The targets are our held-out expert-agreement labels. See the caveats in docs/compare_jev.md.", "",
        "## Overall", "",
        "| | Jev | ours |", "|---|---|---|",
        f"| accuracy | {f(s['jev']['overall']['acc'])} | {f(s['ours']['overall']['acc'])} |",
        f"| log-loss (lower is better) | {f(s['jev']['overall']['logloss'])} | {f(s['ours']['overall']['logloss'])} |",
        f"| Brier (lower is better) | {f(s['jev']['overall']['brier'])} | {f(s['ours']['overall']['brier'])} |",
        f"| latency p50 / p95, ms per request | {f(s['jev']['latency_p50_ms'], 0)} / {f(s['jev']['latency_p95_ms'], 0)} (incl. network) | "
        f"{f(s['ours']['latency_p50_ms'], 0)} / {f(s['ours']['latency_p95_ms'], 0)} (local) |", "",
        "## By question type", "",
        "| type | n | accuracy Jev / ours | log-loss Jev / ours | Brier Jev / ours | ECE Jev / ours |",
        "|---|---|---|---|---|---|"]
    for kind in sorted(s["ours"]["by_type"]):
        j, o = s["jev"]["by_type"].get(kind, {}), s["ours"]["by_type"][kind]
        lines.append(f"| {kind} | {o['n']} | {f(j.get('acc'))} / {f(o['acc'])} | {f(j.get('logloss'))} / {f(o['logloss'])} | "
                     f"{f(j.get('brier'))} / {f(o['brier'])} | {f(j.get('ece'))} / {f(o['ece'])} |")
    lo, hi = h["logloss_diff_ci95"]
    lines += ["", "## Head to head (same questions)", "",
              f"- Same top answer: {100 * h['agreement']:.1f}%",
              f"- Only ours correct: {h['only_ours_correct']} · only Jev correct: {h['only_jev_correct']} · "
              f"both: {h['both_correct']} · neither: {h['neither_correct']}",
              f"- Lower log-loss per question: ours {h['ours_lower_logloss']} · Jev {h['jev_lower_logloss']}",
              f"- Mean log-loss difference (Jev − ours): {f(h['logloss_diff_mean'], 4)}, 95% CI [{f(lo, 4)}, {f(hi, 4)}]. "
              "A positive value means ours is better.", "",
              "## By difficulty", "", "| difficulty | n | accuracy Jev / ours | log-loss Jev / ours |", "|---|---|---|---|"]
    for g, r in rep["by_difficulty"].items():
        lines.append(f"| {g} | {r['n']} | {f(r['jev_acc'])} / {f(r['ours_acc'])} | {f(r['jev_logloss'])} / {f(r['ours_logloss'])} |")
    lines += ["", "## By domain", "", "| domain | n | accuracy Jev / ours | log-loss Jev / ours |", "|---|---|---|---|"]
    for g, r in rep["by_domain"].items():
        lines.append(f"| {g} | {r['n']} | {f(r['jev_acc'])} / {f(r['ours_acc'])} | {f(r['jev_logloss'])} / {f(r['ours_logloss'])} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--our-model", default=str(ROOT / "models/decision-model"), help="local folder of our model")
    ap.add_argument("--jev-model", default="jev-latest")
    ap.add_argument("--limit", type=int, default=0, help="only the first N scenarios (0 = all)")
    ap.add_argument("--max-input-tokens", type=int, default=5_000_000,
                    help="stop sending to Jev after this many billed input tokens (cost guard)")
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds per Jev request")
    ap.add_argument("--out", default=str(ROOT / "results/compare_jev.json"))
    ap.add_argument("--dry-run", action="store_true", help="print the first Jev request and exit (no key, no calls)")
    args = ap.parse_args()

    scs = [json.loads(l) for l in (ROOT / f"data/splits/{args.split}.jsonl").read_text().splitlines() if l.strip()]
    if args.limit:
        scs = scs[: args.limit]
    n_q = sum(len(sc["questions"]) for sc in scs)
    print(f"{args.split} split: {len(scs)} scenarios, {n_q} questions")

    if args.dry_run:
        sc = scs[0]
        print(json.dumps({"state": sc["state"], "model": args.jev_model, "questions": sc["questions"]}, indent=2)[:4000])
        return

    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise SystemExit(f"{KEY_ENV} is not set. Run:  read -s {KEY_ENV} && export {KEY_ENV}")
    if not Path(args.our_model, "config.json").exists():
        raise SystemExit(f"our model not found at {args.our_model}. Download it first (see docs/compare_jev.md).")

    # ---- Jev (cached, resumable)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = out_path.parent / f"jev_cache_{args.split}.jsonl"
    cache = load_cache(cache_path)
    jev_tokens = sum((r.get("response") or {}).get("usage", {}).get("input_tokens", 0) for r in cache.values())
    jev_failed: dict[str, str] = {}
    with cache_path.open("a") as cache_file:
        for i, sc in enumerate(scs, 1):
            if sc["id"] in cache:
                continue
            if jev_tokens >= args.max_input_tokens:
                print(f"cost guard: {jev_tokens:,} input tokens reached --max-input-tokens; not sending more.")
                break
            try:
                resp, secs = call_jev(key, args.jev_model, sc["state"], sc["questions"], args.timeout)
            except JevError as e:
                jev_failed[sc["id"]] = str(e)
                print(f"  [{i}/{len(scs)}] {sc['id']}: Jev error {e}", flush=True)
                continue
            rec = {"id": sc["id"], "latency_ms": round(1000 * secs, 1), "response": resp}
            cache[sc["id"]] = rec
            cache_file.write(json.dumps(rec) + "\n")
            cache_file.flush()
            jev_tokens += resp.get("usage", {}).get("input_tokens", 0)
            if i % 10 == 0 or i == len(scs):
                print(f"  Jev [{i}/{len(scs)}] input tokens so far {jev_tokens:,}", flush=True)

    # ---- our model (local)
    from jevlite.model import SystemOne
    engine = SystemOne(args.our_model)
    print(f"our model: {engine.name} on {engine.device}")
    ours: dict[str, dict] = {}
    ours_failed: dict[str, str] = {}
    for i, sc in enumerate(scs, 1):
        try:
            resp, secs = engine.timed(sc["state"], sc["questions"])
            ours[sc["id"]] = {"latency_ms": round(1000 * secs, 1), "response": resp}
        except Exception as e:  # noqa: BLE001 - record and continue
            ours_failed[sc["id"]] = f"{type(e).__name__}: {e}"
        if i % 25 == 0 or i == len(scs):
            print(f"  ours [{i}/{len(scs)}]", flush=True)

    # ---- score both on the same questions
    rows, jev_rec, ours_rec = [], [], []
    for sc in scs:
        j, o = cache.get(sc["id"]), ours.get(sc["id"])
        if not j or not o:
            continue
        for qid, q in sc["questions"].items():
            t = target_distribution(q, sc["targets"][qid])
            pj = to_distribution(q, j["response"].get("answers", {}).get(qid))
            po = to_distribution(q, o["response"]["answers"].get(qid))
            if pj is None or po is None:
                continue
            mj, mo = question_metrics(q["type"], pj, t), question_metrics(q["type"], po, t)
            jev_rec.append((q["type"], pj, t))
            ours_rec.append((q["type"], po, t))
            rows.append({"id": sc["id"], "qid": qid, "domain": sc["domain"], "type": q["type"],
                         "difficulty": sc["targets"][qid]["difficulty"], "keys": option_keys(q), "target": t,
                         "jev": {"dist": pj, "correct": mj["correct"], "logloss": mj["logloss"]},
                         "ours": {"dist": po, "correct": mo["correct"], "logloss": mo["logloss"]}})
    if not rows:
        raise SystemExit("no questions were answered by both models; see the errors above.")

    def overall(recs):
        ms = [question_metrics(k, p, t) for k, p, t in recs]
        return {"n": len(ms), "acc": statistics.fmean(m["correct"] for m in ms),
                "logloss": statistics.fmean(m["logloss"] for m in ms), "brier": statistics.fmean(m["brier"] for m in ms)}

    jev_lat = [cache[sc["id"]]["latency_ms"] for sc in scs if sc["id"] in cache]
    our_lat = [ours[sc["id"]]["latency_ms"] for sc in scs if sc["id"] in ours]
    diffs = [r["jev"]["logloss"] - r["ours"]["logloss"] for r in rows]
    reported = sorted({cache[sc["id"]]["response"].get("model", "?") for sc in scs if sc["id"] in cache})
    report = {
        "meta": {"split": args.split, "scenarios_requested": len(scs),
                 "scenarios_scored": len({r["id"] for r in rows}), "questions_scored": len(rows),
                 "jev_model_requested": args.jev_model, "jev_model_reported": ", ".join(reported),
                 "our_model": engine.name, "our_device": engine.device,
                 "our_calibration_temperatures": engine.temperature,
                 "jev_input_tokens": jev_tokens,
                 "jev_estimated_cost_usd": jev_tokens / 1e6 * JEV_PRICE_PER_MTOK,
                 "jev_failed_scenarios": len(jev_failed), "ours_failed_scenarios": len(ours_failed),
                 "errors": {"jev": jev_failed, "ours": ours_failed},
                 "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
        "summary": {
            "jev": {"overall": overall(jev_rec), "by_type": metrics_by_type(jev_rec),
                    "latency_p50_ms": pct(jev_lat, 0.5), "latency_p95_ms": pct(jev_lat, 0.95)},
            "ours": {"overall": overall(ours_rec), "by_type": metrics_by_type(ours_rec),
                     "latency_p50_ms": pct(our_lat, 0.5), "latency_p95_ms": pct(our_lat, 0.95)}},
        "head_to_head": {
            "agreement": statistics.fmean(
                max(range(len(r["jev"]["dist"])), key=r["jev"]["dist"].__getitem__)
                == max(range(len(r["ours"]["dist"])), key=r["ours"]["dist"].__getitem__) for r in rows),
            "both_correct": sum(r["jev"]["correct"] and r["ours"]["correct"] for r in rows),
            "only_ours_correct": sum(r["ours"]["correct"] and not r["jev"]["correct"] for r in rows),
            "only_jev_correct": sum(r["jev"]["correct"] and not r["ours"]["correct"] for r in rows),
            "neither_correct": sum(not r["jev"]["correct"] and not r["ours"]["correct"] for r in rows),
            "ours_lower_logloss": sum(d > 0 for d in diffs), "jev_lower_logloss": sum(d < 0 for d in diffs),
            "logloss_diff_mean": statistics.fmean(diffs), "logloss_diff_ci95": paired_bootstrap(diffs)},
        "by_type_accuracy_note": "accuracy = top option equals the target's top option",
        "by_difficulty": grouped(rows, "difficulty"),
        "by_domain": grouped(rows, "domain"),
        "questions": rows,
    }
    report = clean(report)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    md = to_markdown(report)
    out_path.with_suffix(".md").write_text(md)
    print(md)
    print(f"wrote {out_path} and {out_path.with_suffix('.md')}")


if __name__ == "__main__":
    main()
