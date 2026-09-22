"""Evaluate one or more models on the test split and write reports/eval.md.

    python scripts/evaluate.py models/Qwen3.5-0.8B-Base runs/main/best

Targets are soft expert-agreement distributions, so:
- accuracy       = predicted argmax equals the target argmax
- log-loss       = cross-entropy against the target distribution
- Brier          = squared error between the distributions
- ECE (10 bins)  = |mean predicted top prob - mean target prob of the predicted option| (Choice/Score),
                   and |mean predicted p(yes) - mean target p(yes)| (Noul)
- coverage       = share of Choice/Score answers with confidence >= tau, and accuracy on that share
- permutation    = mean total-variation distance of Choice probabilities under a shuffled option order
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevlite.model import SystemOne  # noqa: E402
from jevlite.schema import confidence, option_keys, target_distribution  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TAUS = (0.5, 0.7, 0.9)


def ece(pairs: list[tuple[float, float]], bins: int = 10) -> float:
    buckets = defaultdict(list)
    for pred, obs in pairs:
        buckets[min(bins - 1, int(pred * bins))].append((pred, obs))
    n = len(pairs)
    return sum(len(b) / n * abs(statistics.fmean(p for p, _ in b) - statistics.fmean(o for _, o in b))
               for b in buckets.values())


def question_metrics(kind: str, p: list[float], t: list[float]) -> dict:
    """Per-question numbers for a predicted distribution `p` against target `t` (both aligned with option_keys)."""
    k = max(range(len(p)), key=p.__getitem__)
    tk = max(range(len(t)), key=t.__getitem__)
    out = {"correct": k == tk,
           "logloss": -sum(ti * math.log(max(pi, 1e-9)) for pi, ti in zip(p, t)),
           "brier": sum((pi - ti) ** 2 for pi, ti in zip(p, t)),
           "cal": (p[0], t[0]) if kind == "noul" else (p[k], t[k])}
    if kind != "noul":
        out["confidence"] = confidence(p)
    if kind == "score":
        out["abs_err"] = abs(sum(i * x for i, x in enumerate(p)) - sum(i * x for i, x in enumerate(t)))
    return out


def metrics_by_type(records: list[tuple[str, list[float], list[float]]]) -> dict:
    """Aggregate accuracy, log-loss, Brier, ECE, score MAE and confidence coverage per question type."""
    m = defaultdict(lambda: defaultdict(list))
    for kind, p, t in records:
        q = question_metrics(kind, p, t)
        m[kind]["acc"].append(q["correct"])
        m[kind]["logloss"].append(q["logloss"])
        m[kind]["brier"].append(q["brier"])
        m[kind]["cal"].append(q["cal"])
        if "confidence" in q:
            m[kind]["conf"].append((q["confidence"], q["correct"]))
        if "abs_err" in q:
            m[kind]["mae"].append(q["abs_err"])
    out = {}
    for kind, d in m.items():
        row = {"n": len(d["acc"]), "acc": statistics.fmean(d["acc"]), "logloss": statistics.fmean(d["logloss"]),
               "brier": statistics.fmean(d["brier"]), "ece": ece(d["cal"])}
        if d["mae"]:
            row["score_mae"] = statistics.fmean(d["mae"])
        for tau in TAUS:
            if d["conf"]:
                sel = [ok for c, ok in d["conf"] if c >= tau]
                row[f"cov@{tau}"] = len(sel) / len(d["conf"])
                row[f"acc@{tau}"] = statistics.fmean(sel) if sel else float("nan")
        out[kind] = row
    return out


def evaluate(path: str, scs: list[dict], rng: random.Random) -> dict:
    eng = SystemOne(path)
    items, targets, types = [], [], []
    for sc in scs:
        for qid, q in sc["questions"].items():
            items.append((sc["state"], q, None))
            targets.append(target_distribution(q, sc["targets"][qid]))
            types.append(q["type"])
    probs, _ = eng.distributions(items)

    tvds = []
    for sc in scs:
        for q in sc["questions"].values():
            if q["type"] != "choice":
                continue
            keys = option_keys(q)
            order = keys[:]
            rng.shuffle(order)
            base = eng.permuted_choice(sc["state"], q, keys)
            perm = eng.permuted_choice(sc["state"], q, order)
            tvds.append(0.5 * sum(abs(a - b) for a, b in zip(base, perm)))

    lat = sorted(eng.timed(sc["state"], sc["questions"])[1] for sc in scs[:40])
    return {"model": path, "by_type": metrics_by_type(list(zip(types, probs, targets))),
            "perm_tvd": statistics.fmean(tvds) if tvds else float("nan"),
            "latency_p50_ms": 1000 * lat[len(lat) // 2], "latency_p95_ms": 1000 * lat[int(0.95 * (len(lat) - 1))]}


def to_markdown(results: list[dict]) -> str:
    lines = ["# Evaluation (test split)", "",
             "Targets are expert-agreement distributions written by Claude, not real-world outcomes; "
             "small test sets make ECE noisy.", ""]
    for r in results:
        title = "base: " + r["model"] if r["model"].startswith("Qwen/") else "fine-tuned: " + r["model"]
        lines += [f"## {title}", "",
                  f"- permutation TVD (choice): {r['perm_tvd']:.4f}",
                  f"- latency per request: p50 {r['latency_p50_ms']:.0f} ms, p95 {r['latency_p95_ms']:.0f} ms", "",
                  "| type | n | acc | log-loss | Brier | ECE | score MAE | cov/acc @0.5 | cov/acc @0.7 | cov/acc @0.9 |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for kind, d in sorted(r["by_type"].items()):
            cov = " | ".join(f"{d[f'cov@{t}']:.2f} / {d[f'acc@{t}']:.2f}" if f"cov@{t}" in d else "–" for t in TAUS)
            lines.append(f"| {kind} | {d['n']} | {d['acc']:.3f} | {d['logloss']:.3f} | {d['brier']:.3f} | "
                         f"{d['ece']:.3f} | {d.get('score_mae', float('nan')):.3f} | {cov} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    scs = [json.loads(l) for l in (ROOT / f"data/splits/{args.split}.jsonl").read_text().splitlines() if l.strip()]
    results = [evaluate(p, scs, random.Random(0)) for p in args.models]
    out = ROOT / "reports/eval.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(to_markdown(results))
    (ROOT / "reports/eval.json").write_text(json.dumps(results, indent=2))
    print(out.read_text())


if __name__ == "__main__":
    main()
