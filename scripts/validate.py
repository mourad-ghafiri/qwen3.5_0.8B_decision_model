"""Validate scenario JSONL files: schema, API limits, probability sums, duplicates, PII, quotas.

Usage:
    python scripts/validate.py data/raw/batch_01*.jsonl            # one batch (used by generator agents)
    python scripts/validate.py data/scenarios.jsonl --report        # full dataset with distribution tables
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevlite.schema import validate_scenario  # noqa: E402

DUP_JACCARD = 0.6
PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@(?!example\.(com|org|net)\b)[\w-]+\.[\w.]+"),
    "card": re.compile(r"\b(?:\d[ -]?){15,16}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


def load(paths: list[str]) -> list[tuple[str, int, dict | None, str | None]]:
    rows = []
    for p in paths:
        for i, line in enumerate(Path(p).read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append((p, i, json.loads(line), None))
            except json.JSONDecodeError as e:
                rows.append((p, i, None, f"invalid JSON: {e}"))
    return rows


def state_text(sc: dict) -> str:
    s = sc["state"]
    return s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)


def shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def near_duplicates(scs: list[dict]) -> list[tuple[str, str, float]]:
    """Pairs of states with 5-gram Jaccard >= DUP_JACCARD. Uses an inverted index so only pairs that share
    n-grams are compared (all-pairs is too slow for 2000 long states)."""
    from collections import defaultdict
    sh = {sc["id"]: shingles(state_text(sc)) for sc in scs}
    index = defaultdict(list)
    for sid, grams in sh.items():
        for g in grams:
            index[g].append(sid)
    candidates = set()
    for ids in index.values():
        if 1 < len(ids) <= 50:  # n-grams shared by >50 states are boilerplate, not evidence of duplication
            candidates.update((ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids)))
    dups = []
    for a, b in candidates:
        jac = len(sh[a] & sh[b]) / len(sh[a] | sh[b])
        if jac >= DUP_JACCARD:
            dups.append((a, b, jac))
    return dups


def pii_hits(sc: dict) -> list[str]:
    text = json.dumps(sc, ensure_ascii=False)
    return [name for name, rx in PII_PATTERNS.items() if rx.search(text)]


COUNT_RE = re.compile(r"\bhow many\b|\bcount the\b|\bnumber of (items|times|entries)\b", re.I)
DATE_RE = re.compile(r"\b(arrive before|before the (deadline|due date|cutoff)|in time|on time|overdue|days? (between|since|late)|"
                     r"how long ago|within \d+ (days|hours)|earlier date|later date)\b", re.I)


def is_structured(q: dict) -> bool:
    crit = q.get("criteria")
    vals = list(crit.values()) if isinstance(crit, dict) else (crit or [])
    return isinstance(q["instructions"], (dict, list)) or any(isinstance(v, (dict, list)) for v in vals)


def words(sc: dict) -> int:
    return len(re.findall(r"\w+", state_text(sc)))


def multi_source(sc: dict) -> bool:
    st = sc["state"]
    return isinstance(st, dict) and sum(1 for v in st.values() if isinstance(v, (str, dict, list)) and v) >= 2


def warnings(scs: list[dict]) -> list[str]:
    out = []
    shapes: dict[str, Counter] = {}
    for sc in scs:
        batch = sc["id"][:3]
        for qid, q in sc["questions"].items():
            ins = q["instructions"]
            if isinstance(ins, dict):
                shapes.setdefault(batch, Counter())[tuple(sorted(ins))] += 1
            text = json.dumps(ins, ensure_ascii=False)
            if COUNT_RE.search(text):
                out.append(f"{sc['id']}.{qid}: counting-like wording: {text[:90]}")
            if DATE_RE.search(text):
                out.append(f"{sc['id']}.{qid}: date/time-comparison wording: {text[:90]}")
    for batch, c in sorted(shapes.items()):
        for keys, n in c.items():
            if n > 8:
                out.append(f"{batch}: instruction key set {keys} used {n}x (formulaic?)")
    return out


def report(scs: list[dict]) -> None:
    qtypes, diffs, formats, domains, npos, noul_bins, nopts = (Counter() for _ in range(7))
    for sc in scs:
        domains[sc["domain"]] += 1
        formats[type(sc["state"]).__name__] += 1
        for qid, q in sc["questions"].items():
            t = sc["targets"][qid]
            qtypes[q["type"]] += 1
            diffs[t["difficulty"]] += 1
            if q["type"] == "noul":
                p = t["noul"]
                noul_bins["<0.3" if p < 0.3 else ">0.7" if p > 0.7 else "0.3-0.7"] += 1
            else:
                probs = t["probabilities"]
                keys = list(probs) if q["type"] == "score" else list(q["criteria"])
                top = max(keys, key=lambda k: probs[k])
                npos[f"{keys.index(top)}/{len(keys)}" if q["type"] == "score" else keys.index(top)] += 1
                nopts[len(keys)] += 1
    total_q = sum(qtypes.values())

    def show(title: str, c: Counter, denom: int) -> None:
        print(f"\n{title}")
        for k, v in sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0]))):
            print(f"  {str(k):<28} {v:>5}  {100 * v / denom:5.1f}%")

    print(f"\nscenarios: {len(scs)}   questions: {total_q}")
    show("question types", qtypes, total_q)
    show("difficulty", diffs, total_q)
    show("state format", formats, len(scs))
    show("domains", domains, len(scs))
    show("noul target bins", noul_bins, max(1, qtypes["noul"]))
    show("choice winner position (0-based)", Counter({k: v for k, v in npos.items() if isinstance(k, int)}),
         max(1, qtypes["choice"]))
    show("choice/score option counts", nopts, max(1, qtypes["choice"] + qtypes["score"]))

    allq = [(sc, q) for sc in scs for q in sc["questions"].values()]
    choice_n = [len(q["criteria"]) for _, q in allq if q["type"] == "choice"]
    score_n = [len(q["criteria"]) for _, q in allq if q["type"] == "score"]
    print("\ncomplexity")
    rows = [
        ("structured questions", sum(is_structured(q) for _, q in allq), len(allq)),
        ("backtick-path questions", sum("`" in json.dumps(q["instructions"]) for _, q in allq), len(allq)),
        ("multi-source states", sum(multi_source(sc) for sc in scs), len(scs)),
        ("states >= 250 words", sum(words(sc) >= 250 for sc in scs), len(scs)),
        ("states >= 400 words", sum(words(sc) >= 400 for sc in scs), len(scs)),
        ("choices with 2 options", sum(n == 2 for n in choice_n), len(choice_n)),
        ("choices with 9-52 options", sum(n >= 9 for n in choice_n), len(choice_n)),
        ("scores with 5-10 levels", sum(n >= 5 for n in score_n), len(score_n)),
    ]
    for name, k, n in rows:
        print(f"  {name:<28} {k:>5}  {100 * k / max(1, n):5.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--no-dups", action="store_true", help="skip O(n^2) duplicate scan")
    args = ap.parse_args()

    rows = load(args.paths)
    errors: list[str] = []
    good: list[dict] = []
    seen: set[str] = set()
    for path, line, sc, err in rows:
        where = f"{Path(path).name}:{line}"
        if err:
            errors.append(f"{where}: {err}")
            continue
        errs = validate_scenario(sc)
        if not errs and sc["id"] in seen:
            errs = [f"{sc['id']}: duplicate id"]
        if not errs and (hits := pii_hits(sc)):
            errs = [f"{sc['id']}: possible real PII ({', '.join(hits)}); use example.com / synthetic values"]
        if errs:
            errors += [f"{where}: {e}" for e in errs]
        else:
            seen.add(sc["id"])
            good.append(sc)

    if not args.no_dups:
        for a, b, j in near_duplicates(good):
            errors.append(f"near-duplicate states: {a} ~ {b} (jaccard {j:.2f})")

    if args.report and good:
        report(good)
    warns = warnings(good)
    if warns:
        print(f"\n{len(warns)} warnings (review; not errors)")
        for w in warns[:80]:
            print("  WARN", w)
    print(f"\n{len(good)} valid scenarios, {len(errors)} errors")
    for e in errors[:200]:
        print("  ERROR", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
