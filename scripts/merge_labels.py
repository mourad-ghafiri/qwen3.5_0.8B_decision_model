"""Blind second-annotator workflow.

    prepare:  data/raw/*.jsonl -> data/relabel_inputs/part_N.jsonl   (targets and notes stripped)
    merge:    raw + data/relabel/*.jsonl (+ data/review/adjudicated.jsonl) -> data/scenarios.jsonl
              and data/review/disagreements.jsonl (items that need adjudication)

Relabel file format, one line per scenario:
    {"id": "b01-0001", "targets": {"qid": {"probabilities": {...}} | {"noul": p}, ...}}

Merge rule per question:
    - top option agrees -> final = 0.6 * author + 0.4 * reviewer
    - disagrees         -> goes to disagreements.jsonl. An adjudicated.jsonl line {"id", "targets": {...},
                           "drop": [qid, ...]} resolves it. Unresolved items are dropped from the
                           question set, and a scenario left with < 3 questions is dropped.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevlite.schema import option_keys, target_distribution, validate_scenario  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AUTHOR_WEIGHT = 0.6


def read_jsonl(pattern: str) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(str(ROOT / pattern))):
        rows += [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def batch_range(spec: str | None) -> set[str] | None:
    """'b09-b16' -> {'b09', ..., 'b16'}; None -> all batches."""
    if not spec:
        return None
    lo, hi = (int(x.strip().lstrip("b")) for x in spec.split("-"))
    return {f"b{i:02d}" for i in range(lo, hi + 1)}


def prepare(parts: int, batches: str | None, first_part: int) -> None:
    keep = batch_range(batches)
    raw = [s for s in read_jsonl("data/raw/*.jsonl") if keep is None or s["id"][:3] in keep]
    blind = [{"id": s["id"], "domain": s["domain"], "state": s["state"], "questions": s["questions"]} for s in raw]
    size = -(-len(blind) // parts)
    for i in range(parts):
        write_jsonl(ROOT / f"data/relabel_inputs/part_{first_part + i}.jsonl", blind[i * size:(i + 1) * size])
    print(f"wrote parts {first_part}..{first_part + parts - 1}, {len(blind)} scenarios")


def as_target(q: dict, dist: list[float], base: dict) -> dict:
    out = {k: v for k, v in base.items() if k not in ("probabilities", "noul")}
    if q["type"] == "noul":
        out["noul"] = round(dist[0], 3)
    else:
        rounded = [round(p, 3) for p in dist]
        rounded[rounded.index(max(rounded))] += round(1.0 - sum(rounded), 3)  # keep the exact sum
        out["probabilities"] = dict(zip(option_keys(q), rounded))
    return out


def argmax(dist: list[float]) -> int:
    return max(range(len(dist)), key=dist.__getitem__)


def agree(q: dict, a: list[float], b: list[float]) -> bool:
    if q["type"] == "noul":  # agree on side of 0.5, or both uncertain
        return (a[0] - 0.5) * (b[0] - 0.5) > 0 or (abs(a[0] - 0.5) <= 0.15 and abs(b[0] - 0.5) <= 0.15)
    return argmax(a) == argmax(b)


def merge() -> None:
    raw = {s["id"]: s for s in read_jsonl("data/raw/*.jsonl")}
    relabel = {r["id"]: r["targets"] for r in read_jsonl("data/relabel/*.jsonl")}
    adjud: dict[str, dict] = {}
    for r in read_jsonl("data/review/adjudicated*.jsonl"):  # several files may resolve the same scenario
        entry = adjud.setdefault(r["id"], {"id": r["id"], "targets": {}, "drop": []})
        entry["targets"].update(r.get("targets", {}))
        entry["drop"] += r.get("drop", [])
    stats = defaultdict(lambda: [0, 0])
    out, disagreements, dropped = [], [], []
    for sid, sc in raw.items():
        sc = copy.deepcopy(sc)
        rev = relabel.get(sid)
        if rev is None:
            print(f"warning: {sid} has no relabel; keeping author targets")
        keep = {}
        for qid, q in sc["questions"].items():
            tgt = sc["targets"][qid]
            a = target_distribution(q, tgt)
            b = None
            if rev is not None and qid in rev:
                try:
                    b = target_distribution(q, rev[qid])
                except (KeyError, TypeError):
                    b = None
            ok = b is not None and agree(q, a, b)
            if b is not None:  # raw author-vs-reviewer agreement, counted before any adjudication
                key = (q["type"], tgt["difficulty"])
                stats[key][0] += ok
                stats[key][1] += 1
            adj = adjud.get(sid, {})
            if qid in adj.get("drop", []):
                continue
            if qid in adj.get("targets", {}):
                keep[qid] = as_target(q, target_distribution(q, adj["targets"][qid]), tgt)
                continue
            if b is None:
                keep[qid] = tgt
                continue
            if ok:
                keep[qid] = as_target(q, [AUTHOR_WEIGHT * x + (1 - AUTHOR_WEIGHT) * y for x, y in zip(a, b)], tgt)
            else:
                disagreements.append({"id": sid, "qid": qid, "state": sc["state"], "question": q,
                                      "author": tgt, "reviewer": rev[qid]})
        if len(keep) < 3:
            dropped.append(sid)
            continue
        sc["questions"] = {q: sc["questions"][q] for q in keep}
        sc["targets"] = keep
        errs = validate_scenario(sc)
        if errs:
            print("invalid after merge:", errs[:3])
            dropped.append(sid)
            continue
        out.append(sc)

    write_jsonl(ROOT / "data/scenarios.jsonl", out)
    write_jsonl(ROOT / "data/review/disagreements.jsonl", disagreements)
    tot_ok = sum(v[0] for v in stats.values())
    tot = sum(v[1] for v in stats.values())
    print(f"scenarios kept: {len(out)}  dropped: {len(dropped)}  pending disagreements: {len(disagreements)}")
    print(f"overall agreement: {tot_ok}/{tot} = {tot_ok / max(1, tot):.1%}")
    for (t, d), (ok, n) in sorted(stats.items()):
        print(f"  {t:<7} {d:<13} {ok:>4}/{n:<4} {ok / n:6.1%}")


def check(part: int) -> None:
    """Verify that data/relabel/part_<part>*.jsonl covers every question of relabel_inputs/part_<part>.jsonl."""
    from jevlite.schema import validate_target
    inputs = {s["id"]: s for s in read_jsonl(f"data/relabel_inputs/part_{part}.jsonl")}
    got = {r["id"]: r for r in read_jsonl(f"data/relabel/part_{part}*.jsonl")}
    errors = [f"{sid}: missing" for sid in inputs if sid not in got]
    errors += [f"{sid}: not in part {part} inputs" for sid in got if sid not in inputs]
    for sid, r in got.items():
        if sid not in inputs:
            continue
        qs = inputs[sid]["questions"]
        tg = r.get("targets", {})
        if set(tg) != set(qs):
            errors.append(f"{sid}: target keys {sorted(tg)} != questions {sorted(qs)}")
            continue
        for qid, q in qs.items():
            full = {"difficulty": "clear", "note": "-", **tg[qid]}  # reviewers give only the distribution
            errors += [f"{sid}: {e}" for e in validate_target(qid, q, full)]
    print(f"part {part}: {len(got)}/{len(inputs)} scenarios labelled, {len(errors)} errors")
    for e in errors[:100]:
        print("  ERROR", e)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--parts", type=int, default=4)
    p.add_argument("--batches", help="e.g. b09-b16 (default: all)")
    p.add_argument("--first-part", type=int, default=1)
    sub.add_parser("merge")
    c = sub.add_parser("check")
    c.add_argument("part", type=int)
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args.parts, args.batches, args.first_part)
    elif args.cmd == "check":
        check(args.part)
    else:
        merge()


if __name__ == "__main__":
    main()
