"""Split data/scenarios.jsonl into train/val/test by scenario, stratified by domain.

    python scripts/split.py --val 150 --test 150
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=int, default=150)
    ap.add_argument("--test", type=int, default=150)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    scs = [json.loads(l) for l in (ROOT / "data/scenarios.jsonl").read_text().splitlines() if l.strip()]
    rng = random.Random(args.seed)
    by_domain = defaultdict(list)
    for sc in scs:
        by_domain[sc["domain"]].append(sc)
    for group in by_domain.values():
        rng.shuffle(group)

    # Round-robin across domains so val/test get proportional coverage.
    interleaved = []
    groups = sorted(by_domain.values(), key=len, reverse=True)
    while any(groups):
        for g in groups:
            if g:
                interleaved.append(g.pop())
    test, val, train = (interleaved[: args.test], interleaved[args.test: args.test + args.val],
                        interleaved[args.test + args.val:])
    out = ROOT / "data/splits"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        print(f"{name}: {len(rows)} scenarios, {sum(len(r['questions']) for r in rows)} questions")


if __name__ == "__main__":
    main()
