"""Fit one softmax temperature per primitive on the val split and write <model>/calibration.json.

    python scripts/calibrate.py runs/main/best
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevlite.model import SystemOne  # noqa: E402
from jevlite.schema import target_distribution  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GRID = [round(0.5 + 0.05 * i, 2) for i in range(51)]  # 0.5 .. 3.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    args = ap.parse_args()

    eng = SystemOne(args.model)
    eng.temperature = {}
    scs = [json.loads(l) for l in (ROOT / "data/splits/val.jsonl").read_text().splitlines() if l.strip()]
    items = [(sc["state"], q, None) for sc in scs for q in sc["questions"].values()]
    targets = [target_distribution(q, sc["targets"][qid]) for sc in scs for qid, q in sc["questions"].items()]
    probs, _ = eng.distributions(items)

    by_type = defaultdict(list)
    for (_, q, _), p, t in zip(items, probs, targets):
        by_type[q["type"]].append((torch.tensor(p).clamp_min(1e-9).log(), torch.tensor(t)))

    result = {}
    for kind, pairs in by_type.items():
        def nll(temp: float) -> float:
            return sum(-(t * F.log_softmax(lp / temp, -1)).sum().item() for lp, t in pairs) / len(pairs)
        best = min(GRID, key=nll)
        result[kind] = best
        print(f"{kind:<7} n={len(pairs):<4} T=1.00 nll={nll(1.0):.4f} -> T={best:.2f} nll={nll(best):.4f}")
    (Path(args.model) / "calibration.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
