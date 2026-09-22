"""Expand scenario splits into one row per question, tokenized and ready for training.

Train augmentation:
- one extra option-order shuffle for Choice questions (anti position bias / mode dropping)
- object states rendered either compact or indented at random

Output rows: {"sid", "qid", "type", "input_ids", "target"}, where target is aligned with the rendered labels.

    python scripts/build_sft.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevlite.model import DEFAULT_MAX_TOKENS, truncate_state_tokens  # noqa: E402
from jevlite.prompt import render  # noqa: E402
from jevlite.schema import option_keys, target_distribution  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOCAL_MODEL = ROOT / "models/Qwen3.5-0.8B-Base"
HUB_MODEL = "Qwen/Qwen3.5-0.8B-Base"


def rows_for(sc: dict, tok, rng: random.Random, augment: bool) -> list[dict]:
    rows = []
    for qid, q in sc["questions"].items():
        keys = option_keys(q)
        target = target_distribution(q, sc["targets"][qid])
        variants = [None]
        if augment and q["type"] == "choice":
            order = keys[:]
            while order == keys:
                rng.shuffle(order)
            variants.append(order)
        for order in variants:
            indent = 1 if augment and not isinstance(sc["state"], str) and rng.random() < 0.3 else None
            prompt = render(sc["state"], q, order, indent)
            full = tok.encode(prompt, add_special_tokens=False)
            ids = truncate_state_tokens(tok, full, DEFAULT_MAX_TOKENS)
            tgt = target if order is None else [target[keys.index(k)] for k in order]
            rows.append({"sid": sc["id"], "qid": qid, "type": q["type"], "input_ids": ids, "target": tgt,
                         "truncated": len(full) > len(ids)})
    return rows


def main() -> None:
    tok = AutoTokenizer.from_pretrained(LOCAL_MODEL if LOCAL_MODEL.exists() else HUB_MODEL)
    rng = random.Random(7)
    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        scs = [json.loads(l) for l in (ROOT / f"data/splits/{split}.jsonl").read_text().splitlines() if l.strip()]
        rows = [r for sc in scs for r in rows_for(sc, tok, rng, augment=split == "train")]
        if split == "train":
            rng.shuffle(rows)
        (out / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        n_tok = sum(len(r["input_ids"]) for r in rows)
        trunc = sum(r["truncated"] for r in rows)
        longest = max((len(r["input_ids"]) for r in rows), default=0)
        print(f"{split}: {len(rows)} rows, {n_tok:,} tokens (mean {n_tok / max(1, len(rows)):.0f}, "
              f"max {longest}), truncated rows: {trunc}")


if __name__ == "__main__":
    main()
