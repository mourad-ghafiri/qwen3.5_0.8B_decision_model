"""RLCD-style training: optimize the label distribution directly against proper scoring rules.

The model's output at the answer position *is* a probability distribution over the declared labels. For a
policy that outputs a distribution, the expected reward under the log score is exactly the negative
cross-entropy against the target distribution. Minimizing CE (plus a Brier term, also strictly proper) is
therefore the exact, zero-variance form of "RL with a proper-scoring reward". TypeSafe has not published
RLCD's actual objective; this is our approximation.

    python scripts/train.py --out runs/main
    python scripts/train.py --out runs/smoke --epochs 1 --max-steps 2            # smoke test
    python scripts/train.py --out runs/main0 --epochs 10 --init Qwen/Qwen3.5-0.8B-Base   # HF Jobs (see hf_job/)

The token embeddings (tied with the LM head) stay frozen: they are ~1/3 of the parameters, and the answer
labels' output rows are shared with the input vocabulary.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from jevlite.model import label_logits, label_token_ids, last_hidden, pad_batch, pick_device  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_rows(split: str) -> list[dict]:
    return [json.loads(l) for l in (ROOT / f"data/processed/{split}.jsonl").read_text().splitlines() if l.strip()]


def batches(rows: list[dict], max_tokens: int, rng: random.Random | None) -> list[list[dict]]:
    """Length-bucketed batches capped by padded token count (batch * max_len <= max_tokens)."""
    order = sorted(rows, key=lambda r: len(r["input_ids"]))
    out, cur, cur_max = [], [], 0
    for r in order:
        n = len(r["input_ids"])
        if cur and max(cur_max, n) * (len(cur) + 1) > max_tokens:
            out.append(cur)
            cur, cur_max = [], 0
        cur.append(r)
        cur_max = max(cur_max, n)
    if cur:
        out.append(cur)
    if rng:
        rng.shuffle(out)
    return out


def row_label_ids(r: dict, letter_ids: list[int], noul_ids: list[int]) -> list[int]:
    return noul_ids if r["type"] == "noul" else letter_ids[: len(r["target"])]


def batch_loss(model, batch, letter_ids, noul_ids, device, brier_weight):
    ids, lengths = pad_batch([r["input_ids"] for r in batch], 0, device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        hidden = last_hidden(model, ids, lengths)
    logits = label_logits(model, hidden.float(), [row_label_ids(r, letter_ids, noul_ids) for r in batch])
    ce = brier = 0.0
    hits = 0
    for lg, r in zip(logits, batch):
        t = torch.tensor(r["target"], device=device)
        logp = F.log_softmax(lg, dim=-1)
        ce = ce + -(t * logp).sum()
        brier = brier + ((logp.exp() - t) ** 2).sum()
        hits += int(lg.argmax().item() == int(t.argmax().item()))
    n = len(batch)
    return ce / n + brier_weight * brier / n, (ce / n).item(), (brier / n).item(), hits


@torch.no_grad()
def evaluate(model, rows, letter_ids, noul_ids, device, max_tokens) -> dict:
    model.eval()
    ce_sum = brier_sum = 0.0
    hits = n = 0
    for b in batches(rows, max_tokens, None):
        _, ce, brier, h = batch_loss(model, b, letter_ids, noul_ids, device, 0.0)
        ce_sum += ce * len(b)
        brier_sum += brier * len(b)
        hits += h
        n += len(b)
    model.train()
    n = max(1, n)
    return {"val_ce": round(ce_sum / n, 4), "val_brier": round(brier_sum / n, 4), "val_acc": round(hits / n, 4)}


def save(model, tok, path: Path) -> None:
    model.save_pretrained(path, safe_serialization=True)
    tok.save_pretrained(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(ROOT / "models/Qwen3.5-0.8B-Base"), help="local path or hub id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--batch-tokens", type=int, default=4096)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--brier", type=float, default=0.5)
    ap.add_argument("--eval-every", type=int, default=0,
                    help="extra val checks every N optimizer steps; val always runs at each epoch end")
    ap.add_argument("--patience", type=int, default=0,
                    help="stop after N epochs without val_ce improvement; 0 = always run every epoch")
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=0,
                    help="wall-clock budget: never start an epoch that would not finish in time; 0 = no limit")
    ap.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                    help="weight/optimizer dtype; float32 keeps exact AdamW updates (CUDA uses bf16 autocast)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = pick_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.init)
    model = AutoModelForCausalLM.from_pretrained(args.init, dtype=getattr(torch, args.dtype)).to(device)
    model.get_input_embeddings().weight.requires_grad_(False)
    model.train()
    letter_ids, noul_ids = label_token_ids(tok)

    train_rows, val_rows = load_rows("train"), load_rows("val")
    steps_per_epoch = math.ceil(len(batches(train_rows, args.batch_tokens, None)) / args.accum)
    total_steps = args.max_steps or steps_per_epoch * args.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    warm = max(1, int(args.warmup * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    print(f"device={device} trainable={sum(p.numel() for p in params) / 1e6:.0f}M train_rows={len(train_rows)} "
          f"epochs={args.epochs} steps/epoch={steps_per_epoch} total_steps={total_steps}", flush=True)

    log = (out / "log.jsonl").open("a")

    def write(rec: dict) -> None:
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)

    base = evaluate(model, val_rows, letter_ids, noul_ids, device, args.batch_tokens)
    write({"epoch": 0, "step": 0, **base})
    best, best_epoch, stale = float("inf"), 0, 0
    step, t0 = 0, time.time()
    stop_reason = "completed all epochs"
    epoch_secs: list[float] = []
    for epoch in range(1, args.epochs + 1):
        if args.max_hours and epoch_secs and time.time() - t0 + max(epoch_secs) * 1.1 > args.max_hours * 3600:
            stop_reason = f"time budget {args.max_hours:.2f}h: next epoch would not finish"
            break
        epoch_start = time.time()
        losses = []
        for i, b in enumerate(batches(train_rows, args.batch_tokens, rng)):
            loss, ce, _, _ = batch_loss(model, b, letter_ids, noul_ids, device, args.brier)
            (loss / args.accum).backward()
            losses.append(ce)
            if (i + 1) % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0:
                write({"step": step, "loss": round(loss.item(), 4), "ce": round(ce, 4),
                       "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0)})
            if args.max_hours and time.time() - t0 > args.max_hours * 3600:
                break  # hard stop mid-epoch; the best checkpoint so far is already saved
            if args.eval_every and step % args.eval_every == 0:
                write({"step": step, **evaluate(model, val_rows, letter_ids, noul_ids, device, args.batch_tokens)})
            if step >= total_steps:
                break
        opt.zero_grad(set_to_none=True)  # drop a partial accumulation at epoch end
        epoch_secs.append(time.time() - epoch_start)

        rec = {"epoch": epoch, "step": step, "train_ce": round(sum(losses) / max(1, len(losses)), 4),
               **evaluate(model, val_rows, letter_ids, noul_ids, device, args.batch_tokens),
               "elapsed_s": round(time.time() - t0)}
        if rec["val_ce"] < best:
            best, best_epoch, stale = rec["val_ce"], epoch, 0
            save(model, tok, out / "best")
            rec["saved_best"] = True
        else:
            stale += 1
        write(rec)
        if args.max_hours and time.time() - t0 > args.max_hours * 3600:
            stop_reason = f"time budget {args.max_hours:.2f}h reached"
            break
        if args.patience and stale >= args.patience:
            stop_reason = f"no val improvement for {args.patience} epochs"
            break
        if step >= total_steps:
            break

    save(model, tok, out / "final")
    summary = {"best_epoch": best_epoch, "best_val_ce": best, "epochs_completed": len(epoch_secs),
               "epochs_requested": args.epochs, "steps": step, "stop_reason": stop_reason,
               "elapsed_s": round(time.time() - t0)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary, flush=True)


if __name__ == "__main__":
    main()
