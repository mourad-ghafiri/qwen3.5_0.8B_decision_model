# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "torch>=2.7",
#     "transformers==5.17.0",
#     "accelerate",
#     "safetensors",
#     "numpy",
#     "pyyaml",
#     "huggingface_hub>=1.30",
#     "flash-linear-attention",
# ]
# ///
"""End-to-end HF Jobs pipeline: build rows -> train N epochs -> calibrate -> evaluate -> model card -> push.

Runs inside the job container. The project bundle (code + data splits) is mounted read-only at /bundle by
`hf jobs uv run -v ./hf_job/bundle:/bundle ...` (see hf_job/README.md).

Args:
    --epochs N        epochs to run (default 10). Validation runs after every epoch and the best epoch is published.
    --repo-name NAME  model repo name under your namespace (default qwen3.5-0.8B-decision-model)
    --private         create the repo as private
    --no-push         run everything but skip the upload (dry run on real hardware)
    --budget-hours H  the job's hard --timeout; training gets whatever time remains minus a reserve for
                      calibration, evaluation and upload, so the job finishes cleanly before HF kills it
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE = "Qwen/Qwen3.5-0.8B-Base"
RESERVE_HOURS = 0.75  # dependency install (before START) + calibrate + evaluate + model card + upload
START = time.time()
BUNDLE = Path("/bundle")
WORK = Path("/work")


def run(*cmd: str) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t = time.time()
    subprocess.run([sys.executable, *cmd], cwd=WORK, check=True)
    print(f"  ({time.time() - t:.0f}s)", flush=True)


def export_bf16(src: Path, dst: Path) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16)
    model.save_pretrained(dst, safe_serialization=True)
    AutoTokenizer.from_pretrained(src).save_pretrained(dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--repo-name", default="qwen3.5-0.8B-decision-model")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--budget-hours", type=float, required=True, help="must equal the job's --timeout in hours")
    args = ap.parse_args()

    import torch
    print(f"cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name() if torch.cuda.is_available() else '-'}")
    try:
        import fla  # noqa: F401
        print("flash-linear-attention: available")
    except ImportError:
        print("flash-linear-attention: NOT available (falls back to slower torch kernels)")

    shutil.copytree(BUNDLE, WORK, dirs_exist_ok=True)
    run_dir, export = WORK / "runs/job", WORK / "export"

    run("scripts/build_sft.py")
    train_hours = args.budget_hours - (time.time() - START) / 3600 - RESERVE_HOURS
    if train_hours <= 0.1:
        raise SystemExit(f"not enough budget left to train ({train_hours:.2f}h); aborting before spending more")
    print(f"training time budget: {train_hours:.2f}h of the {args.budget_hours}h job budget", flush=True)
    run("scripts/train.py", "--init", BASE, "--out", str(run_dir), "--epochs", str(args.epochs),
        "--lr", str(args.lr), "--dtype", "float32", "--max-hours", f"{train_hours:.3f}",
        "--batch-tokens", "16384", "--accum", "1")
    if not (run_dir / "best").exists():
        raise SystemExit("no completed epoch produced a checkpoint; nothing to publish")
    export_bf16(run_dir / "best", export)
    run("scripts/calibrate.py", str(export))
    run("scripts/evaluate.py", BASE, str(export))

    repo_id = args.repo_name
    if not args.no_push:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        repo_id = f"{api.whoami()['name']}/{args.repo_name}"
    run("scripts/model_card.py", "--model-dir", str(export), "--run-dir", str(run_dir), "--repo-id", repo_id)

    # Ship the readout package, reports and training log alongside the weights.
    shutil.copytree(WORK / "src/jevlite", export / "jevlite", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "api.py"))
    shutil.copytree(WORK / "reports", export / "reports", dirs_exist_ok=True)
    for name in ("log.jsonl", "summary.json"):
        shutil.copy2(run_dir / name, export / "reports" / f"training_{name}")
    shutil.copy2(WORK / "docs/jev_spec.md", export / "DATA_SPEC.md")

    print((WORK / "reports/eval.md").read_text())
    print(json.dumps(json.loads((run_dir / "summary.json").read_text()), indent=2))
    if args.no_push:
        print("--no-push: skipping upload")
        return
    api.create_repo(repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(export), repo_id=repo_id, repo_type="model",
                      commit_message=f"Train {args.epochs} epochs; publish best epoch with eval report")
    print(f"published: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
