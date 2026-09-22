"""Budget-guarded launcher for the HF Jobs pipeline. Use this instead of calling `hf jobs` directly.

Hugging Face bills Jobs per minute and stops a job, and its billing, when it reaches --timeout
(https://huggingface.co/docs/hub/jobs-pricing). So the worst case for a run is price/hour × timeout. This
script:
  * only allows the flavors listed in PRICES, with a fixed timeout per mode
  * records every launch's worst-case cost in hf_job/budget_ledger.json
  * refuses any launch that would push the total worst case above BUDGET_USD
  * asks you to type "yes" before anything is submitted

    python hf_job/launch.py status
    python hf_job/launch.py smoke                 # 1 epoch, no upload: worst case $2.50
    python hf_job/launch.py full --epochs 10      # 10 epochs, publish: worst case $8.75
    python hf_job/launch.py full --dry-run        # show the resolved job, spend nothing

The ledger only knows about launches made through this script. Jobs started any other way are not counted.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LEDGER = HERE / "budget_ledger.json"
HF = ROOT / ".venv/bin/hf"

BUDGET_USD = 20.00
# Hourly prices from https://huggingface.co/docs/hub/jobs-pricing (checked 2026-09-22).
PRICES = {"a100-large": 2.50, "l40sx1": 1.80, "a10g-large": 1.50}
DEFAULT_FLAVOR = "a100-large"
MODES = {
    # mode: (timeout hours, job.py args)
    "smoke": (1.0, ["--epochs", "1", "--no-push"]),
    "full": (3.5, []),
}


def load_ledger() -> list[dict]:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else []


def committed(ledger: list[dict]) -> float:
    return round(sum(e["worst_case_usd"] for e in ledger), 2)


def status() -> None:
    ledger = load_ledger()
    for e in ledger:
        print(f"  {e['time']}  {e['mode']:<5} {e['flavor']:<11} timeout {e['timeout_h']}h  "
              f"worst ${e['worst_case_usd']:.2f}  job {e.get('job_id') or '?'}")
    used = committed(ledger)
    print(f"worst-case committed: ${used:.2f} of ${BUDGET_USD:.2f}  (remaining ${BUDGET_USD - used:.2f})")
    print("actual charges are usually lower: see https://huggingface.co/settings/billing (Compute Usage)")


def launch(mode: str, flavor: str, epochs: int | None, private: bool, dry_run: bool) -> None:
    if flavor not in PRICES:
        sys.exit(f"flavor {flavor!r} not allowed; choose one of {sorted(PRICES)}")
    import os
    if not dry_run and not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN is not set in this terminal. Run:  read -s HF_TOKEN && export HF_TOKEN  "
                 "(paste your token, press Enter) and try again. See GUIDE.md step 3.")
    bundle = HERE / "bundle"
    if not (bundle / "data/splits/train.jsonl").exists():
        sys.exit("hf_job/bundle is missing or incomplete: run `python hf_job/make_bundle.py` first")

    timeout_h, job_args = MODES[mode]
    job_args = list(job_args)
    if mode == "full":
        job_args += ["--epochs", str(epochs or 10)] + (["--private"] if private else [])
    job_args += ["--budget-hours", str(timeout_h)]
    worst = round(PRICES[flavor] * timeout_h, 2)

    ledger = load_ledger()
    used = committed(ledger)
    cmd = [str(HF), "jobs", "uv", "run", "--flavor", flavor, "--timeout", f"{int(timeout_h * 60)}m",
           "--secrets", "HF_TOKEN", "--name", f"decision-model-{mode}", "-v", f"{bundle}:/bundle",
           *([] if dry_run else ["--detach"]), *(["--dry-run"] if dry_run else []),
           str(HERE / "job.py"), *job_args]

    print(f"mode={mode} flavor={flavor} (${PRICES[flavor]:.2f}/h) hard timeout={timeout_h}h")
    print(f"worst-case cost of this run: ${worst:.2f}")
    print(f"worst-case committed so far: ${used:.2f}; after this run: ${used + worst:.2f} of ${BUDGET_USD:.2f}")
    print("command:", " ".join(cmd))
    if dry_run:
        subprocess.run(cmd, cwd=ROOT, check=False)
        return
    if used + worst > BUDGET_USD + 1e-9:
        sys.exit(f"REFUSED: would exceed the ${BUDGET_USD:.2f} budget. Check real usage on the billing page; "
                 f"if it is lower, edit {LEDGER.name} to record actual costs.")
    if input('type "yes" to submit this job and accept the worst-case cost: ').strip() != "yes":
        sys.exit("not submitted")

    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": mode, "flavor": flavor,
             "timeout_h": timeout_h, "worst_case_usd": worst, "job_id": None}
    ledger.append(entry)  # record before submitting: a crash after submit still counts against the budget
    LEDGER.write_text(json.dumps(ledger, indent=2))
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(res.stdout, res.stderr, sep="")
    m = re.search(r"\b([0-9a-f]{24})\b", res.stdout)
    entry["job_id"] = m.group(1) if m else None
    if res.returncode != 0 and not m:
        entry["worst_case_usd"] = 0.0  # submission failed: nothing is running
        entry["note"] = "submission failed"
    LEDGER.write_text(json.dumps(ledger, indent=2))
    if m:
        print(f"\nfollow:  {HF} jobs logs {m.group(1)}\ncancel:  {HF} jobs cancel {m.group(1)}   (stops billing)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["status", *MODES])
    ap.add_argument("--flavor", default=DEFAULT_FLAVOR)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.mode == "status":
        status()
    else:
        if args.epochs is not None and args.mode != "full":
            sys.exit("--epochs only applies to `full`")
        if args.epochs is not None and args.epochs < 10:
            sys.exit("full runs use at least 10 epochs (your requirement)")
        launch(args.mode, args.flavor, args.epochs, args.private, args.dry_run)


if __name__ == "__main__":
    main()
