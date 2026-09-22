# Train, evaluate and publish on Hugging Face Jobs

A single job runs the whole pipeline on a GPU:

1. build the training rows
2. train for **N epochs** (default 10), with validation after every epoch
3. keep the **best** epoch
4. fit calibration temperatures
5. evaluate the base model against the fine-tuned one on the test split
6. write the model card from the real numbers
7. push `<your-username>/qwen3.5-0.8B-decision-model`

These commands touch your Hugging Face account and are billed to it. You run them yourself; Claude only
prepared them.

## Prerequisites

- A positive **credit balance** at https://huggingface.co/settings/billing. Jobs are billed per minute, only
  while Starting or Running ([docs](https://huggingface.co/docs/hub/jobs-pricing)).
- Log in locally with `hf auth login`, using a token with **write** access.
- The data pipeline must be finished (`data/splits/{train,val,test}.jsonl` exist):
  ```bash
  .venv/bin/python scripts/merge_labels.py merge && .venv/bin/python scripts/split.py
  ```

## Run: always through the budget-guarded launcher

**Do not call `hf jobs` directly.** `hf_job/launch.py` is the only way to launch.
- It enforces the **$20 total budget**.
- It only allows `a100-large`, `l40sx1` and `a10g-large`.
- It always sets a hard `--timeout`.
- It asks you to type `yes` before anything is billed.

```bash
cd <project root>
.venv/bin/python hf_job/make_bundle.py            # -> hf_job/bundle/ (a few MB)
.venv/bin/python hf_job/launch.py status          # budget ledger

.venv/bin/python hf_job/launch.py full --epochs 10            # train 10 epochs + publish (worst case $8.75)
.venv/bin/python hf_job/launch.py full --epochs 10 --private  #   or publish as private first
.venv/bin/python hf_job/launch.py full --dry-run  # show the resolved job without submitting
.venv/bin/hf jobs cancel <job_id>                 # stop a job, and its billing, at any time
```

## How the $20 cap is enforced

| Layer | What it guarantees |
|---|---|
| Hard `--timeout` on every job (full run 3.5 h) | HF stops the job **and its billing** at the timeout ([docs](https://huggingface.co/docs/hub/jobs-pricing)). Worst case = price × timeout. |
| Ledger in `hf_job/budget_ledger.json` | Every launch's worst case is recorded *before* submission. A launch that would take the total above **$20** is refused. |
| Hardware allowlist | Expensive multi-GPU flavors (e.g. `a100x8` at $20/h) cannot be selected. |
| In-job time budget (`--budget-hours`) | Training stops itself in time to calibrate, evaluate and publish before the timeout, so reaching the limit doesn't waste the run. |
| Auto-suspend on failure | HF suspends a failing job and stops billing ([docs](https://huggingface.co/docs/hub/jobs-pricing)). |

**Worst-case plan:** the earlier smoke test ($2.50) + full $8.75 + one retry $8.75 = **$20.00**. The launcher
refuses anything beyond that.

Actual cost should be well under the worst case: the full run on 2000 scenarios (about 4.3M training tokens
per epoch) is estimated at about 1–2 h, or $3–5.
Real usage is on https://huggingface.co/settings/billing under "Compute Usage".

**Not guaranteed by these tools:**
- Jobs launched outside `launch.py` (the HF website, another terminal) are not in the ledger.
- HF documents no spending-limit setting for personal accounts. It is unverified what happens if your credit
  balance runs out mid-job, so don't rely on the balance as a cap.
- Hub storage for the uploaded bundle (a few MB) and the published model (about 1.6 GB) is billed separately,
  if at all, under HF's storage terms. I have not verified those prices.

## About running 10+ epochs

The training set has 1700 scenarios. After a few epochs, a 0.8B model starts memorizing them, and its
probabilities become overconfident. That hurts calibration, which is the point of this model. The job protects
against this:

- It runs every epoch you ask for.
- It measures validation cross-entropy, Brier and accuracy after each one.
- It publishes the epoch with the **lowest validation cross-entropy**, not necessarily the last.

The per-epoch table in the model card shows where it peaked.

## What gets published

| path | contents |
|---|---|
| root | bf16 weights and tokenizer |
| `calibration.json` | per-primitive temperatures |
| `jevlite/` | the readout package: prompt rendering, the label-logit softmax, Jev-shaped answers |
| `reports/eval.md`, `reports/eval.json` | test metrics for the base and fine-tuned models |
| `reports/training_log.jsonl`, `reports/training_summary.json` | per-step and per-epoch curves |
| `DATA_SPEC.md` | the data spec and labelling rubric |
| `README.md` | the model card |

The training data itself is uploaded only to your private jobs bucket, as the `/bundle` mount. It is **not**
published.
