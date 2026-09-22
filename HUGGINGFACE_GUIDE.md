# Step-by-step guide: train, evaluate and publish on Hugging Face

This guide trains the decision model on **2000 scenarios** (9,716 questions) on a Hugging Face GPU, evaluates
it, and publishes it as **https://huggingface.co/<your-username>/qwen3.5-0.8B-decision-model**.

- **Expected cost:** about **$3–5**.
- **Hard cap:** the launcher never lets the worst case go above **$20** in total.
- **Time:** about 5 minutes of your time, plus about **1–2 hours** of waiting.

Run every command in **one Terminal window**, from the project folder:

```bash
cd /path/to/qwen3.5_0.8B_decision_model
```

---

## Step 1: One-time setup (already done)

You already did this:
- **Credits:** a positive credit balance at https://huggingface.co/settings/billing.
- **Token:** a token from https://huggingface.co/settings/tokens with **Repositories: write** and **Jobs:
  start and manage** (`job.write`) permissions.

If you ever need a new token, create it on that page and delete the old one.

## Step 2: Log in (every new Terminal window)

```bash
.venv/bin/hf auth login                 # paste your token (nothing is shown; that's normal)
read -s HF_TOKEN && export HF_TOKEN     # paste the token again, press Enter
echo ${#HF_TOKEN}                       # must print a number (not 0)
.venv/bin/hf auth whoami                # must print: <your-username>
```

Never paste the token into a chat, a file or a screenshot.

## Step 3: Check the bundle and the budget

```bash
.venv/bin/python hf_job/make_bundle.py
.venv/bin/python hf_job/launch.py status
```

You should see:
- `bundle ready ... (7.0 MB)`
- `worst-case committed: $2.50 of $20.00` (the earlier smoke test)

**Skip the smoke test.** It already passed, and the budget is reserved for the real run plus one retry.

## Step 4: Train, evaluate and publish (about 1–2 h, worst case $8.75)

```bash
.venv/bin/python hf_job/launch.py full --epochs 10 --private
```

Type `yes`. It prints a **job id**. Follow the job's progress:

```bash
.venv/bin/hf jobs logs -f <your-username>/<job_id>
```

Press **Ctrl+C** to stop watching. The job keeps running. You can also watch it in the browser at
`https://huggingface.co/jobs/<your-username>/<job_id>`.

**What you'll see:**
1. Setup: `cuda=True ... A100` and `flash-linear-attention: available`.
2. `train: 11342 rows, 4,338,131 tokens`, then `training time budget: ~2.7h`.
3. One line per epoch, like `{'epoch': 3, ... 'val_ce': ..., 'val_acc': ...}`.
   - `'saved_best': True` marks a new best epoch. The best epoch is the one published.
   - `val_ce` usually falls for a few epochs, then flattens or rises. The job still runs all 10 epochs, and
     keeps the best one.
4. The evaluation table (`# Evaluation (test split)`) comparing base Qwen with this model.
5. `published: https://huggingface.co/<your-username>/qwen3.5-0.8B-decision-model`

It is published **private** first, so you can review it.

**To stop the job at any time** (this also stops billing):
```bash
.venv/bin/hf jobs cancel <your-username>/<job_id>
```

**If it fails:** billing stops automatically. Copy the last ~40 lines of the log and send them to Claude. One
retry fits inside your $20.

## Step 5: Check the results

- Model page and model card: https://huggingface.co/<your-username>/qwen3.5-0.8B-decision-model
- Download the report to your Mac:
  ```bash
  .venv/bin/hf download <your-username>/qwen3.5-0.8B-decision-model reports/eval.md reports/training_summary.json --local-dir results
  cat results/reports/eval.md
  ```

**What good looks like:**
- **Accuracy:** higher is better.
- **Log-loss, Brier and ECE:** lower is better. The fine-tuned model should beat base Qwen on all of them.
- **Option-order sensitivity (permutation TVD):** lower is better. Below 0.05 is the goal.
- **Epoch table in the model card:** shows which epoch was best and was published.

## Step 6: Make it public

On the model page, go to **Settings** → **Change model visibility** → **Public**. Share the link:
**https://huggingface.co/<your-username>/qwen3.5-0.8B-decision-model**

## Step 7: Check what you spent

```bash
.venv/bin/python hf_job/launch.py status
```
Real charges are at https://huggingface.co/settings/billing, under **Compute Usage**.

---

## How your $20 is protected

| Protection | Effect |
|---|---|
| Hard timeout of 3.5 h on the job | Hugging Face **stops the job and its billing** at the timeout ([docs](https://huggingface.co/docs/hub/jobs-pricing)). Worst case $8.75. |
| Training time budget | Training stops itself in time to evaluate and publish before the timeout. |
| Budget ledger (`hf_job/budget_ledger.json`) | `launch.py` refuses any launch that would push the total worst case above $20. |
| Hardware allowlist | Only single-GPU machines ($1.50–2.50/h). |
| Typed `yes` | Nothing is billed without your confirmation. |

**Worst-case total:** smoke $2.50 (done) + full $8.75 + one retry $8.75 = **$20.00**. The launcher refuses
anything beyond that.

**Only launch with `hf_job/launch.py`.** Jobs started from the website or with `hf jobs` directly aren't
tracked by the ledger.

## Troubleshooting

| Problem | Fix |
|---|---|
| `HF_TOKEN is not set in this terminal` | Repeat Step 2 in the same window. |
| `403 ... missing permissions: job.write` | The token needs the Jobs permission (Step 1). |
| `401` / not authorized | Log in again (Step 2). |
| `REFUSED: would exceed the $20.00 budget` | The cap is working. Check real usage on the billing page first. |
| Job stopped before 10 epochs (`epochs_completed < 10` in the card) | Training ran out of its time budget. The best epoch so far is still published. |
