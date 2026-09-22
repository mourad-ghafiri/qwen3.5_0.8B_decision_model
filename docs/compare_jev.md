# Compare Jev with our model on unseen scenarios

`scripts/compare_jev.py` sends every held-out scenario to **TypeSafe's Jev** (the real API) and to **our model**
(locally) as the same request. It scores both against our held-out labels and writes the results to JSON.

- **Data:** the **test split**, 150 scenarios and 714 questions. It was never used for training, epoch
  selection or calibration.
- **Requests:** each scenario is **one** `POST /v1/systemone` request, with the same `state` and the same
  `questions` map for both models ([API reference](https://docs.typesafe.ai/api.md)).
- **Metrics:** the same as our published evaluation.
  - **Accuracy:** the top option matches the label's top option.
  - **Log-loss and Brier:** measured against the soft labels.
  - **ECE:** calibration error.
  - **Also reported:** coverage at confidence thresholds, head-to-head counts, and breakdowns by type,
    difficulty and domain.
- **Cost of a full run:** well under **$0.05**. Jev is billed at $0.042 per million input tokens and output
  tokens are free ([Models](https://docs.typesafe.ai/models.md)). The full test split is a few hundred thousand
  input tokens. A built-in guard stops sending after 5M input tokens (about $0.21).

---

## Step 1: Get a TypeSafe API key

Create a key in the TypeSafe console: https://console.typesafe.ai/. Treat it like a password: never paste it
into a file, a commit, a chat or a screenshot.

## Step 2: Put the key in your terminal (not in a file)

In the Terminal window you will run the comparison from:

```bash
cd <project folder>
read -s JEV_MODEL_API_KEY && export JEV_MODEL_API_KEY    # paste the key, press Enter (nothing is shown)
echo ${#JEV_MODEL_API_KEY}                               # prints the key length, never the key
```

`export JEV_MODEL_API_KEY=...` also works, but it leaves the key in your shell history. `read -s` doesn't. The
script only reads the key from the environment and never prints or saves it.

## Step 3: Make sure our model is available locally

```bash
ls models/decision-model/config.json 2>/dev/null \
  || .venv/bin/hf download mghafiri/qwen3.5-0.8B-decision-model --local-dir models/decision-model
```

To use another folder, pass `--our-model <path>`.

## Step 4: Check the request (free, no calls)

```bash
.venv/bin/python scripts/compare_jev.py --dry-run
```

This prints the first request exactly as Jev will receive it, then exits.

## Step 5: Smoke test on 5 scenarios

```bash
.venv/bin/python scripts/compare_jev.py --limit 5
```

This costs a fraction of a cent. It should end with a small results table and
`wrote results/compare_jev.json and results/compare_jev.md`.

## Step 6: Full comparison (150 unseen scenarios)

```bash
.venv/bin/python scripts/compare_jev.py
```

It takes a few minutes: Jev answers each request in well under a second, and our model runs on your Mac.

**Re-running is free for Jev.** Every Jev response is saved in `results/jev_cache_test.jsonl`, and a re-run
reuses the cache and only sends scenarios that are missing. To force fresh Jev answers, for example after a
new Jev release, delete or rename the cache file first.

Options:

| flag | default | effect |
|---|---|---|
| `--split test\|val` | `test` | `test` is strictly unseen. `val` was used to pick the best epoch and fit calibration temperatures, so it slightly favours our model. |
| `--limit N` | all | only the first N scenarios |
| `--jev-model` | `jev-latest` | any model name Jev accepts, e.g. a pinned version such as `jev-1.13.0` |
| `--our-model` | `models/decision-model` | folder of our model |
| `--max-input-tokens` | 5,000,000 | cost guard: stop sending to Jev after this many billed input tokens |
| `--out` | `results/compare_jev.json` | where to write the report (the `.md` goes next to it) |

## Step 7: Read the results

**`results/compare_jev.md`**, the human summary:
- overall accuracy, log-loss, Brier and latency for both models
- a per-type table (Choice / Noul / Score) with ECE
- **head to head:**
  - how often the models agree
  - how many questions only one of them got right
  - the mean log-loss difference, with a 95% bootstrap confidence interval. A positive value means our model
    is better. If the interval includes 0, the difference is not significant on this test set.
- tables by difficulty and by domain

**`results/compare_jev.json`**, everything, machine-readable:

| key | contents |
|---|---|
| `meta` | split, counts, the Jev version that actually answered, our model and device, Jev tokens and estimated cost, any errors per scenario |
| `summary.jev`, `summary.ours` | `overall` metrics, `by_type` (the same fields as our `eval.json`), latency p50/p95 |
| `head_to_head` | agreement, correct/incorrect counts, log-loss wins, mean difference with its 95% CI |
| `by_difficulty`, `by_domain` | accuracy and log-loss for both models per group |
| `questions` | one entry per question: ids, domain, type, difficulty, option keys, target, both distributions, and both correctness and log-loss values |

Quick look with `jq`:
```bash
jq '.summary | {jev: .jev.overall, ours: .ours.overall}' results/compare_jev.json
jq '.head_to_head' results/compare_jev.json
jq '[.questions[] | select(.jev.correct != .ours.correct) | {id, qid, type, jev: .jev.correct, ours: .ours.correct}]' results/compare_jev.json
```

---

## How to interpret the comparison fairly

- **The labels come from our side.** Claude wrote the targets with our rubric (`docs/jev_spec.md`), and our
  model was trained on 1,700 scenarios labelled the same way. The test scenarios are unseen, but their *style*
  and labelling conventions are familiar to our model, and new to Jev. Expect this to favour our model most on
  **log-loss, Brier and ECE**, which measure agreement with our probability conventions.
- **Accuracy is the fairest single number.** Whether the top answer matches the label's top answer depends
  least on how probabilities are spread.
- **The test set is small** (714 questions). Use the confidence interval, and treat per-domain rows with only
  a few questions as anecdotes.
- **Latency is not like-for-like.** Jev's includes network round-trip time. Ours is local inference on your
  machine, for example Apple MPS.
- **Jev versions change.** The report records the version that answered (`meta.jev_model_reported`). Pin a
  version with `--jev-model` for reproducible comparisons.
- **Jev behaviour is documented by TypeSafe:** literal reading, and weaker results with math, dates and
  indirection ([jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)). Our test questions avoid
  math and date comparisons, following the same guidance.

## Data and publishing

- **What is sent to TypeSafe:** only the test scenarios, which are synthetic (fictional people and
  organisations).
- **The key:** it is read from `JEV_MODEL_API_KEY` at runtime and is never written to any file.
- **Before publishing Jev's outputs:** `results/jev_cache_*.jsonl` and `results/compare_jev.json` contain
  Jev's answers. Check TypeSafe's terms at https://docs.typesafe.ai/legal.md before sharing them in a public
  repository. If in doubt, share only the summary tables.

## Troubleshooting

| message | fix |
|---|---|
| `JEV_MODEL_API_KEY is not set` | Run Step 2 in the same Terminal window. |
| `Jev returned 401 Unauthorized` | The key is wrong or revoked. Create a new one (Step 1). |
| `Jev HTTP 429` / `529`, retrying | Rate limit or overload. The script backs off and retries automatically. |
| `Jev error HTTP 422` for a scenario | Jev rejected that request. It is listed in `meta.errors.jev` and skipped; the rest continue. |
| `cost guard: ... not sending more` | Raise `--max-input-tokens` if you really want to send more. |
| `our model not found` | Run Step 3. |
| `ModuleNotFoundError` | Use `.venv/bin/python`, not the system `python`. |
| `CERTIFICATE_VERIFY_FAILED` / `TLS certificate verification failed` | The python.org macOS Python doesn't use the system certificate store. The script verifies against `certifi`'s CA bundle; make sure it is installed: `.venv/bin/pip install certifi`. Certificate checking is never turned off. |
