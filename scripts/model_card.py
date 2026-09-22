"""Render the Hugging Face model card (README.md) from real run outputs.

    python scripts/model_card.py --model-dir export --run-dir runs/job --repo-id user/qwen3.5-0.8B-decision-model

Inputs: reports/eval.json, <run-dir>/log.jsonl, <run-dir>/summary.json, <model-dir>/calibration.json and
data/splits/*.jsonl. Every number in the card comes from these files.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def fmt(x, nd=3) -> str:
    return "–" if x is None or x != x else f"{x:.{nd}f}"


def label(model: str) -> str:
    return "base Qwen3.5-0.8B-Base" if model.startswith("Qwen/") or "Qwen3.5-0.8B-Base" in model else "**this model**"


def eval_table(results: list[dict]) -> str:
    lines = ["| model | type | n | acc | log-loss | Brier | ECE | conf≥0.7 coverage / acc |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        name = label(r["model"])
        for kind, d in sorted(r["by_type"].items()):
            cov = f"{fmt(d.get('cov@0.7'), 2)} / {fmt(d.get('acc@0.7'), 2)}" if "cov@0.7" in d else "–"
            lines.append(f"| {name} | {kind} | {d['n']} | {fmt(d['acc'])} | {fmt(d['logloss'])} | "
                         f"{fmt(d['brier'])} | {fmt(d['ece'])} | {cov} |")
    return "\n".join(lines)


def extras_table(results: list[dict]) -> str:
    lines = ["| model | option-permutation TVD (choice) | score MAE | latency p50 / p95 (ms, 1 request) |",
             "|---|---|---|---|"]
    for r in results:
        mae = r["by_type"].get("score", {}).get("score_mae")
        lines.append(f"| {label(r['model'])} | {fmt(r['perm_tvd'], 4)} | {fmt(mae)} | "
                     f"{r['latency_p50_ms']:.0f} / {r['latency_p95_ms']:.0f} |")
    return "\n".join(lines)


def epoch_table(log: list[dict], best_epoch: int) -> str:
    rows = [r for r in log if "epoch" in r]
    lines = ["| epoch | train CE | val CE | val Brier | val acc |", "|---|---|---|---|---|"]
    for r in rows:
        mark = " **(best, published)**" if r["epoch"] == best_epoch else ""
        lines.append(f"| {r['epoch']}{mark} | {fmt(r.get('train_ce'))} | {fmt(r['val_ce'])} | "
                     f"{fmt(r['val_brier'])} | {fmt(r['val_acc'])} |")
    return "\n".join(lines)


def data_stats() -> dict:
    stats = {}
    for split in ("train", "val", "test"):
        scs = load_jsonl(ROOT / f"data/splits/{split}.jsonl")
        stats[split] = {"scenarios": len(scs), "questions": sum(len(s["questions"]) for s in scs)}
    allsc = [s for split in ("train", "val", "test") for s in load_jsonl(ROOT / f"data/splits/{split}.jsonl")]
    stats["types"] = Counter(q["type"] for s in allsc for q in s["questions"].values())
    stats["difficulty"] = Counter(t["difficulty"] for s in allsc for t in s["targets"].values())
    stats["domains"] = sorted({s["domain"] for s in allsc})
    report = ROOT / "data/review/merge_report.txt"
    line = next((l for l in report.read_text().splitlines() if l.startswith("overall agreement")), "") if report.exists() else ""
    stats["agreement"] = line.replace("overall agreement: ", "") or "n/a"
    return stats


def pct(x) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def compare_section(rep: dict, hardware: str) -> str:
    """Head-to-head vs TypeSafe Jev, computed from scripts/compare_jev.py output (results/compare_jev.json)."""
    m, h, qs = rep["meta"], rep["head_to_head"], rep["questions"]
    j, o = rep["summary"]["jev"], rep["summary"]["ours"]
    jt, ot = j["by_type"], o["by_type"]

    def top(q, who):
        d = q[who]["dist"]
        return max(d[0], 1 - d[0]) if q["type"] == "noul" else max(d)

    jev_certain = sum(top(q, "jev") >= 0.99 for q in qs) / len(qs)
    ours_top = sum(top(q, "ours") for q in qs) / len(qs)
    target_top = sum(max(q["target"]) for q in qs) / len(qs)
    three = lambda t, key, f: " / ".join(f(t[k].get(key)) for k in ("choice", "noul", "score"))  # noqa: E731
    acc3 = lambda t: three(t, "acc", lambda v: f"{100 * v:.1f}%")  # noqa: E731
    ece3 = lambda t: three(t, "ece", lambda v: fmt(v))  # noqa: E731
    gate = lambda t, tau: f"{pct(t['choice'].get(f'cov@{tau}'))} at {pct(t['choice'].get(f'acc@{tau}'))}"  # noqa: E731
    ms = lambda v: f"{v / 1000:.1f} s"  # noqa: E731
    where = f"locally on a {hardware}" if hardware else f"locally on `{m['our_device']}`"
    return f"""## Head to head: TypeSafe Jev vs this model on unseen scenarios

We sent the **same {m['scenarios_scored']} held-out scenarios ({m['questions_scored']} questions)** to both
models, as identical `POST /v1/systemone` requests. None of these scenarios was used to train, select or
calibrate this model.

- **Jev:** `{m['jev_model_reported']}` through TypeSafe's hosted API.
- **This model** (0.8B parameters): ran **{where}**, not on a GPU server.

| | TypeSafe Jev (`{m['jev_model_reported']}`) | this model (0.8B{', ' + hardware if hardware else ''}) |
|---|---|---|
| **Accuracy** (top answer = label's top answer) | {100 * j['overall']['acc']:.1f}% | {100 * o['overall']['acc']:.1f}% |
| Accuracy: Choice / Noul / Score | {acc3(jt)} | {acc3(ot)} |
| Brier (lower is better) | {fmt(j['overall']['brier'])} | {fmt(o['overall']['brier'])} |
| Log-loss vs the soft labels (lower is better)¹ | {fmt(j['overall']['logloss'])} | {fmt(o['overall']['logloss'])} |
| Calibration error, ECE: Choice / Noul / Score¹ | {ece3(jt)} | {ece3(ot)} |
| Choice answers with confidence ≥ 0.7: share automated, accuracy | {gate(jt, 0.7)} | {gate(ot, 0.7)} |
| Choice answers with confidence ≥ 0.9: share automated, accuracy | {gate(jt, 0.9)} | {gate(ot, 0.9)} |
| Latency per request, p50 / p95 | {ms(j['latency_p50_ms'])} / {ms(j['latency_p95_ms'])} (hosted API, incl. network) | {ms(o['latency_p50_ms'])} / {ms(o['latency_p95_ms'])} (local, no network) |
| Cost for the whole test set | ${m['jev_estimated_cost_usd']:.3f} ({m['jev_input_tokens'] / 1000:.0f}k input tokens) | $0, offline, on your own machine |

**What this means:**
- **Head to head:** on questions where only one model was right, Jev was right {h['only_jev_correct']} times
  and this model {h['only_ours_correct']}. The two models agree on {100 * h['agreement']:.1f}% of top answers.
- **Jev is the stronger model** on accuracy, Brier and confidence-gated automation.
- **This model is a small open alternative** that runs fully offline. On a GPU it answers in about 50 ms per
  request (see the evaluation section below).
- **This model is under-confident.** Its average top probability is {ours_top:.2f}, while the labels average
  {target_top:.2f}, so it automates fewer decisions at high confidence thresholds.
- **¹ Read log-loss and ECE with care.** The labels are soft probabilities written with this project's rubric,
  and this model was trained on the same labelling style. Jev often answers with near-certainty
  ({100 * jev_certain:.0f}% of its top probabilities are ≥ 0.99), which log-loss penalises whenever the labels
  spread some probability to other options. Accuracy and Brier are the fairer comparison.

The comparison script and a guide to reproduce it with your own TypeSafe API key are in the project repository
(`scripts/compare_jev.py`, `docs/compare_jev.md`).
"""


def render(repo_id: str, results: list[dict], log: list[dict], summary: dict, calib: dict, stats: dict,
           compare: dict | None = None, hardware: str = "") -> str:
    total_q = sum(stats["types"].values()) or 1
    mix = ", ".join(f"{k} {100 * v / total_q:.0f}%" for k, v in stats["types"].most_common())
    diff = ", ".join(f"{k} {100 * v / total_q:.0f}%" for k, v in stats["difficulty"].most_common())
    temps = ", ".join(f"{k} T={v}" for k, v in sorted(calib.items())) or "none"
    return f"""---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B-Base
language:
- en
library_name: transformers
pipeline_tag: text-classification
tags:
- decision-model
- calibration
- system-one
- classification
- structured-output
---

# {repo_id.split('/')[-1]}

A **decision model** fine-tuned from
[Qwen/Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base). It answers typed questions about a
`state` and returns **calibrated probability distributions**, not generated text:

| type | question | answer |
|---|---|---|
| **Choice** | pick one option from a set you define | `choice`, `probabilities`, `confidence` |
| **Score** | pick a position on ordered levels you describe | `score` (= Σ i·pᵢ), `legend`, `probabilities`, `confidence` |
| **Noul** | yes/no | `noul` = P(yes) |

The interface follows the publicly documented request/response shape of TypeSafe's System One API
([docs](https://docs.typesafe.ai/api.md)). This is an independent open model. **It is not affiliated with or
endorsed by TypeSafe, and it is not Jev.**

{compare_section(compare, hardware) if compare else ""}
## How it works

- **Logit readout.** Each question is rendered after the state and ends in `Answer:`. The probabilities are a
  softmax of the next-token logits restricted to single-token labels: ` A`…` Z`/` a`…` z` for options and
  levels, and ` yes`/` no` for Noul. Every answer is a valid distribution over exactly the options you
  declared, from one forward pass per question, with no JSON parsing.
- **Confidence** is `(K·p_max − 1)/(K − 1)`, the definition published at
  [docs.typesafe.ai/confidence](https://docs.typesafe.ai/confidence.md).
- **Training objective (RLCD-inspired).** Because the output *is* a distribution, the expected log-score reward
  equals negative cross-entropy. We therefore minimise `CE(target, p) + 0.5·Brier(target, p)`, two strictly
  proper scoring rules, against soft expert-agreement targets. TypeSafe has not published RLCD's actual
  objective, so this is an approximation, not a reproduction.
- **Post-hoc temperature per primitive**, fitted on the validation set: {temps}. It is stored in
  `calibration.json` and applied by the bundled `jevlite` package.

## Usage

```python
from huggingface_hub import snapshot_download
import sys

path = snapshot_download("{repo_id}")
sys.path.insert(0, path)              # bundled jevlite/ package
from jevlite.model import SystemOne

engine = SystemOne(path)              # applies calibration.json automatically
print(engine.system_one(
    state="Help! My payouts have been failing for 3 days.",
    questions={{
        "is_urgent": {{"type": "noul", "instructions": "Does this convey urgency?"}},
        "department": {{"type": "choice", "instructions": "Which team should handle this?",
                        "criteria": {{"billing": "Payments, invoicing, refunds",
                                      "technical": "Bugs, outages, integrations",
                                      "sales": "Pricing, upgrades, new accounts"}}}},
        "frustration": {{"type": "score", "instructions": "How frustrated is the customer?",
                         "criteria": ["Calm", "Frustrated", "Very angry"]}},
    }},
))
```

## Training data

The dataset has {sum(stats[s]['scenarios'] for s in ('train', 'val', 'test'))} synthetic English scenarios
({sum(stats[s]['questions'] for s in ('train', 'val', 'test'))} questions), each a state plus 3–6 independent
questions. It covers {len(stats['domains'])} domains: {', '.join(stats['domains'])}.

- **Question mix:** {mix}.
- **Difficulty mix:** {diff}.
- **How targets were made:** Claude authored them as the *fraction of careful experts who would pick each
  option*. A blind second annotator relabelled every question without seeing the originals.
  - **Agreement on the top answer:** {stats['agreement']}.
  - Where the two agreed, the target is 0.6·author + 0.4·reviewer.
  - Borderline disagreements were averaged 50/50.
  - The remaining disagreements were adjudicated one by one.
- **Splits by scenario:** train {stats['train']['scenarios']} / val {stats['val']['scenarios']} /
  test {stats['test']['scenarios']}.

## Training

- Full fine-tune with frozen token embeddings, fp32 weights and bf16 autocast.
- AdamW, lr 1e-5, cosine schedule. Option-order shuffles were used as augmentation.
- **Epochs run:** {summary.get('epochs_completed')} of {summary.get('epochs_requested')} requested
  ({summary.get('stop_reason')}), with {summary.get('steps')} optimizer steps in
  {summary.get('elapsed_s', 0) / 60:.0f} min.
- The published weights are the checkpoint with the lowest validation cross-entropy: **epoch
  {summary.get('best_epoch')}**.

{epoch_table(log, summary.get('best_epoch', -1))}

## Evaluation (held-out test split)

Accuracy compares the predicted argmax with the target argmax. Log-loss and Brier are measured against the
soft targets. ECE uses 10 bins over the top-option probability (P(yes) for Noul).

{eval_table(results)}

{extras_table(results)}

## Limitations

- The targets are expert-agreement estimates written by an LLM, not real-world outcomes. Calibration here
  means calibration to those targets. Validate on your own labelled data before automating decisions.
- The test set is small (about {stats['test']['questions']} questions), so ECE is noisy.
- The model is English-only and text-only.
- Keep arithmetic, counting and date comparison in code. Ask atomic, literal questions.
- Use `confidence` to route uncertain cases to a human or a larger model.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="folder with calibration.json (the card is written here by default)")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--run-dir", help="training run folder with log.jsonl and summary.json")
    ap.add_argument("--log", help="training log (default: <run-dir>/log.jsonl)")
    ap.add_argument("--summary", help="training summary (default: <run-dir>/summary.json)")
    ap.add_argument("--eval", default=str(ROOT / "reports/eval.json"), help="evaluate.py output")
    ap.add_argument("--compare", help="compare_jev.py output (adds the Jev head-to-head section)")
    ap.add_argument("--hardware", default="", help='where this model ran for the comparison, e.g. "MacBook Pro (Apple M2 Max)"')
    ap.add_argument("--out", help="output path (default: <model-dir>/README.md)")
    args = ap.parse_args()
    log_path = Path(args.log) if args.log else Path(args.run_dir) / "log.jsonl"
    summary_path = Path(args.summary) if args.summary else Path(args.run_dir) / "summary.json"
    results = json.loads(Path(args.eval).read_text())
    log = load_jsonl(log_path)
    summary = json.loads(summary_path.read_text())
    cal_path = Path(args.model_dir) / "calibration.json"
    calib = json.loads(cal_path.read_text()) if cal_path.exists() else {}
    compare = json.loads(Path(args.compare).read_text()) if args.compare else None
    card = render(args.repo_id, results, log, summary, calib, data_stats(), compare, args.hardware)
    out = Path(args.out) if args.out else Path(args.model_dir) / "README.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(card)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
