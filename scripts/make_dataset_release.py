"""Stage the dataset for a Hugging Face dataset repo in hf_dataset/.

    python scripts/make_dataset_release.py
    hf upload mghafiri/decision-model-scenarios hf_dataset . --repo-type dataset

`state`, `questions` and `targets` differ in shape between rows (string / object / array), so they are stored as
JSON-encoded strings. That keeps the files loadable with `datasets.load_dataset`; decode them with json.loads.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from jevlite.schema import target_distribution  # noqa: E402

OUT = ROOT / "hf_dataset"
REPO_ID = "mghafiri/decision-model-scenarios"
MODEL_ID = "mghafiri/qwen3.5-0.8B-decision-model"


def load(split: str) -> list[dict]:
    return [json.loads(l) for l in (ROOT / f"data/splits/{split}.jsonl").read_text().splitlines() if l.strip()]


def flat(sc: dict) -> dict:
    return {"id": sc["id"], "domain": sc["domain"], "pattern": sc["pattern"],
            "state": json.dumps(sc["state"], ensure_ascii=False),
            "questions": json.dumps(sc["questions"], ensure_ascii=False),
            "targets": json.dumps(sc["targets"], ensure_ascii=False)}


def main() -> None:
    splits = {"train": load("train"), "validation": load("val"), "test": load("test")}
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        (OUT / f"data/{name}.jsonl").write_text("".join(json.dumps(flat(r), ensure_ascii=False) + "\n" for r in rows))
    shutil.copy2(ROOT / "LICENSE", OUT / "LICENSE")
    shutil.copy2(ROOT / "docs/jev_spec.md", OUT / "DATA_SPEC.md")

    allsc = [sc for rows in splits.values() for sc in rows]
    qs = [(sc, qid, q) for sc in allsc for qid, q in sc["questions"].items()]
    types = Counter(q["type"] for _, _, q in qs)
    diffs = Counter(sc["targets"][qid]["difficulty"] for sc, qid, _ in qs)
    domains = sorted({sc["domain"] for sc in allsc})
    floor = sum(-sum(p * math.log(p) for p in target_distribution(q, sc["targets"][qid]) if p > 0)
                for sc, qid, q in qs) / len(qs)
    report = (ROOT / "data/review/merge_report.txt").read_text().splitlines()
    agreement = next((l.split(": ", 1)[1] for l in report if l.startswith("overall agreement")), "n/a")
    pct = lambda c: ", ".join(f"{k} {100 * v / len(qs):.0f}%" for k, v in c.most_common())  # noqa: E731
    counts = {k: (len(v), sum(len(s["questions"]) for s in v)) for k, v in splits.items()}

    card = f"""---
license: mit
language:
- en
pretty_name: Decision Model Scenarios
size_categories:
- 1K<n<10K
task_categories:
- text-classification
tags:
- calibration
- decision-model
- synthetic
- system-one
- soft-labels
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
  - split: validation
    path: data/validation.jsonl
  - split: test
    path: data/test.jsonl
---

# Decision Model Scenarios

**{len(allsc)} synthetic English scenarios with {len(qs)} typed questions and calibrated soft labels.** It is the
training data for [{MODEL_ID}](https://huggingface.co/{MODEL_ID}), a small model that answers
typed questions about a text "state" with probability distributions instead of generated text.

Each scenario has three parts:
- a **state**: a message, a record, an email thread, a policy plus a ticket, a log, and so on
- **3–6 independent questions** of three types:
  - **Choice:** pick one option
  - **Score:** pick a level on an ordered scale
  - **Noul:** yes/no
- a **target probability distribution** for each question, with a difficulty tag and a short reasoning note

The question format follows the publicly documented request shape of TypeSafe's System One API
([docs.typesafe.ai](https://docs.typesafe.ai/api.md)). This dataset is independent and not affiliated with
TypeSafe.

## Splits

| split | scenarios | questions |
|---|---|---|
{chr(10).join(f"| {k} | {n} | {m} |" for k, (n, m) in counts.items())}

- **Domains ({len(domains)}):** {", ".join(domains)}.
- **Question types:** {pct(types)}.
- **Difficulty:** {pct(diffs)}.

About half of the scenarios are realistic multi-document cases where the correct label needs multi-step
reasoning: policy exceptions, later corrections, 2-hop lookups, and instructions planted in the text that must be
ignored. Each question is still a single literal judgment.

## Fields

| field | type | content |
|---|---|---|
| `id` | string | scenario id (`bNN-NNNN`) |
| `domain` | string | one of the {len(domains)} domains |
| `pattern` | string | fan-out, routing, detection, scoring, verification, extraction-choice, ranking-relevance, matching, guardrail |
| `state` | JSON string | the content to judge (string, object or array) |
| `questions` | JSON string | `{{question_id: {{"type": "choice"|"score"|"noul", "instructions": ..., "criteria": ...}}}}` |
| `targets` | JSON string | `{{question_id: {{"probabilities": {{...}}}} or {{"noul": p}}, "difficulty": ..., "note": ...}}}}` |

- **Choice targets** are keyed by option.
- **Score targets** are keyed by level index (`"0"`…`"K-1"`).
- **Noul targets** give `noul` = P(yes).

```python
import json
from datasets import load_dataset

ds = load_dataset("{REPO_ID}")
row = ds["train"][0]
state, questions, targets = (json.loads(row[k]) for k in ("state", "questions", "targets"))
```

## How the labels were made

1. **Authoring.** Claude (Anthropic) wrote each scenario and its targets. It followed a written rubric
   (`DATA_SPEC.md`): *the target probability of an option is the fraction of careful domain experts who would
   pick it*.
2. **Blind second annotation.** An independent Claude annotator relabelled every question without seeing the
   original labels. Agreement on the top answer was **{agreement}**.
3. **Merge:**
   - Where the two agreed, the target is 0.6·author + 0.4·reviewer.
   - Borderline disagreements were averaged 50/50.
   - The remaining disagreements were adjudicated one by one, and defective questions were removed.
4. **Automated checks:**
   - schema and probability sums
   - near-duplicate states
   - synthetic-only personal data (example.com emails, 555-01xx phones, fictional organisations)
   - no arithmetic, counting or date-comparison questions

**Minimum achievable log-loss on these soft targets:** about {floor:.2f}, the mean target entropy.

## Limitations

- **The data is synthetic and LLM-authored.** The targets estimate expert agreement. They are not real-world
  outcomes.
- **English only.** The domains are business and operations workflows; there is no clinical or legal advice
  content.
- **Personal data is synthetic.** Any resemblance to real people or organisations is unintended.

## License

The data is released under the **MIT License**; see `LICENSE`.
"""
    (OUT / "README.md").write_text(card)
    print(f"staged {OUT}: " + ", ".join(f"{k} {n}" for k, (n, _) in counts.items()) + f"; agreement {agreement}")


if __name__ == "__main__":
    main()
