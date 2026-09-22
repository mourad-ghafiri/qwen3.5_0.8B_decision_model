---
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

**2000 synthetic English scenarios with 9716 typed questions and calibrated soft labels.** It is the
training data for [mghafiri/qwen3.5-0.8B-decision-model](https://huggingface.co/mghafiri/qwen3.5-0.8B-decision-model), a small model that answers
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
| train | 1700 | 8282 |
| validation | 150 | 720 |
| test | 150 | 714 |

- **Domains (32):** agent_tool_routing, airline_travel_ops, aml_kyc_compliance, app_marketplace_policy, banking_fintech, citation_verification, clinical_literature_screening, customer_support, cybersecurity_soc, ecommerce, education_admissions, energy_utilities, entity_matching, government_services, healthcare_admin, hr_recruiting, insurance_claims, it_helpdesk, legal_compliance, llm_guardrails, logistics_sales_crm, manufacturing_qa, news_claim_verification, payroll_benefits, procurement_vendor_risk, property_management, rag_passage_relevance, smart_home_iot, software_engineering, span_selection_extraction, telecom_support, trust_safety_moderation.
- **Question types:** noul 41%, choice 37%, score 22%.
- **Difficulty:** clear 39%, moderate 31%, borderline 16%, insufficient 9%, adversarial 6%.

About half of the scenarios are realistic multi-document cases where the correct label needs multi-step
reasoning: policy exceptions, later corrections, 2-hop lookups, and instructions planted in the text that must be
ignored. Each question is still a single literal judgment.

## Fields

| field | type | content |
|---|---|---|
| `id` | string | scenario id (`bNN-NNNN`) |
| `domain` | string | one of the 32 domains |
| `pattern` | string | fan-out, routing, detection, scoring, verification, extraction-choice, ranking-relevance, matching, guardrail |
| `state` | JSON string | the content to judge (string, object or array) |
| `questions` | JSON string | `{question_id: {"type": "choice"|"score"|"noul", "instructions": ..., "criteria": ...}}` |
| `targets` | JSON string | `{question_id: {"probabilities": {...}} or {"noul": p}, "difficulty": ..., "note": ...}}` |

- **Choice targets** are keyed by option.
- **Score targets** are keyed by level index (`"0"`…`"K-1"`).
- **Noul targets** give `noul` = P(yes).

```python
import json
from datasets import load_dataset

ds = load_dataset("mghafiri/decision-model-scenarios")
row = ds["train"][0]
state, questions, targets = (json.loads(row[k]) for k in ("state", "questions", "targets"))
```

## How the labels were made

1. **Authoring.** Claude (Anthropic) wrote each scenario and its targets. It followed a written rubric
   (`DATA_SPEC.md`): *the target probability of an option is the fraction of careful domain experts who would
   pick it*.
2. **Blind second annotation.** An independent Claude annotator relabelled every question without seeing the
   original labels. Agreement on the top answer was **9434/9718 = 97.1%**.
3. **Merge:**
   - Where the two agreed, the target is 0.6·author + 0.4·reviewer.
   - Borderline disagreements were averaged 50/50.
   - The remaining disagreements were adjudicated one by one, and defective questions were removed.
4. **Automated checks:**
   - schema and probability sums
   - near-duplicate states
   - synthetic-only personal data (example.com emails, 555-01xx phones, fictional organisations)
   - no arithmetic, counting or date-comparison questions

**Minimum achievable log-loss on these soft targets:** about 0.36, the mean target entropy.

## Limitations

- **The data is synthetic and LLM-authored.** The targets estimate expert agreement. They are not real-world
  outcomes.
- **English only.** The domains are business and operations workflows; there is no clinical or legal advice
  content.
- **Personal data is synthetic.** Any resemblance to real people or organisations is unintended.

## License

The data is released under the **MIT License**; see `LICENSE`.
