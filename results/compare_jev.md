# Jev vs our model on unseen scenarios

- Split: **test**, 150 scenarios, 714 questions scored by both (0 Jev failures, 0 of ours)
- Jev model: `jev-1.13.0` · our model: `decision-model` on `mps`
- Jev input tokens: 128,148 (estimated $0.0054)
- The targets are our held-out expert-agreement labels. See the caveats in docs/compare_jev.md.

## Overall

| | Jev | ours |
|---|---|---|
| accuracy | 0.931 | 0.811 |
| log-loss (lower is better) | 0.957 | 0.615 |
| Brier (lower is better) | 0.042 | 0.158 |
| latency p50 / p95, ms per request | 1314 / 2268 (incl. network) | 1661 / 5243 (local) |

## By question type

| type | n | accuracy Jev / ours | log-loss Jev / ours | Brier Jev / ours | ECE Jev / ours |
|---|---|---|---|---|---|
| choice | 261 | 0.939 / 0.812 | 1.312 / 0.702 | 0.051 / 0.191 | 0.088 / 0.037 |
| noul | 288 | 0.941 / 0.816 | 0.314 / 0.468 | 0.022 / 0.141 | 0.018 / 0.026 |
| score | 165 | 0.903 / 0.800 | 1.517 / 0.734 | 0.062 / 0.134 | 0.118 / 0.059 |

## Head to head (same questions)

- Same top answer: 82.5%
- Only ours correct: 17 · only Jev correct: 103 · both: 562 · neither: 32
- Lower log-loss per question: ours 394 · Jev 320
- Mean log-loss difference (Jev − ours): 0.3417, 95% CI [0.2690, 0.4127]. A positive value means ours is better.

## By difficulty

| difficulty | n | accuracy Jev / ours | log-loss Jev / ours |
|---|---|---|---|
| adversarial | 55 | 1.000 / 0.909 | 0.645 / 0.433 |
| borderline | 105 | 0.714 / 0.610 | 1.651 / 0.943 |
| clear | 278 | 0.996 / 0.899 | 0.493 / 0.414 |
| insufficient | 61 | 0.836 / 0.803 | 0.693 / 0.606 |
| moderate | 215 | 0.963 / 0.772 | 1.371 / 0.764 |

## By domain

| domain | n | accuracy Jev / ours | log-loss Jev / ours |
|---|---|---|---|
| agent_tool_routing | 23 | 0.913 / 0.478 | 0.783 / 0.809 |
| airline_travel_ops | 20 | 0.900 / 0.850 | 0.893 / 0.593 |
| aml_kyc_compliance | 18 | 1.000 / 1.000 | 0.640 / 0.357 |
| app_marketplace_policy | 21 | 1.000 / 0.905 | 0.978 / 0.584 |
| banking_fintech | 21 | 0.905 / 1.000 | 0.824 / 0.545 |
| citation_verification | 22 | 0.909 / 0.682 | 0.979 / 0.699 |
| clinical_literature_screening | 24 | 0.917 / 0.667 | 0.807 / 0.634 |
| customer_support | 23 | 0.913 / 0.739 | 1.089 / 0.931 |
| cybersecurity_soc | 24 | 0.917 / 0.833 | 1.056 / 0.460 |
| ecommerce | 20 | 1.000 / 0.900 | 1.512 / 0.568 |
| education_admissions | 22 | 0.955 / 0.864 | 0.919 / 0.541 |
| energy_utilities | 21 | 1.000 / 0.905 | 1.250 / 0.489 |
| entity_matching | 21 | 0.762 / 0.714 | 0.628 / 0.660 |
| government_services | 25 | 0.920 / 0.760 | 1.458 / 0.695 |
| healthcare_admin | 26 | 1.000 / 0.769 | 0.707 / 0.684 |
| hr_recruiting | 22 | 1.000 / 0.955 | 0.940 / 0.423 |
| insurance_claims | 22 | 1.000 / 0.818 | 0.966 / 0.536 |
| it_helpdesk | 30 | 0.833 / 0.633 | 0.836 / 0.946 |
| legal_compliance | 21 | 0.952 / 1.000 | 0.897 / 0.390 |
| llm_guardrails | 20 | 0.850 / 0.800 | 0.653 / 0.527 |
| logistics_sales_crm | 19 | 0.947 / 0.895 | 0.925 / 0.490 |
| manufacturing_qa | 27 | 0.926 / 0.852 | 0.843 / 0.504 |
| news_claim_verification | 25 | 0.960 / 0.840 | 1.048 / 0.577 |
| payroll_benefits | 24 | 0.875 / 0.917 | 0.977 / 0.627 |
| procurement_vendor_risk | 22 | 0.864 / 0.727 | 0.873 / 0.675 |
| property_management | 27 | 0.926 / 0.741 | 1.159 / 0.582 |
| rag_passage_relevance | 21 | 1.000 / 0.857 | 0.637 / 0.411 |
| smart_home_iot | 17 | 0.941 / 0.941 | 1.029 / 0.532 |
| software_engineering | 21 | 0.952 / 0.810 | 0.748 / 0.624 |
| span_selection_extraction | 18 | 0.944 / 0.722 | 1.401 / 0.934 |
| telecom_support | 26 | 0.923 / 0.731 | 1.201 / 0.772 |
| trust_safety_moderation | 21 | 0.952 / 0.857 | 0.921 / 0.671 |
