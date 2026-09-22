import json
from pathlib import Path

import pytest

from jevlite.prompt import LETTER_LABELS, labels_for, render
from jevlite.schema import confidence, derive_answer, option_keys, target_distribution, validate_scenario

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("probs,expected", [
    ([0.88, 0.12, 0.0], 0.81),   # docs.typesafe.ai/api.md choice example (rounded probabilities)
    ([0.0, 0.57, 0.43], 0.35),   # docs.typesafe.ai/primitives/score.md table
    ([0.45, 0.55, 0.0], 0.33),   # same page, numeric-level counterexample
    ([0.0, 0.95, 0.05], 0.92),   # docs.typesafe.ai/api.md score example
    ([0.0, 0.89, 0.11], 0.84),
    ([1 / 3, 1 / 3, 1 / 3], 0.0),
])
def test_confidence_matches_docs(probs, expected):
    assert confidence(probs) == pytest.approx(expected, abs=0.011)


def test_score_answer_shape():
    q = {"type": "score", "instructions": "How frustrated?", "criteria": ["Calm", "Frustrated", "Very angry"]}
    a = derive_answer(q, [0.0, 0.95, 0.05])
    assert a["score"] == pytest.approx(1.05)
    assert a["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert set(a["probabilities"]) == {"0", "1", "2"}


def test_choice_answer_is_argmax_and_keys_match():
    q = {"type": "choice", "instructions": "Team?", "criteria": {"billing": "x", "technical": None, "sales": "y"}}
    a = derive_answer(q, [0.1, 0.7, 0.2])
    assert a["choice"] == "technical"
    assert list(a["probabilities"]) == option_keys(q)
    assert sum(a["probabilities"].values()) == pytest.approx(1.0)


def test_noul_answer():
    assert derive_answer({"type": "noul", "instructions": "?"}, [0.93, 0.07]) == {"type": "noul", "noul": 0.93}


def test_labels_are_letters_and_prompt_lists_every_option():
    q = {"type": "choice", "instructions": "Pick", "criteria": {f"opt{i}": None for i in range(30)}}
    assert labels_for(q) == LETTER_LABELS[:30]
    p = render("state", q)
    assert p.endswith("Answer:") and "A. opt0" in p and "Z. opt25" in p and "d. opt29" in p


def test_permuted_prompt_follows_order():
    q = {"type": "choice", "instructions": "Pick", "criteria": {"x": None, "y": None}}
    assert "A. y\nB. x" in render("s", q, order=["y", "x"])


def test_gold_examples_validate_and_align():
    for line in (ROOT / "data" / "gold.jsonl").read_text().splitlines():
        sc = json.loads(line)
        assert validate_scenario(sc) == []
        for qid, q in sc["questions"].items():
            dist = target_distribution(q, sc["targets"][qid])
            assert len(dist) == len(option_keys(q)) and sum(dist) == pytest.approx(1.0, abs=1e-3)
