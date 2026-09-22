"""Render one (state, question) pair into the model's plain-text prompt.

Every question is rendered on its own row, because Jev evaluates questions independently against the same
state. The prompt ends with "Answer:" and the model's next token is one of the answer labels:
" A".." Z", " a".." z" for Choice options and Score levels, and " yes" / " no" for Noul. All of these are
single tokens in the Qwen3.5 tokenizer.
"""

from __future__ import annotations

import json
import string
from typing import Any

LETTER_LABELS = [" " + c for c in string.ascii_uppercase + string.ascii_lowercase]
NOUL_LABELS = [" yes", " no"]


def render_value(value: Any, indent: int | None = None) -> str:
    """Strings verbatim; JSON objects/arrays serialized (compact by default)."""
    if isinstance(value, str):
        return value
    if indent is None:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(value, ensure_ascii=False, indent=indent)


def labels_for(question: dict) -> list[str]:
    t = question["type"]
    if t == "noul":
        return NOUL_LABELS
    n = len(question["criteria"])
    if n > len(LETTER_LABELS):
        raise ValueError(f"{n} options exceeds the {len(LETTER_LABELS)} single-token labels; chunk in code")
    return LETTER_LABELS[:n]


def render(state: Any, question: dict, order: list[str] | None = None, indent: int | None = None) -> str:
    """Build the prompt.

    `order` optionally permutes Choice options (a list of option keys). Probabilities read from the labels
    must then be mapped back through the same order. Score levels are never permuted, because they are
    ordered.
    """
    t = question["type"]
    parts = [
        "### State",
        render_value(state, indent),
        "### Question",
        render_value(question["instructions"], indent),
    ]
    if t == "choice":
        keys = order or list(question["criteria"])
        parts.append("### Options (pick exactly one)")
        for label, key in zip(LETTER_LABELS, keys):
            desc = question["criteria"][key]
            parts.append(f"{label.strip()}. {key}" + ("" if desc is None else f": {render_value(desc)}"))
    elif t == "score":
        parts.append("### Levels (pick the one that best describes the state)")
        for label, level in zip(LETTER_LABELS, question["criteria"]):
            parts.append(f"{label.strip()}. {render_value(level)}")
    else:
        crit = question.get("criteria") or {}
        parts.append("### Answer yes or no")
        if crit.get("true") is not None:
            parts.append(f"yes: {render_value(crit['true'])}")
        if crit.get("false") is not None:
            parts.append(f"no: {render_value(crit['false'])}")
    parts.append("Answer:")
    return "\n".join(parts)
