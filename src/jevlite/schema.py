"""Scenario and question schema checks plus Jev answer derivation.

The scenario format is documented in docs/jev_spec.md. Answers are always derived from probability
distributions here, never authored, so `choice`, `score`, `legend` and `confidence` stay consistent.
"""

from __future__ import annotations

import re
from typing import Any

QUESTION_TYPES = ("choice", "score", "noul")
DIFFICULTIES = ("clear", "moderate", "borderline", "insufficient", "adversarial")
PATTERNS = (
    "fan-out", "routing", "detection", "scoring", "verification",
    "extraction-choice", "ranking-relevance", "matching", "guardrail",
)
MAX_CHOICE_OPTIONS = 52  # A-Z, a-z single-token labels
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10
PROB_TOL = 1e-3
ID_RE = re.compile(r"^(b\d{2}|gold)-\d{4}$")
JSON_SCALAR = (str, dict, list, type(None))


def confidence(probs: list[float]) -> float:
    """TypeSafe's documented confidence: (K * p_max - 1) / (K - 1), clipped to [0, 1]."""
    k = len(probs)
    if k < 2:
        return 1.0
    return max(0.0, min(1.0, (k * max(probs) - 1) / (k - 1)))


def option_keys(question: dict) -> list[str]:
    """Ordered answer keys: choice option names, score level indices, or yes/no for noul."""
    t = question["type"]
    if t == "choice":
        return list(question["criteria"].keys())
    if t == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return ["yes", "no"]


def target_distribution(question: dict, target: dict) -> list[float]:
    """Target as a probability vector aligned with option_keys(question)."""
    if question["type"] == "noul":
        p = float(target["noul"])
        return [p, 1.0 - p]
    probs = target["probabilities"]
    return [float(probs[k]) for k in option_keys(question)]


def derive_answer(question: dict, probs: list[float]) -> dict[str, Any]:
    """Build a Jev-shaped answer from a probability vector aligned with option_keys(question)."""
    t = question["type"]
    if t == "noul":
        return {"type": "noul", "noul": round(probs[0], 4)}
    keys = option_keys(question)
    rounded = {k: round(p, 4) for k, p in zip(keys, probs)}
    conf = round(confidence(probs), 4)
    if t == "choice":
        best = keys[max(range(len(probs)), key=probs.__getitem__)]
        return {"type": "choice", "choice": best, "probabilities": rounded, "confidence": conf}
    legend = {str(i): _describe(level) for i, level in enumerate(question["criteria"])}
    score = sum(i * p for i, p in enumerate(probs))
    return {"type": "score", "score": round(score, 4), "legend": legend,
            "probabilities": rounded, "confidence": conf}


def _describe(level: Any) -> str:
    if isinstance(level, str):
        return level
    if isinstance(level, dict):
        for key in ("level", "name", "label", "definition"):
            if isinstance(level.get(key), str):
                return level[key]
    import json
    return json.dumps(level, ensure_ascii=False)


def validate_question(qid: str, q: Any) -> list[str]:
    errs: list[str] = []
    if not isinstance(q, dict):
        return [f"{qid}: question is not an object"]
    t = q.get("type")
    if t not in QUESTION_TYPES:
        return [f"{qid}: bad type {t!r}"]
    if "instructions" not in q or not isinstance(q["instructions"], JSON_SCALAR) or q["instructions"] in ("", None):
        errs.append(f"{qid}: missing instructions")
    crit = q.get("criteria")
    if t == "choice":
        if not isinstance(crit, dict) or not (2 <= len(crit) <= MAX_CHOICE_OPTIONS):
            errs.append(f"{qid}: choice criteria must be a map of 2..{MAX_CHOICE_OPTIONS} options")
        elif not all(isinstance(v, JSON_SCALAR) for v in crit.values()):
            errs.append(f"{qid}: choice option descriptions must be string/object/array/null")
    elif t == "score":
        if not isinstance(crit, list) or not (MIN_SCORE_LEVELS <= len(crit) <= MAX_SCORE_LEVELS):
            errs.append(f"{qid}: score criteria must be a list of {MIN_SCORE_LEVELS}..{MAX_SCORE_LEVELS} levels")
        elif any(isinstance(c, str) and c.strip().isdigit() for c in crit):
            errs.append(f"{qid}: score levels must describe situations, not bare numbers")
    elif crit is not None:
        if not isinstance(crit, dict) or not set(crit) <= {"true", "false"}:
            errs.append(f"{qid}: noul criteria may only have 'true'/'false'")
    return errs


def validate_target(qid: str, q: dict, tgt: Any) -> list[str]:
    errs: list[str] = []
    if not isinstance(tgt, dict):
        return [f"{qid}: target is not an object"]
    if tgt.get("difficulty") not in DIFFICULTIES:
        errs.append(f"{qid}: bad difficulty {tgt.get('difficulty')!r}")
    if not isinstance(tgt.get("note"), str) or not tgt["note"].strip():
        errs.append(f"{qid}: missing note")
    if q["type"] == "noul":
        p = tgt.get("noul")
        if not isinstance(p, (int, float)) or not (0.0 <= p <= 1.0):
            errs.append(f"{qid}: noul target must be a number in [0,1]")
        elif not (0.01 <= p <= 0.99):
            errs.append(f"{qid}: noul target must be within [0.01, 0.99]")
        return errs
    probs = tgt.get("probabilities")
    if not isinstance(probs, dict):
        return errs + [f"{qid}: missing probabilities"]
    keys = option_keys(q)
    if set(probs) != set(keys):
        return errs + [f"{qid}: probability keys {sorted(probs)} != options {keys}"]
    vals = [probs[k] for k in keys]
    if not all(isinstance(v, (int, float)) and 0.0 <= v <= 1.0 for v in vals):
        errs.append(f"{qid}: probabilities must be numbers in [0,1]")
    elif abs(sum(vals) - 1.0) > PROB_TOL:
        errs.append(f"{qid}: probabilities sum to {sum(vals):.4f}")
    elif max(vals) > 0.99 + PROB_TOL:
        errs.append(f"{qid}: top probability {max(vals)} > 0.99")
    return errs


def validate_scenario(sc: Any) -> list[str]:
    if not isinstance(sc, dict):
        return ["scenario is not an object"]
    sid = sc.get("id", "?")
    errs: list[str] = []
    if not isinstance(sid, str) or not ID_RE.match(sid):
        errs.append(f"bad id {sid!r}")
    if not isinstance(sc.get("domain"), str):
        errs.append("missing domain")
    if sc.get("pattern") not in PATTERNS:
        errs.append(f"bad pattern {sc.get('pattern')!r}")
    state = sc.get("state")
    if not isinstance(state, (str, dict, list)) or not state:
        errs.append("state must be a non-empty string/object/array")
    qs, ts = sc.get("questions"), sc.get("targets")
    if not isinstance(qs, dict) or not (3 <= len(qs) <= 6):
        errs.append("questions must be a map of 3..6 questions")
        return [f"{sid}: {e}" for e in errs]
    if not isinstance(ts, dict) or set(ts) != set(qs):
        errs.append("targets keys must equal questions keys")
        return [f"{sid}: {e}" for e in errs]
    for qid, q in qs.items():
        qerrs = validate_question(qid, q)
        errs += qerrs
        if not qerrs:
            errs += validate_target(qid, q, ts[qid])
    return [f"{sid}: {e}" for e in errs]
