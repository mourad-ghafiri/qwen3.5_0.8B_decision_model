"""Logit readout: turn the model's next-token logits over answer labels into Jev-style answers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .prompt import LETTER_LABELS, NOUL_LABELS, labels_for, render
from .schema import derive_answer, option_keys

DEFAULT_MAX_TOKENS = 3072


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def label_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    def one(label: str) -> int:
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"label {label!r} is not a single token: {ids}")
        return ids[0]
    return [one(l) for l in LETTER_LABELS], [one(l) for l in NOUL_LABELS]


def last_hidden(model, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Hidden state at each row's last real token. Rows are right-padded, so causal attention keeps pads
    from affecting earlier positions (including the linear-attention recurrent state)."""
    h = model.model(input_ids=input_ids, attention_mask=(torch.arange(input_ids.shape[1], device=input_ids.device)[None]
                                                         < lengths[:, None]).long(), use_cache=False).last_hidden_state
    return h[torch.arange(h.shape[0], device=h.device), lengths - 1]


def label_logits(model, hidden: torch.Tensor, label_ids: list[list[int]]) -> list[torch.Tensor]:
    """Per-row logits restricted to that row's allowed label tokens, without computing the full vocabulary."""
    weight = model.get_output_embeddings().weight
    out = []
    for i, ids in enumerate(label_ids):
        w = weight[torch.tensor(ids, device=hidden.device)].to(hidden.dtype)
        out.append((hidden[i] @ w.T).float())
    return out


def pad_batch(token_lists: list[list[int]], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(len(t) for t in token_lists)
    ids = torch.full((len(token_lists), n), pad_id, dtype=torch.long)
    for i, t in enumerate(token_lists):
        ids[i, :len(t)] = torch.tensor(t)
    lengths = torch.tensor([len(t) for t in token_lists])
    return ids.to(device), lengths.to(device)


def truncate_state_tokens(tokenizer, prompt_ids: list[int], max_tokens: int) -> list[int]:
    """Keep the head of the prompt (state) and the whole tail (question + options) when over budget."""
    if len(prompt_ids) <= max_tokens:
        return prompt_ids
    keep_tail = max_tokens // 2
    return prompt_ids[: max_tokens - keep_tail] + prompt_ids[-keep_tail:]


class SystemOne:
    """Local Jev-compatible evaluator: `system_one(state, questions)` returns {model, answers, usage}."""

    def __init__(self, model_path: str, device: str | None = None, dtype=torch.bfloat16,
                 max_tokens: int = DEFAULT_MAX_TOKENS, batch_size: int = 8):
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(self.device).eval()
        self.letter_ids, self.noul_ids = label_token_ids(self.tokenizer)
        self.max_tokens, self.batch_size = max_tokens, batch_size
        self.name = Path(model_path).name
        cal = Path(model_path) / "calibration.json"
        self.temperature = json.loads(cal.read_text()) if cal.exists() else {}

    def allowed_ids(self, question: dict) -> list[int]:
        return self.noul_ids if question["type"] == "noul" else self.letter_ids[: len(labels_for(question))]

    @torch.no_grad()
    def distributions(self, items: list[tuple[Any, dict, list[str] | None]]) -> tuple[list[list[float]], int]:
        """Probability vectors (aligned with option_keys, or with `order` if given) for (state, question, order)."""
        encoded = [truncate_state_tokens(self.tokenizer,
                                         self.tokenizer.encode(render(s, q, order), add_special_tokens=False),
                                         self.max_tokens) for s, q, order in items]
        results: list[list[float]] = [None] * len(items)  # type: ignore[list-item]
        idx = sorted(range(len(items)), key=lambda i: len(encoded[i]))
        for start in range(0, len(idx), self.batch_size):
            chunk = idx[start:start + self.batch_size]
            ids, lengths = pad_batch([encoded[i] for i in chunk], self.tokenizer.pad_token_id, self.device)
            hidden = last_hidden(self.model, ids, lengths)
            logits = label_logits(self.model, hidden, [self.allowed_ids(items[i][1]) for i in chunk])
            for i, lg in zip(chunk, logits):
                t = self.temperature.get(items[i][1]["type"], 1.0)
                results[i] = F.softmax(lg / t, dim=-1).tolist()
        return results, sum(len(e) for e in encoded)

    def system_one(self, state: Any, questions: dict[str, dict]) -> dict:
        qids = list(questions)
        probs, n_in = self.distributions([(state, questions[q], None) for q in qids])
        answers = {q: derive_answer(questions[q], p) for q, p in zip(qids, probs)}
        return {"model": self.name, "answers": answers, "usage": {"input_tokens": n_in, "output_tokens": 0}}

    def permuted_choice(self, state: Any, question: dict, order: list[str]) -> list[float]:
        """Choice probabilities with options presented in `order`, mapped back to option_keys order."""
        (p,), _ = self.distributions([(state, question, order)])
        by_key = dict(zip(order, p))
        return [by_key[k] for k in option_keys(question)]

    def timed(self, state: Any, questions: dict[str, dict]) -> tuple[dict, float]:
        t = time.perf_counter()
        out = self.system_one(state, questions)
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()
        return out, time.perf_counter() - t
