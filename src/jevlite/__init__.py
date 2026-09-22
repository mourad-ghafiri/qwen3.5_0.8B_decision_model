"""jevlite: a Jev-style System One decision model (Choice / Score / Noul) on Qwen3.5-0.8B-Base."""

from .schema import confidence, derive_answer, option_keys

__all__ = ["confidence", "derive_answer", "option_keys", "SystemOne"]


def __getattr__(name):
    if name == "SystemOne":  # lazy: avoids importing torch for data-only scripts
        from .model import SystemOne
        return SystemOne
    raise AttributeError(name)
