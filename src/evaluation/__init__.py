"""Entity-level NER evaluation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.evaluation.evaluate_ner import (
        EvalCounts,
        EvalEntity,
        EvalResult,
        Evaluator,
        ResultFormatter,
    )

__all__ = [
    "EvalCounts",
    "EvalEntity",
    "EvalResult",
    "Evaluator",
    "ResultFormatter",
]


def __getattr__(name: str):
    """Lazily re-export from ``evaluate_ner`` (keeps ``python -m`` clean)."""
    if name in __all__:
        import importlib

        module = importlib.import_module("src.evaluation.evaluate_ner")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
