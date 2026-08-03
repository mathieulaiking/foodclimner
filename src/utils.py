"""Shared string and path helpers for NER experiments.

Contains small, stateless utilities used across the codebase.

Dataset loading is handled by :class:`src.dataset_loader.DatasetLoader`
and experiment output writing by :class:`src.output_writer.OutputWriter`.
"""

from __future__ import annotations

import os
from typing import Any


def normalize_example_id(example_id: Any) -> str:
    """Return a stable string identifier for one example."""
    return str(example_id)


def strip_extension(filename: str) -> str:
    """Return filename stem by removing one extension level."""
    stem, ext = os.path.splitext(str(filename))
    if ext:
        return stem
    return str(filename)
