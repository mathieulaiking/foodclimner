"""Writing of experiment output directories.

Provides the ``OutputWriter`` class which owns the standard experiment
output layout (``config.json``, ``stats.json``, ``predictions.jsonl``,
``raw_preds/``) described in AGENTS.md, plus the ``RawPredictionSink``
abstract class that serializes single raw model predictions to disk
(``.json`` or ``.txt``).
"""

from __future__ import annotations

import abc
import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any


class RawPredictionSink(abc.ABC):
    """Abstract base for serializing one raw model prediction to a file.

    Each concrete subclass defines the file ``extension`` (including the
    leading dot) and how the raw prediction payload is written.
    """

    extension: str

    @abc.abstractmethod
    def write(self, raw_dir: Path, doc_id: str, data: Any) -> None:
        """Write *data* for document *doc_id* into *raw_dir*."""
        ...


class JsonSink(RawPredictionSink):
    """Write raw predictions as indented JSON files."""

    extension = ".json"

    def write(self, raw_dir: Path, doc_id: str, data: Any) -> None:
        """Write *data* as ``<doc_id>.json``."""
        (raw_dir / f"{doc_id}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class TextSink(RawPredictionSink):
    """Write raw predictions as plain text files."""

    extension = ".txt"

    def write(self, raw_dir: Path, doc_id: str, data: Any) -> None:
        """Write ``str(data)`` as ``<doc_id>.txt``."""
        (raw_dir / f"{doc_id}.txt").write_text(str(data), encoding="utf-8")


class OutputWriter:
    """Own the standard experiment output directory tree.

    Parameters
    ----------
    output_dir:
        Path to the experiment output directory.
    sink:
        ``RawPredictionSink`` used for raw predictions. Defaults to
        ``TextSink`` for plain text output (e.g. LLM responses).
    """

    def __init__(self, output_dir: Path | str, sink: RawPredictionSink = TextSink()) -> None:
        self.output_dir = Path(output_dir)
        self.sink = sink
        self.raw_dir = self.output_dir / "raw_preds"

    def setup(self) -> Path:
        """Create the output directory tree.

        Returns
        -------
        Path to the ``raw_preds`` directory.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)
        return self.raw_dir

    def skipped_documents(self, doc_ids: list[str]) -> set[str]:
        """Return document IDs that already have non-empty raw prediction files.

        Parameters
        ----------
        doc_ids:
            List of document IDs to check.

        Returns
        -------
        Set of document IDs that already have non-empty raw prediction files.
        """
        if not self.raw_dir.is_dir():
            return set()
        existing_files: set[str] = {
            p.stem
            for p in self.raw_dir.iterdir()
            if p.suffix == self.sink.extension and p.stat().st_size > 0
        }
        existing = {doc_id for doc_id in doc_ids if doc_id in existing_files}
        if existing:
            pct = len(existing) * 100 // len(doc_ids)
            logging.info(
                "%d / %d documents skipped (%d%% already done)",
                len(existing),
                len(doc_ids),
                pct,
            )
        return existing

    def save_raw(self, doc_id: str, data: Any) -> None:
        """Save a single raw model prediction via the configured sink."""
        self.sink.write(self.raw_dir, doc_id, data)

    def save_prompt_template(self, template: str) -> None:
        """Write ``prompt_template.txt`` with the entity definitions prompt.

        The template contains entity classes and definitions but retains the
        ``[input_text]`` placeholder (not yet filled with document text).
        """
        (self.output_dir / "prompt_template.txt").write_text(template, encoding="utf-8")

    def save_predictions(self, predictions: list[dict[str, Any]]) -> None:
        """Write ``predictions.jsonl`` in NER-Annotated format.

        Each line is a JSON object with ``id``, ``text``, and ``entities``.
        """
        pred_path = self.output_dir / "predictions.jsonl"
        with pred_path.open("w", encoding="utf-8") as fh:
            for pred in predictions:
                fh.write(json.dumps(pred, ensure_ascii=False) + "\n")

    def load_predictions(self) -> dict[str, dict]:
        """Load existing ``predictions.jsonl`` into a ``{doc_id: record}`` dict.

        Returns an empty dict when the file does not exist.
        """
        pred_path = self.output_dir / "predictions.jsonl"
        if not pred_path.is_file():
            return {}
        predictions: dict[str, dict] = {}
        with pred_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                pred = json.loads(stripped)
                predictions[pred["id"]] = pred
        return predictions

    def update_predictions(self, predictions: list[dict[str, Any]]) -> int:
        """Merge *predictions* into the existing ``predictions.jsonl``.

        Existing records from a previous run are kept (resume-safe) and
        overwritten by same-ID predictions in *predictions*.

        Parameters
        ----------
        predictions:
            New predictions in NER-Annotated format.

        Returns
        -------
        Total number of prediction records after the merge.
        """
        existing = self.load_predictions()
        existing.update({p["id"]: p for p in predictions})
        self.save_predictions(list(existing.values()))
        return len(existing)

    def save_stats(
        self,
        total_time: float,
        num_examples: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write ``stats.json`` with timing and throughput information."""
        stats: dict[str, Any] = {
            "total_time_seconds": round(total_time, 3),
            "num_examples": num_examples,
            "examples_per_second": (
                round(num_examples / total_time, 3) if total_time > 0 else 0.0
            ),
        }
        if extra:
            stats.update(extra)
        self._write_json(self.output_dir / "stats.json", stats)

    def save_config(
        self,
        args: Namespace,
        model_info: dict[str, Any] | None = None,
    ) -> None:
        """Write ``config.json`` with CLI arguments and optional model metadata."""
        config: dict[str, Any] = vars(args)
        if model_info:
            config["model_info"] = model_info
        self._write_json(self.output_dir / "config.json", config)

    def save_gpu_trace(self, trace: list[dict[str, float]]) -> None:
        """Save GPU monitoring trace as JSON and optionally render a line plot.

        Parameters
        ----------
        trace:
            List of dicts with keys ``timestamp_relative_s``, ``memory_mib``,
            ``power_w``, and ``gpu_util_pct``.
        """
        self._write_json(self.output_dir / "gpu_trace.json", trace)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        if not trace:
            return

        timestamps = [r["timestamp_relative_s"] for r in trace]
        mem = [r["memory_mib"] for r in trace]
        power = [r["power_w"] for r in trace]
        util = [r["gpu_util_pct"] for r in trace]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        ax1.plot(timestamps, mem, color="tab:blue")
        ax1.set_ylabel("GPU Memory (MiB)")
        ax1.grid(True, alpha=0.3)

        ax2.plot(timestamps, power, color="tab:red")
        ax2.set_ylabel("Power (W)")
        ax2.grid(True, alpha=0.3)

        ax3.plot(timestamps, util, color="tab:green")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("GPU Util (%)")
        ax3.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.output_dir / "gpu_utilization.png", dpi=150)
        plt.close(fig)

    def _write_json(self, path: Path, data: Any) -> None:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
