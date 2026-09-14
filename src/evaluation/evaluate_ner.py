"""Entity-level NER evaluation with support for nested and discontinuous entities.

Provides the core evaluation API through a set of focused classes:
loading, format detection, matching, scoring, and output formatting.

Run as::

    python -m src.evaluation.evaluate_ner \\
        path/to/predictions.jsonl \\
        path/to/gold.jsonl \\
        [--mode strict|exact|partial]

    python -m src.evaluation.evaluate_ner -r \\
        path/to/predictions_dir \\
        path/to/gold_dir
"""

from __future__ import annotations

import abc
import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


# ── Evaluation schemas (owned by this module) ───────────────────────


@dataclass(frozen=True)
class EvalEntity:
    """One entity mention for strict evaluation.

    Contiguous mention: spans has one ``(start, end)`` pair.
    Discontinuous mention: spans has multiple ``(start, end)`` pairs,
    all belonging to the same logical entity.
    """

    type: str
    spans: tuple[tuple[int, int], ...]


@dataclass
class EvalCounts:
    """TP/FP/FN counts with computed precision, recall, F1."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        denom = p + r
        return 2 * p * r / denom if denom else 0.0


class MatchMode(abc.ABC):
    """Abstract matching strategy for entity comparison.

    Each concrete subclass implements a different matching criterion:
    * ``StrictMatch`` — exact type + exact spans.
    * ``ExactMatch`` — exact spans only, type ignored.
    * ``PartialMatch`` — any span overlap + exact type.
    """

    @abc.abstractmethod
    def matches(self, gold: EvalEntity, pred: EvalEntity) -> bool:
        ...

    @staticmethod
    def from_string(mode: str) -> MatchMode:
        """Resolve a mode string to a ``MatchMode`` instance."""
        mapping: dict[str, type[MatchMode]] = {
            "strict": StrictMatch,
            "exact": ExactMatch,
            "partial": PartialMatch,
        }
        cls = mapping.get(mode)
        if cls is None:
            raise ValueError(f"Unknown mode: {mode!r}. Choices: {list(mapping)}")
        return cls()


class StrictMatch(MatchMode):
    def matches(self, gold: EvalEntity, pred: EvalEntity) -> bool:
        return gold == pred


class ExactMatch(MatchMode):
    def matches(self, gold: EvalEntity, pred: EvalEntity) -> bool:
        return gold.spans == pred.spans


class PartialMatch(MatchMode):
    def matches(self, gold: EvalEntity, pred: EvalEntity) -> bool:
        return gold.type == pred.type and _spans_overlap(gold.spans, pred.spans)


def _spans_overlap(
    a: tuple[tuple[int, int], ...],
    b: tuple[tuple[int, int], ...],
) -> bool:
    """Return ``True`` if any interval in *a* overlaps any interval in *b*.

    Each interval is an inclusive-exclusive ``(start, end)`` pair.
    """
    for s1, e1 in a:
        for s2, e2 in b:
            if max(s1, s2) < min(e1, e2):
                return True
    return False



class Matcher:
    """Matches predicted entities against gold entities under a given mode.

    Both lists are sorted deterministically before matching. Each gold
    entity can be matched at most once (greedy 1-to-1 assignment).
    """

    def __init__(self, mode: MatchMode) -> None:
        self._mode = mode

    @staticmethod
    def _entity_key(e: EvalEntity) -> tuple:
        """Sort key for deterministic matching."""
        return (e.type, e.spans)

    def match(
        self,
        gold: list[EvalEntity],
        pred: list[EvalEntity],
    ) -> EvalCounts:
        """Run matching and return aggregated ``EvalCounts``."""
        gold_sorted = sorted(gold, key=self._entity_key)
        pred_sorted = sorted(pred, key=self._entity_key)

        counts = EvalCounts()
        matched_gold: set[int] = set()

        for p_ent in pred_sorted:
            found = False
            for g_idx, g_ent in enumerate(gold_sorted):
                if g_idx in matched_gold:
                    continue
                if self._mode.matches(g_ent, p_ent):
                    counts.tp += 1
                    matched_gold.add(g_idx)
                    found = True
                    break
            if not found:
                counts.fp += 1

        counts.fn = len(gold_sorted) - len(matched_gold)
        return counts



class DocumentLoader:
    """Reads JSONL files, detects document format, and loads entity type definitions."""

    @staticmethod
    def read_jsonl(path: Path) -> list[dict[str, Any]]:
        """Read a JSONL file and return the list of parsed records."""
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                records.append(json.loads(stripped))
        return records

    @staticmethod
    def detect_format(records: list[dict[str, Any]]) -> str:
        """Detect the document format from the keys of the first record.

        Returns ``"multispan"`` (has ``entities`` key) or
        ``"tokentag"`` (has ``labels`` key).
        """
        if not records:
            raise ValueError("Cannot detect format from empty file.")
        record = records[0]
        has_entities = "entities" in record
        has_labels = "labels" in record
        if has_entities and not has_labels:
            return "multispan"
        if has_labels and not has_entities:
            return "tokentag"
        raise ValueError(
            f"Unable to detect format. Record keys: {list(record.keys())}. "
            "Expected 'entities' (MultiSpanNERDocument) or 'labels' (TokenTagNERDocument)."
        )

    @staticmethod
    def load_entity_types(gold_path: Path) -> list[str] | None:
        """Find ``entities.json`` in the gold file's parent directory and return names.

        Returns ``None`` if the file does not exist.
        """
        entities_path = gold_path.parent / "entities.json"
        if entities_path.is_file():
            with entities_path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            return [e["name"] for e in raw]
        return None



class EntityConverter:
    """Converts JSONL records to lists of ``EvalEntity`` objects."""

    @staticmethod
    def from_record(
        record: dict[str, Any],
        fmt: str,
    ) -> list[EvalEntity]:
        """Convert a single JSONL record to a list of ``EvalEntity``.

        Parameters
        ----------
        record :
            A parsed JSON object from a JSONL file.
        fmt :
            ``"multispan"`` or ``"tokentag"``.

        Returns
        -------
        List of ``EvalEntity`` objects ready for evaluation.
        """
        if fmt == "multispan":
            raw = record.get("entities", [])
            entities: list[EvalEntity] = []
            for ent in raw:
                ent_type = str(ent["type"])
                raw_spans = ent.get("offsets", ent.get("spans"))
                spans = tuple((int(s), int(e)) for s, e in raw_spans)
                entities.append(EvalEntity(type=ent_type, spans=spans))
            return entities

        if fmt == "tokentag":
            tokens = record.get("tokens", [])
            labels = record.get("labels", [])
            return EntityConverter.from_bio(tokens, labels)

        raise ValueError(f"Unknown format: {fmt}")

    @staticmethod
    def from_bio(
        tokens: list[str],
        labels: list[str],
    ) -> list[EvalEntity]:
        """Convert BIO-labelled tokens to a list of contiguous ``EvalEntity``."""
        entities: list[EvalEntity] = []
        i = 0
        n = len(labels)
        while i < n:
            tag = labels[i]
            if tag == "O" or not tag.startswith("B-"):
                i += 1
                continue
            ent_type = tag[2:]
            start = i
            i += 1
            while i < n and labels[i] == f"I-{ent_type}":
                i += 1
            end = i - 1
            entities.append(EvalEntity(type=ent_type, spans=((start, end),)))
        return entities



class DocumentAligner:
    """Aligns gold and prediction records by document ID."""

    @staticmethod
    def align(
        gold_records: list[dict[str, Any]],
        pred_records: list[dict[str, Any]],
        gold_fmt: str,
        pred_fmt: str,
    ) -> tuple[list[list[EvalEntity]], list[list[EvalEntity]]]:
        """Align gold and prediction records by ``id`` and convert to entity lists."""
        gold_by_id: dict[str, dict[str, Any]] = {str(r["id"]): r for r in gold_records}
        pred_by_id: dict[str, dict[str, Any]] = {str(r["id"]): r for r in pred_records}

        common_ids = sorted(set(gold_by_id) & set(pred_by_id))

        if not common_ids:
            LOGGER.warning("No document IDs overlap between gold and predictions.")

        gold_docs: list[list[EvalEntity]] = []
        pred_docs: list[list[EvalEntity]] = []

        for doc_id in common_ids:
            gold_docs.append(EntityConverter.from_record(gold_by_id[doc_id], gold_fmt))
            pred_docs.append(EntityConverter.from_record(pred_by_id[doc_id], pred_fmt))

        return gold_docs, pred_docs



@dataclass
class EvalResult:
    """Aggregated evaluation result for a collection of documents."""

    mode: str
    average: str
    overall: EvalCounts
    per_type: dict[str, EvalCounts]
    type_order: list[str]



class Evaluator:
    """Orchestrates entity-level evaluation over aligned document pairs.

    Usage::

        evaluator = Evaluator(mode="strict", average="both")
        result = evaluator.evaluate(gold_docs, pred_docs, entity_types=types)

    Or for a full file-to-result pipeline::

        result = Evaluator.run_pair(pred_path, gold_path, mode="strict")
    """

    def __init__(
        self,
        mode: str | MatchMode = "strict",
        average: str = "both",
    ) -> None:
        self._mode: MatchMode = mode if isinstance(mode, MatchMode) else MatchMode.from_string(mode)
        self._average = average

    def evaluate(
        self,
        gold_docs: list[list[EvalEntity]],
        pred_docs: list[list[EvalEntity]],
        entity_types: list[str] | None = None,
    ) -> EvalResult:
        """Run entity-level evaluation over aligned document pairs.

        Parameters
        ----------
        gold_docs, pred_docs :
            Per-document entity lists, aligned by document index.
        entity_types :
            Optional list of valid entity types. Entities whose type is not
            in this list are filtered out before evaluation.

        Returns
        -------
        ``EvalResult`` with overall counts and per-type breakdown.
        """
        if len(gold_docs) != len(pred_docs):
            raise ValueError(
                f"Document count mismatch: gold={len(gold_docs)} vs pred={len(pred_docs)}"
            )

        if entity_types is None:
            seen: set[str] = set()
            for doc in gold_docs:
                seen.update(e.type for e in doc)
            for doc in pred_docs:
                seen.update(e.type for e in doc)
            entity_types = sorted(seen)

        overall = EvalCounts()
        per_type = {t: EvalCounts() for t in entity_types}
        matcher = Matcher(self._mode)

        for g_doc, p_doc in zip(gold_docs, pred_docs):
            g_clean = [e for e in g_doc if e.type in per_type]
            p_clean = [e for e in p_doc if e.type in per_type]

            doc_counts = matcher.match(g_clean, p_clean)
            overall.tp += doc_counts.tp
            overall.fp += doc_counts.fp
            overall.fn += doc_counts.fn

            for t in entity_types:
                g_t = [e for e in g_clean if e.type == t]
                p_t = [e for e in p_clean if e.type == t]
                dc = matcher.match(g_t, p_t)
                per_type[t].tp += dc.tp
                per_type[t].fp += dc.fp
                per_type[t].fn += dc.fn

        return EvalResult(
            mode=str(self._mode.__class__.__name__).replace("Match", "").lower(),
            average=self._average,
            overall=overall,
            per_type=per_type,
            type_order=entity_types,
        )

    @classmethod
    def run_pair(
        cls,
        pred_path: Path,
        gold_path: Path,
        mode: str = "strict",
        average: str = "both",
        input_format: str = "auto",
        quiet: bool = False,
    ) -> EvalResult | None:
        """Evaluate a single prediction/gold pair.

        Reads both files, detects formats, aligns by document ID, runs
        evaluation, writes ``results.csv`` into the prediction file's
        parent directory.

        Returns the ``EvalResult`` on success, or ``None`` if any error
        occurs (logged via ``LOGGER.exception``).
        """
        try:
            gold_records = DocumentLoader.read_jsonl(gold_path)
            pred_records = DocumentLoader.read_jsonl(pred_path)

            if not gold_records:
                raise ValueError(f"Gold file is empty: {gold_path}")
            if not pred_records:
                raise ValueError(f"Predictions file is empty: {pred_path}")

            if input_format == "auto":
                pred_fmt = DocumentLoader.detect_format(pred_records)
            else:
                pred_fmt = input_format

            gold_fmt = DocumentLoader.detect_format(gold_records)

            if not quiet:
                LOGGER.info("Gold format: %s | Predictions format: %s", gold_fmt, pred_fmt)

            gold_docs, pred_docs = DocumentAligner.align(
                gold_records, pred_records, gold_fmt, pred_fmt
            )

            if not gold_docs:
                LOGGER.warning("No overlapping document IDs — evaluation is empty.")

            entity_types = DocumentLoader.load_entity_types(gold_path)

            evaluator = cls(mode=mode, average=average)
            result = evaluator.evaluate(
                gold_docs, pred_docs, entity_types=entity_types,
            )

            out_dir = pred_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            csv_path = out_dir / "results.csv"
            ResultFormatter.write_csv(result, csv_path)

            return result
        except Exception:
            LOGGER.exception("Evaluation failed for pred=%s gold=%s", pred_path, gold_path)
            return None



class ResultFormatter:
    """Formats evaluation results as CSV or human-readable text."""

    CSV_HEADER = ["mode", "entity_type", "tp", "fp", "fn", "precision", "recall", "f1"]

    @staticmethod
    def _counts_to_row(mode: str, name: str, c: EvalCounts) -> list:
        """One CSV row for an ``EvalCounts``."""
        return [
            mode,
            name,
            c.tp,
            c.fp,
            c.fn,
            f"{c.precision:.4f}",
            f"{c.recall:.4f}",
            f"{c.f1:.4f}",
        ]

    @staticmethod
    def _compute_macro(
        per_type: dict[str, EvalCounts],
        type_order: list[str],
    ) -> tuple[float, float, float]:
        """Compute macro-averaged precision, recall, F1 across types."""
        precisions = [per_type[t].precision for t in type_order]
        recalls = [per_type[t].recall for t in type_order]
        f1s = [per_type[t].f1 for t in type_order]
        n = len(type_order)
        if n == 0:
            return 0.0, 0.0, 0.0
        return sum(precisions) / n, sum(recalls) / n, sum(f1s) / n

    @staticmethod
    def write_csv(result: EvalResult, path: Path) -> None:
        """Write overall + per-type + optional macro-avg row to a CSV file."""
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(ResultFormatter.CSV_HEADER)
            if result.average == "both":
                writer.writerow(ResultFormatter._counts_to_row(result.mode, "overall_micro", result.overall))
                mp, mr, mf = ResultFormatter._compute_macro(result.per_type, result.type_order)
                writer.writerow([
                    result.mode, "overall_macro", 0, 0, 0,
                    f"{mp:.4f}", f"{mr:.4f}", f"{mf:.4f}",
                ])
            elif result.average == "macro":
                writer.writerow(ResultFormatter._counts_to_row(result.mode, "overall", result.overall))
                mp, mr, mf = ResultFormatter._compute_macro(result.per_type, result.type_order)
                writer.writerow([
                    result.mode, "macro_avg", 0, 0, 0,
                    f"{mp:.4f}", f"{mr:.4f}", f"{mf:.4f}",
                ])
            else:
                writer.writerow(ResultFormatter._counts_to_row(result.mode, "overall", result.overall))
            for t in result.type_order:
                writer.writerow(ResultFormatter._counts_to_row(result.mode, t, result.per_type[t]))

    @staticmethod
    def print_result(result: EvalResult) -> None:
        """Print a human-readable summary of evaluation results to stdout."""
        overall = result.overall
        print(f"\nEntity-level NER evaluation ({result.mode}, {result.average})")
        print("=" * 50)

        if result.average == "both":
            mp, mr, mf = ResultFormatter._compute_macro(result.per_type, result.type_order)
            print("  Micro-averaged:")
            print(f"    Precision: {overall.precision:.4f}")
            print(f"    Recall:    {overall.recall:.4f}")
            print(f"    F1:        {overall.f1:.4f}")
            print(f"    TP: {overall.tp}  FP: {overall.fp}  FN: {overall.fn}")
            print("  Macro-averaged:")
            print(f"    Precision: {mp:.4f}")
            print(f"    Recall:    {mr:.4f}")
            print(f"    F1:        {mf:.4f}")
        elif result.average == "micro":
            print(f"  Precision: {overall.precision:.4f}")
            print(f"  Recall:    {overall.recall:.4f}")
            print(f"  F1:        {overall.f1:.4f}")
            print(f"  TP: {overall.tp}  FP: {overall.fp}  FN: {overall.fn}")
        else:
            mp, mr, mf = ResultFormatter._compute_macro(result.per_type, result.type_order)
            print(f"  Macro Precision: {mp:.4f}")
            print(f"  Macro Recall:    {mr:.4f}")
            print(f"  Macro F1:        {mf:.4f}")
            print(f"  (micro aggregated — TP: {overall.tp}  FP: {overall.fp}  FN: {overall.fn})")


__all__ = [
    "EvalCounts",
    "EvalEntity",
    "EvalResult",
    "MatchMode", "StrictMatch", "ExactMatch", "PartialMatch",
    "Matcher",
    "DocumentLoader",
    "EntityConverter",
    "DocumentAligner",
    "Evaluator",
    "ResultFormatter",
]


# ── CLI ───────────────────────────────────────────────────────────────

MODE_CHOICES = ["strict", "exact", "partial"]
AVG_CHOICES = ["micro", "macro", "both"]
FMT_CHOICES = ["auto", "multispan", "tokentag"]


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run entity-level NER evaluation with support for nested "
            "and discontinuous entities."
        ),
    )
    parser.add_argument(
        "pred",
        type=Path,
        help="Path to predictions JSONL (or directory in --recursive mode).",
    )
    parser.add_argument(
        "gold",
        type=Path,
        help="Path to gold-standard JSONL (or directory in --recursive mode).",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Recursive directory mode.",
    )
    parser.add_argument(
        "-m", "--mode",
        choices=MODE_CHOICES,
        default="strict",
        help="Evaluation mode (default: strict).",
    )
    parser.add_argument(
        "-a", "--average",
        choices=AVG_CHOICES,
        default="both",
        help="Averaging method (default: both).",
    )
    parser.add_argument(
        "-f", "--input-format",
        choices=FMT_CHOICES,
        default="auto",
        help="Input format for predictions file (default: auto).",
    )
    args = parser.parse_args()

    # Argument validation
    if args.recursive:
        if not args.pred.is_dir():
            raise NotADirectoryError(
                f"Prediction path must be a directory in recursive mode: {args.pred}"
            )
        if not args.gold.is_dir():
            raise NotADirectoryError(
                f"Gold path must be a directory in recursive mode: {args.gold}"
            )
    else:
        if not args.pred.is_file():
            raise FileNotFoundError(f"Predictions file not found: {args.pred}")
        if not args.gold.is_file():
            raise FileNotFoundError(f"Gold file not found: {args.gold}")

    return args


def main() -> None:
    """Run entity-level NER evaluation from CLI arguments."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_args()

    if args.recursive:
        _run_recursive(args)
    else:
        _run_single(args)


def _run_single(args: argparse.Namespace) -> None:
    """Run evaluation on a single prediction/gold file pair."""
    result = Evaluator.run_pair(
        args.pred,
        args.gold,
        mode=args.mode,
        average=args.average,
        input_format=args.input_format,
    )
    if result is None:
        raise RuntimeError(f"Evaluation failed for {args.pred}")

    ResultFormatter.print_result(result)
    out_path = args.pred.parent / "results.csv"
    print(f"\nResults saved to {out_path}")


def _run_recursive(args: argparse.Namespace) -> None:
    """Discover dataset directories and evaluate each one.

    ``pred`` is a directory tree containing ``<dataset>/predictions.jsonl``.
    ``gold`` is a directory tree containing ``<dataset>/test.jsonl`` (and
    ``entities.json``).
    """
    # Discover all dataset directories by their predictions.jsonl files
    pred_files = sorted(args.pred.rglob("**/predictions.jsonl"))
    if not pred_files:
        LOGGER.warning("No predictions.jsonl files found under %s", args.pred)
        return

    pairs: list[tuple[Path, Path, str]] = []
    for pf in pred_files:
        dataset_name = pf.parent.name
        gold_test = args.gold / dataset_name / "test.jsonl"
        if not gold_test.is_file():
            LOGGER.warning(
                "Skipping %s: gold file not found at %s", dataset_name, gold_test
            )
            continue
        pairs.append((pf, gold_test, dataset_name))

    if not pairs:
        LOGGER.warning("No valid prediction/gold pairs found.")
        return

    succeeded = 0
    skipped = 0
    failed = 0

    tqdm_bar = tqdm(pairs, desc="Evaluating", unit="dataset")
    for pred_path, gold_path, dataset_name in tqdm_bar:
        out_csv = pred_path.parent / "results.csv"
        if out_csv.is_file():
            skipped += 1
            tqdm_bar.set_description(f"Evaluating (skipped: {skipped})")
            continue

        result = Evaluator.run_pair(
            pred_path,
            gold_path,
            mode=args.mode,
            average=args.average,
            input_format=args.input_format,
            quiet=True,
        )
        if result is None:
            failed += 1
        else:
            succeeded += 1

    tqdm_bar.close()

    parts: list[str] = []
    if succeeded:
        parts.append(f"{succeeded} succeeded")
    if skipped:
        parts.append(f"{skipped} skipped")
    if failed:
        parts.append(f"{failed} failed")
    print(f"Done: {', '.join(parts)}")


if __name__ == "__main__":
    main()
