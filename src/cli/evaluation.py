"""CLI entrypoint for entity-level NER evaluation.

Usage::

    uv run python -m src.cli.eval_ner \\
        path/to/predictions.jsonl \\
        path/to/gold.jsonl \\
        [--mode strict|exact|partial] \\
        [--average micro|macro|both] \\
        [--input-format auto|multispan|tokentag]


    uv run python -m src.cli.eval_ner -r \\
        path/to/predictions_dir \\
        path/to/gold_dir \\
        [--mode strict|exact|partial] \\
        [--average micro|macro|both]


Recursive mode (``-r``) discovers ``<dataset_name>/predictions.jsonl`` files
under *pred* and evaluates them against ``<gold>/<dataset_name>/test.jsonl``.
Output is a progress bar with a final succeeded/skipped/failed summary.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from src.eval_ner import Evaluator, ResultFormatter

LOGGER = logging.getLogger(__name__)

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