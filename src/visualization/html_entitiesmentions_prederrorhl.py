"""Visualize prediction errors against ground-truth BIO labels."""

from __future__ import annotations

import argparse
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.utils import normalize_example_id

CORRECT_COLOR = "#DDF7E2"
WRONG_TYPE_COLOR = "#FFECCF"
WRONG_MENTION_COLOR = "#FFDCDD"
MISSED_MENTION_COLOR = "#D3D3D3"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Render sampled prediction-vs-truth BIO comparisons with displacy.",
    )
    parser.add_argument(
        "--true_path",
        type=Path,
        required=True,
        help="Path to ground-truth .jsonl file with id/tokens/labels.",
    )
    parser.add_argument(
        "--prediction_path",
        type=Path,
        required=True,
        help="Path to prediction .jsonl file with id/tokens(optional)/labels.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of randomly sampled examples to visualize.",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=Path("out/prediction_comparison.html"),
        help="Output HTML file path.",
    )
    parser.add_argument(
        "--sort",
        action="store_true",
        help="Sort examples by identifier before rendering/sampling.",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class EntitySpan:
    """One entity span in token-index space."""

    start: int
    end: int
    entity_type: str


@dataclass(frozen=True)
class ClassifiedSpan:
    """One predicted span with a display label and error category."""

    start: int
    end: int
    display_label: str
    category: str


def _validate_required_columns(
    table: pd.DataFrame,
    required_columns: set[str],
    table_name: str,
) -> None:
    """Raise an error when required columns are missing."""
    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {table_name}: "
            f"{sorted(missing_columns)}"
        )


def _normalize_label_sequence(labels: Any, token_count: int) -> list[str]:
    """Normalize labels to a BIO list with one entry per token."""
    normalized = _coerce_string_sequence(labels, field_name="labels")

    if len(normalized) >= token_count:
        return normalized[:token_count]

    return normalized + ["O"] * (token_count - len(normalized))


def _coerce_string_sequence(value: Any, field_name: str) -> list[str]:
    """Convert sequence-like values (including ndarray) to string lists."""
    if value is None or isinstance(value, (str, bytes)):
        raise ValueError(
            f"Column '{field_name}' must contain sequence values, not "
            f"{type(value).__name__}."
        )

    if isinstance(value, SequenceABC):
        return [str(item) for item in value]

    if hasattr(value, "tolist"):
        candidate = value.tolist()
        if isinstance(candidate, list):
            return [str(item) for item in candidate]

    raise ValueError(
        f"Column '{field_name}' must contain sequence values, got "
        f"{type(value).__name__}."
    )


def _extract_bio_spans(labels: Sequence[str]) -> list[EntitySpan]:
    """Convert BIO labels into token spans.

    Invalid transitions are handled conservatively by starting a new span.
    """
    spans: list[EntitySpan] = []
    current_start: int | None = None
    current_type: str | None = None

    def close_current(end_index: int) -> None:
        if current_start is None or current_type is None:
            return
        spans.append(
            EntitySpan(
                start=current_start,
                end=end_index,
                entity_type=current_type,
            )
        )

    for index, raw_tag in enumerate(labels):
        tag = str(raw_tag).strip()

        if not tag or tag.upper() == "O":
            close_current(index)
            current_start = None
            current_type = None
            continue

        if "-" in tag:
            prefix, entity_type = tag.split("-", 1)
        else:
            prefix, entity_type = "B", tag

        prefix = prefix.upper()
        entity_type = entity_type.strip()

        if not entity_type:
            close_current(index)
            current_start = None
            current_type = None
            continue

        if prefix == "B":
            close_current(index)
            current_start = index
            current_type = entity_type
            continue

        if prefix == "I":
            if current_start is None or current_type != entity_type:
                close_current(index)
                current_start = index
                current_type = entity_type
            continue

        close_current(index)
        current_start = index
        current_type = entity_type

    close_current(len(labels))
    return spans


def _classify_predicted_spans(
    true_spans: Sequence[EntitySpan],
    predicted_spans: Sequence[EntitySpan],
) -> list[ClassifiedSpan]:
    """Classify predicted spans against true spans.

    Categories:
    - correct: same mention boundaries and same type
    - wrong_type: same mention boundaries but different type
    - wrong_mention: mention boundaries not present in ground truth
    - missed_mention: ground-truth mention boundaries missing in predictions
    """
    true_exact = {
        (span.start, span.end, span.entity_type)
        for span in true_spans
    }
    true_by_bounds = {
        (span.start, span.end): span.entity_type for span in true_spans
    }

    classified: list[ClassifiedSpan] = []
    for span in predicted_spans:
        exact_key = (span.start, span.end, span.entity_type)
        boundary_key = (span.start, span.end)

        if exact_key in true_exact:
            classified.append(
                ClassifiedSpan(
                    start=span.start,
                    end=span.end,
                    display_label=span.entity_type,
                    category="correct",
                )
            )
            continue

        true_type = true_by_bounds.get(boundary_key)
        if true_type is not None:
            classified.append(
                ClassifiedSpan(
                    start=span.start,
                    end=span.end,
                    display_label=f"{span.entity_type} -> {true_type}",
                    category="wrong_type",
                )
            )
            continue

        classified.append(
            ClassifiedSpan(
                start=span.start,
                end=span.end,
                display_label=f"{span.entity_type} (FP)",
                category="wrong_mention",
            )
        )

    predicted_bounds = {
        (span.start, span.end)
        for span in predicted_spans
    }
    for true_span in true_spans:
        true_bounds = (true_span.start, true_span.end)
        if true_bounds in predicted_bounds:
            continue
        classified.append(
            ClassifiedSpan(
                start=true_span.start,
                end=true_span.end,
                display_label=f"{true_span.entity_type} (MISS)",
                category="missed_mention",
            )
        )

    return classified


def _build_text_and_offsets(tokens: Sequence[str]) -> tuple[str, list[tuple[int, int]]]:
    """Build whitespace-joined text and token-level character offsets."""
    token_strings = [str(token) for token in tokens]

    text = " ".join(token_strings)
    offsets: list[tuple[int, int]] = []

    cursor = 0
    for index, token in enumerate(token_strings):
        if index > 0:
            cursor += 1
        start = cursor
        end = start + len(token)
        offsets.append((start, end))
        cursor = end

    return text, offsets


def _build_manual_doc(
    example_id: str,
    tokens: Sequence[str],
    true_labels: Any,
    predicted_labels: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Create one manual displacy document and local color mapping."""
    token_count = len(tokens)
    normalized_true = _normalize_label_sequence(true_labels, token_count)
    normalized_pred = _normalize_label_sequence(predicted_labels, token_count)

    true_spans = _extract_bio_spans(normalized_true)
    predicted_spans = _extract_bio_spans(normalized_pred)
    classified_spans = _classify_predicted_spans(true_spans, predicted_spans)

    text, offsets = _build_text_and_offsets(tokens)
    entities: list[dict[str, Any]] = []
    colors: dict[str, str] = {}

    for span in classified_spans:
        start_char = offsets[span.start][0]
        end_char = offsets[span.end - 1][1]

        entities.append(
            {
                "start": start_char,
                "end": end_char,
                "label": span.display_label,
            }
        )

        if span.category == "correct":
            colors[span.display_label] = CORRECT_COLOR
        elif span.category == "wrong_type":
            colors[span.display_label] = WRONG_TYPE_COLOR
        elif span.category == "wrong_mention":
            colors[span.display_label] = WRONG_MENTION_COLOR
        else:
            colors[span.display_label] = MISSED_MENTION_COLOR

    return {
        "text": text,
        "ents": entities,
        "title": f"id: {example_id}",
    }, colors


def _load_aligned_examples(
    true_path: str | Path,
    prediction_path: str | Path,
) -> pd.DataFrame:
    """Load true/pred tables and align them by normalized id."""
    prediction_path_obj = Path(prediction_path)
    if prediction_path_obj.suffix != ".jsonl":
        raise ValueError("Prediction file must be a .jsonl file.")

    true_table = load_dataset_table(true_path)
    prediction_table = load_dataset_table(prediction_path)

    _validate_required_columns(true_table, {"id", "tokens", "labels"}, "truth")
    _validate_required_columns(prediction_table, {"id", "labels"}, "prediction")

    true_table = true_table.copy()
    prediction_table = prediction_table.copy()

    true_table["id_key"] = true_table["id"].apply(normalize_example_id)
    prediction_table["id_key"] = prediction_table["id"].apply(
        normalize_example_id
    )

    true_table = true_table.drop_duplicates(subset=["id_key"], keep="first")
    prediction_table = prediction_table.drop_duplicates(
        subset=["id_key"],
        keep="last",
    )

    merged = pd.merge(
        true_table[["id", "id_key", "tokens", "labels"]],
        prediction_table[["id_key", "labels"]],
        on="id_key",
        how="inner",
        suffixes=("_true", "_pred"),
    )

    if merged.empty:
        raise ValueError("No matching ids found between truth and prediction files.")

    return merged


def build_prediction_comparison_docs(
    true_path: str | Path,
    prediction_path: str | Path,
    n_samples: int = 5,
    random_state: int | None = None,
    sort: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, str], list[str]]:
    """Build sampled displacy-ready docs comparing predictions to truth."""
    if n_samples <= 0:
        raise ValueError("n_samples must be strictly positive.")

    merged = _load_aligned_examples(true_path, prediction_path)
    sampled_count = min(n_samples, len(merged))

    if sort:
        try:
            merged["_sort_key"] = pd.to_numeric(merged["id_key"], errors="coerce")
            merged = merged.sort_values(by=["_sort_key", "id_key"], na_position="last")
            merged = merged.drop(columns=["_sort_key"])
        except Exception:
            merged = merged.sort_values(by="id_key")
        sampled_table = merged.head(sampled_count)
    else:
        sampled_table = merged.sample(n=sampled_count, random_state=random_state)

    docs: list[dict[str, Any]] = []
    colors: dict[str, str] = {}
    sampled_ids: list[str] = []

    for row in sampled_table.itertuples(index=False):
        token_list = _coerce_string_sequence(row.tokens, field_name="tokens")

        example_id = normalize_example_id(row.id)
        sampled_ids.append(example_id)

        doc, doc_colors = _build_manual_doc(
            example_id=example_id,
            tokens=token_list,
            true_labels=row.labels_true,
            predicted_labels=row.labels_pred,
        )
        docs.append(doc)
        colors.update(doc_colors)

    return docs, colors, sampled_ids


def visualize_prediction_comparison(
    true_path: str | Path,
    prediction_path: str | Path,
    n_samples: int = 5,
    random_state: int | None = None,
    save_path: str | Path | None = None,
    sort: bool = False,
) -> tuple[Path, list[str]]:
    """Render a prediction-vs-truth displacy HTML page and save it."""
    from spacy import displacy

    docs, colors, sampled_ids = build_prediction_comparison_docs(
        true_path=true_path,
        prediction_path=prediction_path,
        n_samples=n_samples,
        random_state=random_state,
        sort=sort,
    )

    options = {"colors": colors}
    html = displacy.render(
        docs,
        style="ent",
        options=options,
        manual=True,
        jupyter=False,
        page=True,
    )

    output_path = Path(save_path) if save_path else Path("prediction_comparison.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        output_file.write(html)

    return output_path, sampled_ids


def run_visualize_predictions(
    true_path: Path,
    prediction_path: Path,
    n_samples: int,
    random_state: int | None,
    save_path: Path,
    sort: bool = False,
) -> None:
    """Run prediction comparison visualization with explicit parameters."""
    output_path, sampled_ids = visualize_prediction_comparison(
        true_path=true_path,
        prediction_path=prediction_path,
        n_samples=n_samples,
        random_state=random_state,
        save_path=save_path,
        sort=sort,
    )

    print(f"Saved visualization to: {output_path.resolve()}")
    print(f"Rendered samples: {len(sampled_ids)}")
    print("Sample ids:")
    for sample_id in sampled_ids:
        print(f"- {sample_id}")


def main() -> None:
    """Run prediction comparison visualization CLI."""
    args = parse_args()
    run_visualize_predictions(
        true_path=args.true_path,
        prediction_path=args.prediction_path,
        n_samples=args.n_samples,
        random_state=args.random_state,
        save_path=args.save_path,
        sort=args.sort,
    )


if __name__ == "__main__":
    main()
