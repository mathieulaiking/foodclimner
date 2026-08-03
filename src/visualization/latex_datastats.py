from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate LaTeX statistics tables from stats.json.",
    )
    parser.add_argument(
        "--stats_path",
        type=Path,
        required=True,
        help="Path to stats.json file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output LaTeX file path "
            "(default: {stats_path.parent}/{stats_path.stem}_table.tex)"
        ),
    )
    parser.add_argument(
        "--corpus-caption",
        type=str,
        default=None,
        help="Caption for the corpus stats table.",
    )
    parser.add_argument(
        "--entities-caption",
        type=str,
        default=None,
        help="Caption for the entity type stats table.",
    )
    parser.add_argument(
        "--corpus-label",
        type=str,
        default="tab:corpus-stats",
        help="Label for the corpus stats table (default: tab:corpus-stats).",
    )
    parser.add_argument(
        "--entities-label",
        type=str,
        default="tab:ent-stats",
        help="Label for the entity type stats table (default: tab:ent-stats).",
    )
    return parser.parse_args()


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters in plain text."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(c, c) for c in text)


def _format_entity_type(name: str) -> str:
    """Convert snake_case entity type name to Title Case for display."""
    return " ".join(word.capitalize() for word in name.split("_"))


def _format_mean_std(mean: float, std: float) -> str:
    """Format a mean and standard deviation as a LaTeX math expression."""
    return f"${mean:.1f} \\pm {std:.0f}$"


def _build_corpus_sections(
    stats: dict[str, Any],
) -> list[list[tuple[str, str]]]:
    """Build corpus rows grouped into three ``\\hline``-separated sections.

    Section 1 — document statistics
    Section 2 — mention / entity-type statistics
    Section 3 — NER structural stats (discontinuous / nested / flat).
    """
    sections: list[list[tuple[str, str]]] = []

    # --- Section 1: Document statistics ---
    doc_rows: list[tuple[str, str]] = [
        ("Number of sentences", str(stats["num_documents"])),
    ]
    doc_len_chars = stats.get("document_length_chars")
    if doc_len_chars:
        doc_rows.append((
            "Sent. avg length (char)",
            _format_mean_std(doc_len_chars["mean"], doc_len_chars["std"]),
        ))
    doc_len_words = stats.get("document_length_words")
    if doc_len_words:
        doc_rows.append((
            "Sent. avg length (word)",
            _format_mean_std(doc_len_words["mean"], doc_len_words["std"]),
        ))
    sections.append(doc_rows)

    # --- Section 2: Mention / entity-type statistics ---
    mention_rows: list[tuple[str, str]] = [
        ("Number of entity types", str(stats["num_entity_types"])),
        ("Number of mentions", str(stats["num_mentions"])),
        ("Number of unique mentions", str(stats["num_unique_mentions"])),
    ]
    mention_len_chars = stats.get("mention_length_chars")
    if mention_len_chars:
        mention_rows.append((
            "Mention avg length (char)",
            _format_mean_std(mention_len_chars["mean"], mention_len_chars["std"]),
        ))
    mention_len_words = stats.get("mention_length_words")
    if mention_len_words:
        mention_rows.append((
            "Mention avg length (word)",
            _format_mean_std(mention_len_words["mean"], mention_len_words["std"]),
        ))
    mentions_per_doc = stats.get("mentions_per_document")
    if mentions_per_doc:
        mention_rows.append((
            "Mentions per sentence",
            f"{mentions_per_doc['mean']:.2f}",
        ))
    sections.append(mention_rows)

    # --- Section 3: NER structural statistics ---
    ner_rows: list[tuple[str, str]] = []
    for key in ("discontinuous", "nested", "flat"):
        entry = stats.get(key, {})
        count = entry.get("count", 0)
        pct = entry.get("percentage", 0.0)
        label = f"{key.capitalize()} mentions"
        value = f"{count} ({pct:.0f}\\%)"
        ner_rows.append((label, value))
    sections.append(ner_rows)

    return sections


def generate_corpus_table(stats: dict[str, Any], caption: str, label: str) -> str:
    """Generate a LaTeX ``table*`` environment for global corpus statistics."""
    sections = _build_corpus_sections(stats)

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{lr}",
        "\\hline",
        "\\textbf{Statistic} & \\textbf{Value} \\\\",
        "\\hline",
    ]
    for si, section in enumerate(sections):
        if si > 0:
            lines.append("\\hline")
        for stat_name, stat_value in section:
            lines.append(f"{_escape_latex(stat_name)} & {stat_value} \\\\")
    lines.extend([
        "\\hline",
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def _build_entities_rows(
    stats: dict[str, Any],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Build food and climate entity rows grouped and independently sorted by count desc."""
    per_type = stats.get("per_type", {})

    def _to_row(type_name: str, type_stats: dict) -> tuple[str, str]:
        count = type_stats.get("count", 0)
        pct = type_stats.get("percentage", 0.0)
        return (
            _format_entity_type(type_name),
            f"{count}({pct:.0f}\\%)",
        )

    food_rows = sorted(
        (_to_row(n, s) for n, s in per_type.items() if n.startswith("food_")),
        key=lambda r: int(r[1].split("(")[0]),
        reverse=True,
    )
    climate_rows = sorted(
        (
            _to_row(n, s)
            for n, s in per_type.items()
            if n.startswith("climate_") or n == "environmental_concern"
        ),
        key=lambda r: int(r[1].split("(")[0]),
        reverse=True,
    )
    return food_rows, climate_rows


def generate_entities_table(stats: dict[str, Any], caption: str, label: str) -> str:
    """Generate a LaTeX ``table*`` environment for per-entity-type statistics."""
    food_rows, climate_rows = _build_entities_rows(stats)

    if not food_rows and not climate_rows:
        return "% No per-type statistics available."

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\begin{tabular}{lr}",
        "\\hline",
        "\\textbf{Entity Type} & \\textbf{Count} \\\\",
        "\\hline",
    ]
    for group_rows, comment in (
        (food_rows, "% --- Food practices ---"),
        (climate_rows, "% --- Climate related ---"),
    ):
        lines.append(comment)
        for type_name, count_str in group_rows:
            escaped_type = _escape_latex(type_name)
            lines.append(f"{escaped_type} & {count_str} \\\\")
        lines.append("\\hline")
    lines.extend([
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def main() -> None:
    """Load stats.json and write the LaTeX output file."""
    args = parse_args()

    if not args.stats_path.is_file():
        raise SystemExit(
            f"Error: '{args.stats_path}' is not a file or does not exist."
        )

    with open(args.stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    dataset_name = args.stats_path.parent.name

    corpus_caption = (
        args.corpus_caption
        or f"Corpus statistics for {dataset_name}."
    )
    entities_caption = (
        args.entities_caption
        or f"Entity type statistics for {dataset_name}."
    )

    corpus_table = generate_corpus_table(
        stats, caption=corpus_caption, label=args.corpus_label,
    )
    entities_table = generate_entities_table(
        stats, caption=entities_caption, label=args.entities_label,
    )

    latex_content = f"{corpus_table}\n\n{entities_table}\n"

    output_path = args.output or (
        args.stats_path.parent / f"{args.stats_path.stem}_table.tex"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex_content, encoding="utf-8")
    print(f"LaTeX statistics tables written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
