from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPERIMENT_ORDER = [
    "baselines",
    "prompting",
]

MODEL_INFO: dict[str, str] = {
    # baselines
    "gliner_large-v2.5": "GLiNER",
    "GoLLIE-7B": "GoLLIE-7B",
    "GoLLIE-34B": "GoLLIE-34B",
    # prompting models
    "gemma-4-31B-it": "Gemma4",
    "Qwen3.5-9B": "Qwen3.5-9B",
    "Qwen3.5-27B": "Qwen3.5-27B",
    "Qwen3.6-27B": "Qwen3.6",
    # proprietary
    "claude-opus-4.8": "Claude Opus 4.8",
    "gpt-5.6-sol": "GPT5.6-Sol",
    "glm-5.2": "GLM 5.2",
}

EVAL_MODES = ["partial", "strict"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate LaTeX performance table from experiment results.",
    )
    parser.add_argument(
        "--exp_dir",
        type=Path,
        required=True,
        help="Directory containing [exp_type]/[model]/[dataset]/results.csv "
        "(default) or the dataset directory itself in --entity_types mode",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="Dataset subdirectory name (e.g., sf4cd-v0.4). "
        "Required unless --entity_types is set",
    )
    parser.add_argument(
        "--entity_types",
        action="store_true",
        default=False,
        help="Generate a per-entity-type table instead of an overall table. "
        "In this mode --exp_dir points to the dataset directory and "
        "--dataset_name must not be provided",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output LaTeX file path (default: {exp_dir}/table.tex "
        "or {exp_dir}/entity_type_table.tex in --entity_types mode)",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=None,
        help="LaTeX table caption (auto-generated if omitted)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="LaTeX table label (auto-generated if omitted)",
    )
    args = parser.parse_args()

    # --- mode-aware validation ---
    if args.entity_types:
        if args.dataset_name is not None:
            parser.error(
                "--dataset_name must not be provided when --entity_types is set"
            )
        if not args.exp_dir.is_dir():
            parser.error(
                f"--exp_dir '{args.exp_dir}' is not a directory or does not exist"
            )
        csv_found = list(args.exp_dir.glob("results.csv"))
        if not csv_found:
            parser.error(f"--exp_dir '{args.exp_dir}' contains no results.csv file")
    else:
        if args.dataset_name is None:
            parser.error("--dataset_name is required when --entity_types is not set")
        if not args.exp_dir.is_dir():
            parser.error(
                f"--exp_dir '{args.exp_dir}' is not a directory or does not exist"
            )

    return args


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


def find_results(
    exp_dir: Path, dataset_name: str
) -> list[tuple[str, str, Path]]:
    """Find all ``results.csv`` files under ``exp_dir/*/*/{dataset_name}/``.

    Returns
        List of ``(experiment_type, model_name, csv_path)`` tuples.
    """
    results: list[tuple[str, str, Path]] = []
    for model_dir in sorted(exp_dir.glob(f"*/*/{dataset_name}/")):
        exp_type = model_dir.parent.parent.name
        model_name = model_dir.parent.name
        csv_path = model_dir / "results.csv"
        if csv_path.is_file():
            results.append((exp_type, model_name, csv_path))
    return results


def read_metrics(results_path: Path, mode: str | None = None) -> tuple[float, float, float]:
    """Read the ``overall_micro`` precision / recall / F1 from a results CSV.

    Parameters
    ----------
    results_path :
        Path to the results CSV file.
    mode :
        If set, only consider rows whose ``mode`` column matches this value.

    Returns
        ``(precision, recall, f1)`` as floats in [0, 1].

    Raises
        ValueError
            When no ``overall_micro`` row is found for the given mode.
    """
    with results_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if mode is not None and row.get("mode") != mode:
                continue
            if row.get("entity_type") == "overall_micro":
                return float(row["precision"]), float(row["recall"]), float(row["f1"])
    raise ValueError(f"No 'overall_micro' row found in {results_path}")


def read_entity_type_metrics(
    results_path: Path,
    mode: str | None = None,
) -> list[tuple[str, float, float, float]]:
    """Read per-entity-type precision / recall / F1 from a results CSV.

    Skips ``overall_micro`` and ``overall_macro`` rows.  Preserves the
    order of rows as they appear in the CSV file.

    Parameters
    ----------
    results_path :
        Path to the results CSV file.
    mode :
        If set, only consider rows whose ``mode`` column matches this value.

    Returns
        List of ``(entity_type, precision, recall, f1)`` tuples.
    """
    _OVERALL_KEYS = {"overall_micro", "overall_macro"}
    rows: list[tuple[str, float, float, float]] = []
    with results_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if mode is not None and row.get("mode") != mode:
                continue
            etype = row.get("entity_type", "")
            if etype in _OVERALL_KEYS:
                continue
            rows.append(
                (etype, float(row["precision"]), float(row["recall"]), float(row["f1"]))
            )
    return rows


def normalize_entity_name(raw: str) -> str:
    """Normalize an entity type name for display.

    Replaces underscores and hyphens with spaces, then title-cases.

    ``climate_greenhouse_gases`` → ``Climate Greenhouse Gases``
    """
    return raw.replace("_", " ").replace("-", " ").title()


def _format_pct(value: float) -> str:
    """Format a 0-1 float as a percentage string with two decimal places."""
    return f"{value * 100:.2f}"


def _maybe_format(value: float, best: float, second_best: float) -> str:
    """Wrap value in ``\\textbf`` or ``\\underline`` when it matches best or second."""
    s = _format_pct(value)
    if value == best:
        return f"\\textbf{{{s}}}"
    if value == second_best:
        return f"\\underline{{{s}}}"
    return s


def _best_and_second(values: list[float]) -> tuple[float, float]:
    """Return ``(largest, second_largest)`` from *values*."""
    unique = sorted(set(values), reverse=True)
    if len(unique) >= 2:
        return unique[0], unique[1]
    if len(unique) == 1:
        return unique[0], unique[0]
    return 0.0, 0.0


def generate_latex_table(
    entries: list[tuple[str, str, dict[str, tuple[float, float, float]]]],
    dataset_name: str,
    eval_modes: list[str],
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """Generate a LaTeX ``table*`` environment from grouped experiment entries.

    Parameters
    ----------
    entries:
        ``(experiment_type, model_name, {mode: (precision, recall, f1)})`` tuples.
        Order follows *entries* — group together entries sharing the same
        ``experiment_type`` before calling.
    eval_modes:
        List of evaluation modes to display as column groups.

    Returns
        Complete LaTeX source for the table.
    """
    model_rank = {name: i for i, name in enumerate(MODEL_INFO)}

    groups: dict[str, list[tuple[str, dict[str, tuple[float, float, float]]]]] = {}
    group_order: list[str] = []
    for exp_type, model_name, per_mode_metrics in entries:
        if exp_type not in groups:
            groups[exp_type] = []
            group_order.append(exp_type)
        groups[exp_type].append((model_name, per_mode_metrics))

    # Sort models within each group by MODEL_INFO order
    for exp_type in groups:
        groups[exp_type].sort(key=lambda x: model_rank.get(x[0], len(MODEL_INFO)))

    ordered_groups: list[
        tuple[str, list[tuple[str, dict[str, tuple[float, float, float]]]]]
    ] = []
    seen = set()
    for key in EXPERIMENT_ORDER:
        if key in groups:
            ordered_groups.append((key, groups[key]))
            seen.add(key)
    for key in group_order:
        if key not in seen:
            ordered_groups.append((key, groups[key]))

    # --- best / second-best per mode ---
    all_p: dict[str, list[float]] = {m: [] for m in eval_modes}
    all_r: dict[str, list[float]] = {m: [] for m in eval_modes}
    all_f: dict[str, list[float]] = {m: [] for m in eval_modes}
    for _, models in ordered_groups:
        for _, per_mode in models:
            for mode in eval_modes:
                if mode in per_mode:
                    p, r, f = per_mode[mode]
                    all_p[mode].append(p)
                    all_r[mode].append(r)
                    all_f[mode].append(f)

    best_second: dict[str, tuple[float, float, float, float, float, float]] = {}
    for mode in eval_modes:
        if all_p[mode]:
            bp, sp = _best_and_second(all_p[mode])
            br, sr = _best_and_second(all_r[mode])
            bf, sf = _best_and_second(all_f[mode])
            best_second[mode] = (bp, sp, br, sr, bf, sf)

    escaped_dataset = _escape_latex(dataset_name)
    if caption is None:
        mode_names = ", ".join(m.capitalize() for m in eval_modes)
        caption = (
            f"F1 scores on {escaped_dataset} "
            f"({mode_names}). "
            "Best per metric in bold, second-best underlined."
        )
    if label is None:
        label = f"tab:f1-{dataset_name}"

    num_modes = len(eval_modes)
    col_spec = "l" + "|ccc" * num_modes
    total_cols = 1 + 3 * num_modes

    # Header row 1 — mode group names
    header1_parts = [
        "\\multirow{2}{*}{\\textbf{Model}}",
    ]
    for i, mode in enumerate(eval_modes):
        if i == len(eval_modes) - 1:
            fmt = "{|c}"  # last mode: no right border (avoids orphan |)
        else:
            fmt = "{|c|}"
        header1_parts.append(
            f"\\multicolumn{{3}}{{{fmt}}}{{\\textbf{{{mode.capitalize()}}}}}"
        )
    header1 = " & ".join(header1_parts) + " \\\\"

    # Header row 2 — sub-column names
    header2_parts = [""]
    for _ in eval_modes:
        header2_parts.extend(["P", "R", "F1"])
    header2 = " & ".join(header2_parts) + " \\\\"

    lines: list[str] = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        header1,
        header2,
        "\\hline",
    ]

    num_groups = len(ordered_groups)
    for gi, (exp_type, models) in enumerate(ordered_groups):
        exp_title = exp_type.capitalize()
        lines.append(f"\\textit{{{exp_title}}} " + "& " * (total_cols - 1) + "\\\\")

        for mi, (model_key, per_mode_metrics) in enumerate(models):
            name = MODEL_INFO.get(model_key, model_key)

            cells = [f"\\hspace{{0.5em}}{name}"]
            for mode in eval_modes:
                if mode in per_mode_metrics:
                    p, r, f = per_mode_metrics[mode]
                    bp, sp, br, sr, bf, sf = best_second.get(
                        mode, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                    )
                    cells.append(_maybe_format(p, bp, sp))
                    cells.append(_maybe_format(r, br, sr))
                    cells.append(_maybe_format(f, bf, sf))
                else:
                    cells.extend(["-", "-", "-"])

            lines.append(" & ".join(cells) + " \\\\")

        if gi < num_groups - 1:
            lines.append("\\hline")

    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines)


def generate_entity_type_latex_table(
    entity_metrics: dict[str, list[tuple[str, float, float, float]]],
    dataset_name: str,
    model_name: str,
    eval_modes: list[str],
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """Generate a LaTeX ``table*`` of per-entity-type performance.

    Parameters
    ----------
    entity_metrics:
        Mapping of eval mode name to a list of
        ``(entity_type, precision, recall, f1)`` tuples.  Entity type
        order must be consistent across all modes.
    dataset_name:
        Dataset name for the caption / label.
    model_name:
        Model display name shown in the caption.
    eval_modes:
        List of evaluation modes to display as column groups.
    """
    escaped_dataset = _escape_latex(dataset_name)
    escaped_model = _escape_latex(model_name)
    mode_names = ", ".join(m.capitalize() for m in eval_modes)

    if caption is None:
        caption = (
            f"Entity-type performance on {escaped_dataset} "
            f"({mode_names}) for {escaped_model}."
        )
    if label is None:
        label = f"tab:entity-f1-{dataset_name}"

    num_modes = len(eval_modes)
    col_spec = "l|" + "|ccc" * num_modes

    # Header row 1 — mode group names
    header1_parts = ["\\textbf{Entity Type}"]
    for i, mode in enumerate(eval_modes):
        if i == num_modes - 1:
            fmt = "{|c}"
        else:
            fmt = "{|c|}"
        header1_parts.append(
            f"\\multicolumn{{3}}{{{fmt}}}{{\\textbf{{{mode.capitalize()}}}}}"
        )
    header1 = " & ".join(header1_parts) + " \\\\"

    # Header row 2 — sub-column names
    header2_parts = [""]
    for _ in eval_modes:
        header2_parts.extend(["P", "R", "F1"])
    header2 = " & ".join(header2_parts) + " \\\\"

    lines: list[str] = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        header1,
        header2,
        "\\hline",
    ]

    # Use the first mode's entity list as the canonical ordering
    first_mode = eval_modes[0]
    for etype_raw, *_ in entity_metrics[first_mode]:
        display_name = normalize_entity_name(etype_raw)
        escaped_name = _escape_latex(display_name)
        cells = [escaped_name]

        mode_data: dict[str, tuple[str, str, str]] = {}
        for mode in eval_modes:
            for et, p, r, f in entity_metrics[mode]:
                if et == etype_raw:
                    mode_data[mode] = (
                        _format_pct(p),
                        _format_pct(r),
                        _format_pct(f),
                    )
                    break
            else:
                mode_data[mode] = ("-", "-", "-")

        for mode in eval_modes:
            p_str, r_str, f_str = mode_data[mode]
            cells.extend([p_str, r_str, f_str])

        lines.append(" & ".join(cells) + " \\\\")

    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run the LaTeX table generator from the CLI."""
    args = parse_args()

    if args.entity_types:
        _main_entity_types(args)
    else:
        _main_overall(args)


def _main_overall(args: argparse.Namespace) -> None:
    """Original overall-metrics table path."""
    results = find_results(args.exp_dir, args.dataset_name)
    if not results:
        raise SystemExit(
            f"Error: No results CSV files found under "
            f"'{args.exp_dir}/**/{args.dataset_name}/'"
        )

    entries: list[tuple[str, str, dict[str, tuple[float, float, float]]]] = []
    for exp_type, model_name, csv_path in results:
        per_mode_metrics: dict[str, tuple[float, float, float]] = {}
        for mode in EVAL_MODES:
            try:
                per_mode_metrics[mode] = read_metrics(csv_path, mode=mode)
            except (ValueError, KeyError) as exc:
                print(
                    f"Warning: Skipping {csv_path} ({mode}): {exc}",
                    file=__import__("sys").stderr,
                )
        if per_mode_metrics:
            entries.append((exp_type, model_name, per_mode_metrics))

    if not entries:
        raise SystemExit("Error: No valid metrics could be read.")

    latex = generate_latex_table(
        entries=entries,
        dataset_name=args.dataset_name,
        eval_modes=EVAL_MODES,
        caption=args.caption,
        label=args.label,
    )

    output_path = args.output or (args.exp_dir / "table.tex")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")
    print(f"LaTeX table written to {output_path.resolve()}")


def _main_entity_types(args: argparse.Namespace) -> None:
    """Per-entity-type table path."""
    dataset_name = args.exp_dir.name
    model_name = args.exp_dir.parent.name

    # Discover available modes from the single results.csv
    results_csv = args.exp_dir / "results.csv"
    if not results_csv.is_file():
        raise SystemExit(f"Error: No results.csv found in '{args.exp_dir}'")

    available_modes: list[str] = []
    for mode in EVAL_MODES:
        metrics = read_entity_type_metrics(results_csv, mode=mode)
        if metrics:
            available_modes.append(mode)
    if not available_modes:
        raise SystemExit(f"Error: No entity-type rows found in '{results_csv}'")

    entity_metrics: dict[str, list[tuple[str, float, float, float]]] = {}
    for mode in available_modes:
        metrics = read_entity_type_metrics(results_csv, mode=mode)
        entity_metrics[mode] = metrics

    latex = generate_entity_type_latex_table(
        entity_metrics=entity_metrics,
        dataset_name=dataset_name,
        model_name=model_name,
        eval_modes=available_modes,
        caption=args.caption,
        label=args.label,
    )

    output_path = args.output or (args.exp_dir / "entity_type_table.tex")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex, encoding="utf-8")
    print(f"LaTeX table written to {output_path.resolve()}")


if __name__ == "__main__":
    main()
