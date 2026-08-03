"""CLI entrypoint to convert a corpus directory between formats.

Auto-detects the input format and writes the output in the requested format::

    python -m src.cli.convert data/brat-sf4cd-annots data/sf4cd-v0.5

    python -m src.cli.convert data/sf4cd-v0.4 data/sf4cd-v0.5 \\
        --format brat --no-ent-file
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.convert import FORMAT_CHOICES, auto_detect_format, convert


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a corpus directory between NER annotation formats. "
            "The input format is auto-detected."
        ),
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to the input directory (format auto-detected).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help=(
            "Path to the output directory. Must not exist or must be empty."
        ),
    )
    parser.add_argument(
        "-f",
        "--format",
        default="jsonl_txtmultispan",
        choices=FORMAT_CHOICES,
        help="Target format (default: jsonl_txtmultispan).",
    )
    parser.add_argument(
        "--no-ent-file",
        action="store_true",
        default=False,
        help="Skip creation of entities.json.",
    )
    parser.add_argument(
        "--no-rule-file",
        action="store_true",
        default=False,
        help="Skip creation of rules.json.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    if not args.input_dir.is_dir():
        raise NotADirectoryError(
            f"Input directory does not exist: {args.input_dir}"
        )

    source_format = auto_detect_format(args.input_dir)

    if args.format == source_format:
        print(
            f"Warning: input and target format are both {source_format!r}. "
            "No conversion is necessary."
        )

    convert(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_format=args.format,
        no_ent_file=args.no_ent_file,
        no_rule_file=args.no_rule_file,
    )


if __name__ == "__main__":
    main()
