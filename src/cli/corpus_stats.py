"""Compute corpus statistics for a NER dataset in JSONL format.

Usage::

    python -m src.cli.corpus_stats data/sf4cd-v0.5/test.jsonl
    python -m src.cli.corpus_stats data/sf4cd-v0.5/test.jsonl -o stats.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute corpus statistics for a NER dataset.",
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="Path to a test.jsonl file containing the dataset.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Path to write stats.json. "
            "Defaults to the same directory as data_path."
        ),
    )
    return parser.parse_args(argv)


def _resolve_output(args: argparse.Namespace) -> Path:
    if args.output_path is not None:
        return args.output_path
    return args.data_path.parent / "stats.json"


def _load_jsonl(path: Path) -> list[dict]:
    docs: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def _spans_overlap(s1: int, e1: int, s2: int, e2: int) -> bool:
    return s1 < e2 and s2 < e1


def _entity_spans_overlap(
    spans_a: list[list[int]], spans_b: list[list[int]],
) -> bool:
    for s1, e1 in spans_a:
        for s2, e2 in spans_b:
            if _spans_overlap(s1, e1, s2, e2):
                return True
    return False


def compute_stats(docs: list[dict]) -> dict:
    n_docs = len(docs)

    all_mentions = [e for d in docs for e in d["entities"]]
    n_mentions = len(all_mentions)

    unique_mentions_set: set[str] = set()
    for e in all_mentions:
        norm = " ".join(e["text"].split()).lower()
        unique_mentions_set.add(norm)
    n_unique_mentions = len(unique_mentions_set)

    doc_char_lens = np.array([len(d["text"]) for d in docs], dtype=np.float64)
    doc_word_lens = np.array(
        [len(d["text"].split()) for d in docs], dtype=np.float64
    )
    mentions_per_doc = np.array(
        [len(d["entities"]) for d in docs], dtype=np.float64
    )

    mention_char_lens = np.array(
        [len(e["text"]) for e in all_mentions], dtype=np.float64
    )
    mention_word_lens = np.array(
        [len(e["text"].split()) for e in all_mentions], dtype=np.float64
    )

    type_counts: Counter[str] = Counter()
    type_char_lens: dict[str, list[int]] = defaultdict(list)
    for e in all_mentions:
        t = e["type"]
        type_counts[t] += 1
        type_char_lens[t].append(len(e["text"]))

    sorted_types = sorted(type_counts.keys())

    discontinuous_mentions: list[dict] = []
    nested_mentions: list[dict] = []
    for d in docs:
        ents = d["entities"]
        for i, e in enumerate(ents):
            if len(e["offsets"]) > 1:
                discontinuous_mentions.append(e)
            for j in range(i + 1, len(ents)):
                if _entity_spans_overlap(e["offsets"], ents[j]["offsets"]):
                    nested_mentions.append(e)
                    nested_mentions.append(ents[j])

    n_discontinuous = len(discontinuous_mentions)
    n_nested = len(set(id(e) for e in nested_mentions))
    n_flat = n_mentions - n_discontinuous - n_nested

    per_type = {}
    for t in sorted_types:
        ct = type_counts[t]
        arr = np.array(type_char_lens[t], dtype=np.float64)
        per_type[t] = {
            "count": ct,
            "avg_length_chars": float(arr.mean()) if len(arr) > 0 else 0.0,
            "percentage": round(ct / n_mentions * 100, 2) if n_mentions else 0.0,
        }

    def _mean_std(arr: np.ndarray) -> dict:
        return {
            "mean": float(arr.mean()) if len(arr) > 0 else 0.0,
            "std": float(arr.std(ddof=0)) if len(arr) > 0 else 0.0,
        }

    return {
        "num_documents": n_docs,
        "document_length_chars": {**_mean_std(doc_char_lens), "unit": "char"},
        "document_length_words": {**_mean_std(doc_word_lens), "unit": "word"},
        "num_entity_types": len(sorted_types),
        "entity_types": sorted_types,
        "num_mentions": n_mentions,
        "num_unique_mentions": n_unique_mentions,
        "mention_length_chars": {
            **_mean_std(mention_char_lens), "unit": "char",
        },
        "mention_length_words": {
            **_mean_std(mention_word_lens), "unit": "word",
        },
        "mentions_per_document": _mean_std(mentions_per_doc),
        "per_type": per_type,
        "discontinuous": {
            "count": n_discontinuous,
            "percentage": round(
                n_discontinuous / n_mentions * 100, 2
            ) if n_mentions else 0.0,
        },
        "nested": {
            "count": n_nested,
            "percentage": round(
                n_nested / n_mentions * 100, 2
            ) if n_mentions else 0.0,
        },
        "flat": {
            "count": n_flat,
            "percentage": round(
                n_flat / n_mentions * 100, 2
            ) if n_mentions else 0.0,
        },
    }


def _print_summary(stats: dict) -> None:
    print(f"Sentences               : {stats['num_documents']}")
    print(f"Document length (chars) : {stats['document_length_chars']['mean']:.1f} ± {stats['document_length_chars']['std']:.1f}")
    print(f"Document length (words) : {stats['document_length_words']['mean']:.1f} ± {stats['document_length_words']['std']:.1f}")
    print(f"Entity types            : {stats['num_entity_types']}")
    print(f"Total mentions          : {stats['num_mentions']}")
    pct_unique = stats['num_unique_mentions'] / stats['num_mentions'] * 100 if stats['num_mentions'] else 0.0
    print(f"Unique mentions         : {stats['num_unique_mentions']} ({pct_unique:.1f}%)")
    print(f"Mention length (chars)  : {stats['mention_length_chars']['mean']:.1f} ± {stats['mention_length_chars']['std']:.1f}")
    print(f"Mention length (words)  : {stats['mention_length_words']['mean']:.1f} ± {stats['mention_length_words']['std']:.1f}")
    print(f"Mentions / doc          : {stats['mentions_per_document']['mean']:.1f} ± {stats['mentions_per_document']['std']:.1f}")
    print()
    print(f"{'Type':<30} {'Count':>6} {'AvgChars':>9} {'Percent':>8}")
    print("-" * 55)
    for t, v in stats["per_type"].items():
        print(f"{t:<30} {v['count']:>6} {v['avg_length_chars']:>8.1f} {v['percentage']:>7.1f}%")
    print()
    print(f"Discontinuous : {stats['discontinuous']['count']:>5} ({stats['discontinuous']['percentage']:.1f}%)")
    print(f"Nested        : {stats['nested']['count']:>5} ({stats['nested']['percentage']:.1f}%)")
    print(f"Flat          : {stats['flat']['count']:>5} ({stats['flat']['percentage']:.1f}%)")


def main() -> None:
    args = parse_args()

    if not args.data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    output_path = _resolve_output(args)
    docs = _load_jsonl(args.data_path)
    stats = compute_stats(docs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        f.write("\n")

    _print_summary(stats)


if __name__ == "__main__":
    main()
