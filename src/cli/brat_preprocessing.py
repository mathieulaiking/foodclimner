"""BRAT preprocessing pipeline: combine extracts, clean references,
and split into sentences.

Usage:
    python -m src.cli.brat_preprocessing --source data/brat \\
        --target data/brat-preprocessed \\
        --skip-clean
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from src.brat_utils import (
    PreprocessingStep,
    Removal,
    _entity_ranges,
    _overlaps,
    combine_extracts_into_docs,
    load_brat_directory,
    load_sentencizer,
    normalize_text_and_annotations,
    split_document,
    validate_annotation_texts,
    write_brat_directory,
)
from src.schemas import BratAnnotation, BratDoc, BratEntity


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BRAT preprocessing pipeline: combine extracts, clean references, split sentences.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source directory containing BRAT .txt/.ann pairs.",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=None,
        help="Target output directory. Defaults to <source>-preprocessed.",
    )
    parser.add_argument(
        "--skip-combine",
        action="store_true",
        default=False,
        help="Skip extract-combining step.",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        default=False,
        help="Skip reference-cleaning step.",
    )
    parser.add_argument(
        "--skip-split",
        action="store_true",
        default=False,
        help="Skip sentence-splitting step.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="ISO 639-1 language code used by spaCy sentencizer.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        default=False,
        help="Skip source offset verification in combine and split steps.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run on a small subset (10 docs) for quick checks.",
    )
    return parser.parse_args(argv)


# ── Step 1: Combine extracts ────────────────────────────────────────


class CombineExtractsStep(PreprocessingStep):
    """Combine AUTHOR-YEAR-N.{ann,txt} extract files into full documents."""

    def __init__(self, verify: bool = True) -> None:
        self._verify = verify

    @property
    def name(self) -> str:
        return "combine"

    def run(self, docs: list[BratDoc], source_dir: Path) -> list[BratDoc]:
        print("Step 1/3: Combining extracts ...")
        result = combine_extracts_into_docs(source_dir, verify=self._verify)
        print(f"  {len(result)} document(s) combined")
        return result


# ── Step 2: Clean references ────────────────────────────────────────


class CleanRefStep(PreprocessingStep):
    """Remove reference patterns (citations, brackets, Abstract lines)."""

    @property
    def name(self) -> str:
        return "clean"

    def run(self, docs: list[BratDoc], source_dir: Path) -> list[BratDoc]:
        print("Step 2/3: Cleaning references ...")
        patterns = self._build_reference_patterns()
        result: list[BratDoc] = []
        for doc in docs:
            cleaned_text, cleaned_entities = self._clean_single(doc, patterns)
            result.append(
                BratDoc(doc_id=doc.doc_id, text=cleaned_text, entities=cleaned_entities)
            )
        print(f"  {len(result)} document(s) cleaned")
        return result

    @staticmethod
    def _build_reference_patterns() -> dict[str, re.Pattern]:
        return {
            "bare_bracket": re.compile(r"\[(?:\[?\d+\]?(?:[,\s–]+\[?\d+\]?)*[,\s]*?)\]"),
            "paren_citation": re.compile(
                r"\([A-Za-z\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF]"
                r"[A-Za-z\u00C0-\u024F\u0370-\u03FF\u0400-\u04FF\s,.\x27&]*"
                r"(?:et\s+al\.)?,?\s*"
                r"(?:Citation\s*)?"
                r"(?:19|20)\d{2}"
                r"[^()]*?\)"
            ),
            "author_etal_bracket": re.compile(
                r"[A-Z][a-z]+\s+et\s+al\.\s*\[\d+(?:[,\s]\d+)*\]"
            ),
            "author_and_author_bracket": re.compile(
                r"[A-Z][a-z]+\s+and\s+[A-Z][a-z]+\s*\[\d+(?:[,\s]\d+)*\]"
            ),
            "author_bracket": re.compile(
                r"(?<=[\s(])[A-Z][a-z]+\s+\[\d+(?:[,\s]\d+)*\]"
            ),
            "inline_see": re.compile(
                r"see\s+[A-Z][a-z]+(?:\s+et\s+al\.)?,\s*(?:19|20)\d{2}"
            ),
            "inline_according_to": re.compile(
                r"according\s+to\s+[A-Z][a-z]+(?:\s+et\s+al\.)?"
                r"\s*\(?(?:19|20)\d{2}\)?"
                r"(?:\s*\[\d+(?:[,\s]\d+)*\])?"
            ),
        }

    @staticmethod
    def _find_abstract_removal(text: str) -> Removal | None:
        m = re.search(r"(?:^|\n)Abstract(?=\n|$)", text)
        if m:
            return Removal(start=m.start(), end=m.end(), reason="abstract_line")
        return None

    @staticmethod
    def _find_reference_matches(text: str, patterns: dict[str, re.Pattern]) -> list[Removal]:
        removals: list[Removal] = []
        seen: set[tuple[int, int]] = set()
        for name, pattern in patterns.items():
            for m in pattern.finditer(text):
                rng = (m.start(), m.end())
                if rng not in seen:
                    seen.add(rng)
                    matched = m.group()
                    is_multi = name == "bare_bracket" and ("[" in matched[1:] or "]" in matched[:-1])
                    placeholder = " [MREF]" if is_multi else " [REF]"
                    removals.append(Removal(m.start(), m.end(), name, placeholder=placeholder))
        return removals

    @staticmethod
    def _merge_and_expand(removals: list[Removal], text: str) -> list[Removal]:
        if not removals:
            return []
        sorted_removals = sorted(removals, key=lambda r: r.start)
        merged: list[Removal] = [sorted_removals[0]]
        for r in sorted_removals[1:]:
            if r.start <= merged[-1].end:
                if r.end > merged[-1].end:
                    merged[-1].end = r.end
                if r.placeholder and not merged[-1].placeholder:
                    merged[-1].placeholder = r.placeholder
            else:
                merged.append(r)
        for r in merged:
            while r.start > 0 and text[r.start - 1] in (" ", "\n", "\t"):
                r.start -= 1
        return merged

    @staticmethod
    def _filter_overlapping(removals: list[Removal], entities: list[BratEntity]) -> list[Removal]:
        ent_ranges = _entity_ranges(entities)
        return [r for r in removals if not any(_overlaps((r.start, r.end), er) for er in ent_ranges)]

    @staticmethod
    def _apply_removals(
        text: str,
        removals: list[Removal],
        entities: list[BratEntity],
    ) -> tuple[str, list[BratEntity]]:
        sorted_removals = sorted(removals, key=lambda r: r.start, reverse=True)
        for rem in sorted_removals:
            removal_len = rem.end - rem.start
            placeholder_len = len(rem.placeholder)
            delta = placeholder_len - removal_len
            text = text[: rem.start] + rem.placeholder + text[rem.end :]
            new_entities: list[BratEntity] = []
            for ent in entities:
                new_spans: list[tuple[int, int]] = []
                for s, e in ent.spans:
                    if s >= rem.end:
                        new_spans.append((s + delta, e + delta))
                    elif e > rem.start and s < rem.start:
                        new_spans.append((s, min(e, rem.start + placeholder_len)))
                    elif e <= rem.start:
                        new_spans.append((s, e))
                new_entities.append(
                    BratEntity(
                        entity_id=ent.entity_id,
                        entity_type=ent.entity_type,
                        spans=new_spans,
                        text=ent.text,
                    )
                )
            entities = new_entities
        return text, entities

    @staticmethod
    def _postprocess(text: str, entities: list[BratEntity]) -> tuple[str, list[BratEntity]]:
        ent_ranges = _entity_ranges(entities)
        cleanup: list[Removal] = []
        for m in re.finditer(r"  +", text):
            r = Removal(start=m.start() + 1, end=m.end(), reason="collapse_spaces")
            if not any(_overlaps((r.start, r.end), er) for er in ent_ranges):
                cleanup.append(r)
        for m in re.finditer(r"\n[ \t]+(?=\n)", text):
            r = Removal(start=m.start() + 1, end=m.end(), reason="empty_line")
            if not any(_overlaps((r.start, r.end), er) for er in ent_ranges):
                cleanup.append(r)
        cleanup.sort(key=lambda r: r.start, reverse=True)
        return CleanRefStep._apply_removals(text, cleanup, entities)

    def _clean_single(
        self,
        doc: BratDoc,
        patterns: dict[str, re.Pattern],
    ) -> tuple[str, list[BratEntity]]:
        removals = self._find_reference_matches(doc.text, patterns)
        abstract_rem = self._find_abstract_removal(doc.text)
        if abstract_rem:
            removals.append(abstract_rem)
        removals = self._merge_and_expand(removals, doc.text)
        removals = self._filter_overlapping(removals, doc.entities)
        cleaned_text, adjusted_entities = self._apply_removals(doc.text, removals, doc.entities)
        cleaned_text, adjusted_entities = self._postprocess(cleaned_text, adjusted_entities)
        errors = validate_annotation_texts(cleaned_text, adjusted_entities)
        if errors:
            for err in errors:
                print(f"  VALIDATION ERROR in {doc.doc_id}: {err}")
        return cleaned_text, adjusted_entities


# ── Step 3: Sentence splitting ──────────────────────────────────────


class SentSplitStep(PreprocessingStep):
    """Split BRAT documents into one sentence per BratDoc."""

    def __init__(self, lang: str = "en", verify: bool = True) -> None:
        self._lang = lang
        self._verify = verify
        self._nlp = load_sentencizer(lang)

    @property
    def name(self) -> str:
        return "split"

    def run(self, docs: list[BratDoc], source_dir: Path) -> list[BratDoc]:
        print("Step 3/3: Splitting sentences ...")
        result: list[BratDoc] = []
        for doc in docs:
            annotations = [
                BratAnnotation(raw_line="", line_type="T", entity_id=e.entity_id,
                               entity_type=e.entity_type, spans=list(e.spans), text=e.text)
                for e in doc.entities
            ]
            text, annotations = normalize_text_and_annotations(doc.text, annotations)
            sentence_pairs = split_document(text, annotations, self._nlp)
            width = len(str(len(sentence_pairs)))
            for idx, (sent_text, ann_lines) in enumerate(sentence_pairs, start=1):
                sent_entities = self._parse_ann_lines(ann_lines, sent_text)
                result.append(
                    BratDoc(
                        doc_id=f"{doc.doc_id}_{idx:0{width}d}",
                        text=sent_text,
                        entities=sent_entities,
                    )
                )
        print(f"  {len(result)} sentence(s) produced")
        return result

    @staticmethod
    def _parse_ann_lines(ann_lines: list[str], sent_text: str) -> list[BratEntity]:
        entities: list[BratEntity] = []
        for ann_line in ann_lines:
            if not ann_line.startswith("T"):
                continue
            parts = ann_line.split("\t")
            if len(parts) < 3:
                continue
            entity_id = parts[0]
            type_and_spans = parts[1]
            first_space = type_and_spans.index(" ")
            entity_type = type_and_spans[:first_space]
            span_str = type_and_spans[first_space + 1 :]
            spans: list[tuple[int, int]] = []
            for fragment in span_str.split(";"):
                offsets = fragment.strip().split()
                spans.append((int(offsets[0]), int(offsets[1])))
            text_fragments = [sent_text[s:e] for s, e in spans]
            text = " ".join(text_fragments)
            entities.append(
                BratEntity(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    spans=spans,
                    text=text,
                )
            )
        return entities


# ── Pipeline runner ─────────────────────────────────────────────────


def run_pipeline(args: argparse.Namespace) -> None:
    source = args.source.resolve()
    target = (
        args.target.resolve()
        if args.target is not None
        else source.parent / f"{source.name}-preprocessed"
    )

    if not source.exists():
        print(f"ERROR: source directory not found: {source}")
        sys.exit(1)

    print(f"Source: {source}")
    print(f"Target: {target}")

    skip_combine = args.skip_combine
    skip_clean = args.skip_clean
    skip_split = args.skip_split

    if skip_combine and skip_clean and skip_split:
        print("ERROR: all steps are skipped, nothing to do.")
        sys.exit(1)

    # Step 1: initial load (from combine step or raw directory)
    if skip_combine:
        docs = load_brat_directory(source)
        print(f"Loaded {len(docs)} document(s) from source directory")
    else:
        combine_step = CombineExtractsStep(verify=not args.no_verify)
        docs = combine_step.run([], source)

    if args.debug:
        docs = docs[:10]
        print(f"Debug mode: limiting to {len(docs)} document(s)")

    # Step 2: clean references
    if not skip_clean:
        clean_step = CleanRefStep()
        docs = clean_step.run(docs, source)

    # Step 3: sentence split
    if not skip_split:
        split_step = SentSplitStep(lang=args.lang, verify=not args.no_verify)
        docs = split_step.run(docs, source)

    # Write output
    write_brat_directory(docs, target)
    print(f"\nDone. {len(docs)} document(s) written to {target}")


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
