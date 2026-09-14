"""BRAT preprocessing: parsing, cleaning, sentence-splitting, and CLI.

Run as::

    python -m src.data_processing.brat --source data/brat \\
        --target data/brat-preprocessed \\
        --skip-clean
"""

from __future__ import annotations

import abc
import argparse
import bisect
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import spacy
from spacy.language import Language


# ── BRAT schemas (owned by this module) ─────────────────────────────


@dataclass
class BratDoc:
    """In-memory BRAT document passed between preprocessing steps.

    Attributes
    ----------
    doc_id:
        Document identifier (stem of the .txt/.ann pair).
    text:
        Full document text.
    entities:
        List of BratEntity objects.
    """

    doc_id: str
    text: str
    entities: list[BratEntity]


@dataclass(frozen=True)
class BratEntity:
    """One BRAT entity annotation, possibly discontinuous."""

    entity_id: str
    entity_type: str
    spans: list[tuple[int, int]]
    text: str


@dataclass
class BratAnnotation:
    """Parsed BRAT annotation line used by sentence-splitting logic."""

    raw_line: str
    line_type: str = ""
    entity_id: str = ""
    entity_type: str = ""
    spans: list[tuple[int, int]] = field(default_factory=list)
    text: str = ""
    ref_entity_id: str = ""


__all__ = [
    "BratAnnotation",
    "BratDoc",
    "BratEntity",
    "PreprocessingStep",
    "Removal",
    "combine_extracts_into_docs",
    "load_brat_directory",
    "load_sentencizer",
    "normalize_text_and_annotations",
    "parse_ann_entities",
    "parse_ann_file",
    "rebuild_ann_lines",
    "split_brat_directory",
    "split_document",
    "validate_annotation_texts",
    "write_brat_directory",
    "CombineExtractsStep",
    "CleanRefStep",
    "SentSplitStep",
    "run_pipeline",
    "parse_args",
    "main",
]


def parse_ann_entities(ann_filepath: str | Path) -> list[BratEntity]:
    """Parse BRAT entity lines (T*) from a .ann file."""
    entities: list[BratEntity] = []
    with open(ann_filepath, "r", encoding="utf-8") as ann_file:
        for line in ann_file:
            line = line.strip()
            if not line or not line.startswith("T"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue

            entity_id = parts[0]
            type_and_spans = parts[1]
            text = parts[2]

            first_space = type_and_spans.index(" ")
            entity_type = type_and_spans[:first_space]
            span_str = type_and_spans[first_space + 1 :]

            spans: list[tuple[int, int]] = []
            for fragment in span_str.split(";"):
                offsets = fragment.strip().split()
                spans.append((int(offsets[0]), int(offsets[1])))

            entities.append(
                BratEntity(
                    entity_id=entity_id,
                    entity_type=entity_type,
                    spans=spans,
                    text=text,
                )
            )

    return entities


def parse_ann_file(ann_path: Path) -> list[BratAnnotation]:
    """Parse every line of a .ann file into BratAnnotation objects."""
    annotations: list[BratAnnotation] = []
    with open(ann_path, "r", encoding="utf-8") as ann_file:
        for line in ann_file:
            line = line.rstrip("\n")
            if not line:
                continue
            annotation = BratAnnotation(raw_line=line, line_type=line[0])

            if line.startswith("T"):
                parts = line.split("\t")
                annotation.entity_id = parts[0]
                type_and_spans = parts[1]
                annotation.text = parts[2] if len(parts) > 2 else ""
                first_space = type_and_spans.index(" ")
                annotation.entity_type = type_and_spans[:first_space]
                for fragment in type_and_spans[first_space + 1 :].split(";"):
                    offsets = fragment.strip().split()
                    annotation.spans.append((int(offsets[0]), int(offsets[1])))

            elif line.startswith("#"):
                parts = line.split("\t")
                if len(parts) >= 2:
                    note_meta = parts[1].split()
                    if len(note_meta) >= 2:
                        annotation.ref_entity_id = note_meta[1]

            annotations.append(annotation)

    return annotations


def normalize_text_and_annotations(
    raw_text: str,
    annotations: list[BratAnnotation],
) -> tuple[str, list[BratAnnotation]]:
    """Normalize CRLF endings and shift BRAT offsets accordingly."""
    cr_positions = sorted(index for index, char in enumerate(raw_text) if char == "\r")
    if not cr_positions:
        return raw_text, annotations

    clean_text = raw_text.replace("\r", "")

    def adjust(position: int) -> int:
        return position - bisect.bisect_left(cr_positions, position)

    for annotation in annotations:
        if annotation.line_type == "T" and annotation.spans:
            annotation.spans = [
                (adjust(start), adjust(end))
                for start, end in annotation.spans
            ]

    return clean_text, annotations


def load_sentencizer(lang: str) -> Language:
    """Load a blank spaCy model with a sentencizer."""
    nlp = spacy.blank(lang)
    nlp.add_pipe("sentencizer")
    return nlp


def _entity_inside_sentence(
    spans: list[tuple[int, int]],
    sent_start: int,
    sent_end: int,
) -> bool:
    return all(start >= sent_start and end <= sent_end for start, end in spans)


def _entity_overlaps_sentence(
    spans: list[tuple[int, int]],
    sent_start: int,
    sent_end: int,
) -> bool:
    return any(start < sent_end and end > sent_start for start, end in spans)


def _clip_spans_to_sentence(
    spans: list[tuple[int, int]],
    sent_start: int,
    sent_end: int,
) -> list[tuple[int, int]]:
    clipped: list[tuple[int, int]] = []
    for start, end in spans:
        clipped_start = max(start, sent_start) - sent_start
        clipped_end = min(end, sent_end) - sent_start
        if clipped_end > clipped_start:
            clipped.append((clipped_start, clipped_end))
    return clipped


def _rebuild_entity_line(
    entity_id: str,
    entity_type: str,
    spans: list[tuple[int, int]],
    sent_text: str,
) -> str:
    span_str = ";".join(f"{start} {end}" for start, end in spans)
    text_fragments = [sent_text[start:end] for start, end in spans]
    text = " ".join(text_fragments)
    return f"{entity_id}\t{entity_type} {span_str}\t{text}"


def split_document(
    text: str,
    annotations: list[BratAnnotation],
    nlp: Language,
) -> list[tuple[str, list[str]]]:
    """Split one BRAT document into sentence-local text and annotation lines."""
    doc = nlp(text)
    sentences = list(doc.sents)

    entity_map = {
        annotation.entity_id: annotation
        for annotation in annotations
        if annotation.line_type == "T"
    }
    note_map: dict[str, list[BratAnnotation]] = {}
    for annotation in annotations:
        if annotation.line_type == "#" and annotation.ref_entity_id:
            note_map.setdefault(annotation.ref_entity_id, []).append(annotation)

    results: list[tuple[str, list[str]]] = []

    for sentence in sentences:
        sent_start = sentence.start_char
        sent_end = sentence.end_char
        sent_text = text[sent_start:sent_end]

        ann_lines: list[str] = []
        entity_counter = 1

        for entity_ann in entity_map.values():
            if _entity_inside_sentence(entity_ann.spans, sent_start, sent_end):
                new_spans = [
                    (start - sent_start, end - sent_start)
                    for start, end in entity_ann.spans
                ]
                new_id = f"T{entity_counter}"
                ann_lines.append(
                    _rebuild_entity_line(
                        new_id,
                        entity_ann.entity_type,
                        new_spans,
                        sent_text,
                    )
                )

                for note in note_map.get(entity_ann.entity_id, []):
                    note_parts = note.raw_line.split("\t")
                    note_meta = note_parts[1].split()
                    note_meta[1] = new_id
                    note_parts[1] = " ".join(note_meta)
                    ann_lines.append("\t".join(note_parts))

                entity_counter += 1

            elif _entity_overlaps_sentence(entity_ann.spans, sent_start, sent_end):
                clipped_spans = _clip_spans_to_sentence(
                    entity_ann.spans,
                    sent_start,
                    sent_end,
                )
                if clipped_spans:
                    new_id = f"T{entity_counter}"
                    ann_lines.append(
                        _rebuild_entity_line(
                            new_id,
                            entity_ann.entity_type,
                            clipped_spans,
                            sent_text,
                        )
                    )
                    entity_counter += 1

        results.append((sent_text, ann_lines))

    return results


def _verify_round_trip(source_dir: Path, target_dir: Path, sample_txt: Path) -> None:
    """Verify sentence files contain annotation offsets consistent with text."""
    stem = sample_txt.stem

    with open(sample_txt, "r", encoding="utf-8", newline="") as source_file:
        raw_text = source_file.read()
    original_text = raw_text.replace("\r", "")

    split_files = sorted(
        target_dir.glob(f"{stem}_*.txt"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    reconstructed_parts = [path.read_text(encoding="utf-8") for path in split_files]

    cursor = 0
    for index, part in enumerate(reconstructed_parts, start=1):
        if not part:
            continue

        next_pos = original_text.find(part, cursor)
        if next_pos < 0:
            raise ValueError(
                "Round-trip verification failed: sentence text cannot be "
                f"realigned for {stem} at split index {index}."
            )

        skipped = original_text[cursor:next_pos]
        if skipped and not skipped.isspace():
            raise ValueError(
                "Round-trip verification failed: reconstructed text skipped "
                f"non-whitespace content for {stem}."
            )

        cursor = next_pos + len(part)

    trailing = original_text[cursor:]
    if trailing and not trailing.isspace():
        raise ValueError(
            "Round-trip verification failed: reconstructed text is shorter "
            f"for {stem}."
        )

    for split_txt in split_files:
        split_ann = split_txt.with_suffix(".ann")
        sent_text = split_txt.read_text(encoding="utf-8")
        if not split_ann.exists():
            continue
        for line in split_ann.read_text(encoding="utf-8").splitlines():
            if not line.startswith("T"):
                continue
            parts = line.split("\t")
            expected_text = parts[2]
            type_and_spans = parts[1]
            first_space = type_and_spans.index(" ")
            for fragment in type_and_spans[first_space + 1 :].split(";"):
                offsets = fragment.strip().split()
                start, end = int(offsets[0]), int(offsets[1])
                actual = sent_text[start:end]
                if actual not in expected_text:
                    raise ValueError(
                        "Offset mismatch during verification for "
                        f"{split_txt.name}: expected fragment from "
                        f"{expected_text!r}, got {actual!r}"
                    )


def split_brat_directory(
    source_dir: str | Path,
    target_dir: str | Path | None = None,
    lang: str = "en",
    verify: bool = True,
) -> tuple[int, int]:
    """Split BRAT files into one-sentence-per-file BRAT documents."""
    source_path = Path(source_dir).resolve()
    resolved_target = (
        Path(target_dir).resolve()
        if target_dir is not None
        else source_path.parent / f"{source_path.name}-split"
    )

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_path}")

    nlp = load_sentencizer(lang)

    if resolved_target.exists():
        shutil.rmtree(resolved_target)
    resolved_target.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(source_path.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {source_path}")

    total_sentences = 0
    total_entities = 0

    for txt_path in txt_files:
        stem = txt_path.stem
        ann_path = txt_path.with_suffix(".ann")

        with open(txt_path, "r", encoding="utf-8", newline="") as txt_file:
            raw_text = txt_file.read()

        annotations = parse_ann_file(ann_path) if ann_path.exists() else []
        text, annotations = normalize_text_and_annotations(raw_text, annotations)

        sentence_pairs = split_document(text, annotations, nlp)

        for index, (sent_text, ann_lines) in enumerate(sentence_pairs, start=1):
            out_stem = f"{stem}_{index}"
            out_txt = resolved_target / f"{out_stem}.txt"
            out_ann = resolved_target / f"{out_stem}.ann"

            out_txt.write_text(sent_text, encoding="utf-8")
            content = "\n".join(ann_lines)
            out_ann.write_text(
                content + ("\n" if content else ""),
                encoding="utf-8",
            )

            total_entities += sum(1 for line in ann_lines if line.startswith("T"))

        total_sentences += len(sentence_pairs)

    if verify:
        _verify_round_trip(source_path, resolved_target, txt_files[0])

    return total_sentences, total_entities


# ── Preprocessing step base class ────────────────────────────────────


class PreprocessingStep(abc.ABC):
    """Abstract base for a single BRAT preprocessing step.

    Each step consumes a list of ``BratDoc`` objects (the output of the
    previous step) and the original source directory, then returns a new
    list of ``BratDoc`` objects.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable step name used in logging and flag names."""

    @abc.abstractmethod
    def run(self, docs: list[BratDoc], source_dir: Path) -> list[BratDoc]:
        """Execute this preprocessing step.

        Parameters
        ----------
        docs:
            Documents produced by the previous step (empty for the
            first step in the pipeline).
        source_dir:
            Original source directory passed on the command line.
            Steps that need to read from disk (e.g. combine extracts)
            use this as a fallback.

        Returns
        -------
        list[BratDoc]
            Processed documents ready for the next step.
        """


# ── BRAT file I/O ───────────────────────────────────────────────────


def load_brat_directory(source_dir: Path) -> list[BratDoc]:
    """Load every .txt/.ann pair in *source_dir* into ``BratDoc`` objects.

    Returns documents in sorted order by filename.
    """
    docs: list[BratDoc] = []
    for txt_path in sorted(source_dir.glob("*.txt")):
        ann_path = txt_path.with_suffix(".ann")
        doc_id = txt_path.stem
        raw_bytes = txt_path.read_bytes()
        text = raw_bytes.decode("utf-8")
        if ann_path.exists():
            annotations = parse_ann_file(ann_path)
            text, annotations = normalize_text_and_annotations(text, annotations)
            entities = [
                BratEntity(
                    entity_id=a.entity_id,
                    entity_type=a.entity_type,
                    spans=a.spans,
                    text=a.text,
                )
                for a in annotations
                if a.line_type == "T"
            ]
        else:
            entities = []
        docs.append(BratDoc(doc_id=doc_id, text=text, entities=entities))
    return docs


def write_brat_directory(docs: list[BratDoc], target_dir: Path) -> None:
    """Write a list of ``BratDoc`` objects to  .txt/.ann pairs.

    Existing files in *target_dir* are overwritten; no cleanup is
    performed beforehand.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        out_txt = target_dir / f"{doc.doc_id}.txt"
        out_ann = target_dir / f"{doc.doc_id}.ann"
        out_txt.write_text(doc.text, encoding="utf-8")
        ann_lines = rebuild_ann_lines(doc.entities)
        out_ann.write_text("\n".join(ann_lines) + "\n", encoding="utf-8")


# ── Reference-removal helpers (shared by CleanRefStep) ──────────────


@dataclass
class Removal:
    """One text span slated for replacement."""
    start: int
    end: int
    reason: str = ""
    placeholder: str = ""


def _entity_ranges(entities: list[BratEntity]) -> list[tuple[int, int]]:
    """Flatten all entity span ranges."""
    ranges: list[tuple[int, int]] = []
    for e in entities:
        for s, e_ in e.spans:
            ranges.append((s, e_))
    return ranges


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Return True if interval *a* and interval *b* overlap."""
    return a[0] < b[1] and a[1] > b[0]


def rebuild_ann_lines(entities: list[BratEntity]) -> list[str]:
    """Serialize ``BratEntity`` objects to BRAT T-lines."""
    lines: list[str] = []
    for ent in entities:
        span_str = ";".join(f"{s} {e}" for s, e in ent.spans)
        lines.append(f"{ent.entity_id}\t{ent.entity_type} {span_str}\t{ent.text}")
    return lines


def validate_annotation_texts(text: str, entities: list[BratEntity]) -> list[str]:
    """Verify that each entity's stored text matches its span content.

    Returns a list of error messages (empty means all valid).
    """
    errors: list[str] = []
    for ent in entities:
        fragments: list[str] = []
        for s, e in ent.spans:
            fragments.append(text[s:e])
        reconstructed = " ".join(fragments)
        if reconstructed != ent.text:
            errors.append(
                f"{ent.entity_id} ({ent.entity_type}): "
                f"expected {ent.text!r}, got {reconstructed!r} "
                f"at spans {ent.spans}"
            )
    return errors


# ── Extract-combining helpers (shared by CombineExtractsStep) ────────


FILE_PATTERN = re.compile(r"^(.+)-(\d{4})-(\d+)\.(ann|txt)$")


def _group_extracts_for_prefix(ann_path: Path) -> list[tuple[str, list[tuple[int, int]], str]]:
    """Parse T-lines from a single .ann file for extract combining.

    Returns a list of ``(entity_type, spans, text)`` tuples.
    """
    entities: list[tuple[str, list[tuple[int, int]], str]] = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("T"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            type_and_spans = parts[1]
            text = parts[2]
            first_space = type_and_spans.index(" ")
            entity_type = type_and_spans[:first_space]
            span_str = type_and_spans[first_space + 1 :]
            spans: list[tuple[int, int]] = []
            for fragment in span_str.split(";"):
                offsets = fragment.strip().split()
                spans.append((int(offsets[0]), int(offsets[1])))
            entities.append((entity_type, spans, text))
    return entities


def _group_extracts(source: Path) -> dict[str, list[tuple[int, Path, Path]]]:
    """Group .ann/.txt files by author-year prefix, sorted by extract number.

    Returns ``{prefix: [(num, ann_path, txt_path), ...]}``.
    """
    raw: dict[str, dict[int, dict[str, Path]]] = {}
    for f in source.iterdir():
        if not f.is_file():
            continue
        m = FILE_PATTERN.match(f.name)
        if not m:
            continue
        author, year, num_str, ext = m.groups()
        prefix = f"{author}-{year}"
        num = int(num_str)
        raw.setdefault(prefix, {})
        raw[prefix].setdefault(num, {"ann": None, "txt": None})
        raw[prefix][num][ext] = f
    result: dict[str, list[tuple[int, Path, Path]]] = {}
    for prefix in sorted(raw):
        extracts: list[tuple[int, Path, Path]] = []
        for num in sorted(raw[prefix]):
            pair = raw[prefix][num]
            if pair["ann"] is None or pair["txt"] is None:
                print(f"Warning: {prefix}-{num} missing .ann or .txt, skipping")
                continue
            extracts.append((num, pair["ann"], pair["txt"]))
        if extracts:
            result[prefix] = extracts
    return result


def combine_extracts_into_docs(source_dir: Path, verify: bool = True) -> list[BratDoc]:
    """Combine AUTHOR-YEAR-N extracts into full BratDoc objects.

    Returns one ``BratDoc`` per unique AUTHOR-YEAR prefix.
    """
    groups = _group_extracts(source_dir)
    if not groups:
        raise ValueError(f"No extract files found in {source_dir}")

    docs: list[BratDoc] = []
    for prefix in sorted(groups):
        extracts = groups[prefix]
        combined_text = ""
        all_entities_data: list[tuple[str, list[tuple[int, int]], str]] = []

        for _num, ann_path, txt_path in extracts:
            chunk = txt_path.read_text(encoding="utf-8")
            entities = _group_extracts_for_prefix(ann_path)
            if verify:
                for etype, spans, expected in entities:
                    actual = " ".join(chunk[s:e] for s, e in spans)
                    if actual != expected:
                        raise ValueError(
                            f"Source verification failed for {ann_path.name}: "
                            f"expected {expected!r}, got {actual!r} "
                            f"at spans {spans} (entity type: {etype})"
                        )
            shift = len(combined_text)
            shifted = [
                (etype, [(s + shift, e + shift) for s, e in spans], text_val)
                for etype, spans, text_val in entities
            ]
            all_entities_data.extend(shifted)
            combined_text += chunk

        entities = [
            BratEntity(
                entity_id=f"T{i}",
                entity_type=etype,
                spans=spans,
                text=text_val,
            )
            for i, (etype, spans, text_val) in enumerate(all_entities_data, start=1)
        ]
        docs.append(BratDoc(doc_id=prefix, text=combined_text, entities=entities))

    return docs


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


# ── Pipeline runner + CLI ───────────────────────────────────────────


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
