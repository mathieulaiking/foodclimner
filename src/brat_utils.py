"""BRAT parsing and sentence-splitting utilities.

Migrated from the old ``src.process.converters.brat`` module.
"""

from __future__ import annotations

import abc
import bisect
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import spacy
from spacy.language import Language

from src.schemas import BratAnnotation, BratDoc, BratEntity


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
