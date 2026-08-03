"""Corpus format conversion: readers and writers for BRAT, CoNLL, JSONL-BIO,
and JSONL-TXT-MultiSpan formats.

All readers produce ``list[MultiSpanNERDocument]``.
All writers consume ``list[MultiSpanNERDocument]`` and produce files
in the target format inside the given output directory.

Usage::

    from src.convert import convert, auto_detect_format, Normalizer

    normalizer = Normalizer()
    fmt = auto_detect_format("/path/to/input")
    convert("/path/to/input", "/path/to/output", target_format="jsonl_txtmultispan",
            normalizer=normalizer)
"""

from __future__ import annotations

import abc
import json
import re
from pathlib import Path
from typing import Any
from src.brat_utils import parse_ann_entities
from src.schemas import (
    EntityInfo,
    MultiSpanEntity,
    MultiSpanNERDocument,
    RuleSection,
)
from src.utils import normalize_example_id


# ── Helpers ──────────────────────────────────────────────────────────


def _tokens_to_text_and_offsets(
    tokens: list[str],
) -> tuple[str, list[tuple[int, int]]]:
    """Join tokens with spaces and return (text, list_of_char_offsets)."""
    text = " ".join(tokens)
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = cursor
        end = start + len(token)
        offsets.append((start, end))
        cursor = end + 1
    return text, offsets


def _bio_to_entities(
    tokens: list[str],
    labels: list[str],
) -> list[MultiSpanEntity]:
    """Convert BIO token/label lists to character-offset MultiSpanEntity list."""
    text, offsets = _tokens_to_text_and_offsets(tokens)
    entities: list[MultiSpanEntity] = []
    i = 0
    while i < len(labels):
        label = labels[i]
        if label.startswith("B-"):
            entity_type = label[2:]
            start_idx = i
            end_idx = i + 1
            while end_idx < len(labels) and labels[end_idx] == f"I-{entity_type}":
                end_idx += 1
            char_start = offsets[start_idx][0]
            char_end = offsets[end_idx - 1][1]
            entity_text = text[char_start:char_end]
            entities.append(
                MultiSpanEntity(
                    type=entity_type,
                    text=entity_text,
                    spans=[(char_start, char_end)],
                )
            )
            i = end_idx
        else:
            i += 1
    return entities


def _entities_to_bio(
    text: str,
    entities: list[MultiSpanEntity],
) -> tuple[list[str], list[str]]:
    """Tokenize *text* by whitespace and assign BIO labels from *entities*.

    Returns ``(tokens, labels)``.
    """
    tokens = text.split()
    labels = ["O"] * len(tokens)

    token_spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = text.index(token, cursor)
        end = start + len(token)
        token_spans.append((start, end))
        cursor = end

    for entity in entities:
        indices: list[int] = []
        for ent_start, ent_end in entity.spans:
            for idx, (tok_start, tok_end) in enumerate(token_spans):
                if tok_start >= ent_start and tok_end <= ent_end:
                    indices.append(idx)
        if not indices:
            continue
        sorted_idx = sorted(set(indices))
        groups: list[list[int]] = []
        cur = [sorted_idx[0]]
        for idx in sorted_idx[1:]:
            if idx == cur[-1] + 1:
                cur.append(idx)
            else:
                groups.append(cur)
                cur = [idx]
        groups.append(cur)
        for group in groups:
            labels[group[0]] = f"B-{entity.type}"
            for idx in group[1:]:
                labels[idx] = f"I-{entity.type}"

    return tokens, labels


def _collect_entity_types(
    docs: list[MultiSpanNERDocument],
) -> list[EntityInfo]:
    """Collect unique entity types across all documents, sorted."""
    seen: set[str] = set()
    for doc in docs:
        for ent in doc.entities:
            seen.add(ent.type)
    return [EntityInfo(type=t) for t in sorted(seen)]


# ── Normalizer ────────────────────────────────────────────────────────


class Normalizer:
    """Normalize entity type names and document identifiers."""

    @staticmethod
    def normalize_type(name: str) -> str:
        """Convert CamelCase, space-separated, or hyphen-separated names
        to snake_case.

        Examples::

            >>> Normalizer.normalize_type("Food Repertoire")
            'food_repertoire'
            >>> Normalizer.normalize_type("AnimalProduct")
            'animal_product'
        """
        name = re.sub(r"[\s-]+", "_", name)
        name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        return name.lower()

    @staticmethod
    def normalize_id(doc_id: Any) -> str:
        """Normalize a document identifier to a stable string."""
        return normalize_example_id(doc_id)


# ── Abstract base ────────────────────────────────────────────────────


class FormatHandler(abc.ABC):
    """Abstract base for a corpus format reader/writer pair."""

    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        """Canonical format name (e.g. ``'brat'``, ``'conll'``)."""

    @abc.abstractmethod
    def read(
        self,
        input_dir: Path,
        normalizer: Normalizer | None = None,
    ) -> list[MultiSpanNERDocument]:
        """Read all documents from *input_dir*.

        Parameters
        ----------
        input_dir:
            Directory containing files in this handler's format.
        normalizer:
            Optional normalizer; when given, entity types and document IDs
            are normalised as they are read.
        """

    @abc.abstractmethod
    def write(
        self,
        docs: list[MultiSpanNERDocument],
        output_dir: Path,
        normalizer: Normalizer | None = None,
    ) -> None:
        """Write *docs* into *output_dir*.

        Parameters
        ----------
        docs:
            Documents to write.
        output_dir:
            Directory to create the output files in. Must exist and be empty
            (or will be created by the caller).
        normalizer:
            Optional normalizer; when given, entity types and document IDs
            are normalised before writing.
        """

    @classmethod
    @abc.abstractmethod
    def detect(cls, input_dir: Path) -> bool:
        """Return True if *input_dir* looks like this format."""


# ── BRAT handler ─────────────────────────────────────────────────────


class BratHandler(FormatHandler):
    """Read/write BRAT ``.txt`` / ``.ann`` pairs."""

    @classmethod
    def name(cls) -> str:
        return "brat"

    def read(
        self,
        input_dir: Path,
        normalizer: Normalizer | None = None,
    ) -> list[MultiSpanNERDocument]:
        docs: list[MultiSpanNERDocument] = []
        norm = normalizer or Normalizer()

        txt_files = sorted(input_dir.glob("*.txt"))
        for txt_path in txt_files:
            ann_path = txt_path.with_suffix(".ann")
            text = txt_path.read_text(encoding="utf-8")
            entities: list[MultiSpanEntity] = []
            if ann_path.exists():
                brat_entities = parse_ann_entities(ann_path)
                for be in brat_entities:
                    fragments = [text[s:e] for s, e in be.spans]
                    entities.append(
                        MultiSpanEntity(
                            type=norm.normalize_type(be.entity_type),
                            text=" ".join(fragments),
                            spans=[list(s) for s in be.spans],
                        )
                    )
            docs.append(
                MultiSpanNERDocument(
                    id=norm.normalize_id(txt_path.stem),
                    text=text,
                    entities=entities,
                )
            )
        return docs

    def write(
        self,
        docs: list[MultiSpanNERDocument],
        output_dir: Path,
        normalizer: Normalizer | None = None,
    ) -> None:
        norm = normalizer or Normalizer()
        for doc in docs:
            doc_id = norm.normalize_id(doc.id)
            txt_path = output_dir / f"{doc_id}.txt"
            ann_path = output_dir / f"{doc_id}.ann"

            txt_path.write_text(doc.text, encoding="utf-8")

            ann_lines: list[str] = []
            for idx, entity in enumerate(doc.entities, start=1):
                span_str = ";".join(f"{s} {e}" for s, e in entity.spans)
                ann_lines.append(
                    f"T{idx}\t{norm.normalize_type(entity.type)} {span_str}"
                    f"\t{entity.text}"
                )
            ann_path.write_text(
                "\n".join(ann_lines) + ("\n" if ann_lines else ""),
                encoding="utf-8",
            )

    @classmethod
    def detect(cls, input_dir: Path) -> bool:
        return any(f.suffix == ".ann" for f in input_dir.iterdir())


# ── CoNLL handler ────────────────────────────────────────────────────


class ConllHandler(FormatHandler):
    """Read/write CoNLL tab-separated token/label files."""

    @classmethod
    def name(cls) -> str:
        return "conll"

    def read(
        self,
        input_dir: Path,
        normalizer: Normalizer | None = None,
    ) -> list[MultiSpanNERDocument]:
        docs: list[MultiSpanNERDocument] = []
        norm = normalizer or Normalizer()

        conll_files = sorted(input_dir.glob("*.conll"))
        for conll_path in conll_files:
            with open(conll_path, "r", encoding="utf-8") as fh:
                tokens: list[str] = []
                labels: list[str] = []
                sent_id = 0
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        if tokens:
                            entities = _bio_to_entities(tokens, labels)
                            text, _ = _tokens_to_text_and_offsets(tokens)
                            docs.append(
                                MultiSpanNERDocument(
                                    id=norm.normalize_id(
                                        f"{conll_path.stem}_{sent_id}"
                                    ),
                                    text=text,
                                    entities=entities,
                                )
                            )
                            sent_id += 1
                            tokens = []
                            labels = []
                        continue
                    parts = stripped.split("\t")
                    if len(parts) < 2:
                        continue
                    tokens.append(parts[0])
                    labels.append(parts[-1])
                if tokens:
                    entities = _bio_to_entities(tokens, labels)
                    text, _ = _tokens_to_text_and_offsets(tokens)
                    docs.append(
                        MultiSpanNERDocument(
                            id=norm.normalize_id(f"{conll_path.stem}_{sent_id}"),
                            text=text,
                            entities=entities,
                        )
                    )
        return docs

    def write(
        self,
        docs: list[MultiSpanNERDocument],
        output_dir: Path,
        normalizer: Normalizer | None = None,
    ) -> None:
        lines: list[str] = []
        for doc in docs:
            tokens, labels = _entities_to_bio(doc.text, doc.entities)
            for token, label in zip(tokens, labels):
                lines.append(f"{token}\t{label}")
            lines.append("")
        (output_dir / "test.conll").write_text(
            "\n".join(lines), encoding="utf-8"
        )

    @classmethod
    def detect(cls, input_dir: Path) -> bool:
        return list(input_dir.glob("*.conll")) != []


# ── JSONL BIO handler ────────────────────────────────────────────────


class JsonlBioHandler(FormatHandler):
    """Read/write JSONL with ``{"id", "tokens", "labels"}`` per line."""

    @classmethod
    def name(cls) -> str:
        return "jsonl_bio"

    def read(
        self,
        input_dir: Path,
        normalizer: Normalizer | None = None,
    ) -> list[MultiSpanNERDocument]:
        docs: list[MultiSpanNERDocument] = []
        norm = normalizer or Normalizer()

        for jsonl_path in sorted(input_dir.glob("*.jsonl")):
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    data = json.loads(stripped)
                    tokens: list[str] = data["tokens"]
                    labels: list[str] = data["labels"]
                    entities = _bio_to_entities(tokens, labels)
                    text, _ = _tokens_to_text_and_offsets(tokens)
                    docs.append(
                        MultiSpanNERDocument(
                            id=norm.normalize_id(data.get("id", "")),
                            text=text,
                            entities=entities,
                        )
                    )
        return docs

    def write(
        self,
        docs: list[MultiSpanNERDocument],
        output_dir: Path,
        normalizer: Normalizer | None = None,
    ) -> None:
        norm = normalizer or Normalizer()
        jsonl_path = output_dir / "test.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for doc in docs:
                doc_id = norm.normalize_id(doc.id)
                tokens, labels = _entities_to_bio(doc.text, doc.entities)
                record = {
                    "id": doc_id,
                    "tokens": tokens,
                    "labels": labels,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    @classmethod
    def detect(cls, input_dir: Path) -> bool:
        for jsonl_path in input_dir.glob("*.jsonl"):
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        data = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    return "tokens" in data and "labels" in data
        return False


# ── JSONL TXT MultiSpan handler ──────────────────────────────────────


class JsonlTxtMultiSpanHandler(FormatHandler):
    """Read/write the standard NER-Annotated JSONL format.

    Each line of the JSONL file has ``{"id", "text", "entities"}`` where
    each entity carries ``{"type", "text", "offsets"}``.
    """

    @classmethod
    def name(cls) -> str:
        return "jsonl_txtmultispan"

    def read(
        self,
        input_dir: Path,
        normalizer: Normalizer | None = None,
    ) -> list[MultiSpanNERDocument]:
        docs: list[MultiSpanNERDocument] = []
        norm = normalizer or Normalizer()

        for jsonl_path in sorted(input_dir.glob("*.jsonl")):
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    data = json.loads(stripped)
                    raw_entities = data.get("entities", [])
                    entities = [
                        MultiSpanEntity(
                            type=norm.normalize_type(e["type"]),
                            text=e["text"],
                            spans=[tuple(s) for s in e.get("offsets", [])],
                        )
                        for e in raw_entities
                    ]
                    docs.append(
                        MultiSpanNERDocument(
                            id=norm.normalize_id(data["id"]),
                            text=data["text"],
                            entities=entities,
                        )
                    )
        return docs

    def write(
        self,
        docs: list[MultiSpanNERDocument],
        output_dir: Path,
        normalizer: Normalizer | None = None,
        entities: list[EntityInfo] | None = None,
        rules: list[RuleSection] | None = None,
    ) -> None:
        norm = normalizer or Normalizer()
        jsonl_path = output_dir / "test.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for doc in docs:
                doc_id = norm.normalize_id(doc.id)
                record: dict[str, Any] = {
                    "id": doc_id,
                    "text": doc.text,
                    "entities": [
                        {
                            "type": norm.normalize_type(e.type),
                            "text": e.text,
                            "offsets": [list(s) for s in e.spans],
                        }
                        for e in doc.entities
                    ],
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        if entities is not None:
            ent_list = [
                {
                    "name": norm.normalize_type(e.type),
                    "definition": e.definition or "",
                    "examples": e.examples or [],
                    "counter_examples": e.counter_examples or [],
                }
                for e in entities
            ]
            (output_dir / "entities.json").write_text(
                json.dumps(ent_list, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if rules is not None:
            rule_list = [
                {
                    "title": r.title,
                    "rules": r.rules,
                    "examples": r.examples,
                }
                for r in rules
            ]
            (output_dir / "rules.json").write_text(
                json.dumps(rule_list, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    @classmethod
    def detect(cls, input_dir: Path) -> bool:
        for jsonl_path in input_dir.glob("*.jsonl"):
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        data = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    return "text" in data and "entities" in data
        return False


# ── Format registry ──────────────────────────────────────────────────


_HANDLERS: dict[str, type[FormatHandler]] = {
    handler.name(): handler
    for handler in [
        BratHandler,
        ConllHandler,
        JsonlBioHandler,
        JsonlTxtMultiSpanHandler,
    ]
}


def get_handler(format_name: str) -> FormatHandler:
    """Return a ``FormatHandler`` instance for *format_name*.

    Raises
    ------
    ValueError
        If the format is not registered.
    """
    cls = _HANDLERS.get(format_name)
    if cls is None:
        raise ValueError(
            f"Unknown format {format_name!r}. "
            f"Available: {', '.join(sorted(_HANDLERS))}"
        )
    return cls()


def auto_detect_format(input_dir: str | Path) -> str:
    """Detect the corpus format in *input_dir*.

    Detection priority: BRAT > JSONL-multispan > JSONL-bio > CoNLL.

    Raises
    ------
    ValueError
        If the format cannot be determined.
    """
    path = Path(input_dir)
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    if BratHandler.detect(path):
        return "brat"
    if JsonlTxtMultiSpanHandler.detect(path):
        return "jsonl_txtmultispan"
    if JsonlBioHandler.detect(path):
        return "jsonl_bio"
    if ConllHandler.detect(path):
        return "conll"

    raise ValueError(
        f"Unable to auto-detect format in {path}. "
        "Directory must contain .ann files (BRAT), "
        ".jsonl files with 'tokens'/'labels' or 'text'/'entities', "
        "or .conll files."
    )


# ── Top-level conversion ─────────────────────────────────────────────


FORMAT_CHOICES = sorted(_HANDLERS)


def convert(
    input_dir: str | Path,
    output_dir: str | Path,
    target_format: str = "jsonl_txtmultispan",
    normalizer: Normalizer | None = None,
    no_ent_file: bool = False,
    no_rule_file: bool = False,
) -> None:
    """Read a corpus from *input_dir* and write it in *target_format*.

    Parameters
    ----------
    input_dir:
        Source directory in any supported format.
    output_dir:
        Target directory. Must not exist, or must be empty.
    target_format:
        One of ``FORMAT_CHOICES``.
    normalizer:
        Normalizer for IDs and entity type names. Created automatically
        when ``None``.
    no_ent_file:
        If True, skip writing ``entities.json`` (only relevant for
        ``jsonl_txtmultispan``).
    no_rule_file:
        If True, skip writing ``rules.json``.

    Raises
    ------
    ValueError
        If *target_format* is unknown, or *output_dir* exists and is
        non-empty.
    """
    norm = normalizer or Normalizer()
    src = Path(input_dir)
    dst = Path(output_dir)

    source_format = auto_detect_format(src)

    if target_format not in _HANDLERS:
        raise ValueError(
            f"Unknown target format {target_format!r}. "
            f"Available: {', '.join(FORMAT_CHOICES)}"
        )

    if dst.exists():
        contents = list(dst.iterdir())
        if contents:
            raise ValueError(
                f"Output directory {dst} exists and is not empty. "
                "Please provide an empty or non-existent directory."
            )
    else:
        dst.mkdir(parents=True)

    reader = get_handler(source_format)
    docs = reader.read(src, normalizer=norm)

    writer = get_handler(target_format)

    if isinstance(writer, JsonlTxtMultiSpanHandler):
        entities_info = _collect_entity_types(docs) if not no_ent_file else None
        rules_info: list[RuleSection] | None = [] if not no_rule_file else None
        writer.write(
            docs,
            dst,
            normalizer=norm,
            entities=entities_info,
            rules=rules_info,
        )
    else:
        writer.write(docs, dst, normalizer=norm)

    print(
        f"Converted {len(docs)} document(s) from {source_format} "
        f"to {target_format} in {dst}"
    )
