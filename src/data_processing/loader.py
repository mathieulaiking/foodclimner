"""Loading of standard NER dataset directories.

Provides the NER document schemas (``MultiSpanNERDocument``,
``TokenTagNERDocument``, ``NERCorpus``, ``EntityInfo``, ``RuleSection``)
and the ``DatasetLoader`` class which reads ``test.jsonl``,
``entities.json``, and the optional ``rules.json`` from a dataset
directory.

See AGENTS.md for the full dataset format specification.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── NER Document schemas ────────────────────────────────────────────


@dataclass
class TokenTagNERDocument:
    """Token-tagged NER document with BIO labels."""

    id: str | int
    tokens: list[str]
    labels: list[str]


@dataclass
class MultiSpanEntity:
    """One NER entity mention with character-offset spans.

    ``type`` is the canonical entity name.
    ``text`` is the surface string as it appears in the document.
    ``spans`` is a list of ``(start, end)`` character-offset pairs,
    which supports both nested entities (different types overlapping)
    and discontinuous entities (multiple span tuples for one entity).
    """

    type: str
    text: str
    spans: list[tuple[int, int]]


@dataclass
class MultiSpanNERDocument:
    """Document-level NER annotations with multi-span entities.

    ``entities`` is a list of ``MultiSpanEntity`` objects, each carrying
    the canonical type, surface text, and character-offset spans.
    """

    id: str
    text: str
    entities: list[MultiSpanEntity]


# ── Dataset / data-loading schemas ──────────────────────────────────


@dataclass
class NERCorpus:
    """Loaded dataset contents.

    Attributes
    ----------
    documents:
        List of NER-annotated documents.
    entities_info:
        List of entity type definitions.
    """

    documents: list[TokenTagNERDocument | MultiSpanNERDocument] = field(default_factory=list)
    entities_info: list[EntityInfo] = field(default_factory=list)

    @property
    def entity_names(self) -> list[str]:
        """Return the canonical name of each entity type definition."""
        return [e.type for e in self.entities_info]


@dataclass
class RuleSection:
    """One rule group with a title, list of rules, and examples."""

    title: str
    rules: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render this rule section as markdown."""
        parts = [f"### {self.title}", ""]
        for rule in self.rules:
            parts.append(f"- {rule}")
        if self.examples:
            parts.append("")
            parts.append(f"**Examples:** {', '.join(self.examples)}")
        return "\n".join(parts)


@dataclass
class EntityInfo:
    """One entity type definition with optional documentation.

    Attributes
    ----------
    type:
        Canonical entity type name (required).
    definition:
        Optional human-readable definition of the entity.
    examples:
        Optional list of example mentions.
    counter_examples:
        Optional list of counter-example strings that look similar
        but should **not** be annotated with this type.
    """

    type: str
    definition: Optional[str] = None
    examples: Optional[list[str]] = None
    counter_examples: Optional[list[str]] = None

    def to_markdown(self) -> str:
        """Render this entity info as a markdown bullet list item."""
        lines = [f"* {self.type}"]
        if self.definition:
            lines.append(f"  - Definition: {self.definition}")
        if self.examples:
            examples_str = ", ".join(self.examples)
            lines.append(f"  - Examples: {examples_str}")
        if self.counter_examples:
            counter_str = ", ".join(self.counter_examples)
            lines.append(f"  - Counter-examples: {counter_str}")
        return "\n".join(lines)


__all__ = [
    "EntityInfo",
    "MultiSpanEntity",
    "MultiSpanNERDocument",
    "NERCorpus",
    "RuleSection",
    "TokenTagNERDocument",
    "DatasetLoader",
]


class DatasetLoader:
    """Load a standard NER dataset directory.

    Parameters
    ----------
    data_dir:
        Path to the dataset directory.
    """

    def __init__(self, data_dir: str) -> None:
        self.data_dir = Path(data_dir)

    def load_corpus(self, lazy_entities: bool = False) -> NERCorpus:
        """Load ``test.jsonl`` and ``entities.json`` from the dataset directory.

        Parameters
        ----------
        lazy_entities:
            When ``True``, skip parsing gold ``MultiSpanEntity`` objects
            (saves CPU when entities are not needed, e.g. during inference).

        Returns
        -------
        ``NERCorpus`` with loaded documents and entity type definitions.

        Raises
        ------
        FileNotFoundError
            If either required file is missing.
        ValueError
            If file contents are invalid.
        """
        test_path = self.data_dir / "test.jsonl"
        entities_path = self.data_dir / "entities.json"

        if not test_path.is_file():
            raise FileNotFoundError(f"test.jsonl not found in {self.data_dir}")
        if not entities_path.is_file():
            raise FileNotFoundError(f"entities.json not found in {self.data_dir}")

        documents: list[MultiSpanNERDocument] = []
        with test_path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    doc = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in test.jsonl at line {line_number}: {exc}"
                    ) from exc
                if not isinstance(doc, dict) or "id" not in doc or "text" not in doc:
                    raise ValueError(
                        "Each test.jsonl line must be a dict with 'id' and 'text'"
                    )
                if lazy_entities:
                    parsed_entities = []
                else:
                    raw_entities = doc.get("entities", [])
                    parsed_entities = [
                        MultiSpanEntity(
                            type=e["type"],
                            text=e["text"],
                            spans=[tuple(s) for s in e.get("offsets", [])],
                        )
                        for e in raw_entities
                    ]
                documents.append(
                    MultiSpanNERDocument(
                        id=doc["id"],
                        text=doc["text"],
                        entities=parsed_entities,
                    )
                )

        with entities_path.open("r", encoding="utf-8") as fh:
            raw_entity_types = json.load(fh)

        if not isinstance(raw_entity_types, list):
            raise ValueError("entities.json must contain a JSON array")

        entity_infos: list[EntityInfo] = []
        for e in raw_entity_types:
            if "name" not in e:
                continue
            entity_infos.append(
                EntityInfo(
                    type=e["name"],
                    definition=e.get("definition"),
                    examples=e.get("examples"),
                    counter_examples=e.get("counter_examples"),
                )
            )

        return NERCorpus(documents=documents, entities_info=entity_infos)

    def load_rules(self) -> list[RuleSection]:
        """Load annotation rules from ``rules.json`` in the dataset directory.

        The file is optional — returns an empty list when not present.

        Returns
        -------
        List of ``RuleSection`` dataclass instances.
        """
        rules_path = self.data_dir / "rules.json"
        if not rules_path.is_file():
            return []
        with rules_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return [
            RuleSection(
                title=item["title"],
                rules=item.get("rules", []),
                examples=item.get("examples", []),
            )
            for item in raw
        ]
