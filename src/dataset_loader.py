"""Loading of standard NER dataset directories.

Provides the ``DatasetLoader`` class which reads ``test.jsonl``,
``entities.json``, and the optional ``rules.json`` from a dataset
directory and returns the corresponding schemas.

See AGENTS.md for the full dataset format specification.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.schemas import EntityInfo, MultiSpanEntity, MultiSpanNERDocument, NERCorpus, RuleSection


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
