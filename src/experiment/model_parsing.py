"""Unified parsers for model outputs and BIO tagging."""

from __future__ import annotations

import abc
import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from src.data_processing.loader import MultiSpanEntity
from src.evaluation import EvalEntity


def _normalize_example_id(example_id: Any) -> str:
    """Return a stable string identifier for one example."""
    return str(example_id)


def _strip_extension(filename: str) -> str:
    """Return filename stem by removing one extension level."""
    stem, ext = os.path.splitext(str(filename))
    if ext:
        return stem
    return str(filename)


class BaseParser(abc.ABC):
    """Base class for model-output parsers.

    Concrete subclasses implement three output formats:

    * ``parse_to_bio`` — token-level BIO labels (FlatNerDocument).
    * ``_extract_predicted_entities`` — token-span entities for evaluation.
    * ``parse_to_annotated_entities`` — char-offset entities (MultiSpanNERDocument).
    """

    def __init__(self, entity_names: list[str]):
        """Store canonical entity names for label normalization."""
        self.entity_names = entity_names

    @abc.abstractmethod
    def parse_to_bio(self, model_output: Any, tokens: list[str]) -> list[str]:
        """Parse model output and convert to BIO tags aligned with tokens."""
        raise NotImplementedError

    @abc.abstractmethod
    def parse_to_annotated_entities(
        self,
        model_output: Any,
        text: str,
        **kwargs: Any,
    ) -> list[MultiSpanEntity]:
        """Parse model output to char-offset entity list.

        Returns ``MultiSpanEntity`` objects with canonical type, surface text,
        and character-offset spans — suitable for the NER-Annotated format.

        Parameters
        ----------
        model_output:
            Raw model prediction output.
        text:
            Original document text (used to derive character offsets).
        **kwargs:
            Parser-specific options (e.g. score threshold for GLiNER).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def parse_directory(
        self,
        output_dir: str,
        reference_filepath: str,
    ) -> list[dict[str, Any]]:
        """Parse all compatible files in a directory against a reference JSONL."""
        raise NotImplementedError

    def _normalize_label(self, label: str) -> str:
        """Normalize labels for matching across formats."""
        return re.sub(r"[^a-z0-9]+", "", label.lower())

    def _build_canonical_by_norm(self) -> dict[str, str]:
        """Build a normalized label map to canonical entity names."""
        canonical_by_norm: dict[str, str] = {}
        for name in self.entity_names:
            if not name:
                continue
            prompt_label = re.sub(r"[-_]+", " ", name).strip()
            canonical_by_norm[self._normalize_label(name)] = name
            canonical_by_norm[self._normalize_label(prompt_label)] = name
        return canonical_by_norm

    def _build_text_offsets(
        self,
        tokens: list[str],
    ) -> tuple[str, list[tuple[int, int]]]:
        """Return whitespace-joined text and token char offsets."""
        text = " ".join(tokens)
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for token in tokens:
            start = cursor
            end = start + len(token)
            offsets.append((start, end))
            cursor = end + 1
        return text, offsets

    def _span_to_indices(
        self,
        offsets: list[tuple[int, int]],
        start: int,
        end: int,
    ) -> list[int]:
        """Convert character span to contiguous token indices."""
        indices = [
            idx
            for idx, (tok_start, tok_end) in enumerate(offsets)
            if tok_end > start and tok_start < end
        ]
        if not indices:
            return []
        expected = list(range(indices[0], indices[-1] + 1))
        return indices if indices == expected else []

    def _apply_indices(
        self,
        labels: list[str],
        indices: list[int],
        label: str,
    ) -> bool:
        """Apply BIO tags to free indices and report success."""
        if not indices or any(labels[idx] != "O" for idx in indices):
            return False
        labels[indices[0]] = f"B-{label}"
        for idx in indices[1:]:
            labels[idx] = f"I-{label}"
        return True

    def _load_reference_records(
        self,
        reference_filepath: str,
    ) -> list[dict[str, Any]]:
        """Load reference rows from a JSONL file and validate their shape."""
        reference_path = Path(reference_filepath)
        if not reference_path.is_file():
            raise FileNotFoundError(
                f"Reference file not found: {reference_path}"
            )

        reference_records: list[dict[str, Any]] = []
        with reference_path.open("r", encoding="utf8") as reference_file:
            for line_number, line in enumerate(reference_file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(
                        "Reference JSONL rows must be objects in "
                        f"{reference_path} at line {line_number}."
                    )
                if "id" not in payload:
                    raise ValueError(
                        "Reference JSONL rows must include an id field in "
                        f"{reference_path} at line {line_number}."
                    )
                if "tokens" not in payload:
                    raise ValueError(
                        "Reference JSONL rows must include a tokens field in "
                        f"{reference_path} at line {line_number}."
                    )

                reference_records.append(payload)

        return reference_records

    def _parse_directory(
        self,
        output_dir: str,
        reference_filepath: str,
        expected_suffix: str,
        load_model_output: Callable[[Path], Any],
    ) -> list[dict[str, Any]]:
        """Parse a directory of model outputs using a reference JSONL file."""
        output_path = Path(output_dir)
        if not output_path.is_dir():
            raise FileNotFoundError(f"Output directory not found: {output_path}")

        reference_records = self._load_reference_records(reference_filepath)
        reference_by_id: dict[str, dict[str, Any]] = {}
        reference_order: list[str] = []
        for record in reference_records:
            example_id = _normalize_example_id(record["id"])
            if example_id in reference_by_id:
                raise ValueError(
                    "Reference JSONL contains duplicate example id: "
                    f"{example_id}"
                )
            reference_by_id[example_id] = record
            reference_order.append(example_id)

        output_files_by_id: dict[str, Path] = {}
        for file_path in sorted(output_path.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() != expected_suffix:
                continue

            example_id = _normalize_example_id(_strip_extension(file_path.name))
            if example_id in output_files_by_id:
                raise ValueError(
                    "Duplicate output files for example id "
                    f"{example_id} in {output_path}"
                )
            output_files_by_id[example_id] = file_path

        unknown_ids = sorted(set(output_files_by_id) - set(reference_by_id))
        if unknown_ids:
            raise ValueError(
                "Output directory contains files that do not match the "
                f"reference JSONL: {', '.join(unknown_ids)}"
            )

        missing_ids = [
            example_id
            for example_id in reference_order
            if example_id not in output_files_by_id
        ]
        if missing_ids:
            raise FileNotFoundError(
                "Missing output files for reference examples: "
                f"{', '.join(missing_ids)}"
            )

        parsed_rows: list[dict[str, Any]] = []
        for example_id in reference_order:
            file_path = output_files_by_id[example_id]
            record = reference_by_id[example_id]
            tokens = record["tokens"]
            if not isinstance(tokens, list):
                raise ValueError(
                    "Reference tokens must be a list for example id "
                    f"{example_id}"
                )

            normalized_tokens = [str(token) for token in tokens]
            model_output = load_model_output(file_path)
            labels = self.parse_to_bio(
                model_output,
                normalized_tokens,
            )
            parsed_rows.append(
                {
                    "id": example_id,
                    "tokens": normalized_tokens,
                    "labels": labels,
                }
            )

        return parsed_rows

    # -- entity-list format (nested-friendly) --------------------------------

    @abc.abstractmethod
    def _extract_predicted_entities(
        self, model_output: Any, tokens: list[str]
    ) -> list[EvalEntity]:
        """Extract entities from model output as ``EvalEntity`` objects.

        Each ``EvalEntity`` carries a ``type`` (canonical entity name) and
        ``spans`` (tuple of ``(start, end)`` token-index pairs). Nested
        entities of different types are represented as separate entries;
        discontinuous entities use multiple span tuples.
        """

    def parse_to_entities(
        self, model_output: Any, tokens: list[str]
    ) -> list[dict[str, Any]]:
        """Parse model output to entity-list format (nested-friendly).

        Returns a JSON-serialisable list of::

            {"type": "disease", "spans": [[0, 2]]}

        The format is compatible with ``evaluate.py``.
        """
        entities = self._extract_predicted_entities(model_output, tokens)
        return self._eval_entities_to_output(entities)

    @staticmethod
    def _eval_entities_to_output(
        entities: list[EvalEntity],
    ) -> list[dict[str, Any]]:
        """Convert a list of ``EvalEntity`` to JSON-serialisable dicts."""
        return [
            {"type": e.type, "spans": [list(s) for s in e.spans]}
            for e in entities
        ]


class JSONEntParser(BaseParser):
    """Base class for JSON-format parsers with integrated payload extraction."""

    @staticmethod
    def _extract_json_payload(response_text: str) -> Any | None:
        """Extract a JSON object or list from a model response string."""
        if not response_text:
            return None

        match = re.search(r"(\{.*?\}|\[.*?\])", response_text, re.DOTALL)
        if not match:
            return None

        payload = match.group(0)
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(payload)
            except (ValueError, SyntaxError):
                return None


class GlinerParser(BaseParser):
    """Parse GLiNER JSON outputs into token-level BIO labels and entity lists."""

    @staticmethod
    def _to_int(value: Any) -> int:
        """Convert a value to int, returning 0 on failure."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _iter_gliner_entities(
        self, model_output: Any
    ) -> list[tuple[str, int, int, str, float]]:
        """Yield sorted (canonical, start, end, text, score) tuples."""
        if not isinstance(model_output, list):
            return []

        canonical_by_norm = self._build_canonical_by_norm()
        sorted_entities = sorted(
            (e for e in model_output if isinstance(e, dict)),
            key=lambda e: (
                float(e.get("score", 0.0)),
                self._to_int(e.get("end")) - self._to_int(e.get("start")),
            ),
            reverse=True,
        )

        result: list[tuple[str, int, int, str, float]] = []
        for entity in sorted_entities:
            raw_label = str(entity.get("label", ""))
            canonical = canonical_by_norm.get(self._normalize_label(raw_label))
            if not canonical:
                continue
            result.append(
                (
                    canonical,
                    self._to_int(entity.get("start")),
                    self._to_int(entity.get("end")),
                    str(entity.get("text", "")),
                    float(entity.get("score", 0.0)),
                )
            )
        return result

    def parse_to_bio(self, model_output: Any, tokens: list[str]) -> list[str]:
        """Parse GLiNER JSON output and align to BIO labels."""
        labels = ["O"] * len(tokens)
        if not tokens:
            return labels

        text, offsets = self._build_text_offsets(tokens)

        for canonical, start, end, span_text, _ in self._iter_gliner_entities(
            model_output
        ):
            if end > start and self._apply_indices(
                labels,
                self._span_to_indices(offsets, start, end),
                canonical,
            ):
                continue

            span_text = span_text.strip()
            if not span_text:
                continue
            for match in re.finditer(re.escape(span_text), text):
                if self._apply_indices(
                    labels,
                    self._span_to_indices(
                        offsets,
                        match.start(),
                        match.end(),
                    ),
                    canonical,
                ):
                    break

        return labels

    def _extract_predicted_entities(
        self, model_output: Any, tokens: list[str]
    ) -> list[EvalEntity]:
        """Extract entities from GLiNER output as ``EvalEntity`` objects."""
        if not tokens:
            return []

        text, offsets = self._build_text_offsets(tokens)

        seen: set[tuple[str, int, int]] = set()
        results: list[EvalEntity] = []

        for canonical, start, end, span_text, _ in self._iter_gliner_entities(
            model_output
        ):
            indices: list[int] = []
            if end > start:
                indices = self._span_to_indices(offsets, start, end)

            if not indices:
                span_text = span_text.strip()
                if not span_text:
                    continue
                for match in re.finditer(re.escape(span_text), text):
                    indices = self._span_to_indices(
                        offsets, match.start(), match.end()
                    )
                    if indices:
                        break

            if not indices:
                continue

            token_start = indices[0]
            token_end = indices[-1]

            key = (canonical, token_start, token_end)
            if key in seen:
                continue
            seen.add(key)

            results.append(
                EvalEntity(
                    type=canonical,
                    spans=((token_start, token_end),),
                )
            )

        return results

    def parse_to_annotated_entities(
        self,
        model_output: Any,
        text: str,
        threshold: float = 0.5,
    ) -> list[MultiSpanEntity]:
        """Parse GLiNER output to char-offset entity list.

        Parameters
        ----------
        model_output:
            Raw GLiNER prediction list.
        text:
            Original document text (unused for GLiNER — character offsets
            are returned directly by the model).
        threshold:
            Minimum confidence score to include an entity.
        """
        seen: set[tuple[str, int, int]] = set()
        results: list[MultiSpanEntity] = []

        for canonical, start, end, span_text, score in self._iter_gliner_entities(
            model_output
        ):
            if score < threshold:
                continue
            if end <= start:
                continue

            key = (canonical, start, end)
            if key in seen:
                continue
            seen.add(key)

            results.append(
                MultiSpanEntity(
                    type=canonical,
                    text=span_text,
                    spans=[(start, end)],
                )
            )

        return results

    def parse_directory(
        self,
        output_dir: str,
        reference_filepath: str,
    ) -> list[dict[str, Any]]:
        """Parse a directory of GLiNER JSON outputs."""

        def load_json_output(file_path: Path) -> Any:
            with file_path.open("r", encoding="utf8") as output_file:
                return json.load(output_file)

        return self._parse_directory(
            output_dir=output_dir,
            reference_filepath=reference_filepath,
            expected_suffix=".json",
            load_model_output=load_json_output,
        )


class GollieParser(BaseParser):
    """Parse GoLLIE class spans into token-level BIO labels and entity lists."""

    def parse_to_bio(self, model_output: Any, tokens: list[str]) -> list[str]:
        """Parse GoLLIE output text and align to BIO labels."""
        labels = ["O"] * len(tokens)
        if not tokens or not isinstance(model_output, str):
            return labels

        canonical_by_norm = self._build_canonical_by_norm()
        text, offsets = self._build_text_offsets(tokens)

        pattern = re.compile(
            r"(?P<label>[A-Za-z_][A-Za-z0-9_]*)\s*"
            r"\(\s*span\s*=\s*(?P<quote>[\"\'])"
            r"(?P<span>.*?)(?P=quote)\s*\)",
        )

        entities: list[tuple[str, str]] = []
        for match in pattern.finditer(model_output):
            raw_label = match.group("label")
            snake_label = re.sub(r"(?<!^)([A-Z])", r"_\1", raw_label)
            snake_label = snake_label.lower()
            canonical = canonical_by_norm.get(
                self._normalize_label(snake_label),
            )
            if not canonical:
                canonical = canonical_by_norm.get(
                    self._normalize_label(raw_label),
                )
            if not canonical:
                continue

            span_text = match.group("span").strip()
            if span_text:
                entities.append((canonical, span_text))

        entities.sort(key=lambda item: len(item[1]), reverse=True)
        for canonical, span_text in entities:
            for match in re.finditer(re.escape(span_text), text):
                if self._apply_indices(
                    labels,
                    self._span_to_indices(
                        offsets,
                        match.start(),
                        match.end(),
                    ),
                    canonical,
                ):
                    break

        return labels

    def _extract_predicted_entities(
        self, model_output: Any, tokens: list[str]
    ) -> list[EvalEntity]:
        """Extract entities from GoLLIE output as ``EvalEntity`` objects."""
        if not tokens or not isinstance(model_output, str):
            return []

        canonical_by_norm = self._build_canonical_by_norm()
        text, offsets = self._build_text_offsets(tokens)

        pattern = re.compile(
            r"(?P<label>[A-Za-z_][A-Za-z0-9_]*)\s*"
            r"\(\s*span\s*=\s*(?P<quote>[\"\'])"
            r"(?P<span>.*?)(?P=quote)\s*\)",
        )

        raw_entities: list[tuple[str, str]] = []
        for match in pattern.finditer(model_output):
            raw_label = match.group("label")
            snake_label = re.sub(r"(?<!^)([A-Z])", r"_\1", raw_label)
            snake_label = snake_label.lower()
            canonical = canonical_by_norm.get(
                self._normalize_label(snake_label),
            )
            if not canonical:
                canonical = canonical_by_norm.get(
                    self._normalize_label(raw_label),
                )
            if not canonical:
                continue

            span_text = match.group("span").strip()
            if span_text:
                raw_entities.append((canonical, span_text))

        seen: set[tuple[str, int, int]] = set()
        results: list[EvalEntity] = []

        raw_entities.sort(key=lambda item: len(item[1]), reverse=True)
        for canonical, span_text in raw_entities:
            for match in re.finditer(re.escape(span_text), text):
                indices = self._span_to_indices(
                    offsets, match.start(), match.end()
                )
                if not indices:
                    continue
                token_start = indices[0]
                token_end = indices[-1]
                key = (canonical, token_start, token_end)
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    EvalEntity(
                        type=canonical,
                        spans=((token_start, token_end),),
                    )
                )
                break

        return results

    def parse_to_annotated_entities(
        self,
        model_output: Any,
        text: str,
    ) -> list[MultiSpanEntity]:
        """Parse GoLLIE output to char-offset entity list."""
        if not isinstance(model_output, str):
            return []

        pattern = re.compile(
            r"(?P<label>[A-Za-z_][A-Za-z0-9_]*)\s*"
            r"\(\s*span\s*=\s*(?P<quote>[\"\'])"
            r"(?P<span>.*?)(?P=quote)\s*\)",
        )

        raw_entities: list[tuple[str, str]] = []
        for match in pattern.finditer(model_output):
            raw_label = match.group("label")
            snake_label = re.sub(r"(?<!^)([A-Z])", r"_\1", raw_label)
            snake_label = snake_label.lower()
            canonical = self._build_canonical_by_norm().get(
                self._normalize_label(snake_label),
            )
            if not canonical:
                canonical = self._build_canonical_by_norm().get(
                    self._normalize_label(raw_label),
                )
            if not canonical:
                continue

            span_text = match.group("span").strip()
            if span_text:
                raw_entities.append((canonical, span_text))

        seen: set[tuple[str, int, int]] = set()
        results: list[MultiSpanEntity] = []

        raw_entities.sort(key=lambda item: len(item[1]), reverse=True)
        for canonical, span_text in raw_entities:
            for match in re.finditer(re.escape(span_text), text):
                start, end = match.start(), match.end()
                key = (canonical, start, end)
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    MultiSpanEntity(
                        type=canonical,
                        text=span_text,
                        spans=[(start, end)],
                    )
                )
                break

        return results

    def parse_directory(
        self,
        output_dir: str,
        reference_filepath: str,
    ) -> list[dict[str, Any]]:
        """Parse a directory of GoLLIE TXT outputs."""

        def load_text_output(file_path: Path) -> Any:
            return file_path.read_text(encoding="utf8")

        return self._parse_directory(
            output_dir=output_dir,
            reference_filepath=reference_filepath,
            expected_suffix=".txt",
            load_model_output=load_text_output,
        )


class JSONMultiEntParser(JSONEntParser):
    """Parse JSON dictionaries of entity lists into BIO labels."""

    def __init__(self, entity_names: list[str]):
        super().__init__(entity_names)
        self._canonical_by_norm = self._build_canonical_by_norm()

    def parse_to_bio(self, model_output: str, tokens: list[str]) -> list[str]:
        """Parse JSON entity lists from raw model output and align to BIO labels."""
        labels = ["O"] * len(tokens)
        if not tokens or not isinstance(model_output, str):
            return labels

        parsed_payload = self._extract_json_payload(model_output)
        if not isinstance(parsed_payload, dict):
            return labels

        canonical_by_norm = self._canonical_by_norm
        text, offsets = self._build_text_offsets(tokens)

        entities: list[tuple[str, str]] = []
        for raw_label, mentions in parsed_payload.items():
            canonical = canonical_by_norm.get(
                self._normalize_label(str(raw_label)),
            )
            if not canonical:
                continue

            if isinstance(mentions, str):
                mentions_list = [mentions]
            elif isinstance(mentions, list):
                mentions_list = mentions
            else:
                continue

            for mention in mentions_list:
                if not isinstance(mention, str):
                    continue
                span_text = re.sub(r"\s+", " ", mention).strip()
                if span_text:
                    entities.append((canonical, span_text))

        entities.sort(key=lambda item: len(item[1]), reverse=True)
        for canonical, span_text in entities:
            for match in re.finditer(re.escape(span_text), text):
                if self._apply_indices(
                    labels,
                    self._span_to_indices(
                        offsets,
                        match.start(),
                        match.end(),
                    ),
                    canonical,
                ):
                    break

        return labels

    def _extract_predicted_entities(
        self, model_output: str, tokens: list[str]
    ) -> list[EvalEntity]:
        """Extract entities from JSON multi-entity output as ``EvalEntity`` objects."""
        if not tokens or not isinstance(model_output, str):
            return []

        parsed_payload = self._extract_json_payload(model_output)
        if not isinstance(parsed_payload, dict):
            return []

        canonical_by_norm = self._canonical_by_norm
        text, offsets = self._build_text_offsets(tokens)

        raw_entities: list[tuple[str, str]] = []
        for raw_label, mentions in parsed_payload.items():
            canonical = canonical_by_norm.get(
                self._normalize_label(str(raw_label)),
            )
            if not canonical:
                continue

            if isinstance(mentions, str):
                mentions_list = [mentions]
            elif isinstance(mentions, list):
                mentions_list = mentions
            else:
                continue

            for mention in mentions_list:
                if not isinstance(mention, str):
                    continue
                span_text = re.sub(r"\s+", " ", mention).strip()
                if span_text:
                    raw_entities.append((canonical, span_text))

        seen: set[tuple[str, int, int]] = set()
        results: list[EvalEntity] = []

        raw_entities.sort(key=lambda item: len(item[1]), reverse=True)
        for canonical, span_text in raw_entities:
            for match in re.finditer(re.escape(span_text), text):
                indices = self._span_to_indices(
                    offsets, match.start(), match.end()
                )
                if not indices:
                    continue
                token_start = indices[0]
                token_end = indices[-1]
                key = (canonical, token_start, token_end)
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    EvalEntity(
                        type=canonical,
                        spans=((token_start, token_end),),
                    )
                )
                break

        return results

    def parse_to_annotated_entities(
        self,
        model_output: str,
        text: str,
    ) -> list[MultiSpanEntity]:
        """Parse JSON multi-entity output to char-offset entity list."""
        if not isinstance(model_output, str):
            return []

        parsed_payload = self._extract_json_payload(model_output)
        if not isinstance(parsed_payload, dict):
            return []

        canonical_by_norm = self._canonical_by_norm

        raw_entities: list[tuple[str, str]] = []
        for raw_label, mentions in parsed_payload.items():
            canonical = canonical_by_norm.get(
                self._normalize_label(str(raw_label)),
            )
            if not canonical:
                continue

            if isinstance(mentions, str):
                mentions_list = [mentions]
            elif isinstance(mentions, list):
                mentions_list = mentions
            else:
                continue

            for mention in mentions_list:
                if not isinstance(mention, str):
                    continue
                span_text = re.sub(r"\s+", " ", mention).strip()
                if span_text:
                    raw_entities.append((canonical, span_text))

        seen: set[tuple[str, int, int]] = set()
        results: list[MultiSpanEntity] = []

        raw_entities.sort(key=lambda item: len(item[1]), reverse=True)
        for canonical, span_text in raw_entities:
            for match in re.finditer(re.escape(span_text), text):
                start, end = match.start(), match.end()
                key = (canonical, start, end)
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    MultiSpanEntity(
                        type=canonical,
                        text=span_text,
                        spans=[(start, end)],
                    )
                )
                break

        return results

    def parse_directory(
        self,
        output_dir: str,
        reference_filepath: str,
    ) -> list[dict[str, Any]]:
        """Parse a directory of TXT outputs containing JSON objects."""

        def load_text_output(file_path: Path) -> Any:
            return file_path.read_text(encoding="utf8")

        return self._parse_directory(
            output_dir=output_dir,
            reference_filepath=reference_filepath,
            expected_suffix=".txt",
            load_model_output=load_text_output,
        )


class JSONSingleEntParser(JSONEntParser):
    """Parse a JSON list of mentions for a single entity type."""

    def __init__(self, entity_name: str) -> None:
        """Initialize with the canonical entity name for tagging."""
        super().__init__([entity_name])
        self._label = entity_name

    def parse_to_bio(self, model_output: str, tokens: list[str]) -> list[str]:
        """Parse JSON list output from raw model output and align to BIO labels."""
        labels = ["O"] * len(tokens)
        if not tokens or not isinstance(model_output, str):
            return labels

        parsed_payload = self._extract_json_payload(model_output)
        if parsed_payload is None:
            return labels

        if isinstance(parsed_payload, str):
            mentions_list = [parsed_payload]
        elif isinstance(parsed_payload, list):
            mentions_list = parsed_payload
        else:
            return labels

        text, offsets = self._build_text_offsets(tokens)

        mentions: list[str] = []
        for mention in mentions_list:
            if not isinstance(mention, str):
                continue
            span_text = re.sub(r"\s+", " ", mention).strip()
            if span_text:
                mentions.append(span_text)

        for span_text in sorted(set(mentions), key=len, reverse=True):
            for match in re.finditer(re.escape(span_text), text):
                if self._apply_indices(
                    labels,
                    self._span_to_indices(
                        offsets,
                        match.start(),
                        match.end(),
                    ),
                    self._label,
                ):
                    break

        return labels

    def _extract_predicted_entities(
        self, model_output: str, tokens: list[str]
    ) -> list[EvalEntity]:
        """Extract entities from JSON single-entity output as ``EvalEntity`` objects."""
        if not tokens or not isinstance(model_output, str):
            return []

        parsed_payload = self._extract_json_payload(model_output)
        if parsed_payload is None:
            return []

        if isinstance(parsed_payload, str):
            mentions_list = [parsed_payload]
        elif isinstance(parsed_payload, list):
            mentions_list = parsed_payload
        else:
            return []

        text, offsets = self._build_text_offsets(tokens)

        mentions: list[str] = []
        for mention in mentions_list:
            if not isinstance(mention, str):
                continue
            span_text = re.sub(r"\s+", " ", mention).strip()
            if span_text:
                mentions.append(span_text)

        seen: set[tuple[str, int, int]] = set()  # type is always self._label
        results: list[EvalEntity] = []

        for span_text in sorted(set(mentions), key=len, reverse=True):
            for match in re.finditer(re.escape(span_text), text):
                indices = self._span_to_indices(
                    offsets, match.start(), match.end()
                )
                if not indices:
                    continue
                token_start = indices[0]
                token_end = indices[-1]
                key = (self._label, token_start, token_end)
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    EvalEntity(
                        type=self._label,
                        spans=((token_start, token_end),),
                    )
                )
                break

        return results

    def parse_to_annotated_entities(
        self,
        model_output: str,
        text: str,
    ) -> list[MultiSpanEntity]:
        """Parse JSON single-entity output to char-offset entity list."""
        if not isinstance(model_output, str):
            return []

        parsed_payload = self._extract_json_payload(model_output)
        if parsed_payload is None:
            return []

        if isinstance(parsed_payload, str):
            mentions_list = [parsed_payload]
        elif isinstance(parsed_payload, list):
            mentions_list = parsed_payload
        else:
            return []

        mentions: list[str] = []
        for mention in mentions_list:
            if not isinstance(mention, str):
                continue
            span_text = re.sub(r"\s+", " ", mention).strip()
            if span_text:
                mentions.append(span_text)

        seen: set[tuple[str, int, int]] = set()
        results: list[MultiSpanEntity] = []

        for span_text in sorted(set(mentions), key=len, reverse=True):
            for match in re.finditer(re.escape(span_text), text):
                start, end = match.start(), match.end()
                key = (self._label, start, end)
                if key in seen:
                    continue
                seen.add(key)
                results.append(
                    MultiSpanEntity(
                        type=self._label,
                        text=span_text,
                        spans=[(start, end)],
                    )
                )
                break

        return results

    def parse_directory(
        self,
        output_dir: str,
        reference_filepath: str,
    ) -> list[dict[str, Any]]:
        """Parse a directory of TXT outputs containing JSON lists."""

        def load_text_output(file_path: Path) -> Any:
            return file_path.read_text(encoding="utf8")

        return self._parse_directory(
            output_dir=output_dir,
            reference_filepath=reference_filepath,
            expected_suffix=".txt",
            load_model_output=load_text_output,
        )


__all__ = [
    "BaseParser",
    "GlinerParser",
    "GollieParser",
    "JSONEntParser",
    "JSONMultiEntParser",
    "JSONSingleEntParser",
]
