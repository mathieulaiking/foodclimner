"""Core dataclasses and typed dictionaries for dataset payloads."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional


# ── Evaluation schemas ──────────────────────────────────────────────


@dataclass(frozen=True)
class EvalEntity:
    """One entity mention for strict evaluation.

    Contiguous mention: spans has one ``(start, end)`` pair.
    Discontinuous mention: spans has multiple ``(start, end)`` pairs,
    all belonging to the same logical entity.
    """

    type: str
    spans: tuple[tuple[int, int], ...]


@dataclass
class EvalCounts:
    """TP/FP/FN counts with computed precision, recall, F1."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        denom = p + r
        return 2 * p * r / denom if denom else 0.0


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


# ── BRAT parsing schemas ────────────────────────────────────────────


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


# ── Prompt construction schemas ─────────────────────────────────────


@dataclass
class BasicSection:
    """A generic titled section rendered as a level-3 markdown heading."""

    title: str
    content: str

    def to_markdown(self) -> str:
        """Render this section as markdown."""
        return f"### {self.title}\n\n{self.content}"


@dataclass
class EntitiesSection:
    """A prompt section listing all annotatable entity types."""

    title: str
    entities_infos: list[EntityInfo] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the entities section as markdown."""
        parts = [f"### {self.title}", ""]
        for info in self.entities_infos:
            parts.append(info.to_markdown())
        return "\n".join(parts)


@dataclass
class InputSection:
    """A prompt section whose content contains a placeholder for input text.

    The placeholder ``{input_text}`` is replaced by the actual document
    text at inference time via the ``to_markdown(text=...)`` method.
    """

    title: str
    template: str

    def to_markdown(self, text: str = "{input_text}") -> str:
        """Render this section with *text* substituted for the placeholder."""
        content = self.template.replace("{input_text}", text)
        return f"### {self.title}\n\n{content}"


@dataclass
class NERUserPrompt:
    """The user portion of a NER prompt.

    Attributes
    ----------
    instruction:
        A brief directive telling the model what to do (e.g. "Extract
        all entities of the following types from the text below.").
    entities_section:
        Section listing the entity types to extract.
    rules_section:
        Optional list of rule sections (each with a title and content).
    output_format_section:
        Section describing the expected output format.
    input_text_section:
        Section containing the document text (via placeholder).
    """

    instruction: str
    entities_section: EntitiesSection
    rules_section: list[RuleSection] = field(default_factory=list)
    output_format_section: BasicSection = field(
        default_factory=lambda: BasicSection(
            title="Output Format",
            content="Return a JSON object mapping entity types to lists of mentions.",
        )
    )
    input_text_section: InputSection = field(
        default_factory=lambda: InputSection(
            title="Input Text",
            template="```txt\n{input_text}\n```",
        )
    )

    def to_markdown(self, input_text: str = "{input_text}") -> str:
        """Render the full user prompt as markdown.

        Sections are concatenated in order: instruction, entities,
        rules, output format, input text.
        """
        parts = [
            self.instruction,
            "",
            self.entities_section.to_markdown(),
            "",
        ]
        for rule in self.rules_section:
            parts.append(rule.to_markdown())
            parts.append("")
        parts.append(self.output_format_section.to_markdown())
        parts.append("")
        parts.append(self.input_text_section.to_markdown(text=input_text))
        return "\n".join(parts)


@dataclass
class NERPrompt:
    """Complete NER prompt with a system message and a user prompt.

    Use ``to_messages(input_text=...)`` to obtain the HuggingFace
    chat-template format, or ``to_markdown(input_text=...)`` for a
    plain markdown string.
    """

    system_prompt: str
    user_prompt: NERUserPrompt

    def to_markdown(self, input_text: str = "{input_text}") -> str:
        """Render the full prompt as a single markdown string."""
        return (
            f"{self.system_prompt}\n\n"
            f"{self.user_prompt.to_markdown(input_text=input_text)}"
        )

    def to_messages(self, input_text: str = "{input_text}") -> list[dict[str, str]]:
        """Return the prompt in HuggingFace messages format.

        Returns a list of ``{"role": ..., "content": ...}`` dicts
        suitable for ``tokenizer.apply_chat_template()``.
        """
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": self.user_prompt.to_markdown(input_text=input_text),
            },
        ]


class OutputFormat(abc.ABC):
    """Abstract base for output format definitions.

    Each concrete subclass renders the expected prediction format
    as a ``BasicSection`` that can be placed in the prompt.
    """

    @abc.abstractmethod
    def to_basic_section(self) -> BasicSection:
        """Return a ``BasicSection`` describing the output format."""
        ...


class StandardJsonOutputFormat(OutputFormat):
    """Standard JSON output format: entity-type → list-of-mentions.

    The model is instructed to return a JSON object where keys are
    entity type names and values are lists of mention strings found
    in the input text.
    """

    def to_basic_section(self) -> BasicSection:
        """Return the output format description as a ``BasicSection``."""
        return BasicSection(
            title="Output Format",
            content=(
                "Return a JSON object where each key is an entity type "
                "and the value is a list of all mentions of that entity "
                "type found in the text.\n\n"
                "Example:\n"
                "```json\n"
                '{\n'
                '  "entity_type1" : [\n'
                '    "mention_type1_1",\n'
                '    "mention_type1_2",\n'
                '    "..."\n'
                '  ],\n'
                '  "entity_type2" : [\n'
                '    "mention_type2_1",\n'
                '    "...",\n'
                '  ],\n'
                '  "..."\n'
                "}\n"
                "```\n\n"
                "Return only valid JSON, no additional text."
            ),
        )


__all__ = [
    "BasicSection",
    "BratAnnotation",
    "BratDoc",
    "BratEntity",
    "EntitiesSection",
    "EntityInfo",
    "EvalCounts",
    "EvalEntity",
    "InputSection",
    "MultiSpanEntity",
    "MultiSpanNERDocument",
    "NERCorpus",
    "NERPrompt",
    "NERUserPrompt",
    "OutputFormat",
    "RuleSection",
    "StandardJsonOutputFormat",
    "TokenTagNERDocument",
]
