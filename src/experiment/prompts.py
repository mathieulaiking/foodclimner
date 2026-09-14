"""NER prompt schemas owned by the experiment module.

Each section implements :class:`Renderable` so prompt parts carry their
own rendering behavior instead of living in a central schema bag.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from src.data_processing.loader import EntityInfo, RuleSection


class Renderable(abc.ABC):
    """Abstract base for prompt parts renderable as markdown."""

    @abc.abstractmethod
    def to_markdown(self) -> str:
        """Render this part as markdown."""
        ...


class OutputFormat(abc.ABC):
    """Abstract base for output format definitions.

    Each concrete subclass renders the expected prediction format
    as a ``BasicSection`` that can be placed in the prompt.
    """

    @abc.abstractmethod
    def to_basic_section(self) -> BasicSection:
        """Return a ``BasicSection`` describing the output format."""
        ...


@dataclass
class BasicSection(Renderable):
    """A generic titled section rendered as a level-3 markdown heading."""

    title: str
    content: str

    def to_markdown(self) -> str:
        """Render this section as markdown."""
        return f"### {self.title}\n\n{self.content}"


@dataclass
class EntitiesSection(Renderable):
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
class InputSection(Renderable):
    """A prompt section whose content contains a placeholder for input text.

    The placeholder ``{input_text}`` is replaced by the actual document
    text at inference time via the ``to_markdown(text=...)`` method.
    """

    title: str
    template: str

    def to_markdown(self, text: str = "{input_text}") -> str:  # type: ignore[override]
        """Render this section with *text* substituted for the placeholder."""
        content = self.template.replace("{input_text}", text)
        return f"### {self.title}\n\n{content}"


@dataclass
class NERUserPrompt(Renderable):
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

    def to_markdown(self, input_text: str = "{input_text}") -> str:  # type: ignore[override]
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
class NERPrompt(Renderable):
    """Complete NER prompt with a system message and a user prompt.

    Use ``to_messages(input_text=...)`` to obtain the HuggingFace
    chat-template format, or ``to_markdown(input_text=...)`` for a
    plain markdown string.
    """

    system_prompt: str
    user_prompt: NERUserPrompt

    def to_markdown(self, input_text: str = "{input_text}") -> str:  # type: ignore[override]
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
    "EntitiesSection",
    "InputSection",
    "NERPrompt",
    "NERUserPrompt",
    "OutputFormat",
    "Renderable",
    "StandardJsonOutputFormat",
]
