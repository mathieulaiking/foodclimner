"""LLM NER experiment runner using prompt-based extraction with transformers.

Expects a dataset directory structured as::

    <data_dir>/
        test.jsonl      # NER-Annotated documents (one JSON per line)
        entities.json   # Entity type definitions

Output layout (see AGENTS.md)::

    <output_dir>/
        raw_preds/          # Individual model outputs (*.txt)
        predictions.jsonl   # Parsed predictions in NER-Annotated format
        prompt_template.txt # Prompt template (with {input_text} placeholder)
        config.json         # CLI arguments + model metadata
        stats.json          # Timing and throughput statistics
"""

from __future__ import annotations

import logging
import os
import sys
import time
from argparse import ArgumentParser, Namespace

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.schemas import (
    EntitiesSection,
    EntityInfo,
    InputSection,
    NERPrompt,
    NERUserPrompt,
    RuleSection,
    StandardJsonOutputFormat,
)
from src.dataset_loader import DatasetLoader
from src.output_writer import OutputWriter
from src.parse import JSONMultiEntParser

DEFAULT_INSTRUCTION = "Extract all mentions of the declared entities types from the input text in the specified format."
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant specialized in named entity recognition."
NEUTRAL_SYSTEM_PROMPT = "You are a helpful assistant."


def parse_args() -> Namespace:
    """Parse and validate command-line arguments."""
    parser = ArgumentParser(
        description="Run prompt-based LLM NER on a dataset using AutoModelForCausalLM."
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name or path (e.g. 'mistralai/Mistral-7B-Instruct-v0.3').",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the dataset directory (test.jsonl + entities.json).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to generate per document (default: 4096).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run on the first 10 examples only.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Override the default system prompt.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="Override the default user instruction.",
    )
    parser.add_argument(
        "--print_prompt",
        action="store_true",
        default=False,
        help="Print the prompt template to stdout and exit "
        "(no model loading, no inference).",
    )
    parser.add_argument(
        "--enable_thinking",
        action="store_true",
        default=False,
        help="Enable thinking/reasoning tokens (Qwen 3.5). Disabled by default.",
    )

    # ── Ablation flags ────────────────────────────────────────────
    ablation = parser.add_argument_group(
        "ablation",
        "Remove individual prompt components to study their contribution.",
    )
    ablation.add_argument(
        "--no_definitions",
        action="store_true",
        default=False,
        help="Strip entity type definitions from the prompt.",
    )
    ablation.add_argument(
        "--no_examples",
        action="store_true",
        default=False,
        help="Strip entity type examples and counter-examples from the prompt.",
    )
    ablation.add_argument(
        "--no_rules",
        action="store_true",
        default=False,
        help="Omit dataset-specific rules from the prompt.",
    )
    ablation.add_argument(
        "--no_persona",
        action="store_true",
        default=False,
        help="Use a neutral system prompt instead of the NER-specific persona.",
    )

    args = parser.parse_args()

    if args.print_prompt:
        if args.model_name or args.output_dir:
            pass  # silently ignored in print_prompt mode
    else:
        if not args.model_name:
            parser.error("--model_name is required (unless --print_prompt is set)")
        if not args.output_dir:
            parser.error("--output_dir is required (unless --print_prompt is set)")

    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be >= 1")

    return args


def _load_model(args: Namespace) -> tuple[AutoModelForCausalLM, AutoProcessor]:
    """Load model and processor from HuggingFace Hub or local path."""
    logging.info("Loading model and processor: %s", args.model_name)

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype="auto",
        device_map="auto",
    )
    model.eval()

    return model, processor


def _strip_entity_infos(
    entity_infos: list[EntityInfo],
    no_definitions: bool = False,
    no_examples: bool = False,
) -> list[EntityInfo]:
    """Return new ``EntityInfo`` copies with requested fields cleared.

    Parameters
    ----------
    entity_infos:
        Original entity type definitions.
    no_definitions:
        If ``True``, remove the ``definition`` field from each entity.
    no_examples:
        If ``True``, remove ``examples`` and ``counter_examples``.

    Returns
    -------
    A new list of ``EntityInfo`` instances with the specified fields
    set to ``None``.  Original objects are not mutated.
    """
    if not no_definitions and not no_examples:
        return entity_infos
    stripped: list[EntityInfo] = []
    for info in entity_infos:
        stripped.append(
            EntityInfo(
                type=info.type,
                definition=None if no_definitions else info.definition,
                examples=None if no_examples else info.examples,
                counter_examples=None if no_examples else info.counter_examples,
            )
        )
    return stripped


def _build_prompt(
    entity_infos: list[EntityInfo],
    rules_sections: list[RuleSection],
    system_prompt: str | None,
    instruction: str | None,
    no_definitions: bool = False,
    no_examples: bool = False,
    no_rules: bool = False,
    no_persona: bool = False,
) -> NERPrompt:
    """Construct a reusable ``NERPrompt`` from entity definitions and optional rules.

    Parameters
    ----------
    entity_infos:
        List of ``EntityInfo`` dataclass instances as returned by ``load_entity_infos()``.
    rules_sections:
        List of ``RuleSection`` dataclass instances as returned by ``load_rules()``.
        Pass an empty list when no rules are available.
    system_prompt:
        Override for the default system prompt, or ``None``.
    instruction:
        Override for the default user instruction, or ``None``.
    no_definitions:
        If ``True``, strip entity definitions from the prompt.
    no_examples:
        If ``True``, strip entity examples and counter-examples.
    no_rules:
        If ``True``, omit dataset-specific rules.
    no_persona:
        If ``True``, use a neutral system prompt.

    Returns
    -------
    A fully configured ``NERPrompt`` instance (input text placeholder
    is filled per document at inference time).
    """
    output_format = StandardJsonOutputFormat()

    stripped_infos = _strip_entity_infos(
        entity_infos,
        no_definitions=no_definitions,
        no_examples=no_examples,
    )

    user_prompt = NERUserPrompt(
        instruction=instruction or DEFAULT_INSTRUCTION,
        entities_section=EntitiesSection(
            title="Entity Types",
            entities_infos=stripped_infos,
        ),
        rules_section=[] if no_rules else rules_sections,
        output_format_section=output_format.to_basic_section(),
        input_text_section=InputSection(
            title="Input Text",
            template="```txt\n{input_text}\n```",
        ),
    )

    resolved_system = system_prompt or DEFAULT_SYSTEM_PROMPT
    if no_persona and not system_prompt:
        resolved_system = NEUTRAL_SYSTEM_PROMPT

    return NERPrompt(
        system_prompt=resolved_system,
        user_prompt=user_prompt,
    )


def _format_messages(messages: list[dict[str, str]]) -> str:
    """Render HuggingFace messages as a human-readable role/content display.

    Each message is separated by a divider with its index, role, and content.
    """
    parts: list[str] = []
    for i, msg in enumerate(messages):
        parts.append(f"{'─' * 22} message {i} {'─' * 22}")
        parts.append(f"role: {msg['role']}")
        parts.append("")
        parts.append(msg["content"])
        parts.append("")
    return "\n".join(parts)


def main() -> None:
    """Run the prompt-based LLM NER experiment."""
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    loader = DatasetLoader(args.data_dir)
    dataset = loader.load_corpus(lazy_entities=True)
    entity_infos = dataset.entities_info
    entity_names = dataset.entity_names
    documents = dataset.documents

    if args.debug:
        logging.info("Debug mode: running on first 10 examples")
        documents = documents[:10]

    logging.info(
        "Loaded %d documents, %d entity types",
        len(documents),
        len(entity_names),
    )

    rules_sections = loader.load_rules()
    if rules_sections:
        logging.info("Loaded %d rule sections", len(rules_sections))

    prompt = _build_prompt(
        entity_infos,
        rules_sections,
        system_prompt=args.system_prompt,
        instruction=args.instruction,
        no_definitions=args.no_definitions,
        no_examples=args.no_examples,
        no_rules=args.no_rules,
        no_persona=args.no_persona,
    )

    if args.print_prompt:
        print(_format_messages(prompt.to_messages()))
        sys.exit(0)

    writer = OutputWriter(args.output_dir)
    writer.setup()

    doc_ids = [doc.id for doc in documents]
    skip_ids = writer.skipped_documents(doc_ids)
    if skip_ids:
        documents = [d for d in documents if d.id not in skip_ids]
        logging.info("Skipping inference for %d documents", len(skip_ids))
    if not documents:
        logging.info("All documents already processed — nothing to do.")
        return

    model, processor = _load_model(args)
    parser = JSONMultiEntParser(entity_names)

    # Save the prompt template (with placeholder) for reproducibility
    template_md = prompt.to_markdown()
    writer.save_prompt_template(template_md)

    logging.info(
        "Running inference on %d documents (max_new_tokens=%d)",
        len(documents),
        args.max_new_tokens,
    )

    predictions: list[dict] = []
    total_start = time.time()

    for doc in documents:
        messages = prompt.to_messages(input_text=doc.text)
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
        model_inputs = processor(text=text, return_tensors="pt").to(model.device)
        input_len = model_inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
            )
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

        parsed_entities = parser.parse_to_annotated_entities(response, doc.text)
        predictions.append(
            {
                "id": doc.id,
                "text": doc.text,
                "entities": [
                    {"type": e.type, "text": e.text, "offsets": [list(s) for s in e.spans]}
                    for e in parsed_entities
                ],
            }
        )

        writer.save_raw(doc.id, response)

    total_time = time.time() - total_start

    total_count = writer.update_predictions(predictions)

    writer.save_stats(
        total_time=total_time,
        num_examples=total_count,
    )

    model_info = {
        "model_name": args.model_name,
        "model_class": type(model).__name__,
    }
    writer.save_config(args, model_info=model_info)

    logging.info(
        "Done — %d documents in %.2fs (%.2f examples/s)",
        len(predictions),
        total_time,
        len(predictions) / total_time if total_time > 0 else 0.0,
    )


if __name__ == "__main__":
    main()
