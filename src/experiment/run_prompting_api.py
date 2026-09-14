"""LLM NER experiment runner using prompt-based extraction via OpenRouter API.

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
        stats.json          # Timing, throughput, and token usage statistics
"""

from __future__ import annotations

import logging
import os
import sys
import time
from argparse import ArgumentParser, Namespace
from dataclasses import asdict

import httpx

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.data_processing.loader import DatasetLoader, EntityInfo, RuleSection
from src.experiment.model_parsing import JSONMultiEntParser
from src.experiment.output_writer import OutputWriter
from src.experiment.prompts import (
    EntitiesSection,
    InputSection,
    NERPrompt,
    NERUserPrompt,
    StandardJsonOutputFormat,
)

DEFAULT_INSTRUCTION = "Extract all mentions of the declared entities types from the input text in the specified format."
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant specialized in named entity recognition."
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0


def parse_args() -> Namespace:
    """Parse and validate command-line arguments."""
    parser = ArgumentParser(
        description="Run prompt-based LLM NER on a dataset via OpenRouter API."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="OpenRouter model slug (e.g. 'openai/gpt-4o').",
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
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum number of tokens to generate per document (default: 4096).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy, default: 0.0).",
    )
    parser.add_argument(
        "--structured_output",
        action="store_true",
        default=False,
        help="Request JSON structured output via response_format (model must support it).",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenRouter API key. Falls back to OPENROUTER_API_KEY environment variable.",
    )
    parser.add_argument(
        "--site_url",
        type=str,
        default=None,
        help="Site URL sent as HTTP-Referer header for OpenRouter rankings.",
    )
    parser.add_argument(
        "--site_name",
        type=str,
        default=None,
        help="App name sent as X-OpenRouter-Title header.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run on the first 10 examples only.",
    )
    parser.add_argument(
        "--disable_logging",
        action="store_true",
        default=False,
        help="Disable logging output.",
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
        "--disable_reasoning",
        action="store_true",
        default=False,
        help="Disable model thinking/reasoning (sends reasoning: {effort: none} if the model supports it).",
    )

    args = parser.parse_args()

    if args.max_tokens < 1:
        raise ValueError("--max_tokens must be >= 1")
    if not 0.0 <= args.temperature <= 2.0:
        raise ValueError("--temperature must be in [0.0, 2.0]")

    if args.api_key is None:
        args.api_key = os.environ.get("OPENROUTER_API_KEY")
    if not args.api_key:
        raise ValueError(
            "No API key provided. Pass --api_key or set the OPENROUTER_API_KEY "
            "environment variable."
        )

    return args


def _build_prompt(
    entity_infos: list[EntityInfo],
    rules_sections: list[RuleSection],
    system_prompt: str | None,
    instruction: str | None,
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

    Returns
    -------
    A fully configured ``NERPrompt`` instance (input text placeholder
    is filled per document at inference time).
    """
    output_format = StandardJsonOutputFormat()

    user_prompt = NERUserPrompt(
        instruction=instruction or DEFAULT_INSTRUCTION,
        entities_section=EntitiesSection(
            title="Entity Types",
            entities_infos=entity_infos,
        ),
        rules_section=rules_sections,
        output_format_section=output_format.to_basic_section(),
        input_text_section=InputSection(
            title="Input Text",
            template="```txt\n{input_text}\n```",
        ),
    )

    return NERPrompt(
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )


def _call_api(
    client: httpx.Client,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    structured_output: bool,
    disable_reasoning: bool = False,
) -> tuple[str, dict[str, int]]:
    """Call the OpenRouter chat completions endpoint with retry logic.

    Parameters
    ----------
    client:
        An ``httpx.Client`` instance (reused across calls).
    model:
        OpenRouter model slug.
    messages:
        List of ``{"role": ..., "content": ...}`` dicts.
    max_tokens:
        Maximum tokens in the response.
    temperature:
        Sampling temperature.
    structured_output:
        If ``True``, pass ``response_format: {"type": "json_object"}``.
    disable_reasoning:
        If ``True``, send ``reasoning: {effort: none}`` to disable thinking.

    Returns
    -------
    ``(response_text, usage)`` where *usage* has ``prompt_tokens``,
    ``completion_tokens``, and ``total_tokens`` keys.

    Raises
    ------
    httpx.HTTPStatusError
        If the request fails after all retries.
    """
    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if structured_output:
        body["response_format"] = {"type": "json_object"}
    if disable_reasoning:
        body["reasoning"] = {"effort": "none"}

    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                json=body,
            )
            response.raise_for_status()
            data = response.json()
            message = data["choices"][0]["message"]
            content = message.get("content")
            if content is None:
                content = message.get("reasoning", "")
                if not content:
                    content = ""
                logging.warning(
                    "Model returned content=None; fell back to reasoning field (%d chars)",
                    len(content),
                )
            usage = data.get("usage", {})
            return content, {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
        except httpx.TimeoutException as exc:
            last_exc = exc
            logging.warning(
                "Request timed out (attempt %d/%d)", attempt + 1, MAX_RETRIES
            )
        except httpx.HTTPStatusError as exc:
            last_exc = exc
            status = exc.response.status_code
            if status in (429, 502, 503):
                logging.warning(
                    "Received HTTP %d (attempt %d/%d)",
                    status,
                    attempt + 1,
                    MAX_RETRIES,
                )
            else:
                raise

        if attempt < MAX_RETRIES - 1:
            backoff = INITIAL_BACKOFF * (2 ** attempt)
            logging.info("Retrying in %.1fs ...", backoff)
            time.sleep(backoff)

    raise RuntimeError(
        f"OpenRouter API call failed after {MAX_RETRIES} retries"
    ) from last_exc


def main() -> None:
    """Run the prompt-based LLM NER experiment via OpenRouter."""
    args = parse_args()

    if args.disable_logging:
        logging.disable(logging.CRITICAL)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )

    loader = DatasetLoader(args.data_dir)
    dataset = loader.load_corpus()
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
    )
    parser = JSONMultiEntParser(entity_names)
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

    template_md = prompt.to_markdown()
    writer.save_prompt_template(template_md)

    headers = {
        "Authorization": f"Bearer {args.api_key}",
    }
    if args.site_url:
        headers["HTTP-Referer"] = args.site_url
    if args.site_name:
        headers["X-OpenRouter-Title"] = args.site_name

    logging.info(
        "Running inference (model=%s, max_tokens=%d, temperature=%.2f, "
        "structured_output=%s)",
        args.model,
        args.max_tokens,
        args.temperature,
        args.structured_output,
    )

    total_start = time.time()
    inference_start = time.time()

    total_usage: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    predictions: list[dict] = []

    with httpx.Client(headers=headers, timeout=120.0) as client:
        for doc in documents:
            doc_id = doc.id
            text = doc.text

            messages = prompt.to_messages(input_text=text)

            model_output, usage = _call_api(
                client,
                args.model,
                messages,
                args.max_tokens,
                args.temperature,
                args.structured_output,
                disable_reasoning=args.disable_reasoning,
            )

            for key in total_usage:
                total_usage[key] += usage.get(key, 0)

            parsed_entities = parser.parse_to_annotated_entities(model_output, text)
            predictions.append(
                {
                    "id": doc_id,
                    "text": text,
                    "entities": [asdict(e) for e in parsed_entities],
                }
            )
            writer.save_raw(doc_id, model_output)

    inference_time = time.time() - inference_start

    total_count = writer.update_predictions(predictions)
    writer.save_stats(
        total_time=time.time() - total_start,
        num_examples=total_count,
        extra={
            "inference_time_seconds": round(inference_time, 3),
            **total_usage,
        },
    )

    model_info = {
        "model_name": args.model,
        "structured_output": args.structured_output,
    }
    writer.save_config(args, model_info=model_info)

    total_time = time.time() - total_start
    logging.info(
        "Done — %d documents in %.2fs (%.2f examples/s) | "
        "tokens: %d prompt / %d completion / %d total",
        total_count,
        total_time,
        total_count / total_time if total_time > 0 else 0.0,
        total_usage["prompt_tokens"],
        total_usage["completion_tokens"],
        total_usage["total_tokens"],
    )


if __name__ == "__main__":
    main()
