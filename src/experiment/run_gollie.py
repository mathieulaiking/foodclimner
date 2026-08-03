"""GoLLIE NER experiment runner.

Expects a dataset directory structured as::

    <data_dir>/
        test.jsonl      # NER-Annotated documents (one JSON per line)
        entities.json   # Entity type definitions

Output layout (see AGENTS.md)::

    <output_dir>/
        raw_preds/          # Individual GoLLIE outputs (*.txt)
        predictions.jsonl   # Parsed predictions in NER-Annotated format
        config.json         # CLI arguments + model metadata
        stats.json          # Timing and throughput statistics
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from argparse import ArgumentParser, Namespace
from dataclasses import asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.dataset_loader import DatasetLoader
from src.output_writer import OutputWriter
from src.parse import GollieParser
from src.schemas import EntityInfo


def parse_args() -> Namespace:
    """Parse and validate command-line arguments."""
    parser = ArgumentParser(description="Run GoLLIE NER on a dataset.")

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="GoLLIE model name or path (e.g. 'HiTZ/GoLLIE-7B').",
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
        "--revision",
        type=str,
        default=None,
        help="Model revision (e.g. 'refs/pr/3' for safetensors).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache the model and tokenizer.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate per document (default: 512).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Run on the first 10 examples only.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        default=False,
        help="Do not connect to HuggingFace Hub.",
    )
    parser.add_argument(
        "--disable_logging",
        action="store_true",
        default=False,
        help="Disable logging output.",
    )
    args = parser.parse_args()

    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens must be >= 1")

    return args


SINGLE_ENTITY_TEMPLATE = '''@dataclass
class [name]:
    """[description]"""

    span: str [examples]
'''

GUIDELINES_TEMPLATE = '''from dataclasses import dataclass

# The following lines describe the task definition
[entities_python_classes]

ENTITY_DEFINITIONS = [
    [entity_names]
 ]

# This is the text to analyze
text = [input_text]

# The annotation instances that take place in the text above are listed here
result = ['''


def _multi_line_desc(description: str, max_char_nb: int = 79) -> str:
    """Format the description into multiple lines with a max number of chars per line."""
    words = description.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= max_char_nb:
            current_line += " " + word if current_line else word
        else:
            lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)
    return "\n    ".join(lines)


def _to_camel_case(split_str: str) -> str:
    """Convert a string to CamelCase."""
    if re.match(r'^[A-Z][a-zA-Z0-9]*$', split_str):
        return split_str
    split_str = re.split(r'[-_\s]+', split_str)
    return ''.join(word.capitalize() for word in split_str)


def create_guidelines(entities: list[EntityInfo]) -> str:
    """Generate guideline classes from entity type definitions."""
    entity_classes = []
    entity_names = []

    for entity in entities:
        name = _to_camel_case(entity.type)
        description = _multi_line_desc(
            entity.definition or ""
        )
        examples_list = entity.examples or []
        examples = "# Such as : " + ", ".join(examples_list) if examples_list else ""
        entity_class = (
            SINGLE_ENTITY_TEMPLATE
            .replace("[name]", name)
            .replace("[description]", description)
            .replace("[examples]", examples)
        )
        entity_classes.append(entity_class)
        entity_names.append(name)

    guidelines = GUIDELINES_TEMPLATE.replace(
        "[entities_python_classes]", "\n\n".join(entity_classes)
    ).replace("[entity_names]", ",\n    ".join(entity_names))
    return guidelines


def main() -> None:
    """Run the GoLLIE NER experiment."""
    args = parse_args()

    if args.disable_logging:
        logging.disable(logging.CRITICAL)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
        )

    dataset = DatasetLoader(args.data_dir).load_corpus()
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

    logging.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.revision,
        use_fast=False,
        local_files_only=args.local_files_only,
        cache_dir=args.cache_dir,
    )

    logging.info("Loading model: %s", args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.revision,
        device_map="auto",
        dtype=torch.float16,
        local_files_only=args.local_files_only,
        use_safetensors=True,
        cache_dir=args.cache_dir,
    )
    model.eval()

    parser = GollieParser(entity_names)

    guidelines = create_guidelines(dataset.entities_info)
    writer = OutputWriter(args.output_dir)
    writer.setup()
    writer.save_prompt_template(guidelines)

    doc_ids = [doc.id for doc in documents]
    skip_ids = writer.skipped_documents(doc_ids)
    if skip_ids:
        documents = [d for d in documents if d.id not in skip_ids]
        logging.info("Skipping inference for %d documents", len(skip_ids))
    if not documents:
        logging.info("All documents already processed — nothing to do.")
        return

    logging.info(
        "Running inference (greedy decoding, max_new_tokens=%d)",
        args.max_new_tokens,
    )

    total_start = time.time()

    predictions: list[dict] = []
    for doc in documents:
        doc_id = doc.id
        text = doc.text

        prompt = guidelines.replace("[input_text]", repr(text))
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        model_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

        parsed_entities = parser.parse_to_annotated_entities(model_output, text)
        predictions.append(
            {
                "id": doc_id,
                "text": text,
                "entities": [asdict(e) for e in parsed_entities],
            }
        )

        writer.save_raw(doc_id, model_output)

    total_time = time.time() - total_start

    total_count = writer.update_predictions(predictions)

    writer.save_stats(
        total_time=total_time,
        num_examples=total_count,
    )

    model_info = {
        "model_name": args.model_name,
        "revision": args.revision,
        "model_class": type(model).__name__,
    }
    writer.save_config(args, model_info=model_info)

    logging.info(
        "Done — %d documents in %.2fs (%.2f examples/s)",
        total_count,
        total_time,
        total_count / total_time if total_time > 0 else 0.0,
    )


if __name__ == "__main__":
    main()
