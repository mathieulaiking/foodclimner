"""GLiNER NER experiment runner.

Expects a dataset directory structured as::

    <data_dir>/
        test.jsonl      # NER-Annotated documents (one JSON per line)
        entities.json   # Entity type definitions

Output layout (see AGENTS.md)::

    <output_dir>/
        raw_preds/          # Individual GLiNER predictions (*.json)
        predictions.jsonl   # Parsed predictions in NER-Annotated format
        config.json         # CLI arguments + model metadata
        stats.json          # Timing and throughput statistics
"""

from __future__ import annotations

import logging
import os
import sys
import time
from argparse import ArgumentParser, Namespace
from dataclasses import asdict

from gliner import GLiNER

if __package__ is None or __package__ == "":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from src.dataset_loader import DatasetLoader
from src.output_writer import JsonSink, OutputWriter
from src.parse import GlinerParser


def parse_args() -> Namespace:
    """Parse and validate command-line arguments."""
    parser = ArgumentParser(description="Run GLiNER NER on a dataset.")

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="GLiNER model name or path (e.g. 'urchade/gliner_small-v2.1').",
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
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold (0-1). Entities below this are discarded.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of documents to process per batch (default: 8).",
    )
    parser.add_argument(
        "--flat_ner",
        action="store_true",
        default=False,
        help="Disable nested NER (flat_ner=True). Default is nested (flat_ner=False).",
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

    if args.threshold < 0.0 or args.threshold > 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")

    return args


def _load_model(args: Namespace) -> GLiNER:
    """Load GLiNER model from HuggingFace Hub or local path.

    When ``--local_files_only`` is set, the model is loaded from the
    HuggingFace cache without internet access (required for Jean-Zay).
    """
    logging.info("Loading model: %s", args.model_name)
    model = GLiNER.from_pretrained(
        args.model_name,
        local_files_only=args.local_files_only,)
    model.eval()
    return model


def main() -> None:
    """Run the GLiNER NER experiment."""
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

    writer = OutputWriter(args.output_dir, sink=JsonSink())
    writer.setup()

    doc_ids = [doc.id for doc in documents]
    skip_ids = writer.skipped_documents(doc_ids)
    if skip_ids:
        documents = [d for d in documents if d.id not in skip_ids]
        logging.info("Skipping inference for %d documents", len(skip_ids))
    if not documents:
        logging.info("All documents already processed — nothing to do.")
        return

    model = _load_model(args)
    parser = GlinerParser(entity_names)

    texts = [doc.text for doc in documents]
    doc_ids = [doc.id for doc in documents]

    logging.info(
        "Running inference (%s NER, threshold=%.2f)",
        "flat" if args.flat_ner else "nested",
        args.threshold,
    )

    total_start = time.time()
    inference_start = time.time()

    all_gliner_outputs: list[list[dict]] = []
    for text in texts:
        entities = model.predict_entities(
            text,
            entity_names,
            threshold=args.threshold,
            flat_ner=args.flat_ner,
        )
        all_gliner_outputs.append(entities)

    inference_time = time.time() - inference_start

    predictions: list[dict] = []
    for doc_id, text, gliner_output in zip(doc_ids, texts, all_gliner_outputs):
        parsed_entities = parser.parse_to_annotated_entities(
            gliner_output,
            text,
            threshold=args.threshold,
        )
        predictions.append(
            {
                "id": doc_id,
                "text": text,
                "entities": [asdict(e) for e in parsed_entities],
            }
        )
        writer.save_raw(doc_id, gliner_output)

    total_time = time.time() - total_start

    total_count = writer.update_predictions(predictions)
    writer.save_stats(
        total_time=total_time,
        num_examples=total_count,
        extra={"inference_time_seconds": round(inference_time, 3)},
    )

    model_info = {
        "model_name": args.model_name,
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
