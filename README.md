# FoodClimNER : A corpus for food practice entities related to climate change

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-blue">
  <img alt="License" src="https://img.shields.io/badge/license-CC_BY_4.0-lightgrey">
</p>

<p align="center">
  🤗 <a href="https://huggingface.co/datasets/inrae-bibliome/foodclimner">Dataset</a> &nbsp;·&nbsp; 
  📄 <a href="">Paper (coming soon!)</a>
</p>

We present FoodClimNER a new corpus for food entities and climate change entities. Our article was published in "The 3rd Workshop of Natural Language Processing meets Climate Change" at EMNLP 2026.

## Installation

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/). We tested the code with python 3.14

```bash
uv sync
source .venv/bin/activate
```

## 1. Corpus and guidelines

Download the dataset using uv and HF CLI

```
mkdir -p data/foodclimner
uvx hf download --type dataset inrae-bibliome/foodclimner --local_dir data/foodclimner
```

Guidelines are available [here](guidelines.pdf).

We also provide scripts to reproduce the preprocessing and conversion of raw brat annotation in [`src/data_processing`](src/data_processing/). `brat.py` for preprocessing the files, `convert.py` for converting brat to the test.jsonl file. We also have the `simplify_types.py` script for our experiments with only the macro types of entities, that maps the detailed types to the macro ones.

## 2. Experiments

### GLiNER

Encoder-based NER supporting flat and nested modes.

```bash
python -m src.experiment.run_gliner \
    --model_name urchade/gliner_large-v2.5 \
    --data_dir data/foodclimner \
    --output_dir out/gliner_large \
    --threshold 0.5 \
    --flat_ner
```

| Flag | Default | Description |
|---|---|---|
| `--model_name` | required | GLiNER model name or path |
| `--threshold` | 0.5 | Confidence threshold |
| `--batch_size` | 8 | Documents per batch |
| `--flat_ner` | False | Disable nested NER |
| `--debug` | False | Run on first 10 examples |
| `--local_files_only` | False | Offline mode (no HuggingFace Hub) |

### GoLLIE

Instruction-tuned LLM with code-like entity guidelines.

```bash
python -m src.experiment.run_gollie \
    --model_name HiTZ/GoLLIE-7B \
    --data_dir data/foodclimner \
    --output_dir out/gollie_7b \
    --max_new_tokens 512
```

| Flag | Default | Description |
|---|---|---|
| `--model_name` | required | GoLLIE model name or path |
| `--max_new_tokens` | 512 | Maximum generation length |
| `--revision` | — | Model revision (e.g. `refs/pr/3`) |
| `--cache_dir` | — | Model cache directory |
| `--debug` | False | Run on first 10 examples |
| `--local_files_only` | False | Offline mode |

### Prompt-based LLM (local)

Structured markdown prompting with system instruction, entity definitions, and output format specification.

```bash
python -m src.experiment.run_prompting \
    --model_name google/gemma-4-31B-it \
    --data_dir data/foodclimner \
    --output_dir out/gemma4-31b \
    --max_new_tokens 4096
```

| Flag | Default | Description |
|---|---|---|
| `--model_name` | required | HuggingFace model name or path |
| `--max_new_tokens` | 4096 | Maximum generation length |
| `--system_prompt` / `--instruction` | — | Override default prompts |
| `--print_prompt` | False | Print prompt template and exit |
| `--no_definitions` / `--no_examples` / `--no_rules` / `--no_persona` | False | Prompt ablations |
| `--debug` | False | Run on first 10 examples |

### Prompt-based LLM (API)

Same prompting approach via the OpenRouter API:

```bash
python -m src.experiment.run_prompting_api \
    --model openai/gpt-5.6-sol \
    --data_dir data/foodclimner \
    --output_dir out/gpt-5.6-sol \
    --max_tokens 4096
```

| Flag | Default | Description |
|---|---|---|
| `--model` | required | OpenRouter model slug |
| `--max_tokens` | 4096 | Maximum generation length |
| `--temperature` | 0.0 | Sampling temperature |
| `--structured_output` | False | Request JSON structured output |
| `--api_key` | `$OPENROUTER_API_KEY` | OpenRouter API key |
| `--debug` | False | Run on first 10 examples |

### Output Structure

All experiment runners produce the same output layout:

```
out/[experiment_name]
├── raw_preds/             # Per-document model outputs (.json or .txt)
├── config.json            # CLI arguments + model metadata
├── predictions.jsonl      # Parsed predictions in NER-Annotated format
├── stats.json             # Timing and throughput statistics
└── prompt_template.txt    # Saved prompt (GoLLIE and prompting runs)
```

## 3. Evaluation

Entity-level evaluation with support for nested and discontinuous entities.

```bash
python -m src.evaluation.evaluate_ner \
    out/experiment/predictions.jsonl \
    data/foodclimner/test.jsonl \
    --mode strict \
    --average both
```

| Mode | Match Criteria |
|---|---|
| `strict` | Exact character boundaries + exact entity type |
| `exact` | Exact character boundaries (type ignored) |
| `partial` | Any span overlap + exact entity type |

### Recursive Mode

Evaluate multiple experiment directories at once:

```bash
python -m src.evaluation.evaluate_ner -r \
    out/ \
    data/ \
    --mode strict
```

This discovers all `<dataset>/predictions.jsonl` files under `out/` and evaluates them against `data/<dataset>/test.jsonl`.

Results are saved as `results.csv` in each prediction directory.
