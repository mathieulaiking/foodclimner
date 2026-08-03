# FoodClimNER : A corpus for food practice entities related to climate change

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-blue">
  <img alt="License" src="https://img.shields.io/badge/license-CC_BY--NC--SA_4.0-lightgrey">
</p>

<p align="center">
  <a href="#quick-start">⚡ Quick Start</a> |
  <a href="#corpus">📚 Corpus</a> |
  <a href="#reproducing-experiments">🧪 Experiments</a> |
  <a href="#evaluation">📊 Evaluation</a> |
  <a href="#cli-tools">🔧 CLI Tools</a> |
  <a href="#citation">📖 Citation</a>
</p>

A manually annotated corpus of scientific articles on food practices related to climate change, with baseline NER experiments using GLiNER, GoLLIE, and prompt-based LLMs.

This repository accompanies the paper accepted at **ClimatNLP 2026** (EMNLP 2026 workshop). It provides:

- **FoodClimNER** — a corpus of 10 full text open scientific abstracts annotated with 12 entity types capturing food practices and climate concepts, supporting both **nested** and **discontinuous** entity annotations
- **Three NER baselines** — GLiNER (encoder), GoLLIE (instruction-tuned LLM), and a prompt-based LLM pipeline
- **Entity-level evaluation** — supporting strict, exact, and partial matching modes

---

## Quick Start

```bash
# Install the package
uv pip install -e .

# Run tests
uv run pytest

# Lint
uv run ruff check
```

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/). See `.python-version` (3.14).

---

## Corpus

### Entity Types

| Entity | Description | Examples |
|---|---|---|
| `food_repertoire` | Food choices integrating diet criteria | "increased consumption of plant-based foods", "reduction in meat quantity" |
| `food_logistics_travel` | Pre-purchase practices and travel to supply | "write a shopping list", "plan meals in advance" |
| `food_supply_purchases` | Purchasing practices and non-market supply | "self-production through gardening", "foraging", "barter" |
| `food_storage` | Storage after purchase, before preparation | "canning", "freezing", "smart fridges" |
| `food_culinary_preparation` | Cooking and preparation practices | "raw foodism", "energy-saving cooking equipment" |
| `food_consumption_uses` | Eating and meal-time practices | "reducing plate size", "portion control" |
| `food_waste_management` | Post-meal waste practices | "keeping leftovers", "composting" |
| `climate_greenhouse_gases` | Greenhouse gas mentions | "carbon dioxide", "CH4" |
| `climate_hazards` | Climate-related hazards | "floods", "wildfires", "droughts", "heatwaves" |
| `climate_mitigations` | Climate change mitigation activities | — |
| `climate_problem_origins` | Causes of climate change | "fossil fuel", "deforestation", "transport sector" |
| `environmental_concern` | Broader environmental concepts | "biodiversity", "ecological footprint" |

### Annotation Format

The corpus is tokenization-independent: entities are stored as character offsets supporting **nested** (overlapping spans of different types) and **discontinuous** (multiple non-consecutive spans for one entity) mentions.

```json
{
  "id": "document_001",
  "text": "Increased consumption of plant-based foods...",
  "entities": [
    {"type": "food_repertoire", "text": "Increased consumption", "offsets": [[0, 21]]},
    {"type": "food_repertoire", "text": "plant-based foods", "offsets": [[37, 53]]}
  ]
}
```

Source annotations (BRAT standoff format) are in `data/brat-sf4cd-annots/`. The ready-to-use dataset is in `data/sf4cd-v0.4/`.

### Dataset Structure

```
data/sf4cd-v0.4/
├── test.jsonl       # Documents with entity annotations (JSONL)
├── entities.json    # Entity type definitions
└── rules.json       # Annotation rules
```

---

## Reproducing Experiments

All experiment runners accept a `--debug` flag to run on 10 examples for quick validation, and `--local_files_only` for offline execution on clusters without internet access.

### GLiNER

Encoder-based NER supporting flat and nested modes.

```bash
python -m src.experiment.run_gliner \
    --model_name urchade/gliner_small-v2.1 \
    --data_dir data/sf4cd-v0.4 \
    --output_dir out/gliner_small \
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
    --data_dir data/sf4cd-v0.4 \
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

### Prompt-based LLM

Structured markdown prompting with system instruction, entity definitions, and output format specification.

```bash
python -m src.experiment.run_prompting \
    --model_name mistralai/Mistral-7B-Instruct-v0.3 \
    --data_dir data/sf4cd-v0.4 \
    --output_dir out/mistral7b_prompt \
    --max_new_tokens 512
```

| Flag | Default | Description |
|---|---|---|
| `--model_name` | required | HuggingFace model name or path |
| `--max_new_tokens` | 512 | Maximum generation length |
| `--system_prompt` | — | Override default system prompt |
| `--instruction` | — | Override default user instruction |
| `--debug` | False | Run on first 10 examples |
| `--local_files_only` | False | Offline mode |

### Output Structure

All experiment runners produce the same output layout:

```
out/experiment_name/
├── raw_preds/             # Per-document model outputs (.json or .txt)
├── predictions.jsonl      # Parsed predictions in NER-Annotated format
├── config.json            # CLI arguments + model metadata
├── stats.json             # Timing and throughput statistics
└── prompt_template.txt    # Saved prompt (GoLLIE and prompting runs)
```

---

## Evaluation

Entity-level evaluation with support for nested and discontinuous entities.

```bash
python -m src.cli.eval_ner \
    out/experiment/predictions.jsonl \
    data/sf4cd-v0.4/test.jsonl \
    --mode strict \
    --average both
```

### Evaluation Modes

| Mode | Match Criteria |
|---|---|
| `strict` | Exact character boundaries + exact entity type |
| `exact` | Exact character boundaries (type ignored) |
| `partial` | Any span overlap + exact entity type |

### Recursive Mode

Evaluate multiple experiment directories at once:

```bash
python -m src.cli.eval_ner -r \
    out/ \
    data/ \
    --mode strict
```

This discovers all `<dataset>/predictions.jsonl` files under `out/` and evaluates them against `data/<dataset>/test.jsonl`.

Results are saved as `results_<mode>.csv` in each prediction directory.

---

## CLI Tools

### BRAT Preprocessing

Convert raw BRAT annotation files through a three-step pipeline: combine extracts, clean references, split sentences.

```bash
python -m src.cli.brat_preprocessing \
    --source data/brat-sf4cd-annots/raw-v0.4 \
    --target data/brat-preprocessed \
    --debug
```

### Format Conversion

Convert between annotation formats (BRAT ↔ JSONL).

```bash
python -m src.cli.convert \
    data/brat-sf4cd-annots/raw-v0.4 \
    data/sf4cd-v0.5
```

---

## Cluster Jobs (Jean-Zay)

SLURM scripts for the Jean-Zay cluster are in `scripts/jz/`. They support GPU types H100, A100, and V100 with appropriate partitions and QoS settings.

```bash
# Edit the model and dataset lists in the script, then:
bash scripts/jz/exp_gliner.sh
bash scripts/jz/exp_gollie.sh
bash scripts/jz/exp_prompting.sh
```

The cluster requires `--local_files_only` (no internet access). Set `$SLURM_ACCOUNT` for your project.

---

## Visualization

```bash
# LaTeX performance table
python -m src.visualization.latex_perf_table out/ --mode strict

# HTML error visualization (highlights correct/wrong predictions)
python -m src.visualization.html_entitiesmentions_prederrorhl \
    out/experiment/predictions.jsonl \
    data/sf4cd-v0.4/test.jsonl
```

---

## Project Structure

```
src/
├── cli/
│   ├── brat_preprocessing.py   # BRAT preprocessing pipeline
│   └── convert.py               # Format conversion
├── experiment/
│   ├── run_gliner.py            # GLiNER experiment runner
│   ├── run_gollie.py            # GoLLIE experiment runner
│   └── run_prompting.py         # Prompt-based LLM experiment runner
├── process/
│   ├── brat_utils.py            # BRAT file utilities
│   └── parse.py                 # Model output parsers
├── visualization/
│   ├── html_entitiesmentions_prederrorhl.py
│   ├── latex_perf_table.py
│   └── latex_prompt_boxfigure.py
├── convert.py                   # Core conversion logic
├── eval_ner.py                  # Entity-level NER evaluation
├── schemas.py                   # Core dataclasses
└── utils.py                     # Shared experiment utilities
```

---

## Citation

If you use this corpus or code, please cite:

```bibtex
@inproceedings{sf4cd2026,
  title     = {SafeFood4ClimDiet: A Corpus for Food Practices Related to Climate Change},
  author    = {},
  booktitle = {Proceedings of the Workshop on Climate Change and NLP (ClimatNLP 2026)},
  year      = {2026},
  address   = {Stroudsburg, PA, USA},
  publisher = {Association for Computational Linguistics}
}
```

---

## License

The corpus annotations are distributed under CC BY-NC-SA 4.0. The code is available under the MIT license. See the paper for further details on data sources and annotation methodology.
