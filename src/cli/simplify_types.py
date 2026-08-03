from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

TYPE_MAP: dict[str, str] = {
    "food_repertoire": "food_practice",
    "food_logistics_travel": "food_practice",
    "food_supply_purchases": "food_practice",
    "food_storage": "food_practice",
    "food_culinary_preparation": "food_practice",
    "food_consumption_uses": "food_practice",
    "food_waste_management": "food_practice",
    "climate_greenhouse_gases": "climate_change",
    "climate_hazards": "climate_change",
    "climate_mitigations": "climate_change",
    "climate_problem_origins": "climate_change",
    "climate_change": "climate_change",
    "environmental_concern": "climate_change",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simplify entity types in a dataset to two classes: FoodPractices and ClimateRelated.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/sf4cd-v0.4"),
        help="Path to the input dataset directory (default: data/sf4cd-v0.4).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sf4cd-v0.4-simple"),
        help="Path to the output directory (default: data/sf4cd-v0.4-simple).",
    )
    return parser.parse_args(argv)


def simplify_entities_json(input_path: Path, output_path: Path) -> None:
    simplified = [
        {"name": "food_practice", "definition": "", "examples": [], "counter_examples": []},
        {"name": "climate_change", "definition": "", "examples": [], "counter_examples": []},
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(simplified, f, indent=2, ensure_ascii=False)
        f.write("\n")


def simplify_test_jsonl(input_path: Path, output_path: Path) -> None:
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            for entity in doc["entities"]:
                original_type = entity["type"]
                entity["type"] = TYPE_MAP.get(original_type, original_type)
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    if not args.data_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    src_entities = args.data_dir / "entities.json"
    if src_entities.is_file():
        simplify_entities_json(src_entities, args.output_dir / "entities.json")

    src_test = args.data_dir / "test.jsonl"
    if src_test.is_file():
        simplify_test_jsonl(src_test, args.output_dir / "test.jsonl")

    src_rules = args.data_dir / "rules.json"
    if src_rules.is_file():
        shutil.copy2(src_rules, args.output_dir / "rules.json")


if __name__ == "__main__":
    main()
