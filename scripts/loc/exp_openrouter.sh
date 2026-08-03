#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Load .env (for OPENROUTER_API_KEY)
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    source "${REPO_ROOT}/.env"
    set +a
fi

DEBUG="0"
for arg in "$@"; do
    case "${arg}" in
        --debug)
            DEBUG="1"
            ;;
        *)
            echo "Unknown argument: ${arg}" >&2
            echo "Usage: $0 [--debug]" >&2
            exit 1
            ;;
    esac
done

# -- Argument lists (edit these) --------------------------------------------

models=(
    "openai/gpt-5.6-sol"
    # "z-ai/glm-5.2"
    # "anthropic/claude-opus-4.8"
    # "openai/gpt-4o"
    # "anthropic/claude-sonnet-4-20250514"
)

datasets=(
    "${REPO_ROOT}/data/foodclimner-v1"
    "${REPO_ROOT}/data/foodclimner-v1-simple"
)

# -- Runtime settings --------------------------------------------------------
MAX_TOKENS="4096"
TEMPERATURE="0.0"
STRUCTURED_OUTPUT="--structured_output"  # set to "" to disable

# -- Run one model x dataset combination at a time ---------------------------
for model in "${models[@]}"; do
    model_short="${model##*/}"
    for dataset in "${datasets[@]}"; do
        dataset_short="${dataset##*/}"
        output_dir="${REPO_ROOT}/out/experiments/proprietary/${model_short}/${dataset_short}"

        echo ""
        echo "========================================================================"
        echo "Model:    ${model}"
        echo "Dataset:  ${dataset_short}"
        echo "Output:   ${output_dir}"
        echo "========================================================================"

        DEBUG_FLAG=""
        if [[ "${DEBUG}" == "1" ]]; then
            DEBUG_FLAG="--debug"
        fi

        uv run python src/experiment/run_prompting_api.py \
            --model "${model}" \
            --data_dir "${dataset}" \
            --output_dir "${output_dir}" \
            --max_tokens "${MAX_TOKENS}" \
            --temperature "${TEMPERATURE}" \
            ${STRUCTURED_OUTPUT} \
            ${DEBUG_FLAG}

        echo "Done: ${model} / ${dataset_short}"

        if [[ "${DEBUG}" == "1" ]]; then
            echo ""
            echo "Debug mode: ran one model only."
            exit 0
        fi
    done
done
