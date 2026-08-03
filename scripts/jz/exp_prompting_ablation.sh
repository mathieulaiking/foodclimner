#!/bin/bash
set -euo pipefail

# shellcheck source=common.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

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

# -- Paths ------------------------------------------------------------------
TEMPLATE="${SCRIPT_DIR}/template.slurm"
REPO_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORK_DIR="${WORK_DIR:-${REPO_ROOT_DEFAULT}}"
GENERATED_DIR="${WORK_DIR}/out/slurm/scripts"

JZ_MODEL_DIR="${DSDIR}/HuggingFace_Models"
SCRATCH_MODEL_DIR="${SCRATCH}/models"

# -- Model & dataset -------------------------------------------------------
MODEL_NAME="google/gemma-4-31B-it"
GPU_TYPE="a100"
GPU_NB="1"

DATASET="${WORK_DIR}/data/foodclimner-v1"
DATASET_SHORT="foodclimner-v1"

MODEL_PATH="$(common_resolve_model_path "${MODEL_NAME}" "${JZ_MODEL_DIR}" "${SCRATCH_MODEL_DIR}")"
MODEL_SHORT="${MODEL_NAME##*/}"

# -- Prompting runtime settings ---------------------------------------------
MAX_NEW_TOKENS="4096"

common_require_file "${TEMPLATE}" "SLURM template"
common_prepare_slurm_dirs "${WORK_DIR}" "${GENERATED_DIR}"

TIME_LIMIT="20:00:00"
if [[ "${DEBUG}" == "1" ]]; then
    TIME_LIMIT="2:00:00"
fi

IFS='|' read -r slurm_partition gpu_constraint arch_module cuda_module \
    <<< "$(common_resolve_gpu_settings "${GPU_TYPE}")"
slurm_account="$(common_resolve_slurm_account "${GPU_TYPE}")"
qos_line="$(common_resolve_debug_qos "${GPU_TYPE}" "${DEBUG}")"

# -- Ablation conditions ---------------------------------------------------
# Each entry: "suffix|flag1|flag2|..." (empty string = full baseline)
ablations=(
    # "full"
    # "no_definitions|--no_definitions"
    # "no_examples|--no_examples"
    # "no_rules|--no_rules"
    # "no_persona|--no_persona"
    "nothing|--no_definitions --no_examples --no_rules --no_persona"
)

# -- Submit one job per ablation -------------------------------------------
for entry in "${ablations[@]}"; do
    IFS='|' read -r suffix flags <<< "${entry}"
    output_dir="out/experiments_ablation/${suffix}/${MODEL_SHORT}/${DATASET_SHORT}"

    job_script="${GENERATED_DIR}/run_ablation_${MODEL_SHORT}_${suffix}.slurm"

    run_command="python -u src/experiment/run_prompting.py --model_name \"${MODEL_PATH}\" --data_dir \"${DATASET}\" --output_dir \"${output_dir}\" --max_new_tokens \"${MAX_NEW_TOKENS}\""
    if [[ -n "${flags}" ]]; then
        run_command="${run_command} ${flags}"
    fi

    common_render_template \
        "${TEMPLATE}" \
        "${job_script}" \
        "__JOB_NAME__=abl-${MODEL_SHORT}-${suffix}" \
        "__LOG_PREFIX__=abl-${MODEL_SHORT}-${suffix}" \
        "__GPU_NB__=${GPU_NB}" \
        "__GPU_CONSTRAINT__=${gpu_constraint}" \
        "__SLURM_PARTITION__=${slurm_partition}" \
        "__SLURM_ACCOUNT__=${slurm_account}" \
        "__ARCH_MODULE__=${arch_module}" \
        "__CUDA_MODULE__=${cuda_module}" \
        "__VENV_PATH__=.venv" \
        "__TIME_LIMIT__=${TIME_LIMIT}" \
        "__QOS_LINE__=${qos_line}" \
        "__WORK_DIR__=${WORK_DIR}" \
        "__RUN_COMMAND__=${run_command}"

    common_submit_job "${job_script}" "ablation=${suffix} model=${MODEL_SHORT}"

    if [[ "${DEBUG}" == "1" ]]; then
        echo "Debug mode enabled: submitted only one job." >&2
        exit 0
    fi
done
