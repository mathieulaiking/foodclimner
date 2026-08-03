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

# -- Argument lists (edit these) --------------------------------------------
JZ_MODEL_DIR="${DSDIR}/HuggingFace_Models"
SCRATCH_MODEL_DIR="${SCRATCH}/models/local"

models=(
    "gliner-community/gliner_large-v2.5|v100|1"
)

datasets=(
    "${WORK_DIR}/data/foodclimner-v1"
    "${WORK_DIR}/data/foodclimner-v1-simple"
)

# -- GLiNER runtime settings ------------------------------------------------
THRESHOLD="0.5"
BATCH_SIZE="8"
FLAT_NER=""  # set to "--flat_ner" to disable nested NER

common_require_file "${TEMPLATE}" "SLURM template"
common_prepare_slurm_dirs "${WORK_DIR}" "${GENERATED_DIR}"

TIME_LIMIT="20:00:00"
if [[ "${DEBUG}" == "1" ]]; then
    TIME_LIMIT="2:00:00"
fi

# -- Submit one job per model x dataset -------------------------------------
for model_entry in "${models[@]}"; do
    IFS='|' read -r model_name gpu_type gpu_nb <<< "${model_entry}"

    if [[ -z "${model_name}" || -z "${gpu_type}" || -z "${gpu_nb}" ]]; then
        common_die "Invalid model tuple: ${model_entry}"
    fi
    if ! [[ "${gpu_nb}" =~ ^[0-9]+$ ]]; then
        common_die "gpu_nb must be an integer in model tuple: ${model_entry}"
    fi

    IFS='|' read -r slurm_partition gpu_constraint arch_module cuda_module \
        <<< "$(common_resolve_gpu_settings "${gpu_type}")"
    slurm_account="$(common_resolve_slurm_account "${gpu_type}")"
    qos_line="$(common_resolve_debug_qos "${gpu_type}" "${DEBUG}")"

    model_path="$(common_resolve_model_path "${model_name}" "${JZ_MODEL_DIR}" "${SCRATCH_MODEL_DIR}")"
    model_short="${model_name##*/}"

    for dataset in "${datasets[@]}"; do
        dataset_short="${dataset##*/}"
        output_dir="out/experiments/baselines/${model_short}/${dataset_short}"

        job_script="${GENERATED_DIR}/run_gliner_${model_short}_${dataset_short}.slurm"
        run_command="python -u src/experiment/run_gliner.py --model_name \"${model_path}\" --data_dir \"${dataset}\" --output_dir \"${output_dir}\" --threshold \"${THRESHOLD}\" --batch_size \"${BATCH_SIZE}\" --local_files_only ${FLAT_NER}"

        common_render_template \
            "${TEMPLATE}" \
            "${job_script}" \
            "__JOB_NAME__=gliner-${model_short}-${dataset_short}" \
            "__LOG_PREFIX__=gliner-${model_short}-${dataset_short}" \
            "__GPU_NB__=${gpu_nb}" \
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

        common_submit_job "${job_script}" "gliner model=${model_short} dataset=${dataset_short}"

        if [[ "${DEBUG}" == "1" ]]; then
            echo "Debug mode enabled: submitted only one job." >&2
            exit 0
        fi
    done
done