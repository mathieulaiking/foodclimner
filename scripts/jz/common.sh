#!/bin/bash

common_die() {
    echo "[ERROR] $*" >&2
    exit 1
}

common_require_file() {
    local file_path="$1"
    local file_label="${2:-file}"

    [[ -f "${file_path}" ]] || common_die "Missing ${file_label}: ${file_path}"
}

common_prepare_slurm_dirs() {
    local work_dir="$1"
    local generated_dir="$2"

    mkdir -p "${generated_dir}"
    mkdir -p "${work_dir}/out/slurm/logs"
    mkdir -p "${work_dir}/out/slurm/errs"
}

common_sed_escape() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//|/\\|}"
    value="${value//&/\\&}"
    printf '%s' "${value}"
}

common_render_template() {
    local template="$1"
    local output_file="$2"
    shift 2

    common_require_file "${template}" "template"

    local sed_args=()
    local pair
    local key
    local value

    for pair in "$@"; do
        key="${pair%%=*}"
        value="${pair#*=}"
        sed_args+=("-e" "s|${key}|$(common_sed_escape "${value}")|g")
    done

    sed "${sed_args[@]}" "${template}" > "${output_file}"
}

common_submit_job() {
    local job_script="$1"
    local label="${2:-}"

    if [[ -n "${label}" ]]; then
        echo "Submitting: ${label}"
    else
        echo "Submitting: ${job_script}"
    fi
    sbatch "${job_script}"
}

common_resolve_gpu_settings() {
    local gpu_type="$1"
    local partition=""
    local constraint=""
    local arch_module=""
    local cuda_module="cuda/12.4.1"

    gpu_type="${gpu_type,,}"
    case "${gpu_type}" in
        h100)
            partition="gpu_p6"
            constraint="h100"
            arch_module="arch/h100"
            ;;
        a100)
            partition="gpu_p5"
            constraint="a100"
            arch_module="arch/a100"
            ;;
        v100)
            partition="gpu_p13"
            constraint="v100-32g"
            arch_module=""
            ;;
        *)
            common_die "Unsupported gpu_type: ${gpu_type} (use a100, h100, v100)."
            ;;
    esac

    printf '%s|%s|%s|%s' "${partition}" "${constraint}" "${arch_module}" "${cuda_module}"
}

common_resolve_slurm_account() {
    local gpu_type="$1"
    local upper_type
    local account_var
    local account=""

    upper_type="${gpu_type^^}"
    account_var="SLURM_ACCOUNT_${upper_type}"
    account="${!account_var:-}"
    if [[ -n "${account}" ]]; then
        printf '%s' "${account}"
        return 0
    fi

    if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
        printf '%s' "${SLURM_ACCOUNT}"
        return 0
    fi

    printf 'xpc@%s' "${gpu_type}"
}

common_resolve_debug_qos() {
    local gpu_type="$1"
    local debug="${2:-0}"

    if [[ "${debug}" != "1" ]]; then
        printf ''
        return 0
    fi

    gpu_type="${gpu_type,,}"
    case "${gpu_type}" in
        h100)
            printf '#SBATCH --qos=qos_gpu_h100-dev'
            ;;
        a100)
            printf '#SBATCH --qos=qos_gpu_a100-dev'
            ;;
        v100)
            printf '#SBATCH --qos=qos_gpu-dev'
            ;;
        *)
            common_die "Unsupported gpu_type for qos: ${gpu_type}"
            ;;
    esac
}

common_resolve_model_path() {
    local model_name="$1"
    shift

    if [[ "${model_name}" == /* ]]; then
        printf '%s' "${model_name}"
        return 0
    fi

    local base_dir
    for base_dir in "$@"; do
        if [[ -n "${base_dir}" && -d "${base_dir}/${model_name}" ]]; then
            printf '%s' "${base_dir}/${model_name}"
            return 0
        fi
    done

    printf '%s' "${model_name}"
}
