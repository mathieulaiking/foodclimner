#!/bin/bash
#SBATCH --job-name=__JOB_NAME__
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:__GPU_NB__
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --partition=__SLURM_PARTITION__
#SBATCH -C __GPU_CONSTRAINT__
#SBATCH --time=__TIME_LIMIT__
__QOS_LINE__
#SBATCH --account=__SLURM_ACCOUNT__
#SBATCH --output=__WORK_DIR__/out/slurm/logs/%j___LOG_PREFIX__.out
#SBATCH --error=__WORK_DIR__/out/slurm/errs/%j___LOG_PREFIX__.err

set -euo pipefail

echo "starting job at $(date)"

WORK_DIR="__WORK_DIR__"

cd "$WORK_DIR"

mkdir -p out/slurm/logs out/slurm/errs

# load the matching architecture and CUDA modules first.
module purge
if [[ -n "__ARCH_MODULE__" ]]; then
	module load __ARCH_MODULE__
fi
module load __CUDA_MODULE__

# Ensure nvcc is on PATH (needed by flashinfer JIT compilation at runtime)
export CUDA_HOME="${CUDA_ROOT:-${CUDATOOLKIT_HOME:-}}"
export PATH="${CUDA_HOME}/bin:$PATH"

# Clear stale flashinfer JIT cache that may reference a different nvcc path
rm -rf "${HOME}/.cache/flashinfer"

# Load uv environment and environment variables
source __VENV_PATH__/bin/activate
export HF_HUB_CACHE="${HF_HUB_CACHE:-$SCRATCH/.cache/huggingface/hub}"
export HF_HUB_DISABLE_PROGRESS_BARS=1
# export TRANSFORMERS_VERBOSITY=error

__RUN_COMMAND__

echo "job finished at $(date)"
