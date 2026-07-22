#!/bin/bash
#SBATCH --job-name=smoke-nki
#SBATCH --partition=trainium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:15:00
#SBATCH --output=logs/smoke_nki_%j.out
#SBATCH --error=logs/smoke_nki_%j.err
#
# Smoke test: run 4-rank all_reduce two ways -- framework path
# (torch.distributed.all_reduce) and NKI path (wrap_nki around
# nkilib's all_reduce_hbm_kernel) -- and check that they return
# the same sum.  Prints one-shot timing for a 1 MB tensor.
#
# Usage:
#   sbatch --export=LNC=2,NPROC=4 scripts/smoke_nki_allreduce.sbatch
#
# Environment variables:
#   LNC       -- both NEURON_LOGICAL_NC_CONFIG and NKI_LNC_DEGREE.  Default 2.
#   NPROC     -- world size (defaults to 4 which is a good match for LNC=2).
#   NEURON_VENV     -- path to Neuron venv.  Default $HOME/workspace/native_venv.
#   NKI_LIBRARY_SRC -- optional path to nki-library src/ to prepend to PYTHONPATH.

set -uo pipefail

export PATH="/opt/aws/pcs/scheduler/slurm-25.11/bin:${PATH:-/usr/bin:/bin}"

NEURON_VENV="${NEURON_VENV:-$HOME/workspace/native_venv}"
NKI_LIBRARY_SRC="${NKI_LIBRARY_SRC:-}"
LNC="${LNC:-2}"
NPROC="${NPROC:-4}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
SMOKE_PY="$SCRIPT_DIR/scripts/smoke_nki_allreduce.py"
if [ ! -f "$SMOKE_PY" ]; then
  SMOKE_PY="$SCRIPT_DIR/smoke_nki_allreduce.py"
fi
if [ ! -f "$SMOKE_PY" ]; then
  echo "ERROR: could not find smoke_nki_allreduce.py near $SCRIPT_DIR." >&2
  exit 2
fi
if [ ! -f "$NEURON_VENV/bin/activate" ]; then
  echo "ERROR: $NEURON_VENV/bin/activate not found.  Set NEURON_VENV." >&2
  exit 2
fi

source "$NEURON_VENV/bin/activate"
if [ -n "$NKI_LIBRARY_SRC" ]; then
  export PYTHONPATH="$NKI_LIBRARY_SRC:${PYTHONPATH:-}"
fi
export NEURON_LOGICAL_NC_CONFIG=$LNC
export NKI_LNC_DEGREE=$LNC

echo "=== $(date -Iseconds) node=${SLURMD_NODENAME:-$(hostname)} LNC=$LNC NPROC=$NPROC ==="
torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
    "$SMOKE_PY"
echo "=== exit=$? ==="
