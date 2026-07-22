#!/bin/bash
#SBATCH --job-name=nki-sweep
#SBATCH --partition=trainium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:45:00
#SBATCH --output=logs/nki_sweep_%j.out
#SBATCH --error=logs/nki_sweep_%j.err
#
# Standalone framework-vs-NKI sweep for one collective (see
# scripts/sweep_nki_vs_framework.py for the code and the arg spec).
#
# Usage:
#   sbatch --export=LNC=2,NPROC=4,COLL=all_reduce,MAXSIZE=24 scripts/sweep_nki.sbatch
#
# Environment variables:
#   LNC      -- NEURON_LOGICAL_NC_CONFIG.  Default 2.
#   NPROC    -- world size.  Default 8.
#   COLL     -- collective to sweep: all_reduce, all_gather, reduce_scatter, all_to_all.  Default all_reduce.
#   MAXSIZE  -- max message size (power of 2 in bytes).  Default 24.
#   TRIALS   -- timed iterations.  Default 50.
#   WARMUPS  -- warmup iterations.  Default 20.
#   NEURON_VENV -- path to Neuron venv.  Default $HOME/workspace/native_venv.
#   NKI_LIBRARY_SRC -- optional path to nki-library src/ to prepend to PYTHONPATH.

set -uo pipefail

export PATH="/opt/aws/pcs/scheduler/slurm-25.11/bin:${PATH:-/usr/bin:/bin}"

NEURON_VENV="${NEURON_VENV:-$HOME/workspace/native_venv}"
NKI_LIBRARY_SRC="${NKI_LIBRARY_SRC:-}"
LNC="${LNC:-2}"
NPROC="${NPROC:-8}"
COLL="${COLL:-all_reduce}"
MAXSIZE="${MAXSIZE:-24}"
TRIALS="${TRIALS:-50}"
WARMUPS="${WARMUPS:-20}"

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}"
SWEEP_PY="$SCRIPT_DIR/scripts/sweep_nki_vs_framework.py"
if [ ! -f "$SWEEP_PY" ]; then
  # If submitted from inside scripts/ directly
  SWEEP_PY="$SCRIPT_DIR/sweep_nki_vs_framework.py"
fi
if [ ! -f "$SWEEP_PY" ]; then
  echo "ERROR: could not find sweep_nki_vs_framework.py near $SCRIPT_DIR." >&2
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

echo "=== $(date -Iseconds) node=${SLURMD_NODENAME:-$(hostname)} LNC=$LNC NPROC=$NPROC COLL=$COLL ==="
torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
    "$SWEEP_PY" \
    --coll=$COLL --maxsize=$MAXSIZE --trials=$TRIALS --warmups=$WARMUPS
echo "=== exit=$? ==="
