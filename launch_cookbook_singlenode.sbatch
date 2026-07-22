#!/bin/bash
#SBATCH --job-name=cookbook-single-node
#SBATCH --partition=trainium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:30:00
#SBATCH --output=logs/single_node_%j.out
#SBATCH --error=logs/single_node_%j.err
#
# Intra-node cookbook communication benchmark on one trn2 node.
#
# Usage:
#   sbatch --export=WS=16,LNC=1,COLL=all_reduce launch_cookbook_singlenode.sbatch
#
# Environment variables the script honors:
#   WS       -- world size (per-node worker count).  Default 8.
#   LNC      -- NEURON_LOGICAL_NC_CONFIG.  Default 2.
#   COLL     -- which cookbook collective to run: all_reduce, all_gather,
#               reduce_scatter, all_to_all, broadcast, pt2pt.  Default all_reduce.
#   VISIBLE  -- optional NEURON_RT_VISIBLE_CORES pin (e.g. "3-4" for a cross-die pair).
#   MAXSIZE  -- max message size as a power of 2 in bytes.  Default 28 (256 MB).
#   TRIALS   -- timed iterations per size.  Default 100.
#   WARMUPS  -- warmup iterations per size.  Default 20.
#   NEURON_VENV -- path to a Neuron venv with torch-neuronx (Beta 3 or later).
#                  Default: $HOME/workspace/native_venv (Beta 3 DLC layout).
#   USE_NKI  -- set to 1 to add --use-nki (invokes the nkilib HBM kernels via
#               wrap_nki).  Requires nkilib on PYTHONPATH.  See NEURON.md.
#   NKI_LIBRARY_SRC -- path to the src/ of an nki-library checkout (needed if
#                      nkilib is not already installed in NEURON_VENV).  Optional.

set -uo pipefail

# The Slurm binaries may live outside the default PATH on some clusters.
# Add the common locations if the caller has not already done so.
export PATH="/opt/aws/pcs/scheduler/slurm-25.11/bin:${PATH:-/usr/bin:/bin}"

WS="${WS:-8}"
LNC="${LNC:-2}"
COLL="${COLL:-all_reduce}"
VISIBLE="${VISIBLE:-}"
MAXSIZE="${MAXSIZE:-28}"
TRIALS="${TRIALS:-100}"
WARMUPS="${WARMUPS:-20}"
USE_NKI="${USE_NKI:-0}"
NEURON_VENV="${NEURON_VENV:-$HOME/workspace/native_venv}"
NKI_LIBRARY_SRC="${NKI_LIBRARY_SRC:-}"

# Locate the benchmarks/communication directory relative to this sbatch file.
# `${BASH_SOURCE[0]}` under sbatch expands to the copied script path in the
# spool, so we prefer the caller's $SLURM_SUBMIT_DIR when available.
BENCH_DIR="${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/benchmarks"
if [ ! -d "$BENCH_DIR/communication" ]; then
  echo "ERROR: benchmarks/communication not found under $BENCH_DIR." >&2
  echo "       Submit from the repository root, or set SLURM_SUBMIT_DIR." >&2
  exit 2
fi

# Activate the Neuron venv.
if [ ! -f "$NEURON_VENV/bin/activate" ]; then
  echo "ERROR: $NEURON_VENV/bin/activate not found." >&2
  echo "       Set NEURON_VENV to point at a torch-neuronx Beta 3 (or later) venv." >&2
  exit 2
fi
source "$NEURON_VENV/bin/activate"

# Optional: prepend an nki-library src/ checkout to PYTHONPATH so --use-nki can
# import nkilib.experimental.collectives.collectives.  Skip this if nkilib is
# already installed in the venv.
if [ -n "$NKI_LIBRARY_SRC" ]; then
  export PYTHONPATH="$NKI_LIBRARY_SRC:${PYTHONPATH:-}"
fi

cd "$BENCH_DIR"

echo "=== job=${SLURM_JOB_ID:-local} node=${SLURMD_NODENAME:-$(hostname)} WS=$WS LNC=$LNC COLL=$COLL USE_NKI=$USE_NKI ==="
echo "=== VISIBLE=$VISIBLE MAXSIZE=$MAXSIZE TRIALS=$TRIALS WARMUPS=$WARMUPS ==="
echo "=== NEURON_VENV=$NEURON_VENV NKI_LIBRARY_SRC=$NKI_LIBRARY_SRC ==="
echo "=== $(date -Iseconds) ==="

flag="--${COLL//_/-}"
extra=()
if [ "$USE_NKI" = "1" ]; then
  extra+=("--use-nki")
fi

env_args=(
  "NEURON_LOGICAL_NC_CONFIG=$LNC"
  "NKI_LNC_DEGREE=$LNC"
)
if [ -n "$VISIBLE" ]; then
  env_args+=("NEURON_RT_VISIBLE_CORES=$VISIBLE")
fi

env "${env_args[@]}" \
    torchrun \
    --standalone --nnodes=1 --nproc_per_node="$WS" \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
    -m communication.run_all \
    --dist=torch --backend=neuron \
    "$flag" --scan \
    --trials "$TRIALS" --warmups "$WARMUPS" \
    --maxsize "$MAXSIZE" \
    --raw --bw-unit GBps \
    "${extra[@]}"

echo "=== $(date -Iseconds) exit=$? ==="
