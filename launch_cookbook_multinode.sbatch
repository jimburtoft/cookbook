#!/bin/bash
#SBATCH --job-name=cookbook-multi-node
#SBATCH --partition=trainium
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:45:00
#SBATCH --output=logs/multi_node_%j.out
#SBATCH --error=logs/multi_node_%j.err
#
# Cross-node cookbook communication benchmark across two trn2 nodes.
# Each node runs one torchrun process; torchrun spawns nproc_per_node
# workers within the node, all rendezvousing through rdzv on node 0.
# EFA transport is exercised whenever a rank on node 0 talks to a rank
# on node 1.
#
# Usage:
#   sbatch --export=NPROC=4,LNC=2,COLL=all_reduce launch_cookbook_multinode.sbatch
#
# Environment variables:
#   NPROC    -- per-node worker count.  World size = NPROC * nodes.  Default 8.
#               NOTE: on Beta 3 the Neuron distributed init has been observed to
#               fail with 'ncclInitComm' errors at NPROC=8 in the 2-node case
#               (WS=16).  NPROC<=4 works reliably.  Retry the higher values
#               once a newer Neuron SDK is available.
#   LNC      -- NEURON_LOGICAL_NC_CONFIG.  Default 2.
#   COLL     -- which cookbook collective to run.  Default all_reduce.
#   MAXSIZE  -- max message size (power of 2 in bytes).  Default 28 (256 MB).
#   TRIALS   -- timed iterations per size.  Default 100.
#   WARMUPS  -- warmup iterations per size.  Default 20.
#   NEURON_VENV -- path to the Neuron venv on shared storage.  Default:
#                  $HOME/workspace/native_venv.  Must be readable from every node.
#   USE_NKI  -- set to 1 to add --use-nki.  Requires nkilib on PYTHONPATH.
#   NKI_LIBRARY_SRC -- optional path to nki-library src/ to prepend to PYTHONPATH.

set -uo pipefail

# Slurm binaries may live outside the default PATH on some clusters.
export PATH="/opt/aws/pcs/scheduler/slurm-25.11/bin:${PATH:-/usr/bin:/bin}"

NPROC="${NPROC:-8}"
LNC="${LNC:-2}"
COLL="${COLL:-all_reduce}"
MAXSIZE="${MAXSIZE:-28}"
TRIALS="${TRIALS:-100}"
WARMUPS="${WARMUPS:-20}"
USE_NKI="${USE_NKI:-0}"
NEURON_VENV="${NEURON_VENV:-$HOME/workspace/native_venv}"
NKI_LIBRARY_SRC="${NKI_LIBRARY_SRC:-}"

# Locate benchmarks/ relative to the sbatch file's submission directory.
BENCH_DIR="${SLURM_SUBMIT_DIR:-$(dirname "${BASH_SOURCE[0]}")}/benchmarks"
if [ ! -d "$BENCH_DIR/communication" ]; then
  echo "ERROR: benchmarks/communication not found under $BENCH_DIR." >&2
  echo "       Submit from the repository root, or set SLURM_SUBMIT_DIR." >&2
  exit 2
fi
if [ ! -f "$NEURON_VENV/bin/activate" ]; then
  echo "ERROR: $NEURON_VENV/bin/activate not found." >&2
  echo "       Set NEURON_VENV to a Neuron venv reachable from every node." >&2
  exit 2
fi

# Head node for c10d rendezvous.  `scontrol getaddrs` prints "hostname: IP:PORT";
# take the IP and let torchrun pick its own rdzv port (29500).
NODELIST=$(scontrol show hostnames "$SLURM_NODELIST")
HEAD_NODE=$(echo "$NODELIST" | head -1)
HEAD_ADDR=$(scontrol getaddrs "$HEAD_NODE" 2>/dev/null | awk '{print $2}' | cut -d: -f1 | head -1)
if [ -z "$HEAD_ADDR" ]; then
  HEAD_ADDR=$(getent hosts "$HEAD_NODE" | awk '{print $1}' | head -1)
fi

echo "=== job=${SLURM_JOB_ID:-local} nnodes=$SLURM_NNODES NPROC=$NPROC LNC=$LNC COLL=$COLL USE_NKI=$USE_NKI ==="
echo "=== HEAD_NODE=$HEAD_NODE HEAD_ADDR=$HEAD_ADDR ==="
echo "=== NEURON_VENV=$NEURON_VENV NKI_LIBRARY_SRC=$NKI_LIBRARY_SRC ==="
echo "=== $(date -Iseconds) ==="

flag="--${COLL//_/-}"
extra=""
if [ "$USE_NKI" = "1" ]; then
  extra="--use-nki"
fi

# srun launches one torchrun per node; torchrun on each node spawns NPROC
# python workers and rendezvouses at HEAD_ADDR:29500.  --export=ALL propagates
# the parent PATH (needed on some clusters where /bin/bash isn't in the
# default srun env).
srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 --cpu-bind=none --export=ALL /bin/bash -c "
  export NEURON_LOGICAL_NC_CONFIG=$LNC
  export NKI_LNC_DEGREE=$LNC
  if [ -n \"$NKI_LIBRARY_SRC\" ]; then
    export PYTHONPATH=\"$NKI_LIBRARY_SRC:\${PYTHONPATH:-}\"
  fi
  cd $BENCH_DIR
  source $NEURON_VENV/bin/activate
  torchrun \
      --nnodes=$SLURM_NNODES \
      --nproc_per_node=$NPROC \
      --node_rank=\$SLURM_NODEID \
      --rdzv_backend=c10d \
      --rdzv_endpoint=$HEAD_ADDR:29500 \
      -m communication.run_all \
      --dist=torch --backend=neuron \
      $flag --scan \
      --trials $TRIALS --warmups $WARMUPS \
      --maxsize $MAXSIZE \
      --raw --bw-unit GBps \
      $extra
"

echo "=== $(date -Iseconds) exit=$? ==="
