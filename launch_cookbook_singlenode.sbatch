#!/bin/bash
#SBATCH --job-name=cookbook-single-node
#SBATCH --partition=trainium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:30:00
#SBATCH --output=/fsx/self-managed/jburtoft/collective/logs/single_node_%j.out
#SBATCH --error=/fsx/self-managed/jburtoft/collective/logs/single_node_%j.err
#
# Intra-node cookbook communication benchmark on one trn2.48xlarge.
# Usage:
#   sbatch --export=WS=16,LNC=1,COLL=all_reduce launch_cookbook_singlenode.sbatch
# WS defaults to 8, LNC to 2 (48xlarge has 16 devices; LNC=2 -> 64 logical cores).
# Set NEURON_RT_VISIBLE_CORES via --export to pin a subset (e.g. VISIBLE=0-7).

set -uo pipefail

# Slurm binaries live outside the default PATH on paragao-ml-cluster
export PATH="/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH"

WS="${WS:-8}"
LNC="${LNC:-2}"
COLL="${COLL:-all_reduce}"
VISIBLE="${VISIBLE:-}"
MAXSIZE="${MAXSIZE:-28}"    # 2^27 = 128 MB
TRIALS="${TRIALS:-100}"
WARMUPS="${WARMUPS:-20}"

# Beta 3 venv on the shared FSx
source /fsx/self-managed/beta3/native_venv/bin/activate

cd /fsx/self-managed/jburtoft/collective/cookbook/benchmarks

echo "=== job=$SLURM_JOB_ID node=$SLURMD_NODENAME WS=$WS LNC=$LNC COLL=$COLL ==="
echo "=== VISIBLE=$VISIBLE MAXSIZE=$MAXSIZE TRIALS=$TRIALS WARMUPS=$WARMUPS ==="
echo "=== $(date -Iseconds) ==="

flag="--${COLL//_/-}"

env_args=(
  "NEURON_LOGICAL_NC_CONFIG=$LNC"
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
    --raw --bw-unit GBps

echo "=== $(date -Iseconds) exit=$? ==="
