#!/bin/bash
#SBATCH --job-name=cookbook-multi-node
#SBATCH --partition=trainium
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:45:00
#SBATCH --output=/fsx/self-managed/jburtoft/collective/logs/multi_node_%j.out
#SBATCH --error=/fsx/self-managed/jburtoft/collective/logs/multi_node_%j.err
#
# Cross-node cookbook communication benchmark across 2x trn2.48xlarge.
# Each node runs one torchrun process; torchrun spawns nproc_per_node
# workers within the node, all rendezvousing through rdzv on node 0.
#
# Usage:
#   sbatch --export=NPROC=8,LNC=2,COLL=all_reduce launch_cookbook_multinode.sbatch
# NPROC = per-node worker count. World size = NPROC * 2.
# EFA transport is exercised whenever a rank on trainium-1 talks to a
# rank on trainium-2.

set -uo pipefail

# Slurm binaries live outside the default PATH on paragao-ml-cluster
export PATH="/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH"

NPROC="${NPROC:-8}"
LNC="${LNC:-2}"
COLL="${COLL:-all_reduce}"
MAXSIZE="${MAXSIZE:-28}"   # 128 MB
TRIALS="${TRIALS:-100}"
WARMUPS="${WARMUPS:-20}"

# Beta 3 venv on the shared FSx (visible to every node via /fsx)
source /fsx/self-managed/beta3/native_venv/bin/activate

cd /fsx/self-managed/jburtoft/collective/cookbook/benchmarks

# Head node for c10d rendezvous (rank-0 side of torchrun).
# `scontrol getaddrs` prints "hostname: IP:PORT" -- take the IP only,
# then let torchrun pick its own rdzv port (29500).
NODELIST=$(scontrol show hostnames "$SLURM_NODELIST")
HEAD_NODE=$(echo "$NODELIST" | head -1)
HEAD_ADDR=$(scontrol getaddrs "$HEAD_NODE" 2>/dev/null | awk '{print $2}' | cut -d: -f1 | head -1)
if [ -z "$HEAD_ADDR" ]; then
  HEAD_ADDR=$(getent hosts "$HEAD_NODE" | awk '{print $1}' | head -1)
fi

echo "=== job=$SLURM_JOB_ID nnodes=$SLURM_NNODES NPROC=$NPROC LNC=$LNC COLL=$COLL ==="
echo "=== HEAD_NODE=$HEAD_NODE HEAD_ADDR=$HEAD_ADDR ==="
echo "=== $(date -Iseconds) ==="

flag="--${COLL//_/-}"

# srun runs one torchrun per node. torchrun on each node spawns NPROC
# python workers and rendezvouses at HEAD_ADDR:29500.
# --export=ALL is required so PATH (which contains the Slurm binaries)
# and the shell utilities on the compute node are visible to bash.
srun --nodes="$SLURM_NNODES" --ntasks-per-node=1 --cpu-bind=none --export=ALL /bin/bash -c "
  export NEURON_LOGICAL_NC_CONFIG=$LNC
  cd /fsx/self-managed/jburtoft/collective/cookbook/benchmarks
  source /fsx/self-managed/beta3/native_venv/bin/activate
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
      --raw --bw-unit GBps
"

echo "=== $(date -Iseconds) exit=$? ==="
