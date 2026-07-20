#!/usr/bin/env bash
# Single-node launcher for the Neuron-ported communication benchmarks.
# Usage:
#   ./launch_neuron.sh <world_size> <extra args to run_all.py>
# Examples:
#   ./launch_neuron.sh 4 --all-reduce --scan
#   ./launch_neuron.sh 8 --all-reduce --all-gather --reduce-scatter --scan --trials 200 --warmups 50 --raw
#
# LNC mode is inherited from the host environment. Verify by running
#   NEURON_LOGICAL_NC_CONFIG=1 neuron-ls
# before launch, or by looking at 'logical-neuroncore-config' at the top
# of `neuron-ls`. Under LNC=2 the max world size on trn2.3xlarge is 4;
# under LNC=1 it is 8.
#
# NEURON_RT_VISIBLE_CORES can be set from the environment to pin specific
# core ranges (e.g. NEURON_RT_VISIBLE_CORES=3-4 for cross-die pt2pt).
set -euo pipefail

WORLD_SIZE="${1:-2}"
shift || true

# torchrun sets RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT
# for every worker process. Our port reads these directly.
cd "$(dirname "$0")/benchmarks"

exec torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$WORLD_SIZE" \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29500 \
    -m communication.run_all \
    --dist=torch \
    --backend=neuron \
    "$@"
