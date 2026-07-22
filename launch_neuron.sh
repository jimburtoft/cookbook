#!/usr/bin/env bash
# Single-node launcher for the Neuron-ported communication benchmarks.
# For multi-node runs, use launch_cookbook_multinode.sbatch under Slurm.
#
# Usage:
#   ./launch_neuron.sh <world_size> [extra args to run_all.py]
#
# Examples:
#   ./launch_neuron.sh 4 --all-reduce --scan
#   ./launch_neuron.sh 8 --all-reduce --all-gather --reduce-scatter --scan \
#       --trials 200 --warmups 50 --raw
#   ./launch_neuron.sh 4 --all-reduce --use-nki --scan --raw --bw-unit GBps
#
# LNC mode is inherited from the host environment (`NEURON_LOGICAL_NC_CONFIG`).
# Verify with `neuron-ls` -- look at the 'logical-neuroncore-config' header.
# On trn2.3xlarge:
#   * LNC=2 -> 4 logical cores, so max WS = 4
#   * LNC=1 -> 8 logical cores, so max WS = 8
# On trn2.48xlarge multiply those by the chip count (up to 16).
#
# For `--use-nki` to work you must also set NKI_LNC_DEGREE to match LNC:
#   export NEURON_LOGICAL_NC_CONFIG=2
#   export NKI_LNC_DEGREE=2
# and have the nkilib source (or wheel) available on PYTHONPATH.  See NEURON.md.
#
# NEURON_RT_VISIBLE_CORES can be set from the environment to pin specific
# core ranges (e.g. NEURON_RT_VISIBLE_CORES=3-4 for a cross-die pair).

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
