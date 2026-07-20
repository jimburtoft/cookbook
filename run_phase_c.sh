#!/usr/bin/env bash
# Phase C intra-node measurement driver.
# Runs each collective at a given world size and writes structured JSON output
# to logs/phase_c/<lnc>_ws<N>/<collective>.log.
#
# Usage:
#   NEURON_LOGICAL_NC_CONFIG={1,2} bash run_phase_c.sh <world_size> [collective...]
# If no collectives are listed, runs all six (all_reduce, all_gather,
# reduce_scatter, all_to_all, broadcast, pt2pt).
#
# LNC mode is passed as env var; verified with a probe call to neuron-ls.

set -uo pipefail
export PATH="/home/ubuntu/workspace/native_venv/bin:$PATH"
source /home/ubuntu/workspace/native_venv/bin/activate

WS="${1:-4}"
shift || true

# Detect LNC mode: default to whatever the current shell has (2 unless overridden).
LNC="${NEURON_LOGICAL_NC_CONFIG:-2}"

LOG_DIR="/home/ubuntu/logs/phase_c/lnc${LNC}_ws${WS}"
mkdir -p "$LOG_DIR"
echo "LNC=$LNC WS=$WS log_dir=$LOG_DIR (NEURON_LOGICAL_NC_CONFIG=${NEURON_LOGICAL_NC_CONFIG:-unset})"

COLLECTIVES="$@"
if [ -z "$COLLECTIVES" ]; then
  COLLECTIVES="all_reduce all_gather reduce_scatter all_to_all broadcast pt2pt"
fi

cd /home/ubuntu/cookbook/benchmarks

# maxsize=28 -> up to 2^27 = 128 MB per rank. 256 MB (2^28) may blow HBM
# under LNC=1 (12 GB per logical core).
MAXSIZE=28
TRIALS=200
WARMUPS=50

for op in $COLLECTIVES; do
  flag="--${op//_/-}"
  out="$LOG_DIR/${op}.log"
  echo "=== $(date -Iseconds) $op WS=$WS LNC=$LNC ==="
  echo "=== $(date -Iseconds) $op WS=$WS LNC=$LNC ===" > "$out"
  # NEURON_LOGICAL_NC_CONFIG must be exported before torchrun so the child
  # python processes inherit it.
  timeout 900 env NEURON_LOGICAL_NC_CONFIG="$LNC" \
      torchrun \
      --standalone --nnodes=1 --nproc_per_node="$WS" \
      --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
      -m communication.run_all \
      --dist=torch --backend=neuron \
      "$flag" --scan \
      --trials "$TRIALS" --warmups "$WARMUPS" \
      --maxsize "$MAXSIZE" \
      --raw --bw-unit GBps \
      >> "$out" 2>&1
  rc=$?
  echo "=== exit=$rc ===" | tee -a "$out"
done

echo "Phase C run complete. Logs in $LOG_DIR"
