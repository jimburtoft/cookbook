#!/usr/bin/env bash
# Full Phase C matrix driver. Runs, in order:
#   LNC=2 WS=2  (2 cores, same-die)   -- WS=4 already done
#   LNC=1 WS=2  (2 cores, same-die)
#   LNC=1 WS=4  (4 cores, cross-die crossing at rank 3-4 boundary)
#   LNC=1 WS=8  (8 cores, full trn2.3xlarge single-chip)
#
# All output lands under /home/ubuntu/logs/phase_c/lnc{1,2}_ws{2,4,8}/.
# reduce_scatter under LNC=2 WS=4 is already running separately.
set -uo pipefail

for cfg in "2 2" "1 2" "1 4" "1 8"; do
  read lnc ws <<< "$cfg"
  echo "==================== LNC=$lnc WS=$ws ===================="
  export NEURON_LOGICAL_NC_CONFIG=$lnc
  bash /home/ubuntu/run_phase_c.sh "$ws"
done

echo "==================== Full Phase C matrix complete ===================="
