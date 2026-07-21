#!/bin/bash
#SBATCH --job-name=nki-sweep
#SBATCH --partition=trainium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:45:00
#SBATCH --output=/fsx/self-managed/jburtoft/collective/logs/nki_sweep_%j.out
#SBATCH --error=/fsx/self-managed/jburtoft/collective/logs/nki_sweep_%j.err

export PATH="/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH"
source /fsx/self-managed/beta3/native_venv/bin/activate
export PYTHONPATH=/fsx/self-managed/jburtoft/collective/nki-library/src:$PYTHONPATH
export NEURON_LOGICAL_NC_CONFIG=${LNC:-1}
export NKI_LNC_DEGREE=${LNC:-1}
NPROC=${NPROC:-8}
COLL=${COLL:-all_reduce}
MAXSIZE=${MAXSIZE:-24}
TRIALS=${TRIALS:-50}
WARMUPS=${WARMUPS:-20}

echo "=== $(date -Iseconds) node=$SLURMD_NODENAME LNC=$NEURON_LOGICAL_NC_CONFIG NPROC=$NPROC COLL=$COLL ==="
torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
    /fsx/self-managed/jburtoft/collective/sweep_nki_vs_framework.py \
    --coll=$COLL --maxsize=$MAXSIZE --trials=$TRIALS --warmups=$WARMUPS
echo "=== exit=$? ==="
