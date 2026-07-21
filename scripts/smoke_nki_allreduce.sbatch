#!/bin/bash
#SBATCH --job-name=smoke-nki
#SBATCH --partition=trainium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:15:00
#SBATCH --output=/fsx/self-managed/jburtoft/collective/logs/smoke_nki_%j.out
#SBATCH --error=/fsx/self-managed/jburtoft/collective/logs/smoke_nki_%j.err

export PATH="/opt/aws/pcs/scheduler/slurm-25.11/bin:$PATH"
source /fsx/self-managed/beta3/native_venv/bin/activate
export PYTHONPATH=/fsx/self-managed/jburtoft/collective/nki-library/src:$PYTHONPATH
export NEURON_LOGICAL_NC_CONFIG=${NEURON_LOGICAL_NC_CONFIG:-2}
export NKI_LNC_DEGREE=${NKI_LNC_DEGREE:-1}
NPROC=${NPROC:-4}

echo "=== $(date -Iseconds) node=$SLURMD_NODENAME LNC=$NEURON_LOGICAL_NC_CONFIG NKI_LNC=$NKI_LNC_DEGREE NPROC=$NPROC ==="
torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
    /fsx/self-managed/jburtoft/collective/smoke_nki_allreduce.py
echo "=== exit=$? ==="
