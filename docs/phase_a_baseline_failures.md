# Phase A: Baseline attempt failures (unmodified EleutherAI/cookbook)

**Date**: 2026-07-20
**Instance**: i-089e28365d28e4354 (trn2.3xlarge, sa-east-1b, SDK 2.31 DLAMI + Beta 3 DLC)
**Repo**: `/home/ubuntu/cookbook_upstream`, at commit `827ee3b` (EleutherAI/cookbook main HEAD as of 2026-07-20)
**Environment**: `/home/ubuntu/workspace/native_venv/bin/activate`
  - torch 2.11.0+cpu
  - torch_neuronx 2.11.3.0.1278+5013c208
  - nki 0.4.0+25407465723.g29063adb

The task predicted three failure modes for running the unmodified cookbook on a Neuron host. All three reproduced, in the order they appear as we work around each one.

## Failure 1: `mpi4py` missing (launcher assumption)

**Command**:
```
python -m communication.run_all --dist=torch --scan --all-reduce --backend=nccl
```

**Output**:
```
ModuleNotFoundError: No module named 'mpi4py'
...
Exception
Cannot import mpi4py and MASTER_ADDR not set. Please either install mpi4py
or set the MASTER_ADDR on all ranks
```

**Root cause**: `utils.init_torch_distributed` assumes MPI-based launch. When `MASTER_ADDR` is not in the environment (typical for a non-MPI Python invocation), it falls back to `from mpi4py import MPI`. The Beta 3 venv does not include mpi4py.

**Task 010's fix**: Use torchrun instead (which sets `MASTER_ADDR`/`RANK`/`WORLD_SIZE`/`LOCAL_RANK`/`MASTER_PORT`), and delete the mpi4py fallback for the Neuron path.

## Failure 2: `torch.distributed` has no NCCL backend built in

**Command** (after wrapping with torchrun to defeat Failure 1):
```
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
    -m communication.run_all --dist=torch --scan --all-reduce --backend=nccl
```

**Output**:
```
RuntimeError: Distributed package doesn't have NCCL built in
```

**Root cause**: PyTorch Native Beta 3 ships `torch 2.11.0+cpu`. There is no CUDA in this build, so no NCCL backend is registered.

**Task 010's fix**: Add a `neuron` choice to `--backend` and register the Neuron distributed backend via `import torch_neuronx` before calling `init_process_group(backend='neuron')`.

## Failure 3: argparse rejects any non-CUDA backend

**Command**:
```
torchrun ... -m communication.run_all ... --backend=gloo
```

**Output**:
```
run_all.py: error: argument --backend: invalid choice: 'gloo' (choose from 'nccl', 'ccl', 'mpi')
```

**Root cause**: `utils.benchmark_parser()` hardcodes `choices=['nccl', 'ccl', 'mpi']`. Not only is there no `neuron` option -- there is no `gloo` fallback that would work on a CPU torch build either. The launcher / argparse layer must be extended.

**Task 010's fix**: Add `neuron` to the argparse `choices` alongside the original three.

## Not tested (predicted but not reached)

The following failure modes from the task description would have hit later in the trace, but were shadowed by the three above:

- `.cuda(local_rank)` / `f"cuda:{local_rank}"` -- CUDA-only tensor placement, ~26 sites
- `torch.cuda.Event(enable_timing=True)` -- device-side timing, ~12 sites across 6 files
- `torch.cuda.synchronize()` / `torch.cuda.get_device_properties()` -- device introspection
- `torch.cuda.empty_cache()` -- cache cleanup

These were all replaced in Phase B commits `28602ef` and `fef4008` before we ever hit them at runtime.

## Conclusion

Every layer of the upstream benchmark (launcher, backend registration, argparse validation, device placement, timing, memory query, cache cleanup) is CUDA-specific. The task's characterization ("Every launcher and device assumption needs replacement. Budget the port as a real piece of work, not a one-line change.") is accurate. The port is on branch `neuron-pytorch-native` at `github.com/jimburtoft/cookbook`, and consists of ~15 commits across `constants.py`, `utils.py`, and the 6 per-collective files.
