# Running the EleutherAI Cookbook Communication Benchmarks on AWS Trainium (Neuron)

This branch adds first-class AWS Neuron support to
[`benchmarks/communication`](benchmarks/communication).  The rest of the
Cookbook is unchanged from upstream.

## What's added

1. **A `neuron` backend** for `torch.distributed`.  Pass `--backend=neuron`
   to any of the benchmark scripts (`all_reduce.py`, `run_all.py`, etc.) and
   the port picks up the Neuron distributed backend that `torch_neuronx`
   registers when imported.  All CUDA/NCCL code paths are unchanged.

2. **Neuron-friendly device placement and timing**.  `.cuda(local_rank)`
   becomes `.to("neuron")`, `torch.cuda.Event(enable_timing=True)` becomes a
   `torch.neuron.synchronize()`-gated `time.perf_counter()` pair, and
   `torch.cuda.get_device_properties()` falls back to a fixed HBM budget on
   Neuron.  Under CUDA the behavior is byte-equivalent to upstream.

3. **A `--use-nki` flag** on `all_reduce`, `all_gather`, `reduce_scatter`,
   and `all_to_all`.  When set, those collectives are executed by wrapping
   the corresponding HBM kernel from
   [`nki-library`](https://github.com/aws-neuron/nki-library) with
   `torch_neuronx.wrap_nki` instead of going through `torch.distributed`.
   For the message sizes where the framework layer's dispatch cost
   dominates (up to a few MB), this bypasses the framework overhead and
   dispatches directly to the compiled NEFF.

4. **Slurm launchers and standalone helper scripts** for both single-node
   and cross-node runs.

## Environment

Everything below assumes an AWS Neuron environment with `torch-neuronx`
installed (Beta 3 or later).  On a fresh trn2 instance the simplest path is
to use the DLC that AWS publishes and activate the `native_venv` it ships:

```bash
source $HOME/workspace/native_venv/bin/activate
python -c "import torch_neuronx; print(torch_neuronx.__version__)"
```

For a customer-specific install (from-source, custom base image, etc.) the
same code works as long as you can `import torch_neuronx` and get a
`torch.device("neuron")`.

## Quick start

```bash
git clone -b neuron-pytorch-native https://github.com/jimburtoft/cookbook.git
cd cookbook

# Single-node baseline: 8-way all_reduce through torch.distributed
./launch_neuron.sh 8 --all-reduce --scan --trials 200 --warmups 50 --raw
```

`launch_neuron.sh` is a small `torchrun` wrapper.  Every argument after the
first (the world size) is passed straight to `run_all.py`.  See
[`benchmarks/communication/README.md`](benchmarks/communication/README.md)
for the full list of `run_all.py` options.

## Using the NKI-kernel path

Add `--use-nki` to any run and the four HBM-kernel collectives will bypass
`torch.distributed`:

```bash
./launch_neuron.sh 8 --all-reduce --use-nki --scan --raw --bw-unit GBps
```

Prerequisites:

1. **`torch_neuronx.wrap_nki`** must be importable (Beta 3 and later ship
   it).
2. **`nkilib.experimental.collectives.collectives`** must be importable.
   This module ships with `nki-library` 2.31 and later.  If it is not
   already installed in your venv, clone the reference source and prepend
   it to `PYTHONPATH`:
   ```bash
   git clone --branch 2.31 https://github.com/aws-neuron/nki-library.git
   export PYTHONPATH=$PWD/nki-library/src:$PYTHONPATH
   ```
3. **LNC config**: set both `NEURON_LOGICAL_NC_CONFIG` (Neuron runtime) and
   `NKI_LNC_DEGREE` (the NKI SPMD launch grid) to matching values before
   invoking:
   ```bash
   export NEURON_LOGICAL_NC_CONFIG=2   # or 1
   export NKI_LNC_DEGREE=2             # keep in sync with the line above
   ```

Sizes below the NKI kernel's minimum (`128` fp32 elements for
`all_reduce`, `all_gather`, `all_to_all`, and `world_size * 128` for
`reduce_scatter`) transparently fall back to the framework path -- you get
one uniform scan without having to skip small message sizes manually.

`broadcast` and `pt2pt` are always run through the framework because
`nki-library` 2.31 does not export HBM kernels for those collectives.
`--all-to-all-v` also stays on the framework path (the `all_to_all_hbm_kernel`
does not support the vector variant).

### Which pattern to use in your own code

If you copy the wrap_nki calls out of `benchmarks/communication/nki_ops.py`,
note that the LNC degree must be set on the *HOP caller*, not on the
kernel:

```python
# CORRECT:
wrapped = torch_neuronx.wrap_nki(all_reduce_hbm_kernel)[2]  # LNC=2 on the caller
y = wrapped(input_tensor, replica_group)

# INCORRECT (silently uses LNC=1 -> compile error at LNC=2 runtime):
wrapped = torch_neuronx.wrap_nki(all_reduce_hbm_kernel[2])  # LNC=2 on the kernel
y = wrapped(input_tensor, replica_group)
```

The reason is inside `torch_neuronx/nki_hop.py`: `wrap_nki` builds a fresh
`NKIHOPCaller` with an empty `grid`, and it is that caller's grid (not the
kernel's own `lnc` field) that reaches the compiler.

## Slurm launchers

`launch_cookbook_singlenode.sbatch` and `launch_cookbook_multinode.sbatch`
wrap the same `run_all.py` invocation for cluster use.  They pick up config
via `--export`:

```bash
# Single node, LNC=2, WS=8, all_reduce with the NKI path enabled
mkdir -p logs
sbatch --export=WS=8,LNC=2,COLL=all_reduce,USE_NKI=1 \
       launch_cookbook_singlenode.sbatch

# Two nodes, LNC=2, 4 workers/node (WS=8) via EFA
sbatch --export=NPROC=4,LNC=2,COLL=all_reduce \
       launch_cookbook_multinode.sbatch
```

Both scripts write to `logs/` under the submission directory; create it
first if it does not exist.

Environment variables the scripts honor (see the sbatch headers for the
complete list):

| Variable | Meaning | Default |
|----------|---------|---------|
| `WS` / `NPROC` | World size (single-node) / per-node worker count (multi-node) | 8 |
| `LNC` | `NEURON_LOGICAL_NC_CONFIG` and `NKI_LNC_DEGREE` | 2 |
| `COLL` | one of `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `broadcast`, `pt2pt` | `all_reduce` |
| `MAXSIZE` | Max scan size as a power of 2 in bytes | 28 (256 MB) |
| `TRIALS` / `WARMUPS` | Timing loop counts | 100 / 20 |
| `USE_NKI` | Set to `1` to add `--use-nki` | 0 |
| `NEURON_VENV` | Path to your Neuron venv | `$HOME/workspace/native_venv` |
| `NKI_LIBRARY_SRC` | Optional: path to an `nki-library/src` checkout to prepend to `PYTHONPATH` | unset |

## Standalone smoke test and framework-vs-NKI sweep

Two helper scripts under `scripts/` are useful when characterizing NKI
speedup on your own hardware without going through the full `run_all.py`
argument surface:

* `scripts/smoke_nki_allreduce.py` -- runs 4-rank `all_reduce` two ways
  (framework and NKI kernel), verifies they return the same sum, and prints
  a 1 MB timing comparison.
* `scripts/sweep_nki_vs_framework.py` -- scans powers of two from 512 B up
  to a caller-controlled `--maxsize` and prints framework vs NKI duration,
  throughput, bus BW, and speedup side by side for one collective at a
  time.

Slurm wrappers `scripts/smoke_nki_allreduce.sbatch` and
`scripts/sweep_nki.sbatch` run them under `torchrun` with the standard
Neuron env vars set.

Direct invocation:

```bash
source $HOME/workspace/native_venv/bin/activate
export PYTHONPATH=/path/to/nki-library/src:$PYTHONPATH
export NEURON_LOGICAL_NC_CONFIG=2
export NKI_LNC_DEGREE=2
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    scripts/smoke_nki_allreduce.py
```

## What's still upstream framework-only

* `broadcast.py` and `pt2pt.py` never use the NKI path -- there is no
  matching HBM kernel in `nki-library` 2.31.  On the Neuron backend
  `pt2pt` (`dist.send`/`dist.recv`) is currently not supported at all;
  those runs will exit early with an error.
* Non-fp32 dtypes fall back to the framework path even when `--use-nki` is
  set.  The nki-library HBM kernels export fp32-only surfaces today.
* The `--all-to-all-v` variant stays on the framework path.

## Known constraints on Beta 3

Not bugs in this fork -- properties of the Neuron backend as of Beta 3:

* `all_to_all` on the framework path (`AllToAllXlaOp`) requires
  `world_size` to be one of `{4, 8, 16, or multiples of 32}`.  `WS=2` is
  rejected.  The NKI path (`--use-nki` + `all_to_all_hbm_kernel`) has its
  own constraints -- see the nki-library docs for details.
* In the 2-node cross-node case, `init_process_group(backend='neuron')`
  has been observed to fail at `nproc_per_node=8` (WS=16) in some
  configurations.  Reduce to `nproc_per_node<=4` if you see NCCL init
  errors.
* Cross-node `all_to_all` currently hits a runtime assertion outside a
  specific 4-rank-per-pod configuration.  Retry with a newer SDK.

## Verifying your install

Quick self-test that ensures `--use-nki` works end-to-end at LNC=2:

```bash
source $HOME/workspace/native_venv/bin/activate
export PYTHONPATH=/path/to/nki-library/src:$PYTHONPATH
export NEURON_LOGICAL_NC_CONFIG=2
export NKI_LNC_DEGREE=2

# Framework path baseline
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m communication.run_all \
    --dist=torch --backend=neuron --all-reduce --scan \
    --trials 20 --warmups 5 --maxsize 20 --raw --bw-unit GBps

# NKI path
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
    -m communication.run_all \
    --dist=torch --backend=neuron --all-reduce --use-nki --scan \
    --trials 20 --warmups 5 --maxsize 20 --raw --bw-unit GBps
```

The two runs should produce the same tensor sizes and reasonable
per-collective latency numbers, with the NKI run showing substantially
lower latency for messages in the 512 B - few MB range.
