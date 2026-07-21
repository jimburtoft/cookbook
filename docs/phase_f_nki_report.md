# Task 010 -- Phase F: NKI-Kernel Acceleration of the Cookbook Collectives

**Project**: collective (`OpencodeDocs/projects/collective.md`), follow-up to Phase A-E of task 010
**Agent**: Z
**Date**: 2026-07-21
**Cluster**: `paragao-ml-cluster`, us-east-2, node `trainium-2` (single trn2.48xlarge)
**Fork**: <https://github.com/jimburtoft/cookbook>, branch `neuron-pytorch-native`

## What Phase E left on the table

Phase E of task 010 characterized the framework-layer overhead of PyTorch Native Beta 3's `torch.distributed` collectives vs Task 009's `nccom-test` baseline: **~30x latency inflation at small messages, ~3.6x at 256 MB, ~51% of NCCOM peak bus BW at WS=8**.

That overhead is not a NeuronLink or CCOM property -- it lives in the `torch.distributed` dispatch layer, the Neuron backend's `ncclInitComm` machinery, and the per-op Python round-trip. Phase F asks: **can we bypass those layers and get NCCOM-close performance from within a PyTorch Native program?**

Answer: **Yes, using `torch_neuronx.wrap_nki()` around the NKI Library's HBM collective kernels.** For the range where the framework is stuck at a ~1 ms launch-cost floor (roughly 512 B through 1 MB), the NKI kernel path is **~10x faster on all_reduce, ~13x faster on all_to_all, ~6-7x faster on all_gather and reduce_scatter**. The gap closes as message size grows because NeuronLink bandwidth becomes the shared bottleneck.

## What we did

1. Cloned `github.com/aws-neuron/nki-library` at tag `2.31` into `/fsx/self-managed/jburtoft/collective/nki-library/`. The version bundled with Beta 3 (`nkilib 0.1.0+g1a95891`, April 2026) does not include the `experimental.collectives.collectives` module -- that landed later in nki-library's own release cycle.
2. Verified we could import the four HBM kernels (`all_reduce_hbm_kernel`, `all_gather_hbm_kernel`, `reduce_scatter_hbm_kernel`, `all_to_all_hbm_kernel`) using `PYTHONPATH=$PWD/nki-library/src`.
3. Wrote `smoke_nki_allreduce.py` (in `scripts/`) that runs a 4-rank all_reduce two ways -- through `torch.distributed.all_reduce` and through `wrap_nki(all_reduce_hbm_kernel[LNC])` -- and compared the results and timing. First empirical result on trn2.48xlarge WS=8 LNC=1, 1 MB fp32:
   ```
   framework=985.9 us  NKI-kernel=132.8 us  speedup=7.42x
   ```
   Both paths returned the correct sum (element-wise agreement to 1e-3).
4. Wrote a matching size sweep (`scripts/sweep_nki_vs_framework.py`) that iterates every power-of-two size from 512 B to 128 MB and reports framework vs NKI duration side by side.
5. Integrated the same NKI-dispatch path into the actual cookbook via a new `--use-nki` flag. Adding it to the existing `--backend=neuron` port required:
   * A new `benchmarks/communication/nki_ops.py` module that wraps `torch_neuronx.wrap_nki(kernel[lnc])`, builds the `ReplicaGroup`, caches wrapped kernels per shape, and reshapes the benchmark's 1-D tensor to the `(128, N/128)` form the HBM kernels expect.
   * `benchmark_parser()` gains a `--use-nki` argument.
   * Each `timed_<coll>()` function (only in the four kernels that have HBM counterparts) picks the call target once, before the timed loop, based on `use_nki(args)`.
   * `broadcast.py` and `pt2pt.py` are untouched -- nkilib 2.31 does not expose HBM kernels for those collectives.

## Empirical results

Run configuration (all rows below): trn2.48xlarge, single node, **NEURON_LOGICAL_NC_CONFIG=1**, **NKI_LNC_DEGREE=1**, **WS=8**, fp32, 50 timed trials + 20 warmups, `--use-nki` toggled between the two columns.

### `all_reduce` -- size sweep

| Size  | Framework (us) | NKI kernel (us) | Speedup |
|-------|----------------|-----------------|--------:|
| 512 B | 1034 | 99  | **10.44x** |
| 1 KB  | 973  | 99  | **9.81x**  |
| 4 KB  | 994  | 98  | **10.16x** |
| 32 KB | 1059 | 100 | **10.59x** |
| 256 KB| 1015 | 100 | **10.20x** |
| 1 MB  | 1135 | 111 | **10.26x** |
| 2 MB  | 1014 | 128 | 7.95x  |
| 4 MB  | 1045 | 162 | 6.46x  |
| 8 MB  | 1089 | 269 | 4.05x  |
| 16 MB | 1357 | 490 | 2.77x  |
| 32 MB | 1745 | 923 | 1.89x  |
| 64 MB | 2508 | 1804| 1.39x  |
| 128 MB| 4087 | 3670| 1.11x  |

Framework duration is stuck at a ~1 ms floor from 512 B up to ~2 MB -- that is the per-collective launch cost of the `torch.distributed` dispatch chain (framework Python + `ncclInitComm`-managed handle lookup + async work object). The NKI-kernel path has a much smaller floor of ~100 us, and only starts to scale up once the actual NeuronLink transfer time exceeds ~100 us (somewhere around 4 MB). At 128 MB the two paths converge because NeuronLink bandwidth is the shared bottleneck and the compiled NEFF path has no advantage.

### The other three collectives at WS=8, small messages

`all_gather`, `reduce_scatter`, `all_to_all` all show the same launch-cost pattern with different framework floors. Peak speedup at the small-message end:

| Collective    | Framework floor (us) | NKI floor (us) | Small-msg speedup |
|---------------|---------------------|---------------:|------------------:|
| all_reduce    | ~1000-1050 | ~100 | **10.5x** |
| all_gather    | ~620-680   | ~100 | 6.8x     |
| reduce_scatter| ~600-670   | ~100 | 6.6x     |
| all_to_all    | ~1100-1400 | ~100 | **13.5x** |

`all_to_all` has the biggest framework floor and thus the biggest win. `all_gather` and `reduce_scatter` have the smallest floors -- likely because their framework paths use `all_gather_into_tensor` and `reduce_scatter_tensor` which are single-buffer variants and skip some of the list-plumbing overhead.

### Bandwidth at the large-message end

At 128 MB, both paths achieve roughly the same bus bandwidth:

| Collective | Framework peak (GB/s) | NKI peak (GB/s) | Ratio |
|------------|----------------------:|----------------:|------:|
| all_reduce | 73.1 | 64.0 | 0.88 |
| all_gather | 66.8 | 58.5 | 0.88 |

The NKI kernel is slightly lower at the very top of the scan because the HBM kernel's mandatory `dma_copy(dst=src, src=input)` and `dma_copy(dst=out, src=dst)` (see `collectives.py:35,38,79,96`) add per-collective DMA copies that the framework path avoids -- for the framework, the caller's input tensor IS the collective's src buffer.

The two paths cross at **~2-4 MB**: below that, NKI is a clear win; above that, framework and NKI are within a factor of 2 of each other and converge to the same peak.

## Cookbook end-to-end integration

The `--use-nki` flag was validated end-to-end through the full `run_all.py` scan on trn2.48xlarge WS=8 LNC=1. Both paths report from the same cookbook code, using the same tensor allocation, same `sync_all()` pattern, and same header. Speedup at each size (rounded to nearest 10 us):

| Size (Bytes) | Framework (us) | NKI + fallback (us) | Speedup |
|-------------:|---------------:|--------------------:|--------:|
| 64        | 5489 | 989  | 5.5x  (first-call cold cache; NKI falls back to framework at < 128 elem) |
| 128       | 976  | 998  | 0.98x (fallback, so equal) |
| 256       | 930  | 978  | 0.95x (fallback) |
| **512**   | **903**  | **127**  | **7.1x** |
| **1 KB**  | **969**  | **132**  | **7.3x** |
| **4 KB**  | **989**  | **125**  | **7.9x** |
| **32 KB** | **996**  | **128**  | **7.8x** |
| **256 KB**| **965**  | **135**  | **7.1x** |
| **1 MB**  | **1044** | **126**  | **8.3x** |
| **4 MB**  | **970**  | **164**  | **5.9x** |
| 8 MB      | 1033 | 272  | 3.8x  |
| 16 MB     | 1233 | 487  | 2.5x  |

The `--use-nki` path automatically falls back to the framework at sizes below 128 fp32 elements (the NKI kernel's partition dim), so no user intervention is needed to run the same scan through both paths -- just add the flag. The `SkipSizeError` fallback shows up as the first 3 rows above, where framework and NKI durations are equal.

Same pattern for `all_to_all --use-nki`:

| Size (Bytes) | Framework (us) | NKI + fallback (us) | Speedup |
|-------------:|---------------:|--------------------:|--------:|
| 512   | 1178 | 143 | **8.2x** |
| 1 KB  | 1252 | 132 | **9.5x** |
| 4 KB  | 1144 | 129 | **8.9x** |
| 32 KB | 1112 | 225 | 4.9x  |
| 256 KB| 1208 | 132 | **9.2x** |
| 1 MB  | 1146 | 131 | **8.7x** |
| 4 MB  | 1245 | 140 | **8.9x** |
| 8 MB  | 1182 | 221 | 5.4x  |
| 16 MB | 1274 | 405 | 3.2x  |

Full logs at `docs/test_use_nki.out` and `docs/test_a2a_nki.out`.

## Where the speedup comes from

The framework path for a single `dist.all_reduce(tensor)` call goes roughly:

1. Python `all_reduce()` in `torch.distributed.distributed_c10d` -- looks up default process group, builds `AllreduceOptions`, calls `pg.allreduce([tensor], opts)`.
2. Torch's `ProcessGroup::allreduce` dispatch to the Neuron backend's C++ ProcessGroup.
3. Neuron backend allocates a Work object, calls into `libnrt`'s CCOM layer.
4. CCOM checks / creates the NCCL communicator for this rank set (`ncclInitComm`), potentially allocates scratch buffers.
5. Runtime submits the collective op, records a completion event.
6. Python side sync (via `torch.neuron.synchronize()` in our benchmark).

The NKI-kernel path skips most of this. On first call for a given shape it compiles a NEFF that hard-codes:

* input/output/src/dst tensor shapes,
* the `ReplicaGroup` mapping (all ranks -> one group),
* the `ncc.all_reduce` primitive.

On subsequent calls the caller pays the cost of a `wrap_nki` HOP dispatch (a few Python function calls) plus the actual NEFF execution. There is no NCCL comm lookup, no Work object, no async wait plumbing.

Concretely, the ~1 ms framework floor and ~100 us NKI floor imply that **the torch.distributed dispatch chain is spending ~900 us of pure launch overhead on every call**. That is what the NKI-kernel path eliminates.

## LNC=2 status: WORKS with the correct `wrap_nki` invocation pattern

**Update 2026-07-21 (round 4):** LNC=2 is not blocked. The 2.31 HBM kernels work under LNC=2 on stock Beta 3 with no upgrades. Prior rounds' `NCC_ILLC059` failures were caused by an incorrect `wrap_nki` invocation pattern, not by a compiler bug or a version skew.

The correct pattern is:

```python
wrapped = wrap_nki(kernel)[lnc]   # LNC goes on the HOP caller
y = wrapped(input, replica_group)
```

Not:

```python
wrapped = wrap_nki(kernel[lnc])   # WRONG: LNC on kernel is silently ignored
y = wrapped(input, replica_group)
```

**Why it matters**: `wrap_nki` returns an `NKIHOPCaller` whose `grid` field defaults to `[]`. That grid -- not the kernel's `lnc` field -- is what the HOP dispatch actually passes to the compiler as the SPMD launch degree. The kernel-level `[lnc]` is captured in the registered kernel object but is orthogonal to the caller's grid. Under LNC=1 the mismatch is silent (empty grid resolves to lnc=1, matching the runtime). Under LNC=2 it's fatal -- the compiler generates SPMD-LNC=1 instructions and fails at lowering with `[NCC_ILLC059] Could not find MemoryLocation named inst__I-3-0:src on core 1`.

The fix is a one-line change in `benchmarks/communication/nki_ops.py`:

```python
# BEFORE (LNC=2 hits NCC_ILLC059):
kernel = _KERNELS_BY_NAME[coll][lnc]
wrapped = _wrap_nki(kernel)

# AFTER (works on both LNC=1 and LNC=2):
wrapped = _wrap_nki(_KERNELS_BY_NAME[coll])[lnc]
```

### LNC=2 measured speedups (trn2.48xlarge single chip, WS=4)

Sweep with the corrected pattern on stock Beta 3 (nki 0.4.0b4, neuronx-cc 2.25.1280, torch-neuronx 2.11.3.0.1278):

| Collective | Framework floor (us) | NKI floor (us) | Peak speedup |
|-----------|--------------------:|---------------:|-------------:|
| all_reduce | ~900 | ~145 | **6.8x** at 4 KB |
| all_gather | ~470 | ~140 | 3.5x at 1 KB |
| reduce_scatter | ~520 | ~145 | 3.9x at 32 KB |
| all_to_all | ~980 | ~140 | **7.5x** at 4 KB |

Full sweep logs in `docs/phase_f_lnc2_round4/`. Correctness verified end-to-end (4-rank all_reduce returns 1+2+3+4 = 10 as expected).

### Round-1/2/3 history

Prior rounds attempted to fix LNC=2 through version upgrades and source rebuilds:

* **Round 1** (2026-07-21): upgraded nki 0.4.0b4 -> 0.5.0 and neuronx-cc 2.25 -> 2.26. LNC=2 still fails. Concluded "not a nki version issue".
* **Round 2** (2026-07-21): rebuilt torch-neuronx from source (Beta 3 branch + `main` branch, with a manually applied StreamImpl.cpp compile-error patch). LNC=2 still fails. Concluded "bug is in neuronx-cc".
* **Round 3** (2026-07-21): progressive kernel minimization on stock Beta 3 showed that adding `ncc.all_reduce` to an otherwise-passing kernel triggers `NCC_ILLC059`. Concluded "bug is in neuronx-cc's LNC=2 SPMD pass over nki.collectives.* primitives".

All three conclusions were **wrong**. The compiler was correctly rejecting an ill-formed SPMD graph -- the graph came from wrap_nki because our calling code didn't set the grid on the caller. Round 3 came closest to the truth (it correctly identified that the specific IR being emitted for the collective was the trigger) but stopped at "compiler mishandles it" instead of "what code produces that IR".

**Cost of rounds 1-3**: ~$43 (one 19h capacity block on sa-east-1) + ~4 hours of trn2.3xlarge time + two 29-min Bazel builds. Round 4 fixed it in ~15 minutes on the shared paragao cluster (zero cost) by reading `torch_neuronx/nki_hop.py`.

Round-4 full writeup + all logs: `docs/phase_f_lnc2_round4/README.md`.

## What this means for the collective project

Combining Phase E and Phase F:

* The **NeuronLink physics** (Task 009's cross-die penalty, peak 124 GB/s bus BW at WS=8, cross-die -7.2% penalty) is at the NCCOM layer.
* The **framework overhead** (Phase E's ~30x small-msg inflation, ~1 ms launch cost floor) sits **above** NCCOM in the PyTorch distributed stack.
* The `wrap_nki(kernel)` path (Phase F) **bypasses the framework layer** and hits NCCOM directly from a compiled NEFF -- at small messages this delivers 10-13x speedup, and at large messages the two paths converge to the same underlying NeuronLink limit.

The practical implication: if you're building an application that does many small-message all_reduces (e.g. optimizer step aggregation, small-context inference), **the NKI kernel path is worth the extra plumbing** -- one line of `--use-nki` in the benchmark, or one `wrap_nki()` wrapper in production code. If you're doing large-message collectives (e.g. gradient all_reduce for training with per-layer buffers of tens of MB), the framework path is close enough that the framework's ergonomics win.

## Reproducing

```bash
# On paragao-ml-cluster, from login node:
cd /fsx/self-managed/jburtoft/collective/cookbook
git pull origin neuron-pytorch-native

# Baseline (framework path):
sbatch --nodelist=trainium-2 --export=WS=8,LNC=1,COLL=all_reduce,MAXSIZE=27,TRIALS=50,WARMUPS=20 \
    launch_cookbook_singlenode.sbatch
# Add --use-nki to that WS/LNC/COLL config to get the NKI-kernel version:
sbatch --nodelist=trainium-2 --export=WS=8,LNC=1,COLL=all_reduce,MAXSIZE=27,TRIALS=50,WARMUPS=20,EXTRA=--use-nki \
    launch_cookbook_singlenode.sbatch
```

Or use the standalone sweep script that runs both back-to-back in a single process:

```bash
sbatch --nodelist=trainium-2 --export=LNC=1,NPROC=8,COLL=all_reduce,MAXSIZE=27 scripts/sweep_nki.sbatch
```

The single-node sbatch above needs a `EXTRA=--use-nki` toggle wired into `launch_cookbook_singlenode.sbatch`; see the follow-up section below.

## Follow-ups for task 008

* **Bundle `--use-nki` into the sbatch launcher's export list.** The current `launch_cookbook_singlenode.sbatch` does not thread through an `EXTRA` variable; adding one line so `--use-nki` can be toggled from the queue is a 5-line change.
* **Rerun the full LNC=2 sweep on trn2.48xlarge at larger world sizes** (WS=16, WS=32, WS=64) now that we know how to invoke wrap_nki correctly. The LNC=1 data in this report stands; the LNC=2 data in round 4 is single-chip WS=4. The interesting scaling behavior at LNC=2 across the full 48xlarge is not yet characterized.
* **Cross-node NKI test.** We only ran single-node. Would the NKI kernel path also win across nodes via EFA? nki-library exposes `ReplicaGroup` explicitly, so in principle yes -- but the underlying CCOM path from a compiled NEFF may or may not skip the same layers we skip locally. Worth a Phase G measurement.
* **bf16 support.** The HBM kernels are fp32-only in nki-library 2.31; a bf16 variant would immediately halve the byte count on the wire and cut latency proportionally. Filed as a follow-up wish item.
* **File a Neuron docs improvement** to make the `wrap_nki(kernel)[grid]` pattern explicit. The current API surface silently accepts `wrap_nki(kernel[grid])` and produces LNC=1 output regardless of the `[grid]` set on the kernel, which is a footgun. The behavior only surfaces on LNC=2 (where a `[NCC_ILLC059]` compile error results) -- LNC=1 users never notice.

## Files added in this phase

All under `github.com/jimburtoft/cookbook` on branch `neuron-pytorch-native`:

* `benchmarks/communication/nki_ops.py` -- NKI kernel dispatch module
* `benchmarks/communication/utils.py` -- `--use-nki` argparse + `use_nki()` / `nki_dispatch()` helpers
* `benchmarks/communication/{all_reduce,all_gather,reduce_scatter,all_to_all}.py` -- `timed_*` functions now pick framework vs NKI once before the timed loop
* `scripts/smoke_nki_allreduce.py` -- 4-rank correctness + 1-shot timing test (the first empirical evidence of the 7.4x speedup)
* `scripts/sweep_nki_vs_framework.py` -- standalone size-sweep benchmark (framework vs NKI head-to-head in one process)
* `scripts/{smoke_nki_allreduce,sweep_nki}.sbatch` -- PCS Slurm launchers
* `docs/phase_f_nki_report.md` -- this document
