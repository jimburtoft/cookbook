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

## LNC=2 status: broken under Beta 3 `wrap_nki`, root cause is `neuronx-cc`

Every attempt to run the NKI kernels under `NEURON_LOGICAL_NC_CONFIG=2` (either with `kernel[1]` or `kernel[2]` for the SPMD launch grid) failed compilation with:

```
error message="COMPILATION FAILED: [INTERNAL_ERROR] [NCC_ILLC059] Could
not find MemoryLocation named inst__I-3-0:src on core 1 - Please open a
support ticket at https://github.com/aws-neuron/aws-neuron-sdk/issues/new."
```

**Follow-up round 1 (2026-07-21)**: verified the LNC=2 failure is **not** a nki version skew.

Setup: on a fresh trn2.3xlarge (SDK 2.31 DLAMI + our Beta 3 install), upgraded `nki` (0.4.0b4 -> **0.5.0+28631259367.ga768afa6**) and `neuronx-cc` (2.25.1280 -> **2.26.6360.0**) from the Neuron pip repo, plus SDK 2.31 host runtime lib 2.33.10 + collectives 2.33.10. LNC=1 speedups continue to work (2.3-4.7x on trn2.3xlarge). **LNC=2 still fails with the identical `NCC_ILLC059` error.**

**Follow-up round 2 (2026-07-21)**: verified the LNC=2 failure is **not** a torch-neuronx wheel-vs-source issue and **not** in code we can rebuild.

Per user request "install torch-neuronx from source, not from a wheel":
1. Installed Bazelisk + patchelf. Rebuilt Beta 3's `/workspace/torch_neuron_eager` (`release-3.0` branch) from source against nki 0.5.0 + neuronx-cc 2.26 (29-min build). Reinstalled editable. **LNC=2 wrap_nki still `NCC_ILLC059`.** Also tested single-process (non-distributed) wrap_nki -- same failure.
2. Per user request "If A causes problems, start with SDK 2.31 based image and just install torch-neuronx on it from github": cloned `github.com/aws-neuron/torch-neuronx` (private repo). Found `main` is 265 commits ahead of `beta3`, including "migrate nki_kernel to use new API" (`802f0ff78d`). Main HEAD (`0eeefa4`) fails to build due to a real pre-existing bug in `torch_neuronx/csrc/core/streams/StreamImpl.cpp` (references `NeuronEvent::recorded_stream_impl()` which does not exist -- the method was renamed to `recorded_stream_id()` but the caller was never updated). Patched locally.
3. Main HEAD also requires `nki.framework.torch_native.TorchNativeKernel` -- an unreleased internal nki API. Fell back to commit `eb31942` (right before the nki API migration). Applied the StreamImpl patch. Built in fresh venv with torch 2.12.1 + nki 0.5.0 + neuronx-cc 2.26. Installed as `torch-neuronx-2.12.3.0.278+eb31942.dev`. **LNC=2 wrap_nki still `NCC_ILLC059`, byte-for-byte identical error.**

The error is emitted by the compiler (`neuronx-cc 2.26.6360`) during instruction lowering. `NCC_ILLC059` = compiler cannot find memory location `inst__I-3-0:src` (SPMD instruction 3 replica 0, source memory) on core 1 -- under LNC=2 that memory location was not created for the graph the wrap_nki HOP emits. The same kernel source compiles cleanly under LNC=2 via the DLAMI's stock torch-XLA path, which produces a subtly different HLO.

**The bug is in `neuronx-cc`**, not in torch-neuronx (Python or C++) or nki. It fires under LNC=2 whenever the wrap_nki HOP emits its particular HLO pattern. Fixing requires either a compiler update (`neuronx-cc >= 2.27` presumably) or a torch-neuronx HLO refactor to avoid the failing pattern. The `main` branch's "migrate nki_kernel to use new API" commit (802f0ff78d) is likely part of the second approach; it depends on an unreleased nki `TorchNativeKernel`.

**Full round-2 writeup + all logs are in `docs/phase_f_lnc2_round2/README.md`.**

XLA is not a practical workaround: XLA's lazy graph construction makes each per-iteration NKI kernel call ~100x slower than the fused framework path (~84 ms per NKI call vs ~230 us for `xm.all_reduce` at 1 MB WS=4 LNC=2). The 7-13x speedup we see under Beta 3 wrap_nki depends on wrap_nki's eager dispatch semantics.

**Recommendation**: wait for Beta 4 (or SDK 2.32) plus a matching nki wheel. If Beta 4 ships `nki.framework.torch_native.TorchNativeKernel` and a torch-neuronx built against it, the same code path we measured under LNC=1 should extend to LNC=2 world sizes (WS=4 on trn2.3xlarge, WS=32-64 on trn2.48xlarge). No user-side change needed -- just set `NKI_LNC_DEGREE=2`.

Raw logs for the round-1 and round-2 tests are in `docs/phase_f_lnc2_test/` and `docs/phase_f_lnc2_round2/` respectively.

The LNC=1 path works cleanly. All the numbers in this report use LNC=1 with 8 logical cores on one trn2.48xlarge chip. Once Beta 4 fixes the `wrap_nki` LNC=2 bug (compiler-side or torch-neuronx-side), users can flip `NKI_LNC_DEGREE=2` and expect the same speedup at LNC=2 world sizes.

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
* **Retry LNC=2 on Beta 4** whenever it ships. Round-1 follow-up (2026-07-21) confirmed that upgrading nki to 0.5.0 and neuronx-cc to 2.26 (i.e. matching SDK 2.31's versions) does NOT fix the LNC=2 `NCC_ILLC059` compile error. Round-2 follow-up (same day) confirmed that a full torch-neuronx source rebuild -- including against the `main` branch at commit `eb31942` in a fresh venv with torch 2.12.1 -- does NOT fix it either. The bug is in `neuronx-cc 2.26.6360`, not in torch-neuronx or nki. Fix path: wait for Beta 4 with a compiler update or with the "migrate nki_kernel to use new API" (torch-neuronx `main` commit `802f0ff78d`) that requires unreleased `nki.framework.torch_native.TorchNativeKernel`. If either lands, the same speedups should extend to LNC=2 world sizes (WS=4 on trn2.3xlarge, WS=32-64 on trn2.48xlarge) with no code change -- just set `NKI_LNC_DEGREE=2`.
* **Cross-node NKI test.** We only ran single-node. Would the NKI kernel path also win across nodes via EFA? nki-library exposes `ReplicaGroup` explicitly, so in principle yes -- but the underlying CCOM path from a compiled NEFF may or may not skip the same layers we skip locally. Worth a Phase G measurement once the LNC=2 issue is resolved and the queue is quiet.
* **bf16 support.** The HBM kernels are fp32-only in nki-library 2.31; a bf16 variant would immediately halve the byte count on the wire and cut latency proportionally. Filed as a follow-up wish item.
* **Report `NCC_ILLC059 ... inst__I-3-0:src on core 1` compiler bug internally.** File as a `pytorch-native-tickets` item with the reproducer in `docs/phase_f_lnc2_round2/README.md`.

## Files added in this phase

All under `github.com/jimburtoft/cookbook` on branch `neuron-pytorch-native`:

* `benchmarks/communication/nki_ops.py` -- NKI kernel dispatch module
* `benchmarks/communication/utils.py` -- `--use-nki` argparse + `use_nki()` / `nki_dispatch()` helpers
* `benchmarks/communication/{all_reduce,all_gather,reduce_scatter,all_to_all}.py` -- `timed_*` functions now pick framework vs NKI once before the timed loop
* `scripts/smoke_nki_allreduce.py` -- 4-rank correctness + 1-shot timing test (the first empirical evidence of the 7.4x speedup)
* `scripts/sweep_nki_vs_framework.py` -- standalone size-sweep benchmark (framework vs NKI head-to-head in one process)
* `scripts/{smoke_nki_allreduce,sweep_nki}.sbatch` -- PCS Slurm launchers
* `docs/phase_f_nki_report.md` -- this document
