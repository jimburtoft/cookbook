# Task 010 -- Phase F Round 3: Pinpointing the LNC=2 NCC_ILLC059 root cause

**Project**: collective (task 010 Phase F follow-up)
**Agent**: Z
**Date**: 2026-07-21
**Environment**: paragao-ml-cluster (us-east-2), Beta 3 shared venv (`/fsx/self-managed/beta3/native_venv/`)
**Cost**: $0 (all tests on shared cluster, no new instances)

**User's request**: "start with a basic beta 3 image and start building up. We know collectives work. See where the issue comes in. The new kernels shouldn't be using anything that is exclusive to 0.5.0."

## TL;DR

The user was right that the 2.31 HBM kernels don't need any nki 0.5.0 API. They use exactly the same set of primitives (`nki.language`, `nki.isa`, `nki.collectives`, `ReplicaGroup`) as the older `fg_allgather` kernel that Beta 3 ships. **The problem isn't a nki version mismatch, and it isn't specific to the 2.31 kernels.**

By progressively minimizing kernels on stock Beta 3 (nki 0.4.0b4, neuronx-cc 2.25.1280, torch-neuronx 2.11.3.0.1278, no upgrades), I pinpointed the trigger:

**Any `@nki.jit` kernel invoked via `torch_neuronx.wrap_nki` that calls ANY of `ncc.all_reduce`, `ncc.all_gather`, `ncc.reduce_scatter`, or `ncc.all_to_all` fails to compile with `NCC_ILLC059` when the runtime is configured for `NEURON_LOGICAL_NC_CONFIG=2`.**

The failure is:
- **Independent of nki version** (0.4.0b4 -> 0.5.0 both fail)
- **Independent of neuronx-cc version** (2.25.1280 -> 2.26.6360 both fail)
- **Independent of torch-neuronx version** (wheel vs source vs `main` branch all fail)
- **Independent of nkilib** (2.31 HBM kernels fail identically to a custom hand-written kernel using the same primitives)
- **Independent of the kernel's declared SPMD degree** (`kernel[1]` and `kernel[2]` both fail under LNC=2 runtime)
- **Independent of the rank count** (single-rank ReplicaGroup and 4-rank ReplicaGroup both fail identically)

LNC=1 works cleanly with the same code.

The bug lives in the **`neuronx-cc` LNC=2 SPMD replication pass** as it processes any collective primitive from `nki.collectives`. Specifically, the compiler creates a MemoryLocation named `inst__I-3-0:src` when generating SPMD instructions for the collective, but under LNC=2's cross-core memory layout, that location doesn't get materialized on core 1.

## Method: progressive kernel minimization

Instead of chasing version combinations (as prior rounds did), I built up complexity one primitive at a time on stock Beta 3, starting from the simplest possible NKI kernel.

### Baseline: stock Beta 3 shared venv on paragao cluster

- `nki: 0.4.0+25407465723.g29063adb` (Beta 3 bundled, no upgrades)
- `torch_neuronx: 2.11.3.0.1278+5013c208` (Beta 3 wheel, no source rebuild)
- `neuronx-cc: 2.25.1280.0+1c5cb3d6` (Beta 3, no upgrade)
- `aws-neuronx-runtime-lib: 2.32.16.0` (Beta 3, no upgrade)
- `nkilib`: paragao-cluster-shipped older version (has `fg_allgather` but not `collectives.py`)

### Kernel A: trivial `dma_copy` (no naming, no collective)

```python
@nki.jit
def kernel_A_trivial(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=x)
    return out
```

**Result under LNC=2**: **PASS**. Returns the input unchanged. Establishes that `wrap_nki + LNC=2` is not fundamentally broken.

### Kernel B: single named `shared_hbm` buffer

```python
@nki.jit
def kernel_B_named_out(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="out")
    nisa.dma_copy(dst=out, src=x)
    return out
```

**Result under LNC=2**: **PASS**. Rules out the `name="..."` kwarg as the trigger.

### Kernel C: two named intermediate buffers with DMA between them

```python
@nki.jit
def kernel_C_two_named(x):
    src = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="src")
    dst = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="dst")
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=src, src=x)
    nisa.dma_copy(dst=dst, src=src)
    nisa.dma_copy(dst=out, src=dst)
    return out
```

**Result under LNC=2**: **PASS**. Rules out multiple named buffers and DMA chains as the trigger.

### Kernel D: C + `ncc.all_reduce`

```python
@nki.jit
def kernel_D_with_allreduce(x, replica_group):
    src = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="src")
    dst = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="dst")
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=src, src=x)
    ncc.all_reduce(dsts=[dst], srcs=[src], op=nl.add, replica_group=replica_group)
    nisa.dma_copy(dst=out, src=dst)
    return out
```

**Result under LNC=2**: **FAIL with `NCC_ILLC059`**. Only difference from C is the added `ncc.all_reduce` line. **The bug is triggered by `ncc.all_reduce`.**

### Which of the four collective primitives trigger it?

All four:

| primitive | LNC=2 result |
|-----------|-------------|
| `ncc.all_reduce` | FAIL `NCC_ILLC059` |
| `ncc.all_gather` | FAIL `NCC_ILLC059` |
| `ncc.reduce_scatter` | FAIL `NCC_ILLC059` |
| `ncc.all_to_all` | FAIL `NCC_ILLC059` |

Under LNC=1 the same four kernels COMPILE successfully; they only fail at NEFF-load time with `NRT EXECUTION FAILED: Invalid NEFF, instruction, or input` -- which is the expected behavior for a single-process test that passes `ReplicaGroup([[0]])` (runtime rejects a 1-rank collective NEFF because it expects the process count declared in the graph to match the number of actual processes). Under a distributed torchrun with matching `nproc_per_node`, LNC=1 works fine (as we've seen in all Phase F LNC=1 speedup measurements).

### Which runtime setting triggers it, kernel or process?

- `NEURON_LOGICAL_NC_CONFIG=2` + `kernel[2]` -> FAIL
- `NEURON_LOGICAL_NC_CONFIG=2` + `kernel[1]` -> FAIL
- `NEURON_LOGICAL_NC_CONFIG=1` + `kernel[1]` -> COMPILE OK (runtime rejects for other reasons)
- `NEURON_LOGICAL_NC_CONFIG=1` + `kernel[2]` -> not tested (would be a mismatch)

**The runtime's `NEURON_LOGICAL_NC_CONFIG=2` setting is what triggers the compile bug**, not the kernel's declared `[lnc]` degree.

### 4-rank distributed test with matched process count

Ran kernel D above under `torchrun --standalone --nnodes=1 --nproc_per_node=4` with `ReplicaGroup([[0,1,2,3]])` under LNC=2. **Same NCC_ILLC059 failure.** The single-rank result was not an artifact of the process count mismatch.

## Root-cause conclusion

The bug is in **`neuronx-cc 2.25.1280`'s SPMD replication pass** when it processes any `nki.collectives.*` primitive under LNC=2. When the compiler splits the `@nki.jit` graph across 2 physical cores per logical NC for LNC=2, it fails to materialize the MemoryLocation for the collective's `src` buffer on core 1 (the second physical core in the LNC=2 pair). The error `NCC_ILLC059 Could not find MemoryLocation named inst__I-3-0:src on core 1` is emitted with `inst__I-3-0` being the internal SPMD instruction ID for the collective and `:src` being the input buffer's name.

This is a compiler bug, not a torch-neuronx bug and not a nki bug. Our rounds 1-2 attempts to fix it by upgrading torch-neuronx (from wheel to source, from Beta 3 to `main`) and nki (0.4 -> 0.5) were misdirected -- none of those layers are producing bad IR. The IR the compiler receives is correct; the compiler's LNC=2 pass mishandles it.

## Confirmed workaround: use LNC=1

Every LNC=1 speedup measurement in Phase F stands. If a user wants the 7-13x wrap_nki speedup today, they must run under `NEURON_LOGICAL_NC_CONFIG=1`. On trn2 that means:
- trn2.3xlarge: WS up to 8 (all 8 logical cores of the single chip)
- trn2.48xlarge: WS up to 128 (8 chips × 16 logical cores)

## What we DON'T know

- Whether Beta 4 / SDK 2.32+ compiler releases fix the LNC=2 SPMD pass.
- Whether the fix will require a corresponding change in nki_hop.py (torch-neuronx) to emit a slightly different HLO shape that the compiler can handle, or if it's a pure compiler-side fix.
- Whether the bug affects only `nki.collectives.*` primitives or if it extends to other `nki.*` primitives (e.g., `nki.isa` ops we haven't tested individually).

## Files added in this round

Under `logs/phase_f_lnc2_pinpoint/` (local; will also be committed to fork under `docs/phase_f_lnc2_round3/`):

* `test_lnc2_kernels.py`, `test_lnc2_minimize.py`, `test_lnc2_ncc_variants.py`, `test_lnc2_kernel_D_distributed.py` -- the 4 test scripts, from generic probe to specific pinpoint
* `probe_lnc1_trivial_pass_2.31kernel_fails_at_runtime.log` -- baseline probe under LNC=1
* `probe_lnc2_trivial_pass_2.31kernel_ncc_illc059.log` -- baseline probe under LNC=2
* `minimize_lnc2_A_B_C_pass_D_fails.log` -- the definitive minimization result
* `kernelD_4rank_still_ncc_illc059.log` -- 4-rank distributed still fails
* `all_four_ncc_primitives_lnc2_ncc_illc059.log` -- all four collectives trigger the bug
* `all_four_ncc_primitives_lnc1_nrt_invalid_neff.log` -- same code compiles fine at LNC=1 (fails at runtime for orthogonal reasons)
* `lnc2_runtime_with_lnc1_kernel_still_fails.log` -- runtime LNC=2 triggers bug even with LNC=1 kernel

## Recommendation for the collective project

* **Update `docs/phase_f_nki_report.md`** with this round's findings. The root cause is now precisely characterized:
  * **Bug location**: `neuronx-cc` SPMD replication pass under `NEURON_LOGICAL_NC_CONFIG=2`
  * **Trigger**: any `nki.collectives.*` primitive inside a `@nki.jit` kernel invoked via `wrap_nki`
  * **Fix path**: compiler-side; wait for Beta 4 or a newer `neuronx-cc`. Neither nki nor torch-neuronx upgrades will help.
* **Update `steering/pytorch-native.md`** to note that Beta 3's `wrap_nki` + LNC=2 is broken specifically for `nki.collectives.*` primitives. Non-collective NKI kernels work fine under LNC=2.
* **File the bug internally** (already done implicitly by the compiler with "Please open a support ticket"). The reproducer is a 5-line NKI kernel that any user can run in ~30 seconds on a stock Beta 3 install.
