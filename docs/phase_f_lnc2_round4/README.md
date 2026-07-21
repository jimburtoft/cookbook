# Task 010 -- Phase F Round 4: THE FIX -- correct `wrap_nki` invocation pattern enables LNC=2

**Project**: collective (task 010 Phase F)
**Agent**: Z
**Date**: 2026-07-21
**Environment**: paragao-ml-cluster (us-east-2), stock Beta 3 shared venv, single-chip trn2.48xlarge (trainium-2)
**Cost**: **$0** (all tests on shared cluster)

**User's insight (round 4)**: "the kernels in nki-library are designed for beta 3 and 2.31 SDK, so they should work out of the box. Maybe you need to call them in a different way? research in documentation and in nki-lib code comments"

## TL;DR

**User was right. The 2.31 HBM kernels DO work out of the box under LNC=2 on stock Beta 3 -- we were calling `wrap_nki` incorrectly.**

The correct invocation pattern is:

```python
wrapped = wrap_nki(kernel)[lnc]   # <-- LNC goes on the HOP caller (right)
y = wrapped(input, replica_group)
```

**Not**:

```python
wrapped = wrap_nki(kernel[lnc])   # <-- LNC on the kernel (wrong; silently ignored)
y = wrapped(input, replica_group)
```

With the correct pattern, LNC=2 4-rank distributed all_reduce works cleanly on stock Beta 3 (nki 0.4.0b4, neuronx-cc 2.25.1280, torch-neuronx 2.11.3.0.1278):

* Correctness: PASS (`10.0 = 1+2+3+4` across 4 ranks)
* Framework: 834 us at 1 MB
* NKI kernel: **161 us** at 1 MB
* **Speedup: 5.2x**

**Rounds 1, 2, and 3 were all chasing the wrong thing** -- upgrading nki, upgrading neuronx-cc, rebuilding torch-neuronx from source (Beta 3 branch and `main` branch), digging into the compiler internals. None of those changes would have fixed the problem, because the bug was in our own calling code.

## The two patterns, in detail

`torch_neuronx.wrap_nki` returns an `NKIHOPCaller` object. The `NKIHOPCaller` has its own `grid` field which is INITIALIZED TO EMPTY (`[]`) and is what actually gets passed to the compiler at kernel dispatch time. Look at `torch_neuronx/nki_hop.py`:

```python
def wrap_nki(nki_kernel, **kwargs) -> Any:
    sig = inspect.signature(nki_kernel.func)
    arg_names = list(sig.parameters.keys())
    kernel_default_args = {...}
    # k_fields captures the kernel's OWN fields (including lnc=2 if set)
    k_fields = {f.name: getattr(nki_kernel, f.name) for f in fields(nki_kernel) if f.name != "func"}
    kernel_idx = register_kernel_to_torch(nki_kernel.func, k_fields, arg_names, kernel_default_args, **kwargs)
    # NKIHOPCaller is created with grid=[]  <-- empty!
    return NKIHOPCaller(kernel_idx, [], arg_names, kernel_default_args)
```

The `NKIHOPCaller.__getitem__` is how you set the grid on the caller:

```python
def __getitem__(self, grid: "NKIGridType") -> "NKIHOPCaller":
    if isinstance(grid, int):
        grid = (grid,)
    grid = tuple(map(int, grid))
    return NKIHOPCaller(self.kernel_idx, grid, ...)
```

Then in the dispatch path (`get_dumped_config`), the `grid` from the caller is what determines the LNC that reaches the compiler:

```python
lnc = grid[0] if grid else 1  # empty grid -> lnc=1 (default!)
dconfig = kernel[lnc].dump_config(**meta_args)
```

So:
* `wrap_nki(kernel)` returns a caller with `grid=[]`, which resolves to `lnc=1` at compile time.
* `wrap_nki(kernel)[2]` returns a caller with `grid=(2,)`, which resolves to `lnc=2` at compile time.
* `wrap_nki(kernel[2])` returns a caller with `grid=[]` (still empty). The `[2]` on the kernel sets `nki_kernel.lnc=2` which is captured in `k_fields` and stored on the registered `TorchNeuronNKIKernelV3`, but this is orthogonal to the caller's `grid` field. At dispatch time the caller's empty grid overrides everything and lnc=1 is used.

**This mismatch is silent under LNC=1** (empty-grid default matches runtime), so all of Phase F's LNC=1 measurements were correct. **Under LNC=2 it's fatal**: the compiler generates SPMD-LNC=1 instructions, but the runtime is LNC=2, and the compiler backend errors out with:

```
[NCC_ILLC059] Could not find MemoryLocation named inst__I-3-0:src on core 1
```

## Verification, step by step

Test script `test_lnc2_wrap_grid.py` compares the three plausible patterns:

```
LNC=2 NKI_LNC_DEGREE=2
nki: 0.4.0+25407465723.g29063adb torch_neuronx: 2.11.3.0.1278+5013c208

Attempt 1: wrap_nki(kernel[lnc])(args)   # our old (wrong) pattern
  A1 FAIL: NCC_ILLC059

Attempt 2: wrap_nki(kernel)[lnc](args)   # CORRECT pattern
  A2 FAIL: NRT EXECUTION FAILED (single-rank ReplicaGroup, unrelated)

Attempt 3: wrap_nki(kernel[lnc])[lnc](args)   # both set (redundant)
  A3 FAIL: NRT EXECUTION FAILED (single-rank ReplicaGroup, unrelated)
```

Attempt 2 and Attempt 3 both **compile cleanly under LNC=2**. The remaining runtime error (`Invalid NEFF`) is an orthogonal issue caused by our single-process test using `ReplicaGroup([[0]])` -- the runtime rejects a 1-rank collective when only 1 process is running.

Confirmed with proper 4-rank distributed test (`test_lnc2_correct_pattern.py` under torchrun):

```
=== LNC=2 NPROC=4 ===
rank=0/4 LNC=2 NKI_LNC=2
rank=1/4 LNC=2 NKI_LNC=2
rank=2/4 LNC=2 NKI_LNC=2
rank=3/4 LNC=2 NKI_LNC=2
got: 10.0, expected: 10.0
CORRECTNESS: PASS

1 MB all_reduce (LNC=2 WS=4):
  framework: 834.4 us
  NKI kernel: 161.1 us
  speedup: 5.18x
```

## Full LNC=2 sweep across all 4 HBM collectives

Ran the sweep for each of `all_reduce, all_gather, reduce_scatter, all_to_all` at LNC=2 WS=4 with the corrected pattern:

| Collective | Framework floor (us) | NKI floor (us) | Peak speedup | Where it wins |
|-----------|--------------------:|---------------:|-------------:|---------------|
| all_reduce | ~900 | ~145 | **6.8x** at 4 KB | 512 B - 1 MB |
| all_gather | ~470 | ~140 | 3.5x at 1 KB | 512 B - 256 KB |
| reduce_scatter | ~520 | ~145 | 3.9x at 32 KB | 512 B - 256 KB |
| all_to_all | ~980 | ~140 | **7.5x** at 4 KB | 512 B - 1 MB |

At larger message sizes (2-16 MB), NKI still wins but by less (~2-3x), converging toward parity as NeuronLink bandwidth becomes the bottleneck.

Full data in `logs/phase_f_lnc2_working/lnc2_ws4_*.log`.

## Round-3 "compiler bug" claim: revised

Round 3 (2026-07-21 earlier) narrowed the bug to "`neuronx-cc`'s LNC=2 SPMD replication pass over any `nki.collectives.*` primitive". That was **correct in the sense that the compiler was the source of the error message**, but wrong about the CAUSE. The compiler is not buggy -- it correctly rejects the ill-formed SPMD graph it was fed. The ill-formed graph came from wrap_nki because we didn't set the grid on the caller.

The round-3 minimization test (Kernel A = trivial DMA, Kernel D = D+all_reduce; A passes and D fails) was on the right track but incomplete. Even Kernel D DOES compile under LNC=2 -- if you invoke it with `wrap_nki(kernel_D)[2](args)`. Our test used `wrap_nki(kernel_D[2])(args)`, which is why it failed.

## Cookbook fix

The main fix in the fork is a 1-line change in `benchmarks/communication/nki_ops.py`:

```python
# BEFORE (broken on LNC=2):
kernel = _KERNELS_BY_NAME[coll][lnc]
wrapped = _wrap_nki(kernel)

# AFTER (works on LNC=2 and LNC=1):
wrapped = _wrap_nki(_KERNELS_BY_NAME[coll])[lnc]
```

Also updated:
* Default `NKI_LNC_DEGREE` from 1 to 2 (matches trn2 default LNC=2)
* Docstring on `get_wrapped` explaining the invocation nuance and its consequence

Same fix applied to `scripts/smoke_nki_allreduce.py` and `scripts/sweep_nki_vs_framework.py`.

## Files added in this round

Under `logs/phase_f_lnc2_working/` (local; will also be committed to fork under `docs/phase_f_lnc2_round4/`):

* `test_lnc2_wrap_grid.py` -- single-process test comparing the three invocation patterns
* `test_lnc2_correct_pattern.py` -- 4-rank distributed test with the correct pattern
* `probe_correct_pattern_discovered.log` -- output of the wrap_grid test showing patterns 1 vs 2 vs 3
* `probe_4rank_correctness_5.18x_speedup.log` -- 4-rank correctness + 5.18x speedup at 1 MB
* `lnc2_ws4_all_reduce_sweep_5-6.8x.log` -- full size sweep for all_reduce
* `lnc2_ws4_all_gather_sweep_3.2-3.5x.log` -- full size sweep for all_gather
* `lnc2_ws4_reduce_scatter_sweep_3.0-3.9x.log` -- full size sweep for reduce_scatter
* `lnc2_ws4_all_to_all_sweep_6.5-7.5x.log` -- full size sweep for all_to_all

## Lessons

1. **Read the source, not the docs.** The `NKIHOPCaller.grid` shadowing `nki_kernel.lnc` is not called out in any documentation I found. It's visible in the source of `torch_neuronx/nki_hop.py` if you know to look.
2. **Bisect the smallest failing example, don't upgrade layers.** Rounds 1-3 sunk hours (and one $43 capacity block) into upgrading nki, neuronx-cc, and torch-neuronx from source. All wasted -- the versions had nothing to do with it. Round 3 came close with the progressive kernel minimization approach, but stopped at "the compiler emits the error" instead of "what does wrap_nki actually feed the compiler". The final step -- reading the wrap_nki source -- was the one that mattered.
3. **When the user says "the kernels should work out of the box", believe them and look for user error first.** The nkilib maintainers had already tested these kernels under LNC=2 (see the test file's `RANKS_LNC_2RANK = [(2, 2), (2, 1)]` parametrization). They wouldn't have shipped kernels that couldn't be called under LNC=2.

## Recommendation for the collective project

* **Update `docs/phase_f_nki_report.md`** with the corrected LNC=2 story: kernels are fine, the bug was in our calling pattern, and LNC=2 speedups are 3-7.5x depending on collective.
* **Consider filing a Neuron docs improvement** to make the `wrap_nki(kernel)[grid]` pattern explicit. The current API surface silently accepts `wrap_nki(kernel[grid])` and produces LNC=1 output, which is a footgun.
* **Rerun the full Phase F speedup measurements under LNC=2** on the full trn2.48xlarge (WS=16, WS=32, WS=64) with the corrected pattern. Previous rounds' LNC=1 sweep data stands, but LNC=2 is now the interesting configuration to characterize.
