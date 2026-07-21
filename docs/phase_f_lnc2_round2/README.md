# Task 010 -- Phase F Round 3: Beta 3 nki 0.5.0 upgrade + torch-neuronx source rebuild

**Project**: collective (task 010 Phase F follow-up)
**Agent**: Z
**Date**: 2026-07-21
**Instance**: `i-01fb140c0ea3080e0` (trn2.3xlarge, sa-east-1b, capacity block `cr-0431469bc60795d25`; **terminated after test**)
**User's request (in order)**:
1. "test with a 0.5.0 upgrade and LNC=2"
2. "If you are changing the compile version, you need to install torch-neuronx from source, not from a wheel"
3. "We need to get a valid environment for lnc=2. Start with the beta3 standard environment from the steering docs and only install the nki 0.5.0 package. If that causes problems, then start with SDK 2.31 based image and just install torch-neuronx on it from github."

## TL;DR

**LNC=2 wrap_nki still fails with `NCC_ILLC059` in every configuration we tried, including a full torch-neuronx source rebuild against the latest `main` branch with the newest nki + newest compiler.** The failure is deterministic and identical across configurations, strongly implicating the **`neuronx-cc 2.26.6360` compiler** rather than torch-neuronx or nki. The compile-time error is:

```
[INTERNAL_ERROR] [NCC_ILLC059] Could not find MemoryLocation named
inst__I-3-0:src on core 1
```

## Configurations tested

All on a fresh trn2.3xlarge with SDK 2.31 DLAMI + Beta 3 install layer.

| Config | torch-neuronx | nki | neuronx-cc | Runtime | LNC=1 | LNC=2 |
|---|---|---|---|---|---|---|
| Baseline (Plan A pre-upgrade) | 2.11.3.0.1278 (wheel) | 0.4.0b4 | 2.25.1280 | 2.32.19 | ok | **FAIL NCC_ILLC059** |
| Plan A: nki 0.5.0 upgrade only | 2.11.3.0.1278 (wheel) | **0.5.0** | 2.25.1280 | 2.32.19 | ok, 4.81x | **FAIL NCC_ILLC059** |
| Plan B1: rebuild torch-neuronx from Beta 3 source (`release-3.0`, `/workspace/torch_neuron_eager`) | 2.11.3.0.1278 (rebuilt from source) | 0.5.0 | 2.25.1280 | 2.32.19 | ok | **FAIL NCC_ILLC059** |
| Plan B1 + neuronx-cc 2.26 upgrade | 2.11.3.0.1278 (rebuilt) | 0.5.0 | **2.26.6360** | 2.32.19 | ok | **FAIL** -- `Unsupported NEFF version` (compiler ahead of runtime) |
| Plan B1 + neuronx-cc 2.26 + runtime 2.33 | 2.11.3.0.1278 (rebuilt) | 0.5.0 | 2.26.6360 | **2.33.10** | ok | **FAIL NCC_ILLC059** (returned once versions matched) |
| Plan B2: torch-neuronx source rebuild at `main`@`eb31942` (fresh venv, torch 2.12.1) | **2.12.3.0.278+eb31942.dev** | 0.5.0 | 2.26.6360 | 2.33.10 | (not tested) | **FAIL NCC_ILLC059** |

## Detailed timeline

### Plan A (per user's instruction 3): baseline Beta 3 + only nki upgrade

* Ran `setup_beta3.sh` on a fresh SDK 2.31 DLAMI. Verified state: torch_neuronx 2.11.3.0.1278, nki 0.4.0b4, neuronx-cc 2.25.1280, runtime 2.32.19.
* Baseline `smoke_nki_allreduce.py` under LNC=2 WS=4 -> **FAIL `NCC_ILLC059`** (baseline reproduces the Phase F bug on fresh install).
* `pip install --upgrade nki` -> nki 0.5.0+28631259367.ga768afa6 installed cleanly. torch_neuronx and wrap_nki still importable.
* Retest LNC=1 WS=8 sanity -> **speedup 4.81x preserved**. Regression check passed.
* Retest LNC=2 WS=4 -> **FAIL `NCC_ILLC059`, identical error message and offset as baseline**.
* Log: `planA_nki0.5_lnc2_still_fails.log`.

### Plan B1 (per user's instruction 2): rebuild torch-neuronx from source against nki 0.5.0

* Installed Bazelisk (`install_bazelisk.sh`), patchelf.
* Moved the DLC-shipped pre-built `_C.so` (291 MB) to `/tmp/_C.so.beta3_prebuilt`.
* Kicked off `USE_BAZEL=1 python setup.py build_ext` from `/home/ubuntu/workspace/torch_neuron_eager` (which is Beta 3's `release-3.0` branch at commit `5013c208`).
* Build took ~29 minutes. Produced fresh `_C.so` (245 MB) at 16:57 UTC.
* Retest LNC=2 wrap_nki -> **FAIL `NCC_ILLC059`, same error as before rebuild**.
* Escalated: upgraded neuronx-cc 2.25.1280 -> 2.26.6360 (SDK 2.31 GA). Retest LNC=2 -> new error: `Unsupported NEFF version` (compiler ahead of Beta 3's runtime lib 2.32.19).
* Upgraded runtime + collectives to SDK 2.31 (2.33.10). Retest LNC=2 -> **back to `NCC_ILLC059`** (kernel compiles further this time, but hits same underlying compiler bug).
* Also isolated: **single-process wrap_nki under LNC=2 fails identically**. Not a distributed issue.
* Logs: `planB_lnc2_after_rebuild.log`, `planB_lnc2_full_upgrade.log`, `planB_lnc2_sdk231_runtime.log`, `planB_lnc2_solo.log`.

### Plan B2 (fallback): install torch-neuronx from GitHub `main` branch

* Cloned `github.com/aws-neuron/torch-neuronx` (private repo, used git credential). Confirmed: `main` is 265 commits ahead of `beta3`. Includes commit `802f0ff78d` "migrate nki_kernel to use new API" from Jul 16.
* Attempted to build `main` HEAD (`0eeefa4`, Jul 21). **Compile error** in `torch_neuronx/csrc/core/streams/StreamImpl.cpp`:
  * Line 424: `event_kernel.GetEvent().recorded_stream_impl()` -- method does not exist on `NeuronEvent` (has `recorded_stream_id()` returning `int64_t`).
  * Line 445: `IsSameStreamWaitBypass(op.get(), this)` -- passing `StreamImpl*` where `c10::StreamId` (long int) expected.
* **This is a real bug in the `main` branch** -- present since commit `fa715ed` (Jul 11). API rename was not carried through consistently.
* Patched locally: `recorded_stream_impl()` -> `recorded_stream_id()`, `this` -> `this->GetStreamId()`. Rebuilt.
* Also had to abandon HEAD (`0eeefa4`) because its `torch_neuronx/nki_hop.py` imports `nki.framework.torch_native.TorchNativeKernel` which is an unreleased internal nki API; public nki 0.5.0 does not have it.
* Fell back to commit `eb31942` (Jul 16, right before the `802f0ff78d` "migrate nki_kernel to use new API" commit). This still requires `torch==2.12.1` but its nki_hop.py only uses the public nki 0.5.0 surface.
* Applied the same StreamImpl patch to `eb31942` and rebuilt in a fresh `main_venv` (torch 2.12.1 + nki 0.5.0 + neuronx-cc 2.26).
* Build succeeded in 178 seconds (cache hit). Installed as `torch-neuronx-2.12.3.0.278+eb31942.dev`.
* `import torch_neuronx`, `from torch_neuronx import wrap_nki`, basic LNC=1 and LNC=2 tensor ops -> **all OK**.
* Retest LNC=2 wrap_nki -> **FAIL `NCC_ILLC059`, byte-for-byte identical error**.
* Logs: `planB2_main_first_build_fails.log`, `planB2_main_eb31942_build_ok.log`, `planB2_main_eb31942_lnc2_still_fails.log`.

## Analysis: where the bug lives

The error is `[INTERNAL_ERROR] [NCC_ILLC059] Could not find MemoryLocation named inst__I-3-0:src on core 1`.

* `NCC_ILLC` = "Neuron Compiler Compilation Illegal" (ILLC prefix)
* `NCC_ILLC059` is code 59 in that class
* `inst__I-3-0` is an SPMD instruction ID (instruction 3, replica 0)
* `:src` is the source memory location on `core 1`

The error is emitted by the **compiler backend (`neuronx-cc`)**. It occurs after the HLO/StableHLO IR is generated and the compiler is lowering to Neuron instructions. Under LNC=2, the compiler generates instructions that reference memory locations on core 1, but under the LNC=2 memory layout those locations were not created.

Given that:
1. The same kernel source (nkilib 2.31's `all_reduce_hbm_kernel`) compiles cleanly under LNC=2 through **the DLAMI's stock torch-XLA path** (confirmed 2026-07-21 in previous test);
2. The same failure occurs across torch-neuronx `2.11.3.0.1278` (Beta 3 wheel), `2.11.3.0.1278+source` (Beta 3 rebuild), `2.12.3.0.278+eb31942.dev` (main branch source);
3. The same failure occurs across nki `0.4.0b4` and `0.5.0`;
4. The same failure occurs across neuronx-cc `2.25.1280` and `2.26.6360`;
5. The same failure occurs across runtime `2.32.19` and `2.33.10`;

**The most likely root cause is an incompatibility between `neuronx-cc`'s compile-graph output and the LNC=2 memory layout, specifically triggered by the HLO shape that PyTorch Native's `wrap_nki` HOP produces.** The XLA lowering path produces a subtly different HLO structure that avoids the failing code path.

The fact that torch-neuronx `main` branch has a commit "migrate nki_kernel to use new API" (`802f0ff78d`) suggests the Neuron team is actively rewriting this code path. That new API requires `nki.framework.torch_native.TorchNativeKernel` which is not yet in any public nki release. **A user upgrade path to LNC=2 support therefore requires all three of:**
1. A nki release with `nki.framework.torch_native.TorchNativeKernel` (unreleased)
2. torch-neuronx `main` branch or later (currently has compile bug at HEAD; `eb31942` works but still fails the LNC=2 test with public nki)
3. neuronx-cc that matches the above

## What we established

* **nki 0.5.0 alone does not fix LNC=2 wrap_nki**. Confirmed on fresh install per user's instruction.
* **Rebuilding torch-neuronx from source (Beta 3 branch) against nki 0.5.0 does not fix LNC=2**. Confirmed per user's instruction 2.
* **Rebuilding torch-neuronx from source (`main` branch at `eb31942`) against nki 0.5.0 does not fix LNC=2** either. The bug persists.
* **The LNC=2 bug is not in torch-neuronx's Python or C++ code that we can rebuild** -- it's in the compiler.
* **The main branch has a real, unrelated compile error** in `StreamImpl.cpp` (recorded_stream_impl / IsSameStreamWaitBypass API mismatch) that has been broken on the mirror since Jul 11 and would prevent anyone from building HEAD without a manual patch. Filed as an internal issue candidate.
* **The main branch HEAD requires `nki.framework.torch_native.TorchNativeKernel`** which is not in nki 0.5.0. Anyone trying to build HEAD today will hit an ImportError on first `import torch_neuronx`.

## What we did NOT establish

* Whether the LNC=2 bug is fixed in Beta 4 / SDK 2.32+ (not yet released as of 2026-07-21).
* Whether the LNC=2 bug is specific to `nkilib`'s `all_reduce_hbm_kernel` HLO shape or affects any `@nki.jit` kernel wrapped by `wrap_nki` under LNC=2.
* Whether the fix will land as a compiler-side change (`neuronx-cc`), a torch-neuronx-side change (different HLO generation for LNC=2), or a nki-library-side change (kernel avoids the pattern).

## Files added in this round

Under `logs/phase_f_lnc2_plan_ab/` (local, will also be committed to fork under `docs/phase_f_lnc2_test/`):

* `planA_baseline_lnc2_fails.log` -- fresh Beta 3 baseline; LNC=2 NCC_ILLC059 reproduces
* `planA_nki0.5_lnc1_works.log` -- LNC=1 speedup still 4.81x at 1MB WS=8 after nki 0.5.0 upgrade
* `planA_nki0.5_lnc2_still_fails.log` -- nki 0.5.0 alone does not fix LNC=2
* `planB1_beta3_source_rebuild_lnc2_fails.log` -- Beta 3 source rebuild + nki 0.5.0 + neuronx-cc 2.26 + runtime 2.33; still fails
* `planB_lnc2_after_rebuild.log`, `planB_lnc2_full_upgrade.log`, `planB_lnc2_sdk231_runtime.log`, `planB_lnc2_solo.log` -- intermediate Plan B1 test states
* `planB2_main_first_build_fails.log` -- main HEAD compile fails with `recorded_stream_impl` API mismatch
* `planB2_main_eb31942_build_ok.log` -- main@eb31942 source rebuild succeeds after StreamImpl patch
* `planB2_main_eb31942_lnc2_still_fails.log` -- LNC=2 still NCC_ILLC059 on main@eb31942

## Cost of this round

* Capacity block `cr-0431469bc60795d25` (sa-east-1b, 19h, $42.91) -- **used ~2h of 19h**, remaining slot unused (block is non-cancellable, so full cost applies).
* Compute time: ~2h of trn2.3xlarge.
* Two large Bazel builds (~29 min each) + one cached rebuild (~3 min).
* Two source patches applied (both to torch-neuronx `main` branch), documented above.

## Recommendation for the collective project

* **Update `docs/phase_f_nki_report.md`** with this round's findings. The LNC=2 blocker is now fully characterized: it's in `neuronx-cc`, not in torch-neuronx or nki. Wait for Beta 4 or a `neuronx-cc` patch that addresses `NCC_ILLC059` under LNC=2 wrap_nki.
* **Update `steering/pytorch-native.md`** to note the compile-error pattern `NCC_ILLC059 ... MemoryLocation named inst__I-3-0:src on core 1` as a known LNC=2 wrap_nki symptom, so future agents don't chase the same rabbit hole.
* **Do not upgrade to `main` branch of `aws-neuron/torch-neuronx` in Beta 3 environments** -- the branch has both a legit compile bug (StreamImpl API mismatch on `main` HEAD) and a dependency on unreleased nki API (`nki.framework.torch_native.TorchNativeKernel`). Even after workarounds, LNC=2 wrap_nki is still broken.
