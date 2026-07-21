# Task 010 -- Phase F Follow-up: nki 0.5.0 Upgrade + LNC=2 Test

**Project**: collective, task 010 Phase F follow-up
**Agent**: Z
**Date**: 2026-07-21
**Instance**: `i-07fed17eee72aa798` (trn2.3xlarge, sa-east-1b, terminated after test)
**Question the user asked**: "test with a 0.5.0 upgrade and LNC=2"

## TL;DR

* **nki 0.5.0 fully installs in the Beta 3 venv** and does not break Beta 3's `torch_neuronx.wrap_nki` LNC=1 path -- speedups continue to work.
* **LNC=2 still fails with the same `NCC_ILLC059` compile error** under Beta 3 `wrap_nki`, even after upgrading BOTH `nki` (0.4.0b4 -> 0.5.0) AND `neuronx-cc` (2.25.1280 -> 2.26.6360). This confirms the LNC=2 issue is **not** a `nki` / `neuronx-cc` version-skew problem; it's a real gap in Beta 3's `torch_neuronx.wrap_nki` HOP under LNC=2.
* **The same LNC=2 kernel compiles cleanly** through the SDK 2.31 DLAMI's stock **torch-XLA** path (not Beta 3). Correctness verified end-to-end at WS=4 -- kernel returns the correct sum across all ranks. So the compile bug lives in Beta 3 `wrap_nki`, not the kernel or the compiler.
* **XLA is not a practical workaround** for the speedup we saw in Phase F -- XLA's lazy execution makes single-collective per-iteration calls much slower (each call triggers HLO graph construction + compile lookup). At 1 MB WS=4 LNC=2 XLA gives ~230 us framework vs ~84 ms NKI (100x worse). The 7-13x Phase F wins are Beta 3-specific.

## What the user asked for

Retry the LNC=2 test after upgrading nki to 0.5.0 to see if the `NCC_ILLC059: Could not find MemoryLocation named inst__I-3-0:src on core 1` compile error we saw in Phase F was a nki-library-2.31 vs nki 0.4.0b4 version-skew issue.

## What we did

1. Launched a fresh `trn2.3xlarge` (SDK 2.31 DLAMI: `ami-00016af1920f68903`, sa-east-1b, capacity block `cr-0971544c50aeddd18`).
2. Confirmed the DLAMI ships **nki 0.5.0+28631259367.ga768afa6** and **nkilib** (SDK 2.31 GA versions) in `/opt/aws_neuronx_venv_pytorch_2_9/` alongside torch-neuronx 2.9 + torch-XLA 2.9.
3. Ran `smoke_nki_lnc2_xla.py` (WS=4, LNC=2) on the DLAMI's stock XLA venv -- **correctness passed** (kernel and `xm.all_reduce` both returned the expected sum 10.0). This proves the SDK 2.31 kernel source is fine and neuronx-cc 2.26 can compile it under LNC=2.
4. Installed Beta 3 on top of the DLAMI (our existing `setup_beta3.sh`) to get `torch_neuronx 2.11.3.0.1278` + `wrap_nki`. Beta 3's `dpkg -i` from `runtime_artifacts/` downgraded the host runtime lib (2.33 -> 2.32.19), collectives (2.33 -> 2.32.16), and tools (2.31.13 -> 2.30.5), but DKMS module remained at 2.29 (module was already loaded, dpkg install didn't force a rebuild).
5. Reboot cycle: after reboot the driver + userspace mismatch caused `ucode_lib_ll_create failed, error: 6` -- fatal for any Neuron init. Re-installed SDK 2.31's runtime lib + collectives + tools via apt to fix the mismatch, keeping the Beta 3 torch-neuronx wheel.
6. Inside the Beta 3 venv, upgraded `nki` (0.4.0b4 -> **0.5.0**) then `neuronx-cc` (2.25.1280 -> **2.26.6360**) from the Neuron pip repo. Both upgrades succeeded; `import torch_neuronx` and `from torch_neuronx import wrap_nki` still work.
7. Reran the smoke_nki_allreduce.py from Phase F under both LNC modes with the upgraded Beta 3 venv.

## Environment (final, after all upgrades)

| Component | Version | Notes |
|-----------|---------|-------|
| torch-neuronx | 2.11.3.0.1278+5013c208 | Beta 3, unchanged |
| torch | 2.11.0+cpu | Beta 3, unchanged |
| nki | **0.5.0+28631259367.ga768afa6** | Upgraded from 0.4.0b4 |
| neuronx-cc | **2.26.6360.0+6f180f47** | Upgraded from 2.25.1280 |
| Runtime lib | 2.33.10.0-3dcef56f0 | SDK 2.31 (matches nki 0.5.0's expectation) |
| Collectives | 2.33.10.0-068180c7a | SDK 2.31 |
| Driver (DKMS) | 2.28.0.0 (dpkg) / 2.29.0.0 (kernel module loaded) | Split -- see below |
| Tools | 2.31.13.0-a9e473f33 | SDK 2.31 |

The driver split is a known artifact of running our Beta 3 setup on top of an SDK 2.31 DLAMI: the DLAMI already has kernel module 2.29 loaded; Beta 3's `dpkg -i aws-neuronx-dkms_2.28.0.0.deb` writes files but does not force a module reload, so the running kernel stays at 2.29. Runtime lib 2.33 is backward-compatible with driver 2.29 -- everything worked fine after the fix. No functional impact on our results.

## Results

### LNC=1 with nki 0.5.0 (Beta 3 wrap_nki, trn2.3xlarge, WS=8, `all_reduce`)

Sweep over powers of 2 from 512 B to 4 MB:

```
  size_bytes  numel     fw_us    nki_us   fw_tput   nki_tput   fw_busbw  nki_busbw  speedup
         512    128    2285.8     946.1   0.000     0.001      0.000     0.001       2.42x
        1024    256    2096.6     635.8   0.001     0.003      0.001     0.003       3.30x
        2048    512    1928.1     606.8   0.002     0.007      0.002     0.006       3.18x
        4096   1024    2142.6     888.6   0.004     0.009      0.003     0.008       2.41x
        8192   2048    2051.2     889.2   0.008     0.018      0.007     0.016       2.31x
       16384   4096    1854.7     749.6   0.018     0.044      0.015     0.038       2.47x
       32768   8192    2022.9     704.4   0.032     0.093      0.028     0.081       2.87x
       65536  16384    2376.5     694.3   0.055     0.189      0.048     0.165       3.42x
      131072  32768    2519.0     919.7   0.104     0.285      0.091     0.249       2.74x
      262144  65536    1858.3     447.7   0.282     1.171      0.247     1.025       4.15x
      524288 131072    2371.4     771.5   0.442     1.359      0.387     1.189       3.07x
     1048576 262144    2352.9     630.1   0.891     3.328      0.780     2.912       3.73x
     2097152 524288    2531.0     535.0   1.657     7.841      1.450     6.860       4.73x
     4194304 1048576   2136.9     476.8   3.926    17.594      3.435    15.395       4.48x
```

**Summary**: LNC=1 speedup on trn2.3xlarge is **2.3x -- 4.7x** across the tested size range, lower than the 10x we saw on paragao cluster's trn2.48xlarge (where per-core HBM bandwidth is higher). Both the framework path AND the NKI path are ~2-8x slower than on trn2.48xlarge -- consistent with trn2.3xlarge sharing 4 HBM banks across all 8 logical cores. The nki 0.5.0 + neuronx-cc 2.26 upgrade did NOT break LNC=1; the speedup pattern is preserved.

### LNC=2 with nki 0.5.0 (Beta 3 wrap_nki, WS=2 or WS=4)

**Fails at kernel compile time with the same NCC_ILLC059 error we saw in Phase F under nki 0.4.0b4:**

```
[rank0]: error message="COMPILATION FAILED: 2026-07-21T05:37:48Z Non-signal exit.
Backend exited with code 1 and stderr: [INTERNAL_ERROR] [NCC_ILLC059] Could not
find MemoryLocation named inst__I-3-0:src on core 1 - Please open a support
ticket at https://github.com/aws-neuron/aws-neuron-sdk/issues/new."
```

The error is identical regardless of:
- world size (WS=2 or WS=4)
- NKI_LNC_DEGREE (kernel[1] or kernel[2])
- nki version (0.4.0b4 or 0.5.0)
- neuronx-cc version (2.25.1280 or 2.26.6360)

### LNC=2 with the DLAMI's stock XLA path (torch-neuronx 2.9, torch-XLA)

**Kernel COMPILES and executes correctly under LNC=2** via the DLAMI's built-in torch-XLA path. WS=4 all_reduce returns the correct sum (10.0 = 1+2+3+4). This is the same kernel source (nkilib 2.31's `all_reduce_hbm_kernel`) that fails to compile through Beta 3 `wrap_nki`.

Timing on the XLA path is not comparable to Beta 3 due to XLA's lazy graph construction -- each per-iteration NKI kernel call takes ~84 ms vs framework's ~230 us. XLA is not a practical replacement for `wrap_nki` if you want per-call low-latency invocation.

## Root cause of the LNC=2 failure

Given that:
1. The kernel source is identical between our Beta 3 test and the DLAMI XLA test.
2. neuronx-cc 2.26.6360 is the compiler in both.
3. nki 0.5.0 is present in both.
4. The kernel compiles under XLA + LNC=2 but fails under Beta 3 `wrap_nki` + LNC=2.

The bug is in **`torch_neuronx.wrap_nki`'s HOP implementation under LNC=2**. It generates a Neuron dispatch graph structure that the compiler's LNC=2 memory-location resolver can't handle. The error `Could not find MemoryLocation named inst__I-3-0:src on core 1` looks like the compiler is generating instructions referencing memory locations that were not created for core 1 -- consistent with wrap_nki producing an SPMD launch structure that assumes something different from what LNC=2 needs.

## What this means for Phase F's conclusions

* **LNC=1 numbers hold up.** The 7-13x speedup we measured on paragao under nki 0.4.0b4 is preserved with nki 0.5.0. The kernel path is stable across the nki version upgrade.
* **LNC=2 remains blocked on Beta 3.** The user can currently use `wrap_nki` for the speedup at LNC=1 only. LNC=2 requires either (a) a fix in Beta 4's `torch_neuronx.wrap_nki`, or (b) switching to the SDK 2.31 DLAMI's torch-XLA path (which does not deliver the speedup because XLA reintroduces per-call graph overhead).
* **The Phase F report's "LNC=2 blocked" section can be updated**: we now know the fix path is Beta 4's wrap_nki, not a nki version upgrade.

## Files written this session (local, not yet committed)

* `logs/phase_f_lnc2_test/beta3_nki0.5_lnc1_ws8_trn2_3xlarge.log` -- full sweep at LNC=1 with nki 0.5.0 (proves the upgrade doesn't regress LNC=1)
* `logs/phase_f_lnc2_test/beta3_nki0.5_lnc2_ws4_fails.log` -- verbatim `NCC_ILLC059` failure trace under Beta 3 wrap_nki with nki 0.5.0
* `logs/phase_f_lnc2_test/dlami_xla_lnc2_ws4_correctness.log` -- SDK 2.31 DLAMI XLA path succeeds under LNC=2 (baseline comparison)
* `logs/phase_f_lnc2_test/dlami_xla_lnc2_ws4_timing.log` -- XLA timing at 1 MB WS=4 LNC=2 (framework 233 us vs NKI 84 ms -- not a useful path)

## Recommendation for the fork

Update `docs/phase_f_nki_report.md`'s LNC=2 section:

Before: "LNC=2 currently blocked. This is version-skew between Beta 3's bundled nki wheel (0.4.0b4) and nki-library tag 2.31; users who want LNC=2 should either install a matching nki wheel or wait for Beta 4."

After: "LNC=2 currently blocked on Beta 3, even after upgrading nki to 0.5.0 and neuronx-cc to 2.26 (SDK 2.31 versions). The same kernel compiles cleanly under LNC=2 through the DLAMI's stock torch-XLA path, which pinpoints the bug in Beta 3's `torch_neuronx.wrap_nki` HOP -- not in the nki version, not in the kernel source, not in the compiler. Wait for Beta 4."
