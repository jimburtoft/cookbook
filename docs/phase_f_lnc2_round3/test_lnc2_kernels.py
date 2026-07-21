"""Test various NKI kernels under LNC=2 to isolate the NCC_ILLC059 bug.

Runs on paragao cluster's Beta 3 environment (nki 0.4.0b4, neuronx-cc 2.25.1280,
torch-neuronx 2.11.3.0.1278). All kernels tested single-process (no torchrun)
to isolate the wrap_nki compile path from the distributed layer.

Tests, in order of complexity:
  1. Trivial custom @nki.jit kernel (just a copy) -- pure wrap_nki + LNC=2 baseline
  2. fg_allgather (from Beta 3-shipped nkilib, older API) -- known-good older kernel
  3. all_reduce_hbm_kernel from nki-library 2.31 (via PYTHONPATH override) -- new kernel

Each test prints which stage fails so we can pinpoint what specifically breaks
LNC=2.
"""
import os
import sys
import traceback

# LNC config MUST be set before touching neuron
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")

import torch
import torch_neuronx
from torch_neuronx import wrap_nki

import nki
import nki.language as nl
import nki.isa as nisa
import nki.collectives as ncc
from nki.collectives import ReplicaGroup

print("=" * 70)
print(f"nki version: {getattr(nki, '__version__', '?')}")
print(f"torch_neuronx version: {getattr(torch_neuronx, '__version__', '?')}")
print(f"NEURON_LOGICAL_NC_CONFIG: {os.environ['NEURON_LOGICAL_NC_CONFIG']}")
print(f"NKI_LNC_DEGREE (used as kernel[lnc]): {os.environ.get('NKI_LNC_DEGREE', '2 (default for this test)')}")
lnc_degree = int(os.environ.get("NKI_LNC_DEGREE", "2"))
print("=" * 70)

device = torch.device("neuron")

# ------------------------------------------------------------------
# Test 1: Trivial @nki.jit kernel -- just copies input to output.
# No collectives, no shared_hbm names, no complications.
# If this fails under LNC=2, wrap_nki + LNC=2 is fundamentally broken.
# ------------------------------------------------------------------
print()
print("Test 1: Trivial NKI copy kernel under LNC=2")

@nki.jit
def trivial_copy(x: nl.ndarray) -> nl.ndarray:
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=x)
    return out

try:
    trivial_kernel = trivial_copy[lnc_degree]
    trivial_wrapped = wrap_nki(trivial_kernel)
    x = torch.ones(128, 512, dtype=torch.float32, device=device) * 7.0
    torch.neuron.synchronize()
    y = trivial_wrapped(x)
    torch.neuron.synchronize()
    got = y.cpu()[0, 0].item()
    print(f"  Test 1 result: got {got}, expected 7.0 -- {'PASS' if abs(got - 7.0) < 1e-3 else 'WRONG VALUE'}")
except Exception as e:
    print(f"  Test 1 FAILED: {type(e).__name__}: {str(e)[:300]}")

# ------------------------------------------------------------------
# Test 2: fg_allgather from the Beta 3-shipped nkilib (nki-library 2.30 era).
# This kernel is what the Beta 3 environment ships and is presumably tested.
# ------------------------------------------------------------------
print()
print("Test 2: fg_allgather (Beta 3-shipped nkilib) under LNC=2, single rank")

try:
    from nkilib.experimental.collectives.fg_allgather import fine_grained_allgather
    # tp_degree must be even (from docstring) and >= 4. Try tp_degree=4 with 1 group.
    # Single-process test -- rank 0 of 1 group. This won't actually communicate
    # but will exercise the compile path.
    print(f"  fine_grained_allgather imported: {fine_grained_allgather}")
    # Note: not calling it directly, just seeing if the kernel object exists at LNC=2.
    fg_kernel = fine_grained_allgather[lnc_degree]
    print(f"  kernel[{lnc_degree}] object: {fg_kernel}")
    print("  Skipping actual invocation (would need multi-rank setup)")
except Exception as e:
    print(f"  Test 2 FAILED to import/instantiate: {type(e).__name__}: {str(e)[:200]}")

# ------------------------------------------------------------------
# Test 3: 2.31 all_reduce_hbm_kernel (via PYTHONPATH override)
# ------------------------------------------------------------------
print()
print("Test 3: 2.31 all_reduce_hbm_kernel under LNC=2, single rank")

try:
    # PYTHONPATH should have prepended /fsx/self-managed/jburtoft/collective/nki-library/src
    from nkilib_src.nkilib.experimental.collectives.collectives import all_reduce_hbm_kernel
    print(f"  imported: {all_reduce_hbm_kernel}")
    ar_kernel = all_reduce_hbm_kernel[lnc_degree]
    ar_wrapped = wrap_nki(ar_kernel)
    x = torch.ones(128, 512, dtype=torch.float32, device=device) * 3.0
    torch.neuron.synchronize()
    # Single-rank replica group: all_reduce becomes identity (sum of 1 element)
    rg = ReplicaGroup([[0]])
    y = ar_wrapped(x, rg)
    torch.neuron.synchronize()
    got = y.cpu()[0, 0].item()
    print(f"  Test 3 result: got {got}, expected 3.0 (single rank) -- {'PASS' if abs(got - 3.0) < 1e-3 else 'WRONG VALUE'}")
except Exception as e:
    tb = traceback.format_exc()
    error_message = str(e)
    print(f"  Test 3 FAILED: {type(e).__name__}")
    print(f"  Error message (first 500 chars): {error_message[:500]}")
