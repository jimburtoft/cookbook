"""Test the correct wrap_nki invocation pattern for LNC=2.

Hypothesis: wrap_nki(kernel[2]) does NOT propagate the LNC=2 to the HOP caller's
grid. The grid is [] by default, which makes the dispatch use lnc=1 even when
the kernel object has lnc=2 baked in.

The correct pattern is wrap_nki(kernel)[grid](args) -- setting the grid on the
NKIHOPCaller, not on the kernel.
"""
import os
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")

import torch
import torch_neuronx
from torch_neuronx import wrap_nki
import nki
import nki.language as nl
import nki.isa as nisa
import nki.collectives as ncc
from nki.collectives import ReplicaGroup

lnc = int(os.environ.get("NKI_LNC_DEGREE", "2"))
device = torch.device("neuron")

@nki.jit
def with_all_reduce(x, rg):
    src = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="src")
    dst = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="dst")
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=src, src=x)
    ncc.all_reduce(dsts=[dst], srcs=[src], op=nl.add, replica_group=rg)
    nisa.dma_copy(dst=out, src=dst)
    return out


print("=" * 70)
print(f"LNC={os.environ['NEURON_LOGICAL_NC_CONFIG']} NKI_LNC_DEGREE={lnc}")
print(f"nki: {nki.__version__} torch_neuronx: {torch_neuronx.__version__}")
print("=" * 70)

x = torch.ones(128, 512, dtype=torch.float32, device=device) * 6.0
torch.neuron.synchronize()
rg = ReplicaGroup([[0]])


# ------------------------------------------------------------------
# Attempt 1: what we've been doing WRONG all along
# wrap_nki(kernel[2])(args) -- LNC set on kernel, NOT propagated to caller grid
# ------------------------------------------------------------------
print("\nAttempt 1: wrap_nki(kernel[lnc])(args)  # our old (wrong?) pattern")
try:
    wrapped = wrap_nki(with_all_reduce[lnc])
    y = wrapped(x, rg)
    torch.neuron.synchronize()
    print(f"  A1 PASS: got {y.cpu()[0,0].item()}")
except Exception as e:
    msg = str(e)
    if 'NCC_ILLC059' in msg:
        print(f"  A1 FAIL: NCC_ILLC059")
    else:
        print(f"  A1 FAIL: {type(e).__name__}: {msg[:200]}")


# ------------------------------------------------------------------
# Attempt 2: the CORRECT pattern
# wrap_nki(kernel)[lnc](args) -- LNC set on the HOP caller's grid
# ------------------------------------------------------------------
print("\nAttempt 2: wrap_nki(kernel)[lnc](args)  # correct pattern per nki_hop.py inspection")
try:
    wrapped = wrap_nki(with_all_reduce)
    # Set grid on the caller, then invoke
    y = wrapped[lnc](x, rg)
    torch.neuron.synchronize()
    print(f"  A2 PASS: got {y.cpu()[0,0].item()}")
except Exception as e:
    msg = str(e)
    if 'NCC_ILLC059' in msg:
        print(f"  A2 FAIL: NCC_ILLC059")
    else:
        print(f"  A2 FAIL: {type(e).__name__}: {msg[:300]}")


# ------------------------------------------------------------------
# Attempt 3: both
# ------------------------------------------------------------------
print("\nAttempt 3: wrap_nki(kernel[lnc])[lnc](args)  # both set")
try:
    wrapped = wrap_nki(with_all_reduce[lnc])
    y = wrapped[lnc](x, rg)
    torch.neuron.synchronize()
    print(f"  A3 PASS: got {y.cpu()[0,0].item()}")
except Exception as e:
    msg = str(e)
    if 'NCC_ILLC059' in msg:
        print(f"  A3 FAIL: NCC_ILLC059")
    else:
        print(f"  A3 FAIL: {type(e).__name__}: {msg[:300]}")

