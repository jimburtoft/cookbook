"""Progressively strip down the 2.31 all_reduce_hbm_kernel to find the minimal
reproducer for NCC_ILLC059 under LNC=2.

We test 4 kernels:
  A. Trivial copy (baseline: PASSES LNC=2 in test 1 of previous run)
  B. Same as A but with name="src" -- tests whether naming triggers the bug
  C. A + name="src"/name="dst" -- 2 named ndarrays (like the failing kernel)
  D. C + ncc.all_reduce call -- add the actual collective primitive
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
import traceback

lnc = int(os.environ.get("NKI_LNC_DEGREE", "2"))
device = torch.device("neuron")

def run_kernel(name, kernel_fn, *args):
    print(f"\n--- Kernel {name} ---")
    try:
        kernel = kernel_fn[lnc]
        wrapped = wrap_nki(kernel)
        y = wrapped(*args)
        torch.neuron.synchronize()
        val = y.cpu()[0, 0].item()
        print(f"  {name}: PASS (got {val})")
        return True
    except Exception as e:
        msg = str(e)
        if 'NCC_ILLC059' in msg:
            print(f"  {name}: FAIL with NCC_ILLC059 (compiler bug)")
        else:
            print(f"  {name}: FAIL: {type(e).__name__}: {msg[:200]}")
        return False

# ------------------------------------------------------------------
# Kernel A: trivial copy, no naming
# ------------------------------------------------------------------
@nki.jit
def kernel_A_trivial(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=out, src=x)
    return out

# ------------------------------------------------------------------
# Kernel B: same as A but with name="out" -- tests naming alone
# ------------------------------------------------------------------
@nki.jit
def kernel_B_named_out(x):
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="out")
    nisa.dma_copy(dst=out, src=x)
    return out

# ------------------------------------------------------------------
# Kernel C: two named intermediate buffers, no collective
# ------------------------------------------------------------------
@nki.jit
def kernel_C_two_named(x):
    src = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="src")
    dst = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="dst")
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=src, src=x)
    nisa.dma_copy(dst=dst, src=src)
    nisa.dma_copy(dst=out, src=dst)
    return out

# ------------------------------------------------------------------
# Kernel D: C + ncc.all_reduce -- adds the collective primitive
# ------------------------------------------------------------------
@nki.jit
def kernel_D_with_allreduce(x, replica_group):
    src = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="src")
    dst = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="dst")
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=src, src=x)
    ncc.all_reduce(dsts=[dst], srcs=[src], op=nl.add, replica_group=replica_group)
    nisa.dma_copy(dst=out, src=dst)
    return out

print("=" * 70)
print(f"LNC={os.environ['NEURON_LOGICAL_NC_CONFIG']} NKI_LNC_DEGREE={lnc}")
print(f"nki: {nki.__version__} torch_neuronx: {torch_neuronx.__version__}")
print("=" * 70)

x = torch.ones(128, 512, dtype=torch.float32, device=device) * 4.0
torch.neuron.synchronize()

run_kernel("A (trivial copy)", kernel_A_trivial, x)
run_kernel("B (named out)", kernel_B_named_out, x)
run_kernel("C (two named + intermediate)", kernel_C_two_named, x)
rg = ReplicaGroup([[0]])
run_kernel("D (C + ncc.all_reduce, single rank)", kernel_D_with_allreduce, x, rg)
