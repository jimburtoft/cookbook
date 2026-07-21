"""Test each of the four ncc.* collective primitives under LNC=2 (single-rank
kernel, single-process). Determines whether all four hit NCC_ILLC059 or if only
some do.
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
    print(f"\n--- Test: {name} ---")
    try:
        kernel = kernel_fn[lnc]
        wrapped = wrap_nki(kernel)
        y = wrapped(*args)
        torch.neuron.synchronize()
        val = y.cpu().reshape(-1)[0].item()
        print(f"  {name}: PASS (got {val})")
        return True
    except Exception as e:
        msg = str(e)
        if 'NCC_ILLC059' in msg:
            print(f"  {name}: FAIL with NCC_ILLC059")
        else:
            print(f"  {name}: FAIL: {type(e).__name__}: {msg[:200]}")
        return False


# Reference: kernel with just DMA (works)
@nki.jit
def dma_only(x):
    src = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="src")
    dst = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="dst")
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=src, src=x)
    nisa.dma_copy(dst=dst, src=src)
    nisa.dma_copy(dst=out, src=dst)
    return out

@nki.jit
def with_all_reduce(x, rg):
    src = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="src")
    dst = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="dst")
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=src, src=x)
    ncc.all_reduce(dsts=[dst], srcs=[src], op=nl.add, replica_group=rg)
    nisa.dma_copy(dst=out, src=dst)
    return out

@nki.jit
def with_all_gather(x, rg, num_ranks: int):
    H, W = x.shape
    src = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="src")
    dst = nl.ndarray((H * num_ranks, W), dtype=x.dtype, buffer=nl.shared_hbm, name="dst")
    out = nl.ndarray((H * num_ranks, W), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=src, src=x)
    ncc.all_gather(dsts=[dst], srcs=[src], replica_group=rg, collective_dim=0)
    nisa.dma_copy(dst=out, src=dst)
    return out

@nki.jit
def with_reduce_scatter(x, rg, num_ranks: int):
    H, W = x.shape
    src = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="src")
    dst = nl.ndarray((H // num_ranks, W), dtype=x.dtype, buffer=nl.shared_hbm, name="dst")
    out = nl.ndarray((H // num_ranks, W), dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=src, src=x)
    ncc.reduce_scatter(dsts=[dst], srcs=[src], op=nl.add, replica_group=rg, collective_dim=0)
    nisa.dma_copy(dst=out, src=dst)
    return out

@nki.jit
def with_all_to_all(x, rg):
    src = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="src")
    dst = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="dst")
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=src, src=x)
    ncc.all_to_all(dsts=[dst], srcs=[src], replica_group=rg, collective_dim=0)
    nisa.dma_copy(dst=out, src=dst)
    return out


print("=" * 70)
print(f"LNC={os.environ['NEURON_LOGICAL_NC_CONFIG']} NKI_LNC_DEGREE={lnc}")
print(f"nki: {nki.__version__} torch_neuronx: {torch_neuronx.__version__}")
print("=" * 70)

x = torch.ones(128, 512, dtype=torch.float32, device=device) * 5.0
torch.neuron.synchronize()

run_kernel("Reference DMA-only (no collective)", dma_only, x)

rg = ReplicaGroup([[0]])
run_kernel("with_all_reduce single-rank", with_all_reduce, x, rg)
run_kernel("with_all_gather single-rank", with_all_gather, x, rg, 1)
run_kernel("with_reduce_scatter single-rank", with_reduce_scatter, x, rg, 1)
run_kernel("with_all_to_all single-rank", with_all_to_all, x, rg)

