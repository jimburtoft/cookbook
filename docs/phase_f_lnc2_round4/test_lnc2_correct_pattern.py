"""Test the CORRECT wrap_nki pattern under distributed LNC=2 4-rank.

wrap_nki(kernel)[grid](args) -- grid goes on the HOP caller, not on the kernel.
"""
import os
import time
import torch
import torch_neuronx
from torch_neuronx import wrap_nki
import nki
import nki.language as nl
import nki.isa as nisa
import nki.collectives as ncc
from nki.collectives import ReplicaGroup
import torch.distributed as dist


@nki.jit
def all_reduce_kernel(x, rg):
    src = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="src")
    dst = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="dst")
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=src, src=x)
    ncc.all_reduce(dsts=[dst], srcs=[src], op=nl.add, replica_group=rg)
    nisa.dma_copy(dst=out, src=dst)
    return out


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    lnc = int(os.environ.get("NKI_LNC_DEGREE", "2"))
    print(f"rank={rank}/{world_size} LNC={os.environ.get('NEURON_LOGICAL_NC_CONFIG')} NKI_LNC={lnc}", flush=True)

    dist.init_process_group(backend="neuron")
    device = torch.device("neuron")

    # CORRECT INVOCATION: wrap the kernel, then use bracket syntax on the caller
    wrapped = wrap_nki(all_reduce_kernel)[lnc]
    rg = ReplicaGroup([list(range(world_size))])

    # Warmup + correctness check
    x = torch.ones(128, 512, dtype=torch.float32, device=device) * float(rank + 1)
    torch.neuron.synchronize()
    y = wrapped(x, rg)
    torch.neuron.synchronize()
    got = y.cpu()[0, 0].item()
    expected = float(sum(r + 1 for r in range(world_size)))
    if rank == 0:
        print(f"got: {got}, expected: {expected}", flush=True)
        print(f"CORRECTNESS: {'PASS' if abs(got - expected) < 1e-3 else 'FAIL'}", flush=True)

    # Simple timing (1 MB, 20 warm + 50 timed)
    N = 1024 * 1024 // 4
    H2, W2 = 128, N // 128
    x_time = torch.ones((H2, W2), dtype=torch.float32, device=device) * float(rank + 1)
    torch.neuron.synchronize(); dist.barrier()
    for _ in range(20):
        _ = wrapped(x_time, rg)
    torch.neuron.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    for _ in range(50):
        _ = wrapped(x_time, rg)
    torch.neuron.synchronize()
    nki_us = (time.perf_counter() - t0) / 50 * 1e6

    # Framework baseline (torch.distributed)
    x_fw = torch.ones(N, dtype=torch.float32, device=device) * float(rank + 1)
    torch.neuron.synchronize(); dist.barrier()
    for _ in range(20):
        dist.all_reduce(x_fw)
    torch.neuron.synchronize(); dist.barrier()
    t0 = time.perf_counter()
    for _ in range(50):
        dist.all_reduce(x_fw)
    torch.neuron.synchronize()
    fw_us = (time.perf_counter() - t0) / 50 * 1e6

    if rank == 0:
        print(f"\n1 MB all_reduce (LNC={lnc} WS={world_size}):", flush=True)
        print(f"  framework: {fw_us:.1f} us", flush=True)
        print(f"  NKI kernel: {nki_us:.1f} us", flush=True)
        print(f"  speedup: {fw_us/nki_us:.2f}x", flush=True)


if __name__ == "__main__":
    main()
