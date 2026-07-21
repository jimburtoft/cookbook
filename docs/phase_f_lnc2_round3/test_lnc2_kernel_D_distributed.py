"""Test kernel D (ncc.all_reduce) under torchrun with actual 4-rank replica group.
This mirrors the actual all_reduce_hbm_kernel usage.
"""
import os
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
def kernel_D_with_allreduce(x, replica_group):
    src = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="src")
    dst = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm, name="dst")
    out = nl.ndarray(x.shape, dtype=x.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=src, src=x)
    ncc.all_reduce(dsts=[dst], srcs=[src], op=nl.add, replica_group=replica_group)
    nisa.dma_copy(dst=out, src=dst)
    return out


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    lnc = int(os.environ.get("NKI_LNC_DEGREE", "2"))
    print(f"rank={rank}/{world_size} LNC={os.environ.get('NEURON_LOGICAL_NC_CONFIG')} NKI_LNC={lnc}", flush=True)

    dist.init_process_group(backend="neuron")
    device = torch.device("neuron")

    kernel = kernel_D_with_allreduce[lnc]
    wrapped = wrap_nki(kernel)

    x = torch.ones(128, 512, dtype=torch.float32, device=device) * float(rank + 1)
    torch.neuron.synchronize()

    rg = ReplicaGroup([list(range(world_size))])
    try:
        y = wrapped(x, rg)
        torch.neuron.synchronize()
        val = y.cpu()[0, 0].item()
        expected = sum(r+1 for r in range(world_size))
        if rank == 0:
            print(f"got: {val}, expected: {expected}", flush=True)
            print(f"Kernel D distributed: {'PASS' if abs(val - expected) < 1e-3 else 'WRONG VALUE'}", flush=True)
    except Exception as e:
        msg = str(e)
        if rank == 0:
            if 'NCC_ILLC059' in msg:
                print(f"Kernel D distributed: FAIL with NCC_ILLC059", flush=True)
            else:
                print(f"Kernel D distributed: FAIL: {type(e).__name__}: {msg[:400]}", flush=True)


if __name__ == "__main__":
    main()
