"""Minimal smoke test: run NKI all_reduce_hbm_kernel across 4 ranks
on a single trainium node, compare vs torch.distributed.all_reduce.

Kernel launch grid is set to LNC_DEGREE=2 to match trn2 LNC=2 mode.
"""
import os
import time
import socket

import torch
import torch_neuronx  # noqa: F401
from torch_neuronx import wrap_nki
from nki.collectives import ReplicaGroup
from nkilib_src.nkilib.experimental.collectives.collectives import all_reduce_hbm_kernel

import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    lnc_degree = int(os.environ.get("NKI_LNC_DEGREE", os.environ.get("NEURON_LOGICAL_NC_CONFIG", "1")))
    print(f"rank={rank}/{world_size} local_rank={local_rank} lnc={lnc_degree} host={socket.gethostname()}", flush=True)

    dist.init_process_group(backend="neuron")
    device = torch.device("neuron")

    H, W = 128, 512
    x_local = torch.ones(H, W, dtype=torch.float32, device=device) * (rank + 1.0)
    torch.neuron.synchronize()

    # --- Framework path ---
    y_fw = x_local.clone()
    dist.all_reduce(y_fw)
    torch.neuron.synchronize()
    y_fw_cpu = y_fw.cpu()

    # --- NKI kernel path: wrap_nki(kernel)[lnc_degree] sets the SPMD launch grid ---
    # NOTE: the LNC grid MUST be set on the HOP caller (result of wrap_nki),
    # not on the raw kernel object. wrap_nki(kernel[lnc]) is a silent no-op
    # at the HOP layer -- see benchmarks/communication/nki_ops.py for the
    # underlying failure mode this causes on LNC=2.
    replica_group = ReplicaGroup([list(range(world_size))])
    wrapped = wrap_nki(all_reduce_hbm_kernel)[lnc_degree]

    x_nki = torch.ones(H, W, dtype=torch.float32, device=device) * (rank + 1.0)
    torch.neuron.synchronize()
    y_nki = wrapped(x_nki, replica_group)
    torch.neuron.synchronize()
    y_nki_cpu = y_nki.cpu() if isinstance(y_nki, torch.Tensor) else y_nki[0].cpu()

    expected = float(sum(r + 1 for r in range(world_size)))
    got_fw = y_fw_cpu[0, 0].item()
    got_nki = y_nki_cpu[0, 0].item()

    if rank == 0:
        print(f"expected={expected} framework={got_fw} nki={got_nki}", flush=True)
        assert abs(got_fw - expected) < 1e-3, "framework result wrong"
        assert abs(got_nki - expected) < 1e-3, "nki result wrong"
        print("SMOKE TEST PASSED", flush=True)

    # 1 MB fp32
    N = 1024 * 1024 // 4
    x_time = torch.ones(N, dtype=torch.float32, device=device)
    torch.neuron.synchronize(); dist.barrier()

    for _ in range(20):
        dist.all_reduce(x_time)
    torch.neuron.synchronize(); dist.barrier()

    t0 = time.perf_counter()
    for _ in range(50):
        dist.all_reduce(x_time)
    torch.neuron.synchronize()
    t_fw = (time.perf_counter() - t0) / 50 * 1e6

    H2, W2 = 512, N // 512
    x_time_2d = torch.ones(H2, W2, dtype=torch.float32, device=device)
    torch.neuron.synchronize(); dist.barrier()

    for _ in range(20):
        _ = wrapped(x_time_2d, replica_group)
    torch.neuron.synchronize(); dist.barrier()

    t0 = time.perf_counter()
    for _ in range(50):
        _ = wrapped(x_time_2d, replica_group)
    torch.neuron.synchronize()
    t_nki = (time.perf_counter() - t0) / 50 * 1e6

    if rank == 0:
        print(f"1 MB all_reduce  framework={t_fw:.1f} us  NKI-kernel={t_nki:.1f} us  speedup={t_fw/t_nki:.2f}x", flush=True)


if __name__ == "__main__":
    main()
