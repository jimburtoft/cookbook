"""Sweep comparison: torch.distributed collective vs wrap_nki(nki_hbm_kernel)
for each of {all_reduce, all_gather, reduce_scatter, all_to_all}.

Runs under torchrun on a single node. Both paths use identical:
  * device=torch.device("neuron")
  * float32 tensors filled with rank+1
  * torch.neuron.synchronize() gated timer (perf_counter with pre-sync)
  * 50 warmup + 100 timed iterations per size
  * 2D tensor of shape (H, W) where H*W matches the target byte count

Config comes from env: NPROC, NKI_LNC_DEGREE, COLL.
"""
import argparse
import os
import time
import socket

import torch
import torch_neuronx  # noqa: F401 -- registers 'neuron' backend + device
from torch_neuronx import wrap_nki
from nki.collectives import ReplicaGroup

from nkilib_src.nkilib.experimental.collectives.collectives import (
    all_reduce_hbm_kernel,
    all_gather_hbm_kernel,
    reduce_scatter_hbm_kernel,
    all_to_all_hbm_kernel,
)

import torch.distributed as dist


COLLECTIVE_MAP = {
    "all_reduce": all_reduce_hbm_kernel,
    "all_gather": all_gather_hbm_kernel,
    "reduce_scatter": reduce_scatter_hbm_kernel,
    "all_to_all": all_to_all_hbm_kernel,
}


def _sync():
    torch.neuron.synchronize()


def _framework_call(coll, tensor, world_size):
    """Invoke the framework's torch.distributed version of `coll`."""
    if coll == "all_reduce":
        dist.all_reduce(tensor)
        return tensor
    if coll == "all_gather":
        # NOTE: the fastest form is all_gather_into_tensor (used by our port)
        out = torch.empty(tensor.numel() * world_size, dtype=tensor.dtype, device=tensor.device)
        dist.all_gather_into_tensor(out, tensor.view(-1))
        return out
    if coll == "reduce_scatter":
        # Input is (world_size * H, W). Framework wants a flat view of size N*ws
        # scattered into a chunk of size N.
        n = tensor.numel() // world_size
        inp_flat = tensor.reshape(-1)
        out = torch.empty(n, dtype=tensor.dtype, device=tensor.device)
        dist.reduce_scatter_tensor(out, inp_flat)
        return out
    if coll == "all_to_all":
        # Framework's dist.all_to_all_single is the fastest form
        out = torch.empty_like(tensor)
        dist.all_to_all_single(out, tensor)
        return out
    raise ValueError(coll)


def _make_2d_tensor(numel_per_rank: int, rank: int, device):
    """Return (H, W) tensor with numel exactly numel_per_rank.

    NKI kernels require rank-2 input. We pick H, W = 128, numel/128 so H
    stays constant and W varies with size. The Partition dimension of NKI
    is 128 for fp32.
    """
    H = 128
    if numel_per_rank < H:
        # Kernels are unhappy with H > numel; skip this size upstream instead
        raise ValueError(f"numel_per_rank={numel_per_rank} < H=128 -- caller should filter")
    W = numel_per_rank // H
    if H * W != numel_per_rank:
        raise ValueError(f"numel_per_rank={numel_per_rank} not divisible by H=128")
    return torch.ones((H, W), dtype=torch.float32, device=device) * float(rank + 1)


def _make_input_for_coll(coll: str, per_rank_elem: int, world_size: int, rank: int, device):
    """Return the input tensor for the NKI kernel, in the shape the kernel expects.

    The HBM kernels expect:
      * all_reduce, all_to_all: input shape (H, W) with H % world_size not required
      * all_gather: input (H, W), output (H*ws, W)
      * reduce_scatter: input (H*ws, W), output (H, W). So we pass world_size larger H.

    For a fair per-rank byte comparison we pin *output-byte* size, matching
    the cookbook framework's convention (per-rank input in ByPass sender's
    frame). Actually the cookbook's `size` variable = sender's input bytes.
    So we hold input_numel constant across all collectives.
    """
    if coll in ("all_reduce", "all_gather", "all_to_all"):
        return _make_2d_tensor(per_rank_elem, rank, device)
    if coll == "reduce_scatter":
        # reduce_scatter input is world_size larger than output
        return _make_2d_tensor(per_rank_elem * world_size, rank, device)
    raise ValueError(coll)


def _wrap_kernel_for_size(coll: str, lnc_degree: int, world_size: int):
    """Return a wrapped NKI kernel + kwargs for the collective.

    Correct pattern: wrap_nki(kernel)[lnc_degree] -- LNC on the HOP caller,
    NOT on the raw kernel object. See benchmarks/communication/nki_ops.py
    for the failure mode of the alternative wrap_nki(kernel[lnc_degree]).
    """
    kernel_fn = COLLECTIVE_MAP[coll]
    wrapped = wrap_nki(kernel_fn)[lnc_degree]
    replica_group = ReplicaGroup([list(range(world_size))])
    if coll == "all_reduce" or coll == "all_to_all":
        def call(inp):
            return wrapped(inp, replica_group)
    else:
        # all_gather + reduce_scatter both need num_ranks
        def call(inp):
            return wrapped(inp, replica_group, world_size)
    return call


def _cookbook_bw(coll: str, size_bytes: int, duration_s: float, world_size: int, bw_unit="GBps"):
    """Match the cookbook's bandwidth formulas exactly."""
    n = world_size
    if coll == "all_to_all":
        tput = size_bytes / duration_s
        busbw = tput * (n - 1) / n
    elif coll == "all_gather":
        size_bytes *= n
        tput = size_bytes / duration_s
        busbw = tput * (n - 1) / n
    elif coll == "all_reduce":
        tput = size_bytes * 2 / duration_s
        busbw = size_bytes / duration_s * 2 * (n - 1) / n
    elif coll == "reduce_scatter":
        tput = size_bytes / duration_s
        busbw = tput * (n - 1) / n
    else:
        tput = busbw = 0.0
    if bw_unit == "Gbps":
        tput *= 8
        busbw *= 8
    return tput, busbw


def run_size(coll: str, per_rank_elem: int, world_size: int, rank: int, device,
             lnc_degree: int, warmups=20, trials=50):
    """Return (framework_us, nki_us, size_bytes)."""
    # --- Prepare tensors ---
    try:
        inp_fw = _make_input_for_coll(coll, per_rank_elem, world_size, rank, device)
        inp_nki = _make_input_for_coll(coll, per_rank_elem, world_size, rank, device)
    except ValueError:
        return None, None, None

    size_bytes = inp_fw.element_size() * inp_fw.numel()
    _sync(); dist.barrier()

    # --- Framework path ---
    try:
        for _ in range(warmups):
            _ = _framework_call(coll, inp_fw, world_size)
        _sync(); dist.barrier()

        t0 = time.perf_counter()
        for _ in range(trials):
            _ = _framework_call(coll, inp_fw, world_size)
        _sync()
        fw_us = (time.perf_counter() - t0) / trials * 1e6
    except Exception as e:
        fw_us = None
        if rank == 0:
            print(f"    framework {coll} failed: {type(e).__name__}: {e}", flush=True)

    # --- NKI kernel path ---
    try:
        call = _wrap_kernel_for_size(coll, lnc_degree, world_size)
        for _ in range(warmups):
            _ = call(inp_nki)
        _sync(); dist.barrier()

        t0 = time.perf_counter()
        for _ in range(trials):
            _ = call(inp_nki)
        _sync()
        nki_us = (time.perf_counter() - t0) / trials * 1e6
    except Exception as e:
        nki_us = None
        if rank == 0:
            print(f"    NKI {coll} failed: {type(e).__name__}: {e}", flush=True)

    return fw_us, nki_us, size_bytes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coll", type=str, required=True, choices=list(COLLECTIVE_MAP.keys()))
    p.add_argument("--maxsize", type=int, default=24, help="max size as power of 2 in bytes")
    p.add_argument("--minsize", type=int, default=9, help="min size as power of 2 in bytes (must be >= 9 for H=128 fp32)")
    p.add_argument("--warmups", type=int, default=20)
    p.add_argument("--trials", type=int, default=50)
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    lnc_degree = int(os.environ.get("NKI_LNC_DEGREE", "1"))
    if rank == 0:
        print(f"=== coll={args.coll} WS={world_size} LNC={os.environ.get('NEURON_LOGICAL_NC_CONFIG','?')} "
              f"NKI_LNC={lnc_degree} host={socket.gethostname()} ===", flush=True)

    dist.init_process_group(backend="neuron")
    device = torch.device("neuron")

    # Cookbook prints: size_bytes duration_us fw_tput fw_busbw nki_tput nki_busbw speedup
    if rank == 0:
        print(f"{'size_bytes':>12} {'per_rank_elem':>14} {'fw_us':>10} {'nki_us':>10} {'fw_tput_GBps':>13} {'nki_tput_GBps':>14} {'fw_busbw_GBps':>14} {'nki_busbw_GBps':>15} {'speedup':>8}",
              flush=True)

    # Element sizes: for fp32 (4 B), min per_rank_elem is 128 for H=128 partition.
    for log2 in range(args.minsize, args.maxsize + 1):
        per_rank_elem = (1 << log2) // 4  # 4 bytes per fp32 element
        if per_rank_elem < 128:
            continue
        if per_rank_elem % 128 != 0:
            continue

        fw_us, nki_us, size_b = run_size(
            args.coll, per_rank_elem, world_size, rank, device,
            lnc_degree=lnc_degree, warmups=args.warmups, trials=args.trials,
        )

        if rank == 0:
            fw_str = f"{fw_us:.1f}" if fw_us else "FAIL"
            nki_str = f"{nki_us:.1f}" if nki_us else "FAIL"
            fw_tput = fw_busbw = nki_tput = nki_busbw = 0.0
            if fw_us:
                fw_tput, fw_busbw = _cookbook_bw(args.coll, size_b, fw_us / 1e6, world_size)
                fw_tput /= 1e9
                fw_busbw /= 1e9
            if nki_us:
                nki_tput, nki_busbw = _cookbook_bw(args.coll, size_b, nki_us / 1e6, world_size)
                nki_tput /= 1e9
                nki_busbw /= 1e9
            speedup = (fw_us / nki_us) if (fw_us and nki_us) else 0.0
            print(f"{size_b:>12} {per_rank_elem:>14} {fw_str:>10} {nki_str:>10} "
                  f"{fw_tput:>13.3f} {nki_tput:>14.3f} {fw_busbw:>14.3f} {nki_busbw:>15.3f} {speedup:>8.2f}",
                  flush=True)


if __name__ == "__main__":
    main()
