"""NKI kernel dispatch for the Neuron collective benchmarks.

When ``--use-nki`` is passed on the command line, the ``timed_*`` functions in
each per-collective file replace ``dist.all_reduce()`` (and friends) with a
call to a cached wrapper around the corresponding NKI HBM kernel from
``nkilib.experimental.collectives.collectives``.

The wrapper:
  * imports the kernel lazily on first use (so plain ``--backend=neuron`` runs
    without ``--use-nki`` do not need the nkilib source on PYTHONPATH),
  * sets the SPMD launch grid via ``kernel[LNC]`` where LNC is read from
    ``NKI_LNC_DEGREE`` (default 1),
  * builds a fresh ``ReplicaGroup`` covering the whole world size,
  * caches per-(collective, input_shape) so each NEFF is compiled once.

We do NOT try to make the NKI path shape-compatible with the framework's
1-D ``torch.ones(N)`` input.  Instead we reshape at call time to the 2-D form
the HBM kernels expect (H, W) with H fixed at 128 (the fp32 partition dim).
This mirrors the pattern used in nki-library tests and in the smoke test that
proved the 7-10x speedup at WS=8 LNC=1.

Only ``all_reduce, all_gather, reduce_scatter, all_to_all`` are supported.
``broadcast`` and ``pt2pt`` have no HBM-kernel counterpart in nkilib 2.31 and
fall through to the framework path.
"""
from __future__ import annotations

import os
from typing import Callable

import torch

# Deferred imports: nki / torch_neuronx / nkilib_src are heavy and are only
# needed when --use-nki is actually set.
_wrap_nki = None
_ReplicaGroup = None
_KERNELS_BY_NAME: dict[str, Callable] = {}
_WRAPPED_CACHE: dict[tuple, Callable] = {}


# NKI HBM kernels partition on 128 rows for fp32; if the tensor is not a
# multiple of 128 elements we can't hand it to the kernel.
_NKI_PARTITION_ROWS = 128


def _lazy_init() -> None:
    """Import wrap_nki, ReplicaGroup, and the four HBM kernels on first use."""
    global _wrap_nki, _ReplicaGroup, _KERNELS_BY_NAME
    if _wrap_nki is not None:
        return
    from torch_neuronx import wrap_nki
    from nki.collectives import ReplicaGroup
    try:
        from nkilib.experimental.collectives.collectives import (  # type: ignore
            all_reduce_hbm_kernel,
            all_gather_hbm_kernel,
            reduce_scatter_hbm_kernel,
            all_to_all_hbm_kernel,
        )
    except ImportError:
        # Fall back to the source layout used by the nki-library git repo
        # (in case the user set PYTHONPATH to point at src/ rather than
        # installing the wheel).
        from nkilib_src.nkilib.experimental.collectives.collectives import (  # type: ignore
            all_reduce_hbm_kernel,
            all_gather_hbm_kernel,
            reduce_scatter_hbm_kernel,
            all_to_all_hbm_kernel,
        )
    _wrap_nki = wrap_nki
    _ReplicaGroup = ReplicaGroup
    _KERNELS_BY_NAME = {
        "all_reduce": all_reduce_hbm_kernel,
        "all_gather": all_gather_hbm_kernel,
        "reduce_scatter": reduce_scatter_hbm_kernel,
        "all_to_all": all_to_all_hbm_kernel,
    }


def _lnc_degree() -> int:
    """Read the SPMD launch grid degree from the environment.

    Default 1.  Matches Beta 3 on trn2 when running under
    NEURON_LOGICAL_NC_CONFIG=1 (8 logical cores).  LNC=2 currently fails at
    kernel compile time with NCC_ILLC059 -- see docs/phase_e_nki_report.md
    for the details.  Once that ticket is resolved, LNC=2 will Just Work by
    setting NKI_LNC_DEGREE=2 in the env.
    """
    return int(os.environ.get("NKI_LNC_DEGREE", "1"))


def _to_kernel_shape(x: torch.Tensor, coll: str, world_size: int) -> torch.Tensor:
    """Reshape a 1-D benchmark tensor to the (H, W) form the kernel expects.

    All HBM kernels take rank-2 input.  H is fixed at 128 (fp32 partition).
    For reduce_scatter the input needs to be world_size times taller (so
    each rank's output chunk is the full 1-D size).
    """
    numel = x.numel()
    if coll == "reduce_scatter":
        h = _NKI_PARTITION_ROWS * world_size
    else:
        h = _NKI_PARTITION_ROWS
    if numel % h != 0:
        raise ValueError(
            f"NKI path: numel={numel} not divisible by required H={h} "
            f"(coll={coll}, world_size={world_size}). Skip this size or "
            f"round up the benchmark tensor."
        )
    w = numel // h
    return x.view(h, w)


def is_nki_supported(coll: str, dtype: torch.dtype) -> bool:
    """True iff we have an HBM kernel for this collective and dtype."""
    if coll not in {"all_reduce", "all_gather", "reduce_scatter", "all_to_all"}:
        return False
    # HBM kernels are compiled per-dtype; nki-library 2.31 exposes them only
    # for float32 by default.  bf16 support exists internally but is not on
    # the export surface for these particular kernels.
    return dtype == torch.float32


def get_wrapped(coll: str, shape: tuple, dtype: torch.dtype, world_size: int) -> Callable:
    """Return a (cached) callable that runs the NKI kernel for `coll` on tensors
    with `shape` and `dtype` under a full-world replica group.
    """
    _lazy_init()
    if coll not in _KERNELS_BY_NAME:
        raise ValueError(f"No NKI kernel for collective={coll}")

    lnc = _lnc_degree()
    key = (coll, shape, str(dtype), world_size, lnc)
    if key in _WRAPPED_CACHE:
        return _WRAPPED_CACHE[key]

    kernel = _KERNELS_BY_NAME[coll][lnc]
    wrapped = _wrap_nki(kernel)
    replica_group = _ReplicaGroup([list(range(world_size))])

    if coll in ("all_reduce", "all_to_all"):
        def call(inp: torch.Tensor):
            return wrapped(inp, replica_group)
    else:
        # all_gather + reduce_scatter both need num_ranks as a third arg
        def call(inp: torch.Tensor):
            return wrapped(inp, replica_group, world_size)

    _WRAPPED_CACHE[key] = call
    return call


def dispatch_call(coll: str, tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Run the NKI-kernel version of `coll` on `tensor` and return the result.

    The benchmark files call us in place of ``dist.<coll>(tensor)``.  Because
    the HBM kernels are out-of-place we return the kernel's output tensor;
    the caller is expected to not rely on in-place semantics for the timed
    loop (which is true for the cookbook -- it just needs *some* execution
    to happen).
    """
    reshaped = _to_kernel_shape(tensor, coll, world_size)
    call = get_wrapped(coll, tuple(reshaped.shape), reshaped.dtype, world_size)
    return call(reshaped)
