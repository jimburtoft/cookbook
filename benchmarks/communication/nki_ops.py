"""NKI kernel dispatch for the Neuron collective benchmarks.

When ``--use-nki`` is passed on the command line, the ``timed_*`` functions in
each per-collective file replace ``dist.all_reduce()`` (and friends) with a
call to a cached wrapper around the corresponding NKI HBM kernel from
``nkilib.experimental.collectives.collectives``.

The wrapper:
  * imports the kernel lazily on first use (so plain ``--backend=neuron`` runs
    without ``--use-nki`` do not need the nkilib source on PYTHONPATH),
  * sets the SPMD launch grid on the HOP caller via ``wrap_nki(kernel)[lnc]``
    with LNC read from ``NKI_LNC_DEGREE`` (default 2 to match trn2 default),
  * builds a fresh ``ReplicaGroup`` covering the whole world size,
  * caches per-(collective, input_shape) so each NEFF is compiled once.

**Invocation nuance**: the LNC grid MUST be set on the HOP caller returned by
``wrap_nki``, not on the raw kernel object.  The wrong pattern
``wrap_nki(kernel[lnc])`` compiles as LNC=1 regardless of the ``[lnc]`` on the
kernel, because ``wrap_nki`` builds a fresh ``NKIHOPCaller`` with an empty
``grid`` field.  See ``get_wrapped`` docstring for the failure mode this
causes on LNC=2.

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

    Default 2 -- matches trn2 default LNC=2 (4 logical cores per chip).  Users
    running under NEURON_LOGICAL_NC_CONFIG=1 (8 logical cores per chip) should
    also set NKI_LNC_DEGREE=1 so the wrap_nki grid matches the runtime.
    """
    return int(os.environ.get("NKI_LNC_DEGREE", "2"))


def _to_kernel_shape(x: torch.Tensor, coll: str, world_size: int) -> torch.Tensor:
    """Reshape a 1-D benchmark tensor to the (H, W) form the kernel expects.

    All HBM kernels take rank-2 input.  H is fixed at 128 (fp32 partition).
    For reduce_scatter the input needs to be world_size times taller (so
    each rank's output chunk is the full 1-D size).

    Raises SkipSizeError if the tensor is too small or has an awkward shape
    for the NKI kernel; the caller should fall back to the framework path
    for that particular size.
    """
    numel = x.numel()
    if coll == "reduce_scatter":
        h = _NKI_PARTITION_ROWS * world_size
    else:
        h = _NKI_PARTITION_ROWS
    if numel < h or numel % h != 0:
        raise SkipSizeError(
            f"NKI path: numel={numel} not compatible with required H={h} "
            f"(coll={coll}, world_size={world_size})"
        )
    w = numel // h
    return x.view(h, w)


class SkipSizeError(ValueError):
    """Raised when a tensor shape is not compatible with the NKI HBM kernel.
    Callers should catch this and fall back to the framework path.
    """


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

    Invocation pattern is ``wrap_nki(kernel)[lnc](args, replica_group)``:
    the SPMD launch grid (LNC degree) is set on the ``NKIHOPCaller`` returned by
    ``wrap_nki``, not on the underlying kernel object.  Setting it on the kernel
    (``wrap_nki(kernel[lnc])``) is a no-op at the HOP layer -- the HOP caller
    stores its own ``grid`` list which is passed to the compiler via
    ``dump_config``.  This nuance is not called out anywhere in the public NKI
    docs; it is only visible in the source of ``torch_neuronx/nki_hop.py``.

    Getting this wrong is silent on LNC=1 (because the HOP caller's default
    grid==[] resolves to lnc=1, which happens to match) but fatal on LNC=2:
    the compiler generates SPMD instructions for LNC=1 while the runtime is
    LNC=2, and lowering fails with
    ``[NCC_ILLC059] Could not find MemoryLocation named inst__I-3-0:src on core 1``.
    """
    _lazy_init()
    if coll not in _KERNELS_BY_NAME:
        raise ValueError(f"No NKI kernel for collective={coll}")

    lnc = _lnc_degree()
    key = (coll, shape, str(dtype), world_size, lnc)
    if key in _WRAPPED_CACHE:
        return _WRAPPED_CACHE[key]

    # Correct pattern: wrap_nki(kernel)[lnc] -- LNC on the HOP caller.
    # See docstring above for why we do NOT use wrap_nki(kernel[lnc]).
    wrapped = _wrap_nki(_KERNELS_BY_NAME[coll])[lnc]
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
