"""Spatial (domain) parallelism: each sample's grid is split into slabs along
one axis, one slab per GPU of a spatial group. Every GPU in a group sees the
same batch; data parallelism runs across groups.

Layers that need to communicate look up the group with get_spatial_context().
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

logger = logging.getLogger(__name__)

SPATIAL_DIM = "spatial"
DATA_PARALLEL_DIM = "dp"


@dataclass(frozen=True)
class SpatialContext:
    group: dist.ProcessGroup
    size: int  # GPUs per spatial group
    rank: int  # this GPU's position in its group (which slab it owns)
    axis: int  # spatial axis that is split (0 = x)
    dp_size: int  # number of spatial groups (data-parallel replicas)
    dp_rank: int  # which group this GPU belongs to


_context: Optional[SpatialContext] = None


def set_spatial_context(mesh: DeviceMesh, axis: int = 0) -> SpatialContext:
    global _context
    spatial, dp = mesh[SPATIAL_DIM], mesh[DATA_PARALLEL_DIM]
    _context = SpatialContext(
        group=spatial.get_group(),
        size=spatial.size(),
        rank=spatial.get_local_rank(),
        axis=axis,
        dp_size=dp.size(),
        dp_rank=dp.get_local_rank(),
    )
    logger.info(
        f"Spatial group: slab {_context.rank}/{_context.size} along axis {axis}, "
        f"data-parallel replica {_context.dp_rank}/{_context.dp_size}"
    )
    return _context


def get_spatial_context() -> Optional[SpatialContext]:
    """The active spatial context, or None when spatial parallelism is off."""
    return _context


def clear_spatial_context() -> None:
    global _context
    _context = None


def slab_bounds(n_points: int, size: int, rank: int, align: int = 1) -> Tuple[int, int]:
    """[start, stop) of `rank`'s slab when `n_points` are split into `size`
    contiguous slabs whose edges fall on multiples of `align` (the patch size),
    so no encoder window straddles two GPUs. Slabs differ by at most one
    `align` block when the blocks don't divide evenly."""
    if n_points % align != 0:
        raise ValueError(f"{n_points} points are not a multiple of the patch size {align}")
    blocks = n_points // align
    if blocks < size:
        raise ValueError(
            f"{n_points} points give {blocks} patch blocks, fewer than {size} GPUs"
        )
    base, extra = divmod(blocks, size)
    start = (rank * base + min(rank, extra)) * align
    stop = start + (base + (1 if rank < extra else 0)) * align
    return start, stop
