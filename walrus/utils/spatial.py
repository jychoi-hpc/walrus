"""Spatial (domain) parallelism: each sample's grid is split into slabs along
one axis, one slab per GPU of a spatial group. Every GPU in a group sees the
same batch; data parallelism runs across groups.

Layers that need to communicate look up the group with get_spatial_context().
A layer only communicates once enable_domain_split() has marked it: until the
model runs on slabs, every GPU of a group holds the whole sample.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
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


@dataclass(frozen=True)
class SlabLayout:
    """How `n_points` along the split axis are divided over `size` GPUs, with
    maps between global indices and (rank, local index)."""

    n_points: int
    size: int
    align: int = 1

    def __post_init__(self):
        slab_bounds(self.n_points, self.size, 0, self.align)  # validates

    def bounds(self, rank: int) -> Tuple[int, int]:
        return slab_bounds(self.n_points, self.size, rank, self.align)

    def all_bounds(self) -> List[Tuple[int, int]]:
        return [self.bounds(r) for r in range(self.size)]

    def owner(self, global_index: int) -> Tuple[int, int]:
        """(rank, local index) of the slab holding `global_index`."""
        if not 0 <= global_index < self.n_points:
            raise IndexError(f"index {global_index} outside 0..{self.n_points - 1}")
        for rank, (start, stop) in enumerate(self.all_bounds()):
            if global_index < stop:
                return rank, global_index - start
        raise AssertionError("unreachable")

    def to_global(self, rank: int, local_index: int) -> int:
        start, stop = self.bounds(rank)
        if not 0 <= local_index < stop - start:
            raise IndexError(f"local index {local_index} outside slab of rank {rank}")
        return start + local_index


def local_slab(
    x: torch.Tensor, dim: int, ctx: SpatialContext, align: int = 1
) -> torch.Tensor:
    """This GPU's slab of a full tensor `x` along `dim` (a view)."""
    start, stop = slab_bounds(x.shape[dim], ctx.size, ctx.rank, align)
    return x.narrow(dim, start, stop - start)


def gather_slabs(x_local: torch.Tensor, dim: int, ctx: SpatialContext) -> torch.Tensor:
    """Reassemble the full tensor from every GPU's slab along `dim` (slabs may
    differ in width). Not differentiable: for tests and diagnostics."""
    widths: List[Optional[int]] = [None] * ctx.size
    dist.all_gather_object(widths, x_local.shape[dim], group=ctx.group)
    pad_to = max(widths)
    padded = torch.zeros(
        *x_local.shape[:dim], pad_to, *x_local.shape[dim + 1 :],
        dtype=x_local.dtype, device=x_local.device,
    )
    padded.narrow(dim, 0, x_local.shape[dim]).copy_(x_local)
    parts = [torch.empty_like(padded) for _ in range(ctx.size)]
    dist.all_gather(parts, padded.contiguous(), group=ctx.group)
    return torch.cat([p.narrow(dim, 0, w) for p, w in zip(parts, widths)], dim=dim)


def slab_offset_and_total(n_local: int, ctx: SpatialContext) -> Tuple[int, int]:
    """Start of this rank's slab along the split axis and the axis's full
    length, from the local lengths of all slabs of the group."""
    local = torch.tensor([n_local], dtype=torch.long)
    if dist.get_backend(ctx.group) == "nccl":
        local = local.cuda()
    widths = [torch.empty_like(local) for _ in range(ctx.size)]
    dist.all_gather(widths, local, group=ctx.group)
    widths = [int(w) for w in widths]
    return sum(widths[: ctx.rank]), sum(widths)


def enable_domain_split(model: "torch.nn.Module", ctx: Optional[SpatialContext] = None):
    """Mark every layer of `model` that supports it (has a `spatial_ctx`
    attribute) to treat its input as this GPU's slab and communicate over the
    spatial group. Stored on the modules, so it also holds when gradient
    checkpointing re-runs the forward during backward. Returns the model."""
    ctx = ctx or get_spatial_context()
    if ctx is None:
        raise RuntimeError("no spatial context: configure distribution=spatial first")
    for module in model.modules():
        if hasattr(module, "spatial_ctx"):
            module.spatial_ctx = ctx
    return model


def check_patch_aligned(ctx: Optional[SpatialContext], length: int, kernels, strides,
                        paddings=None) -> None:
    """On a split domain, a strided (transposed) convolution stack runs on each
    slab without communication only if, along the split axis, every kernel
    equals its stride, there is no padding, and the slab holds whole patches.
    Raise otherwise (a halo exchange would be needed)."""
    if ctx is None or ctx.size == 1:
        return
    a = ctx.axis
    for k, s in zip(kernels, strides):
        if k[a] != s[a]:
            raise NotImplementedError(
                f"kernel {k[a]} != stride {s[a]} along the split axis needs a halo exchange"
            )
    for p in paddings or []:
        if p[a]:
            raise NotImplementedError("padding along the split axis needs a halo exchange")
    patch = 1
    for s in strides:
        patch *= s[a]
    if length % patch:
        raise ValueError(f"slab of {length} points is not a whole number of {patch}-point patches")
