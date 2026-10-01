"""Differentiable halo exchange along the split axis (tracker task t202).

halo_exchange(x, dim, width, ctx, periodic) returns this GPU's slab with
`width` layers of each neighbor's slab attached on both sides, so a layer
with a window wider than one point (a convolution, a roll) can run on the
slab as on the full domain. At the domain's two ends the halo is the other
end of the domain when the axis is periodic, and zeros otherwise (zero
padding of the full domain).

Backward sends each halo's gradient back to the GPU that owns those points,
where it is added to the gradient of its edge layers.

Messages go in a fixed order - rightward first, then leftward - on every
rank, so they also match with only two GPUs (left and right neighbor are the
same rank) and with backends that match messages by order (NCCL).
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist

from walrus.utils.spatial import SpatialContext


def _neighbors(ctx: SpatialContext, periodic: bool) -> Tuple[Optional[int], Optional[int]]:
    """Group ranks of the left and right neighbor (None at a wall)."""
    left, right = ctx.rank - 1, ctx.rank + 1
    if periodic:
        return left % ctx.size, right % ctx.size
    return (left if left >= 0 else None), (right if right < ctx.size else None)


def _exchange(ctx: SpatialContext, to_right: torch.Tensor, to_left: torch.Tensor,
              periodic: bool) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Send to_right to the right neighbor and to_left to the left neighbor;
    return (from left neighbor, from right neighbor), None where there is none."""
    left, right = _neighbors(ctx, periodic)
    if ctx.size == 1:  # periodic with one slab: the neighbor is itself
        return (to_right, to_left) if periodic else (None, None)
    peer = lambda r: dist.get_global_rank(ctx.group, r)  # noqa: E731
    from_left = torch.empty_like(to_right) if left is not None else None
    from_right = torch.empty_like(to_left) if right is not None else None
    ops = []
    # Rightward stream: my right edge -> right neighbor's left halo
    if right is not None:
        ops.append(dist.P2POp(dist.isend, to_right.contiguous(), peer(right), ctx.group, tag=0))
    if left is not None:
        ops.append(dist.P2POp(dist.irecv, from_left, peer(left), ctx.group, tag=0))
    # Leftward stream: my left edge -> left neighbor's right halo
    if left is not None:
        ops.append(dist.P2POp(dist.isend, to_left.contiguous(), peer(left), ctx.group, tag=1))
    if right is not None:
        ops.append(dist.P2POp(dist.irecv, from_right, peer(right), ctx.group, tag=1))
    for request in dist.batch_isend_irecv(ops):
        request.wait()
    return from_left, from_right


class _HaloExchange(torch.autograd.Function):
    @staticmethod
    def forward(fctx, x, dim, width, ctx, periodic):
        fctx.dim, fctx.width, fctx.ctx, fctx.periodic = dim, width, ctx, periodic
        n = x.shape[dim]
        from_left, from_right = _exchange(
            ctx, x.narrow(dim, n - width, width), x.narrow(dim, 0, width), periodic
        )
        zeros = lambda: torch.zeros_like(x.narrow(dim, 0, width))  # noqa: E731
        left_halo = from_left if from_left is not None else zeros()
        right_halo = from_right if from_right is not None else zeros()
        return torch.cat([left_halo, x, right_halo], dim=dim)

    @staticmethod
    def backward(fctx, grad):
        dim, width = fctx.dim, fctx.width
        n = grad.shape[dim] - 2 * width
        grad_x = grad.narrow(dim, width, n).clone()
        # A halo's gradient belongs to the neighbor that owns those points:
        # my right halo is my right neighbor's left edge, and vice versa.
        from_left, from_right = _exchange(
            fctx.ctx,
            grad.narrow(dim, width + n, width),  # right halo -> right neighbor
            grad.narrow(dim, 0, width),  # left halo -> left neighbor
            fctx.periodic,
        )
        if from_left is not None:  # left neighbor's right halo = my left edge
            grad_x.narrow(dim, 0, width).add_(from_left)
        if from_right is not None:  # right neighbor's left halo = my right edge
            grad_x.narrow(dim, n - width, width).add_(from_right)
        return grad_x, None, None, None, None


def halo_exchange(x: torch.Tensor, dim: int, width: int, ctx: SpatialContext,
                  periodic: bool) -> torch.Tensor:
    """x with `width` layers from each neighbor's slab attached on both sides
    of `dim` (zeros beyond a non-periodic domain end). Differentiable. Every
    slab must be at least `width` wide."""
    if width == 0:
        return x
    if x.shape[dim] < width:
        raise ValueError(f"slab of {x.shape[dim]} points is narrower than the halo {width}")
    return _HaloExchange.apply(x, dim, width, ctx, periodic)
