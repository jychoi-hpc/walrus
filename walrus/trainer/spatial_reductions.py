"""Reductions over the whole domain when it is split into slabs across a
spatial group (tracker task t207): RevIN statistics and the training loss.

Means over space become sums over each slab, summed over the group, divided
by the total point count. Statistics carry no gradient (the trainer computes
them under no_grad), so a plain all-reduce is used there. The loss is not
communicated at all: each GPU weights its slab's mean by its share of the
points, so the GPUs' losses add up to the full-domain loss and each GPU's
backward gives exactly its part of the gradient.
"""

import math
from typing import Sequence

import torch
import torch.distributed as dist

from walrus.utils.spatial import SpatialContext


def global_mean(x: torch.Tensor, dims: Sequence[int], ctx: SpatialContext) -> torch.Tensor:
    """Mean of x over `dims` (keepdim) across all slabs of the group. Not
    differentiable."""
    total = x.sum(dim=tuple(dims), keepdim=True)
    count = torch.tensor(float(math.prod(x.shape[d] for d in dims)),
                         dtype=total.dtype, device=total.device)
    dist.all_reduce(total, group=ctx.group)
    dist.all_reduce(count, group=ctx.group)
    return total / count


def global_std_mean(x: torch.Tensor, dims: Sequence[int], ctx: SpatialContext,
                    correction: int = 1):
    """torch.std_mean(x, dims, keepdim=True) across all slabs (two passes:
    global mean, then global sum of squared deviations). Not differentiable."""
    mean = global_mean(x, dims, ctx)
    sq_dev = (x - mean).square().sum(dim=tuple(dims), keepdim=True)
    count = torch.tensor(float(math.prod(x.shape[d] for d in dims)),
                         dtype=sq_dev.dtype, device=sq_dev.device)
    dist.all_reduce(sq_dev, group=ctx.group)
    dist.all_reduce(count, group=ctx.group)
    return (sq_dev / (count - correction)).sqrt(), mean


def slab_share(n_local_points: int, ctx: SpatialContext, device=None) -> float:
    """This slab's fraction of the domain's points. `device` must suit the
    group's backend (a CUDA device for NCCL)."""
    total = torch.tensor(float(n_local_points), device=device)
    dist.all_reduce(total, group=ctx.group)
    return n_local_points / float(total)


def spatial_mean_loss(loss_fn, y_pred: torch.Tensor, y_ref: torch.Tensor, metadata,
                      ctx: SpatialContext, **kwargs) -> torch.Tensor:
    """This slab's contribution to a loss that is a mean over space (MAE, MSE):
    loss_fn on the slab, weighted by the slab's share of the points. Summed over
    the group it equals loss_fn on the full domain. Tensors are (..., *space, C).
    Not valid for losses that are nonlinear in the spatial mean (RMSE, NRMSE)."""
    n_space = metadata.n_spatial_dims
    n_local = math.prod(y_pred.shape[-n_space - 1 : -1])
    return loss_fn(y_pred, y_ref, metadata, **kwargs) * slab_share(n_local, ctx, y_pred.device)
