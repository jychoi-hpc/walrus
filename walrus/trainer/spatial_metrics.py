"""Validation metrics over the whole domain when it is split into slabs across
a spatial group (tracker task t303b).

the_well's spatial metrics reduce over the spatial dims to one value per
(batch, time, channel). On a slab each GPU only sees part of the domain, so
the metrics are rebuilt from sums over each slab, summed over the group (and a
maximum for L-infinity). Every GPU of the group gets the full-domain value -
the same formulas as the_well, up to the order of summation. No gradients.
"""

import torch
import torch.distributed as dist
from the_well.benchmark.metrics import (
    MAE,
    MSE,
    NMAE,
    NMSE,
    NRMSE,
    RMSE,
    VMSE,
    VRMSE,
    LInfinity,
    PearsonR,
)

from walrus.trainer.spatial_reductions import global_mean, global_std_mean
from walrus.utils.spatial import SpatialContext


def _dims(x: torch.Tensor, meta) -> tuple:
    """Spatial dims of (..., *space, C) tensors, as non-negative indices."""
    n = meta.n_spatial_dims
    return tuple(range(x.dim() - n - 1, x.dim() - 1))


def _mean(t, dims, ctx):
    return global_mean(t, dims, ctx).squeeze(dims)


def _mse(x, y, dims, ctx):
    return _mean((x - y) ** 2, dims, ctx)


def _nmse(x, y, dims, ctx, eps, norm_mode):
    if norm_mode == "norm":
        norm = _mean(y**2, dims, ctx)
    else:  # "std": torch.std (unbiased) squared
        norm = global_std_mean(y, dims, ctx)[0].squeeze(dims) ** 2
    return _mse(x, y, dims, ctx) / (norm + eps)


@torch.no_grad()
def spatial_metric(metric, x: torch.Tensor, y: torch.Tensor, meta, ctx: SpatialContext,
                   eps: float = 1e-7) -> torch.Tensor:
    """metric(x, y, meta, eps=eps) of the_well over the whole split domain,
    for x, y holding this GPU's slab, (B, T, *space, C). Computed one time
    step at a time: metrics reduce over space only, so the values are the
    same, and the temporaries (e.g. (x - y)^2) are one step in size instead
    of a whole rollout."""
    if x.dim() == meta.n_spatial_dims + 3 and x.shape[1] > 1:
        return torch.stack([_spatial_metric(metric, x[:, t:t + 1], y[:, t:t + 1], meta, ctx, eps)
                            for t in range(x.shape[1])], dim=1).squeeze(2)
    return _spatial_metric(metric, x, y, meta, ctx, eps)


def _spatial_metric(metric, x, y, meta, ctx, eps):
    dims = _dims(x, meta)
    kind = type(metric) if not isinstance(metric, type) else metric
    if kind is MSE:
        return _mse(x, y, dims, ctx)
    if kind is MAE:
        return _mean((x - y).abs(), dims, ctx)
    if kind is NMAE:
        return _mean((x - y).abs(), dims, ctx) / (_mean(y.abs(), dims, ctx) + eps)
    if kind is RMSE:
        return torch.sqrt(_mse(x, y, dims, ctx))
    if kind is NMSE:
        return _nmse(x, y, dims, ctx, eps, "norm")
    if kind is NRMSE:
        return torch.sqrt(_nmse(x, y, dims, ctx, eps, "norm"))
    if kind is VMSE:
        return _nmse(x, y, dims, ctx, eps, "std")
    if kind is VRMSE:
        return torch.sqrt(_nmse(x, y, dims, ctx, eps, "std"))
    if kind is PearsonR:
        # As the_well: mean (biased) covariance over unbiased standard deviations
        x_mean, y_mean = global_mean(x, dims, ctx), global_mean(y, dims, ctx)
        covariance = _mean((x - x_mean) * (y - y_mean), dims, ctx)
        std_x = global_std_mean(x, dims, ctx)[0].squeeze(dims)
        std_y = global_std_mean(y, dims, ctx)[0].squeeze(dims)
        return covariance / (std_x * std_y + eps)
    if kind is LInfinity:
        local = (x - y).abs().flatten(start_dim=dims[0], end_dim=-2).max(dim=-2).values
        dist.all_reduce(local, op=dist.ReduceOp.MAX, group=ctx.group)
        return local
    raise NotImplementedError(f"{kind.__name__} on a split domain")

