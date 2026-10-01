import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.nn.functional import all_reduce

from walrus.utils.spatial import SpatialContext


def get_spatial_dims(n_dims: int, include_time: bool):
    """Assumes input is ([T], B, H, [W, D], C)"""
    start = 1
    if include_time:
        start += 1
    return list(range(start, start + n_dims))


def spatial_rms_norm(x: torch.Tensor, ctx: SpatialContext) -> torch.Tensor:
    """Same as F.rms_norm(x, x.shape[3:]) - for x (B, groups, channels per
    group, *space) that is the RMS of each channel over space - when space is
    split into slabs over ctx.group: the sum of squares and the point count are
    summed over the group, so every slab is scaled by the RMS of the whole
    domain. Uses F.rms_norm's default eps (finfo(dtype).eps)."""
    reduce_dims = tuple(range(3, x.dim()))
    # Accumulate in at least float32 (half-precision inputs under AMP)
    acc_dtype = torch.promote_types(x.dtype, torch.float32)
    sum_sq = torch.linalg.vector_norm(x, ord=2, dim=reduce_dims, dtype=acc_dtype).square()
    sum_sq = all_reduce(sum_sq, group=ctx.group)  # differentiable
    count = torch.tensor(float(math.prod(x.shape[3:])), device=x.device)
    torch.distributed.all_reduce(count, group=ctx.group)
    inv_rms = torch.rsqrt(sum_sq / count + torch.finfo(x.dtype).eps)
    return x * inv_rms.to(x.dtype).view(*inv_rms.shape, *([1] * len(reduce_dims)))


class RMSGroupNorm(nn.Module):
    r"""Applies RMS version of Group Normalization over a mini-batch of inputs as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::
        y = \frac{x}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. :attr:`num_channels` must be divisible by
    :attr:`num_groups`. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = nn.GroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = nn.GroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = nn.GroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)
    """

    __constants__ = ["num_groups", "num_channels", "eps", "affine"]
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-6,
        affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter("weight", None)

        self.reset_parameters()
        # Set by walrus.utils.spatial.enable_domain_split: the input is then one
        # slab of the domain and the statistics are summed over the group.
        self.spatial_ctx: SpatialContext | None = None

    def reset_parameters(self) -> None:
        if self.affine:
            nn.init.ones_(self.weight)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Assume input is (B, C, H, W, D)
        dims = list(input.shape[2:])
        input = input.view(input.shape[0], self.num_groups, -1, *dims)
        norm_shape = input.shape[3:]
        if self.spatial_ctx is not None and self.spatial_ctx.size > 1:
            input = spatial_rms_norm(input, self.spatial_ctx)
        else:
            input = F.rms_norm(input, normalized_shape=norm_shape)
        input = input.view(input.shape[0], -1, *dims)
        if self.weight is not None:
            indexing_tuple = (slice(None),) + (None,) * len(dims)
            return input * self.weight[indexing_tuple]
        else:
            return input

    def extra_repr(self) -> str:
        return "{num_groups}, {num_channels}, eps={eps}, affine={affine}".format(
            **self.__dict__
        )
