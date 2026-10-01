"""Attention across slabs (tracker task t301), Ulysses style, and stochastic
depth that agrees across the slabs of a sample.

Before attention each GPU holds all heads for its own tokens. heads_to_tokens
swaps that over the spatial group so each GPU holds a 1/size share of the
heads for all tokens of the domain; ordinary attention then runs per GPU, and
tokens_to_heads swaps back. Each swap is the other's backward. Tokens are
ordered by rank, which is the order of the full domain when tokens are
flattened with the split axis outermost. Slabs may hold different numbers of
tokens; the number of heads must be a multiple of the group size.
"""

from typing import List

import torch
import torch.distributed as dist
from timm.layers import drop_path
from torch import nn

from walrus.utils.distributed_gather import _comm_device, _send_recv, _slab_widths
from walrus.utils.spatial import SpatialContext


def _check_heads(n_heads: int, ctx: SpatialContext) -> None:
    if n_heads % ctx.size:
        raise ValueError(
            f"{n_heads} attention heads cannot be shared among {ctx.size} GPUs: "
            "the head count must be a multiple of the spatial group size"
        )


def _swap(ctx: SpatialContext, sends: List[torch.Tensor], recv_shapes, like: torch.Tensor):
    """sends[p] goes to rank p; returns what each rank p sent here."""
    cdev = _comm_device(ctx, like)
    recvs = [like.new_empty(shape, device=cdev) for shape in recv_shapes]
    _send_recv(ctx, [s.contiguous().to(cdev) for s in sends], recvs)
    recvs[ctx.rank] = sends[ctx.rank]
    return [r.to(like.device) for r in recvs]


def _heads_to_tokens(x: torch.Tensor, ctx: SpatialContext, counts: List[int]):
    # x: (B, heads, my tokens, C) -> (B, heads / size, all tokens, C)
    B, H, _, C = x.shape
    chunks = list(x.chunk(ctx.size, dim=1))
    shapes = [(B, H // ctx.size, n, C) for n in counts]
    return torch.cat(_swap(ctx, chunks, shapes, x), dim=2)


def _tokens_to_heads(y: torch.Tensor, ctx: SpatialContext, counts: List[int]):
    # y: (B, heads / size, all tokens, C) -> (B, heads, my tokens, C)
    B, h, _, C = y.shape
    parts = list(y.split(counts, dim=2))
    shapes = [(B, h, counts[ctx.rank], C)] * ctx.size
    return torch.cat(_swap(ctx, parts, shapes, y), dim=1)


class _HeadsToTokens(torch.autograd.Function):
    @staticmethod
    def forward(fctx, x, ctx, counts):
        fctx.ctx, fctx.counts = ctx, counts
        return _heads_to_tokens(x, ctx, counts)

    @staticmethod
    def backward(fctx, grad):
        return _tokens_to_heads(grad, fctx.ctx, fctx.counts), None, None


class _TokensToHeads(torch.autograd.Function):
    @staticmethod
    def forward(fctx, y, ctx, counts):
        fctx.ctx, fctx.counts = ctx, counts
        return _tokens_to_heads(y, ctx, counts)

    @staticmethod
    def backward(fctx, grad):
        return _heads_to_tokens(grad, fctx.ctx, fctx.counts), None, None


def token_counts(n_local: int, ctx: SpatialContext, like: torch.Tensor) -> List[int]:
    """Number of tokens on every rank of the group."""
    return _slab_widths(n_local, ctx, _comm_device(ctx, like))


def slab_attention(q, k, v, ctx: SpatialContext, attend):
    """attend(q, k, v) over all tokens of the domain, for q, k, v of shape
    (B, heads, my tokens, C) holding this slab's tokens. Differentiable."""
    _check_heads(q.shape[1], ctx)
    counts = token_counts(q.shape[2], ctx, q)
    q, k, v = (_HeadsToTokens.apply(t, ctx, counts) for t in (q, k, v))
    return _TokensToHeads.apply(attend(q, k, v), ctx, counts)


class SpatialDropPath(nn.Module):
    """timm's DropPath (stochastic depth per sample). On a split domain the
    keep/drop draw is made on the group's first rank and broadcast, so all
    slabs of a sample agree; otherwise identical to timm's DropPath."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob, self.scale_by_keep = drop_prob, scale_by_keep
        self.spatial_ctx = None  # set by enable_domain_split

    def forward(self, x):
        ctx = self.spatial_ctx
        if ctx is None or ctx.size == 1 or not self.training or self.drop_prob == 0.0:
            return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
        keep = 1 - self.drop_prob
        mask = x.new_empty((x.shape[0],) + (1,) * (x.dim() - 1)).bernoulli_(keep)
        dist.broadcast(mask, src=dist.get_global_rank(ctx.group, 0), group=ctx.group)
        if keep > 0.0 and self.scale_by_keep:
            mask.div_(keep)
        return x * mask
