"""Gather points of a tensor split into slabs by their global index, and the
operations built on it (tracker task t203): a roll of the whole domain along
the split axis, and random numbers shared by the spatial group.

gather_global(x, dim, index, ctx) returns, on each GPU, the points of the
full (split) tensor at the global positions `index` along `dim`, wherever
they live; index -1 gives zeros. Backward sends each point's gradient back
to the GPU that owns it and adds it there. Built from point-to-point
messages: every rank first tells every other rank which of its points it
needs, then the points are sent. Messages to each peer go in a fixed order,
so they also match with NCCL.
"""

from typing import List

import torch
import torch.distributed as dist

from walrus.utils.spatial import SpatialContext


def _slab_widths(n_local: int, ctx: SpatialContext, device) -> List[int]:
    local = torch.tensor([n_local], dtype=torch.long, device=device)
    widths = [torch.empty_like(local) for _ in range(ctx.size)]
    dist.all_gather(widths, local, group=ctx.group)
    return [int(w) for w in widths]


def _comm_device(ctx: SpatialContext, like: torch.Tensor):
    return like.device if dist.get_backend(ctx.group) != "gloo" else torch.device("cpu")


def _send_recv(ctx: SpatialContext, sends: List[torch.Tensor], recvs: List[torch.Tensor]):
    """sends[p] goes to group rank p, recvs[p] is filled from group rank p
    (own rank skipped). Empty tensors are skipped on both sides."""
    peer = lambda r: dist.get_global_rank(ctx.group, r)  # noqa: E731
    ops = []
    for p in range(ctx.size):
        if p == ctx.rank:
            continue
        if sends[p].numel():
            ops.append(dist.P2POp(dist.isend, sends[p].contiguous(), peer(p), ctx.group))
        if recvs[p].numel():
            ops.append(dist.P2POp(dist.irecv, recvs[p], peer(p), ctx.group))
    if ops:
        for request in dist.batch_isend_irecv(ops):
            request.wait()


class _GatherPlan:
    """Who needs which of whose points, for one global index request."""

    def __init__(self, index: torch.Tensor, n_local: int, ctx: SpatialContext, device):
        self.ctx = ctx
        widths = _slab_widths(n_local, ctx, device)
        starts = torch.tensor([0] + widths[:-1], device=index.device).cumsum(0)
        self.n_local, self.n_global = n_local, sum(widths)
        valid = index >= 0
        if bool((index[valid] >= self.n_global).any()):
            raise IndexError(f"global index beyond the axis length {self.n_global}")
        owner = torch.searchsorted(starts, index.clamp(min=0), right=True) - 1
        owner[~valid] = -1
        # What I need from each rank: positions in my output, local index there
        self.want_pos, want_local = [], []
        for p in range(ctx.size):
            pos = (owner == p).nonzero(as_tuple=True)[0]
            self.want_pos.append(pos)
            want_local.append(index[pos] - starts[p])
        # Tell each rank how many of its points I need, then which ones
        cdev = _comm_device(ctx, index)
        counts_out = [torch.tensor([len(w)], device=cdev) for w in want_local]
        counts_in = [torch.zeros(1, dtype=torch.long, device=cdev) for _ in range(ctx.size)]
        _send_recv(ctx, counts_out, counts_in)
        counts_in[ctx.rank] = counts_out[ctx.rank]
        asks_in = [torch.empty(int(c), dtype=torch.long, device=cdev) for c in counts_in]
        _send_recv(ctx, [w.to(cdev) for w in want_local], asks_in)
        asks_in[ctx.rank] = want_local[ctx.rank].to(cdev)
        # give_local[p]: my local indices that rank p needs, in its order
        self.give_local = [a.to(index.device) for a in asks_in]
        self.want_local_self = want_local[ctx.rank]


def _move_points(plan: _GatherPlan, x: torch.Tensor, dim: int, out_len: int,
                 send_index, recv_pos, own_src, own_dst) -> torch.Tensor:
    """Generic exchange: for each peer p send x[send_index[p]] and place what p
    sends at recv_pos[p] of a zero output of length out_len along dim."""
    ctx = plan.ctx
    shape = list(x.shape)
    shape[dim] = out_len
    out = x.new_zeros(shape)
    cdev = _comm_device(ctx, x)
    sends = [x.index_select(dim, send_index[p]).to(cdev) for p in range(ctx.size)]
    recv_shape = lambda n: shape[:dim] + [n] + shape[dim + 1:]  # noqa: E731
    recvs = [x.new_empty(recv_shape(len(recv_pos[p])), device=cdev) for p in range(ctx.size)]
    _send_recv(ctx, sends, recvs)
    recvs[ctx.rank] = x.index_select(dim, own_src)
    for p in range(ctx.size):
        if len(recv_pos[p]):
            out.index_add_(dim, recv_pos[p] if p != ctx.rank else own_dst, recvs[p].to(x.device))
    return out


class _GatherGlobal(torch.autograd.Function):
    @staticmethod
    def forward(fctx, x, dim, index, ctx):
        plan = _GatherPlan(index, x.shape[dim], ctx, x.device)
        fctx.plan, fctx.dim, fctx.n_local = plan, dim, x.shape[dim]
        return _move_points(plan, x, dim, len(index), plan.give_local, plan.want_pos,
                            plan.want_local_self, plan.want_pos[ctx.rank])

    @staticmethod
    def backward(fctx, grad):
        plan, dim = fctx.plan, fctx.dim
        ctx = plan.ctx
        # Reverse the exchange: send gradients of what I received back to the
        # owners, who add them at the local indices they gave.
        grad_x = _move_points(plan, grad, dim, fctx.n_local, plan.want_pos, plan.give_local,
                              plan.want_pos[ctx.rank], plan.want_local_self)
        return grad_x, None, None, None


def gather_global(x: torch.Tensor, dim: int, index: torch.Tensor,
                  ctx: SpatialContext) -> torch.Tensor:
    """Points at global positions `index` (1-D long tensor; -1 = zeros) along
    `dim` of the tensor whose slabs are spread over ctx's group. Differentiable."""
    return _GatherGlobal.apply(x, dim, index.to(x.device), ctx)


def slab_start_and_total(n_local: int, ctx: SpatialContext, device) -> tuple:
    widths = _slab_widths(n_local, ctx, device)
    return sum(widths[: ctx.rank]), sum(widths)


def distributed_roll(x: torch.Tensor, shift: int, dim: int, ctx: SpatialContext) -> torch.Tensor:
    """torch.roll of the whole split domain by `shift` along the split `dim`;
    each GPU keeps its slab of the result. Differentiable."""
    n = x.shape[dim]
    start, total = slab_start_and_total(n, ctx, _comm_device(ctx, x))
    shift = shift % total
    if shift == 0:
        return x
    index = (torch.arange(start, start + n, device=x.device) - shift) % total
    return gather_global(x, dim, index, ctx)


def shared_randint(low: int, high: int, ctx: SpatialContext, device=None) -> int:
    """Random integer in [low, high), the same on every GPU of the group (drawn
    on the group's first rank with torch's generator and broadcast)."""
    value = torch.randint(low, high, (1,), device="cpu")
    if dist.get_backend(ctx.group) != "gloo":
        value = value.to(device or torch.device("cuda", torch.cuda.current_device()))
    dist.broadcast(value, src=dist.get_global_rank(ctx.group, 0), group=ctx.group)
    return int(value)


def roll_split(x: torch.Tensor, shifts, dims, ctx, split_dim: int) -> torch.Tensor:
    """torch.roll(x, shifts, dims) where tensor dim `split_dim` is split over
    ctx's group (ctx None: not split): rolls along other dims stay local."""
    shifts, dims = list(shifts), list(dims)
    if ctx is None or ctx.size == 1 or split_dim not in dims:
        return torch.roll(x, shifts=shifts, dims=dims) if dims else x
    i = dims.index(split_dim)
    split_shift = shifts.pop(i)
    dims.pop(i)
    if dims:
        x = torch.roll(x, shifts=shifts, dims=dims)
    return distributed_roll(x, split_shift, split_dim, ctx)
