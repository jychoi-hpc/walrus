"""Check that a layer split across a spatial group computes what it computes
on the full grid (tracker task t208).

Every process of a CPU (gloo) group builds the same reference module and the
same random input, and runs the reference on the full grid in float64. It then
runs the spatial version on its own slab and compares, for its slab:

  - the output,
  - the gradient with respect to the input (from the same random upstream
    gradient, cut to the slab),
  - every parameter gradient, summed over the group first - the sum of the
    slabs' partial gradients is what data-parallel averaging will see.

Each process compares against the reference it computed itself, so nothing
but a few sizes is exchanged. The spatial module must keep the reference
module's parameter names (it is usually the same layer with a new forward).

    assert_spatially_equivalent(make_conv, input_shape=(2, 4, 32, 8, 8),
                                split_dim=2, align=4)
"""

import copy
import os
import socket
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from walrus.utils.spatial import (
    DATA_PARALLEL_DIM,
    SPATIAL_DIM,
    SpatialContext,
    clear_spatial_context,
    set_spatial_context,
    slab_bounds,
)

# name -> (max |error|, max |reference|, within tolerance)
Report = Dict[str, Tuple[float, float, bool]]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world_size, port, results, fn, axis):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh(
            "cpu", (1, world_size), mesh_dim_names=(DATA_PARALLEL_DIM, SPATIAL_DIM)
        )
        ctx = set_spatial_context(mesh, axis=axis)
        results[rank] = fn(ctx)
    finally:
        clear_spatial_context()
        dist.destroy_process_group()


def run_on_spatial_group(fn: Callable[[SpatialContext], Any], world_size: int = 4,
                         axis: int = 0) -> List[Any]:
    """Run fn(ctx) on `world_size` CPU processes forming one spatial group;
    returns each rank's result. fn must be picklable (module-level or partial)."""
    results = mp.Manager().dict()
    mp.spawn(_worker, args=(world_size, _free_port(), results, fn, axis),
             nprocs=world_size, join=True)
    return [results[r] for r in range(world_size)]


def _compare(name, got, want, atol, rtol, report: Report):
    if got.shape != want.shape:
        report[name] = (float("inf"), float(want.abs().max()), False)
        return
    err = float((got - want).abs().max()) if got.numel() else 0.0
    scale = float(want.abs().max()) if want.numel() else 0.0
    report[name] = (err, scale, torch.allclose(got, want, atol=atol, rtol=rtol))


def _check_on_rank(ctx: SpatialContext, make_module, make_spatial_module, input_shape,
                   split_dim, out_split_dim, align, seed, atol, rtol,
                   output_is_partial_sum=False) -> Report:
    torch.manual_seed(seed)
    reference = make_module().double()
    spatial = (make_spatial_module or (lambda m: m))(copy.deepcopy(reference)).double()
    gen = torch.Generator().manual_seed(seed + 1)
    x_full = torch.randn(input_shape, generator=gen, dtype=torch.float64)

    # Reference on the full grid
    x = x_full.clone().requires_grad_()
    y = reference(x)
    upstream = torch.randn(y.shape, generator=gen, dtype=torch.float64)
    (y * upstream).sum().backward()

    # Spatial version on this rank's slab
    start, stop = slab_bounds(input_shape[split_dim], ctx.size, ctx.rank, align)
    x_local = x_full.narrow(split_dim, start, stop - start).clone().requires_grad_()
    y_local = spatial(x_local)
    report: Report = {}
    if out_split_dim is None:  # output not split (e.g. a reduction)
        y_ref, upstream_local = y, upstream
        if output_is_partial_sum:  # e.g. a loss: the slabs' outputs add up
            total = y_local.detach().clone()
            dist.all_reduce(total, group=ctx.group)
            _compare("output (summed over group)", total, y.detach(), atol, rtol, report)
            (y_local * upstream_local).sum().backward()
            _compare("input grad", x_local.grad,
                     x.grad.narrow(split_dim, start, stop - start), atol, rtol, report)
            _compare_param_grads(ctx, reference, spatial, atol, rtol, report)
            return report
    else:
        widths: List[Optional[int]] = [None] * ctx.size
        dist.all_gather_object(widths, y_local.shape[out_split_dim], group=ctx.group)
        if sum(widths) != y.shape[out_split_dim]:
            report["output"] = (float("inf"), 0.0, False)
            return report
        o_start = sum(widths[: ctx.rank])
        y_ref = y.narrow(out_split_dim, o_start, widths[ctx.rank])
        upstream_local = upstream.narrow(out_split_dim, o_start, widths[ctx.rank])
    _compare("output", y_local.detach(), y_ref.detach(), atol, rtol, report)
    (y_local * upstream_local).sum().backward()
    _compare("input grad", x_local.grad,
             x.grad.narrow(split_dim, start, stop - start), atol, rtol, report)

    _compare_param_grads(ctx, reference, spatial, atol, rtol, report)
    return report


def _compare_param_grads(ctx, reference, spatial, atol, rtol, report):
    ref_params = dict(reference.named_parameters())
    for name, p in spatial.named_parameters():
        grad = p.grad.clone() if p.grad is not None else torch.zeros_like(p)
        dist.all_reduce(grad, group=ctx.group)
        ref = ref_params[name]
        ref_grad = ref.grad if ref.grad is not None else torch.zeros_like(ref)
        _compare(f"grad {name}", grad, ref_grad, atol, rtol, report)


def check_spatial_equivalence(
    make_module: Callable[[], nn.Module],
    input_shape: Sequence[int],
    split_dim: int,
    align: int = 1,
    make_spatial_module: Optional[Callable[[nn.Module], nn.Module]] = None,
    out_split_dim: Optional[int] = -1,
    output_is_partial_sum: bool = False,
    world_size: int = 4,
    seed: int = 0,
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> List[Report]:
    """Per-rank reports comparing make_spatial_module(module) on slabs with
    make_module() on the full grid. out_split_dim: dimension along which the
    output is split (-1: same as split_dim; None: output is not split).
    output_is_partial_sum (with out_split_dim=None): each rank's output is its
    share of the full output (e.g. a loss), so outputs are summed over the group
    before comparing."""
    if out_split_dim == -1:
        out_split_dim = split_dim
    fn = partial(_check_on_rank, make_module=make_module,
                 make_spatial_module=make_spatial_module, input_shape=tuple(input_shape),
                 split_dim=split_dim, out_split_dim=out_split_dim, align=align,
                 seed=seed, atol=atol, rtol=rtol,
                 output_is_partial_sum=output_is_partial_sum)
    return run_on_spatial_group(fn, world_size=world_size)


def format_reports(reports: List[Report]) -> str:
    lines = []
    for rank, report in enumerate(reports):
        for name, (err, scale, ok) in report.items():
            lines.append(f"rank {rank} {'ok  ' if ok else 'FAIL'} {name}: "
                         f"max error {err:.3g} (max reference {scale:.3g})")
    return "\n".join(lines)


def assert_spatially_equivalent(*args, **kwargs) -> List[Report]:
    """check_spatial_equivalence, raising AssertionError listing every mismatch."""
    reports = check_spatial_equivalence(*args, **kwargs)
    if not all(ok for report in reports for (_, _, ok) in report.values()):
        raise AssertionError("spatial version differs from the full grid:\n"
                             + format_reports(reports))
    return reports
