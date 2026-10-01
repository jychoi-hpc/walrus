import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from walrus.utils.spatial import (
    DATA_PARALLEL_DIM,
    SPATIAL_DIM,
    clear_spatial_context,
    get_spatial_context,
    set_spatial_context,
    slab_bounds,
)


@pytest.mark.parametrize(
    "n_points, size, align",
    [(1536, 4, 32), (1536, 2, 32), (512, 4, 32), (384, 4, 24), (1536, 5, 32)],
)
def test_slabs_cover_axis_on_patch_boundaries(n_points, size, align):
    bounds = [slab_bounds(n_points, size, r, align) for r in range(size)]
    assert bounds[0][0] == 0 and bounds[-1][1] == n_points
    for (start, stop), (next_start, _) in zip(bounds, bounds[1:]):
        assert stop == next_start  # contiguous, no gaps or overlaps
    widths = [stop - start for start, stop in bounds]
    assert all(start % align == 0 for start, _ in bounds)
    assert max(widths) - min(widths) <= align  # balanced to within one patch


def test_full_grid_splits_into_equal_slabs():
    assert [slab_bounds(1536, 4, r, 32) for r in range(4)] == [
        (0, 384),
        (384, 768),
        (768, 1152),
        (1152, 1536),
    ]


@pytest.mark.parametrize("n_points, size, align", [(1000, 4, 32), (64, 4, 32)])
def test_slab_bounds_rejects_unsplittable_axes(n_points, size, align):
    with pytest.raises(ValueError):
        slab_bounds(n_points, size, 0, align)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _check_context(rank, world_size, spatial_size, port, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh(
            "cpu",
            (world_size // spatial_size, spatial_size),
            mesh_dim_names=(DATA_PARALLEL_DIM, SPATIAL_DIM),
        )
        ctx = set_spatial_context(mesh, axis=0)
        # Which global ranks share this rank's spatial group?
        members = [None] * ctx.size
        dist.all_gather_object(members, rank, group=ctx.group)
        results[rank] = (ctx.rank, ctx.size, ctx.dp_rank, ctx.dp_size, members)
        assert get_spatial_context() is ctx
    finally:
        clear_spatial_context()
        dist.destroy_process_group()


def test_spatial_groups_on_mesh():
    world_size, spatial_size = 4, 2
    results = mp.Manager().dict()
    mp.spawn(
        _check_context,
        args=(world_size, spatial_size, _free_port(), results),
        nprocs=world_size,
        join=True,
    )
    # Contiguous ranks form a spatial group; groups are data-parallel replicas.
    assert dict(results) == {
        0: (0, 2, 0, 2, [0, 1]),
        1: (1, 2, 0, 2, [0, 1]),
        2: (0, 2, 1, 2, [2, 3]),
        3: (1, 2, 1, 2, [2, 3]),
    }


def test_no_context_without_spatial_mode():
    clear_spatial_context()
    assert get_spatial_context() is None
