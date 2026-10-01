import os
import socket
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh

from walrus.utils.distribution_utils import _first_host, setup_env_from_slurm
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


TORCH_VARS = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]


@mock.patch.dict(os.environ)
def test_setup_env_from_slurm(monkeypatch):
    for var in TORCH_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SLURM_PROCID", "5")
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_LOCALID", "1")
    monkeypatch.delenv("SLURM_NTASKS_PER_NODE", raising=False)
    monkeypatch.setenv("SLURM_TASKS_PER_NODE", "4(x2)")
    monkeypatch.delenv("SLURM_STEP_NODELIST", raising=False)
    monkeypatch.setenv("SLURM_JOB_NODELIST", "nid[001234-001237,001240]")
    setup_env_from_slurm()
    assert {v: os.environ[v] for v in TORCH_VARS} == {
        "RANK": "5",
        "WORLD_SIZE": "8",
        "LOCAL_RANK": "1",
        "LOCAL_WORLD_SIZE": "4",
        "MASTER_ADDR": "nid001234",
        "MASTER_PORT": "29500",
    }


@pytest.mark.parametrize(
    "nodelist, host",
    [
        ("nid[001234-001237,001240]", "nid001234"),
        ("nid001234", "nid001234"),
        ("nid001240,nid[001234-001235]", "nid001240"),
        ("", ""),
    ],
)
def test_first_host(nodelist, host):
    assert _first_host(nodelist) == host


@mock.patch.dict(os.environ)
def test_setup_env_from_slurm_keeps_torchrun_values(monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "5")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("MASTER_ADDR", "node0")
    setup_env_from_slurm()
    assert os.environ["RANK"] == "2" and os.environ["MASTER_ADDR"] == "node0"


@mock.patch.dict(os.environ)
def test_setup_env_without_slurm_does_nothing(monkeypatch):
    for var in TORCH_VARS + ["SLURM_PROCID"]:
        monkeypatch.delenv(var, raising=False)
    setup_env_from_slurm()
    assert not any(v in os.environ for v in TORCH_VARS)


def test_no_context_without_spatial_mode():
    clear_spatial_context()
    assert get_spatial_context() is None
