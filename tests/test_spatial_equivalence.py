"""Tests for the domain-decomposition helpers (t201) and the spatial
equivalence harness (t208). The harness must pass layers that need no
communication and must catch layers that do."""

from functools import partial

import pytest
import torch
from torch import nn

from walrus.utils.spatial import SlabLayout, gather_slabs, local_slab
from walrus.utils.spatial_equivalence import (
    assert_spatially_equivalent,
    check_spatial_equivalence,
    run_on_spatial_group,
)

# Inputs are (batch, channels, x, y, z); the split is along x (dim 2).
SHAPE = (2, 4, 32, 8, 8)


def test_slab_layout_maps_indices_both_ways():
    layout = SlabLayout(n_points=40, size=4, align=4)  # 10 blocks: 3, 3, 2, 2
    assert layout.all_bounds() == [(0, 12), (12, 24), (24, 32), (32, 40)]
    for g in range(40):
        rank, local = layout.owner(g)
        assert layout.to_global(rank, local) == g
    with pytest.raises(IndexError):
        layout.owner(40)
    with pytest.raises(IndexError):
        layout.to_global(3, 8)
    with pytest.raises(ValueError):
        SlabLayout(n_points=42, size=4, align=4)


def _slab_round_trip(ctx, n_points):
    full = torch.arange(2 * 3 * n_points, dtype=torch.float64).reshape(2, 3, n_points)
    mine = local_slab(full, dim=2, ctx=ctx, align=4)
    return bool(torch.equal(gather_slabs(mine.clone(), dim=2, ctx=ctx), full))


@pytest.mark.parametrize("n_points", [32, 40])  # even and uneven slabs
def test_local_slab_and_gather_round_trip(n_points):
    assert all(run_on_spatial_group(partial(_slab_round_trip, n_points=n_points)))


# Module factories live at module level so worker processes can unpickle them.
def pointwise():
    return nn.Sequential(nn.Conv3d(4, 8, kernel_size=1), nn.GELU(), nn.Conv3d(8, 3, 1))


def patch_conv():
    """Like the encoder's first stage: kernel = stride = patch size."""
    return nn.Conv3d(4, 6, kernel_size=4, stride=4)


def halo_conv():
    """Needs one neighboring point on each side: wrong without a halo exchange."""
    return nn.Conv3d(4, 6, kernel_size=3, padding=1)


class Center(nn.Module):
    """Subtracts the mean over space: wrong without an all-reduce."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, 4, 1, 1, 1))

    def forward(self, x):
        return self.scale * (x - x.mean(dim=(2, 3, 4), keepdim=True))


def test_pointwise_layers_need_no_communication():
    assert_spatially_equivalent(pointwise, SHAPE, split_dim=2)


@pytest.mark.parametrize("n_points", [32, 40])
def test_patch_aligned_conv_needs_no_halo(n_points):
    # The premise of t204: slabs on patch boundaries need no halo for the
    # patch-embedding convolution, even when slabs differ in width.
    assert_spatially_equivalent(patch_conv, (2, 4, n_points, 8, 8), split_dim=2, align=4)


def test_harness_catches_missing_halo():
    reports = check_spatial_equivalence(halo_conv, SHAPE, split_dim=2)
    # Output and input gradient are wrong at the slab edges on every rank
    assert all(not r["output"][2] and not r["input grad"][2] for r in reports)
    with pytest.raises(AssertionError, match="FAIL output"):
        assert_spatially_equivalent(halo_conv, SHAPE, split_dim=2)


def test_harness_catches_missing_all_reduce():
    reports = check_spatial_equivalence(Center, SHAPE, split_dim=2)
    assert not all(r["output"][2] for r in reports)
    assert not all(r["grad scale"][2] for r in reports)
