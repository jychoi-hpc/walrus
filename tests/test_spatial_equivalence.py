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


# t205: RMSGroupNorm across the spatial group
def rms_group_norm():
    from walrus.models.shared_utils.normalization import RMSGroupNorm

    torch.manual_seed(1)
    norm = RMSGroupNorm(num_groups=2, num_channels=4)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)  # non-trivial weights so their gradient is tested
    return norm


def patch_conv_then_norm():
    return nn.Sequential(patch_conv(), rms_group_norm_6(), nn.GELU())


def rms_group_norm_6():
    from walrus.models.shared_utils.normalization import RMSGroupNorm

    return RMSGroupNorm(num_groups=3, num_channels=6)


def split(model):
    from walrus.utils.spatial import enable_domain_split

    return enable_domain_split(model)


@pytest.mark.parametrize("n_points", [32, 40])
def test_rms_group_norm_is_equivalent_when_split(n_points):
    assert_spatially_equivalent(rms_group_norm, (2, 4, n_points, 8, 8), split_dim=2,
                                align=4, make_spatial_module=split)


def test_rms_group_norm_differs_without_split():
    reports = check_spatial_equivalence(rms_group_norm, SHAPE, split_dim=2)
    assert not all(r["output"][2] for r in reports)


def test_patch_conv_and_norm_stack_is_equivalent():
    assert_spatially_equivalent(patch_conv_then_norm, (2, 4, 40, 8, 8), split_dim=2,
                                align=4, make_spatial_module=split)


# t207: RevIN statistics and the loss across the spatial group
class RevINNormalize(nn.Module):
    """What the trainer does to inputs: samplewise statistics (no gradient),
    then normalized values and normalized time differences, scaled by a
    parameter so its gradient is checked too. Input: T B C x y z."""

    def __init__(self, kind):
        from walrus.trainer import normalization_strat as ns

        super().__init__()
        self.revin = {"rms": ns.RMSSamplewiseRevNormalization,
                      "meanstd": ns.MeanStdSamplewiseRevNormalization}[kind]()
        self.scale = nn.Parameter(torch.linspace(0.5, 1.5, 4).view(1, 1, 4, 1, 1, 1))
        self.spatial_ctx = None  # set by enable_domain_split

    def forward(self, x):
        from types import SimpleNamespace

        self.revin.spatial_ctx = self.spatial_ctx
        meta = SimpleNamespace(n_spatial_dims=3)
        with torch.no_grad():
            stats = self.revin.compute_stats(x, meta)
        values = self.revin.normalize_stdmean(x, stats)
        deltas = self.revin.normalize_delta(x[1:] - x[:-1], stats)
        return self.scale * torch.cat([values, deltas], dim=0)


class MAELoss(nn.Module):
    """Training loss on (B, T, x, y, z, C) predictions against a target that
    depends on the prediction pointwise; on slabs, each GPU's share."""

    def __init__(self):
        super().__init__()
        self.spatial_ctx = None

    def forward(self, y_pred):
        from types import SimpleNamespace

        from the_well.benchmark.metrics import MAE

        from walrus.trainer.spatial_reductions import spatial_mean_loss

        meta = SimpleNamespace(n_spatial_dims=3)
        y_ref = torch.tanh(1.3 * y_pred).detach()
        if self.spatial_ctx is not None:
            return spatial_mean_loss(MAE(), y_pred, y_ref, meta, self.spatial_ctx).mean()
        return MAE()(y_pred, y_ref, meta).mean()


REVIN_SHAPE = (3, 2, 4, 40, 4, 4)  # T B C x y z, 3 time steps for delta stats


@pytest.mark.parametrize("kind", ["rms", "meanstd"])
def test_revin_statistics_are_equivalent_when_split(kind):
    assert_spatially_equivalent(partial(RevINNormalize, kind), REVIN_SHAPE, split_dim=3,
                                align=4, make_spatial_module=split, atol=1e-5, rtol=1e-5)


def test_revin_statistics_differ_without_split():
    reports = check_spatial_equivalence(partial(RevINNormalize, "rms"), REVIN_SHAPE,
                                        split_dim=3, align=4, atol=1e-5, rtol=1e-5)
    assert not all(r["output"][2] for r in reports)


def test_loss_shares_add_up_to_full_domain_loss():
    assert_spatially_equivalent(MAELoss, (2, 1, 40, 4, 4, 3), split_dim=2, align=4,
                                make_spatial_module=split, out_split_dim=None,
                                output_is_partial_sum=True)


# t206: rotary positions of a slab are those of the full grid
class RotaryQK(nn.Module):
    """FullAttention's rotary embedding applied to queries (B, heads, x, y, z, c)."""

    def __init__(self):
        from walrus.models.spatial_blocks.full_attention import FullAttention

        super().__init__()
        torch.manual_seed(2)
        self.attention = FullAttention(hidden_dim=32, num_heads=2, mlp_dim=64)

    def forward(self, q):
        from walrus.models.shared_utils.lr_rope_temporary import apply_rotary_emb

        pos = self.attention.axial_rotary_freqs(*q.shape[2:5])
        return apply_rotary_emb(pos.to(q.dtype), q)


ROTARY_SHAPE = (2, 2, 10, 4, 3, 16)  # 10 tokens along x: slabs of 3/3/2/2


def test_rotary_positions_are_equivalent_when_split():
    assert_spatially_equivalent(RotaryQK, ROTARY_SHAPE, split_dim=2,
                                make_spatial_module=split)


def test_rotary_positions_differ_without_split():
    reports = check_spatial_equivalence(RotaryQK, ROTARY_SHAPE, split_dim=2)
    assert not all(r["output"][2] for r in reports)


def _attention_refuses_slabs(ctx):
    from walrus.models.spatial_blocks.full_attention import FullAttention
    from walrus.utils.spatial import enable_domain_split

    attention = enable_domain_split(FullAttention(hidden_dim=32, num_heads=2, mlp_dim=64))
    try:
        attention(torch.randn(1, 32, 3, 4, 4), bcs=None)
    except NotImplementedError:
        return True
    return False


def test_full_attention_refuses_to_run_on_slabs_until_t301():
    assert all(run_on_spatial_group(_attention_refuses_slabs))


# t202: halo exchange - a convolution on halo'd slabs equals the full domain
class HaloConv(nn.Module):
    """Conv with an odd kernel along x. Full domain: pad x (zeros or periodic)
    then convolve. Slab: halo exchange along x, then the same convolution."""

    def __init__(self, kernel=3, periodic=False):
        super().__init__()
        torch.manual_seed(3)
        self.width, self.periodic = kernel // 2, periodic
        self.conv = nn.Conv3d(4, 5, kernel_size=(kernel, 3, 3), padding=(0, 1, 1))
        self.spatial_ctx = None

    def forward(self, x):
        from walrus.utils.halo_exchange import halo_exchange

        if self.spatial_ctx is not None:
            x = halo_exchange(x, 2, self.width, self.spatial_ctx, self.periodic)
        else:
            pad = (0, 0, 0, 0, self.width, self.width)
            x = (torch.nn.functional.pad(x, pad, mode="circular") if self.periodic
                 else torch.nn.functional.pad(x, pad))
        return self.conv(x)


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("kernel, n_points", [(3, 32), (3, 40), (5, 40)])
def test_halo_exchange_conv_is_equivalent(periodic, kernel, n_points):
    assert_spatially_equivalent(partial(HaloConv, kernel, periodic), (2, 4, n_points, 6, 6),
                                split_dim=2, align=4, make_spatial_module=split)


@pytest.mark.parametrize("periodic", [False, True])
def test_halo_exchange_with_two_gpus(periodic):
    # Left and right neighbor are the same rank
    assert_spatially_equivalent(partial(HaloConv, 3, periodic), SHAPE, split_dim=2,
                                align=4, make_spatial_module=split, world_size=2)


def test_halo_wider_than_slab_is_rejected():
    reports = None
    with pytest.raises(Exception, match="narrower than the halo"):
        reports = check_spatial_equivalence(partial(HaloConv, 5), (1, 4, 4, 6, 6),
                                            split_dim=2, make_spatial_module=split)
    assert reports is None
