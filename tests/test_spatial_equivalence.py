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


class SpaceAttention(nn.Module):
    """FullAttention block (B, C, x, y, z) -> (B, C, x, y, z). With drop_path,
    the generator is reseeded so the full grid and rank 0 draw the same mask."""

    def __init__(self, heads=4, drop_path=0.0):
        from walrus.models.spatial_blocks.full_attention import FullAttention

        super().__init__()
        torch.manual_seed(9)
        self.block = FullAttention(hidden_dim=64, num_heads=heads, mlp_dim=128,
                                   drop_path=drop_path)

    def forward(self, x):
        torch.manual_seed(10)
        return self.block(x, bcs=None)[0]


ATTENTION_SHAPE = (3, 64, 10, 4, 3)  # 10 tokens along x: slabs of 3/3/2/2


@pytest.mark.parametrize("world_size", [4, 2])
def test_full_attention_is_equivalent_on_slabs(world_size):
    assert_spatially_equivalent(SpaceAttention, ATTENTION_SHAPE, split_dim=2,
                                make_spatial_module=split, world_size=world_size)


def test_full_attention_differs_without_split():
    reports = check_spatial_equivalence(SpaceAttention, ATTENTION_SHAPE, split_dim=2)
    assert not all(r["output"][2] for r in reports)


def test_drop_path_is_shared_on_slabs():
    # High drop rate over 3 samples: some are dropped, all slabs must agree
    assert_spatially_equivalent(partial(SpaceAttention, 4, 0.5), ATTENTION_SHAPE,
                                split_dim=2, make_spatial_module=split)


def _attention_with_too_few_heads(ctx):
    model = split(SpaceAttention(heads=2))
    try:
        model(torch.randn(1, 64, 3, 4, 3))
    except ValueError as err:
        return "multiple of the spatial group size" in str(err)
    return False


def test_full_attention_needs_heads_divisible_by_gpus():
    assert all(run_on_spatial_group(_attention_with_too_few_heads))


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


# t203: distributed gather / roll and shared randomness
class Roll(nn.Module):
    """torch.roll along x (dim 2), then a pointwise layer so parameters get gradients."""

    def __init__(self, shift):
        super().__init__()
        torch.manual_seed(4)
        self.shift = shift
        self.mix = nn.Conv3d(4, 3, kernel_size=1)
        self.spatial_ctx = None

    def forward(self, x):
        from walrus.utils.distributed_gather import distributed_roll

        if self.spatial_ctx is not None:
            x = distributed_roll(x, self.shift, 2, self.spatial_ctx)
        else:
            x = torch.roll(x, self.shift, dims=2)
        return self.mix(x)


@pytest.mark.parametrize("shift", [3, -5, 13, 40, -63, 101])
def test_distributed_roll_is_equivalent(shift):
    # 40 points in slabs of 12/12/8/8; shifts cross one or several slabs and wrap
    assert_spatially_equivalent(partial(Roll, shift), (2, 4, 40, 3, 3), split_dim=2,
                                align=4, make_spatial_module=split)


class GatherEveryOtherReversed(nn.Module):
    """Takes points 2k in reverse order, with zeros in between (index -1)."""

    def __init__(self):
        super().__init__()
        self.spatial_ctx = None

    def forward(self, x):
        from walrus.utils.distributed_gather import gather_global, slab_start_and_total

        n = x.shape[2]
        if self.spatial_ctx is None:
            start, total = 0, n
        else:
            start, total = slab_start_and_total(n, self.spatial_ctx, x.device)
        g = torch.arange(start, start + n)
        index = torch.where(g % 2 == 0, total - 2 - g, torch.full_like(g, -1))
        if self.spatial_ctx is None:
            out = torch.zeros_like(x)
            keep = index >= 0
            out[:, :, keep] = x[:, :, index[keep]]
            return out
        return gather_global(x, 2, index, self.spatial_ctx)


def test_gather_global_with_fill_is_equivalent():
    assert_spatially_equivalent(GatherEveryOtherReversed, (2, 3, 40, 2, 2), split_dim=2,
                                align=4, make_spatial_module=split)


def _shared_draws(ctx):
    from walrus.utils.distributed_gather import shared_randint

    torch.manual_seed(100 + ctx.rank)  # every rank's generator differs
    return [shared_randint(-15, 16, ctx) for _ in range(5)]


def test_shared_randint_is_the_same_on_every_rank():
    draws = run_on_spatial_group(_shared_draws)
    assert all(d == draws[0] for d in draws)
    torch.manual_seed(100)  # rank 0's generator decides
    assert draws[0] == [int(torch.randint(-15, 16, (1,))) for _ in range(5)]


class JitterEncodeDecode(nn.Module):
    """The model's path around the patch jitter: jitter (pad + bc flags +
    roll), patch-embedding conv (kernel = stride = 4), transposed conv back,
    unjitter. x periodic (split), y wall, z periodic. Input: T B C x y z."""

    def __init__(self, rolls=None):
        from walrus.models.shared_utils.patch_jitterers import PatchJittererBoundaryPad

        super().__init__()
        torch.manual_seed(5)
        self.rolls = rolls
        self.jitterer = PatchJittererBoundaryPad(stage_dim=4, max_d=3)
        self.encode = nn.Conv3d(4 + 3, 6, kernel_size=4, stride=4)  # + 3 bc flag channels
        self.decode = nn.ConvTranspose3d(6, 4, kernel_size=4, stride=4)

    def forward(self, x):
        from types import SimpleNamespace

        from the_well.data.datasets import BoundaryCondition as BC

        bcs = torch.tensor([[BC.PERIODIC.value] * 2, [BC.WALL.value] * 2,
                            [BC.PERIODIC.value] * 2])
        kernels = ((2, 2), (2, 2), (2, 2))  # effective kernel 4 = stride 4
        override = None
        if self.rolls is None:
            torch.manual_seed(7)  # same draws for the full grid and rank 0
        else:
            override = {"rolls": (self.rolls, None)}
        x, info = self.jitterer(x, bcs, SimpleNamespace(n_spatial_dims=3),
                                patch_size=[4, 4, 4], base_kernel=kernels,
                                random_kernel=kernels, jitter_override=override)
        T = x.shape[0]
        y = self.decode(self.encode(x.flatten(0, 1)))
        return self.jitterer.unjitter(y.unflatten(0, (T, -1)), info)


JITTER_SHAPE = (1, 2, 4, 40, 8, 8)  # T B C x y z; x slabs 12/12/8/8


@pytest.mark.parametrize("rolls", [[3, 1, -2], [-9, -1, 1], [17, 0, 3]])
def test_jitter_with_fixed_rolls_is_equivalent(rolls):
    assert_spatially_equivalent(partial(JitterEncodeDecode, rolls), JITTER_SHAPE,
                                split_dim=3, align=4, make_spatial_module=split)


def test_jitter_with_random_rolls_is_equivalent():
    # The split run uses rank 0's draws, broadcast to the group
    assert_spatially_equivalent(JitterEncodeDecode, JITTER_SHAPE, split_dim=3, align=4,
                                make_spatial_module=split)


def test_jitter_differs_without_split():
    reports = check_spatial_equivalence(partial(JitterEncodeDecode, [3, 1, -2]), JITTER_SHAPE,
                                        split_dim=3, align=4)
    assert not all(r["output"][2] for r in reports)


# t204: the real Walrus encoder and decoder on slabs
class EncodeDecode(nn.Module):
    """SpaceBagAdaptiveDVstrideEncoder -> AdaptiveDVstrideDecoder with RMSGroupNorm,
    as in the Walrus model, scaled down. Along x (split, periodic) kernels equal
    strides (as for patch 32 from (8, 4) kernels); along y (wall) the first
    kernel is wider than its stride, like Walrus's y patch. Input: T B C x y z."""

    def __init__(self, x_kernels=(2, 2)):
        from walrus.models.decoders.vstride_decoder import AdaptiveDVstrideDecoder
        from walrus.models.encoders.vstride_encoder import SpaceBagAdaptiveDVstrideEncoder
        from walrus.models.shared_utils.normalization import RMSGroupNorm

        super().__init__()
        torch.manual_seed(6)
        kernels = (x_kernels, (3, 2), (2, 2))
        self.encoder = SpaceBagAdaptiveDVstrideEncoder(
            kernel_scales_seq=((2, 2),), base_kernel_size3d=kernels, input_dim=7,
            inner_dim=8, output_dim=12, spatial_dims=3, groups=2,
            norm_layer=RMSGroupNorm, activation=nn.SiLU)
        self.decoder = AdaptiveDVstrideDecoder(
            base_kernel_size3d=kernels, input_dim=12, inner_dim=8, output_dim=4,
            spatial_dims=3, groups=2, norm_layer=RMSGroupNorm, activation=nn.SiLU)

    def forward(self, x):
        from the_well.data.datasets import BoundaryCondition as BC

        strides = ((2, 2), (2, 2), (2, 2))
        bcs = [[BC.PERIODIC.value] * 2, [BC.WALL.value] * 2, [BC.PERIODIC.value] * 2]
        tokens, info = self.encoder(x, torch.arange(7), bcs, random_kernel=strides)
        return self.decoder(tokens, torch.arange(4), bcs, stage_info=info)


ENCDEC_SHAPE = (1, 2, 7, 40, 9, 8)  # T B C x y z; x slabs 12/12/8/8 = whole 4-point patches


def test_walrus_encoder_decoder_are_equivalent_on_slabs():
    assert_spatially_equivalent(EncodeDecode, ENCDEC_SHAPE, split_dim=3, align=4,
                                make_spatial_module=split)


def test_walrus_encoder_decoder_differ_without_split():
    # The norms inside need the split; without it the slabs normalize alone
    reports = check_spatial_equivalence(EncodeDecode, ENCDEC_SHAPE, split_dim=3, align=4)
    assert not all(r["output"][2] for r in reports)


def _encoder_with_wide_x_kernel(ctx):
    model = split(EncodeDecode(x_kernels=(3, 2)))
    try:
        model(torch.randn(1, 2, 7, 12, 9, 8))
    except NotImplementedError as err:
        return "halo" in str(err)
    return False


def test_encoder_refuses_split_axis_kernel_wider_than_stride():
    assert all(run_on_spatial_group(_encoder_with_wide_x_kernel))


# Time attention: attends over time at each point, but its norm pools over space
class TimeAttention(nn.Module):
    def __init__(self):
        from walrus.models.temporal_blocks.axial_time_attention import AxialTimeAttention

        super().__init__()
        torch.manual_seed(8)
        self.block = AxialTimeAttention(hidden_dim=16, num_heads=2)

    def forward(self, x):
        out = self.block(x)
        return out[0] if isinstance(out, tuple) else out


def test_time_attention_is_equivalent_on_slabs():
    assert_spatially_equivalent(TimeAttention, (3, 2, 16, 12, 4, 4), split_dim=3,
                                make_spatial_module=split)


# t303: the whole IsotropicModel on slabs
class WalrusModel(nn.Module):
    """A small Walrus model (full spatial attention, 4 heads) in training mode:
    drop path and field dropout active. The generator is reseeded per forward
    so the full grid and rank 0 draw the same numbers. Input: T B C x y z;
    x periodic (split), y wall, z periodic."""

    def __init__(self, jitter=False):
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate

        from walrus.train import CONFIG_DIR, CONFIG_NAME

        super().__init__()
        overrides = ["server=local", "model=isotropic_model",
                     "model/processor/space_mixing=full_spatial_attention",
                     "model.hidden_dim=64", "model.projection_dim=16",
                     "model.intermediate_dim=32", "model.processor_blocks=2",
                     "model.groups=4", "model.processor.space_mixing.num_heads=4",
                     "model.processor.time_mixing.num_heads=4",
                     f"model.jitter_patches={jitter}"]
        with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
            cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        torch.manual_seed(12)
        self.model = instantiate(cfg.model, n_states=4)

    def forward(self, x):
        from unittest import mock

        from the_well.data.datasets import WellMetadata

        shape = (512, 16, 64)  # full grid, also on slabs
        meta = WellMetadata(dataset_name="dummy", n_spatial_dims=3, grid_type="cartesian",
                            spatial_resolution=shape, scalar_names=[], constant_scalar_names=[],
                            constant_field_names={0: [], 1: [], 2: []},
                            field_names={0: ["a", "b", "c", "d"], 1: [], 2: []},
                            boundary_condition_types=["PERIODIC", "WALL", "PERIODIC"],
                            n_files=1, n_trajectories_per_file=[1], n_steps_per_trajectory=[2])
        bcs = [[[2, 2], [0, 0], [2, 2]]]
        torch.manual_seed(13)
        # The full grid draws token rolls with numpy, the split model with torch
        # on rank 0: route numpy's draw through torch so both see the same numbers
        torch_randint = lambda low, high=None: int(torch.randint(low, high, (1,)))  # noqa: E731
        with mock.patch("numpy.random.randint", torch_randint):
            return self.model(x, torch.arange(4), bcs, metadata=meta)


# T B C x y z. x needs patch 32 (strides (8, 4) = Walrus's base kernels, so no
# padding along the split axis): 512 points, slabs of 128
MODEL_SHAPE = (2, 1, 4, 512, 16, 64)


@pytest.mark.parametrize("jitter", [False, True])
def test_walrus_model_is_equivalent_on_slabs(jitter):
    assert_spatially_equivalent(partial(WalrusModel, jitter), MODEL_SHAPE, split_dim=3,
                                align=32, make_spatial_module=split, atol=1e-9, rtol=1e-7)
