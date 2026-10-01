"""Jitter -> encoder -> decoder -> unjitter must return the input's size for
every axis size in the patch-size table and for both boundary types. A
192- or 384-point periodic axis (stride 6, kernel 8) used to come back 4
points short (tracker task t210)."""

from types import SimpleNamespace

import pytest
import torch
from the_well.data.datasets import BoundaryCondition as BC
from torch import nn

from walrus.models.decoders.vstride_decoder import AdaptiveDVstrideDecoder
from walrus.models.encoders.vstride_encoder import SpaceBagAdaptiveDVstrideEncoder
from walrus.models.shared_utils.flexi_utils import choose_kernel_size_deterministic
from walrus.models.shared_utils.normalization import RMSGroupNorm
from walrus.models.shared_utils.patch_jitterers import PatchJittererBoundaryPad

KERNELS = ((8, 4), (8, 4), (8, 4))  # Walrus's 3D base kernels


@pytest.mark.parametrize("n_points", [16, 64, 128, 192, 256, 384, 512, 1024])
@pytest.mark.parametrize("bc", ["PERIODIC", "WALL"])
def test_round_trip_keeps_axis_size(n_points, bc):
    torch.manual_seed(0)
    encoder = SpaceBagAdaptiveDVstrideEncoder(
        kernel_scales_seq=((2, 2),), base_kernel_size3d=KERNELS, input_dim=4, inner_dim=8,
        output_dim=8, spatial_dims=3, groups=2, norm_layer=RMSGroupNorm, activation=nn.SiLU)
    decoder = AdaptiveDVstrideDecoder(
        base_kernel_size3d=KERNELS, input_dim=8, inner_dim=8, output_dim=1, spatial_dims=3,
        groups=2, norm_layer=RMSGroupNorm, activation=nn.SiLU)
    jitterer = PatchJittererBoundaryPad(stage_dim=4, max_d=3)
    shape = (n_points, 16, 16)
    strides = choose_kernel_size_deterministic(shape)
    bcs = torch.tensor([[BC[bc].value] * 2, [BC.PERIODIC.value] * 2, [BC.PERIODIC.value] * 2])
    x = torch.randn(1, 1, 1, *shape)
    with torch.no_grad():
        x, info = jitterer(x, bcs, SimpleNamespace(n_spatial_dims=3),
                           patch_size=[a * b for a, b in strides],
                           base_kernel=KERNELS, random_kernel=strides)
        tokens, stage = encoder(x, torch.arange(4), bcs.tolist(), random_kernel=strides)
        y = jitterer.unjitter(decoder(tokens, torch.arange(1), bcs.tolist(), stage_info=stage), info)
    assert tuple(y.shape[3:]) == shape
