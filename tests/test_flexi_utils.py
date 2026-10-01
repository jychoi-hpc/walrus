import os.path as osp
import pathlib

import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from the_well.data.datasets import WellMetadata

from walrus.models.shared_utils.flexi_utils import choose_kernel_size_deterministic
from walrus.train import CONFIG_DIR, CONFIG_NAME


def tokens_per_axis(shape, kernels):
    return [size // (k1 * k2) for size, (k1, k2) in zip(shape, kernels)]


@pytest.mark.parametrize(
    "shape, expected",
    [
        ((64, 64, 64), ((2, 2), (2, 2), (2, 2))),
        ((256, 256, 256), ((4, 4), (4, 4), (4, 4))),
        ((512, 192, 512), ((8, 4), (6, 2), (8, 4))),
        ((512, 384, 512), ((8, 4), (6, 4), (8, 4))),
        ((512, 1, 512), ((4, 4), (1, 1), (4, 4))),
    ],
)
def test_table_sizes_unchanged(shape, expected):
    assert choose_kernel_size_deterministic(shape) == expected


@pytest.mark.parametrize(
    "shape, expected_tokens",
    [
        ((1536, 384, 1024), [48, 16, 32]),
        ((768, 192, 512), [24, 16, 16]),
        ((1024, 64, 64), [32, 16, 16]),
    ],
)
def test_large_3d_axes_keep_largest_patch(shape, expected_tokens):
    kernels = choose_kernel_size_deterministic(shape)
    assert tokens_per_axis(shape, kernels) == expected_tokens
    # Stride never exceeds the base kernel (8, 4), so no grid point is skipped.
    assert all(k1 <= 8 and k2 <= 4 for k1, k2 in kernels)


@pytest.mark.parametrize(
    "shape, error",
    [
        ((1536, 1, 1024), KeyError),  # singleton axis: unchanged behaviour
        ((1536, 1024), KeyError),  # 2D: unchanged behaviour
        ((1552, 384, 512), KeyError),  # 1552 is not a multiple of 32
        ((1000, 384, 512), AssertionError),  # not a multiple of 16
    ],
)
def test_unsupported_shapes_still_raise(shape, error):
    with pytest.raises(error):
        choose_kernel_size_deterministic(shape)


def test_model_forward_with_large_axis():
    cfg_dir = osp.relpath(CONFIG_DIR, pathlib.Path(__file__).resolve().parent)
    with initialize(config_path=str(cfg_dir), version_base=None):
        cfg = compose(
            config_name=CONFIG_NAME,
            overrides=["server=local", "logger=none", "model=debug"],
        )
    model = instantiate(cfg.model, n_states=4).eval()
    shape = (1024, 64, 64)
    metadata = WellMetadata(
        dataset_name="dummy",
        n_spatial_dims=3,
        grid_type="cartesian",
        spatial_resolution=shape,
        scalar_names=[],
        constant_scalar_names=[],
        constant_field_names={0: [], 1: [], 2: []},
        field_names={0: ["a", "b", "c", "d"], 1: [], 2: []},
        boundary_condition_types=["PERIODIC", "WALL", "PERIODIC"],
        n_files=1,
        n_trajectories_per_file=[1],
        n_steps_per_trajectory=[2],
    )
    x = torch.randn(1, 1, 4, *shape)  # T B C H W D
    bcs = [[[2, 2], [0, 0], [2, 2]]]  # periodic, wall, periodic
    with torch.no_grad():
        y = model(x, torch.arange(4), bcs, metadata=metadata)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
