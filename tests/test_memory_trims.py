"""The memory trims (tracker task t209) must not change any value: each is
compared bit for bit (torch.equal) with the code it replaced."""

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from walrus.models.isotropic_model import field_dropout_and_scale
from walrus.trainer.normalization_strat import (
    MeanStdSamplewiseRevNormalization,
    RMSSamplewiseRevNormalization,
)


def original_dropout_and_scale(x, dummy, p, training):
    """The code before the trim (IsotropicModel._encoder_forward)."""
    T = x.shape[0]
    x = rearrange(x, "t b c h ... -> b c (t h) ...")
    x = F.dropout3d(x, training=training, p=p)
    x = rearrange(x, "b c (t h) ... -> t b c h ...", t=T)
    return x * dummy


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("shape", [(1, 2, 4, 8, 6, 5), (3, 2, 5, 4, 4, 4)])
def test_field_dropout_and_scale_is_bit_identical(training, shape):
    p = 0.4
    x0 = torch.randn(shape)
    results = []
    for fn in (original_dropout_and_scale, field_dropout_and_scale):
        x = x0.clone().requires_grad_()
        dummy = torch.nn.Parameter(torch.tensor([1.37]))
        torch.manual_seed(11)  # same random numbers for both
        y = fn(x, dummy, p=p, training=training)
        upstream = torch.randn(y.shape, generator=torch.Generator().manual_seed(3))
        (y * upstream).sum().backward()
        results.append((y.detach(), x.grad, dummy.grad))
    for old, new in zip(*results):
        assert torch.equal(old, new)
    if training:  # the test really dropped some fields
        assert (results[0][0] == 0).any()


@pytest.mark.parametrize("revin_cls", [RMSSamplewiseRevNormalization,
                                       MeanStdSamplewiseRevNormalization])
def test_in_place_normalization_is_bit_identical(revin_cls):
    from types import SimpleNamespace

    revin = revin_cls()
    x = torch.randn(2, 3, 4, 8, 6, 5) * 3 + 1  # T B C x y z
    stats = revin.compute_stats(x, SimpleNamespace(n_spatial_dims=3))
    expected = revin.normalize_stdmean(x, stats)
    out = revin.normalize_stdmean_(x, stats)
    assert out.data_ptr() == x.data_ptr()  # really in place
    assert torch.equal(out, expected)


def original_jitter(jitterer, x, bcs, kernels, patch_size, rolls):
    """The jitter as it was before the trim: pad, full boundary-flag tensor,
    concatenate, roll (PatchJittererBoundaryPad.forward)."""
    from the_well.data.datasets import BoundaryCondition

    T, shape = x.shape[0], x.shape[3:]
    constant, periodic, _, _ = jitterer.get_paddings(
        shape, bcs, 3, patch_size, {"base_kernel": kernels, "random_kernel": kernels})
    x = rearrange(x, "t b c h w d -> (t b) c h w d")
    if sum(constant) > 0:
        x = F.pad(x, pad=constant, mode="constant")
    if sum(periodic) > 0:
        x = F.pad(x, pad=periodic, mode="circular")
    x = rearrange(x, "(t b) c h w d -> t b c h w d", t=T)
    base = [slice(None)] * x.dim()
    flags_shape = list(x.shape)
    flags_shape[2] = 3
    flags = torch.zeros(flags_shape, dtype=x.dtype)
    flags[:, :, 0] = 1.0
    dims = []
    for i in range(3):
        if shape[i] == 1:
            continue
        if int(bcs[i][0]) != BoundaryCondition["PERIODIC"].value:
            b, e = base[:], base[:]
            b[i + 3] = slice(None, constant[-2 * i - 2])
            b[2] = 1 + int(bcs[i][0])
            e[i + 3] = slice(-constant[-2 * i - 1], None)
            e[2] = 1 + int(bcs[i][1])
            flags[tuple(b)] = flags[tuple(b)] + 1.0
            flags[tuple(e)] = flags[tuple(e)] + 1.0
            b[2], e[2] = 0, 0
            flags[tuple(b)] = 0.0
            flags[tuple(e)] = 0.0
        dims.append(i + 3)
    return torch.roll(torch.cat((x, flags), dim=2), shifts=rolls, dims=dims)


@pytest.mark.parametrize("bc_names", [("PERIODIC", "WALL", "PERIODIC"),
                                      ("WALL", "WALL", "PERIODIC"),
                                      ("PERIODIC", "PERIODIC", "PERIODIC")])
@pytest.mark.parametrize("rolls", [[3, -2, 1], [0, 0, 0], [-5, 4, -1]])
@pytest.mark.parametrize("shape", [(16, 12, 8), (16, 1, 8)])
def test_jitter_is_bit_identical(bc_names, rolls, shape):
    from types import SimpleNamespace

    from the_well.data.datasets import BoundaryCondition as BC

    from walrus.models.shared_utils.patch_jitterers import PatchJittererBoundaryPad

    jitterer = PatchJittererBoundaryPad(stage_dim=4, max_d=3)
    bcs = torch.tensor([[BC[n].value] * 2 for n in bc_names])
    kernels = ((3, 2), (3, 2), (2, 2))  # kernel > stride on x and y: padding
    patch = [4, 4, 4]
    rolls = [r for r, n in zip(rolls, shape) if n > 1]
    x0 = torch.randn(2, 2, 4, *shape)
    outs = []
    for new in (False, True):
        x = x0.clone().requires_grad_()
        if new:
            y, _ = jitterer(x, bcs, SimpleNamespace(n_spatial_dims=3), patch_size=patch,
                            base_kernel=kernels, random_kernel=kernels,
                            jitter_override={"rolls": (rolls, None)})
        else:
            y = original_jitter(jitterer, x, bcs, kernels, patch, rolls)
        upstream = torch.randn(y.shape, generator=torch.Generator().manual_seed(4))
        (y * upstream).sum().backward()
        outs.append((y.detach(), x.grad))
    assert outs[0][0].shape == outs[1][0].shape
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])


@pytest.mark.parametrize("batch_size", [1, 3])
def test_batch_loading_is_bit_identical(dummy_dataset, batch_size, monkeypatch):
    """Host-side trims (no identity tile of freshly read fields, no zero-width
    padding, unsqueeze instead of stack for one sample) give the same batch."""
    import walrus.data.inflated_dataset as inflated
    from the_well.data.datasets import WellDataset

    from walrus.data.multidatamodule import MixedWellDataModule

    def batch():
        module = MixedWellDataModule(
            well_base_path=dummy_dataset,
            well_dataset_info={"dummy": {"path": dummy_dataset / "dummy",
                                         "include_filters": [], "exclude_filters": []}},
            batch_size=batch_size, data_workers=0)
        dset = module.train_dataset.sub_dsets[0]
        return dset[list(range(batch_size))]

    new = batch()
    pad_axes = WellDataset._pad_axes
    monkeypatch.setattr(inflated, "collate_samples", inflated.default_collate)
    monkeypatch.setattr(WellDataset, "_pad_axes",
                        lambda self, *a, **k: pad_axes(self, *a, **{**k, "copy": True}))
    old = batch()
    assert new.keys() == old.keys()
    for key in new:
        if torch.is_tensor(new[key]):
            assert new[key].shape == old[key].shape and torch.equal(new[key], old[key]), key
