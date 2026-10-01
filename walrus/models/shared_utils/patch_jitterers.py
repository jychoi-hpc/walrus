import itertools
from typing import List, Optional, Sequence, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from the_well.data.datasets import BoundaryCondition

from walrus.utils.distributed_gather import roll_split, shared_randint


def _rolled_ranges(start: int, stop: int, shift: int, length: int):
    """Where [start, stop) of an axis of `length` lands after torch.roll by
    `shift`: one or two (start, stop) ranges."""
    if start >= stop:
        return []
    begin, size = (start + shift) % length, stop - start
    if begin + size <= length:
        return [(begin, begin + size)]
    return [(begin, length), (0, begin + size - length)]


def write_flags(flags: torch.Tensor, flag_ops, shifts) -> None:
    """Boundary flag channels (T, B, 3, *space): BiasCorr = 1 everywhere, then
    the recorded ("add" 1 | "zero", channel, dim, slice) operations, each
    region moved to where torch.roll by `shifts` ({dim: shift}) would put it."""
    flags.zero_()
    flags[:, :, 0] = 1.0
    for kind, channel, dim, region in flag_ops:
        length = flags.shape[dim]
        start, stop, _ = region.indices(length)
        for a, b in _rolled_ranges(start, stop, shifts.get(dim, 0), length):
            index = [slice(None)] * flags.dim()
            index[2], index[dim] = channel, slice(a, b)
            if kind == "add":
                flags[tuple(index)] += 1.0
            else:
                flags[tuple(index)] = 0.0


def roll_into(dst: torch.Tensor, src: torch.Tensor, shifts) -> None:
    """dst = torch.roll(src, shifts) ({dim: shift}) by block copies, without a
    temporary: a roll along one axis moves two blocks."""
    blocks = [((slice(None), slice(None)),)] * src.dim()
    for dim, shift in shifts.items():
        length = src.shape[dim]
        r = shift % length
        if r:
            blocks[dim] = ((slice(r, None), slice(None, length - r)),
                           (slice(None, r), slice(length - r, None)))
    for combo in itertools.product(*blocks):
        dst[tuple(d for d, _ in combo)].copy_(src[tuple(src_s for _, src_s in combo)])


def assemble_rolled(x: torch.Tensor, flag_ops, shifts) -> torch.Tensor:
    """torch.roll(torch.cat((x, flags), dim=2), shifts) built in one buffer."""
    channels = x.shape[2]
    out = x.new_empty(x.shape[:2] + (channels + 3,) + x.shape[3:])
    write_flags(out[:, :, channels:], flag_ops, shifts)
    roll_into(out[:, :, :channels], x, shifts)
    return out


class PatchJitterer(nn.Module):
    """Applies random shifts to patches so that error doesn't accumulate in single patches
    For BCs that don't support periodicity, pads the patches with random values before shifting

    Parameters
    ----------
    stage_dim:
        Dimension of the stage
    patch_size:
        Size of the patch
    num_bcs:
        Number of potential boundary conditions #TODO: autopopulate this
    max_d:
        Maximum dimensionality of the data
    jitter_patches:
        Whether to jitter patches or return shaped identity
    """

    def __init__(
        self,
        stage_dim: int,
        patch_size: Optional[Sequence[int]] = None,
        num_bcs: int = 3,
        max_d: int = 3,
        jitter_patches: bool = True,
    ):
        super().__init__()
        self.jitter_patches = jitter_patches
        self.patch_size = patch_size
        self.padding_tokens = nn.Parameter(torch.randn(num_bcs - 1, stage_dim, 1, 1, 1))
        self.max_d = max_d

    def forward(
        self, x, bcs, metadata, patch_size: Optional[Sequence[int]] = None, **kwargs
    ):
        # x: (T, B, C, H, W, D) - so need to apply to 3D padded data
        # bcs: (n_dims, 2)
        # Allow for identity mapping to simplify code
        if (not self.jitter_patches) and ("learned_pad" not in kwargs):
            return x, {}

        assert (self.patch_size is None) ^ (patch_size is None), (
            "Must provide EITHER `patch_size` as parameter OR `patch_size` as kwarg to the `forward` call, but not both"
        )
        if patch_size is not None:
            _patch_size: Sequence[int] = patch_size
        else:
            _patch_size = cast(Sequence[int], self.patch_size)

        # This will only work if learned padding is needed even when jitter_patches is False
        T = x.shape[0]
        shape: Sequence[int] = x.shape[3:]
        x = rearrange(x, "t b c h w d -> (t b) c h w d")
        n_dims = metadata.n_spatial_dims
        dim_offset = 3  # Offset by T, B, C
        # If not identity, pad and randomly roll
        paddings: List[int] = []
        # Extra paddings if doing strided convolutions
        extra_paddings: List[int] = [0, 0, 0]
        # If not periodic, apply padding first
        for i in range(self.max_d):
            extra_padding = 0
            if i >= n_dims or shape[i] == 1:
                axis_padding: List[int] = [0, 0]
                extra_padding = 0
            else:
                if int(bcs[i][0]) == BoundaryCondition["PERIODIC"].value:
                    axis_padding = [0, 0]
                else:
                    axis_padding = (
                        [_patch_size[i] // 2, _patch_size[i] // 2]
                        if self.jitter_patches
                        else [0, 0]
                    )
                if ("base_kernel" in kwargs) and ("random_kernel" in kwargs):
                    # only true for strided case with learned padding
                    base_kernel1 = kwargs["base_kernel"][i][0]
                    base_kernel2 = kwargs["base_kernel"][i][1]
                    stride1 = kwargs["random_kernel"][i][0]
                    stride2 = kwargs["random_kernel"][i][1]
                    effective_shape = shape[i] + 2 * axis_padding[0]
                    extra_padding = stride1 * stride2 - (
                        (
                            effective_shape
                            - base_kernel1
                            + stride1
                            - base_kernel2 * stride1
                        )
                        % (stride1 * stride2)
                    )
                    extra_padding = extra_padding // 2
                    extra_paddings[i] = extra_padding

            axis_padding_with_extra: List[int] = [
                axis_padding[0] + extra_padding,
                axis_padding[1] + extra_padding,
            ]
            paddings = (
                axis_padding_with_extra + paddings
            )  # Pytorch padding goes [last[start], last[end], ..., first[start], first[end]]

        for i in range(self.max_d):
            indices = 2 * self.max_d - 2 * i - 2, 2 * self.max_d - 2 * i - 1
            axis_pad = [
                paddings[j] if j in indices else 0 for j in range(len(paddings))
            ]
            if i >= n_dims or sum(axis_pad) == 0:
                continue
            if int(bcs[i][0]) == BoundaryCondition["PERIODIC"].value:
                x = F.pad(x, pad=axis_pad, mode="circular")
            else:
                x = F.pad(x, pad=axis_pad, mode="constant")
        x = rearrange(x, "(t b) c h w d -> t b c h w d", t=T)
        # Randomly roll each dimension by a random amount < 1 patch
        base_slices = [slice(None)] * len(x.shape)
        roll_quantities, roll_dims = [], []
        for i in range(self.max_d):
            # If we're beyond the number of spatial dims, skip
            if i >= n_dims or shape[i] == 1:
                continue
            half_patch = (
                _patch_size[i] // 2 + extra_paddings[i]
                if self.jitter_patches
                else extra_paddings[i]
            )
            # Override base slice to specific dimension
            beginning, end = base_slices[:], base_slices[:]
            beginning[i + dim_offset] = slice(None, half_patch)  #
            end[i + dim_offset] = slice(-half_patch, None)
            # apply the padding along the slices (corners are sum of padding tokens)
            if int(bcs[i][0]) != BoundaryCondition["PERIODIC"].value:
                x[tuple(beginning)] += self.padding_tokens[int(bcs[i][0])]
                x[tuple(end)] += self.padding_tokens[int(bcs[i][1])]
            if self.jitter_patches:
                # Compute and log the random roll for this dimension
                from_ = -(half_patch - 1)
                to_ = half_patch
                if from_ < to_:
                    roll_rate = int(torch.randint(from_, to_, ()))
                else:
                    roll_rate = 0
                # TODO - move this to using random state to avoid compilation issues
                roll_quantities.append(roll_rate)
                roll_dims.append(i + dim_offset)
        if self.jitter_patches:
            # Now roll by the randomly sampled values if jitter_patches is true
            x = torch.roll(x, shifts=roll_quantities, dims=roll_dims)
        # Use kwargs for optional compatibility with different versions
        jitter_info = {"paddings": paddings, "rolls": (roll_quantities, roll_dims)}
        return x, jitter_info

    def unjitter(self, x, jitter_info=None, **kwargs):
        if not self.jitter_patches and ("learned_pad" not in kwargs):
            return x
        paddings, rolls = jitter_info["paddings"], jitter_info["rolls"]
        if self.jitter_patches:
            # Reverse the paddings and rolls
            roll_quantities, roll_dims = rolls
            roll_quantities = [-r for r in roll_quantities]
            # Reverse by rolling/padding with negative values
            x = torch.roll(x, shifts=roll_quantities, dims=roll_dims)
        paddings = [-p for p in paddings]
        x = F.pad(x, pad=paddings)
        return x


class PatchJittererBoundaryPad(nn.Module):
    """Applies random shifts to patches so that error doesn't accumulate in single patches
    For BCs that don't support periodicity, pads the patches with random values before shifting

    Parameters
    ----------
    stage_dim:
        Dimension of the stage
    patch_size:
        Size of the patch
    num_bcs:
        Number of potential boundary conditions #TODO: autopopulate this
    max_d:
        Maximum dimensionality of the data
    jitter_patches:
        Whether to jitter patches or return shaped identity
    """

    def __init__(
        self,
        stage_dim: int,
        patch_size: Optional[Sequence[int]] = None,
        num_bcs: int = 3,
        max_d: int = 3,
        jitter_patches: bool = True,
    ):
        super().__init__()
        # Not used, but don't want to disable at the moment.
        self.stage_dim = stage_dim
        self.num_bcs = num_bcs
        self.jitter_patches = jitter_patches
        self.patch_size = patch_size
        self.max_d = max_d
        # Set by walrus.utils.spatial.enable_domain_split: the input is one slab
        self.spatial_ctx = None

    def _split(self):
        ctx = getattr(self, "spatial_ctx", None)
        return ctx if ctx is not None and ctx.size > 1 else None

    def _check_split(self, bcs, constant_paddings, periodic_paddings):
        """Supported when the split axis is periodic and needs no padding
        (encoder kernel = stride along it, e.g. patch 32 from (8, 4) kernels)."""
        ctx = self._split()
        if ctx is None:
            return
        i = ctx.axis
        if int(bcs[i][0]) != BoundaryCondition["PERIODIC"].value:
            raise NotImplementedError("patch jitter on a split wall axis")
        if periodic_paddings[-2 * i - 2] or periodic_paddings[-2 * i - 1]:
            raise NotImplementedError("patch jitter with padding along the split axis")

    def _draw_roll(self, low, high):
        ctx = self._split()
        if ctx is not None:  # every slab of a sample must roll by the same amount
            return shared_randint(low, high, ctx)
        return int(torch.randint(low, high, ()))

    def _roll(self, x, shifts, dims, dim_offset):
        ctx = self._split()
        split_dim = ctx.axis + dim_offset if ctx is not None else None
        return roll_split(x, shifts, dims, ctx, split_dim)

    def get_paddings(self, shape: Sequence[int], bcs, n_dims, _patch_size, kwargs):
        """Compute amount of padding needed on each dimension from the patch size, BC type, and shape"""
        constant_paddings: List[int] = []
        periodic_paddings: List[int] = []
        for i in range(self.max_d):
            extra_padding = 0
            # If this axis is fake, don't pad it
            if i >= n_dims or shape[i] == 1:
                pass
            # Otherwise, we need some level of padding
            else:
                # Now check for padding necessary due to striding
                if ("base_kernel" in kwargs) and ("random_kernel" in kwargs):
                    # only true for strided case with learned padding
                    base_kernel1 = kwargs["base_kernel"][i][0]
                    base_kernel2 = kwargs["base_kernel"][i][1]
                    stride1 = kwargs["random_kernel"][i][0]
                    stride2 = kwargs["random_kernel"][i][1]
                    effective_patch_size = base_kernel1 + stride1 * (base_kernel2 - 1)
                    effective_stride = stride1 * stride2
                    extra_padding = (effective_patch_size - effective_stride) // 2

                if int(bcs[i][0]) == BoundaryCondition["PERIODIC"].value:
                    jitter_padding = [0, 0]
                else:
                    jitter_padding = (
                        [effective_stride // 2, effective_stride // 2]
                        if self.jitter_patches
                        else [0, 0]
                    )
                axis_padding_with_extra: List[int] = [
                    (p + extra_padding) for p in jitter_padding
                ]
            # Pytorch padding goes [last[start], last[end], ..., first[start], first[end]] so we prepend
            if i >= n_dims or shape[i] == 1:
                periodic_paddings = [0, 0] + periodic_paddings
                constant_paddings = [0, 0] + constant_paddings
            elif int(bcs[i][0]) == BoundaryCondition["PERIODIC"].value:
                periodic_paddings = axis_padding_with_extra + periodic_paddings
                constant_paddings = [0, 0] + constant_paddings
            else:
                constant_paddings = axis_padding_with_extra + constant_paddings
                periodic_paddings = [0, 0] + periodic_paddings
        return (
            constant_paddings,
            periodic_paddings,
            effective_patch_size,
            effective_stride,
        )

    def forward(
        self,
        x,
        bcs,
        metadata,
        patch_size: Optional[Sequence[int]] = None,
        jitter_override=None,  # Used for testing
        **kwargs,
    ):
        # x: (T, B, C, H, W, D) - so need to apply to 3D padded data
        # bcs: (n_dims, 2)
        # Allow for identity mapping to simplify code
        if (not self.jitter_patches) and ("learned_pad" not in kwargs):
            return x, {}

        assert (self.patch_size is None) ^ (patch_size is None), (
            "Must provide EITHER `patch_size` as parameter OR `patch_size` as kwarg to the `forward` call, but not both"
        )

        if patch_size is not None:
            _patch_size: Sequence[int] = patch_size
        else:
            _patch_size = cast(Sequence[int], self.patch_size)

        # This will only work if learned padding is needed even when jitter_patches is False
        T = x.shape[0]
        shape: Sequence[int] = x.shape[3:]
        x = rearrange(x, "t b c h w d -> (t b) c h w d")
        n_dims = metadata.n_spatial_dims
        dim_offset = 3  # Offset by T, B, C

        constant_paddings, periodic_paddings, effective_ps, effective_stride = (
            self.get_paddings(shape, bcs, n_dims, _patch_size, kwargs)
        )
        self._check_split(bcs, constant_paddings, periodic_paddings)

        if sum(constant_paddings) > 0:
            x = F.pad(x, pad=constant_paddings, mode="constant")
        if sum(periodic_paddings) > 0:
            x = F.pad(x, pad=periodic_paddings, mode="circular")

        x = rearrange(x, "(t b) c h w d -> t b c h w d", t=T)
        # Randomly roll each dimension by a random amount < 1 patch
        roll_quantities, roll_dims = [], []
        # Boundary flags (BiasCorr/Open/close channels), recorded as operations
        # and written once the rolls are known
        flag_ops = []
        # Padding phase
        for i in range(self.max_d):
            # If we're beyond the number of spatial dims, skip
            if i >= n_dims or shape[i] == 1:
                continue

            # apply the learned BC values along the slices (corners are sum of padding tokens)
            if int(bcs[i][0]) != BoundaryCondition["PERIODIC"].value:
                beginning = slice(None, constant_paddings[-2 * i - 2])
                end = slice(-constant_paddings[-2 * i - 1], None)
                # Use to apply constant padding
                flag_ops.append(("add", 1 + int(bcs[i][0]), i + dim_offset, beginning))
                flag_ops.append(("add", 1 + int(bcs[i][1]), i + dim_offset, end))
                # Hack eproduce the bias term behavioral of previous implementation where projection bias is only applied to non-boundaries
                # Only necessary for experiment consistency - not for actual use.
                flag_ops.append(("zero", 0, i + dim_offset, beginning))
                flag_ops.append(("zero", 0, i + dim_offset, end))
            if self.jitter_patches:
                if _patch_size[i] <= 1:
                    roll_quantities.append(0)
                else:
                    half_patch = _patch_size[i] // 2
                    # Compute and log the random roll for this dimension
                    roll_rate = self._draw_roll(-(half_patch - 1), half_patch)
                    # TODO - move this to using random state to avoid compilation issues
                    roll_quantities.append(roll_rate)
                roll_dims.append(i + dim_offset)
        if self.jitter_patches and jitter_override is not None:
            roll_quantities = jitter_override["rolls"][0]
        shifts = dict(zip(roll_dims, roll_quantities)) if self.jitter_patches else {}
        if self._split() is None:
            # Data and flags written straight into their rolled positions: same
            # values as concatenating and rolling, without those two copies
            x = assemble_rolled(x, flag_ops, shifts)
        else:
            bc_flags = x.new_empty(x.shape[:2] + (3,) + x.shape[3:])
            write_flags(bc_flags, flag_ops, {})
            x = torch.cat((x, bc_flags), dim=2)
            del bc_flags
            if self.jitter_patches:
                x = self._roll(x, roll_quantities, roll_dims, dim_offset)
        # Use kwargs for optional compatibility with different versions
        jitter_info = {
            "constant_paddings": constant_paddings,
            "periodic_paddings": periodic_paddings,
            "rolls": (roll_quantities, roll_dims),
        }
        return x, jitter_info

    def unjitter(self, x, jitter_info=None, **kwargs):
        if not self.jitter_patches and ("learned_pad" not in kwargs):
            return x
        constant_paddings, periodic_paddings, rolls = (
            jitter_info["constant_paddings"],
            jitter_info["periodic_paddings"],
            jitter_info["rolls"],
        )
        if self.jitter_patches:
            # Reverse the paddings and rolls
            roll_quantities, roll_dims = rolls
            roll_quantities = [-r for r in roll_quantities]
            # Reverse by rolling/padding with negative values
            x = self._roll(x, roll_quantities, roll_dims, dim_offset=3)
        paddings = [-(p1 + p2) for p1, p2 in zip(constant_paddings, periodic_paddings)]
        x = F.pad(x, pad=paddings)
        return x


class FixedPatchJittererBoundaryPad(PatchJittererBoundaryPad):
    def forward(
        self,
        x,
        bcs,
        metadata,
        patch_size: Optional[Sequence[int]] = None,
        jitter_override=None,  # Used for testing
        **kwargs,
    ):
        if self._split() is not None:
            raise NotImplementedError("FixedPatchJittererBoundaryPad on a split domain")
        # x: (T, B, C, H, W, D) - so need to apply to 3D padded data
        # bcs: (n_dims, 2)
        # Allow for identity mapping to simplify code
        if (not self.jitter_patches) and ("learned_pad" not in kwargs):
            return x, {}

        assert (self.patch_size is None) ^ (patch_size is None), (
            "Must provide EITHER `patch_size` as parameter OR `patch_size` as kwarg to the `forward` call, but not both"
        )

        if patch_size is not None:
            _patch_size: Sequence[int] = patch_size
        else:
            _patch_size = cast(Sequence[int], self.patch_size)

        # This will only work if learned padding is needed even when jitter_patches is False
        T = x.shape[0]
        shape: Sequence[int] = x.shape[3:]
        x = rearrange(x, "t b c h w d -> (t b) c h w d")
        n_dims = metadata.n_spatial_dims
        dim_offset = 3  # Offset by T, B, C

        constant_paddings, periodic_paddings, effective_ps, effective_stride = (
            self.get_paddings(shape, bcs, n_dims, _patch_size, kwargs)
        )
        if sum(constant_paddings) > 0:
            x = F.pad(x, pad=constant_paddings, mode="constant")

        x = rearrange(x, "(t b) c h w d -> t b c h w d", t=T)
        # Randomly roll each dimension by a random amount < 1 patch
        base_slices = [slice(None)] * len(x.shape)
        roll_quantities, roll_dims = [], []
        bc_flag_shape = list(x.shape)
        bc_flag_shape[2] = 3  # BiasCorr/Open/close
        bc_flags = torch.zeros(bc_flag_shape, device=x.device, dtype=x.dtype)
        bc_flags[:, :, 0] = 1.0  # BiasCorr
        # Padding phase
        for i in range(self.max_d):
            # If we're beyond the number of spatial dims, skip
            if i >= n_dims or shape[i] == 1:
                continue

            # apply the learned BC values along the slices (corners are sum of padding tokens)
            if int(bcs[i][0]) != BoundaryCondition["PERIODIC"].value:
                # Override base slice to specific dimension
                beginning, end = base_slices[:], base_slices[:]
                beginning[i + dim_offset] = slice(
                    None, constant_paddings[-2 * i - 2]
                )  #
                beginning[2] = 1 + int(bcs[i][0])
                end[i + dim_offset] = slice(-constant_paddings[-2 * i - 1], None)
                end[2] = 1 + int(bcs[i][1])
                # Use to apply constant padding
                bc_flags[tuple(beginning)] = bc_flags[tuple(beginning)] + 1.0
                bc_flags[tuple(end)] = bc_flags[tuple(end)] + 1.0
                # Only necessary for experiment consistency - not for training from scratch.
                beginning[2] = 0
                end[2] = 0
                bc_flags[tuple(beginning)] = 0.0
                bc_flags[tuple(end)] = 0.0
            if self.jitter_patches:
                if _patch_size[i] <= 1:
                    roll_quantities.append(0)
                else:
                    total_pad = (
                        constant_paddings[-2 * i - 1] + periodic_paddings[-2 * i - 1]
                    )
                    half_patch = (
                        total_pad  # // 2
                        if int(bcs[i][0]) != BoundaryCondition["PERIODIC"].value
                        else x.shape[i + dim_offset] // 2
                    )
                    # Compute and log the random roll for this dimension
                    roll_rate = int(torch.randint(-(half_patch - 1), half_patch, ()))
                    # TODO - move this to using random state to avoid compilation issues
                    roll_quantities.append(roll_rate)
                roll_dims.append(i + dim_offset)
        x = torch.cat((x, bc_flags), dim=2)
        # Now roll by the randomly sampled values if jitter_patches is true
        if self.jitter_patches:
            if jitter_override is not None:
                roll_quantities = jitter_override["rolls"][0]
            # Now roll by the randomly sampled values if jitter_patches is true
            x = torch.roll(x, shifts=roll_quantities, dims=roll_dims)
        if sum(periodic_paddings) > 0:
            x = rearrange(x, "t b c ... -> (t b) c ...")
            x = F.pad(x, pad=periodic_paddings, mode="circular")
            x = rearrange(x, "(t b) c ... -> t b c ...", t=T)
        # Use kwargs for optional compatibility with different versions
        jitter_info = {
            "constant_paddings": constant_paddings,
            "periodic_paddings": periodic_paddings,
            "rolls": (roll_quantities, roll_dims),
        }

        return x, jitter_info

    def unjitter(self, x, jitter_info=None, **kwargs):
        if not self.jitter_patches and ("learned_pad" not in kwargs):
            return x
        constant_paddings, periodic_paddings, rolls = (
            jitter_info["constant_paddings"],
            jitter_info["periodic_paddings"],
            jitter_info["rolls"],
        )
        paddings = [-p1 for p1 in periodic_paddings]
        x = F.pad(x, pad=paddings)
        if self.jitter_patches:
            # Reverse the paddings and rolls
            roll_quantities, roll_dims = rolls
            roll_quantities = [-r for r in roll_quantities]
            # Reverse by rolling/padding with negative values
            x = torch.roll(x, shifts=roll_quantities, dims=roll_dims)
        paddings = [-p2 for p2 in constant_paddings]
        x = F.pad(x, pad=paddings)
        return x
