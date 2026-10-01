from dataclasses import dataclass
from typing import Any, Optional

import torch
import yaml
from the_well.data.utils import flatten_field_names

from walrus.data.multidataset import MixedWellDataset
from walrus.data.well_to_multi_transformer import AbstractFormatter
from walrus.trainer.spatial_reductions import global_mean, global_std_mean
from walrus.utils.spatial import SpatialContext


@dataclass
class NormalizationStats:
    """
    Class to store normalization statistics for a samplewise normalization.
    """

    sample_mean: torch.Tensor
    sample_std: torch.Tensor
    # sample_norm: torch.Tensor
    delta_mean: torch.Tensor
    delta_std: torch.Tensor
    # delta_norm: torch.Tensor
    epsilon: float = 1e-5


class BaseRevNormalization:
    # Set when each GPU holds one slab of the domain: samplewise statistics
    # are then taken over the whole domain across the spatial group.
    spatial_ctx: Optional[SpatialContext] = None

    def __init__(self, *args, **kwargs):
        pass

    def _split(self) -> bool:
        return self.spatial_ctx is not None and self.spatial_ctx.size > 1

    def _mean(self, x: torch.Tensor, dims) -> torch.Tensor:
        if self._split():
            return global_mean(x, dims, self.spatial_ctx)
        return x.mean(dims, keepdim=True)

    def _std_mean(self, x: torch.Tensor, dims):
        if self._split():
            return global_std_mean(x, dims, self.spatial_ctx)
        return torch.std_mean(x, dims, keepdim=True)

    def compute_stats(
        self, x: torch.Tensor, metadata, epsilon: float = 1e-5
    ) -> NormalizationStats:
        """
        Compute normalization statistics for a batch of data.

        Note - channels are always assumed to be in the 2 dimension.
        """
        raise NotImplementedError

    def get_dims_from_metadata(self, metadata) -> tuple:
        # data assumed to be in T B C H [W D] - MPP format
        if metadata.n_spatial_dims == 1:
            return (0, 3)
        elif metadata.n_spatial_dims == 2:
            return (0, 3, 4)
        else:
            assert metadata.n_spatial_dims == 3
            return (0, 3, 4, 5)

    def normalize_stdmean(
        self,
        x: torch.Tensor,
        stats: NormalizationStats,
        reshape_key: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Normalize data using the samplewise mean and std.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            n_channels = x.shape[2]
            # print("shapes at norm", x.shape, stats.sample_std.shape, n_channels)
            # Restrict channels to those in the input - assumed any extra input channels are last channels
            return (x - stats.sample_mean[:, :, :n_channels]) / stats.sample_std[
                :, :, :n_channels
            ]

    def normalize_delta(
        self,
        x: torch.Tensor,
        stats: NormalizationStats,
        reshape_key: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Normalize data using the delta mean and std.
        """
        # x - T B C H [W D]
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            n_channels = x.shape[2]
            # Restrict channels to those in the input - assumed any extra input channels are last channels
            return (x - stats.delta_mean[:, :, :n_channels]) / stats.delta_std[
                :, :, :n_channels
            ]

    def denormalize_stdmean(
        self, x: torch.Tensor, stats: NormalizationStats
    ) -> torch.Tensor:
        """
        Denormalize data using the samplewise mean and std.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            n_channels = x.shape[2]
            # Restrict channels to those in the input - assumed any extra input channels are last channels
            return (
                x * stats.sample_std[:, :, :n_channels]
                + stats.sample_mean[:, :, :n_channels]
            )

    def denormalize_delta(
        self, x: torch.Tensor, stats: NormalizationStats
    ) -> torch.Tensor:
        """
        Denormalize data using the delta mean and std.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            n_channels = x.shape[2]
            if stats.delta_std.shape[2] < n_channels:
                out = x.clone()
                out[:, :, : stats.delta_std.shape[2]] = (
                    x[:, :, : stats.delta_std.shape[2]] * stats.delta_std
                    + stats.delta_mean
                )
                return out
            # Restrict channels to those in the input - assumed any extra input channels are last channels
            return (
                x * stats.delta_std[:, :, :n_channels]
                + stats.delta_mean[:, :, :n_channels]
            )


class GlobalRevNormBase(BaseRevNormalization):
    def _flatten_fields(
        self, flattened_field_names, name_to_norm_dict, statistic_name, metadata
    ) -> str:
        norm = name_to_norm_dict[statistic_name]
        seen_fields = set()
        output_stats = []
        # print("flattened_field_names", flattened_field_names)
        # print("name_to_norm_dict", name_to_norm_dict)
        for field_name in flattened_field_names:
            # print("before check", field_name)
            if "_" in field_name and "".join(field_name.rsplit("_", 1)[:-1]) in norm:
                # print("after check", field_name)
                flat_name = "".join(field_name.rsplit("_", 1)[:-1])
                if flat_name not in seen_fields:
                    order = len(torch.tensor(norm[flat_name]).shape)
                    repeat = metadata.n_spatial_dims**order
                    flat_norms = (
                        torch.tensor(norm[flat_name]).square().sum().sqrt()
                        / repeat**0.5
                    )
                    output_stats += [flat_norms] * repeat
                    # print("order, repeat, flat_norms", order, repeat, flat_norms, flat_name)
                    seen_fields.add(flat_name)
            elif field_name in norm:
                output_stats.append(norm[field_name])
        output_stats = torch.tensor(output_stats)[None, None, :]
        for i in range(metadata.n_spatial_dims):
            output_stats = output_stats.unsqueeze(-1)
        # print("output_stats", output_stats.shape)
        return output_stats

    def compute_stats(self, x, metadata, epsilon=1e-5):
        return self.name_to_normalization_stats[metadata.dataset_name]


class MeanStdGlobalRevNormalization(GlobalRevNormBase):
    """
    Module computes normalization and inverts normalization for computing loss statistics

    Data assumed to be in T B C H [W D] format for consistency with MPP repo models
    """

    def __init__(self, train_dataset: MixedWellDataset, device):
        super().__init__()
        # Note - we're assuming no rotations - every field is normalized as though it is uniform
        # for ds in train_dataset.sub_dsets:
        #     print( yaml.load(open(ds.normalization_path), Loader=yaml.SafeLoader))
        self.name_to_normalization_stats = {
            ds.dataset_name: NormalizationStats(
                sample_mean=self._flatten_fields(
                    flatten_field_names(ds.metadata, include_constants=True),
                    yaml.load(open(ds.normalization_path), Loader=yaml.SafeLoader),
                    "mean",
                    ds.metadata,
                ).to(device),
                sample_std=self._flatten_fields(
                    flatten_field_names(ds.metadata, include_constants=True),
                    yaml.load(open(ds.normalization_path), Loader=yaml.SafeLoader),
                    "std",
                    ds.metadata,
                ).to(device),
                delta_mean=self._flatten_fields(
                    flatten_field_names(ds.metadata, include_constants=False),
                    yaml.load(open(ds.normalization_path), Loader=yaml.SafeLoader),
                    "mean_delta",
                    ds.metadata,
                ).to(device),
                delta_std=self._flatten_fields(
                    flatten_field_names(ds.metadata, include_constants=False),
                    yaml.load(open(ds.normalization_path), Loader=yaml.SafeLoader),
                    "std_delta",
                    ds.metadata,
                ).to(device),
            )
            for ds in train_dataset.sub_dsets
        }


class RMSGlobalRevNormalization(GlobalRevNormBase):
    """
    Module computes normalization and inverts normalization for computing loss statistics

    Data assumed to be in T B C H [W D] format for consistency with MPP repo models
    """

    def __init__(self, train_dataset: MixedWellDataset, device):
        super().__init__()
        # Note - we're assuming no rotations - every field is normalized as though it is uniform
        self.name_to_normalization_stats = {
            ds.dataset_name: NormalizationStats(
                sample_mean=torch.zeros_like(
                    self._flatten_fields(
                        flatten_field_names(ds.metadata, include_constants=True),
                        yaml.load(open(ds.normalization_path), Loader=yaml.SafeLoader),
                        "rms",  # use RMS as base size so we don't need a field we dont use
                        ds.metadata,
                    )
                ).to(device),
                sample_std=self._flatten_fields(
                    flatten_field_names(ds.metadata, include_constants=True),
                    yaml.load(open(ds.normalization_path), Loader=yaml.SafeLoader),
                    "rms",
                    ds.metadata,
                ).to(device),
                delta_mean=torch.zeros_like(
                    self._flatten_fields(
                        flatten_field_names(ds.metadata, include_constants=False),
                        yaml.load(open(ds.normalization_path), Loader=yaml.SafeLoader),
                        "rms_delta",  # use RMS as base size so we don't need a field we dont use
                        ds.metadata,
                    )
                ).to(device),
                delta_std=self._flatten_fields(
                    flatten_field_names(ds.metadata, include_constants=False),
                    yaml.load(open(ds.normalization_path), Loader=yaml.SafeLoader),
                    "rms_delta",
                    ds.metadata,
                ).to(device),
            )
            for ds in train_dataset.sub_dsets
        }


class MeanStdSamplewiseRevNormalization(BaseRevNormalization):
    """
    Module computes normalization and inverts normalization for computing loss statistics

    Data assumed to be in T B C H [W D] format for consistency with MPP repo models
    """

    def compute_stats(
        self, x: torch.Tensor, metadata, epsilon: float = 1e-5
    ) -> NormalizationStats:
        """
        Compute normalization statistics for a batch of data.

        Note - channels are always assumed to be in the 2 dimension.
        """
        # x - T B C H [W D] - MPP format
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            dims = self.get_dims_from_metadata(metadata)
            # Compute samplewise mean and std
            sample_std, sample_mean = self._std_mean(x, dims)
            sample_std = torch.maximum(
                sample_std, torch.tensor(epsilon, device=sample_std.device)
            )
            # Compute delta mean and std
            # assert x.shape[0] > 1, "Cannot compute delta with only one time frame"
            if x.shape[0] > 1:
                deltas = x[1:] - x[:-1]  # u_t - u_{t-1}
                delta_std, delta_mean = self._std_mean(deltas, dims)
                delta_std = torch.maximum(
                    delta_std, torch.tensor(epsilon, device=delta_std.device)
                )
            else:
                # print("(REPLACE WITH WARNING): Only one sample - delta stats defaulting to zero/1")
                delta_std = torch.ones_like(sample_std)
                delta_mean = torch.zeros_like(sample_mean)

            return NormalizationStats(
                sample_mean, sample_std, delta_mean, delta_std, epsilon
            )


class RMSSamplewiseRevNormalization(MeanStdSamplewiseRevNormalization):
    """
    Module computes normalization and inverts normalization for computing loss statistics

    Data assumed to be in T B C H [W D] format for consistency with MPP repo models
    """

    def compute_stats(
        self, x: torch.Tensor, metadata, epsilon: float = 1e-5
    ) -> NormalizationStats:
        """
        Compute normalization statistics for a batch of data.

        Note - channels are always assumed to be in the 2 dimension.
        """
        # x - T B C H [W D] - MPP format
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            dims = self.get_dims_from_metadata(metadata)
            # Compute samplewise mean and std
            sample_std = self._mean(x.square(), dims).sqrt()
            sample_mean = torch.zeros_like(sample_std)
            sample_std = torch.maximum(
                sample_std, torch.tensor(epsilon, device=sample_std.device)
            )
            # Compute delta mean and std
            # assert x.shape[0] > 1, "Cannot compute delta with only one time frame"
            if x.shape[0] > 1:
                deltas = x[1:] - x[:-1]  # u_t - u_{t-1}
                delta_std = self._mean(deltas.square(), dims).sqrt()
                delta_mean = torch.zeros_like(delta_std)
                delta_std = torch.maximum(
                    delta_std, torch.tensor(epsilon, device=delta_std.device)
                )
            else:
                # print("(REPLACE WITH WARNING): Only one sample - delta stats defaulting to zero/1")
                delta_std = torch.ones_like(sample_std)
                delta_mean = torch.zeros_like(sample_mean)
            return NormalizationStats(
                sample_mean, sample_std, delta_mean, delta_std, epsilon
            )


# Aliases for backwards compat
GlobalRevNormalization = RMSGlobalRevNormalization
SamplewiseRevNormalization = RMSSamplewiseRevNormalization


def normalize_target(
    y_ref: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    formatter: AbstractFormatter,
    metadata: Any,
    device: Any,
) -> torch.Tensor:
    """Helper function to assist in computing targets since this is done in multiple paths.

    1. Transform means/stds from the model format to the validation format
    2. Moves target to device
    3. Normalizes target using reformatted mean/std
    """
    with torch.autocast(device_type=device.type, enabled=False):
        y_ref = y_ref.float()
        mean = mean.float()
        mean = formatter.process_output(mean, metadata)[..., : y_ref.shape[-1]]
        std = std.float()
        std = formatter.process_output(std, metadata)[..., : y_ref.shape[-1]]
        y_ref = (y_ref - mean) / std
        return y_ref
