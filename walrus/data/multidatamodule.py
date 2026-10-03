import logging
from functools import partial
from typing import Dict, List, Literal, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
from the_well.data import WellDataset
from the_well.data.augmentation import Augmentation
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    DistributedSampler,
    RandomSampler,
    Sampler,
)
from torch.utils.data._utils.collate import default_collate

from walrus.data.inflated_dataset import (
    BatchInflatedWellDataset,
)
from walrus.utils.spatial import get_spatial_context

from .mixed_dset_sampler import BatchedMultisetSampler
from .prefetch import ThreadPrefetchLoader
from .multidataset import MixedWellDataset
from .utils import get_dict_depth

logger = logging.getLogger(__name__)


def metadata_aware_collate(batch):
    """Collate function that is aware of the metadata of the dataset."""
    # Metadata constant per batch
    metadata = batch[0]["metadata"]
    # Remove metadata from current dicts
    [sample.pop("metadata") for sample in batch]
    batch = default_collate(batch)  # Returns stacked dictionary
    batch["metadata"] = metadata
    return batch


class MixedWellDataModule:
    def __init__(
        self,
        *,
        well_base_path: str,
        well_dataset_info: Dict[
            str,
            Dict[
                Literal[
                    "include_filters", "exclude_filters", "path", "field_transforms"
                ],
                List[str] | str | Dict[str, str],
            ],
        ],
        batch_size: int,
        use_normalization: bool = False,
        field_index_map_override: Dict[str, int] = {},
        return_grid: bool = True,
        normalize_time_grid: bool = True,
        prefetch_field_names: bool = True,
        max_rollout_steps: int = 100,
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        min_dt_stride: int = 1,
        max_dt_stride: int = 1,
        world_size: int = 1,
        rank: int = 1,
        data_workers: int = 4,
        inner_dataset_type: WellDataset = BatchInflatedWellDataset,
        epoch: int = 0,
        max_samples: int = 2000,
        recycle_datasets: bool = True,
        start_rollout_valid_output_at_t: int = -1,
        restrict_train_num_trajectories: Optional[float | int] = None,
        restrict_train_num_samples: Optional[float | int] = None,
        restriction_seed: int = 0,
        allow_sharding_in_train: bool = False,
        prefetch_factor: Optional[int] = None,
        pin_memory: bool = True,
        eval_data_workers: Optional[int] = None,
        transform: Optional[
            Union[
                Augmentation,
                Dict[str, Augmentation],
                Dict[
                    Literal["train", "val", "rollout_val", "test", "rollout_test"],
                    Dict[str, Augmentation],
                ],
            ]
        ] = None,
        global_field_transforms: Optional[Dict[str, str]] = {},
        storage_kwargs: Optional[Dict] = None,
        dataset_kws: Optional[
            Union[
                dict,
                Dict[str, dict],
                Dict[
                    Literal["train", "val", "rollout_val", "test", "rollout_test"],
                    Dict[str, dict],
                ],
            ]
        ] = None,
    ):
        """Data module class to yield batches of samples.

        Parameters
        ----------
        well_base_path:
            Path to the base directory for the Well dataset.
        well_dataset_info:
            Dictionary containing for each dataset:
            - include_filters: List of strings to filter files to include
            - exclude_filters: List of strings to filter files to exclude
            - path: Optional custom path for this specific dataset
            - field_transforms: Optional dictionary of field transformations
        batch_size:
            Size of the batches yielded by the dataloaders
        use_normalization:
            Whether or not to use normalization.
        field_index_map_override:
            Optional dictionary to override the field to index mapping. Useful when loading an old checkpoint.
        return_grid:
            Whether or not to return the grid information.
        normalize_time_grid:
            Whether or not to normalize the time grid to start at zero.
        prefetch_field_names:
            Whether or not to prefetch the field names from the datasets. Your code
            will probably break if you set this to False and aren't loading a pretrained
            model.
        max_rollout_steps:
            Maximum number of steps to rollout during full trajectory validation and testing.
        n_steps_input:
            Number of input steps to use.
        n_steps_output:
            Number of output steps to use as target.
        min_dt_stride:
            Minimum stride to use when sampling in the time dimension.
        max_dt_stride:
            Maximum stride to use when sampling in the time dimension.
        world_size:
            Number of total processes in the distributed setting.
        rank:
            Rank of the current GPU in the full torchrun world.
        data_workers:
            Number of workers for the dataloaders in the given process.
        inner_dataset_type:
            Type of the inner dataset to use. This is the dataset that loads individual files.
            It must be a subclass of WellDataset.
        epoch:
            Current epoch number.
        max_samples:
            Maximum number of samples to use for a single training loop.
        recycle_datasets:
            Whether or not to allow sampling from sub-datasets who have already been fully used in this epoch. Changes
            sampling to be more similar to "with replacement" sampling of sub-datasets such that "max_samples" now
            defines the epoch length.
        start_rollout_valid_output_at_t:
            During rollout validation, start outputting predictions at this time index.
            This is useful to compare models that require different initial context to make accurate predictions.
            Default is -1, which means to start at the beginning of the trajectory.
        restrict_train_num_trajectories:
            If set to a float in (0, 1), restrict the number of trajectories in the training set to this fraction of the total.
            If set to an int > 1, restrict the number of trajectories in the training set to this number.
            If None, use all trajectories.
        restrict_train_num_samples:
            If set to a float in (0, 1), restrict the number of samples in the training set to this fraction of the total.
            If set to an int > 1, restrict the number of samples in the training set to this number.
            If None, use all samples.
        restriction_seed:
            Random seed to use when restricting the number of trajectories or samples in the training set.
        allow_sharding_in_train:
            Whether to allow sharding in the training set. If True and dataset size ~ replication size,
            drop indices can result in large percentage of data dropped.
        prefetch_factor:
            Batches each loader worker prepares ahead (None: PyTorch's
            default of 2). Each is a full batch in host memory, so at large
            grids a lower value lets more workers fit.
        pin_memory:
            Copy batches to pinned host memory for faster transfers to the GPU
            (one more host copy of each batch).
        eval_data_workers:
            Loader workers for validation and test (None: data_workers).
            Rollout validation loads whole trajectories, so at large grids
            fewer workers keep host memory in bounds.
        transform:
            Transformations to apply to the data.
        storage_kwargs:
            Storage options passed to fsspec for accessing the raw data.
        dataset_kws:
            Additional keyword arguments to pass to each dataset.
        """
        self.global_field_transforms = global_field_transforms or {}
        if transform is not None:
            # If transform is a single Augmentation, apply it to all datasets
            if isinstance(transform, Augmentation):
                transform = {dataset: transform for dataset in well_dataset_info.keys()}

            # If transform is a Dict[str, Augmentation], apply it to all splits
            if isinstance(transform, dict) and all(
                isinstance(k, str) and isinstance(v, Augmentation)
                for k, v in transform.items()
            ):
                transform = {
                    data_split: transform
                    for data_split in [
                        "train",
                        "val",
                        "rollout_val",
                        "test",
                        "rollout_test",
                    ]
                }

            # If transform keys are not a subset of train, val, rollout_val, test, rollout_test, raise an error
            assert set(transform.keys()).issubset(
                set(["train", "val", "rollout_val", "test", "rollout_test"])
            ), (
                f"Expected transform keys {transform.keys()} to be a subset of train, val, rollout_val, test, rollout_test."
            )

        if dataset_kws is not None:
            # Hydra passes configs as DictConfig
            if isinstance(dataset_kws, DictConfig):
                dataset_kws = OmegaConf.to_container(dataset_kws, resolve=True)
            # If dataset_kws is not a dict, raise an error
            if not isinstance(dataset_kws, dict):
                raise ValueError(
                    f"Expected dataset_kws to be None or a dict, got {type(dataset_kws)}."
                )

            # If dataset_kws is a single dict with depth 1, apply it to all datasets
            if isinstance(dataset_kws, dict) and get_dict_depth(dataset_kws) == 1:
                dataset_kws = {
                    dataset: dataset_kws for dataset in well_dataset_info.keys()
                }

            # If dataset_kws is a dict of dicts with depth 2, apply it to all splits
            if (
                isinstance(dataset_kws, dict)
                and all(
                    isinstance(k, str) and isinstance(v, dict)
                    for k, v in dataset_kws.items()
                )
                and get_dict_depth(dataset_kws) == 2
            ):
                dataset_kws = {
                    data_split: dataset_kws
                    for data_split in [
                        "train",
                        "val",
                        "rollout_val",
                        "test",
                        "rollout_test",
                    ]
                }

            # If dataset_kws keys are not a subset of train, val, rollout_val, test, rollout_test, raise an error
            assert set(dataset_kws.keys()).issubset(
                set(["train", "val", "rollout_val", "test", "rollout_test"])
            ), (
                f"Expected dataset_kws keys {dataset_kws.keys()} to be a subset of train, val, rollout_val, test, rollout_test."
            )
        self.allow_sharding_in_train = allow_sharding_in_train
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.eval_data_workers = data_workers if eval_data_workers is None else eval_data_workers
        # > 0: the training loader runs without workers and a background thread
        # fetches this many batches ahead (set when fields come from DDStore)
        self.train_thread_prefetch = 0
        # Train is a single mixed dataset
        self.train_dataset = MixedWellDataset(
            well_base_path=well_base_path,
            well_dataset_info=well_dataset_info,
            well_split_name="train",
            use_normalization=use_normalization,
            return_grid=return_grid,
            normalize_time_grid=normalize_time_grid,
            n_steps_input=n_steps_input,
            n_steps_output=n_steps_output,
            min_dt_stride=min_dt_stride,
            max_dt_stride=max_dt_stride,
            restrict_num_trajectories=restrict_train_num_trajectories,
            restrict_num_samples=restrict_train_num_samples,
            restriction_seed=restriction_seed,
            prefetch_field_names=prefetch_field_names,
            transform=transform["train"]
            if transform is not None and "train" in transform
            else None,
            global_field_transforms=self.global_field_transforms,
            storage_options=storage_kwargs,
            field_index_map_override=field_index_map_override,
            inner_dataset_type=inner_dataset_type,
            dataset_kws=dataset_kws["train"]
            if dataset_kws is not None and "train" in dataset_kws
            else None,
        )
        # In Val/Test, we want stats for each dataset
        # but we still use MixedWellDataset to handle the extra info (field indices, etc.)
        self.val_datasets = [
            MixedWellDataset(
                well_base_path=well_base_path,
                well_dataset_info={dset_name: well_dataset_info[dset_name]},
                well_split_name="valid",
                use_normalization=use_normalization,
                return_grid=return_grid,
                normalize_time_grid=normalize_time_grid,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                min_dt_stride=min_dt_stride,
                max_dt_stride=min_dt_stride,
                transform=transform["val"]  # [dset_name]
                if transform is not None and "val" in transform
                # and dset_name in transform["val"]
                else None,
                global_field_transforms=self.global_field_transforms,
                storage_options=storage_kwargs,
                prefetch_field_names=False,  # Use same from train
                field_index_map_override=self.train_dataset.field_to_index_map,
                inner_dataset_type=inner_dataset_type,
                dataset_kws=dataset_kws["val"]  # [dset_name]
                if dataset_kws is not None
                and "val" in dataset_kws
                and dset_name in dataset_kws["val"]
                else None,
            )
            for dset_name in well_dataset_info
        ]

        self.rollout_val_datasets = [
            MixedWellDataset(
                well_base_path=well_base_path,
                well_dataset_info={dset_name: well_dataset_info[dset_name]},
                well_split_name="valid",
                use_normalization=use_normalization,
                return_grid=return_grid,
                normalize_time_grid=normalize_time_grid,
                max_rollout_steps=max_rollout_steps,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                full_trajectory_mode=True,
                min_dt_stride=min_dt_stride,
                max_dt_stride=min_dt_stride,
                start_output_steps_at_t=start_rollout_valid_output_at_t,
                transform=transform["rollout_val"]  # [dset_name]
                if transform is not None and "rollout_val" in transform
                # and dset_name in transform["rollout_val"]
                else None,
                global_field_transforms=self.global_field_transforms,
                storage_options=storage_kwargs,
                prefetch_field_names=False,  # Use same from train
                field_index_map_override=self.train_dataset.field_to_index_map,
                inner_dataset_type=inner_dataset_type,
                dataset_kws=dataset_kws["rollout_val"][dset_name]
                if dataset_kws is not None
                and "rollout_val" in dataset_kws
                and dset_name in dataset_kws["rollout_val"]
                else None,
            )
            for dset_name in well_dataset_info
        ]

        self.test_datasets = [
            MixedWellDataset(
                well_base_path=well_base_path,
                well_dataset_info={dset_name: well_dataset_info[dset_name]},
                well_split_name="test",
                use_normalization=use_normalization,
                return_grid=return_grid,
                normalize_time_grid=normalize_time_grid,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                min_dt_stride=min_dt_stride,
                max_dt_stride=min_dt_stride,
                transform=transform["test"]  # [dset_name]
                if transform is not None and "test" in transform
                # and dset_name in transform["test"]
                else None,
                global_field_transforms=self.global_field_transforms,
                storage_options=storage_kwargs,
                prefetch_field_names=False,  # Use same from train
                field_index_map_override=self.train_dataset.field_to_index_map,
                inner_dataset_type=inner_dataset_type,
                dataset_kws=dataset_kws["test"][dset_name]
                if dataset_kws is not None
                and "test" in dataset_kws
                and dset_name in dataset_kws["test"]
                else None,
            )
            for dset_name in well_dataset_info
        ]

        self.rollout_test_datasets = [
            MixedWellDataset(
                well_base_path=well_base_path,
                well_dataset_info={dset_name: well_dataset_info[dset_name]},
                well_split_name="test",
                use_normalization=use_normalization,
                return_grid=return_grid,
                normalize_time_grid=normalize_time_grid,
                max_rollout_steps=max_rollout_steps,
                n_steps_input=n_steps_input,
                n_steps_output=n_steps_output,
                full_trajectory_mode=True,
                min_dt_stride=min_dt_stride,
                max_dt_stride=min_dt_stride,
                start_output_steps_at_t=start_rollout_valid_output_at_t,
                transform=transform["rollout_test"]  # [dset_name]
                if transform is not None and "rollout_test" in transform
                # and dset_name in transform["rollout_test"]
                else None,
                global_field_transforms=self.global_field_transforms,
                storage_options=storage_kwargs,
                prefetch_field_names=False,  # Use same from train
                field_index_map_override=self.train_dataset.field_to_index_map,
                inner_dataset_type=inner_dataset_type,
                dataset_kws=dataset_kws["rollout_test"][dset_name]
                if dataset_kws is not None
                and "rollout_test" in dataset_kws
                and dset_name in dataset_kws["rollout_test"]
                else None,
            )
            for dset_name in well_dataset_info
        ]
        self.batch_size = batch_size
        self.world_size = world_size
        self.data_workers = data_workers
        self.rank = rank
        self.epoch = epoch
        self.max_samples = max_samples
        self.recycle = recycle_datasets

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def train_dataloader(self, rank_override=None) -> DataLoader:
        rank = self.rank if rank_override is None else rank_override
        if self.allow_sharding_in_train and self.is_distributed:
            base_sampler: type[Sampler] = DistributedSampler
            if get_spatial_context() is not None:
                # Shard over spatial groups, not over all GPUs
                base_sampler = partial(
                    DistributedSampler, num_replicas=self.world_size, rank=rank
                )
        elif get_spatial_context() is not None:
            # GPUs of a spatial group must draw the same samples, so seed the
            # shuffle by group instead of each process's global RNG
            base_sampler = partial(
                RandomSampler, generator=torch.Generator().manual_seed(rank)
            )
        else:
            base_sampler = RandomSampler

        sampler = BatchedMultisetSampler(
            self.train_dataset,
            base_sampler,
            self.batch_size,  # seed=seed,
            distributed=self.is_distributed,
            max_samples=self.max_samples,  # TODO Fix max_samples later
            recycle=self.recycle,
            drop_last=False,
            rank=rank,
        )

        shuffle = sampler is None
        workers = 0 if self.train_thread_prefetch > 0 else self.data_workers

        loader = DataLoader(
            self.train_dataset,
            num_workers=workers,
            pin_memory=self.pin_memory,
            # Batches each worker prepares ahead (PyTorch default 2). Every one
            # is a full batch in host memory, which limits workers at large grids.
            prefetch_factor=self.prefetch_factor if workers > 0 else None,
            batch_size=None,
            shuffle=shuffle,
            # drop_last=True,
            sampler=sampler,
            collate_fn=None,
        )
        if self.train_thread_prefetch > 0:
            # DDStore reads cannot run in forked workers; a thread fetches ahead
            return ThreadPrefetchLoader(loader, self.train_thread_prefetch)
        return loader

    def build_loaders_from_dset_list(
        self, dset_list, batch_size=1, replicas=None, rank=None, full=True
    ) -> List[DataLoader]:
        dataloaders = []
        for dataset in dset_list:
            # If distributed, don't replicate across GPUs
            if self.is_distributed:
                # However, for large enough worlds, we need drop_last=False which causes some replication
                sampler: Sampler = BatchSampler(
                    DistributedSampler(
                        dataset,
                        seed=0,
                        drop_last=False,
                        shuffle=not full,  # If doing everyhing
                        num_replicas=replicas,  # World size is default if replicas is None otherwise pass size of sync (FSDP) group
                        rank=rank,
                    ),  # Global rank is default if rank is None - otherwise pass within sync (FSDP) group rank
                    batch_size=batch_size,
                    drop_last=False,
                )
            else:
                sampler = BatchSampler(
                    RandomSampler(dataset, generator=torch.Generator().manual_seed(0)),
                    batch_size=batch_size,
                    drop_last=False,
                )

            dataloaders.append(
                DataLoader(
                    dataset,
                    num_workers=self.eval_data_workers,
                    pin_memory=self.pin_memory,
                    prefetch_factor=(
                        self.prefetch_factor if self.eval_data_workers > 0 else None
                    ),
                    batch_size=None,
                    shuffle=None,  # Sampler is set
                    sampler=sampler,
                    collate_fn=None,
                )
            )
        return dataloaders

    def val_dataloaders(
        self,
        replicas: Optional[int] = None,
        rank: Optional[int] = None,
        full: bool = False,
    ) -> List[DataLoader]:
        return self.build_loaders_from_dset_list(
            self.val_datasets, self.batch_size, replicas, rank, full
        )

    def rollout_val_dataloaders(
        self,
        replicas: Optional[int] = None,
        rank: Optional[int] = None,
        full: bool = False,
    ) -> List[DataLoader]:
        return self.build_loaders_from_dset_list(
            self.rollout_val_datasets,
            1,  # Batch size hardcoded to one since 3D data uses so much memory - can be fixed, but not priority
            replicas,
            rank,
            full,
        )

    def test_dataloaders(
        self,
        replicas: Optional[int] = None,
        rank: Optional[int] = None,
        full: bool = True,
    ) -> List[DataLoader]:
        return self.build_loaders_from_dset_list(
            self.test_datasets, self.batch_size, replicas, rank, full
        )

    def rollout_test_dataloaders(
        self,
        replicas: Optional[int] = None,
        rank: Optional[int] = None,
        full: bool = True,
    ) -> List[DataLoader]:
        return self.build_loaders_from_dset_list(
            self.rollout_test_datasets,
            1,  # Batch size hardcoded to one since 3D data uses so much memory - can be fixed, but not priority
            replicas,
            rank,
            full,
        )


if __name__ == "__main__":
    well_base_path = "/mnt/home/polymathic/ceph/the_well/"
    data = MixedWellDataModule(
        well_base_path=well_base_path,
        well_dataset_info={
            "active_matter": {"include_filters": [], "exclude_filters": []},
            "planetswe": {"include_filters": [], "exclude_filters": []},
        },
        batch_size=32,
        data_workers=4,
    )

    for x in data.train_dataloader():
        print(x)
        break
