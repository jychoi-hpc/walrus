import pytest
from the_well.data.augmentation import Resize

from walrus.data.multidatamodule import MixedWellDataModule
from walrus.data.multidataset import MixedWellDataset


def test_datamodule(dummy_dataset):
    well_base_path = dummy_dataset
    data_module = MixedWellDataModule(
        well_base_path=well_base_path,
        well_dataset_info={
            "dummy": {
                "path": dummy_dataset / "dummy",
                "include_filters": [],
                "exclude_filters": [],
            },
        },
        batch_size=1,
        data_workers=1,
        max_samples=20,
    )
    assert hasattr(data_module, "train_dataset")
    assert hasattr(data_module, "train_dataloader")
    for batch_index, batch in enumerate(data_module.train_dataloader(), start=1):
        assert "input_fields" in batch
    assert batch_index == data_module.max_samples
    print(batch_index, len(data_module.train_dataloader()), batch["input_fields"].shape)


@pytest.mark.parametrize(
    "restrict_train_num_trajectories, restrict_train_num_samples",
    [(None, None), (1, None), (0.5, None), (None, 10), (None, 0.5)],
)
def test_dataset_restrictions(
    dummy_dataset, restrict_train_num_trajectories, restrict_train_num_samples
):
    well_base_path = dummy_dataset
    data_module = MixedWellDataModule(
        well_base_path=well_base_path,
        well_dataset_info={
            "dummy": {
                "path": dummy_dataset / "dummy",
                "include_filters": [],
                "exclude_filters": [],
            },
        },
        batch_size=1,
        data_workers=1,
        max_samples=99999999,  # Large enough to ensure max is from data
        recycle_datasets=False,
        restrict_train_num_trajectories=restrict_train_num_trajectories,
        restrict_train_num_samples=restrict_train_num_samples,
    )
    train_dataloader = data_module.train_dataloader()
    # Only one entry in each for dummy
    train_dataset = train_dataloader.dataset.sub_dsets[0]
    train_trajectories = train_dataset.n_trajectories_per_file[0]
    train_windows_per_traj = train_dataset.n_windows_per_trajectory[0]
    # Compute what n trajectories should be - weirdness when float*n_traj/n_windows is not int is expected
    if restrict_train_num_trajectories is not None:
        if isinstance(restrict_train_num_trajectories, float):
            target_samples = (
                int(restrict_train_num_trajectories * train_trajectories)
                * train_windows_per_traj
            )
        elif isinstance(restrict_train_num_trajectories, int):
            target_samples = (
                min(restrict_train_num_trajectories, train_trajectories)
                * train_windows_per_traj
            )
        assert len(train_dataloader) == target_samples, (
            f"Expected {target_samples} samples, got {len(train_dataloader)}"
        )
    if restrict_train_num_samples is not None:
        orig_num_samples = train_trajectories * train_windows_per_traj
        if isinstance(restrict_train_num_samples, float):
            target_samples = int(restrict_train_num_samples * orig_num_samples)
        elif isinstance(restrict_train_num_samples, int):
            target_samples = min(restrict_train_num_samples, orig_num_samples)
        assert len(train_dataloader) == target_samples, (
            f"Expected {target_samples} samples, got {len(train_dataloader)}"
        )


def test_dataset_start_valid_at_t(dummy_dataset, start_at_t=4):
    """Test that setting the start prediction point correctly offsets the prediction targets"""
    well_base_path = dummy_dataset
    control_module = MixedWellDataModule(
        well_base_path=well_base_path,
        well_dataset_info={
            "dummy": {
                "path": dummy_dataset / "dummy",
                "include_filters": [],
                "exclude_filters": [],
            },
        },
        batch_size=1,
        data_workers=1,
        n_steps_input=1,
        normalize_time_grid=False,
        start_rollout_valid_output_at_t=-1,
    )
    exp_module = MixedWellDataModule(
        well_base_path=well_base_path,
        well_dataset_info={
            "dummy": {
                "path": dummy_dataset / "dummy",
                "include_filters": [],
                "exclude_filters": [],
            },
        },
        batch_size=1,
        data_workers=1,
        n_steps_input=1,
        normalize_time_grid=False,
        recycle_datasets=False,
        start_rollout_valid_output_at_t=start_at_t,
    )
    control_dset = control_module.rollout_val_datasets[0]
    exp_dset = exp_module.rollout_val_datasets[0]

    control_batch = control_dset[0]
    exp_batch = exp_dset[0]
    # Check that the correct values are used
    assert (
        control_batch["output_time_grid"][0, start_at_t - 1]
        == exp_batch["output_time_grid"][0, 0]
    )
    assert (
        control_batch["output_time_grid"][0, start_at_t - 2]
        == exp_batch["input_time_grid"][0, 0]
    )


@pytest.fixture()
def dummy_resized_dataset(dummy_dataset):
    augmentation = Resize(target_size=16, interpolation_mode="bilinear")
    dataset = MixedWellDataset(
        well_base_path=dummy_dataset,
        well_dataset_info={
            "dummy": {
                "path": dummy_dataset / "dummy",
                "include_filters": [],
                "exclude_filters": [],
            }
        },
        well_split_name="train",
        use_normalization=False,
        n_steps_input=5,
        n_steps_output=1,
        transform=augmentation,
        dataset_kws={"pad_cartesian_data_to_d": 2},
    )
    return dataset


@pytest.fixture()
def dummy_control_dataset(dummy_dataset):
    dataset = MixedWellDataset(
        well_base_path=dummy_dataset,
        well_dataset_info={
            "dummy": {
                "path": dummy_dataset / "dummy",
                "include_filters": [],
                "exclude_filters": [],
            }
        },
        well_split_name="train",
        use_normalization=False,
        n_steps_input=5,
        n_steps_output=1,
        dataset_kws={"pad_cartesian_data_to_d": 2},
    )
    return dataset


def _compare_samples(resized, control, resize_shape):
    """Compare the resized dataset with the control dataset to ensure
    that the resizing has been applied correctly. This function checks
    the shapes of the input fields, output fields, and any constant
    fields to ensure they match the expected dimensions after resizing.
    """
    metadata = resized["metadata"]
    # Target sizes should be the same outside of resized dims
    for key in ["input_fields", "output_fields", "space_grid", "constant_fields"]:
        ctrl = control[key]
        res = resized[key]
        # Check dims before space
        pre_space_ind = -metadata.n_spatial_dims - 1
        assert ctrl.shape[:pre_space_ind] == res.shape[:pre_space_ind], (
            f"Shape mismatch for {key}: {ctrl.shape} vs {res.shape}"
        )
        # Check space dims
        target_size = [
            resize_shape if n > 1 else 1 for n in ctrl.shape[pre_space_ind:-1]
        ]  # Exclude the last dim (usually the channel dim)
        res_size = [n for n in res.shape[pre_space_ind:-1]]
        assert target_size == res_size, (
            f"Shape mismatch for {key}: {target_size} vs {res_size}"
        )
        # Check last dim (usually the channel dim)
        assert ctrl.shape[-1] == res.shape[-1], (
            f"Shape mismatch for {key}: {ctrl.shape} vs {res.shape}"
        )


def test_dummy_resized_dataset(dummy_resized_dataset, dummy_control_dataset):
    resized = dummy_resized_dataset[0]
    control = dummy_control_dataset[0]
    resize_shape = (
        16  # TODO (mm) - should parameterize this, but just doing quick fixes atm
    )
    _compare_samples(resized, control, resize_shape)


@pytest.fixture()
def dummy_resized_datamodule(dummy_dataset):
    augmentation = Resize(target_size=16, interpolation_mode="bilinear")
    data_module = MixedWellDataModule(
        well_base_path=dummy_dataset,
        well_dataset_info={
            "dummy": {
                "path": dummy_dataset / "dummy",
                "include_filters": [],
                "exclude_filters": [],
            }
        },
        data_workers=1,
        batch_size=8,
        use_normalization=False,
        n_steps_input=5,
        n_steps_output=1,
        transform=augmentation,
        dataset_kws={
            "pad_cartesian_data_to_d": 2,
        },
    )
    return data_module


@pytest.fixture()
def dummy_control_datamodule(dummy_dataset):
    data_module = MixedWellDataModule(
        well_base_path=dummy_dataset,
        well_dataset_info={
            "dummy": {
                "path": dummy_dataset / "dummy",
                "include_filters": [],
                "exclude_filters": [],
            }
        },
        data_workers=1,
        batch_size=8,
        use_normalization=False,
        n_steps_input=5,
        n_steps_output=1,
        dataset_kws={
            "pad_cartesian_data_to_d": 2,
        },
    )
    return data_module


def test_dummy_resized_datamodule(dummy_resized_datamodule, dummy_control_datamodule):
    """Check if the datamodule is producing batches of the correct size and shape
    for both the resized and control datasets."""
    # First train
    resized_trainloader = dummy_resized_datamodule.train_dataloader()
    control_trainloader = dummy_control_datamodule.train_dataloader()
    resized_train_batch = next(iter(resized_trainloader))
    control_train_batch = next(iter(control_trainloader))
    resize_shape = (
        16  # TODO (mm) - should parameterize this, but just doing quick fixes atm
    )
    _compare_samples(resized_train_batch, control_train_batch, resize_shape)

    # Then val - we assume val is the same for both datasets
    resized_val_dataloaders = dummy_resized_datamodule.val_dataloaders()
    control_val_dataloaders = dummy_control_datamodule.val_dataloaders()
    for resized_val, control_val in zip(
        resized_val_dataloaders, control_val_dataloaders
    ):
        resized_val_batch = next(iter(resized_val))
        control_val_batch = next(iter(control_val))
        _compare_samples(resized_val_batch, control_val_batch, resize_shape)

    # Then rollout val
    resized_rollout_dataloaders = dummy_resized_datamodule.rollout_val_dataloaders()
    control_rollout_dataloaders = dummy_control_datamodule.rollout_val_dataloaders()
    for resized_rollout, control_rollout in zip(
        resized_rollout_dataloaders, control_rollout_dataloaders
    ):
        resized_rollout_batch = next(iter(resized_rollout))
        control_rollout_batch = next(iter(control_rollout))
        _compare_samples(resized_rollout_batch, control_rollout_batch, resize_shape)
    # Test uses the same path as val so ignore for now


def test_dataset_kws_from_hydra_config(dummy_dataset):
    """dataset_kws given as an OmegaConf DictConfig (as Hydra passes it) reaches the datasets."""
    from omegaconf import OmegaConf

    data_module = MixedWellDataModule(
        well_base_path=dummy_dataset,
        well_dataset_info={
            "dummy": {"path": dummy_dataset / "dummy", "include_filters": [], "exclude_filters": []},
        },
        batch_size=1,
        data_workers=0,
        dataset_kws=OmegaConf.create({"slab": [0, 8, 24]}),
    )
    sample = data_module.train_dataset[[0]]
    assert sample["input_fields"].shape[2] == 16  # B T [X] ...: only the slab's 16 x-points
