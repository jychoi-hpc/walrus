import gc
import logging
import os
import pickle
import time
from concurrent.futures import Future
from contextlib import nullcontext
from random import shuffle
from subprocess import CalledProcessError
from typing import Any, Callable, Literal, Optional

import numpy as np
import torch
import torch.distributed as dist
import wandb
from the_well.benchmark.metrics import (
    VRMSE,
    make_video,
    validation_plots,
)
from the_well.benchmark.metrics.common import Metric
from the_well.data.datamodule import AbstractDataModule
from the_well.data.datasets import WellDataset
from the_well.data.utils import flatten_field_names
from torch.amp.grad_scaler import GradScaler
from torch.distributed.fsdp.api import ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader

from walrus.trainer.checkpoints import CheckPointLoader
from walrus.trainer.normalization_strat import (
    BaseRevNormalization,
    normalize_target,
)
from walrus.trainer.spatial_reductions import spatial_mean_loss
from walrus.utils.spatial import get_spatial_context

logger = logging.getLogger(__name__)


def log_all_time_metrics(
    time_logs: dict,
    output_dir: str,
    epoch_number: str = "0",
):
    """Plot loss over time for all time metrics.

    Args:
        time_logs: Dict of time metrics
        metadata: Metadata object associated with dset
        output_dir: Directory to save the plots
        epoch_number: Current epoch number
    """
    for dataset_name, logs in time_logs.items():
        os.makedirs(
            f"{output_dir}/{dataset_name}/rollout_losses/epoch_{epoch_number}",
            exist_ok=True,
        )
        print(
            "output_dir",
            f"{output_dir}/{dataset_name}/rollout_losses/epoch_{epoch_number}",
        )
        for k, v in logs.items():
            v = np.array(v)
            title = k.split("/")[-1]
            np.save(
                f"{output_dir}/{dataset_name}/rollout_losses/epoch_{epoch_number}/{title}.npy",
                v,
            )


def expand_mask_to_match(mask, target):
    """Expand mask of shape B H [W D] 1 to
    broadcast with tensor of given shape B T H [W D] C"""
    T = target.shape[1]
    C = target.shape[-1]
    expansion_tuple = (
        -1,
        T,
    )
    expansion_tuple = expansion_tuple + (-1,) * (len(target.shape) - 3) + (C,)
    mask = mask.unsqueeze(1).expand(*expansion_tuple)
    return mask


def get_grad_norm_local(model) -> torch.Tensor:
    """Computes grad norm for the specific device

    From https://github.com/pytorch/pytorch/issues/88621"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            local_norm = torch.linalg.vector_norm(p.grad, dtype=p.dtype)
            total_norm += local_norm**2
    return total_norm**0.5


def get_grad_norm_fsdp(
    model, rank, world_size, sharding_strategy=ShardingStrategy.FULL_SHARD
) -> torch.Tensor:
    """Combines grad norm for the specific device

    From https://github.com/pytorch/pytorch/issues/88621"""
    local_norm = get_grad_norm_local(model)
    op = torch.distributed.ReduceOp.SUM
    return_norm = local_norm.clone().detach().requires_grad_(False) ** 2
    dist.all_reduce(return_norm, op=op)
    if sharding_strategy == ShardingStrategy.NO_SHARD:
        return_norm = return_norm / world_size
    return return_norm**0.5


def param_norm(parameters):
    with torch.no_grad():
        total_norm = 0
        for p in parameters:
            total_norm += p.pow(2).sum().item()
        return total_norm**0.5


class Trainer:
    grad_scaler: GradScaler

    def __init__(
        self,
        experiment_name: str,
        viz_folder: str,
        formatter: Callable,
        model: torch.nn.Module,
        datamodule: AbstractDataModule,
        revin: BaseRevNormalization,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        prediction_type: str,
        max_epoch: int,
        val_frequency: int,
        rollout_val_frequency: int,
        max_rollout_steps: int,
        short_validation_length: int,
        checkpointer: CheckPointLoader,
        num_time_intervals: int,
        skip_checkpointing: bool = False,
        validation_suite: list[Metric] = [VRMSE()],
        validation_trajectory_metrics: Optional[list[Metric]] = None,
        batch_aggregation_fns: list[Callable] = [torch.mean],
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device=torch.device("cuda"),
        device_mesh: Optional[torch.distributed.device_mesh.DeviceMesh] = None,
        sampling_rank_strategy: Literal["gpu", "node", "world"] = "node",
        reuse_batches: bool = False,
        distribution_type: str = "local",
        rank: int = 0,
        world_size: int = 1,
        enable_amp: bool = False,
        amp_type: str = "float16",  # bfloat not supported in FFT
        grad_acc_steps: int = 1,
        big_batch_multiplier: int = 1,
        big_batch_before: int = 0,
        clip_gradient: float = 0.0,
        loss_multiplier: float = 1.0,
        minimum_context: int = 1,
        validation_full_trajectory_ensemble_size: int = 1,
        validation_one_step_ensemble_size: int = 1,
        masked_loss_for_objects: bool = True,
        video_validation: bool = False,
        video_size_multiplier: float = 0.4,
        dump_prediction_to_disk: bool = False,
        num_detailed_logs: int = 3,
        image_validation: bool = False,
        skip_spectral_metrics: bool = False,
        gradient_log_level: int = 0,
        log_interval: int = 10,
        wandb_logging: bool = True,
        start_epoch: int = 1,
        lr_scheduler_per_step: bool = False,
        debug_mode: bool = False,
        start_val_loss: Optional[float] = None,
        epsilon: float = 1e-5,
        validation_epsilon: float = 1e-5,
    ):
        """
        Class in charge of the training loop. It performs train, validation and test.

        Parameters
        ----------
        experiment_name:
            The name of the training experiment to be run
        viz_folder:
            The folder where visualizations are saved
        formatter:
            Callable that initializes formatter object that maps between Well and model formats.
        model:
            PyTorch model used for training.
        datamodule:
            A datamodule that provides dataloaders for each split (train, valid, and test)
        optimizer:
            A Pytorch optimizer to perform the backprop (e.g. Adam)
        loss_fn:
            A loss function that evaluates the model predictions to be used for training
        prediction_type:
            The type of prediction to make. Options are "delta" or "full". "delta" predicts the change in the
            field from the previous timestep. "full" predicts the full field at the next timestep.
            This only affects training since validation losses are computed on full reconstructed fields
            either way.
        max_epoch:
            Number of epochs to train the model.
            One epoch correspond to a full loop over the datamodule's training dataloader which may be a subset
        val_frequency:
            The frequency in terms of number of epochs to perform one-step validation
        rollout_val_frequency:
            The frequency in terms of number of epochs to perform rollout validation
        max_rollout_steps:
            The maximum number of timesteps to rollout the model during long validation.
            Note: for historical reasons, redundant parameter is included in datamodule - both need to be set.
        num_time_intervals:
            The number of time intervals to bin the loss over for logging purposes.
        lr_scheduler:
            A Pytorch learning rate scheduler to update the learning rate during training
        device:
            A Pytorch device (e.g. "cuda" or "cpu")
        device_mesh:
            Device mesh used for distributed training
        reuse_batches:
            A boolean flag to reuse batches during training. If True, the same batch is used for
            two steps of training (separated by a new batch). If False, a new batch is used for each step.
            Mostly use if disk io is bottleneck.
        distribution_type:
            The type of distribution to use. Options are "local", "ddp", "fsdp", "hsdp"
        rank:
            The rank of the current GPU in the PyTorch world.
        world_size:
            The total number of GPUs in the PyTorch world
        enable_amp:
            A boolean flag to enable automatic mixed precision training
        amp_type:
            The type of automatic mixed precision to use. Options are "float16" or "bfloat16"
        grad_acc_steps:
            The number of gradient accumulation steps to perform between optimizer steps
        big_batch_multiplier:
            Use batches this many times larger via grad accumulation during the first big_batch_before epochs
            Was experimented with, but not actually used.
        big_batch_before:
            Use big batches for the first this many epochs
            Was experimented with, but not actually used.
        clip_gradient:
            The maximum gradient norm to clip to. If 0, no clipping is performed.
        loss_multiplier:
            A float to multiply the loss by before backpropagating. Useful for satisfying paranoia
            about underflow error. Generally won't matter when using Adam-family optimizers at FP32.
        minimum_context:
            The minimum number of timesteps needed to evaluate the loss for a given sample
        validation_full_trajectory_ensemble_size:
            The number of times to repeat the trajectory during validation. The final prediction is averaged over all sampled trajectories.
        validation_one_step_ensemble_size:
            The number of samples to draw per step. The final step is averaged over all sampled steps.
        video_validation:
            A boolean flag to enable saving rollouts to disk during validation
        dump_prediction_to_disk:
            A boolean flag to enable saving raw prediction and reference numpy arrays to disk during validation
        num_detailed_logs:
            The number of detailed logs (video, numpy dumps) to save during validation
        image_validation:
            A boolean flag to enable saving images to disk during validation
        skip_spectral_metrics:
            A boolean flag to skip spectral metrics during validation since they can be memory hungry.
        gradient_log_level:
            An integer representing the level of gradient logging. 0 is no logging, 1 full synced gradient only
        log_interval:
            An integer representing how often to log training information. This results in gpu-cpu sync.
        wandb_logging:
            A boolean flag to enable logging to Weights and Biases
        start_epoch:
            The epoch to start training from. Used for resuming training.
        lr_scheduler_per_step:
            A boolean flag to update the learning rate after each optimizer step instead of each epoch.
        debug_mode:
            A boolean flag to run a quick debug run. This limits all validation to small version.
        start_val_loss:
            The validation loss to start from. Used for resuming training.
        epsilon:
            A small float added to denominators in denominators during training for numerical stability. Reasonably varies between runs.
        validation_epsilon:
            A small float added to denominators in loss functions during validation for numerical stability. Should be kept consistent across runs.
        """
        self.experiment_name = experiment_name
        self.viz_folder = viz_folder
        self.wandb_logging = wandb_logging
        self.video_validation = video_validation
        self.image_validation = image_validation
        self.video_size_multiplier = video_size_multiplier
        self.dump_prediction_to_disk = dump_prediction_to_disk
        self.num_detailed_logs = num_detailed_logs
        self.gradient_log_level = gradient_log_level
        self.log_interval = log_interval
        self.device = device
        self.model = model
        self.datamodule = datamodule
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.prediction_type = prediction_type
        self.skip_checkpointing = skip_checkpointing
        self.validation_suite = validation_suite
        self.validation_suite.append(self.loss_fn)
        self.validation_trajectory_metrics = validation_trajectory_metrics
        self.batch_aggregation_fns = batch_aggregation_fns
        self.skip_spectral_metrics = skip_spectral_metrics
        # These starting parameters are just for resuming runs
        self.start_epoch = start_epoch
        self.start_val_loss = start_val_loss
        # Run logistics
        self.max_epoch = max_epoch
        self.val_frequency = val_frequency
        self.rollout_val_frequency = rollout_val_frequency
        self.max_rollout_steps = max_rollout_steps
        self.short_validation_length = short_validation_length
        self.num_time_intervals = num_time_intervals
        self.enable_amp = enable_amp
        self.grad_acc_steps = grad_acc_steps
        self.big_batch_multiplier = big_batch_multiplier
        self.big_batch_before = big_batch_before
        self.clip_gradient = clip_gradient
        self.loss_multiplier = loss_multiplier
        self.minimum_context = minimum_context
        self.validation_full_trajectory_ensemble_size = (
            validation_full_trajectory_ensemble_size
        )
        self.validation_one_step_ensemble_size = validation_one_step_ensemble_size
        self.lr_scheduler_per_step = lr_scheduler_per_step
        self.debug_mode = debug_mode
        self.masked_loss_for_objects = masked_loss_for_objects
        self.amp_type = torch.bfloat16 if amp_type == "bfloat16" else torch.float16
        self.checkpointer = checkpointer
        # If local, DDP or spatial (replicated weights), can use standard grad scaler
        if distribution_type.upper() in ["LOCAL", "DDP", "SPATIAL"]:
            self.grad_scaler = torch.GradScaler(
                device=self.device.type, enabled=enable_amp and amp_type != "bfloat16"
            )
        # Otherwise need sharded version.
        else:
            self.grad_scaler = ShardedGradScaler(
                device=self.device.type, enabled=enable_amp and amp_type != "bfloat16"
            )

        self.device_mesh = device_mesh
        self.is_distributed = device_mesh is not None
        self.distribution_type = distribution_type
        self.rank = rank
        self.world_size = world_size
        self.sampling_rank_strategy = sampling_rank_strategy
        self.reuse_batches = reuse_batches
        # Get derived rank info about which nodes must be synced from the device mesh
        self.spatial = get_spatial_context()
        # True once enable_domain_split is called: each GPU holds one slab
        self.split_domain = False
        if self.spatial is not None:
            # A spatial group shares each sample, like an FSDP group shares a dataset
            self.sync_group = self.spatial.group
            self.sync_group_size = self.spatial.size
        elif self.device_mesh is not None and "fsdp" in self.device_mesh.mesh_dim_names:
            self.sync_group = self.device_mesh.get_group(mesh_dim="fsdp")
            self.sync_group_size = self.sync_group.size()
        else:  # Local or DDP
            self.sync_group = None
            self.sync_group_size = 1
        # Local or DDP do not need all-gather to do forward pass
        self.sync_group_rank = self.rank // self.sync_group_size
        self.rank_in_sync_group = self.rank % self.sync_group_size
        self.num_sync_groups = self.world_size // self.sync_group_size
        # FSDP group members split validation batches among themselves; spatial
        # group members must all see the same batches.
        self.eval_replicas = 1 if self.spatial else self.sync_group_size
        self.eval_rank = 0 if self.spatial else self.rank_in_sync_group
        if self.spatial is not None:
            # Every GPU of a spatial group must draw the same training samples
            self.sampling_rank = self.sync_group_rank
        elif (
            self.sampling_rank_strategy == "gpu"
        ):  # This means sample different dataset per GPU
            self.sampling_rank = self.rank
        elif (
            self.sampling_rank_strategy == "node"
        ):  # This means sample different dataset per node
            self.sampling_rank = self.sync_group_rank
        else:  # This means sample single dataset per step
            self.sampling_rank = 0

        self.dset_metadata = self.datamodule.train_dataset.dset_to_metadata
        self.revin = revin(self.datamodule.train_dataset, self.device)
        self.model_epsilon = (
            epsilon  # Used for compatibility with configs from before split occurred
        )
        self.validation_epsilon = validation_epsilon

        self.formatter_dict = {}
        # Initial formatter for each dataset - right now these are all identical
        # but we might want to differentiate them in the future.
        for dset_name, metadata in self.dset_metadata.items():
            self.formatter_dict[metadata.dataset_name] = formatter()

    def enable_domain_split(self, ctx) -> None:
        """Train on slabs of a domain split across the spatial group ctx: RevIN
        statistics over the whole domain, and each GPU's loss is its slab's
        share of the full-domain loss."""
        self.split_domain = True
        self.split_ctx = ctx
        self.revin.spatial_ctx = ctx

    def save_model_if_necessary(
        self, epoch: int, validation_loss: float, last: bool = False
    ) -> Optional[Future]:
        """Save the model checkpoint.
        Force checkpointing if last.
        """
        checkpoint_future = self.checkpointer.save_if_necessary(
            self.model,
            self.optimizer,
            validation_loss,
            epoch,
            force=last,
            local=self.distribution_type.upper() in ["LOCAL", "DDP", "SPATIAL"],
        )
        return checkpoint_future

    def rollout_model(self, model, batch, formatter, train=True, fake_pass=False):
        """Rollout the model for as many steps as we have data for.

        predict_normalized: bool - If true, output normalized prediction. During one-step training,
            predict normalized values to reduce precision issues/extra FLOPs. During rollout,
            denormalize the output for loss calculation. If multiple steps used during training,
            throw error because not currently supported.
        """

        metadata = batch["metadata"]
        # space_grid is never read past this point; at full resolution it is
        # several GB, so it stays on the host.
        batch = {
            k: v.to(self.device, non_blocking=True)
            if k not in {"metadata", "boundary_conditions", "space_grid"}
            else v
            for k, v in batch.items()
        }
        # Extract mask and move to device for loss eval
        if (
            self.masked_loss_for_objects
            and "mask" in batch["metadata"].constant_field_names[0]
        ):
            mask_index = batch["metadata"].constant_field_names[0].index("mask")
            mask = batch["constant_fields"][..., mask_index : mask_index + 1]
            mask = mask.to(self.device, dtype=torch.bool, non_blocking=True)
        else:
            mask = None

        inputs, y_ref = formatter.process_input(
            batch,
            causal_in_time=model.causal_in_time,
            predict_delta=self.prediction_type == "delta",
            train=train,
        )

        # Inputs T B C H [W D], y_ref B T H [W D] C
        # If causal, during training don't include initial context in rollout length
        T_in = batch["input_fields"].shape[1]
        if model.causal_in_time:
            max_rollout_steps = self.max_rollout_steps + (T_in - 1)
        else:
            max_rollout_steps = self.max_rollout_steps
        rollout_steps = min(
            y_ref.shape[1], max_rollout_steps
        )  # Number of timesteps in target
        train_rollout_limit = T_in if (train and model.causal_in_time) else 1
        if rollout_steps > train_rollout_limit and train:
            raise ValueError("Multiple step prediction in train mode not yet supported")
        y_ref = y_ref[:, :rollout_steps]
        if train:
            # The target is built; nothing reads the device copy of the raw
            # output fields again (at full resolution it is as large as the input)
            batch["output_fields"] = None
        # Create a moving batch of one step at a time
        # A shallow copy is enough: the loop below only rebinds
        # moving_batch["input_fields"] and never modifies tensors in place.
        # A deepcopy would duplicate every field on the GPU.
        moving_batch = dict(batch)
        y_preds = []
        # Fake pass is just a convenience for distributed validation without communcation code
        if fake_pass:
            return y_ref, y_ref
        # Rollout the model - Causal in time gets more predictions from the first step
        for i in range(train_rollout_limit - 1, rollout_steps):
            # Don't fill causal_in_time here since that only affects y_ref
            inputs, _ = formatter.process_input(moving_batch)
            inputs = list(inputs)
            with torch.no_grad():
                normalization_stats = self.revin.compute_stats(
                    inputs[0], metadata, epsilon=self.model_epsilon
                )
            # NOTE - Currently assuming only [0] (fields) needs normalization
            normalized_inputs = inputs[:]  # Map type bugs out
            # In training the raw input is not read again: normalize in place
            # (same values, one input-sized tensor less)
            normalize = (
                self.revin.normalize_stdmean_ if train else self.revin.normalize_stdmean
            )
            normalized_inputs[0] = normalize(normalized_inputs[0], normalization_stats)
            if train:
                ensemble_size = 1  # No ensembling during training right now
            else:
                ensemble_size = self.validation_one_step_ensemble_size
            for jj in range(ensemble_size):
                y_pred_internal = model(
                    normalized_inputs[0],
                    normalized_inputs[1],
                    normalized_inputs[2].tolist(),
                    metadata=metadata,
                )
                if jj == 0:
                    y_pred = y_pred_internal.clone() / ensemble_size
                else:
                    y_pred = y_pred + y_pred_internal / ensemble_size
            # During validation, don't maintain full inner predictions
            if not train and model.causal_in_time:
                y_pred = y_pred[-1:]  # y_pred is T first, y_ref is not
            # Train used normalized values to avoid precision loss
            # Validation on the other hand, reconstructs predictions on original scale
            if train:
                pass  # Do nothing since we're computing loss on predicted value and normalizing "ref"
            elif self.prediction_type == "delta":
                # y_pred - (T_all or T=-1 depending on causal or not), B, C, H, [W, D]. Different from y_ref
                with torch.autocast(
                    self.device.type, enabled=False, dtype=self.amp_type
                ):
                    y_pred = inputs[0][
                        -y_pred.shape[0] :
                    ].float() + self.revin.denormalize_delta(
                        y_pred, normalization_stats
                    )  # Unnormalize delta and add to input
            elif self.prediction_type == "full":
                y_pred = self.revin.denormalize_stdmean(y_pred, normalization_stats)
            else:
                raise ValueError(
                    f"Invalid prediction type {self.prediction_type}. Valid types are delta/full"
                )
            y_pred = formatter.process_output(y_pred, metadata)[
                ..., : y_ref.shape[-1]
            ]  # Cut off constant channels

            # TODO - redo losses to accept losses since this will be more efficient there
            if mask is not None:
                mask_pred = expand_mask_to_match(mask, y_pred)
                y_pred.masked_fill_(mask_pred, 0)

            # NOTE (MM) - The person reading this might ask - why is the mask in-place above and a copy below?
            # The answer is that job failed using in_place below but not above. No idea why.
            if (
                batch["padded_field_mask"].shape[0] != y_pred.shape[-1]
            ):  # Quick hack for Neutron
                batch["padded_field_mask"] = batch["padded_field_mask"][
                    : y_pred.shape[-1]
                ]
            y_pred = y_pred.masked_fill(~batch["padded_field_mask"], 0.0)

            # If not last step, update moving batch for autoregressive prediction
            # TODO - for anyone updating this later, it's the primary reason why
            # multiple steps isn't currently supported since we want to recompute
            # normalization stats at each step, but also want to compute training loss
            # on normalized values
            if i != rollout_steps - 1:
                moving_batch["input_fields"] = torch.cat(
                    [moving_batch["input_fields"][:, 1:], y_pred[:, -1:]], dim=1
                )
            # For causal models, we get use full predictions for the first batch and
            # incremental predictions for subsequent batches - concat 1:T to y_ref for loss eval
            if model.causal_in_time and i == train_rollout_limit - 1:
                y_preds.append(y_pred)
            else:
                y_preds.append(y_pred[:, -1:])
        y_pred_out = torch.cat(y_preds, dim=1)
        # Post-processing y_ref depending on train - if train, normalize y_ref before loss calc
        # If not train, we already denormalized the prediction
        if train:
            mean = (
                normalization_stats.sample_mean
                if self.prediction_type == "full"
                else normalization_stats.delta_mean
            )
            std = (
                normalization_stats.sample_std
                if self.prediction_type == "full"
                else normalization_stats.delta_std
            )
            y_ref = normalize_target(y_ref, mean, std, formatter, metadata, self.device)
        if mask is not None:
            mask_ref = expand_mask_to_match(mask, y_ref)
            y_ref.masked_fill_(mask_ref, 0)

        del moving_batch, batch, mask  # Free up batch memory when done
        return y_pred_out, y_ref

    def temporal_split_losses(
        self, loss_values, temporal_loss_intervals, loss_name, dset_name, fname="full"
    ):
        """Take time series of loss values in split them into aggregate metrics computed on
        different time intervals."""
        # loss_values is B, T here since we either selected from or averaged over fields
        new_losses = {}
        # Average over time interval
        new_losses[f"{dset_name}/{fname}_{loss_name}_T=all"] = loss_values.mean(dim=1)
        # Don't compute sublosses if we only have one interval
        if len(temporal_loss_intervals) <= 2:
            return new_losses
        # Break it down by time interval
        for k in range(len(temporal_loss_intervals) - 1):
            start_ind = temporal_loss_intervals[k]
            end_ind = temporal_loss_intervals[k + 1]
            time_str = f"{start_ind}:{end_ind}"
            loss_subset = loss_values[:, start_ind:end_ind].mean(
                1
            )  # Mean over time in interval
            new_losses[f"{dset_name}/{fname}_{loss_name}_T={time_str}"] = loss_subset
        return new_losses

    def split_up_losses(
        self, loss_values, loss_name, dset_name, field_names: list[str]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Divide raw losses into per-field and mean values, as well as temporal splits.
        Returns new loss dict and time logs for plotting."""
        # loss_values is B, T, C here
        new_losses: dict[str, Any] = {}
        time_logs: dict[str, Any] = {}
        time_steps = loss_values.shape[1]  # we B, T, C
        num_time_intervals = min(time_steps, self.num_time_intervals)
        temporal_loss_intervals = np.linspace(0, np.log(time_steps), num_time_intervals)
        temporal_loss_intervals = [0] + [
            int(np.exp(x)) for x in temporal_loss_intervals
        ]
        # Split up losses by field
        for i, fname in enumerate(field_names):
            time_logs[f"{dset_name}/{fname}_{loss_name}_rollout"] = loss_values[
                :, :, i
            ].cpu()
            new_losses |= self.temporal_split_losses(
                loss_values[:, :, i],
                temporal_loss_intervals,
                loss_name,
                dset_name,
                fname,
            )
        # Compute average over all fields
        new_losses |= self.temporal_split_losses(
            loss_values.mean(2),
            temporal_loss_intervals,
            loss_name,
            dset_name,
            "full",
        )
        time_logs[f"{dset_name}/full_{loss_name}_rollout"] = loss_values.mean(2).cpu()
        # New losses - B, time_logs - B T
        return new_losses, time_logs

    @torch.no_grad()
    def validation_loop(
        self,
        dataloaders: list[WellDataset],
        valid_or_test: str = "valid",
        full=False,
        epoch: int = 0,
    ) -> tuple[float, dict[str, Any]]:
        """Run validation by looping over the dataloader.

        Validate the same dataset over FSDP groups since they're locked
        by syncs but distribute over replication (DDP) groups.

        WARNING: running DDP or HSDP with a number of datasets that either dont
        evenly divide the number of shard groups/GPUs or that have
        significantly varying sizes will probably result in a timeout at one of these barriers.
        """
        full = full if not self.debug_mode else False
        self.model.eval()
        validation_loss = 0.0
        metadatas = []
        # Timeouts get really annoying - barrier at start makes it slightly better
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        # Each dataset being validated gets separate loader
        rank_loss_dict = {}
        rank_time_logs = {}
        for i, dataloader in enumerate(dataloaders):
            # Grab metadata for the current dataset
            assert len(dataloader.dataset.sub_dsets) == 1, (
                "Only one dataset per validation dataloader"
            )
            dataset = dataloader.dataset.sub_dsets[
                0
            ]  # There is only one dset by design
            dset_time_logs = {}
            current_metadata = dataset.metadata
            metadatas.append(current_metadata)
            dset_name = current_metadata.dataset_name
            field_names = flatten_field_names(current_metadata, include_constants=False)
            # NOTE MM - We want each FSDP group to process a different dataset to accelerate
            # partial validation between epochs so we assign datasets to groups in round-robin fashion.
            # If we have more FSDP groups than datasets, some groups will do nothing which can lead to timeouts
            # during full validation.
            # Validation shared over FSDP group, then passed to local rank 0 for aggregation and logging.
            # For giant dataset or metric counts, this could get memory hungry on rank 0.
            rank_assignment = (
                i % self.num_sync_groups
            )  # Use same data across one sync group
            # Only print if we're doing something on this node
            if rank_assignment == self.sync_group_rank:
                logger.info(
                    f"Validating dataset {dataset.metadata.dataset_name} with full_trajectory_mode={dataset.full_trajectory_mode} on rank {self.rank}"
                )
            else:
                continue
            count = 0
            denom = (
                len(dataloader)
                if full
                else min(len(dataloader), self.short_validation_length)
            )
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.enable_amp,
                dtype=self.amp_type,
            ):
                for j, batch in enumerate(dataloader):
                    # Validation datasets don't automatically add metadata
                    start_time = time.time()
                    # Rollout for length of target - fake pass if not evaluating on this node
                    # so that we get the right field names for reduction
                    for kk in range(self.validation_full_trajectory_ensemble_size):
                        y_pred_internal, y_ref_internal = self.rollout_model(
                            self.model,
                            batch,
                            self.formatter_dict[dset_name],
                            train=False,
                            fake_pass=(
                                rank_assignment != self.sync_group_rank
                            ),  # Leftover from earlier design - should never be true due to continue above
                        )
                        if kk == 0:
                            y_pred = (
                                y_pred_internal
                                / self.validation_full_trajectory_ensemble_size
                            )
                            y_ref = y_ref_internal  # This doesn't change
                        else:
                            y_pred += (
                                y_pred_internal
                                / self.validation_full_trajectory_ensemble_size
                            )
                    assert y_ref.shape == y_pred.shape, (
                        f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
                    )
                    # Go through losses
                    model_time = time.time() - start_time
                    if (
                        batch["padded_field_mask"].shape[0] != y_pred.shape[-1]
                    ):  # Quick hack for Neutron and the underscores in the dimension names...
                        batch["padded_field_mask"] = batch["padded_field_mask"][
                            : y_pred.shape[-1]
                        ]
                    # Don't evaluate loss of "padded" dimensions so we're not biased towards predicting 0
                    y_pred, y_ref = (
                        y_pred[..., batch["padded_field_mask"]],
                        y_ref[..., batch["padded_field_mask"]],
                    )

                    # Collecting names to make detailed output logs
                    used_field_names = [
                        f
                        for i, f in enumerate(field_names)
                        if batch["padded_field_mask"][i]
                    ]

                    # Iterate through all validation metrics and log them.
                    for loss_fn in self.validation_suite:
                        # Mean over batch and time per field
                        if (
                            self.skip_spectral_metrics
                            and "spectr" in loss_fn.__class__.__name__
                        ):
                            continue
                        # Loss fn expect B T [H W D] C where [H W D] are described by metadata
                        loss = loss_fn(
                            y_pred, y_ref, current_metadata, eps=self.validation_epsilon
                        )
                        # Some losses return multiple values for efficiency, so if not dict,
                        # wrap in dict here
                        if not isinstance(loss, dict):
                            loss = {loss_fn.__class__.__name__: loss}
                        # Split the losses and update the logging dictionary
                        for k, sub_loss in loss.items():
                            # Each loss is B, T, C
                            new_losses, new_time_logs = self.split_up_losses(
                                sub_loss, k, dset_name, used_field_names
                            )
                            # TODO Break spectral error into second category using new API
                            for loss_name, batch_losses in new_losses.items():
                                if loss_name in rank_loss_dict:
                                    rank_loss_dict[loss_name] = torch.cat(
                                        [rank_loss_dict[loss_name], batch_losses], dim=0
                                    )
                                else:
                                    rank_loss_dict[loss_name] = batch_losses
                                # Let's just store the VRMSE for printout since that's what we're actually looking at on aggregate.
                                if "full_VRMSE_T=all" in loss_name:
                                    vrmse = batch_losses.mean().item()
                            if dataset.full_trajectory_mode:
                                for loss_name, batch_time_logs in new_time_logs.items():
                                    if loss_name in dset_time_logs:
                                        # Time logs live on CPU for mem reasons
                                        dset_time_logs[loss_name] = torch.cat(
                                            [
                                                dset_time_logs[loss_name],
                                                batch_time_logs,
                                            ],
                                            dim=0,
                                        )
                                    else:
                                        dset_time_logs[loss_name] = batch_time_logs
                    # NOW do trajectory losses if we have them - can probably combine with above later
                    if dataset.full_trajectory_mode:
                        for traj_loss_fn in self.validation_trajectory_metrics or []:
                            traj_loss = traj_loss_fn(
                                y_pred, y_ref, current_metadata, batch["metadata"]
                            )
                            if not isinstance(traj_loss, dict):
                                traj_loss = {traj_loss_fn.__class__.__name__: traj_loss}
                            # Note - still use split up losses for per-field even if temporal not relevant
                            for k, sub_traj_loss in traj_loss.items():
                                new_traj_losses, _ = self.split_up_losses(
                                    sub_traj_loss, k, dset_name, used_field_names
                                )
                                for loss_name, batch_losses in new_traj_losses.items():
                                    if loss_name in rank_loss_dict:
                                        rank_loss_dict[loss_name] = torch.cat(
                                            [rank_loss_dict[loss_name], batch_losses],
                                            dim=0,
                                        )
                                    else:
                                        rank_loss_dict[loss_name] = batch_losses
                    total_time = time.time() - start_time
                    max_mem_GB = torch.cuda.max_memory_allocated() / 1024**3
                    # Only print out if local device actually doing something
                    if rank_assignment == self.sync_group_rank:
                        logger.info(
                            f"{valid_or_test}: {dset_name}, Batch {j + 1}/{denom}, Rank {self.rank:>3}: Field-time-averaged VRMSE {vrmse:7.4f}, mem {max_mem_GB:5.2f} GB, total_time {total_time:5.3f}s, model {model_time:5.4f}s"
                        )
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    count += 1
                    # Do some detailed outputs on rank 0 if specified
                    if (
                        self.rank_in_sync_group == 0
                        and rank_assignment == self.sync_group_rank
                    ):
                        if dataset.full_trajectory_mode:
                            if self.video_validation and count < self.num_detailed_logs:
                                try:
                                    make_video(
                                        y_pred[0],  # First sample only in batch
                                        y_ref[0],  # First sample only in batch
                                        current_metadata,
                                        self.viz_folder,
                                        f"{epoch}_rank{self.rank}_{valid_or_test}_batch{j}",  # For the file name
                                        field_name_overrides=used_field_names,  # Fields actually used
                                        size_multiplier=self.video_size_multiplier,  # Shrinking for bulk runs, but visuals tuned around 1
                                    )
                                except CalledProcessError as e:
                                    logger.warning(
                                        f"Error in making video due to FFMPEG: {e}. Skipping video."
                                    )
                            # Write out prediction and reference as npy for later analysis
                            if (
                                self.dump_prediction_to_disk
                                and count < self.num_detailed_logs
                            ):
                                dump_path = os.path.join(
                                    self.viz_folder, dset_name, "full_trajectory_dumps"
                                )
                                if not os.path.exists(dump_path):
                                    os.makedirs(dump_path, exist_ok=True)
                                np.save(
                                    os.path.join(
                                        dump_path,
                                        f"yref_{dset_name}_{valid_or_test}_epoch{epoch}_rank{self.rank}_{j}.npy",
                                    ),
                                    y_ref.cpu().numpy(),
                                )
                                np.save(
                                    os.path.join(
                                        dump_path,
                                        f"ypred_{dset_name}_{valid_or_test}_epoch{epoch}_rank{self.rank}_{j}.npy",
                                    ),
                                    y_pred.cpu().numpy(),
                                )
                                logger.info(f"Wrote out npy dumps to {dump_path}")
                    # For most per-"epoch" validations, we only do a configurably short subset
                    if not full and count >= self.short_validation_length:
                        break
                # Run some last outputs on rank 0 do get a general sense of what's going on
                if (
                    self.rank_in_sync_group == 0
                    and rank_assignment == self.sync_group_rank
                ):
                    if self.image_validation:
                        for plot_fn in validation_plots:
                            if (
                                self.skip_spectral_metrics
                                and "spectr" in plot_fn.__name__
                            ):
                                continue
                            plot_fn(
                                y_pred,
                                y_ref,
                                current_metadata,
                                self.viz_folder,  # Temporary until we port over the resume logic
                                f"{epoch}_rank{self.rank}_{valid_or_test}_batch{j}",  # For the file name
                            )
                    if dataset.full_trajectory_mode:
                        if self.video_validation:
                            try:
                                make_video(
                                    y_pred[0],  # First sample only in batch
                                    y_ref[0],  # First sample only in batch
                                    current_metadata,
                                    self.viz_folder,
                                    f"{epoch}_rank{self.rank}_{valid_or_test}_batch{j}",  # For the file name
                                    field_name_overrides=used_field_names,  # Fields actually used
                                    size_multiplier=self.video_size_multiplier,  # Shrinking for bulk runs, but visuals tuned around 1
                                )
                            except CalledProcessError as e:
                                logger.warning(
                                    f"Error in making video due to FFMPEG: {e}. Skipping video."
                                )
            if dataset.full_trajectory_mode:
                rank_time_logs[current_metadata.dataset_name] = dset_time_logs
        # If we're distributed, now send all per rank results to rank 0 for aggregation
        if self.is_distributed:
            # Wait for all ranks to finish
            logger.info(f"Rank {self.rank} waiting for barrier")
            dist.barrier()
            logger.info(f"Rank {self.rank} passed barrier")

            # If we're distributed, gather all batchwise losses onto rank 0 for summarization and logging
            if self.spatial is not None and self.rank_in_sync_group != 0:
                # Spatial group members evaluated the same batches as their
                # leader; only the leader's copy is counted.
                rank_loss_dict, rank_time_logs = {}, {}
            object_loss_dict = {k: v.cpu() for k, v in rank_loss_dict.items()}
            loss_dicts = [None for _ in range(self.world_size)]
            time_logs_list = [None for _ in range(self.world_size)]
            dist.all_gather_object(loss_dicts, object_loss_dict)
            dist.all_gather_object(time_logs_list, rank_time_logs)
            loss_dict = {}
            # Iterate through loss dicts and recover all keys
            for ld in loss_dicts:
                for k, v in ld.items():
                    if k not in loss_dict:
                        loss_dict[k] = v
                    else:
                        loss_dict[k] = torch.cat([loss_dict[k], v], dim=0)
            time_logs = {}
            # Time logs have two levels {dataset: {lossname: B x T tensor}}
            for tl in time_logs_list:
                for dataset, dset_time_logs in tl.items():
                    if dataset not in time_logs:
                        time_logs[dataset] = {}
                    for k, v in dset_time_logs.items():
                        if k not in time_logs[dataset]:
                            time_logs[dataset][k] = v
                        else:
                            time_logs[dataset][k] = torch.cat(
                                [time_logs[dataset][k], v], dim=0
                            )
        else:  # If we're not distributed, just use the local rank dict
            loss_dict = rank_loss_dict
            time_logs = rank_time_logs
        # Single score validation loss is average of all losses on the training metric
        validation_loss = sum(
            [
                loss_dict[
                    f"{metadata.dataset_name}/full_{self.loss_fn.__class__.__name__}_T=all"
                ]
                .mean()
                .item()  # Losses should all be B sized
                for metadata in metadatas
            ]
        ) / len(metadatas)
        aggregated_losses = {}
        aggregated_time_logs = {}
        for f in self.batch_aggregation_fns:
            for k, v in loss_dict.items():
                aggregated_losses[f"{valid_or_test}_{k}_{f.__name__}"] = f(
                    v
                )  # All losses by this point should be B sized
            # Two levels of time logs
            for dataset, dset_time_logs in time_logs.items():
                if dataset not in aggregated_time_logs:
                    aggregated_time_logs[dataset] = {}
                for k, v in dset_time_logs.items():
                    agged = f(v, dim=0)
                    if not isinstance(
                        agged, torch.Tensor
                    ):  # Some torch args return wrappers - hack for the common pattern
                        agged = (
                            agged.values
                        )  # For torch tensors wrapped in something else
                    aggregated_time_logs[dataset][
                        f"{valid_or_test}_{k}_{f.__name__}"
                    ] = agged
        # Simple function just saves all time logs in place
        if len(aggregated_time_logs) > 0 and self.rank == 0:
            log_all_time_metrics(
                aggregated_time_logs,
                self.viz_folder,
                f"{epoch}_rank{self.rank}_{valid_or_test}",  # For the file name
            )
            # Write the aggretated time log pickles for later analysis
            dump_path = os.path.join(
                self.viz_folder,
                "loss_dicts",
                f"{valid_or_test}_time_logs_epoch{epoch}_rank{self.rank}.pkl",
            )
            if not os.path.exists(dump_path):
                os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            with open(dump_path, "wb") as f:
                pickle.dump(aggregated_time_logs, f)
            logger.info(f"Wrote out time logs to {dump_path}")
        # Just write the pickles out.
        if len(aggregated_losses) > 0 and self.rank == 0:
            # Write the aggretated loss pickles for later analysis
            dump_path = os.path.join(
                self.viz_folder,
                "loss_dicts",
                f"{valid_or_test}_loss_dict_epoch{epoch}_rank{self.rank}.pkl",
            )
            if not os.path.exists(dump_path):
                os.makedirs(os.path.dirname(dump_path), exist_ok=True)
            with open(dump_path, "wb") as f:
                pickle.dump(aggregated_losses, f)
            logger.info(f"Wrote out loss dict to {dump_path}")
        # Misc metrics
        aggregated_losses["param_norm"] = param_norm(self.model.parameters())
        return validation_loss, aggregated_losses

    def train_one_epoch(
        self, epoch: int, dataloader: DataLoader
    ) -> tuple[float, dict[str, Any]]:
        """Train the model for one epoch by looping over the dataloader."""
        self.model.train()
        epoch_loss = 0.0
        avg_grad_norm = torch.tensor(0.0).to(self.device)
        last_grad_norm = torch.tensor(0.0).to(self.device)
        train_logs: dict[str, Any] = {}
        batch_start = time.time()
        interval_start = time.time()
        # When using grad acculuation, it makes sense to zero gradient outside first, then after optimizer step
        self.optimizer.zero_grad()  # Set to none now default
        overall_batch_queue = []
        current_batch_queue = []
        data_iter = iter(dataloader)
        i = 0
        if epoch < self.big_batch_before:
            grad_acc_steps = self.grad_acc_steps * self.big_batch_multiplier
        else:
            grad_acc_steps = self.grad_acc_steps

        while i < len(dataloader):
            # If reuse batches is on, we want to cache grad_acc_steps worth of
            # batches and reuse them. We want the order to be new -> cached -> new
            # Current last batch will just be new.
            if self.reuse_batches and (
                len(overall_batch_queue) >= 2 * grad_acc_steps
                or len(current_batch_queue) > 0
            ):
                # Reuse the batch
                if len(current_batch_queue) == 0:
                    # If grad acc > 1, shuffle so we have unique batches at least
                    if grad_acc_steps > 1:
                        shuffle(overall_batch_queue)
                    current_batch_queue = overall_batch_queue[:grad_acc_steps]
                    overall_batch_queue = overall_batch_queue[grad_acc_steps:]
                batch = current_batch_queue.pop(0)
            else:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                if self.reuse_batches:
                    overall_batch_queue.append(batch.copy())
            batch["padded_field_mask"] = batch["padded_field_mask"].to(
                self.device, non_blocking=True
            )
            # Update grad if we're not using distribution
            update_grad = (i + 1) % grad_acc_steps == 0
            with (
                nullcontext()
                if (update_grad or self.distribution_type == "local")
                else self.model.no_sync()
            ):
                with torch.autocast(
                    device_type=self.device.type,
                    enabled=self.enable_amp,
                    dtype=self.amp_type,
                ):
                    data_time = time.time() - batch_start
                    current_metadata = batch["metadata"]
                    dset_name = current_metadata.dataset_name
                    y_pred, y_ref = self.rollout_model(
                        self.model, batch, self.formatter_dict[dset_name]
                    )
                    # If T > self.minimum_context, then optimize only the predictions with the minimum context
                    # By default this is just removing the zero-context prediction in causal mode.
                    if y_pred.shape[1] > self.minimum_context:
                        y_ref = y_ref[:, self.minimum_context :]
                        y_pred = y_pred[:, self.minimum_context :]
                    forward_time = time.time() - batch_start - data_time
                    assert y_ref.shape == y_pred.shape, (
                        f"Mismatching shapes between reference {y_ref.shape} and prediction {y_pred.shape}"
                    )
                    if self.split_domain:
                        # This slab's share of the full-domain loss, times the
                        # group size: the gradient average over all GPUs is then
                        # the sum over each group, averaged over the groups
                        sample_loss = spatial_mean_loss(
                            self.loss_fn, y_pred, y_ref, current_metadata,
                            self.split_ctx, eps=self.model_epsilon,
                        ).mean() * self.split_ctx.size
                    else:
                        sample_loss = self.loss_fn(
                            y_pred, y_ref, current_metadata, eps=self.model_epsilon
                        ).mean()
                    loss = self.loss_multiplier * sample_loss / grad_acc_steps
                    del y_pred, y_ref  # Let gc free up a little before the BW pass
                # If not AMP, then grad scaler is no op
                self.grad_scaler.scale(loss).backward()
                backward_time = time.time() - batch_start - forward_time - data_time
            # On update_grad steps, we actually perform the steps
            if update_grad:
                if self.clip_gradient > 0 or self.gradient_log_level > 0:
                    self.grad_scaler.unscale_(self.optimizer)
                if self.gradient_log_level == 1:
                    if hasattr(self.model, "sharding_strategy"):
                        last_grad_norm = get_grad_norm_fsdp(
                            self.model,
                            self.rank,
                            self.world_size,
                            self.model.sharding_strategy,
                        )
                    else:
                        last_grad_norm = get_grad_norm_local(self.model)
                        avg_grad_norm += last_grad_norm.detach() / (
                            len(dataloader) / grad_acc_steps
                        )
                if self.clip_gradient > 0:
                    if hasattr(self.model, "clip_grad_norm_"):
                        self.model.clip_grad_norm_(
                            self.clip_gradient,
                            norm_type=2.0,
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.clip_gradient, norm_type=2.0
                        )
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()  # Set to none is now default\
                if self.lr_scheduler_per_step and self.lr_scheduler:
                    self.lr_scheduler.step()
            total_time = time.time() - batch_start
            optimizer_time = total_time - forward_time - backward_time - data_time
            # Syncing for all reduce anyway so may as well compute synchornous metrics
            logged_loss = loss.detach()
            if self.split_domain:  # full-domain loss: sum of the slab shares
                logged_loss = logged_loss / self.split_ctx.size
                dist.all_reduce(logged_loss, group=self.split_ctx.group)
            epoch_loss += (grad_acc_steps * logged_loss) / len(
                dataloader
            )  # Unscale loss for accurate measure.

            max_mem_GB = torch.cuda.max_memory_allocated() / 1024**3
            if i % self.log_interval == 0:
                timing = (time.time() - interval_start) / self.log_interval
                interval_start = time.time()
                logger.info(
                    f"Epoch {epoch:>4}, Batch {i + 1}/{len(dataloader)}, Rank {self.rank:>3}, SyncStep: {update_grad}:\n\t Data: {current_metadata.dataset_name:<32}, loss {(grad_acc_steps * logged_loss.item()) ** 0.5:7.4f}, mem {max_mem_GB:5.2f} GB, total_time {timing:5.3f}s, data {data_time:5.4f}s, fwd {forward_time:5.3f}s, bw {backward_time:5.3f}s, opt {optimizer_time:5.3f}s"
                )
            # Log times and memory stats to wandb - I don't trust wandb numbers
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            batch_start = time.time()
            # Log elapsed times in train_log - NOTE: only accurate if cuda syncing, but can be interpretted either way
            train_logs["avg_data_loading_time"] = train_logs.get(
                "avg_data_loading_time", 0
            ) + data_time / len(dataloader)
            train_logs["avg_forward_time"] = train_logs.get(
                "avg_forward_time", 0
            ) + forward_time / len(dataloader)
            train_logs["avg_backward_time"] = train_logs.get(
                "avg_backward_time", 0
            ) + backward_time / len(dataloader)
            train_logs["avg_optimizer_time"] = train_logs.get(
                "avg_optimizer_time", 0
            ) + optimizer_time / len(dataloader)
            train_logs["avg_time_per_step"] = train_logs.get(
                "avg_time_per_step", 0
            ) + total_time / len(dataloader)
            train_logs["peak_memory"] = max(
                train_logs.get("peak_memory", 0), max_mem_GB
            )
            i += 1
        train_logs["train_loss"] = epoch_loss
        if self.gradient_log_level >= 1:
            train_logs["avg_grad_norm"] = avg_grad_norm.item()
        if self.lr_scheduler:
            if not self.lr_scheduler_per_step:
                self.lr_scheduler.step()
            train_logs["lr"] = self.lr_scheduler.get_last_lr()[-1]
        return epoch_loss, train_logs

    def validate_if_necessary(
        self,
        epoch: int,
        one_step_dataloaders: list[DataLoader],
        rollout_dataloaders: list[DataLoader],
        valid_or_test: Literal["valid", "test"] = "valid",
    ):
        """Check what type of validate/rollouts we need to do for a given epoch.

        Parameters
        ----------
        epoch: int
            The current epoch. Used for logging and saving checkpoints.
        one_step_dataloaders: list[DataLoader]
            List of dataloaders for one step validation
        rollout_dataloaders: list[DataLoader]
            List of dataloaders for rollout validation
        valid_or_test: str
            String to indicate if we are validating or testing. Options are "valid" or "test"
        """
        is_test = valid_or_test == "test"  # Check if test
        val_loss, rollout_val_loss = None, None
        if self.split_domain:
            logger.warning(
                f"Epoch {epoch}/{self.max_epoch}: {valid_or_test} validation skipped: "
                "metrics on a split domain are not implemented yet (tracker task t303b)"
            )
            return val_loss, rollout_val_loss
        # First do one step checks = frequency, last epoch, or test. Only do full validation on last epoch or test
        if epoch % self.val_frequency == 0 or epoch >= self.max_epoch or is_test:
            logger.info(
                f"Epoch {epoch}/{self.max_epoch}: starting {valid_or_test} validation"
            )
            val_loss, loss_dict = self.validation_loop(
                one_step_dataloaders,
                valid_or_test=valid_or_test,
                full=(epoch >= self.max_epoch or is_test),
                epoch=epoch,
            )
            logger.info(
                f"Epoch {epoch}/{self.max_epoch}: {valid_or_test} loss {val_loss}"
            )
            loss_dict |= {f"{valid_or_test}": val_loss, "epoch": epoch}

            if self.wandb_logging and self.rank == 0:
                wandb.log(loss_dict)

        # Rollout if frequency, last epoch, or if this is the test set
        if (
            epoch % self.rollout_val_frequency == 0
            or epoch >= self.max_epoch
            or is_test
        ):
            logger.info(
                f"Epoch {epoch}/{self.max_epoch}: starting rollout {valid_or_test} validation"
            )
            rollout_val_loss, rollout_val_loss_dict = self.validation_loop(
                rollout_dataloaders,
                valid_or_test=f"rollout_{valid_or_test}",
                full=epoch >= self.max_epoch,
                epoch=epoch,
            )
            logger.info(
                f"Epoch {epoch}/{self.max_epoch}: rollout {valid_or_test} loss {rollout_val_loss}"
            )
            rollout_val_loss_dict |= {
                f"rollout_{valid_or_test}": rollout_val_loss,
                "epoch": epoch,
            }
            if self.wandb_logging and self.rank == 0:
                wandb.log(rollout_val_loss_dict)
        return val_loss, rollout_val_loss

    def train(self):
        """Run training, validation and test. The training is run for multiple epochs."""
        checkpoint_future = None
        val_loss = self.start_val_loss
        train_dataloader = self.datamodule.train_dataloader(self.sampling_rank)
        epoch = self.start_epoch
        for epoch in range(
            self.start_epoch, self.max_epoch + 1
        ):  # I like 1 indexing for epochs
            # NOTE - only update train sampler because we want to sample same valid data every time
            if self.is_distributed:
                train_dataloader.sampler.set_epoch(epoch)
            # Empty mem caches before train loop
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"Epoch {epoch}/{self.max_epoch}: starting training")
            train_loss, train_logs = self.train_one_epoch(epoch, train_dataloader)
            logger.info(
                f"Epoch {epoch}/{self.max_epoch}: training loss {train_loss:.4f}"
            )
            train_logs |= {"train": train_loss, "epoch": epoch}
            if self.wandb_logging and self.rank == 0:
                wandb.log(train_logs)
            # Empty mem caches before val
            torch.cuda.empty_cache()
            gc.collect()

            # Recreate loader every time so we're using same data in val
            val_dataloders = self.datamodule.val_dataloaders(
                replicas=self.eval_replicas,
                rank=self.eval_rank,
                full=(epoch >= self.max_epoch and not self.debug_mode),
            )
            rollout_val_dataloaders = self.datamodule.rollout_val_dataloaders(
                replicas=self.eval_replicas,
                rank=self.eval_rank,
                full=(epoch >= self.max_epoch and not self.debug_mode),
            )
            maybe_val_loss, rollout_loss = self.validate_if_necessary(
                epoch, val_dataloders, rollout_val_dataloaders
            )
            val_loss = maybe_val_loss if maybe_val_loss is not None else val_loss
            if checkpoint_future is not None:
                logger.debug(
                    f"Wait for previous checkpointing {checkpoint_future} to complete."
                )
                checkpoint_future.result()  # Make sure previous checkpoint has finished before starting next.
            # Save "last" every epoch plus various intervals/best results
            if not self.skip_checkpointing:
                checkpoint_future = self.save_model_if_necessary(
                    epoch, val_loss, last=(epoch == self.max_epoch)
                )
        # Do test validation
        test_dataloaders = self.datamodule.test_dataloaders(
            self.eval_replicas, rank=self.eval_rank, full=not self.debug_mode
        )
        rollout_test_dataloaders = self.datamodule.rollout_test_dataloaders(
            replicas=self.eval_replicas,
            rank=self.eval_rank,
            full=not self.debug_mode,
        )
        self.validate_if_necessary(
            epoch, test_dataloaders, rollout_test_dataloaders, valid_or_test="test"
        )

    def validate(self):
        """Run validation and test. This is a stand alone path"""
        val_dataloders = self.datamodule.val_dataloaders(
            replicas=self.eval_replicas,
            rank=self.eval_rank,
            full=not self.debug_mode,
        )
        rollout_val_dataloaders = self.datamodule.rollout_val_dataloaders(
            replicas=self.eval_replicas,
            rank=self.eval_rank,
            full=not self.debug_mode,
        )
        test_dataloaders = self.datamodule.test_dataloaders(
            replicas=self.eval_replicas,
            rank=self.eval_rank,
            full=not self.debug_mode,
        )
        rollout_test_dataloaders = self.datamodule.rollout_test_dataloaders(
            replicas=self.eval_replicas,
            rank=self.eval_rank,
            full=not self.debug_mode,
        )
        # Run validation and test
        self.validate_if_necessary(
            self.max_epoch + 1, val_dataloders, rollout_val_dataloaders
        )
        self.validate_if_necessary(
            self.max_epoch + 1,
            test_dataloaders,
            rollout_test_dataloaders,
            valid_or_test="test",
        )
