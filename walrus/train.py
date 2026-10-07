import logging
import os
import pathlib
from typing import Dict, Optional, cast

import hydra
import numpy as np
import torch
import wandb
from hydra.utils import get_method, instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torchinfo import summary

from walrus.data import MixedWellDataModule
from walrus.data.slabs import assign_slabs
from walrus.data.well_to_multi_transformer import (
    ChannelsFirstWithTimeFormatter,
)
from walrus.optim.optim_utils import (
    build_param_groups,
)
from walrus.trainer.checkpoints import CheckPointLoader
from walrus.trainer.training import Trainer
from walrus.utils.distribution_utils import (
    configure_distribution,
    distribute_model,
    setup_env_from_slurm,
)
from walrus.utils.experiment_utils import (
    align_checkpoint_with_field_to_index_map,
    configure_experiment,
)
from walrus.utils.spatial import enable_domain_split, get_spatial_context

logger = logging.getLogger("walrus")
# logger.setLevel(level=logging.DEBUG)

# Retrieve configuration for hydra
CONFIG_DIR = pathlib.Path(__file__).parent / "configs"
CONFIG_NAME = "config"
CONFIG_PATH = CONFIG_DIR / f"{CONFIG_NAME}.yaml"
assert CONFIG_PATH.is_file(), f"Configuration {CONFIG_PATH} is not an existing file."
logger.info(f"Run training script for {CONFIG_PATH}")

# import warnings
# warnings.filterwarnings("ignore")


def load_from_coalesced_checkpoint(
    model: torch.nn.Module,
    coalesced_checkpoint_path: str,
    field_to_index_map: Dict,
    old_field_index_map: Optional[Dict] = None,
    align_fields: bool = True,
):
    """Load model weights from a coalesced checkpoint, aligning field indices if necessary."""
    logger.info(f"Loading coalesced checkpoint {coalesced_checkpoint_path}")
    checkpoint = torch.load(coalesced_checkpoint_path, map_location="cpu")
    # Load the model weights
    model_checkpoint = checkpoint["app"]["model"]
    if align_fields and field_to_index_map != old_field_index_map:
        # NOTE (MM) - "model" n_fields should be strictly larger than "old" n_fields to loading
        # the old field index map values into the new override, but indices may not line
        # up exactly, so it's safer to align regardless
        model_checkpoint = align_checkpoint_with_field_to_index_map(
            checkpoint_state_dict=model_checkpoint,
            model_state_dict=model.state_dict(),
            checkpoint_field_to_index_map=old_field_index_map,
            model_field_to_index_map=field_to_index_map,
        )
    model.load_state_dict(model_checkpoint, strict=True)
    return model


def train(
    cfg: DictConfig,
    experiment_name: str,
    experiment_folder: str,
    viz_folder: str,
    old_field_index_map: Optional[Dict] = None,
    world_size: int = 1,
    rank: int = 0,
    local_rank: int = 0,
    device_mesh: Optional[torch.distributed.device_mesh.DeviceMesh] = None,
):
    """Instantiate the different objects required for training and run the training loop."""
    if cfg.get("seed") is not None:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
    logger.info(f"Instantiate datamodule {cfg.data.wandb_data_name}")
    # With spatial parallelism, every GPU of a spatial group reads the same
    # samples: data is split only across groups.
    spatial = get_spatial_context()
    datamodule: MixedWellDataModule = instantiate(
        cfg.data.module_parameters,
        world_size=spatial.dp_size if spatial else world_size,
        rank=spatial.dp_rank if spatial else rank,
        data_workers=cfg.data_workers,
        well_base_path=cfg.data.well_base_path,
        field_index_map_override=cfg.data.get("field_index_map_override", {}),
        transform=cfg.data.get("transform", None),
    )
    split_domain = spatial is not None and cfg.distribution.get("split_domain", False)
    if split_domain:
        assign_slabs(datamodule, spatial)  # each GPU reads only its slab
    ddstore_cfg = cfg.get("ddstore", None)
    if ddstore_cfg is not None and ddstore_cfg.get("enabled", False):
        from walrus.data.ddstore_source import DDStoreFieldSource

        # Processes reading the same slab share one store (all of them when
        # the domain is not split); each holds part of the training data.
        train_sets = list(getattr(datamodule.train_dataset, "sub_dsets",
                                  [datamodule.train_dataset]))
        source = DDStoreFieldSource(
            train_sets,
            group_color=spatial.rank if split_domain else 0,
            group_key=spatial.dp_rank if split_domain else rank,
            # One registered read buffer per loader thread
            reuse_slots=int(ddstore_cfg.get("workers", 2))
            if ddstore_cfg.get("reuse_buffers", True) else 0,
            # GPUDirect: reads land in this GPU's memory
            device=torch.device(f"cuda:{int(local_rank)}")
            if ddstore_cfg.get("gpu_direct", False) else None,
        )
        for dataset in train_sets:
            dataset.field_source = source
        datamodule.train_threads = int(ddstore_cfg.get("workers", 2))
        # No forked loader workers once MPI and DDStore are running
        datamodule.eval_in_threads = True
    field_to_index_map = datamodule.train_dataset.field_to_index_map
    # Retrieve the number of fields used in training
    # from the mapping of field to index and incrementing by 1
    total_input_fields = max(field_to_index_map.values()) + 1

    logger.info(
        f"Instantiate model {cfg.model._target_}",
    )
    model: torch.nn.Module = instantiate(
        cfg.model,
        n_states=total_input_fields,
    )
    # Checkpointing is a bit of a mess due to adding a second code path to deal with a bug that was later fixed, but the second
    # path remained since things are generally working and neither path currently has all the capabilities of the other.
    # Short version of logic:
    # 1) Standard path (sharded if FSDP/HSDP, standard if DDP/local) is used for finetuning and resuming training normally
    # 2) Secondary path allows for loading from "conventional" (non-sharded) checkpoints that were
    #    saved in any manner.
    # So the logic for handling all of this is as follows:
    # 1) Determine if we need to load from a coalesced checkpoint based on config
    #    1.1) If yes, and "load_chkpt_after_finetuning_expansion" is False or not set, load now
    # .       1.1.1) If align_fields is true, use the new and old field to index maps to make sure the embedding layers are aligned correctly.
    # 2) Do any weight modifications in finetuning_mods
    #    2.1) If coalesced checkpoint loading is specified and "load_chkpt_after_finetuning_expansion" is True, load now
    #        2.1.1) If align_fields is true, use the new and old field to index maps to make sure the embedding layers are aligned correctly.
    # 3) If autoresume is set and standard path checkpointing logic shows a checkpoint to load from, load now (overrides previous loads)
    #    3.1) Note - autoresume assumes this is the same model structure as the checkpoint, so no field alignment is done here.
    load_coalesced_chkpt_cond = (
        hasattr(cfg.checkpoint, "coalesced_checkpoint_path")
        and cfg.checkpoint.coalesced_checkpoint_path is not None
        and (cfg.finetune or cfg.validation_mode)
    )
    if load_coalesced_chkpt_cond and not cfg.checkpoint.get(
        "load_chkpt_after_finetuning_expansion", False
    ):
        model = load_from_coalesced_checkpoint(
            model=model,
            coalesced_checkpoint_path=cfg.checkpoint.coalesced_checkpoint_path,
            field_to_index_map=field_to_index_map,
            old_field_index_map=old_field_index_map,
            align_fields=cfg.checkpoint.get("align_fields", True),
        )
    # Finetuning changes - technically useable without FT too
    if hasattr(cfg, "finetuning_mods"):
        if hasattr(model, "add_ft_options"):
            model.add_ft_options(cfg.finetuning_mods)
    # If we're loading a checkpoint after finetuning expansion, do it here
    if load_coalesced_chkpt_cond and cfg.checkpoint.get(
        "load_chkpt_after_finetuning_expansion", False
    ):
        model = load_from_coalesced_checkpoint(
            model=model,
            coalesced_checkpoint_path=cfg.checkpoint.coalesced_checkpoint_path,
            field_to_index_map=field_to_index_map,
            old_field_index_map=old_field_index_map,
            align_fields=cfg.checkpoint.get("align_fields", True),
        )
    if rank == 0:
        summary(model, depth=5)

    logger.info(
        f"Assigning distribution strategy: {cfg.distribution.distribution_type}"
    )
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{int(local_rank)}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    model = model.to(device)
    if split_domain:
        enable_domain_split(model, spatial)
    model = distribute_model(model, cfg, device_mesh)

    logger.info(f"Instantiate optimizer {cfg.optimizer._target_}")
    if hasattr(cfg.optimizer, "param_groups"):
        with open_dict(cfg):
            param_groups_cfg = OmegaConf.to_container(
                cfg.optimizer.pop("param_groups"), resolve=True
            )
    else:
        param_groups_cfg = None

    # Param group cfg only exists if we want different learning rates for different parts of the model
    if param_groups_cfg is not None:
        param_groups = build_param_groups(model, param_groups_cfg=param_groups_cfg)
        optimizer = cast(
            torch.optim.Optimizer,
            instantiate(
                cfg.optimizer,
                params=param_groups,
                lr=cfg.optimizer.lr,
                _convert_="all",
            ),
        )
    # Otherwise just instantiate normally
    else:
        optimizer = cast(
            torch.optim.Optimizer,
            instantiate(
                cfg.optimizer,
                params=model.parameters(),
                lr=cfg.optimizer.lr,
                _convert_="all",
            ),
        )
    # Set start epoch to 1 before potential retrieval from checkpoint
    # honestly forget why this is 1 and not 0, might have just been aesthetics in the loop printout
    start_epoch = 1
    last_epoch = -1  # Default for Pytorch
    val_loss = torch.tensor(float("inf"))
    # Checkpointer manages standard path checkpoint loading/saving - step 3 in the above logic
    logger.info(f"Instantiate checkpointer {cfg.checkpoint._target_}")
    checkpointer: CheckPointLoader = instantiate(cfg.checkpoint, rank=rank)
    if hasattr(checkpointer, "load_checkpoint_path"):
        load_checkpoint_path = checkpointer.load_checkpoint_path
        if load_checkpoint_path is not None and os.path.exists(load_checkpoint_path):
            # Load model and optimizer from checkpoint
            # If this is a finetuning load, just load model weights
            if (
                cfg.finetune or cfg.validation_mode
            ) and load_checkpoint_path != checkpointer.last_checkpoint:
                logger.info(f"Finetuning from checkpoint {load_checkpoint_path}")
                checkpointer.load(
                    model,
                    local=cfg.distribution.distribution_type.upper()
                    in ["LOCAL", "DDP", "SPATIAL"],
                )
            # Otherwise this is a resume load
            else:
                logger.info(f"Resume from checkpoint {load_checkpoint_path}")
                epoch, val_loss = checkpointer.load(
                    model,
                    optimizer,
                    local=cfg.distribution.distribution_type.upper()
                    in ["LOCAL", "DDP", "SPATIAL"],
                )
                # Ensure initial_lr is set for each parameter group
                for i, param_group in enumerate(optimizer.param_groups):
                    if "initial_lr" not in param_group:
                        if param_groups_cfg is not None:
                            group_cfg = param_groups_cfg[i]
                            if "lr" in group_cfg:
                                param_group["initial_lr"] = group_cfg["lr"]
                            else:
                                param_group["initial_lr"] = cfg.optimizer.lr
                        else:
                            param_group["initial_lr"] = cfg.optimizer.lr

                logger.info(
                    f"Resume from epoch {epoch} with validation loss {val_loss}"
                )
                start_epoch = 1 if epoch is None else epoch + 1
                last_epoch = (
                    start_epoch - 1
                )  # Set last_epoch to the last completed epoch
    if hasattr(cfg, "lr_scheduler"):
        # Instantiate LR scheduler
        logger.info(f"Instantiate learning rate scheduler {cfg.lr_scheduler._target_}")
        # Option to convert from per-epoch scheduler to per-step scheduler
        if cfg.trainer.lr_scheduler_per_step:
            # NOTE(TM): This ideally should be max_iterations or something. Or T_max if we are to go with pytorch.
            step_mult_factor = (
                cfg.data.module_parameters.max_samples / cfg.trainer.grad_acc_steps
            )
        else:
            step_mult_factor = 1

        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = instantiate(
            cfg.lr_scheduler,
            optimizer=optimizer,
            max_epochs=cfg.trainer.max_epoch,
            step_mult_factor=step_mult_factor,
            last_epoch=max(-1, last_epoch - 1),
        )
    else:
        logger.info("No learning rate scheduler")
        lr_scheduler = None

    # Update the config with the newly generated field-to-index map for resuming/knowing what was there
    with open_dict(cfg):
        cfg.data.field_index_map_override = field_to_index_map
    if rank == 0:
        logger.info(f"Final configuration:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"Instantiate trainer {cfg.trainer._target_}")
    # These are used for aggregating per-sample metrics in validation.
    # Defaults to mean.
    batch_aggregation_fns = [
        get_method(name)
        for name in cfg.trainer.get("batch_aggregation_fns", ["torch.mean"])
    ]
    trainer: Trainer = instantiate(
        cfg.trainer,
        experiment_name=experiment_name,
        batch_aggregation_fns=batch_aggregation_fns,
        viz_folder=viz_folder,
        model=model,
        datamodule=datamodule,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpointer=checkpointer,
        device=device,
        device_mesh=device_mesh,
        distribution_type=cfg.distribution.distribution_type,
        rank=rank,
        world_size=world_size,
        formatter=ChannelsFirstWithTimeFormatter,  # TODO change this to function of model
        wandb_logging=cfg.logger.wandb,
        start_epoch=start_epoch,
        start_val_loss=val_loss,
    )
    if split_domain:
        trainer.enable_domain_split(spatial)
    # Validation mode only runs validation loop on valid and test data. No training.
    if cfg.validation_mode:
        # If we're validating to a different directory, still copy the config here so we know where it came from
        if rank == 0 and not os.path.exists(
            pathlib.Path(experiment_folder) / "extended_config.yaml"
        ):
            with open(
                pathlib.Path(experiment_folder) / "extended_config.yaml", "w"
            ) as f:
                OmegaConf.save(cfg, f)
        trainer.validate()
    else:
        # Save config to directory folder so we can track experiment settings
        if rank == 0:
            with open(
                pathlib.Path(experiment_folder) / "extended_config.yaml", "w"
            ) as f:
                OmegaConf.save(cfg, f)
        trainer.train()


@hydra.main(
    version_base=None, config_path=str(CONFIG_DIR), config_name=str(CONFIG_NAME)
)
def main(cfg: DictConfig):
    # Torch optimization settings
    torch.set_float32_matmul_precision("high")  # Use TF32 when supported
    torch.backends.cudnn.allow_tf32 = True
    # Retrieve multiple processes context to setup DDP (from torchrun or srun)
    setup_env_from_slurm()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = (
        cfg.distribution.distribution_type.upper() != "LOCAL" and world_size > 1
    )

    # DDStore's MPI starts before torch.distributed sets up RCCL/NCCL
    if cfg.get("ddstore", None) is not None and cfg.ddstore.get("enabled", False):
        from walrus.data.ddstore_source import init_mpi

        init_mpi()
    # Since configure_experiment uses distributed logic, distribution must be set up first
    device_mesh = configure_distribution(cfg)
    (
        cfg,
        experiment_name,
        experiment_folder,
        checkpoint_folder,  # This path was given to the checkpoint config which is where it is used. This output not used in later code.
        artifact_folder,  # Actually not used anywhere, but kept in case added later.
        viz_folder,
        old_field_index_map,
    ) = configure_experiment(cfg, rank, is_distributed)

    logger.info(f"Run experiment {experiment_name}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    # Initiate wandb logging

    # Make sure we're logging the true batch size
    config_for_wandb = cast(Dict, OmegaConf.to_container(cfg, resolve=True))
    config_for_wandb["world_size"] = world_size
    # Global batch size is microbatch size * number of GPUs * gradient accumulation steps
    # Though grad acc reduces the number of optimizer steps
    spatial = get_spatial_context()
    config_for_wandb["global_batch_size"] = (
        cfg.data.module_parameters.batch_size
        * (spatial.dp_size if spatial else world_size)
    ) * cfg.trainer.grad_acc_steps
    if rank == 0 and cfg.logger.wandb:
        wandb.init(
            project=cfg.logger.wandb_project_name,
            group=f"{cfg.data.wandb_data_name}",
            config=config_for_wandb,
            name=experiment_name,
        )
    train(
        cfg,
        experiment_name,
        experiment_folder,
        viz_folder,
        old_field_index_map,
        world_size,
        rank,
        local_rank,
        device_mesh=device_mesh,
    )
    if rank == 0 and cfg.logger.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
