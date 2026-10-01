import logging
import os
import re
from functools import partial

import torch
from hydra.utils import get_class
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy, size_based_auto_wrap_policy

from walrus.utils.spatial import DATA_PARALLEL_DIM, SPATIAL_DIM, set_spatial_context

logger = logging.getLogger(__name__)


def _first_host(nodelist: str) -> str:
    """First host of a Slurm node list: 'nid[001234-001237,001240],x' -> 'nid001234'."""
    m = re.match(r"([^\[,]*)(?:\[([^\]]*)\])?", nodelist)
    if not m or not m.group(1):
        return ""
    prefix, ranges = m.group(1), m.group(2)
    return prefix + (ranges.split(",")[0].split("-")[0] if ranges else "")


def setup_env_from_slurm():
    """Under `srun` (one task per GPU), fill the torch.distributed variables
    from Slurm's: RANK <- SLURM_PROCID, WORLD_SIZE <- SLURM_NTASKS,
    LOCAL_RANK <- SLURM_LOCALID, LOCAL_WORLD_SIZE <- SLURM_NTASKS_PER_NODE,
    plus MASTER_ADDR and MASTER_PORT. Variables already set (e.g. by
    torchrun) win."""
    env = os.environ
    if "SLURM_PROCID" not in env:
        return
    defaults = {
        "RANK": env["SLURM_PROCID"],
        "WORLD_SIZE": env.get("SLURM_NTASKS", "1"),
        "LOCAL_RANK": env.get("SLURM_LOCALID", "0"),
        "LOCAL_WORLD_SIZE": env.get("SLURM_NTASKS_PER_NODE", "1"),
        # Rank 0 runs on the first node of the step. (Not the launch node: under
        # salloc that is the login node.)
        "MASTER_ADDR": _first_host(
            env.get("SLURM_STEP_NODELIST") or env.get("SLURM_JOB_NODELIST", "")
        )
        or "127.0.0.1",
        "MASTER_PORT": "29500",
    }
    for key, value in defaults.items():
        env.setdefault(key, value)


def configure_distribution(cfg):
    """Configure the appropriate device mesh for the given distribution parameters.

    Parameters:
    ----------
    cfg: DictConfig
        The configuration object for the experiment which contains the appropriate distribution type.

    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = int(
        cfg.distribution.local_size or os.environ.get("LOCAL_WORLD_SIZE", 1)
    )
    # Initialize the device mesh - right now we have FSDP, HSDP, and DDP - HSDP is the only
    # multi-level sharding strategy

    # NOTE - Documentation suggests you do not need torch.cuda.set_device(local_rank) when using
    # device_mesh, but I got errors until I set it in train.py.
    if cfg.distribution.distribution_type.upper() == "LOCAL":
        return None
    elif cfg.distribution.distribution_type.upper() == "SPATIAL":
        spatial_size = int(cfg.distribution.get("spatial_size") or local_world_size)
        if world_size % spatial_size != 0:
            raise ValueError(
                f"World size {world_size} is not a multiple of spatial_size {spatial_size}"
            )
        mesh = init_device_mesh(
            "cuda",
            (world_size // spatial_size, spatial_size),
            mesh_dim_names=(DATA_PARALLEL_DIM, SPATIAL_DIM),
        )
        set_spatial_context(mesh, axis=int(cfg.distribution.get("spatial_axis", 0)))
    elif cfg.distribution.distribution_type.upper() == "HSDP":
        # HSDP is DDP across nodes, FSDP within nodes - useful if  intra-node xfer speed >>> inter-node xfer speed
        mesh = init_device_mesh(
            "cuda",
            (world_size // local_world_size, local_world_size),
            mesh_dim_names=("ddp", "fsdp"),
        )
    else:
        mesh = init_device_mesh(
            "cuda",
            (world_size,),
            mesh_dim_names=(cfg.distribution.distribution_type.lower(),),
        )
    return mesh


def distribute_model(model, cfg, mesh=None):
    """Distributes model over a given mesh using the appropriate distribution strategy.

    For Walrus, default FSDP wrapping is based on the Walrus definition schema. Otherwise
        - If the model has encoder, decoder, and processor attributes, those are wrapped.
        - If the model has blocks attribute, the first block's class is used for wrapping.
        - Otherwise, size_based_auto_wrap_policy with min_num_params=1e6 is used.

    Parameters:
    ----------
    model: nn.Module
        The model to be distributed.
    cfg: DictConfig
        The configuration object for the experiment which contains the appropriate distribution type.
    mesh: None | torch.distributed.device_mesh.DeviceMesh
        The device mesh to distribute the model over.
    """
    # If no mesh is provided, there is no valid distribution strategy
    if mesh is None:
        return model
    elif cfg.distribution.distribution_type.upper() in ["FSDP", "HSDP", "DDP", "SPATIAL"]:
        # FSDP and HSDP both use FSDP API, so need to build in AMP if we're going to use it.
        if cfg.trainer.enable_amp:
            fpSixteen = MixedPrecision(
                param_dtype=torch.float16,
                # Gradient communication precision.
                reduce_dtype=torch.float16,
                # Buffer precision.
                buffer_dtype=torch.float16,
            )
        else:
            fpSixteen = None

        # TODO - Add in more FSDP parameters from the config including Zero2
        if cfg.distribution.distribution_type.upper() == "FSDP":
            sharding = ShardingStrategy.FULL_SHARD
        elif cfg.distribution.distribution_type.upper() == "HSDP":
            sharding = ShardingStrategy.HYBRID_SHARD
        elif cfg.distribution.distribution_type.upper() in ["DDP", "SPATIAL"]:
            sharding = ShardingStrategy.NO_SHARD
        # TODO - should make this configurable at some point, but this
        # is a good default for now. Shards per block where block is the encoder, decoder,
        # or processor. For larger enc/dec it probably makes sense to have more blocks.
        if cfg.distribution.distribution_type.upper() not in ["DDP", "SPATIAL"]:
            if (
                hasattr(cfg.model, "encoder")
                and hasattr(cfg.model, "decoder")
                and hasattr(cfg.model, "processor")
            ):
                wrap_policy = ModuleWrapPolicy(
                    [
                        get_class(
                            cfg.model.encoder._target_
                        ),  # Better performance, but has some bugs with grad accum in FSDP
                        get_class(
                            cfg.model.decoder._target_
                        ),  # Better performance, but has some bugs with grad accum in FSDP
                        get_class(cfg.model.processor._target_),
                    ]
                )
            elif hasattr(model, "blocks"):
                wrap_policy = ModuleWrapPolicy([model.blocks[0].__class__])
            else:
                wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=1e6)
        else:
            wrap_policy = None
        model = FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            mixed_precision=fpSixteen,
            sharding_strategy=sharding,
            # Spatial: weights are replicated and gradients averaged over all
            # ranks (the default group). Until layers split the domain, every
            # rank in a spatial group sees the same batch, so this average is
            # the data-parallel average.
            device_mesh=None
            if cfg.distribution.distribution_type.upper() == "SPATIAL"
            else mesh,
            use_orig_params=True,
        )
    else:
        raise ValueError(
            f"Unknown distribution type {cfg.distribution.distribution_type} - must be LOCAL, DDP, FSDP, HSDP or SPATIAL"
        )
    return model
