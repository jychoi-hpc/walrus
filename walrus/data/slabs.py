"""Give every dataset of a datamodule this GPU's slab of the domain when the
domain is split across a spatial group (tracker task t303)."""

import logging

from walrus.models.shared_utils.flexi_utils import choose_kernel_size_deterministic
from walrus.utils.spatial import SpatialContext, slab_bounds

logger = logging.getLogger(__name__)


def _datasets(datamodule):
    """Every WellDataset of the datamodule (train, valid, rollout, test)."""
    groups = [[datamodule.train_dataset]]
    for name in ("val_datasets", "rollout_val_datasets", "test_datasets",
                 "rollout_test_datasets"):
        groups.append(getattr(datamodule, name, []) or [])
    for group in groups:
        for mixed in group:
            yield from getattr(mixed, "sub_dsets", [mixed])


def assign_slabs(datamodule, ctx: SpatialContext) -> None:
    """Each dataset reads only this rank's slab along the split axis. Slab
    edges fall on multiples of the patch size chosen for the full grid, so
    every patch lies within one slab."""
    for dataset in _datasets(datamodule):
        if getattr(dataset, "transform", None) is not None:
            # A transform of one slab (e.g. a resize) is not the slab of the
            # transformed sample
            raise NotImplementedError(
                f"{dataset.metadata.dataset_name}: data transforms on a split domain"
            )
        shape = tuple(dataset.metadata.spatial_resolution)
        kernels = choose_kernel_size_deterministic(shape)
        patch = kernels[ctx.axis][0] * kernels[ctx.axis][1]
        start, stop = slab_bounds(shape[ctx.axis], ctx.size, ctx.rank, align=patch)
        dataset.slab = (ctx.axis, start, stop)
        logger.info(
            f"{dataset.metadata.dataset_name}: slab {start}:{stop} of {shape[ctx.axis]} "
            f"along axis {ctx.axis} (patch {patch})"
        )
