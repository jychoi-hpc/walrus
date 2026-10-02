## Key settings overview

Mostly self-explanatory:

```yaml
distribution_type: local # Distribution types "local", "ddp", "fsdp", "hsdp", "spatial"
local_size: # For HSDP, if sub-dividing nodes, specify to what degree here.
```

## Spatial parallelism (`distribution: spatial`)

Spatial parallelism splits **each sample's grid** across the GPUs of a
*spatial group*, so grids too large for one GPU can be trained on whole, with
no cropping or downsampling. The domain is cut into slabs along one axis, one
slab per GPU; every GPU runs the full model on its slab and the layers that
need other slabs exchange data (domain decomposition, as in a CFD solver).
Data parallelism runs across groups: with N GPUs and groups of S, N/S groups
train on different samples.

```yaml
distribution_type: spatial
spatial_size: 4      # GPUs per spatial group (default: GPUs per node)
spatial_axis: 0      # axis split across the group (0 = x)
split_domain: true   # false: every GPU of a group holds the whole sample
```

Example: the full-resolution BLASTNet channel flow (1536 x 384 x 1024, 19 GB per
sample) does not fit one 80 GB GPU; split into 4 slabs of 384 it trains at
~24 GB per GPU on one node, and across nodes with data parallelism.

### What changes when the domain is split

| Part | On a split domain |
|---|---|
| Data | each GPU reads only its slab (the_well `slab` option, set by `walrus.data.slabs.assign_slabs`); slab edges fall on the patch size chosen for the full grid |
| Patch jitter, token roll | one shared random shift; data crossing a slab border goes to the GPU that needs it (`walrus/utils/distributed_gather.py`) |
| Attention | Ulysses all-to-all: each GPU holds all tokens for 1/S of the heads (`walrus/utils/spatial_attention.py`) |
| RMSGroupNorm, RevIN | statistics summed over the group |
| Rotary positions | each slab uses its place in the full grid |
| Encoder / decoder | no communication (kernel = stride along the split axis) |
| Drop path, field dropout, rolls | drawn on the group's first rank and broadcast |
| Loss | each GPU's share of the full-domain loss (`walrus/trainer/spatial_reductions.py`); gradients averaged over all GPUs = summed per group, averaged over groups |
| Validation metrics | built from slab sums: whole-domain values (`walrus/trainer/spatial_metrics.py`) |

### Requirements

- The split axis is **periodic** and its patch has **kernel = stride** (Walrus's
  3D base kernels (8, 4): patch 32, i.e. 512 points, or more than 512 and a
  multiple of 32). Other cases raise `NotImplementedError` rather than compute
  something wrong.
- Attention **heads are a multiple of** `spatial_size` (the default model has 12).
- **No data transforms** (a transform of a slab is not the slab of the
  transformed sample).
- Launch one process per GPU with `srun` (Slurm variables are mapped to
  `torch.distributed` by `setup_env_from_slurm`) or `torchrun`.

### Recommended settings at large grids

```bash
distribution=spatial distribution.split_domain=true distribution.spatial_size=4
seed=0                                  # reproducible runs (weights are also broadcast at startup)
trainer.enable_amp=true +trainer.amp_type=bfloat16   # float16 overflows in the feed-forward
+data.module_parameters.return_grid=false            # coordinate grids are unused in training
+data.module_parameters.prefetch_factor=1             # each batch in flight is GBs of host memory
+data.module_parameters.pin_memory=false
+data.module_parameters.eval_data_workers=1           # rollout validation loads whole trajectories
data_workers=3
```

Measured on the full-resolution channel flow (4 x A100 80 GB per node,
bfloat16): ~3.5 s per training step and 24 GB per GPU in training; with
5-step rollout validation and 1 eval worker per GPU, 58 GB per GPU and 194 GB
host memory per node. 2 nodes (2 groups of 4) scale at ~67% efficiency.

### How it is tested

- `tests/test_spatial_equivalence.py`: every split layer, and the whole model,
  against the same layer on the full grid in float64 (outputs, input and
  parameter gradients), on 4 and 2 CPU processes; uneven slabs included.
- End to end (scripts in the walrus-blastnet-ai repository): training on 1 GPU
  vs 4 GPUs with a split domain gives the same losses, and validation/test
  losses to 1.6e-7; with 2 groups of 2 GPUs, gradients match the mean over the
  two samples on 1 GPU.

### Limitations

Spectral metrics, videos, image plots and prediction dumps are skipped on a
split domain (each GPU holds only its slab). Validation metrics without a
split-domain version raise `NotImplementedError`.
`FixedPatchJittererBoundaryPad` is not supported on a split domain.
