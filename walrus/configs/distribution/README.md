## Key settings overview

Mostly self-explanatory:

```yaml
distribution_type: local # Distribution types "local", "ddp", "fsdp", "hsdp", "spatial"
local_size: # For HSDP, if sub-dividing nodes, specify to what degree here.
```

`spatial` (in progress) splits each sample's grid across the GPUs of a spatial group:

```yaml
distribution_type: spatial
spatial_size: # GPUs per spatial group; defaults to the GPUs per node
spatial_axis: 0 # Axis split across the group (0 = x)
```

Every GPU in a spatial group sees the same batches, and data parallelism runs
across groups. Weights are replicated. Layers look up their group with
`walrus.utils.spatial.get_spatial_context()`.