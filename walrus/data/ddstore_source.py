"""Serve the time-varying fields of training data from DDStore (data held in
the nodes' host memory, read over libfabric RDMA) instead of reading the HDF5
files every step.

Each process loads part of the data once at startup. The processes that read
the same slab of the domain (same spatial rank; everyone when the domain is
not split) share one store, and each of them holds a contiguous share of the
(trajectory, time step) pairs of every file. A field's slab of one time step
is stored as rows of `planes` points along the slab axis, at most
`max_row_mb` each (one RDMA read is limited to under 2 GB on Perlmutter's cxi
network).

Only DDStore method 1 (libfabric) is used; set DDSTORE_FABRIC=cxi on
Perlmutter. Reads run without the GIL, so a background thread can fetch the
next sample while the training step runs (see walrus.data.prefetch).
"""

import logging
import os
import time
from typing import Dict, Iterable, Optional, Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)

FIELD_GROUPS = ("t0_fields", "t1_fields", "t2_fields")


def _largest_divisor_at_most(n: int, limit: int) -> int:
    return max(d for d in range(1, max(1, min(n, limit)) + 1) if n % d == 0)


class _Variable:
    """One field of one file: rows = (trajectory, step, chunk), chunk fastest."""

    def __init__(self, name, key, n_samples, n_steps, slab_shape, components, axis, planes):
        self.name, self.key = name, key
        self.n_samples, self.n_steps = n_samples, n_steps
        self.slab_shape, self.components, self.axis = slab_shape, components, axis
        self.planes = planes
        self.chunks = slab_shape[axis] // planes
        plane = int(np.prod(slab_shape)) // slab_shape[axis] * components
        self.disp = plane * planes  # float32 values per row
        self.chunk_shape = tuple(
            planes if d == axis else n for d, n in enumerate(slab_shape)
        ) + ((components,) if components > 1 else ())

    def row(self, sample: int, step: int, chunk: int) -> int:
        return (sample * self.n_steps + step) * self.chunks + chunk


class DDStoreFieldSource:
    """Field source for the_well's WellDataset (its `field_source` attribute)."""

    def __init__(self, datasets: Iterable, group_color: int, group_key: int,
                 max_row_mb: float = 512.0):
        import torch  # noqa: F401  (torch must load before mpi4py initializes MPI)
        import mpi4py

        mpi4py.rc.thread_level = "serialized"
        mpi4py.rc.threads = False
        from mpi4py import MPI
        import pyddstore as dds

        if os.environ.get("DDSTORE_FABRIC") is None:
            logger.warning("DDSTORE_FABRIC is not set (use cxi on Perlmutter)")
        self.comm = MPI.COMM_WORLD.Split(group_color, group_key)
        g_rank, g_size = self.comm.Get_rank(), self.comm.Get_size()
        self.store = dds.PyDDStore(self.comm, method=1)
        self.variables: Dict[Tuple[str, str, Optional[tuple]], _Variable] = {}
        self._files = []
        t0, loaded = time.perf_counter(), 0
        for dataset in datasets:
            slab = getattr(dataset, "slab", None)
            for path in dataset.files_paths:
                loaded += self._load_file(path, slab, f"d{len(self._files)}", g_rank, g_size,
                                          max_row_mb)
                self._files.append(path)
        self.comm.Barrier()
        total = self.comm.allreduce(loaded, op=MPI.SUM)
        logger.info(
            f"DDStore: {len(self.variables)} fields loaded in {time.perf_counter() - t0:.1f} s; "
            f"this process holds {loaded / 1024**3:.1f} GB, its group of {g_size} "
            f"holds {total / 1024**3:.1f} GB"
        )

    def _load_file(self, path, slab, prefix, g_rank, g_size, max_row_mb) -> int:
        loaded = 0
        with h5py.File(path, "r") as f:
            for group in FIELD_GROUPS:
                if group not in f:
                    continue
                for field_name in f[group].attrs["field_names"]:
                    field = f[group][field_name]
                    a = field.attrs
                    if not (a["time_varying"] and a["sample_varying"]) or not all(a["dim_varying"]):
                        continue  # read from the file as before
                    n_samples, n_steps = field.shape[:2]
                    spatial = field.shape[2:2 + len(a["dim_varying"])]
                    components = int(np.prod(field.shape[2 + len(spatial):], dtype=int))
                    if field.dtype != np.float32:
                        continue
                    axis = slab[0] if slab is not None else 0
                    start, stop = (slab[1], slab[2]) if slab is not None else (0, spatial[0])
                    slab_shape = tuple(stop - start if d == axis else n for d, n in enumerate(spatial))
                    plane_bytes = int(np.prod(slab_shape)) // slab_shape[axis] * components * 4
                    planes = _largest_divisor_at_most(
                        slab_shape[axis], int(max_row_mb * 1024**2 // plane_bytes))
                    key = f"{group}/{field_name}"
                    var = _Variable(f"{prefix}/{key}", key, n_samples, n_steps,
                                    slab_shape, components, axis, planes)
                    pairs = np.array_split(np.arange(n_samples * n_steps), g_size)[g_rank]
                    self.store.init(var.name, len(pairs) * var.chunks, var.disp, 4)
                    for i, pair in enumerate(pairs):
                        sample, step = divmod(int(pair), n_steps)
                        for c in range(var.chunks):
                            lo = start + c * planes
                            index = [slice(None)] * len(spatial)
                            index[axis] = slice(lo, lo + planes)
                            block = np.ascontiguousarray(field[(sample, step, *index)])
                            self.store.update(var.name, block.reshape(1, -1), i * var.chunks + c)
                            loaded += block.nbytes
                    self.variables[(path, key, slab)] = var
        return loaded

    def read(self, path, field_key, sample_idx, time_idx, n_steps, dt, slab):
        """Fields of steps time_idx, time_idx + dt, ... (n_steps of them) for
        this slab, shaped as the HDF5 read would return them; None when the
        field is not held."""
        var = self.variables.get((path, field_key, slab))
        if var is None:
            return None
        out = np.empty((n_steps,) + var.slab_shape + ((var.components,) if var.components > 1 else ()),
                       dtype=np.float32)
        for k in range(n_steps):
            step = time_idx + k * dt
            for c in range(var.chunks):
                index = [slice(None)] * len(var.slab_shape)
                index[var.axis] = slice(c * var.planes, (c + 1) * var.planes)
                dst = out[(k, *index)]
                if dst.flags.c_contiguous:
                    self.store.get(var.name, dst.reshape(1, -1), start=var.row(sample_idx, step, c))
                else:
                    buf = np.empty((1, var.disp), np.float32)
                    self.store.get(var.name, buf, start=var.row(sample_idx, step, c))
                    dst[...] = buf.reshape(var.chunk_shape)
        return out

    def free(self):
        self.store.free()
        self.comm.Barrier()
        self.comm.Free()
