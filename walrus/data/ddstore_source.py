"""Serve the time-varying fields of training data from DDStore (data held in
the nodes' host memory, read over libfabric RDMA) instead of reading the HDF5
files every step.

Storage is DDStore's own (pyddstore.torch, DDStore >= 3699e57): one
DistDataset per Walrus dataset, over a ConcatDataset of per-file "frame"
sources. A frame is one time step of one trajectory: this GPU's slab of every
stored field (float32 fields that vary by sample and time over all spatial
dims), stored once, so samples that share a step (i, i+1) do not store it
twice. The processes that read the same slab of the domain (same spatial
rank; everyone when the domain is not split) share one store (the `comm` of
the DistDataset), and each holds a contiguous share of the frames, loaded one
frame at a time (chunk_size=1). DDStore splits reads larger than the
network's message size itself.

A sample (time steps time_idx, time_idx + dt, ...) is one read_rows() per
field: one get_batch(). With reuse_slots > 0, each loader thread reads into
its own buffers from alloc(), registered with the network card once; fresh
buffers would be registered (pinned) on every read, which costs more than the
transfer. A returned array is then a view of the thread's buffer, valid until
that thread's next read: the_well copies the fields (concatenation) before
__getitem__ returns.

Only DDStore method 1 (libfabric) is used; set DDSTORE_FABRIC=cxi on
Perlmutter. Reads run without the GIL, so loader threads fetch upcoming
samples while the training step runs (pyddstore.torch.ThreadDataLoader).
"""

import itertools
import logging
import os
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)

FIELD_GROUPS = ("t0_fields", "t1_fields", "t2_fields")


def _stored_fields(path) -> List[str]:
    """Keys ("t1_fields/velocity", ...) of the fields DDStore holds: float32,
    varying by sample and time, over all spatial dims."""
    keys = []
    with h5py.File(path, "r") as f:
        for group in FIELD_GROUPS:
            if group not in f:
                continue
            for name in f[group].attrs["field_names"]:
                a = f[group][name].attrs
                if (a["time_varying"] and a["sample_varying"] and all(a["dim_varying"])
                        and f[group][name].dtype == np.float32):
                    keys.append(f"{group}/{name}")
    return keys


class _FrameSource:
    """Frames of one HDF5 file: item i is (trajectory, step) = divmod(i,
    n_steps), as a dict of this slab of every stored field."""

    def __init__(self, path, keys, slab, timer, as_torch=False):
        self.path, self.keys, self.slab, self.timer = path, keys, slab, timer
        # torch fields: DistDataset(device=) puts only torch fields on the GPU
        self.as_torch = as_torch
        with h5py.File(path, "r") as f:
            self.n_samples, self.n_steps = f[keys[0]].shape[:2]
        self._file = None

    def __len__(self):
        return self.n_samples * self.n_steps

    def __getitem__(self, i):
        t = time.perf_counter()
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        sample, step = divmod(int(i), self.n_steps)
        frame = {}
        for key in self.keys:
            field = self._file[key]
            index = [slice(None)] * (field.ndim - 2)
            if self.slab is not None:
                axis, start, stop = self.slab
                index[axis] = slice(start, stop)
            frame[key] = np.ascontiguousarray(field[(sample, step, *index)])
            if self.as_torch:
                import torch
                frame[key] = torch.from_numpy(frame[key])
        self.timer["hdf5"] += time.perf_counter() - t
        return frame

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def init_mpi():
    """Initialize MPI (mpi4py) for DDStore: serialized, from one thread.
    Called before torch.distributed starts RCCL/NCCL when DDStore is on: on
    Frontier with the RCCL network plugin, MPI_Init after RCCL's setup failed
    to create its network endpoint ("No space left on device")."""
    import torch  # noqa: F401  (torch must load before mpi4py initializes MPI)
    import mpi4py

    mpi4py.rc.thread_level = "serialized"
    mpi4py.rc.threads = False
    from mpi4py import MPI
    return MPI


class DDStoreFieldSource:
    """Field source for the_well's WellDataset (its `field_source` attribute)."""

    def __init__(self, datasets: Iterable, group_color: int, group_key: int,
                 reuse_slots: int = 0, device=None):
        MPI = init_mpi()
        from torch.utils.data import ConcatDataset

        try:
            from pyddstore.torch import DistDataset, row_of
        except ImportError as exc:
            raise ImportError("ddstore.enabled needs DDStore with pyddstore.torch "
                              "DistDataset.read_rows (check-thread 3699e57 or later)") from exc

        if os.environ.get("DDSTORE_FABRIC") is None:
            logger.warning("DDSTORE_FABRIC is not set (use cxi on Perlmutter)")
        self.comm = MPI.COMM_WORLD.Split(group_color, group_key)
        self.world_rank = MPI.COMM_WORLD.Get_rank()
        self.reuse_slots = int(reuse_slots)
        # GPUDirect (device set): reads and read buffers on this GPU. DDStore's
        # get_batch() synchronizes the device before reading into a GPU
        # buffer, so a reused buffer is not overwritten while a kernel still
        # copies out of it.
        self.device = device
        self._row_of = row_of
        # path -> (store, concat, source index, n_steps per trajectory, keys, slab)
        self._files: Dict[str, tuple] = {}
        self._stores = []
        self._bufs_lock = threading.Lock()
        self._thread = threading.local()
        self._slot_ids = itertools.count()
        # WALRUS_DDSTORE_TIMING=1: log every read (thread, field, sample, time)
        # and, with DDSTORE_PROFILE=1, DDStore's counters every 8th read
        self._timing = os.environ.get("WALRUS_DDSTORE_TIMING", "0") not in ("", "0")
        self._reads: Dict[str, int] = {}
        self.load_times = {"hdf5": 0.0}
        load_error: Optional[str] = None
        t0 = time.perf_counter()
        for k, dataset in enumerate(datasets):
            slab = getattr(dataset, "slab", None)
            paths = list(dataset.files_paths)
            keys = _stored_fields(paths[0]) if paths else []
            if not keys:
                continue
            sources = [_FrameSource(p, keys, slab, self.load_times, as_torch=device is not None)
                       for p in paths]
            concat = ConcatDataset(sources)
            try:
                # DistDataset raises on every rank of its group if any fails
                store = DistDataset(concat, f"walrus{k}", comm=self.comm, method=1,
                                    chunk_size=1, device=device)
            except Exception as exc:  # noqa: BLE001 - raised on all ranks below
                load_error = f"rank {self.world_rank}: {type(exc).__name__}: {exc}"
                break
            finally:
                for s in sources:
                    s.close()
            self._stores.append(store)
            for i, (p, s) in enumerate(zip(paths, sources)):
                self._files[p] = (store, concat, i, s.n_steps, keys, slab)
        # One group's failure raises on every rank, not only in that group
        errors = [e for e in MPI.COMM_WORLD.allgather(load_error) if e]
        if errors:
            raise RuntimeError(f"DDStore loading failed on {len(errors)} rank(s): {errors[0]}")
        self.comm.Barrier()
        self.load_times["total"] = time.perf_counter() - t0
        lt = self.load_times
        n_frames = sum(len(s) for s in self._stores)
        logger.info(
            f"DDStore: {n_frames} frames of {len(self._stores)} dataset(s) loaded in "
            f"{lt['total']:.1f} s (HDF5 reads {lt['hdf5']:.1f} s, store "
            f"{lt['total'] - lt['hdf5']:.1f} s); group of {self.comm.Get_size()}; "
            f"reads into {self.device or 'host memory'}"
        )

    def _slot(self) -> Optional[int]:
        """This thread's buffer slot; None when every slot is taken."""
        slot = getattr(self._thread, "slot", None)
        if slot is None:
            with self._bufs_lock:
                slot = next(self._slot_ids)
            self._thread.slot = slot
            self._thread.bufs = {}
        return slot if slot < self.reuse_slots else None

    def _buffers(self, store, key, n_rows):
        """This thread's registered buffers for `key` (reuse_slots), or None."""
        if not self.reuse_slots or self._slot() is None:
            return None
        bufs = self._thread.bufs
        k = (id(store), key, n_rows)
        if k not in bufs:
            bufs[k] = store.alloc(n_rows, fields=[key])
        return bufs[k]

    def read(self, path, field_key, sample_idx, time_idx, n_steps, dt, slab):
        """Fields of steps time_idx, time_idx + dt, ... (n_steps of them) for
        this slab, shaped as the HDF5 read would return them; None when the
        field is not held. With reuse_slots, a view of this thread's buffer."""
        entry = self._files.get(path)
        if entry is None:
            return None
        store, concat, source, steps_per_traj, keys, held_slab = entry
        if field_key not in keys or slab != held_slab:
            return None
        rows = [self._row_of(concat, source, sample_idx * steps_per_traj + time_idx + k * dt)
                for k in range(n_steps)]
        t0 = time.perf_counter()
        out = self._buffers(store, field_key, n_steps)
        t1 = time.perf_counter()
        value = store.read_rows(rows, fields=[field_key], out=out)[field_key]
        if self._timing:
            self._log_read(store, field_key, sample_idx, time_idx, len(rows),
                           out is not None, t1 - t0, time.perf_counter() - t1,
                           path, rows, value)
        return value

    def _log_read(self, store, key, sample_idx, time_idx, n_rows, reused, prep_s, read_s,
                  path=None, rows=None, value=None):
        n = self._reads[key] = self._reads.get(key, 0) + 1
        # Non-finite values in what was read (diagnosis: bad data vs bad model)
        bad = ""
        if value is not None:
            import torch
            v = torch.as_tensor(value)
            bad = f", non-finite {int(v.numel() - torch.isfinite(v).sum())}"
        logger.info(
            f"DDStore read rank {self.world_rank} {threading.current_thread().name} "
            f"{os.path.basename(path) if path else ''} {key} sample {sample_idx} t {time_idx} "
            f"rows {rows if rows is not None else n_rows} "
            f"{'reused' if reused else 'fresh'} buffer: buffer {prep_s:.3f} s, "
            f"read_rows {read_s:.3f} s{bad}"
        )
        if n % 8 == 0 and hasattr(store.ddstore, "get_profile"):
            prof = store.ddstore.get_profile(f"{store.name}/{key}")
            logger.info(f"DDStore profile rank {self.world_rank} {key} after {n} reads: " + ", ".join(
                f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in prof.items()))

    def free(self):
        for store in self._stores:
            store.ddstore.free()
        self.comm.Barrier()
        self.comm.Free()
