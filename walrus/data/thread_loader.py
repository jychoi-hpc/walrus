"""DataLoader that runs __getitem__ in a pool of threads instead of forked
worker processes, for training data served from DDStore: forking after MPI is
initialized hangs with DDStore, while its reads release the GIL and run in
threads (one at a time per store; the rest of __getitem__ overlaps).

Adapted from ThreadDataLoader in DDStore's examples/vae/ddstore_dataloader.py
(ORNL/DDStore), with two changes for Walrus's multi-GB batches:
  - at most max(num_workers, prefetch) batches are fetched ahead of the
    consumer, by default one per thread (the example submits a whole epoch at
    once, keeping every finished batch in memory);
  - batch_size=None (Walrus's samplers yield whole batches) calls
    dataset[index] directly, as PyTorch's own loader does.
Collation and pinning run in the worker threads, not in the training loop.
Like torch's loader, each epoch draws one number from the global RNG, so the
random numbers the training loop draws afterwards are the same.
"""

import collections
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class ThreadDataLoader(DataLoader):
    def __init__(self, dataset, num_workers: int = 1, prefetch: int = 0, **kwargs):
        kwargs.pop("prefetch_factor", None)
        # The DataLoader itself must not fork workers
        super().__init__(dataset, num_workers=0, **kwargs)
        self.threads = max(1, int(num_workers))
        # Batches submitted ahead of the consumer (in flight or finished)
        self.ahead = max(self.threads, int(prefetch))
        self._worker_ids = iter(range(self.threads))
        self._worker_ids_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(
            max_workers=self.threads, thread_name_prefix="data-loader",
            initializer=self._worker_init,
        )
        logger.info(f"ThreadDataLoader: {self.threads} threads, up to {self.ahead} batches ahead")

    def _worker_init(self):
        """Optionally pin each thread to its own cores, as in DDStore's example:
        DDSTORE_AFFINITY_WIDTH cores per thread from DDSTORE_AFFINITY_OFFSET on."""
        width = int(os.environ.get("DDSTORE_AFFINITY_WIDTH", "0"))
        offset = int(os.environ.get("DDSTORE_AFFINITY_OFFSET", "0"))
        if width <= 0 or not hasattr(os, "sched_setaffinity"):
            return
        with self._worker_ids_lock:
            wid = next(self._worker_ids)
        cores = sorted(os.sched_getaffinity(0))[offset + width * wid: offset + width * (wid + 1)]
        if cores:
            os.sched_setaffinity(0, cores)  # this thread only

    def _fetch(self, index):
        if self._auto_collation:
            batch = [self.dataset[i] for i in index]
        else:
            batch = self.dataset[index]
        if self.collate_fn is not None:
            batch = self.collate_fn(batch)
        if self.pin_memory:
            batch = torch.utils.data._utils.pin_memory.pin_memory(batch)
        return batch

    def __iter__(self):
        sampler_iter = iter(self._index_sampler)
        # torch's DataLoader iterator draws a base seed from the global RNG
        # here, every epoch; draw it too, so the training loop's later random
        # numbers (patch jitter, drop path, ...) match a run with torch's loader
        torch.empty((), dtype=torch.int64).random_(generator=self.generator)
        pending = collections.deque()

        def submit() -> bool:
            try:
                index = next(sampler_iter)
            except StopIteration:
                return False
            pending.append(self.executor.submit(self._fetch, index))
            return True

        for _ in range(self.ahead):
            if not submit():
                break
        try:
            while pending:
                batch = pending.popleft().result()
                submit()
                yield batch
        finally:
            for future in pending:  # an early exit drops batches not started
                future.cancel()

    def __del__(self):
        executor = getattr(self, "executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
