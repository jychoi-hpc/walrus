"""ThreadDataLoader (walrus.data.thread_loader): same batches in the same order
as torch's DataLoader, at most `ahead` batches fetched ahead of the consumer,
and dataset errors raised in the training loop."""

import threading
import time

import pytest
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from walrus.data.thread_loader import ThreadDataLoader


class Squares(Dataset):
    """Item i is {'x': tensor(i*i)}; a list index returns a whole batch, as
    Walrus's datasets do with batch_size=None. Counts fetched items."""

    def __init__(self, n=20, fail_at=None, delay=0.0):
        self.n, self.fail_at, self.delay = n, fail_at, delay
        self.fetched = 0
        self.lock = threading.Lock()

    def __len__(self):
        return self.n

    def _one(self, i):
        if i == self.fail_at:
            raise ValueError(f"bad item {i}")
        time.sleep(self.delay)
        with self.lock:
            self.fetched += 1
        return {"x": torch.tensor(i * i)}

    def __getitem__(self, i):
        if isinstance(i, list):
            items = [self._one(j) for j in i]
            return {"x": torch.stack([it["x"] for it in items])}
        return self._one(i)


class Batches(Sampler):
    """Yields lists of 3 shuffled indices (like Walrus's batched sampler)."""

    def __init__(self, n, seed=0):
        self.order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()

    def __iter__(self):
        return iter(self.order[i:i + 3] for i in range(0, len(self.order), 3))

    def __len__(self):
        return (len(self.order) + 2) // 3


def as_lists(loader):
    return [b["x"].tolist() for b in loader]


@pytest.mark.parametrize("threads", [1, 3])
def test_batch_sampler_without_batch_size_matches_dataloader(threads):
    ds = Squares()
    ref = as_lists(DataLoader(ds, batch_size=None, sampler=Batches(len(ds)), collate_fn=None))
    out = as_lists(ThreadDataLoader(ds, num_workers=threads, batch_size=None,
                                    sampler=Batches(len(ds)), collate_fn=None))
    assert out == ref and len(out) == 7


def test_auto_collation_matches_dataloader():
    ds = Squares()
    ref = as_lists(DataLoader(ds, batch_size=4, shuffle=False))
    out = as_lists(ThreadDataLoader(ds, num_workers=2, batch_size=4, shuffle=False))
    assert out == ref


def test_fetches_at_most_ahead_batches():
    ds = Squares(n=30, delay=0.001)
    loader = ThreadDataLoader(ds, num_workers=2, prefetch=3, batch_size=None,
                              sampler=Batches(len(ds)), collate_fn=None)
    it = iter(loader)
    next(it)
    time.sleep(0.3)  # threads would load the whole epoch without a bound
    # 1 consumed + 3 ahead (one submitted after the consumed batch), 3 items each
    assert ds.fetched <= (1 + 3) * 3
    assert len(list(it)) == len(loader) - 1


def test_dataset_error_reaches_consumer():
    ds = Squares(fail_at=5)
    loader = ThreadDataLoader(ds, num_workers=2, batch_size=None,
                              sampler=Batches(len(ds)), collate_fn=None)
    with pytest.raises(ValueError, match="bad item 5"):
        list(loader)


def test_len_and_reiteration():
    ds = Squares()
    loader = ThreadDataLoader(ds, num_workers=2, batch_size=None,
                              sampler=Batches(len(ds)), collate_fn=None)
    assert len(loader) == 7
    assert as_lists(loader) == as_lists(loader)


def test_global_rng_stream_matches_dataloader():
    """The training loop's random draws after each epoch starts must not
    depend on which loader is used."""
    ds = Squares()

    def draws(loader):
        torch.manual_seed(0)
        out = []
        for _ in range(2):
            for _batch in loader:
                out.append(torch.rand(1).item())
        return out

    ref = draws(DataLoader(ds, batch_size=None, sampler=Batches(len(ds)), collate_fn=None))
    out = draws(ThreadDataLoader(ds, num_workers=2, batch_size=None,
                                 sampler=Batches(len(ds)), collate_fn=None))
    assert out == ref
