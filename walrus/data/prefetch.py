"""Run a DataLoader in a background thread so the next batch is fetched while
the training step runs. Used with DDStore, whose reads cannot run in forked
loader workers but release the GIL."""

import queue
import threading

_END = object()


class _Error:
    def __init__(self, exc: BaseException):
        self.exc = exc


class ThreadPrefetchLoader:
    """Iterates `loader` (num_workers=0) in a thread, `depth` batches ahead.
    Other attributes (dataset, sampler, ...) are the loader's."""

    def __init__(self, loader, depth: int = 1):
        self.loader = loader
        self.depth = max(1, int(depth))

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        q: queue.Queue = queue.Queue(maxsize=self.depth)
        stop = threading.Event()

        def put(item) -> bool:
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    pass
            return False

        def work():
            try:
                for batch in self.loader:
                    if not put(batch):
                        return
            except BaseException as e:  # noqa: BLE001 - re-raised in the consumer
                put(_Error(e))
            put(_END)

        thread = threading.Thread(target=work, name="prefetch", daemon=True)
        thread.start()
        try:
            while True:
                item = q.get()
                if item is _END:
                    break
                if isinstance(item, _Error):
                    raise item.exc
                yield item
        finally:
            stop.set()  # an early exit unblocks the thread
            thread.join()
