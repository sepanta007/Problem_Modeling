from __future__ import annotations

import heapq
import itertools
from typing import Any


class HeapQueue:
    """Binary-heap priority queue with lazy deletion of stale entries."""

    def __init__(self) -> None:
        self._heap: list[tuple[float, int, Any]] = []
        self._counter = itertools.count()  # tie-breaker -> FIFO, avoids comparing items
        self._live = 0

    def push(self, key: float, item: Any) -> None:
        heapq.heappush(self._heap, (key, next(self._counter), item))
        self._live += 1

    def pop_min(self) -> tuple[float, Any]:
        key, _, item = heapq.heappop(self._heap)
        self._live -= 1
        return key, item

    def __len__(self) -> int:
        return len(self._heap)


class BucketQueue:
    """Dial's bucket queue -- an array of stacks indexed by discretised key.

    Parameters
    ----------
    width:
        Bucket width in the same unit as the keys (metres here).  A node whose
        key is ``k`` lands in bucket ``int(k / width)``.  Smaller widths give a
        finer (closer to exact) ordering; larger widths are faster but expand
        nodes in a slightly looser order.  The returned *path cost* is unaffected
        as long as the search is otherwise correct; only the exploration order
        (and hence the node count) can change marginally.

    Notes
    -----
    The cursor ``_cur`` only ever moves forward, exploiting the fact that in
    Dijkstra / consistent-heuristic A* the minimum key is non-decreasing.  This
    is exactly the monotonicity Cazenave relies on for the "array of stacks".
    """

    def __init__(self, width: float = 20.0) -> None:
        self.width = float(width)
        self._buckets: dict[int, list[Any]] = {}
        self._cur = 0
        self._size = 0
        self._max_bucket = -1

    def push(self, key: float, item: Any) -> None:
        b = int(key / self.width)
        if b < self._cur:
            # Should not happen with a consistent heuristic, but stay safe.
            b = self._cur
        self._buckets.setdefault(b, []).append(item)
        self._size += 1
        if b > self._max_bucket:
            self._max_bucket = b

    def pop_min(self) -> tuple[float, Any]:
        while self._cur <= self._max_bucket:
            bucket = self._buckets.get(self._cur)
            if bucket:
                item = bucket.pop()  # LIFO within a bucket -> a "stack"
                self._size -= 1
                return self._cur * self.width, item
            self._cur += 1
        raise IndexError("pop_min from empty BucketQueue")

    def __len__(self) -> int:
        return self._size


def make_queue(kind: str = "heap", **kwargs) -> HeapQueue | BucketQueue:
    """Factory so algorithms can switch open-list implementation by name."""
    if kind == "heap":
        return HeapQueue()
    if kind == "bucket":
        return BucketQueue(**kwargs)
    raise ValueError(f"unknown queue kind {kind!r}")
