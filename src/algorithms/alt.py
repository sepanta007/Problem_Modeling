from __future__ import annotations

import time

import numpy as np

from ..landmarks import Landmarks
from ..network import RoadNetwork
from ..priority_queue import make_queue
from .base import SearchResult, reconstruct_path


def alt(
    net: RoadNetwork,
    src: int,
    dst: int,
    landmarks: Landmarks,
    num_active: int | None = None,
    queue: str = "heap",
    label: str | None = None,
    **queue_kwargs,
) -> SearchResult:
    """A* using the ALT heuristic built from *landmarks*."""
    t0 = time.perf_counter()
    n = net.n
    # Same A* loop as astar.py, but h is the landmark/triangle-inequality bound
    # (a much tighter admissible estimate than the crow-flies distance).
    h = landmarks.heuristic(dst, source=src, num_active=num_active)

    g = np.full(n, np.inf)
    closed = np.zeros(n, dtype=bool)
    came_from: dict[int, int] = {}

    g[src] = 0.0
    pq = make_queue(queue, **queue_kwargs)
    pq.push(h(src), src)

    settled = 0
    generated = 0
    visited: set[int] = set()
    order: list[int] = []

    while len(pq):
        _, u = pq.pop_min()
        if closed[u]:
            continue
        closed[u] = True
        settled += 1
        visited.add(u)
        order.append(u)
        if u == dst:
            break
        gu = g[u]
        for v, w in net.adj[u]:
            if closed[v]:
                continue
            ng = gu + w
            if ng < g[v]:
                g[v] = ng
                came_from[v] = u
                generated += 1
                pq.push(ng + h(v), v)

    path = reconstruct_path(came_from, src, dst) if closed[dst] else []
    if label is None:
        suffix = f",active={num_active}" if num_active else ""
        label = f"ALT[{landmarks.strategy},k={landmarks.k}{suffix}]"
    return SearchResult(
        path=path,
        cost=float(g[dst]),
        settled=settled,
        generated=generated,
        visited=visited,
        order=order,
        elapsed_s=time.perf_counter() - t0,
        algorithm=label,
        meta={"strategy": landmarks.strategy, "k": landmarks.k},
    )
