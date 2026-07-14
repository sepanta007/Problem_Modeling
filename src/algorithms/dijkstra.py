from __future__ import annotations

import time

import numpy as np

from ..network import RoadNetwork
from ..priority_queue import make_queue
from .base import SearchResult, reconstruct_path


def dijkstra(
    net: RoadNetwork,
    src: int,
    dst: int,
    queue: str = "heap",
    **queue_kwargs,
) -> SearchResult:
    """Shortest path from *src* to *dst* by Dijkstra.

    Parameters
    ----------
    queue:
        ``"heap"`` (binary heap) or ``"bucket"`` (Dial array-of-stacks).  Exposed
        so the benchmark can measure the data-structure speed-up from the paper.
    """
    t0 = time.perf_counter()
    n = net.n
    dist = np.full(n, np.inf)
    closed = np.zeros(n, dtype=bool)
    came_from: dict[int, int] = {}

    dist[src] = 0.0
    pq = make_queue(queue, **queue_kwargs)
    pq.push(0.0, src)

    settled = 0
    generated = 0
    visited: set[int] = set()
    order: list[int] = []

    while len(pq):
        _, u = pq.pop_min()
        if closed[u]:
            continue                     # stale entry: u was already settled via a shorter key
        closed[u] = True                 # u is settled -> dist[u] is now final
        settled += 1
        visited.add(u)
        order.append(u)
        if u == dst:
            break                        # early exit: the target's distance is fixed
        du = dist[u]
        for v, w in net.adj[u]:
            if closed[v]:
                continue
            nd = du + w                  # relax the edge u -> v
            if nd < dist[v]:             # found a shorter way to reach v
                dist[v] = nd
                came_from[v] = u
                generated += 1
                pq.push(nd, v)           # lazy update: push a new entry, skip the old later

    path = reconstruct_path(came_from, src, dst) if closed[dst] else []
    return SearchResult(
        path=path,
        cost=float(dist[dst]),
        settled=settled,
        generated=generated,
        visited=visited,
        order=order,
        elapsed_s=time.perf_counter() - t0,
        algorithm=f"Dijkstra[{queue}]",
    )
