from __future__ import annotations

import time
from typing import Callable

import numpy as np

from ..network import RoadNetwork
from ..priority_queue import make_queue
from .base import SearchResult, reconstruct_path

# A heuristic *factory*: given (net, target) it returns h(u) -> estimated metres.
HeuristicFactory = Callable[[RoadNetwork, int], Callable[[int], float]]


def astar(
    net: RoadNetwork,
    src: int,
    dst: int,
    heuristic: HeuristicFactory,
    queue: str = "heap",
    label: str = "A*",
    **queue_kwargs,
) -> SearchResult:
    """A* from *src* to *dst* using the heuristic built by *heuristic*."""
    t0 = time.perf_counter()
    n = net.n
    h = heuristic(net, dst)

    g = np.full(n, np.inf)
    closed = np.zeros(n, dtype=bool)
    came_from: dict[int, int] = {}

    g[src] = 0.0
    pq = make_queue(queue, **queue_kwargs)
    pq.push(h(src), src)                  # priority is f = g + h (here g(src) = 0)

    settled = 0
    generated = 0
    visited: set[int] = set()
    order: list[int] = []

    while len(pq):
        _, u = pq.pop_min()
        if closed[u]:
            continue                      # stale entry left by a lazy update
        closed[u] = True                  # with a consistent h, g[u] is final here
        settled += 1
        visited.add(u)
        order.append(u)
        if u == dst:
            break                         # target reached with its optimal cost
        gu = g[u]
        for v, w in net.adj[u]:
            if closed[v]:
                continue
            ng = gu + w                   # cost from source to v through u
            if ng < g[v]:
                g[v] = ng
                came_from[v] = u
                generated += 1
                pq.push(ng + h(v), v)     # ordered by f = g + h, so the search aims at dst

    path = reconstruct_path(came_from, src, dst) if closed[dst] else []
    return SearchResult(
        path=path,
        cost=float(g[dst]),
        settled=settled,
        generated=generated,
        visited=visited,
        order=order,
        elapsed_s=time.perf_counter() - t0,
        algorithm=label,
    )
