from __future__ import annotations

import time
from typing import Callable

import numpy as np

from ..network import RoadNetwork
from ..priority_queue import make_queue
from .base import SearchResult, reconstruct_path

HeuristicFactory = Callable[[RoadNetwork, int], Callable[[int], float]]


def greedy_best_first(
    net: RoadNetwork,
    src: int,
    dst: int,
    heuristic: HeuristicFactory,
    label: str = "Greedy",
) -> SearchResult:
    t0 = time.perf_counter()
    n = net.n
    h = heuristic(net, dst)

    g = np.full(n, np.inf)          # tracked only to report the path length
    closed = np.zeros(n, dtype=bool)
    came_from: dict[int, int] = {}

    g[src] = 0.0
    discovered = np.zeros(n, dtype=bool)
    discovered[src] = True
    pq = make_queue("heap")
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
            # Keep the smallest g seen so the *reported* path length is truthful,
            # even though the search order is driven purely by h(v).
            if gu + w < g[v]:
                g[v] = gu + w
                came_from[v] = u
            # Each node enters the open list once, when first discovered.
            if not discovered[v]:
                discovered[v] = True
                generated += 1
                pq.push(h(v), v)

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
