from __future__ import annotations

import time
from collections import deque

import numpy as np

from ..network import RoadNetwork
from .base import SearchResult, reconstruct_path


def bfs(net: RoadNetwork, src: int, dst: int) -> SearchResult:
    """Breadth-first search: fewest-hops path (optimal only under unit weights)."""
    t0 = time.perf_counter()
    n = net.n
    seen = np.zeros(n, dtype=bool)
    came_from: dict[int, int] = {}
    q: deque[int] = deque([src])
    seen[src] = True
    order: list[int] = []
    settled = 0

    while q:
        u = q.popleft()
        settled += 1
        order.append(u)
        if u == dst:
            break
        for v, _ in net.adj[u]:
            if not seen[v]:
                seen[v] = True
                came_from[v] = u
                q.append(v)

    path = reconstruct_path(came_from, src, dst) if (seen[dst]) else []
    cost = 0.0
    for a, b in zip(path, path[1:]):
        cost += min(w for v, w in net.adj[a] if v == b)
    return SearchResult(
        path=path,
        cost=cost if path else float("inf"),
        settled=settled,
        generated=int(seen.sum()),
        visited=set(order),
        order=order,
        elapsed_s=time.perf_counter() - t0,
        algorithm="BFS",
        meta={"objective": "fewest hops (not metric-optimal)"},
    )


def dfs(net: RoadNetwork, src: int, dst: int) -> SearchResult:
    """Depth-first search: returns *some* path (usually far from shortest)."""
    t0 = time.perf_counter()
    n = net.n
    seen = np.zeros(n, dtype=bool)
    came_from: dict[int, int] = {}
    stack: list[int] = [src]
    seen[src] = True
    order: list[int] = []
    settled = 0
    found = False

    while stack:
        u = stack.pop()
        settled += 1
        order.append(u)
        if u == dst:
            found = True
            break
        for v, _ in net.adj[u]:
            if not seen[v]:
                seen[v] = True
                came_from[v] = u
                stack.append(v)

    path = reconstruct_path(came_from, src, dst) if found else []
    cost = 0.0
    for a, b in zip(path, path[1:]):
        cost += min(w for v, w in net.adj[a] if v == b)
    return SearchResult(
        path=path,
        cost=cost if path else float("inf"),
        settled=settled,
        generated=int(seen.sum()),
        visited=set(order),
        order=order,
        elapsed_s=time.perf_counter() - t0,
        algorithm="DFS",
        meta={"objective": "any path (not optimal)"},
    )
