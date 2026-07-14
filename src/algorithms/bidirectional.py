from __future__ import annotations

import heapq
import time

import numpy as np

from ..landmarks import Landmarks
from ..network import RoadNetwork
from .base import SearchResult


def _bidirectional_core(
    net: RoadNetwork,
    src: int,
    dst: int,
    p: np.ndarray,
    label: str,
) -> SearchResult:
    """Bidirectional Dijkstra on edge weights reduced by potential array *p*.

    ``p`` must be a feasible potential (``|p[u] - p[v]| <= w(u,v)`` on every
    edge) so that reduced costs are non-negative.  ``p = 0`` gives plain
    bidirectional Dijkstra.
    """
    t0 = time.perf_counter()
    n = net.n

    if src == dst:
        return SearchResult(
            path=[src], cost=0.0, settled=1, generated=0, visited={src},
            elapsed_s=time.perf_counter() - t0, algorithm=label,
        )

    df = np.full(n, np.inf)   # reduced-cost distance from source (forward)
    db = np.full(n, np.inf)   # reduced-cost distance from target (backward)
    df[src] = 0.0
    db[dst] = 0.0
    closed_f = np.zeros(n, dtype=bool)
    closed_b = np.zeros(n, dtype=bool)
    parent_f: dict[int, int] = {}
    parent_b: dict[int, int] = {}

    pq_f: list[tuple[float, int]] = [(0.0, src)]
    pq_b: list[tuple[float, int]] = [(0.0, dst)]

    best = np.inf          # best reduced-cost meeting value found so far
    meet = -1
    settled = 0
    generated = 0
    visited: set[int] = set()
    order: list[int] = []

    while pq_f and pq_b:
        # Stop once the two frontiers can no longer improve the meeting cost.
        if pq_f[0][0] + pq_b[0][0] >= best:
            break

        # Expand whichever frontier is currently smaller (balances the search).
        if pq_f[0][0] <= pq_b[0][0]:
            d, u = heapq.heappop(pq_f)
            if closed_f[u] or d > df[u]:
                continue
            closed_f[u] = True
            settled += 1
            visited.add(u)
            order.append(u)
            for v, w in net.adj[u]:
                rw = w - p[u] + p[v]
                nd = d + rw
                if nd < df[v]:
                    df[v] = nd
                    parent_f[v] = u
                    generated += 1
                    heapq.heappush(pq_f, (nd, v))
                if db[v] < np.inf and df[v] + db[v] < best:
                    best = df[v] + db[v]
                    meet = v
        else:
            d, u = heapq.heappop(pq_b)
            if closed_b[u] or d > db[u]:
                continue
            closed_b[u] = True
            settled += 1
            visited.add(u)
            order.append(u)
            for v, w in net.radj[u]:          # reverse edges: v -> u exists in G
                rw = w - p[v] + p[u]           # reduced cost of the reverse step
                nd = d + rw
                if nd < db[v]:
                    db[v] = nd
                    parent_b[v] = u
                    generated += 1
                    heapq.heappush(pq_b, (nd, v))
                if df[v] < np.inf and df[v] + db[v] < best:
                    best = df[v] + db[v]
                    meet = v

    # Reconstruct s -> meet (forward) and meet -> t (backward).
    path: list[int] = []
    cost = np.inf
    if meet != -1:
        left = [meet]
        node = meet
        while node != src:
            node = parent_f[node]
            left.append(node)
        left.reverse()
        right = []
        node = meet
        while node != dst:
            node = parent_b[node]
            right.append(node)
        path = left + right
        # Exact length: sum original edge weights along the stitched route.
        cost = 0.0
        for a, b in zip(path, path[1:]):
            cost += min(w for v, w in net.adj[a] if v == b)

    return SearchResult(
        path=path,
        cost=float(cost),
        settled=settled,
        generated=generated,
        visited=visited,
        order=order,
        elapsed_s=time.perf_counter() - t0,
        algorithm=label,
    )


def bidirectional_dijkstra(net: RoadNetwork, src: int, dst: int) -> SearchResult:
    """Uninformed bidirectional Dijkstra (potential = 0)."""
    return _bidirectional_core(
        net, src, dst, p=np.zeros(net.n), label="Bi-Dijkstra"
    )


def _average_potential(landmarks: Landmarks, src: int, dst: int) -> np.ndarray:
    """p(v) = ½ (lb(v -> dst) - lb(src -> v)), vectorised over all nodes."""
    to_L = landmarks.to_L
    from_L = landmarks.from_L

    # lb(v -> dst): max over landmarks of the two directed triangle bounds.
    fwd = np.maximum(
        (to_L - to_L[:, dst : dst + 1]).max(axis=0),
        (from_L[:, dst : dst + 1] - from_L).max(axis=0),
    )
    np.maximum(fwd, 0.0, out=fwd)

    # lb(src -> v).
    bwd = np.maximum(
        (to_L[:, src : src + 1] - to_L).max(axis=0),
        (from_L - from_L[:, src : src + 1]).max(axis=0),
    )
    np.maximum(bwd, 0.0, out=bwd)

    return 0.5 * (fwd - bwd)


def bidirectional_alt(
    net: RoadNetwork,
    src: int,
    dst: int,
    landmarks: Landmarks,
) -> SearchResult:
    """Bidirectional A* guided by the ALT heuristic (average-potential form)."""
    p = _average_potential(landmarks, src, dst).astype(np.float64)
    return _bidirectional_core(
        net, src, dst, p=p, label=f"Bi-ALT[{landmarks.strategy},k={landmarks.k}]"
    )
