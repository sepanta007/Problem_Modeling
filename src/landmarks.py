from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .network import RoadNetwork


# ---------------------------------------------------------------------------
# Full single-source shortest paths (used only during preprocessing)
# ---------------------------------------------------------------------------


def _sssp(adj: list[list[tuple[int, float]]], source: int, n: int) -> np.ndarray:
    """Dijkstra from *source* over adjacency *adj*; returns distances to all nodes."""
    dist = np.full(n, np.inf, dtype=np.float64)
    dist[source] = 0.0
    pq: list[tuple[float, int]] = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def _sssp_tree(
    adj: list[list[tuple[int, float]]], source: int, n: int
) -> tuple[np.ndarray, np.ndarray]:
    """Dijkstra returning distances AND shortest-path-tree parents (-1 = none)."""
    dist = np.full(n, np.inf, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int64)
    dist[source] = 0.0
    pq: list[tuple[float, int]] = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, parent


# ---------------------------------------------------------------------------
# Landmark selection
# ---------------------------------------------------------------------------


def _select_random(net: RoadNetwork, k: int, rng: np.random.Generator) -> list[int]:
    return rng.choice(net.n, size=k, replace=False).tolist()


def _select_planar(net: RoadNetwork, k: int) -> list[int]:
    """Most extreme node in each of k angular sectors around the centroid."""
    cx = float(np.mean(net.lon))
    cy = float(np.mean(net.lat))
    # Angle and (squared) distance of every node relative to the centre.
    ang = np.arctan2(net.lat - cy, net.lon - cx)  # (-pi, pi]
    ang = (ang + 2 * np.pi) % (2 * np.pi)
    r2 = (net.lat - cy) ** 2 + (net.lon - cx) ** 2
    chosen: list[int] = []
    edges = np.linspace(0, 2 * np.pi, k + 1)
    for s in range(k):
        mask = (ang >= edges[s]) & (ang < edges[s + 1])
        if not mask.any():
            continue
        idxs = np.nonzero(mask)[0]
        chosen.append(int(idxs[np.argmax(r2[idxs])]))
    # If some sectors were empty, top up with the globally farthest unused nodes.
    if len(chosen) < k:
        order = np.argsort(-r2)
        for node in order:
            node = int(node)
            if node not in chosen:
                chosen.append(node)
            if len(chosen) == k:
                break
    return chosen[:k]


def _select_farthest(
    net: RoadNetwork, k: int, rng: np.random.Generator
) -> tuple[list[int], dict[int, np.ndarray]]:
    """Greedy k-centre on graph distance ("avoid"/farthest-point selection).

    Returns the landmark ids *and* the forward-distance array already computed
    for each of them, so preprocessing need not repeat that work.
    """
    # Seed: farthest node (by graph distance) from a random start.
    start = int(rng.integers(net.n))
    d0 = _sssp(net.adj, start, net.n)
    d0[~np.isfinite(d0)] = -1.0
    first = int(np.argmax(d0))

    landmarks = [first]
    from_cache: dict[int, np.ndarray] = {}
    # min distance from every node to the closest chosen landmark so far
    min_dist = _sssp(net.adj, first, net.n)
    from_cache[first] = min_dist.copy()

    while len(landmarks) < k:
        finite = np.where(np.isfinite(min_dist), min_dist, -1.0)
        nxt = int(np.argmax(finite))
        if nxt in landmarks:  # degenerate; fall back to any unused node
            remaining = [i for i in range(net.n) if i not in landmarks]
            if not remaining:
                break
            nxt = remaining[0]
        landmarks.append(nxt)
        d = _sssp(net.adj, nxt, net.n)
        from_cache[nxt] = d.copy()
        min_dist = np.minimum(min_dist, d)
    return landmarks, from_cache


def _select_avoid(net: RoadNetwork, k: int, rng: np.random.Generator) -> list[int]:
    """Goldberg-Werneck "avoid": add each landmark where the current set is
    weakest.

    For a random root ``r`` we grow its shortest-path tree and give every vertex
    a *weight* = how much farther it is than the best bound the current
    landmarks already give for the pair ``(r, v)``.  The *size* of a subtree is
    the total weight it contains (zeroed if it already holds a landmark).
    Walking from ``r`` towards the heaviest subtree lands on the vertex the
    current landmarks cover worst -- that becomes the next landmark.
    """
    n = net.n
    landmarks: list[int] = []
    from_rows: list[np.ndarray] = []   # d(L, .)  for chosen landmarks
    to_rows: list[np.ndarray] = []     # d(., L)  for chosen landmarks

    def bound(r: int, v: np.ndarray | int):
        """Best current landmark lower bound for pairs (r, v)."""
        if not landmarks:
            return 0.0
        best = np.zeros(n)
        for fr, to in zip(from_rows, to_rows):
            # d(r,v) >= |d(r,L) - d(v,L)|  and  |d(L,r) - d(L,v)|
            best = np.maximum(best, np.abs(to[r] - to))
            best = np.maximum(best, np.abs(fr[r] - fr))
        return best

    for _ in range(k):
        root = int(rng.integers(n))
        dist, parent = _sssp_tree(net.adj, root, n)
        reachable = np.isfinite(dist)
        lb = bound(root, None)
        weight = np.where(reachable, dist - (lb if np.ndim(lb) else 0.0), 0.0)
        np.maximum(weight, 0.0, out=weight)

        size = weight.copy()
        has_lm = np.zeros(n, dtype=bool)
        for L in landmarks:
            has_lm[L] = True
        # Process vertices farthest-first so children precede their parent.
        order = [int(v) for v in np.argsort(-dist) if reachable[v]]
        for v in order:
            if has_lm[v]:
                size[v] = 0.0
            p = int(parent[v])
            if p >= 0:
                size[p] += size[v]
                if has_lm[v]:
                    has_lm[p] = True

        # Descend from the root towards the heaviest child until it dies out.
        children: dict[int, list[int]] = {}
        for v in order:
            p = int(parent[v])
            if p >= 0:
                children.setdefault(p, []).append(v)
        cur = root
        while True:
            ch = children.get(cur, [])
            if not ch:
                break
            best_c = max(ch, key=lambda c: size[c])
            if size[best_c] <= 0.0:
                break
            cur = best_c
        if cur == root or cur in landmarks:
            # Degenerate (everything already covered): fall back to farthest.
            cand = np.where(reachable, dist, -1.0)
            for L in landmarks:
                cand[L] = -1.0
            cur = int(np.argmax(cand))
        landmarks.append(cur)
        from_rows.append(_sssp(net.adj, cur, n))
        to_rows.append(_sssp(net.radj, cur, n))

    return landmarks


# ---------------------------------------------------------------------------
# The Landmarks object: tables + heuristic
# ---------------------------------------------------------------------------


@dataclass
class Landmarks:
    net: RoadNetwork
    nodes: list[int]                 # landmark node ids
    from_L: np.ndarray               # (k, n) float64: d(L_i, v)
    to_L: np.ndarray                 # (k, n) float64: d(v, L_i)
    strategy: str
    preprocess_s: float = 0.0

    @property
    def k(self) -> int:
        return len(self.nodes)

    def lower_bound(self, u: int, t: int) -> float:
        """The ALT admissible heuristic h(u -> t), maximised over all landmarks."""
        t1 = self.to_L[:, u] - self.to_L[:, t]
        t2 = self.from_L[:, t] - self.from_L[:, u]
        return float(max(0.0, t1.max(), t2.max()))

    def heuristic(
        self, target: int, source: int | None = None, num_active: int | None = None
    ) -> Callable[[int], float]:
        """Build the per-query heuristic ``h(u)`` for a fixed *target*.

        If *num_active* is given (and *source* is provided), only the
        ``num_active`` landmarks giving the largest bound for the (source,
        target) pair are used -- the *ALTBestp* optimisation.  This keeps the
        per-node work small when there are many landmarks.
        """
        rows = np.arange(self.k)
        if num_active is not None and source is not None and num_active < self.k:
            b1 = self.to_L[:, source] - self.to_L[:, target]
            b2 = self.from_L[:, target] - self.from_L[:, source]
            bound = np.maximum(b1, b2)
            rows = np.argsort(-bound)[:num_active]

        to_sub = self.to_L[rows]          # (p, n)
        from_sub = self.from_L[rows]      # (p, n)
        to_t = to_sub[:, target]          # (p,)
        from_t = from_sub[:, target]      # (p,)

        def h(u: int) -> float:
            # Two directed triangle bounds, maximised over the active landmarks:
            #   d(u,t) >= d(u,L) - d(t,L)   and   d(u,t) >= d(L,t) - d(L,u)
            t1 = to_sub[:, u] - to_t
            t2 = from_t - from_sub[:, u]
            v = t1.max()
            v2 = t2.max()
            if v2 > v:
                v = v2
            return v if v > 0.0 else 0.0

        return h


def build_landmarks(
    net: RoadNetwork,
    k: int = 16,
    strategy: str = "farthest",
    seed: int = 0,
) -> Landmarks:
    """Select *k* landmarks and pre-compute the two distance tables.

    This is the expensive, *one-off* preprocessing step: ``2k`` full Dijkstra
    runs (k forward for ``from_L``, k reverse for ``to_L``).  On a ~10k-node
    city it takes a couple of seconds; it is then reused for every query.
    """
    import time

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    from_cache: dict[int, np.ndarray] = {}
    if strategy == "random":
        nodes = _select_random(net, k, rng)
    elif strategy == "planar":
        nodes = _select_planar(net, k)
    elif strategy == "farthest":
        nodes, from_cache = _select_farthest(net, k, rng)
    elif strategy == "avoid":
        nodes = _select_avoid(net, k, rng)
    else:
        raise ValueError(f"unknown landmark strategy {strategy!r}")

    k = len(nodes)
    # float64 keeps the triangle-inequality bound *exactly* admissible: the bound
    # is a difference of Dijkstra distances and can equal the true distance, so
    # float32 rounding could nudge it a hair over the real value.
    from_L = np.empty((k, net.n), dtype=np.float64)
    to_L = np.empty((k, net.n), dtype=np.float64)
    for i, L in enumerate(nodes):
        fwd = from_cache.get(L)
        if fwd is None:
            fwd = _sssp(net.adj, L, net.n)
        from_L[i] = fwd
        to_L[i] = _sssp(net.radj, L, net.n)

    return Landmarks(
        net=net,
        nodes=list(nodes),
        from_L=from_L,
        to_L=to_L,
        strategy=strategy,
        preprocess_s=time.perf_counter() - t0,
    )
