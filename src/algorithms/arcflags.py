from __future__ import annotations

import heapq
import time

import numpy as np

from ..network import RoadNetwork
from .base import SearchResult, reconstruct_path


def _partition(net: RoadNetwork, regions: int, seed: int) -> np.ndarray:
    """Assign each node to one of *regions* clusters by geographic k-means."""
    from sklearn.cluster import KMeans

    xy = np.column_stack([net.lon, net.lat])
    km = KMeans(n_clusters=regions, random_state=seed, n_init=4)
    return km.fit_predict(xy).astype(np.int32)


class ArcFlags:
    def __init__(self, net: RoadNetwork, regions: int = 16):
        self.net = net
        self.regions = regions
        self.region = np.zeros(net.n, dtype=np.int32)
        # flags[u][i] = R-bit mask for arc net.adj[u][i]; bit r set => usable for
        # reaching region r.
        self.flags: list[np.ndarray] = []
        self.preprocess_s = 0.0

    def build(self, seed: int = 0) -> "ArcFlags":
        net = self.net
        n = net.n
        t0 = time.perf_counter()
        self.region = _partition(net, self.regions, seed)
        region = self.region

        # Index arcs per node: map head node -> list of arc indices (a node may
        # have several parallel arcs to the same neighbour; flag them all).
        arc_index = [dict() for _ in range(n)]
        self.flags = [np.zeros(len(net.adj[u]), dtype=np.int64) for u in range(n)]
        for u in range(n):
            for i, (v, _) in enumerate(net.adj[u]):
                arc_index[u].setdefault(v, []).append(i)

        # Intra-region arcs are always usable for their own region.
        # NB: cast to Python int -- `1 << np.int32(r)` overflows int32 for r>=31.
        for u in range(n):
            ru = int(region[u])
            for i, (v, _) in enumerate(net.adj[u]):
                if int(region[v]) == ru:
                    self.flags[u][i] |= (1 << ru)

        # Boundary nodes of each region: a node v in r with an incoming arc from
        # outside r (i.e. some u->v with region[u] != r).
        boundary = {r: set() for r in range(self.regions)}
        for u in range(n):
            for v, _ in net.adj[u]:
                if region[u] != region[v]:
                    boundary[region[v]].add(v)

        # For each boundary node b of region r, grow the shortest-path tree TO b
        # on the reversed graph and flag the corresponding forward arcs for r.
        for r in range(self.regions):
            bit = 1 << r
            for b in boundary[r]:
                self._flag_tree_to(b, r, bit, arc_index)

        self.preprocess_s = time.perf_counter() - t0
        return self

    def _flag_tree_to(self, b, r, bit, arc_index) -> None:
        """Reverse Dijkstra from *b*: flag forward arc (u -> next) for region r,
        where `next` is the successor on u's shortest path to b."""
        net = self.net
        dist = {b: 0.0}
        pq = [(0.0, b)]
        while pq:
            d, x = heapq.heappop(pq)
            if d > dist.get(x, np.inf):
                continue
            for u, w in net.radj[x]:          # forward arc u -> x exists
                nd = d + w
                if nd < dist.get(u, np.inf):
                    dist[u] = nd
                    # forward arc(s) (u -> x) are tree arcs heading toward b
                    for i in arc_index[u][x]:
                        self.flags[u][i] |= bit
                    heapq.heappush(pq, (nd, u))

    def query(self, src: int, dst: int) -> SearchResult:
        t0 = time.perf_counter()
        n = self.net.n
        r = int(self.region[dst])
        bit = 1 << r
        g = np.full(n, np.inf)
        closed = np.zeros(n, dtype=bool)
        came_from: dict[int, int] = {}
        g[src] = 0.0
        pq = [(0.0, src)]
        settled = 0
        visited: set[int] = set()
        order: list[int] = []

        while pq:
            d, u = heapq.heappop(pq)
            if closed[u] or d > g[u]:
                continue
            closed[u] = True
            settled += 1
            visited.add(u)
            order.append(u)
            if u == dst:
                break
            flags_u = self.flags[u]
            for i, (v, w) in enumerate(self.net.adj[u]):
                if not (flags_u[i] & bit):      # arc cannot reach dst's region
                    continue
                nd = d + w
                if nd < g[v]:
                    g[v] = nd
                    came_from[v] = u
                    heapq.heappush(pq, (nd, v))

        path = reconstruct_path(came_from, src, dst) if closed[dst] else []
        return SearchResult(
            path=path,
            cost=float(g[dst]),
            settled=settled,
            generated=0,
            visited=visited,
            order=order,
            elapsed_s=time.perf_counter() - t0,
            algorithm=f"ArcFlags[R={self.regions}]",
            meta={"regions": self.regions},
        )


def build_arcflags(net: RoadNetwork, regions: int = 16, seed: int = 0) -> ArcFlags:
    """Preprocess *net* into an Arc-Flags index (one-off, reusable)."""
    return ArcFlags(net, regions=regions).build(seed=seed)
