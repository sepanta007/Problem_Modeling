from __future__ import annotations

import heapq
import time

import numpy as np

from ..network import RoadNetwork
from .base import SearchResult

_EPS = 1e-6


def _witness(out_adj, contracted, source, target, avoid, limit, max_settle):
    """Shortest dist source->target avoiding *avoid*, capped at *limit* / effort.

    Returns the distance if a path no longer than *limit* is found within the
    settle budget, else ``inf`` (meaning: assume no witness -> add a shortcut,
    which is always safe for correctness).
    """
    dist = {source: 0.0}
    pq = [(0.0, source)]
    settled = 0
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, np.inf):
            continue
        if u == target:
            return d
        settled += 1
        if settled > max_settle or d > limit:
            break
        for w2, wt in out_adj[u].items():
            if w2 == avoid or contracted[w2]:
                continue
            nd = d + wt
            if nd <= limit + _EPS and nd < dist.get(w2, np.inf):
                dist[w2] = nd
                heapq.heappush(pq, (nd, w2))
    return dist.get(target, np.inf)


class ContractionHierarchy:
    """A preprocessed hierarchy supporting fast exact shortest-path queries."""

    def __init__(self, net: RoadNetwork):
        self.net = net
        self.rank = np.full(net.n, -1, dtype=np.int64)
        self.f_adj: list[list[tuple[int, float]]] = [[] for _ in range(net.n)]
        self.b_adj: list[list[tuple[int, float]]] = [[] for _ in range(net.n)]
        self.shortcut_mid: dict[tuple[int, int], int] = {}
        self.n_shortcuts = 0
        self.preprocess_s = 0.0

    # ---- preprocessing -------------------------------------------------

    def build(self, max_settle: int = 120) -> "ContractionHierarchy":
        net = self.net
        n = net.n
        # Mutable "remaining graph" as adjacency dicts (min weight per pair).
        out_adj: list[dict[int, float]] = [dict() for _ in range(n)]
        in_adj: list[dict[int, float]] = [dict() for _ in range(n)]
        for u in range(n):
            for v, w in net.adj[u]:
                if v not in out_adj[u] or w < out_adj[u][v]:
                    out_adj[u][v] = w
                    in_adj[v][u] = w
        # Augmented edge set (original + shortcuts) for the query graph.
        aug: dict[tuple[int, int], tuple[float, int]] = {}
        for u in range(n):
            for v, w in out_adj[u].items():
                aug[(u, v)] = (w, -1)

        contracted = np.zeros(n, dtype=bool)
        t0 = time.perf_counter()

        def shortcuts_needed(v: int, do_add: bool) -> int:
            added = 0
            ins = [u for u in in_adj[v] if not contracted[u]]
            outs = [w for w in out_adj[v] if not contracted[w]]
            for u in ins:
                for w in outs:
                    if u == w:
                        continue
                    P = in_adj[v][u] + out_adj[v][w]
                    if _witness(out_adj, contracted, u, w, v, P, max_settle) > P + _EPS:
                        added += 1
                        if do_add:
                            cur = out_adj[u].get(w, np.inf)
                            if P < cur:
                                out_adj[u][w] = P
                                in_adj[w][u] = P
                            a = aug.get((u, w))
                            if a is None or P < a[0]:
                                aug[(u, w)] = (P, v)
                                self.n_shortcuts += 1
            return added

        def priority(v: int) -> int:
            deg = len([u for u in in_adj[v] if not contracted[u]]) + \
                  len([w for w in out_adj[v] if not contracted[w]])
            return shortcuts_needed(v, do_add=False) - deg

        pq = [(priority(v), v) for v in range(n)]
        heapq.heapify(pq)
        order = 0
        while pq:
            pr, v = heapq.heappop(pq)
            if contracted[v]:
                continue
            # Lazy update: if v's priority got worse than the next best, defer it.
            npr = priority(v)
            if pq and npr > pq[0][0]:
                heapq.heappush(pq, (npr, v))
                continue
            # Contract v: materialise its shortcuts, assign its rank, unlink it.
            shortcuts_needed(v, do_add=True)
            self.rank[v] = order
            order += 1
            contracted[v] = True
            for u in list(in_adj[v]):
                out_adj[u].pop(v, None)
            for w in list(out_adj[v]):
                in_adj[w].pop(v, None)
            in_adj[v].clear()
            out_adj[v].clear()

        # Build the upward query graph from the augmented edges.
        rank = self.rank
        for (u, v), (w, mid) in aug.items():
            if rank[u] < rank[v]:
                self.f_adj[u].append((v, w))
            else:
                self.b_adj[v].append((u, w))
            if mid >= 0:
                self.shortcut_mid[(u, v)] = mid

        self.preprocess_s = time.perf_counter() - t0
        return self

    # ---- query ---------------------------------------------------------

    def query(self, src: int, dst: int) -> SearchResult:
        t0 = time.perf_counter()
        n = self.net.n
        INF = np.inf
        df = np.full(n, INF)
        db = np.full(n, INF)
        df[src] = 0.0
        db[dst] = 0.0
        parent_f: dict[int, int] = {}
        parent_b: dict[int, int] = {}
        pf: list[tuple[float, int]] = [(0.0, src)]
        pb: list[tuple[float, int]] = [(0.0, dst)]
        cf = np.zeros(n, dtype=bool)
        cb = np.zeros(n, dtype=bool)
        mu = INF
        meet = -1
        settled = 0
        visited: set[int] = set()
        order: list[int] = []

        while pf or pb:
            topf = pf[0][0] if pf else INF
            topb = pb[0][0] if pb else INF
            if min(topf, topb) >= mu:
                break
            if topf <= topb and pf:
                d, u = heapq.heappop(pf)
                if cf[u] or d > df[u]:
                    continue
                cf[u] = True
                settled += 1
                visited.add(u)
                order.append(u)
                if db[u] < INF and d + db[u] < mu:
                    mu = d + db[u]
                    meet = u
                for v, w in self.f_adj[u]:
                    nd = d + w
                    if nd < df[v]:
                        df[v] = nd
                        parent_f[v] = u
                        heapq.heappush(pf, (nd, v))
            elif pb:
                d, u = heapq.heappop(pb)
                if cb[u] or d > db[u]:
                    continue
                cb[u] = True
                settled += 1
                visited.add(u)
                order.append(u)
                if df[u] < INF and d + df[u] < mu:
                    mu = d + df[u]
                    meet = u
                for v, w in self.b_adj[u]:
                    nd = d + w
                    if nd < db[v]:
                        db[v] = nd
                        parent_b[v] = u
                        heapq.heappush(pb, (nd, v))

        path = self._unpack(src, dst, meet, parent_f, parent_b) if meet >= 0 else []
        return SearchResult(
            path=path,
            cost=float(mu),
            settled=settled,
            generated=0,
            visited=visited,
            order=order,
            elapsed_s=time.perf_counter() - t0,
            algorithm="CH",
            meta={"shortcuts": self.n_shortcuts, "meet_rank": int(self.rank[meet])
                  if meet >= 0 else -1},
        )

    # ---- shortcut unpacking (for the real node path) -------------------

    def _unpack_edge(self, u: int, v: int, out: list[int]) -> None:
        """Append the real intermediate nodes of edge u->v (recursively)."""
        mid = self.shortcut_mid.get((u, v), -1)
        if mid < 0:
            out.append(v)
        else:
            self._unpack_edge(u, mid, out)
            self._unpack_edge(mid, v, out)

    def _unpack(self, src, dst, meet, parent_f, parent_b) -> list[int]:
        # Forward half: follow parent_f from meet back to src (edge u->v = (parent,node)).
        fwd = [meet]
        node = meet
        while node != src:
            node = parent_f[node]
            fwd.append(node)
        fwd.reverse()                    # src .. meet
        # Backward half: follow parent_b from meet to dst (edge cur->parent_b[cur]).
        bwd = []
        node = meet
        while node != dst:
            node = parent_b[node]
            bwd.append(node)             # meet's successors down to dst
        packed = fwd + bwd               # src .. meet .. dst (shortcut edges inside)
        real = [packed[0]]
        for a, b in zip(packed, packed[1:]):
            self._unpack_edge(a, b, real)
        return real


def build_ch(net: RoadNetwork, max_settle: int = 120) -> ContractionHierarchy:
    """Preprocess *net* into a Contraction Hierarchy (one-off, reusable)."""
    return ContractionHierarchy(net).build(max_settle=max_settle)
