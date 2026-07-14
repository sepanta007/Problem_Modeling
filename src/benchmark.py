from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .algorithms import (
    alt,
    astar,
    bfs,
    bidirectional_alt,
    bidirectional_dijkstra,
    dfs,
    dijkstra,
    greedy_best_first,
)
from .algorithms.arcflags import ArcFlags
from .algorithms.base import SearchResult
from .algorithms.contraction import ContractionHierarchy
from .heuristics import haversine_heuristic
from .landmarks import Landmarks
from .network import RoadNetwork

# An algorithm runner takes (net, src, dst) and returns a SearchResult.
Runner = Callable[[RoadNetwork, int, int], SearchResult]


@dataclass
class AlgoStats:
    name: str
    settled: list[int] = field(default_factory=list)
    generated: list[int] = field(default_factory=list)
    times: list[float] = field(default_factory=list)
    costs: list[float] = field(default_factory=list)
    optimal: bool = True          # matches Dijkstra on every query
    max_rel_error: float = 0.0
    is_optimal_algo: bool = True  # whether it is *supposed* to be optimal

    @property
    def mean_settled(self) -> float:
        return float(np.mean(self.settled)) if self.settled else float("nan")

    @property
    def mean_time_ms(self) -> float:
        return float(np.mean(self.times)) * 1000 if self.times else float("nan")

    @property
    def mean_cost_km(self) -> float:
        finite = [c for c in self.costs if np.isfinite(c)]
        return float(np.mean(finite)) / 1000 if finite else float("nan")


@dataclass
class BenchmarkReport:
    stats: dict[str, AlgoStats]
    n_queries: int
    net_name: str
    net_nodes: int

    def to_table(self) -> str:
        base = self.stats.get("Dijkstra")
        base_settled = base.mean_settled if base else float("nan")
        base_time = base.mean_time_ms if base else float("nan")

        header = (
            f"\nBenchmark on {self.net_name!r} "
            f"({self.net_nodes:,} nodes, {self.n_queries} random queries)\n"
        )
        cols = (
            f"{'algorithm':<28}{'mean settled':>13}{'vs Dij':>9}"
            f"{'mean ms':>10}{'speedup':>9}{'mean km':>9}{'optimal':>9}"
        )
        line = "-" * len(cols)
        rows = [header, cols, line]
        for name, s in self.stats.items():
            set_ratio = base_settled / s.mean_settled if s.mean_settled else float("nan")
            speedup = base_time / s.mean_time_ms if s.mean_time_ms else float("nan")
            opt = "yes" if s.optimal else f"NO({s.max_rel_error*100:.1f}%)"
            if not s.is_optimal_algo:
                opt = f"~({s.max_rel_error*100:.1f}%)"
            rows.append(
                f"{name:<28}{s.mean_settled:>13,.0f}{set_ratio:>8.1f}x"
                f"{s.mean_time_ms:>10.2f}{speedup:>8.1f}x"
                f"{s.mean_cost_km:>9.2f}{opt:>9}"
            )
        return "\n".join(rows)


def default_suite(
    net: RoadNetwork,
    landmarks: Landmarks,
    ch: ContractionHierarchy | None = None,
    arcflags: ArcFlags | None = None,
    include_bucket: bool = True,
) -> dict[str, Runner]:
    """The standard line-up: uninformed baselines, the course algorithms, and
    the beyond-course ones (bidirectional, and Contraction Hierarchies if *ch*
    is provided).

    Non-optimal algorithms (BFS, DFS, Greedy) are flagged via :data:`NON_OPTIMAL`.
    """
    suite: dict[str, Runner] = {
        "BFS": lambda n, s, t: bfs(n, s, t),
        "DFS": lambda n, s, t: dfs(n, s, t),
        "Dijkstra": lambda n, s, t: dijkstra(n, s, t),
        "Greedy(haversine)": lambda n, s, t: greedy_best_first(
            n, s, t, haversine_heuristic
        ),
        "A*(haversine)": lambda n, s, t: astar(n, s, t, haversine_heuristic, label="A*"),
        "ALT": lambda n, s, t: alt(n, s, t, landmarks),
        "ALT(active=4)": lambda n, s, t: alt(n, s, t, landmarks, num_active=4),
        "Bi-Dijkstra": lambda n, s, t: bidirectional_dijkstra(n, s, t),
        "Bi-ALT": lambda n, s, t: bidirectional_alt(n, s, t, landmarks),
    }
    if arcflags is not None:
        suite["Arc Flags"] = lambda n, s, t: arcflags.query(s, t)
    if ch is not None:
        suite["CH"] = lambda n, s, t: ch.query(s, t)
    if include_bucket:
        # Width = min edge weight  ->  Dial's bucket queue is *exact* for
        # Dijkstra (a node in the current bucket can never be improved later).
        # We deliberately do NOT bucket A*: with a consistent heuristic the
        # *reduced* edge costs can be zero, so no positive width guarantees
        # exactness there without re-opening nodes.  See report §"Data structures".
        w = net.min_edge_weight()
        suite["Dijkstra(bucket)"] = lambda n, s, t: dijkstra(
            n, s, t, queue="bucket", width=w
        )
    return suite


NON_OPTIMAL = {"Greedy(haversine)", "BFS", "DFS"}


def random_queries(
    net: RoadNetwork, n_queries: int, seed: int = 0
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    pairs = rng.integers(0, net.n, size=(n_queries, 2))
    return [(int(s), int(t)) for s, t in pairs if s != t]


def run_benchmark(
    net: RoadNetwork,
    suite: dict[str, Runner],
    n_queries: int = 100,
    seed: int = 0,
    warmup: bool = True,
) -> BenchmarkReport:
    """Run every algorithm in *suite* on the same random queries."""
    queries = random_queries(net, n_queries, seed)
    stats = {
        name: AlgoStats(name=name, is_optimal_algo=name not in NON_OPTIMAL)
        for name in suite
    }

    if warmup:  # touch code paths / JIT-less warm caches for fairer timing
        s, t = queries[0]
        for run in suite.values():
            run(net, s, t)

    for s, t in queries:
        ref = dijkstra(net, s, t)
        for name, run in suite.items():
            res = run(net, s, t)
            st = stats[name]
            st.settled.append(res.settled)
            st.generated.append(res.generated)
            st.times.append(res.elapsed_s)
            st.costs.append(res.cost)
            if np.isfinite(ref.cost) and ref.cost > 0 and np.isfinite(res.cost):
                rel = abs(res.cost - ref.cost) / ref.cost
                st.max_rel_error = max(st.max_rel_error, rel)
                if st.is_optimal_algo and rel > 1e-6:
                    st.optimal = False

    return BenchmarkReport(
        stats=stats, n_queries=len(queries), net_name=net.name, net_nodes=net.n
    )
