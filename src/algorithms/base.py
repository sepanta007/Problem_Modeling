from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """Everything a single shortest-path query produced.

    The metrics are deliberately the ones the assignment asks us to compare:
    *nombre de nœuds explorés* (``settled``) and *temps de réponse*
    (``elapsed_s``).  ``visited`` is the set drawn as the explored "stain" on
    the map.
    """

    path: list[int]              # node ids from source to target ([] if unreachable)
    cost: float                  # total path length in metres (inf if unreachable)
    settled: int = 0             # nodes permanently expanded (popped with final dist)
    generated: int = 0           # edge relaxations that improved a node (≈ pushes)
    visited: set[int] = field(default_factory=set)  # every node ever settled -> the stain
    order: list[int] = field(default_factory=list)   # settle order, for animation
    elapsed_s: float = 0.0
    algorithm: str = ""
    meta: dict = field(default_factory=dict)  # algorithm-specific extras

    @property
    def found(self) -> bool:
        return len(self.path) > 0

    @property
    def hops(self) -> int:
        return max(0, len(self.path) - 1)

    def summary(self) -> str:
        if not self.found:
            return f"{self.algorithm}: NO PATH ({self.settled} settled)"
        return (
            f"{self.algorithm}: {self.cost/1000:.3f} km, "
            f"{self.hops} hops, {self.settled:,} settled, "
            f"{self.elapsed_s*1000:.1f} ms"
        )


def combine_legs(legs: list["SearchResult"], algorithm: str) -> "SearchResult":
    """Stitch consecutive shortest-path *legs* into one multi-stop result.

    A multi-stop query "visit A, then B, then C in this order" is just the
    concatenation of the optimal legs A→B and B→C: because the intermediate
    stops are fixed, the legs are independent and each is solved on its own.
    (If the *order* of the stops were free instead, the problem would become a
    Travelling-Salesman-style ordering problem — strictly harder.)

    Costs and search effort add up; the shared junction between two legs is not
    counted twice in the path; explored "stains" are unioned.
    """
    path: list[int] = []
    cost = 0.0
    settled = generated = 0
    elapsed = 0.0
    visited: set[int] = set()
    order: list[int] = []
    for i, leg in enumerate(legs):
        # Count this leg's work even if it failed, so an aborted multi-stop
        # result still reports the effort it spent before giving up.
        settled += leg.settled
        generated += leg.generated
        elapsed += leg.elapsed_s
        visited |= leg.visited
        order.extend(leg.order)  # legs explored one after another -> animatable
        if not leg.found:
            return SearchResult(path=[], cost=float("inf"), settled=settled,
                                generated=generated, visited=visited,
                                order=order, elapsed_s=elapsed,
                                algorithm=algorithm, meta={"failed_leg": i})
        cost += leg.cost
        path.extend(leg.path if i == 0 else leg.path[1:])
    return SearchResult(
        path=path, cost=cost, settled=settled, generated=generated,
        visited=visited, order=order, elapsed_s=elapsed, algorithm=algorithm,
        meta={"n_legs": len(legs)},
    )


def reconstruct_path(came_from: dict[int, int], src: int, dst: int) -> list[int]:
    """Rebuild the node sequence src -> ... -> dst from a predecessor map."""
    if dst not in came_from and dst != src:
        return []
    path = [dst]
    node = dst
    while node != src:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path
