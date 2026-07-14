from __future__ import annotations

import math
from dataclasses import dataclass

from .network import RoadNetwork


@dataclass
class Step:
    instruction: str   # short turn phrase, e.g. "à gauche"
    street: str
    distance_m: float
    start_node: int = -1   # graph node where this step begins (for map markers)


def _bearing(net: RoadNetwork, u: int, v: int) -> float:
    """Initial compass bearing (degrees, 0 = North, clockwise) from u to v."""
    lat1, lat2 = math.radians(net.lat[u]), math.radians(net.lat[v])
    dlon = math.radians(net.lon[v] - net.lon[u])
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _turn(prev_b: float | None, next_b: float) -> str:
    """Short turn phrase going from bearing *prev_b* to *next_b*."""
    if prev_b is None:
        return "départ"
    d = (next_b - prev_b + 180.0) % 360.0 - 180.0  # signed angle in (-180, 180]
    a = abs(d)
    if a < 25:
        return "tout droit"
    if a > 150:
        return "demi-tour"
    side = "à droite" if d > 0 else "à gauche"          # bearing increases clockwise
    prefix = "légèrement " if a < 50 else ""
    return f"{prefix}{side}"


def build_itinerary(net: RoadNetwork, path: list[int]) -> list[Step]:
    """Group a node path into named steps with turn instructions."""
    if len(path) < 2:
        return []
    # Per-segment (name, length, bearing) along the path.
    segs = []
    for a, b in zip(path, path[1:]):
        w = min((wt for v, wt in net.adj[a] if v == b), default=net.haversine(a, b))
        segs.append((net.street(a, b), w, _bearing(net, a, b)))

    steps: list[Step] = []
    i = 0
    prev_last_bearing: float | None = None
    while i < len(segs):
        name = segs[i][0]
        first_bearing = segs[i][2]
        total = 0.0
        j = i
        while j < len(segs) and segs[j][0] == name:
            total += segs[j][1]
            j += 1
        steps.append(Step(_turn(prev_last_bearing, first_bearing), name, total,
                          start_node=path[i]))
        prev_last_bearing = segs[j - 1][2]
        i = j
    return steps


def _arrow(instruction: str) -> str:
    if instruction == "départ":
        return "→"
    if instruction == "tout droit":
        return "↑"
    if instruction == "demi-tour":
        return "↻"
    return "↱" if "droite" in instruction else "↰"


def format_itinerary(steps: list[Step], origin: str, dest: str) -> str:
    """Pretty multi-line route sheet (street column fixed, turn phrase last)."""
    lines = [f"Départ — {origin}"]
    for s in steps:
        street = s.street if len(s.street) <= 38 else s.street[:37] + "…"
        dist = (f"{s.distance_m:,.0f} m" if s.distance_m < 1000
                else f"{s.distance_m / 1000:.2f} km")
        turn = "" if s.instruction == "départ" else f"({s.instruction})"
        lines.append(f"  {_arrow(s.instruction)}  {street:<39}{dist:>9}   {turn}")
    total = sum(s.distance_m for s in steps)
    lines.append(f"Arrivée — {dest}   (total {total / 1000:.2f} km, "
                 f"{len(steps)} étapes)")
    return "\n".join(lines)
