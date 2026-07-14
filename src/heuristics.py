from __future__ import annotations

from typing import Callable

from .network import RoadNetwork


def zero_heuristic(_net: RoadNetwork, _target: int) -> Callable[[int], float]:
    """h(u) = 0 for all u.  A* with this heuristic *is* Dijkstra."""
    return lambda _u: 0.0


def haversine_heuristic(net: RoadNetwork, target: int) -> Callable[[int], float]:
    """The "distance à vol d'oiseau" heuristic.

    Straight-line (great-circle) distance to the target.  Admissible because no
    road can be shorter than the straight line joining two points, and
    consistent because the great-circle metric obeys the triangle inequality.
    It is weak on road networks, though: detours around rivers, railways and
    motorway-only corridors make the true distance far larger than the crow
    flies -- which is exactly why ALT does better.
    """
    lat = net.lat
    lon = net.lon
    t_lat = lat[target]
    t_lon = lon[target]

    import math

    R = 6_371_008.8
    t_lat_r = math.radians(t_lat)
    cos_t = math.cos(t_lat_r)

    def h(u: int) -> float:
        lat1 = math.radians(lat[u])
        dlat = t_lat_r - lat1
        dlon = math.radians(t_lon - lon[u])
        a = (
            math.sin(dlat / 2.0) ** 2
            + math.cos(lat1) * cos_t * math.sin(dlon / 2.0) ** 2
        )
        return 2.0 * R * math.asin(math.sqrt(a))

    return h
