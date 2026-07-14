from __future__ import annotations

import math
import os
import pickle
from dataclasses import dataclass, field

import numpy as np

# Mean Earth radius (metres), WGS-84 authalic sphere.
EARTH_RADIUS_M = 6_371_008.8


@dataclass
class RoadNetwork:
    """A directed, weighted road graph with contiguous integer node ids."""

    n: int
    adj: list[list[tuple[int, float]]]
    radj: list[list[tuple[int, float]]]
    lat: np.ndarray  # shape (n,), degrees
    lon: np.ndarray  # shape (n,), degrees
    osmid: np.ndarray  # shape (n,), original OSM node ids (for reference / plotting)
    name: str = "road-network"
    edge_name: dict = field(default_factory=dict)  # (u, v) -> OSM street name

    def street(self, u: int, v: int) -> str:
        """Street name of the arc u -> v (falls back to a placeholder)."""
        return self.edge_name.get((u, v)) or "route sans nom"

    # ---- geometry -------------------------------------------------------

    def haversine(self, u: int, v: int) -> float:
        """Great-circle distance between nodes *u* and *v*, in metres.

        This is the "distance à vol d'oiseau" heuristic of the course.  Because
        travelling along roads can never be shorter than the straight-line
        distance, it is an **admissible** (and consistent) heuristic for A*.
        """
        lat1 = math.radians(self.lat[u])
        lat2 = math.radians(self.lat[v])
        dlat = lat2 - lat1
        dlon = math.radians(self.lon[v] - self.lon[u])
        a = (
            math.sin(dlat / 2.0) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
        )
        return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))

    def haversine_arrays(self, u: int, targets: np.ndarray) -> np.ndarray:
        """Vectorised haversine from one node to many (used by planar landmarks)."""
        lat1 = math.radians(self.lat[u])
        lon1 = math.radians(self.lon[u])
        lat2 = np.radians(self.lat[targets])
        lon2 = np.radians(self.lon[targets])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = (
            np.sin(dlat / 2.0) ** 2
            + math.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        )
        return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))

    # ---- convenience ----------------------------------------------------

    @property
    def n_edges(self) -> int:
        return sum(len(neigh) for neigh in self.adj)

    def min_edge_weight(self) -> float:
        """Smallest strictly-positive edge weight (metres).

        This is the exactness threshold for the bucket queue: Dial's algorithm
        is provably optimal iff the bucket width does not exceed it (a node in
        the current bucket can then never be improved by a later relaxation).
        Cached lazily because it is not stored in the on-disk pickle.
        """
        cached = getattr(self, "_min_w", None)
        if cached is None:
            m = math.inf
            for neigh in self.adj:
                for _, w in neigh:
                    if 0.0 < w < m:
                        m = w
            cached = m if m < math.inf else 1.0
            self._min_w = cached
        return cached

    def nearest_node(self, lat: float, lon: float) -> int:
        """Return the id of the graph node closest to a (lat, lon) point."""
        # Small equirectangular approximation is fine for "snap to nearest".
        lat0 = math.radians(lat)
        dx = (np.radians(self.lon) - math.radians(lon)) * math.cos(lat0)
        dy = np.radians(self.lat) - math.radians(lat)
        return int(np.argmin(dx * dx + dy * dy))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"RoadNetwork(name={self.name!r}, nodes={self.n:,}, "
            f"edges={self.n_edges:,})"
        )


# ---------------------------------------------------------------------------
# Building a RoadNetwork from an osmnx MultiDiGraph
# ---------------------------------------------------------------------------


def _from_osmnx(G, name: str) -> RoadNetwork:
    """Convert an osmnx MultiDiGraph into a compact RoadNetwork."""
    osm_ids = list(G.nodes)
    index = {osm: i for i, osm in enumerate(osm_ids)}
    n = len(osm_ids)

    lat = np.empty(n, dtype=np.float64)
    lon = np.empty(n, dtype=np.float64)
    for osm, i in index.items():
        data = G.nodes[osm]
        lat[i] = data["y"]
        lon[i] = data["x"]

    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    radj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    edge_name: dict = {}

    for u_osm, v_osm, data in G.edges(data=True):
        u, v = index[u_osm], index[v_osm]
        # osmnx stores edge length in metres under "length".
        w = float(data.get("length", 0.0))
        # A MultiDiGraph can hold several edges u->v; keep the shortest.
        adj[u].append((v, w))
        radj[v].append((u, w))
        # Street name (OSM "name" may be a string, a list, or missing).
        nm = data.get("name")
        if isinstance(nm, list):
            nm = nm[0] if nm else None
        if nm and (u, v) not in edge_name:
            edge_name[(u, v)] = str(nm)

    return RoadNetwork(
        n=n,
        adj=adj,
        radj=radj,
        lat=lat,
        lon=lon,
        osmid=np.asarray(osm_ids, dtype=np.int64),
        name=name,
        edge_name=edge_name,
    )


# ---------------------------------------------------------------------------
# Public loader with on-disk caching
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_network(
    place: str | None = None,
    point: tuple[float, float] | None = None,
    dist: int = 3000,
    network_type: str = "drive",
    cache_dir: str = DEFAULT_CACHE_DIR,
    force_download: bool = False,
) -> RoadNetwork:
    """Load a road network, downloading it from OpenStreetMap the first time.

    Parameters
    ----------
    place:
        A place name understood by Nominatim, e.g. ``"Paris, France"`` or
        ``"Boulogne-Billancourt, France"``.  Mutually exclusive with *point*.
    point:
        ``(lat, lon)`` centre; downloads everything within *dist* metres.
    dist:
        Radius in metres when *point* is used.
    network_type:
        osmnx network type -- ``"drive"`` gives the routable car network.
    cache_dir:
        Where the compact ``.pkl`` cache is written.  Subsequent runs load
        instantly from disk, so the Overpass API is queried only once.
    force_download:
        Ignore any cache and re-download.

    Returns
    -------
    RoadNetwork
    """
    if (place is None) == (point is None):
        raise ValueError("Provide exactly one of `place` or `point`.")

    os.makedirs(cache_dir, exist_ok=True)
    if place is not None:
        key = f"{place}_{network_type}".replace(",", "").replace(" ", "_")
        name = place
    else:
        key = f"pt_{point[0]:.4f}_{point[1]:.4f}_{dist}_{network_type}"
        name = f"{point} r={dist}m"
    cache_path = os.path.join(cache_dir, f"{key}.pkl")

    if os.path.exists(cache_path) and not force_download:
        with open(cache_path, "rb") as fh:
            net = pickle.load(fh)
        net.name = name
        return net

    import osmnx as ox

    if place is not None:
        G = ox.graph_from_place(place, network_type=network_type)
    else:
        G = ox.graph_from_point(point, dist=dist, network_type=network_type)

    # Ensure every edge carries a metric length (older extracts sometimes lack it).
    has_length = any("length" in data for _, _, data in G.edges(data=True))
    if not has_length:
        G = ox.distance.add_edge_lengths(G)

    # Keep only the largest strongly connected component so that *every* pair of
    # nodes is mutually reachable.  This makes all queries solvable and keeps the
    # ALT landmark distances finite (no inf special-casing needed downstream).
    G = ox.truncate.largest_component(G, strongly=True)

    net = _from_osmnx(G, name)
    with open(cache_path, "wb") as fh:
        pickle.dump(net, fh)
    return net


# ---------------------------------------------------------------------------
# Offline fallback: a synthetic grid city (used by tests / when offline)
# ---------------------------------------------------------------------------


def grid_network(rows: int = 40, cols: int = 40, spacing_m: float = 100.0) -> RoadNetwork:
    """A deterministic grid graph, handy for unit tests and offline demos.

    Nodes are laid out on a regular lat/lon grid near Paris; edges connect the
    4-neighbourhood with weight equal to the true haversine distance, so A*'s
    straight-line heuristic behaves exactly as it would on a real network.
    """
    n = rows * cols
    lat = np.empty(n)
    lon = np.empty(n)
    # ~111.32 km per degree of latitude; scale longitude by cos(lat).
    lat0, lon0 = 48.85, 2.35
    dlat = spacing_m / 111_320.0
    dlon = spacing_m / (111_320.0 * math.cos(math.radians(lat0)))

    def idx(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            i = idx(r, c)
            lat[i] = lat0 + r * dlat
            lon[i] = lon0 + c * dlon

    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    radj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    net = RoadNetwork(
        n=n, adj=adj, radj=radj, lat=lat, lon=lon,
        osmid=np.arange(n, dtype=np.int64), name=f"grid-{rows}x{cols}",
    )
    for r in range(rows):
        for c in range(cols):
            i = idx(r, c)
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    j = idx(nr, nc)
                    w = net.haversine(i, j)
                    adj[i].append((j, w))
                    radj[j].append((i, w))
    return net
