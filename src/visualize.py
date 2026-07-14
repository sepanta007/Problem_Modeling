from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: render straight to files
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from .algorithms.base import SearchResult
from .network import RoadNetwork


def _edge_segments(net: RoadNetwork) -> np.ndarray:
    """Return an (E, 2, 2) array of (lon, lat) segment endpoints for all edges."""
    segs = []
    for u in range(net.n):
        x1, y1 = net.lon[u], net.lat[u]
        for v, _ in net.adj[u]:
            segs.append([(x1, y1), (net.lon[v], net.lat[v])])
    return np.asarray(segs)


def plot_search(
    net: RoadNetwork,
    result: SearchResult,
    ax: "plt.Axes",
    edge_segments: np.ndarray | None = None,
    stain_color: str = "#1f77b4",
    waypoints: list[int] | None = None,
) -> None:
    """Draw one algorithm's exploration + route on *ax*.

    If *waypoints* is given (a multi-stop query), its first node is marked as the
    start (green circle), its last as the end (red square), and every
    intermediate stop as a numbered orange diamond.
    """
    if edge_segments is None:
        edge_segments = _edge_segments(net)

    ax.add_collection(
        LineCollection(edge_segments, colors="#dddddd", linewidths=0.3, zorder=1)
    )

    # Draw the route FIRST (lower z-order), then the explored nodes ON TOP, so
    # that algorithms which explore very few nodes (Greedy, ALT, Bi-ALT) still
    # show their dots instead of hiding them under the thick route line.
    if result.path:
        px = net.lon[result.path]
        py = net.lat[result.path]
        ax.plot(px, py, "-", color="#d62728", linewidth=1.6, alpha=0.9, zorder=2)

    if result.visited:
        vis = np.fromiter(result.visited, dtype=int, count=len(result.visited))
        # Bigger dots when the exploration is sparse, so it stays visible.
        size = 4 if len(vis) > 400 else 9
        ax.scatter(
            net.lon[vis], net.lat[vis], s=size, color=stain_color, alpha=0.6,
            linewidths=0, zorder=3,
        )
        if waypoints:
            for i, wp in enumerate(waypoints):
                if i == 0:
                    ax.plot(net.lon[wp], net.lat[wp], "o", color="#2ca02c",
                            ms=9, zorder=5)
                elif i == len(waypoints) - 1:
                    ax.plot(net.lon[wp], net.lat[wp], "s", color="#d62728",
                            ms=9, zorder=5)
                else:
                    ax.plot(net.lon[wp], net.lat[wp], "D", color="#ff7f0e",
                            ms=9, markeredgecolor="black", markeredgewidth=0.5,
                            zorder=5)
                    ax.annotate(str(i), (net.lon[wp], net.lat[wp]),
                                fontsize=8, ha="center", va="center", zorder=6)
        else:
            s, t = result.path[0], result.path[-1]
            ax.plot(net.lon[s], net.lat[s], "o", color="#2ca02c", ms=8, zorder=4)
            ax.plot(net.lon[t], net.lat[t], "s", color="#d62728", ms=8, zorder=4)

    if result.found:
        third = f"{result.cost/1000:.2f} km"
    elif result.meta.get("aborted") or result.meta.get("failed_leg") is not None:
        third = "ABORTED (budget)"
    else:
        third = "no path"
    ax.set_title(
        f"{result.algorithm}\n{result.settled:,} settled · "
        f"{result.elapsed_s*1000:.1f} ms · {third}",
        fontsize=10,
    )
    ax.set_aspect(1.0 / np.cos(np.radians(float(np.mean(net.lat)))))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.02)


def compare_searches(
    net: RoadNetwork,
    results: list[SearchResult],
    out_path: str,
    ncols: int = 3,
    suptitle: str | None = None,
    waypoints: list[int] | None = None,
) -> str:
    """Grid of exploration plots, one panel per algorithm; saved to *out_path*."""
    edge_segments = _edge_segments(net)
    n = len(results)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    # A curated palette that avoids green and red (reserved for source/target
    # markers and the route line) so the stain never blends into them.
    palette = ["#1f77b4", "#ff7f0e", "#9467bd", "#17becf",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]
    for i, res in enumerate(results):
        plot_search(net, res, axes[i], edge_segments,
                    stain_color=palette[i % len(palette)], waypoints=waypoints)
    for j in range(n, len(axes)):
        axes[j].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_landmarks(
    net: RoadNetwork,
    landmark_nodes: list[int],
    out_path: str,
    title: str = "Landmark placement",
) -> str:
    """Show where landmarks sit on the map (for the landmark-selection study)."""
    edge_segments = _edge_segments(net)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.add_collection(
        LineCollection(edge_segments, colors="#dddddd", linewidths=0.3, zorder=1)
    )
    lm = np.asarray(landmark_nodes, dtype=int)
    ax.scatter(net.lon[lm], net.lat[lm], s=90, c="#d62728", marker="*",
               edgecolors="black", linewidths=0.5, zorder=3)
    ax.set_title(title, fontsize=12)
    ax.set_aspect(1.0 / np.cos(np.radians(float(np.mean(net.lat)))))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
