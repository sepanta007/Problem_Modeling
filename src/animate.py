from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

from .algorithms.base import SearchResult
from .network import RoadNetwork
from .visualize import _edge_segments

_PALETTE = ["#1f77b4", "#ff7f0e", "#9467bd", "#17becf",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22"]


def _draw_endpoints(net, ax, result, waypoints):
    """Static start/stop/waypoint markers, drawn once so they stay visible."""
    pts = waypoints if waypoints else (result.path[:1] + result.path[-1:]
                                       if result.path else [])
    for i, wp in enumerate(pts):
        if i == 0:
            ax.plot(net.lon[wp], net.lat[wp], "o", color="#2ca02c", ms=9, zorder=6)
        elif i == len(pts) - 1:
            ax.plot(net.lon[wp], net.lat[wp], "s", color="#d62728", ms=9, zorder=6)
        else:
            ax.plot(net.lon[wp], net.lat[wp], "D", color="#ff7f0e", ms=9,
                    markeredgecolor="black", markeredgewidth=0.5, zorder=6)


def animate_comparison(
    net: RoadNetwork,
    results: list[SearchResult],
    out_path: str,
    n_frames: int = 60,
    fps: int = 12,
    ncols: int = 3,
    waypoints: list[int] | None = None,
    suptitle: str | None = None,
) -> str:
    """Save a grid GIF that animates every algorithm's exploration in parallel."""
    edge_segments = _edge_segments(net)
    n = len(results)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    aspect = 1.0 / math.cos(math.radians(float(np.mean(net.lat))))

    coords = []       # per panel: (N,2) array of settled-node coords in order
    scatters = []     # per panel: the growing scatter artist
    path_lines = []   # per panel: the (initially empty) route line
    for i, res in enumerate(results):
        ax = axes[i]
        ax.add_collection(
            LineCollection(edge_segments, colors="#e2e2e2", linewidths=0.3, zorder=1)
        )
        order = res.order if res.order else list(res.visited)
        xy = np.column_stack([net.lon[order], net.lat[order]]) if order \
            else np.empty((0, 2))
        coords.append(xy)
        # Route line UNDER the stain (zorder 2) and dots ON TOP (zorder 3), so
        # sparse explorations (Greedy, ALT, Bi-ALT) stay visible.
        (line,) = ax.plot([], [], "-", color="#d62728", linewidth=1.6,
                          alpha=0.9, zorder=2)
        path_lines.append(line)
        sc = ax.scatter([], [], s=7, color=_PALETTE[i % len(_PALETTE)],
                        alpha=0.65, linewidths=0, zorder=3)
        scatters.append(sc)
        _draw_endpoints(net, ax, res, waypoints)
        ax.set_aspect(aspect)
        ax.set_xticks([]); ax.set_yticks([])
        ax.margins(0.02)
        # Placeholder 2-line title so tight_layout reserves the space now; the
        # real title (same 2 lines) is set per-frame in update(). Without this,
        # tight_layout runs before any title exists and the titles later overlap
        # the panels (visible on the second row).
        ax.set_title(f"{res.algorithm}\n ", fontsize=9)
    for j in range(n, len(axes)):
        axes[j].axis("off")

    max_len = max((len(c) for c in coords), default=1)
    step = max(1, math.ceil(max_len / n_frames))
    n_reveal = math.ceil(max_len / step)
    hold = max(6, n_reveal // 6)          # freeze on the final frame before looping
    total_frames = n_reveal + hold

    def update(frame: int):
        revealed = min(max_len, (frame + 1) * step)
        artists = []
        for i, res in enumerate(results):
            k = min(revealed, len(coords[i]))
            scatters[i].set_offsets(coords[i][:k] if k else np.empty((0, 2)))
            done = k >= len(coords[i])
            if done and res.path and path_lines[i].get_xdata().size == 0:
                path_lines[i].set_data(net.lon[res.path], net.lat[res.path])
            status = (f"{res.cost/1000:.2f} km" if (done and res.found)
                      else f"{k:,}/{len(coords[i]):,} settled")
            axes[i].set_title(f"{res.algorithm}\n{status}", fontsize=9)
            artists += [scatters[i], path_lines[i]]
        return artists

    # Reserve a top band for the suptitle (the GIF cannot use bbox_inches="tight"
    # like the PNG, so without this the title floats far above the panels).
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=1.8)
    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps,
                         blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)
    return out_path


def animate_search(
    net: RoadNetwork,
    result: SearchResult,
    out_path: str,
    n_frames: int = 80,
    fps: int = 15,
    waypoints: list[int] | None = None,
) -> str:
    """Save a single-algorithm GIF (larger, for a close look)."""
    return animate_comparison(
        net, [result], out_path, n_frames=n_frames, fps=fps, ncols=1,
        waypoints=waypoints, suptitle=None,
    )
