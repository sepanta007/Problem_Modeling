#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms import (
    alt, astar, bidirectional_alt, bidirectional_dijkstra, build_arcflags,
    build_ch, dijkstra, greedy_best_first,
)
from src.heuristics import haversine_heuristic
from src.landmarks import build_landmarks
from src.network import load_network

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--place", default="Paris, France")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--landmarks", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    net = load_network(place=args.place)
    print(net)
    lm = build_landmarks(net, k=args.landmarks, strategy="farthest", seed=1)
    ch = build_ch(net)
    af = build_arcflags(net, regions=32, seed=0)
    print(f"landmarks {lm.preprocess_s:.1f}s · CH {ch.preprocess_s:.1f}s · "
          f"Arc Flags {af.preprocess_s:.1f}s")

    algos = {
        "Dijkstra": lambda s, t: dijkstra(net, s, t),
        "Greedy": lambda s, t: greedy_best_first(net, s, t, haversine_heuristic),
        "A*": lambda s, t: astar(net, s, t, haversine_heuristic),
        "ALT": lambda s, t: alt(net, s, t, lm),
        "Bi-Dijkstra": lambda s, t: bidirectional_dijkstra(net, s, t),
        "Bi-ALT": lambda s, t: bidirectional_alt(net, s, t, lm),
        "Arc Flags": lambda s, t: af.query(s, t),
        "CH": lambda s, t: ch.query(s, t),
    }
    names = list(algos)

    rng = np.random.default_rng(args.seed)
    dists = []
    settled = {n: [] for n in names}
    for s, t in rng.integers(0, net.n, size=(args.queries, 2)):
        s, t = int(s), int(t)
        if s == t:
            continue
        dists.append(net.haversine(s, t) / 1000.0)  # km, crow-flies
        for n, run in algos.items():
            settled[n].append(run(s, t).settled)
    dists = np.array(dists)
    for n in names:
        settled[n] = np.array(settled[n])

    # ---- stratified table ----------------------------------------------
    edges = [0, 2, 5, 10, np.inf]
    labels = ["<2 km", "2–5 km", "5–10 km", ">10 km"]
    print(f"\nMedian nodes explored by trip distance ({len(dists)} queries)\n")
    header = f"{'stratum':<10}{'#q':>5}" + "".join(f"{n:>11}" for n in names)
    print(header)
    print("-" * len(header))
    for i, lab in enumerate(labels):
        mask = (dists >= edges[i]) & (dists < edges[i + 1])
        if not mask.any():
            continue
        row = f"{lab:<10}{int(mask.sum()):>5}"
        for n in names:
            row += f"{np.median(settled[n][mask]):>11,.0f}"
        print(row)

    print(f"\n{'algorithm':<10}{'median':>10}{'mean':>10}{'p90':>10}{'max':>10}")
    print("-" * 50)
    for n in names:
        a = settled[n]
        print(f"{n:<10}{np.median(a):>10,.0f}{np.mean(a):>10,.0f}"
              f"{np.percentile(a,90):>10,.0f}{np.max(a):>10,.0f}")

    # ---- figure: box-plot + per-algorithm trend curves -----------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    data = [settled[n] for n in names]
    bp = ax1.boxplot(data, tick_labels=names, showfliers=True, patch_artist=True)
    for patch, c in zip(bp["boxes"], plt.cm.viridis(np.linspace(0.15, 0.9, len(names)))):
        patch.set_facecolor(c)
    ax1.set_yscale("log")
    ax1.set_ylabel("nodes explored (log scale)")
    ax1.set_title(f"Distribution of search effort — {args.place}\n"
                  f"({len(dists)} random queries)")
    ax1.grid(axis="y", alpha=0.3)

    # Right: median nodes explored per 1 km distance band, one line per algorithm
    # (a raw 8x300 scatter would overplot into mush; binned medians stay legible
    # and show the *trend* — effort vs trip length — for every algorithm).
    bins = np.arange(0.0, float(np.ceil(dists.max())) + 1.0, 1.0)
    centers = (bins[:-1] + bins[1:]) / 2
    which = np.digitize(dists, bins)
    for n in names:
        med = np.array([
            np.median(settled[n][which == b]) if np.any(which == b) else np.nan
            for b in range(1, len(bins))
        ])
        ax2.plot(centers, med, marker="o", ms=4, lw=2, label=n)
    ax2.set_yscale("log")
    ax2.set_xlabel("trip distance (km, crow-flies)")
    ax2.set_ylabel("median nodes explored (log scale)")
    ax2.set_title("Effort grows with trip length (per algorithm)")
    ax2.legend(ncol=2, fontsize=8); ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "query_distribution.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure -> {out}")


if __name__ == "__main__":
    main()
