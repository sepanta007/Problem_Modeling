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

from src.algorithms import alt, dijkstra
from src.benchmark import random_queries
from src.landmarks import build_landmarks
from src.network import load_network
from src.visualize import plot_landmarks

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)


def eval_alt(net, lm, queries) -> tuple[float, float, bool]:
    """Mean settled, mean ms, optimal? for ALT with these landmarks."""
    settled, times = [], []
    optimal = True
    for s, t in queries:
        ref = dijkstra(net, s, t)
        r = alt(net, s, t, lm)
        settled.append(r.settled)
        times.append(r.elapsed_s)
        if ref.cost > 0 and abs(r.cost - ref.cost) / ref.cost > 1e-6:
            optimal = False
    return float(np.mean(settled)), float(np.mean(times)) * 1000, optimal


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--place", default="Paris, France")
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    net = load_network(place=args.place)
    print(net)
    queries = random_queries(net, args.queries, seed=args.seed)

    dij_settled = float(np.mean([dijkstra(net, s, t).settled for s, t in queries]))
    print(f"\nDijkstra baseline: {dij_settled:,.0f} mean nodes settled\n")

    strategies = ["random", "planar", "farthest", "avoid"]
    curves: dict[str, list[float]] = {s: [] for s in strategies}
    preprocess: dict[str, list[float]] = {s: [] for s in strategies}

    print(f"{'strategy':<12}{'k':>4}{'prep_s':>9}{'settled':>12}"
          f"{'vs Dij':>9}{'ms':>8}{'opt':>6}")
    print("-" * 60)
    for strat in strategies:
        for k in args.ks:
            lm = build_landmarks(net, k=k, strategy=strat, seed=1)
            ms_settled, ms_time, opt = eval_alt(net, lm, queries)
            curves[strat].append(ms_settled)
            preprocess[strat].append(lm.preprocess_s)
            print(f"{strat:<12}{k:>4}{lm.preprocess_s:>9.2f}{ms_settled:>12,.0f}"
                  f"{dij_settled/ms_settled:>8.1f}x{ms_time:>8.2f}"
                  f"{'yes' if opt else 'NO':>6}")

    # --- curve: mean settled vs k, per strategy -------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    markers = {"random": "o", "planar": "s", "farthest": "^", "avoid": "D"}
    for strat in strategies:
        ax.plot(args.ks, curves[strat], marker=markers[strat], label=strat, lw=2)
    ax.axhline(dij_settled, color="grey", ls="--", label="Dijkstra baseline")
    ax.set_xlabel("number of landmarks k")
    ax.set_ylabel("mean nodes settled by ALT (fewer = better)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(args.ks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_title(f"Effect of landmark selection — {args.place} "
                 f"({args.queries} queries)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    curve_path = os.path.join(RESULTS_DIR, "landmark_study.png")
    fig.tight_layout()
    fig.savefig(curve_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nCurve -> {curve_path}")

    # --- placement maps for k=16 ----------------------------------------
    for strat in strategies:
        lm = build_landmarks(net, k=16, strategy=strat, seed=1)
        p = os.path.join(RESULTS_DIR, f"landmarks_{strat}.png")
        plot_landmarks(net, lm.nodes, p,
                       title=f"{strat} landmarks (k=16) — {args.place}")
        print(f"Placement ({strat}) -> {p}")


if __name__ == "__main__":
    main()
