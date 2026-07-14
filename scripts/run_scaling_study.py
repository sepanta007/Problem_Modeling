#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms import (
    alt, astar, bidirectional_alt, bidirectional_dijkstra, build_ch, dijkstra,
)
from src.heuristics import haversine_heuristic
from src.landmarks import build_landmarks
from src.network import load_network

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
CENTRE = (48.8566, 2.3522)  # Paris


def eval_on(net, algos, n_queries, seed):
    rng = np.random.default_rng(seed)
    pairs = [(int(s), int(t)) for s, t in rng.integers(0, net.n, size=(n_queries, 2))
             if s != t]
    out = {name: {"settled": [], "time": []} for name in algos}
    for s, t in pairs:
        for name, run in algos.items():
            r = run(s, t)
            out[name]["settled"].append(r.settled)
            out[name]["time"].append(r.elapsed_s)
    return {name: (float(np.mean(d["settled"])), float(np.mean(d["time"]) * 1000))
            for name, d in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dists", type=int, nargs="+",
                    default=[1500, 3000, 5000, 8000, 12000],
                    help="extract radii in metres around Paris centre")
    ap.add_argument("--queries", type=int, default=40)
    ap.add_argument("--landmarks", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ch", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    names = ["Dijkstra", "A*", "ALT", "Bi-Dijkstra", "Bi-ALT"] \
        + ([] if args.no_ch else ["CH"])
    sizes = []
    settled = {n: [] for n in names}
    times = {n: [] for n in names}
    prep = {"ALT": [], "CH": []}

    for dist in args.dists:
        print(f"\n=== radius {dist} m ===")
        net = load_network(point=CENTRE, dist=dist)
        print(f"  {net.n:,} nodes, {net.n_edges:,} edges")
        lm = build_landmarks(net, k=args.landmarks, strategy="farthest", seed=1)
        prep["ALT"].append(lm.preprocess_s)
        algos = {
            "Dijkstra": lambda s, t: dijkstra(net, s, t),
            "A*": lambda s, t: astar(net, s, t, haversine_heuristic),
            "ALT": lambda s, t: alt(net, s, t, lm),
            "Bi-Dijkstra": lambda s, t: bidirectional_dijkstra(net, s, t),
            "Bi-ALT": lambda s, t: bidirectional_alt(net, s, t, lm),
        }
        if not args.no_ch:
            ch = build_ch(net)
            prep["CH"].append(ch.preprocess_s)
            algos["CH"] = lambda s, t: ch.query(s, t)
            print(f"  CH: {ch.preprocess_s:.1f}s prep, {ch.n_shortcuts:,} shortcuts")

        res = eval_on(net, algos, args.queries, args.seed)
        sizes.append(net.n)
        base = res["Dijkstra"][0]
        print(f"  {'algo':<10}{'settled':>10}{'ms':>8}{'vs Dij':>9}")
        for n in names:
            s_mean, t_mean = res[n]
            settled[n].append(s_mean)
            times[n].append(t_mean)
            print(f"  {n:<10}{s_mean:>10,.0f}{t_mean:>8.2f}{base/s_mean:>8.1f}x")

    # ---- plots ---------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5))
    markers = {"Dijkstra": "o", "A*": "v", "ALT": "^", "Bi-Dijkstra": "P",
               "Bi-ALT": "s", "CH": "D"}
    for n in names:
        ax1.plot(sizes, settled[n], marker=markers[n], label=n, lw=2)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("graph size (nodes)"); ax1.set_ylabel("mean nodes explored")
    ax1.set_title("Search effort vs graph size"); ax1.legend(); ax1.grid(alpha=0.3)

    for n in names:
        ax2.plot(sizes, times[n], marker=markers[n], label=n, lw=2)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("graph size (nodes)"); ax2.set_ylabel("mean query time (ms)")
    ax2.set_title("Response time vs graph size"); ax2.legend(); ax2.grid(alpha=0.3)

    for n in names:
        if n == "Dijkstra":
            continue
        speedup = [settled["Dijkstra"][i] / settled[n][i] for i in range(len(sizes))]
        ax3.plot(sizes, speedup, marker=markers[n], label=n, lw=2)
    ax3.set_xscale("log")
    ax3.set_xlabel("graph size (nodes)")
    ax3.set_ylabel("speed-up over Dijkstra (nodes explored)")
    ax3.set_title("Speed-up GROWS with scale"); ax3.legend(); ax3.grid(alpha=0.3)

    fig.suptitle("Scaling study — concentric Paris extracts", fontsize=14)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "scaling_study.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(RESULTS_DIR, "scaling_study.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["nodes"] + [f"{n}_settled" for n in names]
                   + [f"{n}_ms" for n in names])
        for i, sz in enumerate(sizes):
            w.writerow([sz] + [f"{settled[n][i]:.0f}" for n in names]
                       + [f"{times[n][i]:.3f}" for n in names])

    print(f"\nChart -> {out}")
    print(f"CSV   -> {os.path.join(RESULTS_DIR, 'scaling_study.csv')}")


if __name__ == "__main__":
    main()
