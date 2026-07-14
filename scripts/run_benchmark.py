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

from src.algorithms import build_arcflags, build_ch
from src.benchmark import default_suite, run_benchmark
from src.landmarks import build_landmarks
from src.network import load_network

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)


def save_csv(report, path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["algorithm", "mean_settled", "mean_time_ms", "mean_cost_km",
                    "optimal", "max_rel_error"])
        for name, s in report.stats.items():
            w.writerow([name, f"{s.mean_settled:.1f}", f"{s.mean_time_ms:.3f}",
                        f"{s.mean_cost_km:.3f}", s.optimal,
                        f"{s.max_rel_error:.6f}"])


def save_chart(report, path: str) -> None:
    names = list(report.stats.keys())
    settled = [report.stats[n].mean_settled for n in names]
    times = [report.stats[n].mean_time_ms for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(names)))

    ax1.barh(names, settled, color=colors)
    ax1.set_xlabel("mean nodes settled (explored)")
    ax1.set_title("Search effort — fewer is better")
    ax1.invert_yaxis()
    for i, v in enumerate(settled):
        ax1.text(v, i, f" {v:,.0f}", va="center", fontsize=8)

    ax2.barh(names, times, color=colors)
    ax2.set_xlabel("mean response time (ms)")
    ax2.set_title("Response time — lower is better")
    ax2.invert_yaxis()
    for i, v in enumerate(times):
        ax2.text(v, i, f" {v:.2f}", va="center", fontsize=8)

    fig.suptitle(
        f"{report.net_name} — {report.net_nodes:,} nodes, "
        f"{report.n_queries} queries",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--place", default="Paris, France")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--landmarks", type=int, default=16)
    ap.add_argument("--strategy", default="farthest",
                    choices=["random", "planar", "farthest", "avoid"])
    ap.add_argument("--no-ch", action="store_true",
                    help="skip Contraction Hierarchies (avoids the heavy preprocessing)")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Loading {args.place!r} ...")
    net = load_network(place=args.place)
    print(net)

    print(f"Building {args.landmarks} '{args.strategy}' landmarks ...")
    lm = build_landmarks(net, k=args.landmarks, strategy=args.strategy, seed=1)
    print(f"  landmark preprocessing: {lm.preprocess_s:.2f}s "
          f"({2*lm.k} Dijkstra runs)")

    ch = None
    arcflags = None
    if not args.no_ch:
        print("Building Contraction Hierarchy (this is the heavy preprocessing) ...")
        ch = build_ch(net)
        print(f"  CH preprocessing: {ch.preprocess_s:.2f}s, "
              f"{ch.n_shortcuts:,} shortcuts added")
        print("Building Arc Flags (32 regions) ...")
        arcflags = build_arcflags(net, regions=32, seed=0)
        print(f"  Arc Flags preprocessing: {arcflags.preprocess_s:.2f}s")

    suite = default_suite(net, lm, ch=ch, arcflags=arcflags)
    print(f"\nRunning {len(suite)} algorithms x {args.queries} queries ...")
    report = run_benchmark(net, suite, n_queries=args.queries, seed=args.seed)
    print(report.to_table())

    csv_path = os.path.join(RESULTS_DIR, "benchmark.csv")
    chart_path = os.path.join(RESULTS_DIR, "benchmark_summary.png")
    save_csv(report, csv_path)
    save_chart(report, chart_path)
    print(f"\nCSV   -> {csv_path}")
    print(f"Chart -> {chart_path}")


if __name__ == "__main__":
    main()
