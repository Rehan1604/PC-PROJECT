#!/usr/bin/env python3
"""
plot_scaling.py - speedup / efficiency / time breakdown plots from results/timings.csv

PC project ODD-2026 - Rehan Mehta (241080039), Bhavya Gothi (241080021)

usage:  python scripts/plot_scaling.py [--csv results/timings.csv] [--nx 600]
writes  results/speedup.png, results/efficiency.png, results/breakdown.png,
        results/summary.md (tables we paste into the report)
"""
import argparse
import csv
import os
from collections import OrderedDict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"omp": "#2a7ab9", "mpi": "#e07a2f", "hybrid": "#3c9d5d"}


def load(path, nx=None):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ("ranks", "threads", "nx", "ny", "steps"):
                r[k] = int(r[k])
            for k in ("dx", "sim_hours", "wall_s", "compute_s", "comm_s", "io_s",
                      "imbalance", "mcups", "vol_err"):
                r[k] = float(r[k])
            r["balance"], r["overlap"] = r["balance"] == "1", r["overlap"] == "1"
            r["p"] = r["ranks"] * r["threads"]
            rows.append(r)
    if nx is None and rows:
        nx = max(rows, key=lambda r: r["nx"])["nx"] if not any(r["mode"] == "serial" for r in rows) \
            else [r for r in rows if r["mode"] == "serial"][-1]["nx"]
    # keep the latest entry per tag for this grid size
    latest = OrderedDict()
    for r in rows:
        if r["nx"] == nx:
            latest[r["tag"]] = r
    return list(latest.values()), nx


def amdahl_fit(p, s):
    """serial fraction f that best explains S(p) = 1 / (f + (1-f)/p)"""
    best, best_err = 0.0, 1e9
    for f in np.linspace(0, 0.5, 5001):
        pred = 1 / (f + (1 - f) / p)
        err = ((pred - s) ** 2).sum()
        if err < best_err:
            best, best_err = f, err
    return best


def label(r):
    if r["mode"] == "omp":
        return f"OpenMP {r['threads']}T"
    if r["mode"] == "mpi":
        return f"MPI {r['ranks']}P"
    if r["mode"] == "hybrid":
        return f"Hybrid {r['ranks']}P x {r['threads']}T"
    return r["mode"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/timings.csv")
    ap.add_argument("--nx", type=int, default=None, help="which grid size to plot (default: serial run's)")
    args = ap.parse_args()

    rows, nx = load(args.csv, args.nx)
    serial = [r for r in rows if r["mode"] == "serial" and not r["tag"].startswith("exp")]
    if not serial:
        raise SystemExit("no serial run in the csv for this grid, run that first")
    base = serial[-1]
    T1 = base["wall_s"]
    scaling = [r for r in rows if r["mode"] in COLORS and not r["tag"].startswith("exp")]
    os.makedirs("results", exist_ok=True)
    maxp = max([r["p"] for r in scaling] + [2])

    # speedup
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot([1, maxp], [1, maxp], "--", color="#999", lw=1, label="ideal")
    fits = {}
    for mode, col in COLORS.items():
        rs = sorted([r for r in scaling if r["mode"] == mode], key=lambda r: r["p"])
        if not rs:
            continue
        p = np.array([r["p"] for r in rs], float)
        s = T1 / np.array([r["wall_s"] for r in rs])
        ax.plot(p, s, "o-", color=col, label=mode if mode != "omp" else "OpenMP", lw=2)
        if mode == "omp" and len(rs) >= 3:
            f = amdahl_fit(p, s)
            fits[mode] = f
            pp = np.linspace(1, maxp, 100)
            ax.plot(pp, 1 / (f + (1 - f) / pp), ":", color=col, lw=1.2,
                    label=f"Amdahl fit, serial part {100 * f:.1f}%")
    ax.set_xlabel("workers (processes x threads)")
    ax.set_ylabel(f"speedup over serial ({T1:.1f} s)")
    ax.set_title(f"Strong scaling, {base['nx']} x {base['ny']} grid ({base['dx']:g} m)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/speedup.png", dpi=150)
    plt.close(fig)

    # efficiency
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for mode, col in COLORS.items():
        rs = sorted([r for r in scaling if r["mode"] == mode], key=lambda r: r["p"])
        if rs:
            p = np.array([r["p"] for r in rs], float)
            e = T1 / np.array([r["wall_s"] for r in rs]) / p
            ax.plot(p, 100 * e, "o-", color=col, lw=2, label=mode if mode != "omp" else "OpenMP")
    ax.axhline(100, ls="--", color="#999", lw=1)
    ax.set_ylim(0, 110)
    ax.set_xlabel("workers")
    ax.set_ylabel("parallel efficiency (%)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/efficiency.png", dpi=150)
    plt.close(fig)

    # where the time goes
    shown = [base] + sorted(scaling, key=lambda r: (list(COLORS).index(r["mode"]), r["p"])) + \
            [r for r in rows if r["tag"].startswith("exp")]
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(shown) + 2), 4.4))
    x = np.arange(len(shown))
    comp = np.array([r["compute_s"] for r in shown])
    comm = np.array([r["comm_s"] for r in shown])
    io = np.array([r["io_s"] for r in shown])
    ax.bar(x, comp, color="#2a7ab9", label="compute")
    ax.bar(x, comm, bottom=comp, color="#e07a2f", label="communication / waiting")
    ax.bar(x, io, bottom=comp + comm, color="#b8b8b8", label="output")
    ax.set_xticks(x)
    ax.set_xticklabels([r["tag"] for r in shown], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("seconds (slowest process)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/breakdown.png", dpi=150)
    plt.close(fig)

    # tables for the report
    lines = [f"Grid {base['nx']} x {base['ny']} at {base['dx']:g} m, {base['sim_hours']:g} simulated hours, "
             f"serial time {T1:.2f} s\n",
             "| Run | Workers | Time (s) | Speedup | Efficiency | Karp-Flatt e | Comm (s) | Imbalance | Mcell/s |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in [base] + sorted(scaling, key=lambda r: (list(COLORS).index(r["mode"]), r["p"])):
        S = T1 / r["wall_s"]
        p = r["p"]
        kf = (1 / S - 1 / p) / (1 - 1 / p) if p > 1 else float("nan")
        lines.append(f"| {label(r)} | {p} | {r['wall_s']:.2f} | {S:.2f} | {100 * S / p:.0f}% | "
                     f"{'' if p == 1 else f'{kf:.3f}'} | {r['comm_s']:.2f} | {r['imbalance']:.2f} | {r['mcups']:.0f} |")
    exps = [r for r in rows if r["tag"].startswith("exp")]
    if exps:
        lines += ["", "Experiments\n", "| Tag | Run | Time (s) | Comm (s) | Imbalance |", "|---|---|---|---|---|"]
        for r in exps:
            lines.append(f"| {r['tag']} | {label(r)} | {r['wall_s']:.2f} | {r['comm_s']:.2f} | {r['imbalance']:.2f} |")
    if fits:
        lines.append(f"\nAmdahl fit on the OpenMP runs: serial fraction ~ {100 * fits['omp']:.1f}%, "
                     f"so the best possible speedup on this code is ~ {1 / max(fits['omp'], 1e-6):.0f}x")
    with open("results/summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nplots: results/speedup.png, results/efficiency.png, results/breakdown.png")


if __name__ == "__main__":
    main()
