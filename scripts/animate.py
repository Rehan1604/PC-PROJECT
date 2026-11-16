#!/usr/bin/env python3
"""
animate.py - turns the depth snapshots into the flood video + the max depth map.

PC project ODD-2026 - Rehan Mehta (241080039), Bhavya Gothi (241080021)

usage:  python scripts/animate.py --run out --domain data/domain_30m.bin
writes  results/flood_<name>.mp4 (or .gif if ffmpeg isn't around)
        results/max_depth_<name>.png
        results/hotspots_<name>.csv
"""
import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LightSource, LinearSegmentedColormap

from common import (read_domain, read_meta, read_frame, read_frames_csv, read_rain,
                    read_hotspots, km_to_cell, SEA, GREEN)

# light blue for ankle deep, dark blue when it is over your head
WATER = LinearSegmentedColormap.from_list("water", ["#9fd3f2", "#3a8fd4", "#1b4f9c", "#0b1f5c"])
DEPTH_MAX = 2.0       # colour scale tops out at 2 m
SHOW_FROM = 0.10      # don't draw puddles under 10 cm


def base_map(z, typ, dx):
    ls = LightSource(azdeg=315, altdeg=40)
    shade = ls.hillshade(z, vert_exag=12, dx=dx, dy=dx)
    rgb = np.dstack([0.20 + 0.70 * shade, 0.20 + 0.68 * shade, 0.21 + 0.62 * shade])
    g = typ == GREEN
    rgb[g] = rgb[g] * np.array([0.72, 0.92, 0.70])
    rgb[typ == SEA] = (0.07, 0.10, 0.17)
    return np.clip(rgb, 0, 1)


def water_rgba(h, typ):
    a = np.clip((h - SHOW_FROM) / 0.4, 0, 1) * 0.92
    a[typ == SEA] = 0
    rgba = WATER(np.clip(h / DEPTH_MAX, 0, 1))
    rgba[..., 3] = a
    return rgba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out", help="output folder of the solver")
    ap.add_argument("--domain", default=None, help="domain file (default: the one in meta.txt)")
    ap.add_argument("--rain", default="data/rain_26jul2005.csv")
    ap.add_argument("--hotspots", default="data/hotspots.csv")
    ap.add_argument("--name", default=None)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--max-px", type=int, default=900, help="downsample so the map is at most this tall")
    args = ap.parse_args()

    meta = read_meta(args.run)
    d = read_domain(args.domain or meta["domain"])
    nx, ny, dx = d["nx"], d["ny"], d["dx"]
    name = args.name or os.path.basename(os.path.normpath(args.run))
    os.makedirs("results", exist_ok=True)

    s = max(1, int(np.ceil(ny / args.max_px)))
    z, typ = d["z"][::s, ::s], d["type"][::s, ::s]
    W, H = nx * dx / 1000, ny * dx / 1000
    ext = (0, W, 0, H)
    bg = base_map(z, typ, dx * s)

    log = read_frames_csv(args.run)
    nframes = len(log)
    hrs, mm = read_rain(args.rain)
    t_h = np.array([float(r["t_sec"]) for r in log]) / 3600
    flooded = np.array([float(r["flooded_km2"]) for r in log])
    hotspots = read_hotspots(args.hotspots) if os.path.exists(args.hotspots) else []

    # ---------------- the video ----------------
    fig = plt.figure(figsize=(10, 9.6), facecolor="#0d1117")
    ax = fig.add_axes([0.02, 0.03, 0.46, 0.92])
    ax.set_axis_off()
    ax.imshow(bg, extent=ext, interpolation="bilinear")
    img = ax.imshow(np.zeros(z.shape + (4,)), extent=ext, interpolation="nearest")
    for nm, x, y in hotspots:
        ax.plot(x, y, "o", ms=3, color="#ffb347", alpha=0.9)
        ax.text(x + 0.3, y, nm, fontsize=6.5, color="#ffe2b8", va="center")

    txt_kw = dict(color="white", transform=fig.transFigure)
    fig.text(0.53, 0.93, "MUMBAI, 26 JULY 2005", fontsize=17, weight="bold", **txt_kw)
    fig.text(0.53, 0.895, "944 mm of rain in 24 hours, replayed on our parallel solver",
             fontsize=9.5, color="#9aa4b2", transform=fig.transFigure)
    clock = fig.text(0.53, 0.80, "", fontsize=34, weight="bold", family="monospace", **txt_kw)
    sub = fig.text(0.53, 0.765, "", fontsize=10, color="#c9d1d9", transform=fig.transFigure)

    axr = fig.add_axes([0.56, 0.47, 0.40, 0.22], facecolor="#0d1117")
    bars = axr.bar(hrs + 0.5, mm, width=0.85, color="#30363d")
    axr.set_xlim(0, 24)
    axr.set_ylabel("rain (mm/h)", color="#9aa4b2", fontsize=9)
    axr.set_title("Rainfall", color="white", fontsize=10, loc="left")

    axf = fig.add_axes([0.56, 0.17, 0.40, 0.20], facecolor="#0d1117")
    axf.plot(t_h, flooded, color="#30363d", lw=2)
    line, = axf.plot([], [], color="#58a6ff", lw=2.5)
    dot, = axf.plot([], [], "o", color="#58a6ff", ms=6)
    axf.set_xlim(0, max(24, t_h[-1]))
    axf.set_ylim(0, max(flooded.max() * 1.15, 1))
    axf.set_xlabel("hours since 08:30 IST", color="#9aa4b2", fontsize=9)
    axf.set_ylabel("km$^2$ under > 30 cm", color="#9aa4b2", fontsize=9)
    axf.set_title("Flooded area", color="white", fontsize=10, loc="left")
    for a in (axr, axf):
        a.tick_params(colors="#9aa4b2", labelsize=8)
        for sp in a.spines.values():
            sp.set_color("#30363d")

    cax = fig.add_axes([0.56, 0.075, 0.40, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=WATER, norm=plt.Normalize(0, DEPTH_MAX)),
                      cax=cax, orientation="horizontal")
    cb.set_label("water depth (m)", color="#9aa4b2", fontsize=8)
    cb.ax.tick_params(colors="#9aa4b2", labelsize=7)
    fig.text(0.53, 0.012, "Rehan Mehta (241080039) · Bhavya Gothi (241080021) · Parallel Computing, VJTI",
             fontsize=7.5, color="#6e7681", transform=fig.transFigure)

    def draw(i):
        h = read_frame(args.run, i, nx, ny)[::s, ::s]
        img.set_data(water_rgba(h, typ))
        r = log[i]
        clock.set_text(r["ist"] + " IST")
        sub.set_text(f"rain {float(r['rain_mmhr']):.0f} mm/h   tide {float(r['tide_m']):+.1f} m   "
                     f"deepest {float(r['max_depth_m']):.1f} m")
        hr_now = int(t_h[i])
        for k, b in enumerate(bars):
            b.set_color("#58a6ff" if k == hr_now else ("#1f6feb55" if k < hr_now else "#30363d"))
        line.set_data(t_h[: i + 1], flooded[: i + 1])
        dot.set_data([t_h[i]], [flooded[i]])
        return img, clock, sub, line, dot

    ani = animation.FuncAnimation(fig, draw, frames=nframes, blit=False)
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    out = f"results/flood_{name}.mp4"
    if animation.FFMpegWriter.isAvailable():
        ani.save(out, writer=animation.FFMpegWriter(fps=args.fps, bitrate=4000), dpi=110)
    else:
        out = out.replace(".mp4", ".gif")
        ani.save(out, writer=animation.PillowWriter(fps=args.fps), dpi=80)
    plt.close(fig)
    print("video:", out)

    # ---------------- max depth map + hotspot table ----------------
    hmax = np.fromfile(os.path.join(args.run, "hmax.bin"), np.float32).reshape(ny, nx)
    fig, ax = plt.subplots(figsize=(5.2, 10))
    ax.imshow(bg, extent=ext)
    ax.imshow(water_rgba(hmax[::s, ::s], typ), extent=ext)
    rows = []
    for nm, x, y in hotspots:
        r, c = km_to_cell(x, y, ny, dx)
        r0, r1, c0, c1 = max(r - 3, 0), min(r + 4, ny), max(c - 3, 0), min(c + 4, nx)
        depth = float(hmax[r0:r1, c0:c1].max())          # deepest point within ~3 cells
        rows.append((nm, depth))
        ax.plot(x, y, "o", ms=4, color="#d1495b")
        ax.annotate(f"{nm}\n{depth:.1f} m", (x, y), xytext=(5, 0), textcoords="offset points",
                    fontsize=7, color="white", va="center")
    ax.set_title("Deepest water reached anywhere during the 24 h", fontsize=10)
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    fig.tight_layout()
    png = f"results/max_depth_{name}.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print("map:  ", png)

    with open(f"results/hotspots_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["place", "max_depth_m"])
        for nm, dep in rows:
            w.writerow([nm, f"{dep:.2f}"])
            print(f"   {nm:<18s} {dep:5.2f} m")


if __name__ == "__main__":
    main()
