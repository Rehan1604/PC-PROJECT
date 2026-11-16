#!/usr/bin/env python3
"""
make_mumbai_dem.py - builds a Mumbai-shaped terrain grid for the flood solver.

PC project ODD-2026 - Rehan Mehta (241080039), Bhavya Gothi (241080021)

Why a generated map and not only the satellite DEM: for the scaling experiments we
need the same city at 30 m, 15 m and 10 m and the Copernicus tile is only 30 m.
So this script draws the city from a hand-traced outline (island city in the
south, suburbs widening to the north, Thane creek on the east) and puts in the
things that matter for flooding:
  - hills: Malabar Hill, Trombay, Powai/Ghatkopar ridge, Sanjay Gandhi NP
  - the Mithi river from Powai down to Mahim creek
  - low pockets at the places that flood every year (Hindmata, Sion, Milan
    subway, Andheri subway ...)
Heights are rough, not surveyed. For a real terrain run use prepare_real_dem.py.

usage:  python scripts/make_mumbai_dem.py --dx 30
        python scripts/make_mumbai_dem.py --dx 10 --no-preview
"""
import argparse
import csv
import os
import numpy as np
from scipy import ndimage
from matplotlib.path import Path

from common import write_domain, URBAN, GREEN, SEA

W_KM, H_KM = 18.0, 40.0   # x = km east of the west edge, y = km north of the south edge

# outline, clockwise from the Colaba tip (traced by hand off Google Maps, very roughly)
COAST = [
    (2.3, 0.2), (1.8, 2.0), (2.0, 3.5), (2.8, 4.5),       # Colaba, Back Bay
    (2.2, 5.2), (1.0, 5.8), (0.7, 7.0),                   # Marine Drive, Malabar Hill
    (1.3, 8.5), (1.6, 10.5), (1.4, 12.5),                 # Haji Ali, Worli
    (2.4, 14.0), (2.0, 15.5), (1.7, 18.0),                # Mahim bay, Bandra
    (1.6, 21.0), (1.3, 24.0), (1.8, 26.0),                # Juhu, Versova
    (1.2, 28.0), (0.8, 31.0), (0.6, 34.0),                # Madh, Gorai
    (0.8, 37.0), (1.0, 40.0),
    (14.5, 40.0),                                          # northern edge (Dahisar/Mulund side)
    (14.0, 36.0), (13.8, 32.0), (13.5, 28.0),             # Thane creek
    (14.5, 24.0), (15.5, 19.0), (13.0, 16.0),             # Mankhurd, Trombay
    (10.5, 15.5), (9.0, 13.5), (7.0, 11.5),               # Mahul, Sewri
    (5.5, 8.5), (4.5, 6.0), (3.6, 3.5), (3.0, 1.5),       # Mazgaon, docks
]

# (x, y, height m, spread x km, spread y km)
HILLS = [
    (1.2, 6.6, 50, 0.45, 0.45),     # Malabar Hill
    (1.8, 11.0, 15, 0.4, 0.4),      # Worli hill
    (6.3, 14.6, 25, 0.35, 0.35),    # Sion fort hill
    (13.6, 19.6, 280, 1.0, 1.1),    # Trombay
    (11.4, 22.0, 90, 0.9, 1.6),     # Ghatkopar ridge, east of Powai
    (8.0, 31.5, 420, 2.4, 4.0),     # Sanjay Gandhi National Park
    (11.5, 27.5, 120, 1.2, 1.8),    # Bhandup / Mulund hills
]

# the usual suspects, places that go under every monsoon (name, x, y, depth m, spread km)
HOTSPOTS = [
    ("Hindmata", 4.3, 9.5, 2.4, 0.35),
    ("King's Circle", 4.8, 12.5, 2.0, 0.30),
    ("Sion", 6.2, 13.7, 2.2, 0.30),
    ("Kurla (LBS Marg)", 7.6, 16.6, 2.0, 0.40),
    ("Kalina", 5.4, 18.6, 1.6, 0.40),
    ("Milan Subway", 3.4, 20.6, 3.0, 0.20),
    ("Andheri Subway", 3.8, 24.6, 3.0, 0.20),
    ("Dahisar", 3.2, 37.0, 1.5, 0.45),
]

# Mithi river, Powai lake outlet to Mahim creek (last point is already in the sea)
MITHI = [(9.6, 23.0), (8.5, 20.5), (7.0, 17.6), (5.6, 16.4), (3.6, 15.2), (2.3, 14.6), (1.6, 14.3)]


def gauss(X, Y, x0, y0, sx, sy):
    return np.exp(-0.5 * (((X - x0) / sx) ** 2 + ((Y - y0) / sy) ** 2))


def dist_to_polyline(X, Y, pts):
    """distance (km) to the polyline and how far along it we are (0 at start, 1 at end)"""
    pts = np.asarray(pts, float)
    seg_len = np.hypot(*(pts[1:] - pts[:-1]).T)
    total = seg_len.sum()
    best = np.full(X.shape, np.inf)
    along = np.zeros(X.shape)
    done = 0.0
    for (x1, y1), (x2, y2), L in zip(pts[:-1], pts[1:], seg_len):
        t = ((X - x1) * (x2 - x1) + (Y - y1) * (y2 - y1)) / (L * L)
        t = np.clip(t, 0, 1)
        d = np.hypot(X - (x1 + t * (x2 - x1)), Y - (y1 + t * (y2 - y1)))
        closer = d < best
        best[closer] = d[closer]
        along[closer] = (done + t[closer] * L) / total
        done += L
    return best, along


def build(dx, seed=2005):
    nx = int(round(W_KM * 1000 / dx))
    ny = int(round(H_KM * 1000 / dx))
    xs = (np.arange(nx) + 0.5) * dx / 1000.0
    ys = H_KM - (np.arange(ny) + 0.5) * dx / 1000.0      # row 0 = north
    X, Y = np.meshgrid(xs, ys)

    land = Path(COAST).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(ny, nx)
    d_land = ndimage.distance_transform_edt(land) * dx / 1000.0     # km inland
    d_sea = ndimage.distance_transform_edt(~land) * dx / 1000.0     # km offshore

    # coastal plain: ~2.5 m at the shore rising to ~6 m inland
    z = 2.5 + 3.5 * (1 - np.exp(-d_land / 1.2))
    for x0, y0, hgt, sx, sy in HILLS:
        z += hgt * gauss(X, Y, x0, y0, sx, sy)

    # some bumpiness so water doesn't sit on a perfectly smooth sheet
    rng = np.random.default_rng(seed)
    coarse = ndimage.gaussian_filter(rng.standard_normal((ny, nx)), sigma=max(300.0 / dx, 1.0))
    coarse /= coarse.std() + 1e-12
    fine = ndimage.gaussian_filter(rng.standard_normal((ny, nx)), sigma=1.0)
    fine /= fine.std() + 1e-12
    z += 0.6 * coarse + 0.1 * fine

    for _, x0, y0, depth, s in HOTSPOTS:
        z -= depth * gauss(X, Y, x0, y0, s, s)

    # carve the Mithi: concave profile, ~38 m below Powai down to -2 m at the mouth
    d_riv, along = dist_to_polyline(X, Y, MITHI)
    bed = 40.0 * (1.0 - along) ** 2 - 2.0
    half_w = 0.04                        # 80 m wide channel
    near = d_riv < 0.25
    z = np.where(near, np.minimum(z, bed + 4.0 * np.clip((d_riv - half_w) / 0.21, 0, 1)), z)
    river = d_riv < half_w
    z[river] = bed[river]
    land |= river & (along < 0.97)       # keep the channel as "land" so it routes water

    # sea floor
    z = np.where(land, z, -4.0 - 3.0 * (1 - np.exp(-d_sea / 2.0)))

    typ = np.full((ny, nx), URBAN, np.uint8)
    green = land & ((z > 40) | (gauss(X, Y, 8.3, 31.5, 1.8, 3.0) > 0.55))
    typ[green] = GREEN
    typ[~land] = SEA

    manning = np.full((ny, nx), 0.035)     # roads + buildings, the usual urban value
    manning[typ == GREEN] = 0.08
    manning[river] = 0.03
    manning[typ == SEA] = 0.025

    infil = np.zeros((ny, nx))             # urban drainage is handled in the solver (--drain)
    infil[typ == GREEN] = 15.0             # mm/h soaked up by forest / open ground

    return z, manning, infil, typ, (X, Y)


def save_preview(path, z, typ, dx, hotspots):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    ny, nx = z.shape
    step = max(1, ny // 1200)
    zz, tt = z[::step, ::step], typ[::step, ::step]
    ls = LightSource(azdeg=315, altdeg=45)
    shade = ls.hillshade(zz, vert_exag=12, dx=dx * step, dy=dx * step)
    rgb = np.dstack([0.93 * shade + 0.05, 0.90 * shade + 0.06, 0.82 * shade + 0.08])
    rgb[tt == SEA] = (0.12, 0.22, 0.36)
    rgb[tt == GREEN] = rgb[tt == GREEN] * (0.75, 0.95, 0.70)

    fig, ax = plt.subplots(figsize=(5, 10))
    ax.imshow(np.clip(rgb, 0, 1), extent=(0, W_KM, 0, H_KM))
    for name, x, y, *_ in hotspots:
        ax.plot(x, y, "o", ms=4, color="#d1495b")
        ax.annotate(name, (x, y), xytext=(4, 2), textcoords="offset points", fontsize=7, color="#5a0e17")
    ax.set_xlabel("km")
    ax.set_ylabel("km")
    ax.set_title(f"Model terrain, {dx:g} m grid ({nx} x {ny})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=30.0, help="cell size in metres (30, 15, 10 ...)")
    ap.add_argument("--out", default=None, help="default data/domain_<dx>m.bin")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    out = args.out or f"data/domain_{args.dx:g}m.bin"
    z, man, infil, typ, _ = build(args.dx)
    write_domain(out, z, man, infil, typ, args.dx)

    land = typ != SEA
    print(f"wrote {out}: {z.shape[1]} x {z.shape[0]} cells at {args.dx:g} m "
          f"({land.sum() * args.dx ** 2 / 1e6:.0f} km2 of land, {os.path.getsize(out) / 1e6:.0f} MB)")

    with open("data/hotspots.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "x_km", "y_km"])
        for name, x, y, *_ in HOTSPOTS:
            w.writerow([name, x, y])

    if not args.no_preview:
        os.makedirs("results", exist_ok=True)
        png = f"results/terrain_{args.dx:g}m.png"
        save_preview(png, z, typ, args.dx, HOTSPOTS)
        print(f"preview: {png}")


if __name__ == "__main__":
    main()
