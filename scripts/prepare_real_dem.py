#!/usr/bin/env python3
"""
prepare_real_dem.py - makes a solver input grid from the real Copernicus DEM.

PC project ODD-2026 - Rehan Mehta (241080039), Bhavya Gothi (241080021)

1. Download Copernicus GLO-30 tiles (free, needs a login) from
   https://dataspace.copernicus.eu  ->  "Copernicus DEM GLO-30"
   Mumbai needs two tiles: N18 E072 (Colaba end) and N19 E072.
2. (optional) ESA WorldCover 10 m tile N18E072 from https://esa-worldcover.org
   for roads vs trees vs mangroves. Without it, anything above 40 m counts as green.
3. pip install rasterio
4. python scripts/prepare_real_dem.py --dem N18.tif N19.tif --dx 30 [--landcover worldcover.tif]

Note: GLO-30 is a surface model, so big buildings are in it as bumps. For a city
flood model that is actually not too bad (water goes around buildings), but small
streets are lost at 30 m.
"""
import argparse
import csv
import os
import numpy as np
from scipy import ndimage

try:
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import reproject, Resampling, transform as warp_transform
    from rasterio.transform import from_origin
except ImportError:
    raise SystemExit("this script needs rasterio:  pip install rasterio")

from common import write_domain, URBAN, GREEN, SEA

# Greater Mumbai, lon/lat
BBOX = (72.77, 18.89, 73.00, 19.28)
UTM43N = "EPSG:32643"

HOTSPOTS_LL = [  # approximate, from Google Maps
    ("Hindmata", 72.8432, 19.0050),
    ("King's Circle", 72.8573, 19.0278),
    ("Sion", 72.8625, 19.0410),
    ("Kurla (LBS Marg)", 72.8795, 19.0700),
    ("Kalina", 72.8650, 19.0760),
    ("Milan Subway", 72.8410, 19.0880),
    ("Andheri Subway", 72.8470, 19.1170),
    ("Dahisar", 72.8580, 19.2500),
]

GREEN_CLASSES = {10, 20, 30, 90, 95}   # WorldCover: trees, shrubs, grass, wetland, mangroves


def to_grid(srcs, dst_transform, shape, resampling):
    out = np.zeros(shape, np.float32)
    data, transform = merge(srcs) if len(srcs) > 1 else (srcs[0].read(), srcs[0].transform)
    reproject(data[0], out, src_transform=transform, src_crs=srcs[0].crs,
              dst_transform=dst_transform, dst_crs=UTM43N, resampling=resampling)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", nargs="+", required=True, help="Copernicus GLO-30 GeoTIFF tile(s)")
    ap.add_argument("--landcover", nargs="*", default=[], help="ESA WorldCover tile(s), optional")
    ap.add_argument("--dx", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dx = args.dx

    # box corners in UTM
    xs, ys = warp_transform("EPSG:4326", UTM43N, [BBOX[0], BBOX[2], BBOX[0], BBOX[2]],
                            [BBOX[1], BBOX[1], BBOX[3], BBOX[3]])
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    nx, ny = int((x1 - x0) // dx), int((y1 - y0) // dx)
    T = from_origin(x0, y1, dx, dx)       # row 0 = north, same as the solver

    srcs = [rasterio.open(p) for p in args.dem]
    z = to_grid(srcs, T, (ny, nx), Resampling.bilinear).astype(np.float64)
    for s in srcs:
        s.close()

    # sea = low cells connected to the edge of the box (so Powai lake doesn't count)
    low = z <= 0.5
    lab, _ = ndimage.label(low)
    edge = np.unique(np.r_[lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    sea = np.isin(lab, edge[edge > 0])
    z[sea] = np.minimum(z[sea], -3.0)     # give the sea some depth

    typ = np.full((ny, nx), URBAN, np.uint8)
    if args.landcover:
        lsrcs = [rasterio.open(p) for p in args.landcover]
        lc = to_grid(lsrcs, T, (ny, nx), Resampling.mode).astype(int)
        for s in lsrcs:
            s.close()
        typ[np.isin(lc, list(GREEN_CLASSES))] = GREEN
    else:
        typ[z > 40] = GREEN
    typ[sea] = SEA

    manning = np.where(typ == GREEN, 0.08, 0.035)
    manning[sea] = 0.025
    infil = np.where(typ == GREEN, 15.0, 0.0)

    out = args.out or f"data/domain_real_{dx:g}m.bin"
    write_domain(out, z, manning, infil, typ, dx)
    print(f"wrote {out}: {nx} x {ny} at {dx:g} m, land {(~sea).sum() * dx * dx / 1e6:.0f} km2")

    # hotspots in the solver's km coordinates (x from the west edge, y from the south edge)
    hx, hy = warp_transform("EPSG:4326", UTM43N, [p[1] for p in HOTSPOTS_LL], [p[2] for p in HOTSPOTS_LL])
    with open("data/hotspots_real.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "x_km", "y_km"])
        for (name, *_), x, y in zip(HOTSPOTS_LL, hx, hy):
            w.writerow([name, f"{(x - x0) / 1000:.3f}", f"{(y - (y1 - ny * dx)) / 1000:.3f}"])
    print("hotspots: data/hotspots_real.csv  (pass --hotspots data/hotspots_real.csv to animate.py)")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    main()
