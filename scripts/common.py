"""Shared helpers for reading the solver's input and output files.

PC project ODD-2026 - Rehan Mehta (241080039), Bhavya Gothi (241080021)
"""
import os
import struct
import csv
import numpy as np

URBAN, GREEN, SEA = 0, 1, 2


def write_domain(path, z, manning, infil, typ, dx):
    ny, nx = z.shape
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"MFLD")
        f.write(struct.pack("<iiid", 1, nx, ny, float(dx)))
        f.write(np.ascontiguousarray(z, dtype="<f8").tobytes())
        f.write(np.ascontiguousarray(manning, dtype="<f8").tobytes())
        f.write(np.ascontiguousarray(infil, dtype="<f8").tobytes())
        f.write(np.ascontiguousarray(typ, dtype=np.uint8).tobytes())


def read_domain(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"MFLD":
            raise ValueError(f"{path} is not a domain file")
        ver, nx, ny, dx = struct.unpack("<iiid", f.read(20))
        n = nx * ny
        z = np.frombuffer(f.read(8 * n), "<f8").reshape(ny, nx)
        man = np.frombuffer(f.read(8 * n), "<f8").reshape(ny, nx)
        infil = np.frombuffer(f.read(8 * n), "<f8").reshape(ny, nx)
        typ = np.frombuffer(f.read(n), np.uint8).reshape(ny, nx)
    return dict(nx=nx, ny=ny, dx=dx, z=z, manning=man, infil=infil, type=typ)


def read_meta(outdir):
    meta = {}
    with open(os.path.join(outdir, "meta.txt")) as f:
        for line in f:
            k, v = line.split(maxsplit=1)
            meta[k] = v.strip()
    for k in ("nx", "ny", "frames"):
        meta[k] = int(meta[k])
    for k in ("dx", "hours", "every_min"):
        meta[k] = float(meta[k])
    return meta


def read_frame(outdir, i, nx, ny):
    return np.fromfile(os.path.join(outdir, f"frame_{i:04d}.bin"), np.float32).reshape(ny, nx)


def read_frames_csv(outdir):
    rows = []
    with open(os.path.join(outdir, "frames.csv")) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def read_rain(path):
    hours, mm = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split(",")
            try:
                hours.append(int(parts[0]))
                mm.append(float(parts[-1]))
            except ValueError:
                continue
    return np.array(hours), np.array(mm)


def read_hotspots(path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out.append((r["name"], float(r["x_km"]), float(r["y_km"])))
    return out


def km_to_cell(x_km, y_km, ny, dx):
    """x from the west edge, y from the south edge -> (row, col). Row 0 is north."""
    col = int(x_km * 1000.0 / dx)
    row = int(ny - 1 - y_km * 1000.0 / dx)
    return row, col
