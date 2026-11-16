#!/usr/bin/env python3
"""
check_same.py - compares the final water depths of two runs.

The parallel versions do exactly the same arithmetic per cell as the serial one,
so the answer should be identical, not just close. Exit code 1 if not.

usage: python scripts/check_same.py runs/serial/final_h.bin runs/mpi-p4/final_h.bin
Rehan Mehta (241080039), Bhavya Gothi (241080021)
"""
import sys
import numpy as np

if len(sys.argv) not in (3, 4):
    print(__doc__)
    sys.exit(2)

a = np.fromfile(sys.argv[1], np.float64)
b = np.fromfile(sys.argv[2], np.float64)
if a.size != b.size:
    print(f"DIFFERENT SIZE: {a.size} vs {b.size}")
    sys.exit(1)

diff = np.abs(a - b)
worst = diff.max()
n_bad = int((diff > 0).sum())
if worst == 0:
    print(f"identical ({a.size} cells)")
else:
    i = int(diff.argmax())
    print(f"max difference {worst:.3e} m at cell {i}, {n_bad} cells differ")
sys.exit(0 if worst <= float(sys.argv[3] if len(sys.argv) > 3 else 0) else 1)
