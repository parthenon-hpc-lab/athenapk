#!/usr/bin/env python3
"""Compare the last row of the *.hst files of two runs: max relative difference per column."""

import glob
import sys

import numpy as np


def last_row(d):
    f = sorted(glob.glob(f"{d}/*.hst"))[0]
    return np.loadtxt(f, comments="#")[-1]


a, b = last_row(sys.argv[1]), last_row(sys.argv[2])
rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-300)
for i, r in enumerate(rel):
    print(f"col {i:2d}  {a[i]: .12e}  {b[i]: .12e}  rel {r:.2e}")
print("max rel diff:", rel.max())
