#!/usr/bin/env python3
r"""
Generate uniformly-spaced (in log10 T) Gnat-Sternberg cooling tables
from the original tab13.txt data.

Background
----------
The original cooling tables shipped with AthenaPK (in
``inputs/cooling_tables/gnat-sternberg.cooling_*Z``) are formatted from the
Gnat & Sternberg (2007) CIE cooling data.  However, the original tab13.txt
data has **non-uniform log10 T spacing**: :math:`\Delta \log T` alternates
between 0.01, 0.02, and 0.03.  The ``TabularCooling`` implementation
(``src/hydro/srcterms/tabular_cooling.cpp``) requires equally spaced tables
for the ``tabular`` integrator — the direct-indexing lookup assumes constant
:math:`\Delta \log T`.

This script creates new tables with uniform :math:`\Delta \log T = 0.02`
using cubic spline interpolation (``scipy.interpolate.CubicSpline``) on
the original tab13 data.  The output tables contain:

* log10 T in [4.00, 8.00] with :math:`\Delta \log T = 0.02` (201 points)
* log10 :math:`\Lambda_N(T)` in [erg cm\ :sup:`3` / s]

and can be used with all cooling integrator types (townsend, tabular,
subcycle).

Usage
-----
.. code-block:: bash

    python scripts/generate_uniform_cooling_table.py

The script expects ``inputs/cooling_tables/tab13.txt`` in the repository
root.  If the file is not available, download it from
http://wise-obs.tau.ac.il/~orlyg/cooling/CIEcool/tab13.txt

Output files are written to ``inputs/cooling_tables/`` with the naming
pattern ``gnat-sternberg.cooling_uniform_<Z>Z``, where ``<Z>`` is one of
0.001, 0.01, 0.1, 1, 2.

References
----------
* Gnat, O. & Sternberg, A. 2007, ApJS, 168, 213
* http://wise-obs.tau.ac.il/~orlyg/cooling/CIEcool/tab13.txt

Dependencies
------------
* numpy
* scipy
"""

import argparse
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline

# ---------------------------------------------------------------------------
# Column mapping in tab13.txt: T[K] Lam(Z=0.001) ... Lam(Z=1) Lam(Z=2)
# ---------------------------------------------------------------------------
METALLICITY_COLS = {
    "0.001": 0,
    "0.01": 1,
    "0.1": 2,
    "1": 3,
    "2": 4,
}

# Target uniform log-T grid
LOG_T_START = 4.00
LOG_T_END = 8.00
D_LOG_T = 0.02

# Header written into every output file
HEADER_LINES = [
    "# Adapted from: http://wise-obs.tau.ac.il/~orlyg/cooling/CIEcool/tab13.txt",
    "# Title: Time-Dependent Ionization in Radiatively Cooling Gas",
    "# Authors: Orly Gnat and Amiel Sternberg",
    "# Table: CIE Cooling Efficiencies",
    "# -------------------------------------------------------------------------",
    "# Our assumed Z=1 solar abundances are listed in Table 1.",
    "# -------------------------------------------------------------------------",
    "# Interpolated to uniform log10 T spacing (d_log_T = 0.02) using cubic spline.",
    "# Original table from tab13.txt had non-uniform temperature spacing.",
    "# -------------------------------------------------------------------------",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate uniformly-spaced Gnat-Sternberg cooling tables"
    )
    parser.add_argument(
        "--tab13",
        default=None,
        help="Path to tab13.txt (default: <repo_root>/inputs/cooling_tables/tab13.txt)",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory (default: <repo_root>/inputs/cooling_tables/)",
    )
    args = parser.parse_args()

    # Locate the repository root (assume this script lives in scripts/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    tab13_path = args.tab13
    if tab13_path is None:
        tab13_path = os.path.join(repo_root, "inputs", "cooling_tables", "tab13.txt")

    outdir = args.outdir
    if outdir is None:
        outdir = os.path.join(repo_root, "inputs", "cooling_tables")

    # -----------------------------------------------------------------------
    # 1. Read tab13.txt
    # -----------------------------------------------------------------------
    # The file has 22 header lines, then data rows:
    #   T[K]  Lam(Z=0.001)  Lam(Z=0.01)  Lam(Z=0.1)  Lam(Z=1)  Lam(Z=2)
    # Lambda values are already in erg cm^3/s.
    if not os.path.isfile(tab13_path):
        print(
            f"ERROR: tab13.txt not found at '{tab13_path}'.\n"
            "Download it from:\n"
            "  http://wise-obs.tau.ac.il/~orlyg/cooling/CIEcool/tab13.txt\n"
            "Or place it at <repo_root>/inputs/cooling_tables/tab13.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    data = np.loadtxt(tab13_path, skiprows=22)

    T_lin = data[:, 0]  # Temperature in K
    lam_cgs = data[:, 1:]  # Lambda in erg cm^3/s [Z=0.001, 0.01, 0.1, 1, 2]

    log_T_orig = np.log10(T_lin)
    log_lam_orig = np.log10(lam_cgs)

    print(f"Read {len(log_T_orig)} data points from {tab13_path}")
    print(f"  T range:  [{T_lin[0]:.1f}, {T_lin[-1]:.1f}] K")
    print(f"  log T range:  [{log_T_orig[0]:.2f}, {log_T_orig[-1]:.2f}]")

    # Report original spacing characteristics
    d_orig = np.diff(log_T_orig)
    unique_d = np.unique(np.round(d_orig, 6))
    print(f"  Original spacing values: {unique_d}")

    # -----------------------------------------------------------------------
    # 2. Create uniform log_T grid
    # -----------------------------------------------------------------------
    num_pts = int(round((LOG_T_END - LOG_T_START) / D_LOG_T)) + 1
    log_T_uniform = np.linspace(LOG_T_START, LOG_T_END, num_pts)

    print(f"\nUniform grid:")
    print(f"  log_T = [{log_T_uniform[0]:.2f}, {log_T_uniform[-1]:.2f}]")
    print(f"  Number of points: {len(log_T_uniform)}")
    print(f"  d_log_T = {log_T_uniform[1] - log_T_uniform[0]:.4f}")

    # -----------------------------------------------------------------------
    # 3. Interpolate & write output files
    # -----------------------------------------------------------------------
    os.makedirs(outdir, exist_ok=True)

    for Z_label, col_idx in METALLICITY_COLS.items():
        # Cubic spline on log_T -> log_Lambda
        cs = CubicSpline(log_T_orig, log_lam_orig[:, col_idx], extrapolate=False)
        log_lam_uniform = cs(log_T_uniform)

        if np.any(np.isnan(log_lam_uniform)):
            print(f"WARNING: NaN values in Z={Z_label} interpolation!", file=sys.stderr)

        out_filename = os.path.join(
            outdir, f"gnat-sternberg.cooling_uniform_{Z_label}Z"
        )
        with open(out_filename, "w") as f:
            for line in HEADER_LINES:
                f.write(line + "\n")
            col_header = f"# log10 T [K] Z={Z_label} log10 Lambda_N [erg cm^3/s]\n"
            f.write(col_header)
            for lt, ll in zip(log_T_uniform, log_lam_uniform):
                f.write(f"{lt:.2f} {ll:.4f}\n")

        print(f"  Written: {out_filename}  ({len(log_T_uniform)} rows)")

    print("\nDone.  All five metallicities (0.001, 0.01, 0.1, 1, 2 Z_solar) generated.")
    print("Use them in AthenaPK input files via, e.g.:")
    print(f"  <cooling>")
    print(f"  table = {outdir}/gnat-sternberg.cooling_uniform_0.1Z")
    print()


if __name__ == "__main__":
    main()
