#!/usr/bin/env python3
r"""
Fix non-uniform log10 T spacing in the Gnat-Sternberg cooling tables.

The original tables (gnat-sternberg.cooling_*Z) have two irregular gaps:
  - dlogT = 0.03 at logT 4.98 -> 5.01  (row 49 -> 50)
  - dlogT = 0.01 at logT 5.99 -> 6.00  (row 99 -> 100)

Rows 0-49 and 100-200 are already uniformly spaced at dlogT = 0.02.
Rows 50-99 are offset by +0.01 from the uniform grid.

This script corrects rows 50-99 by:
  - replacing logT with the correct uniform-grid values
  - interpolating Lambda at the corrected logT from the tab13 source data
Rows 0-49 and 100-200 are preserved exactly (original values, no interpolation).

Usage
-----
    python scripts/fix_cooling_table_spacing.py

Dependencies: numpy, scipy
"""

import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline

# ---------------------------------------------------------------------------
# Column mapping in tab13.txt
# ---------------------------------------------------------------------------
METALLICITY_COLS = {
    "0.001": 0,
    "0.01": 1,
    "0.1": 2,
    "1": 3,
    "2": 4,
}

# Uniform grid parameters
LOG_T_START = 4.00
D_LOG_T = 0.02

# Row range that needs fixing (0-indexed, inclusive)
FIX_ROW_START = 50
FIX_ROW_END = 99


def read_tab13(tab13_path):
    """Read tab13.txt and return (log_T, log_lam_array)."""
    data = np.loadtxt(tab13_path, skiprows=22)
    T_lin = data[:, 0]
    lam_cgs = data[:, 1:]  # columns: Z=0.001, 0.01, 0.1, 1, 2
    log_T = np.log10(T_lin)
    log_lam = np.log10(lam_cgs)
    return log_T, log_lam


def read_cooling_table(filepath):
    """Read a cooling table, returning (header_lines, log_T_array, log_lam_array)."""
    header_lines = []
    log_T_list = []
    log_lam_list = []
    with open(filepath) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                header_lines.append(line.rstrip("\n"))
                continue
            parts = s.split()
            if len(parts) == 2:
                log_T_list.append(float(parts[0]))
                log_lam_list.append(float(parts[1]))
    return header_lines, np.array(log_T_list), np.array(log_lam_list)


def write_cooling_table(filepath, header_lines, log_T, log_lam):
    """Write a cooling table preserving the original header format."""
    with open(filepath, "w") as f:
        for hline in header_lines:
            f.write(hline + "\n")
        for lt, ll in zip(log_T, log_lam):
            f.write(f"{lt:.2f} {ll:.4f}\n")


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tab13_path = os.path.join(repo_root, "inputs", "cooling_tables", "tab13.txt")
    tables_dir = os.path.join(repo_root, "inputs", "cooling_tables")

    if not os.path.isfile(tab13_path):
        print(f"ERROR: tab13.txt not found at {tab13_path}", file=sys.stderr)
        sys.exit(1)

    # Read tab13 source data
    log_T_src, log_lam_src = read_tab13(tab13_path)
    print(f"Read {len(log_T_src)} data points from tab13.txt")

    for Z_label, col_idx in METALLICITY_COLS.items():
        table_path = os.path.join(tables_dir, f"gnat-sternberg.cooling_{Z_label}Z")
        if not os.path.isfile(table_path):
            print(f"SKIP: {table_path} not found", file=sys.stderr)
            continue

        header_lines, log_T_orig, log_lam_orig = read_cooling_table(table_path)
        n_rows = len(log_T_orig)
        print(f"\nProcessing Z={Z_label}: {n_rows} rows")

        # Verify the anomalies
        d_orig = np.diff(log_T_orig)
        for i, d in enumerate(d_orig):
            if abs(d - D_LOG_T) > 1e-6:
                print(f"  Anomaly: row {i}->{i+1}  logT={log_T_orig[i]:.4f}->{log_T_orig[i+1]:.4f}  dlogT={d:.4f}")

        # Build cubic spline interpolant from tab13: log_T -> log_Lambda
        cs = CubicSpline(log_T_src, log_lam_src[:, col_idx], extrapolate=False)

        # Create corrected arrays (start as copies)
        log_T_fixed = log_T_orig.copy()
        log_lam_fixed = log_lam_orig.copy()

        # Fix rows 50-99
        n_changed = 0
        max_diff = 0.0
        for i in range(FIX_ROW_START, min(FIX_ROW_END + 1, n_rows)):
            corrected_logT = LOG_T_START + i * D_LOG_T
            if abs(log_T_orig[i] - corrected_logT) < 1e-6:
                # Already correct, skip
                continue

            # Interpolate Lambda at the corrected logT
            corrected_logLam = float(cs(corrected_logT))

            diff = abs(corrected_logLam - log_lam_orig[i])
            if diff > max_diff:
                max_diff = diff

            log_T_fixed[i] = corrected_logT
            log_lam_fixed[i] = corrected_logLam
            n_changed += 1

        if n_changed == 0:
            print("  No rows needed fixing — table already uniform.")
            continue

        print(f"  Fixed {n_changed} rows (indices {FIX_ROW_START}-{FIX_ROW_END})")
        print(f"  Max |delta logLambda| in fixed region: {max_diff:.6f}")

        # Verify uniform spacing after fix
        d_fixed = np.diff(log_T_fixed)
        remaining_anomalies = np.sum(np.abs(d_fixed - D_LOG_T) > 1e-6)
        if remaining_anomalies > 0:
            print(f"  WARNING: {remaining_anomalies} spacing anomalies remain!", file=sys.stderr)
        else:
            print(f"  All {n_rows - 1} gaps now uniform (dlogT = {D_LOG_T})")

        # Write corrected table
        write_cooling_table(table_path, header_lines, log_T_fixed, log_lam_fixed)
        print(f"  Written: {table_path}")

    print("\nDone. All five metallicities processed.")


if __name__ == "__main__":
    main()
