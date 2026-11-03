#!/usr/bin/env python3
# This file was made in part with generative AI
"""Compute observed convergence orders for the linear MHD wave test."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

COLUMNS = {
    "||dq||": 4,
    "density": 5,
    "M1": 6,
    "M2": 7,
    "M3": 8,
    "E": 9,
    "B1c": 10,
    "B2c": 11,
    "B3c": 12,
}

WAVE_FAMILIES = {
    0: "left-going fast magnetosonic wave",
    1: "left-going Alfven wave",
    2: "left-going slow magnetosonic wave",
    3: "entropy/contact wave",
    4: "right-going slow magnetosonic wave",
    5: "right-going Alfven wave",
    6: "right-going fast magnetosonic wave",
}


def default_error_file() -> Path:
    here = Path(__file__).resolve().parent
    preferred = here / "linearwave-errors.dat"
    return preferred


def wave_label(wave_family: int) -> str:
    return WAVE_FAMILIES[wave_family]


def infer_wave_family(error_file: Path) -> int:
    match = re.search(r"linearwave-errors-(\d+)(?:\D|$)", error_file.name)
    if match is None:
        valid = ", ".join(str(flag) for flag in WAVE_FAMILIES)
        raise SystemExit(
            f"Could not infer wave family from '{error_file.name}'. "
            f"Use a filename like linearwave-errors-0.dat or pass a wave family ({valid})."
        )
    wave_family = int(match.group(1))
    if wave_family not in WAVE_FAMILIES:
        valid = ", ".join(str(flag) for flag in WAVE_FAMILIES)
        raise SystemExit(f"Unknown wave family {wave_family}. Valid choices: {valid}")
    return wave_family


def read_error_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            try:
                values = [float(p) for p in parts]
            except ValueError:
                continue
            if len(values) >= 19:
                rows.append(values)
    rows.sort(key=lambda row: row[0])
    return rows


def observed_order(n0: float, e0: float, n1: float, e1: float) -> float:
    if e0 <= 0.0 or e1 <= 0.0 or n0 <= 0.0 or n1 <= n0:
        return float("nan")
    return math.log(e0 / e1) / math.log(n1 / n0)


def resolution(row: list[float]) -> float:
    """Use transverse N for oblique 2D/3D waves and nx1 for a 1D wave."""
    return row[1] if row[1] > 1 else row[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute observed orders from linearwave-errors.dat."
    )
    parser.add_argument(
        "error_file",
        nargs="?",
        type=Path,
        default=default_error_file(),
        help="Path to the linear MHD wave error .dat file.",
    )
    parser.add_argument(
        "wave_family",
        nargs="?",
        type=int,
        choices=sorted(WAVE_FAMILIES),
        help="MHD wave family number, 0 through 6. Overrides filename inference.",
    )
    args = parser.parse_args()
    wave_family = (
        args.wave_family
        if args.wave_family is not None
        else infer_wave_family(args.error_file)
    )

    rows = read_error_rows(args.error_file)
    if len(rows) < 2:
        raise SystemExit(f"Need at least two data rows in {args.error_file}")
    if len({resolution(row) for row in rows}) != len(rows):
        raise SystemExit("Duplicate resolutions: isolate or clear appended error rows.")

    lines: list[str] = []
    lines.append(f"Wave family {wave_family}: {wave_label(wave_family)}")
    lines.append(f"Reading: {args.error_file}")
    lines.append("")
    header = f"{'N0->N1':>12}  " + "  ".join(f"{name:>10}" for name in COLUMNS)
    lines.append(header)
    lines.append("-" * len(header))
    for coarse, fine in zip(rows[:-1], rows[1:]):
        n0 = resolution(coarse)
        n1 = resolution(fine)
        orders = [
            observed_order(n0, coarse[col], n1, fine[col]) for col in COLUMNS.values()
        ]
        line = f"{int(n0):5d}->{int(n1):<5d}  "
        line += "  ".join(
            f"{p:10.4f}" if math.isfinite(p) else f"{'nan':>10}" for p in orders
        )
        lines.append(line)

    lines.append("")
    lines.append("Errors used:")
    err_header = f"{'N':>8}  " + "  ".join(f"{name:>12}" for name in COLUMNS)
    lines.append(err_header)
    lines.append("-" * len(err_header))
    for row in rows:
        line = f"{int(resolution(row)):8d}  "
        line += "  ".join(f"{row[col]:12.6e}" for col in COLUMNS.values())
        lines.append(line)

    report = "\n".join(lines)
    print(report)


if __name__ == "__main__":
    main()
