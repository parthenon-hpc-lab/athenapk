#!/usr/bin/env python3

from pathlib import Path
import argparse

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np


KPC_IN_CM = 3.0856775814913673e21


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot z versus g(z) from an HSE profile table."
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="inputs/hse_3.0.txt",
        help="Input HSE table. Defaults to inputs/hse_3.0.txt.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image path. Defaults to <input-stem>_g_of_z.png.",
    )
    parser.add_argument(
        "--z-unit",
        choices=("cm", "kpc"),
        default="cm",
        help="Unit to use for the z axis. Defaults to cm.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.filename)
    output_path = (
        Path(args.output)
        if args.output is not None
        else input_path.with_name(f"{input_path.stem}_g_of_z.png")
    )

    data = np.loadtxt(input_path)
    z = data[:, 0]
    g_z = data[:, 3]

    if args.z_unit == "kpc":
        z = z / KPC_IN_CM
        z_label = "z [kpc]"
    else:
        z_label = "z [cm]"

    fig, ax = plt.subplots()
    ax.plot(z, g_z)
    ax.set_xlabel(z_label)
    ax.set_ylabel("g(z) [cm s$^{-2}$]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    print(f"saving to {output_path}...")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
