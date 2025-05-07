import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.insert(
    1,
    "./external/parthenon" + "/scripts/python/packages/parthenon_tools/parthenon_tools",
)

try:
    import phdf
except ModuleNotFoundError:
    print("Couldn't find module to read Parthenon hdf5 files.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs="*")
    args = parser.parse_args()

    for filename in args.filenames:
        data = phdf.phdf(filename)
        tracers = data.GetSwarm("tracers")
        xs = tracers.x
        ys = tracers.y
        zs = tracers.z
        ids = tracers.Get("id")

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax.scatter(xs, ys, zs, s=1, c=ids)

        filepath = Path(filename)
        figfile = str(filepath.stem) + ".png"
        print(f"saving to {figfile}...")
        plt.savefig(figfile)
        plt.close()
