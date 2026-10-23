"""Summarize self-gravity scaling runs from their athenapk.log files.

Usage: python3 sg_scaling_summary.py RUN_DIR [RUN_DIR ...]

Per run: number of ranks and cells, median wall time per cycle and zone-cycles per second
(skipping the first 10 cycles, which include setup and the first Poisson solve), and the
throughput per GPU. If the runs printed solver residuals, the iterations per solve too.
"""

import os
import re
import statistics
import sys

CYCLE = re.compile(r"^cycle=(\d+) .*zone-cycles/wsec_step=(\S+) .* wsec_step=(\S+)$")
RANKS = re.compile(r"Number of MPI ranks\s*=?\s*(\d+)", re.I)
SKIP = 10


def summarize(run_dir):
    text = open(os.path.join(run_dir, "athenapk.log")).read()
    rows = [m.groups() for m in map(CYCLE.match, text.splitlines()) if m]
    rows = [(int(c), float(z), float(w)) for c, z, w in rows if int(c) > SKIP]
    if not rows:
        return f"{run_dir}: no cycles after the first {SKIP} (check athenapk.log)"
    zcs = statistics.median(z for _, z, _ in rows)
    spc = statistics.median(w for _, _, w in rows)
    m = RANKS.search(text)
    ranks = int(m.group(1)) if m else int(os.environ.get("SLURM_NTASKS", "0")) or None
    per_gpu = f"{zcs / ranks:9.3e}" if ranks else "        ?"
    return (
        f"{os.path.basename(run_dir):6s} ranks={ranks}  cells={zcs * spc:9.3e}  "
        f"s/cycle={spc:8.4f}  zone-cycles/s={zcs:9.3e}  per GPU={per_gpu}"
    )


for d in sys.argv[1:]:
    print(summarize(d))
if len(sys.argv) == 3:
    t = [
        statistics.median(
            float(m.group(3))
            for m in map(
                CYCLE.match, open(os.path.join(d, "athenapk.log")).read().splitlines()
            )
            if m and int(m.group(1)) > SKIP
        )
        for d in sys.argv[1:]
    ]
    print(
        f"cost ratio {os.path.basename(sys.argv[2])}/{os.path.basename(sys.argv[1])} "
        f"= {t[1] / t[0]:.2f}x"
    )
