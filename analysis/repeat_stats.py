#!/usr/bin/env python3
"""Mean/std of the step time over repeats: repeat_stats.py ROOT  (ROOT/rep*/<label>/athenapk.log)."""
import glob
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(__file__))
from summarize import parse_log

root = sys.argv[1]
warmup = 20
runs = {}
for d in sorted(glob.glob(f"{root}/rep*/*")):
    step, _ = parse_log(os.path.join(d, "athenapk.log"), warmup)
    if step is not None:
        runs.setdefault(os.path.basename(d), []).append(step * 1e3)
print(f"{'case':16s} {'n':>2s} {'mean ms':>9s} {'std ms':>8s}  values")
for k, v in runs.items():
    sd = st.stdev(v) if len(v) > 1 else 0.0
    print(f"{k:16s} {len(v):2d} {st.mean(v):9.2f} {sd:8.2f}  {' '.join(f'{x:.2f}' for x in v)}")
