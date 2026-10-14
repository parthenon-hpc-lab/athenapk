#!/usr/bin/env python3
"""Summarize stars benchmark runs: step time, stars/swarm/hydro kernel shares, top kernels.

Usage: summarize.py [--warmup 20] [--plot] RUN_DIR [RUN_DIR ...]
A run dir holds athenapk.log (Parthenon stdout) and kernel_times.txt (kp_reader output).
"""
import argparse
import os
import re

STARS_PREFIX = ("StellarFeedback::", "InjectParticles::", "MoveStars::")


def parse_log(path, warmup):
    """Mean wsec_step and zone-cycles/s over cycles > warmup, from the per-cycle lines."""
    steps, zcs = [], []
    pat = re.compile(r"cycle=(\d+).*zone-cycles/wsec_step=(\S+).*wsec_step=(\S+)")
    with open(path, errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m and int(m.group(1)) > warmup:
                zcs.append(float(m.group(2)))
                steps.append(float(m.group(3)))
    if not steps:
        return None, None
    return sum(steps) / len(steps), sum(zcs) / len(zcs)


def parse_kernels(path):
    """Return (regions, kernels) dicts name -> (total_time, calls) from kp_reader output."""
    sections = {"Regions": {}, "Kernels": {}}
    cur, name = None, None
    with open(path, errors="replace") as f:
        for line in f:
            s = line.strip()
            if s in ("Regions:", "Kernels:"):
                cur = s[:-1]
            elif s.startswith("Summary"):
                cur = None
            elif cur and s.startswith("- "):
                name = s[2:].strip()
            elif cur and name and re.search(r"\((ParFor|ParRed|ParScan|Region|REGION)\)", s):
                nums = s.split()[-5:]
                t0, c0 = sections[cur].get(name, (0.0, 0))
                sections[cur][name] = (t0 + float(nums[0]), c0 + int(nums[1]))
                name = None
    return sections["Regions"], sections["Kernels"]


def classify(name):
    if name.startswith(STARS_PREFIX) or name == "LocalReduceStarFormationRate":
        return "stars"
    if "Swarm" in name:
        return "swarm"
    return "hydro"


def summarize(run, warmup, ntop):
    log = os.path.join(run, "athenapk.log")
    kt = os.path.join(run, "kernel_times.txt")
    if not os.path.exists(kt) or os.path.getsize(kt) == 0:
        kt = log  # timer ASCII reports (one per rank) are printed into the log
    step, zcs = parse_log(log, warmup) if os.path.exists(log) else (None, None)
    res = {"run": run, "step": step, "zcs": zcs}
    if os.path.exists(kt):
        regions, kernels = parse_kernels(kt)
        tot = {"stars": 0.0, "swarm": 0.0, "hydro": 0.0}
        for n, (t, _) in kernels.items():
            tot[classify(n)] += t
        allt = sum(tot.values()) or 1.0
        res["share"] = {k: 100 * v / allt for k, v in tot.items()}
        res["top"] = sorted(kernels.items(), key=lambda kv: -kv[1][0])[:ntop]
        res["regions"] = {n: v for n, v in regions.items() if n.split("::")[0] in
                          ("Stars", "StellarFeedback", "ParticlesUtils") or "Stars" in n
                          or "StellarFeedback" in n or "InjectParticles" in n}
    return res


def show(r, ntop):
    step = "n/a" if r["step"] is None else f"{r['step']:.4g} s"
    zcs = "n/a" if r["zcs"] is None else f"{r['zcs']:.4g}"
    print(f"== {r['run']}\n   step time {step}   zone-cycles/s {zcs}")
    if "share" in r:
        sh = r["share"]
        print(f"   kernel time share: stars {sh['stars']:.2f}%  swarm comm {sh['swarm']:.2f}%"
              f"  hydro {sh['hydro']:.2f}%")
        print(f"   top {ntop} kernels (total s summed over ranks, calls, us/call):")
        for n, (t, c) in r["top"]:
            print(f"     {t:10.3f} {c:9d} {1e6 * t / max(c, 1):10.1f}  {n[:90]}")
        for n, (t, c) in r["regions"].items():
            print(f"     [region] {t:10.3f} {c:9d}  {n[:80]}")


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = []
    for r in results:
        m = re.search(r"B_N(\d+)$", r["run"].rstrip("/"))
        if m and "share" in r:
            b.append((int(m.group(1)) * 64, r["share"]["stars"], r["share"]["swarm"]))
    if b:
        b.sort()
        fig, ax = plt.subplots()
        ax.plot([x[0] for x in b], [x[1] for x in b], "o-", label="stars kernels")
        ax.plot([x[0] for x in b], [x[2] for x in b], "s-", label="swarm comm")
        ax.set_xscale("log")
        ax.set_xlabel("N_star")
        ax.set_ylabel("share of kernel time [%]")
        ax.legend()
        fig.savefig("stars_share_vs_N.png", dpi=150)
    d = {"on": [], "off": []}
    for r in results:
        m = re.search(r"D_strong_n(\d+)_(on|off)$", r["run"].rstrip("/"))
        if m and r["step"]:
            d[m.group(2)].append((int(m.group(1)), r["step"]))
    if d["on"] or d["off"]:
        fig, ax = plt.subplots()
        for k, v in d.items():
            v.sort()
            ax.plot([x[0] for x in v], [x[1] for x in v], "o-", label=f"stars {k}")
        ax.set_xlabel("nodes (4 GPUs each)")
        ax.set_ylabel("time per cycle [s]")
        ax.legend()
        fig.savefig("step_time_vs_nodes.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    results = [summarize(r, a.warmup, a.top) for r in a.runs]
    for r in results:
        show(r, a.top)
    if a.plot:
        plot(results)
