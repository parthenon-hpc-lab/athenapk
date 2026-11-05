"""Builds the particle distribution notebook (run with an env that has nbformat).

Usage: python make_notebook.py [run_dir] [initial_velocity: gas|zero] [notebook name]
"""
import sys

import nbformat as nbf

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else "phase2_particles"
INIT_VEL = sys.argv[2] if len(sys.argv) > 2 else "gas"
OUT_NAME = sys.argv[3] if len(sys.argv) > 3 else "particle_distribution.ipynb"
assert INIT_VEL in ("gas", "zero")

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

nb.cells = [
    md(
        r"""# Inertial particles in subsonic, solenoidally driven turbulence

**Setup** (`turb_sol.in`, `run_phase2.sh`)
- $64^3$, periodic unit box, Euler (HLLC/PLM/VL2), $\gamma = 1.0001$ (near isothermal, $c_s \approx 1$)
- purely solenoidal OU forcing at $k \approx 2$, `accel_rms = 0.08`
- **phase 1** (gas only): $t = 0 \to 10$, saturation from $t \approx 5$ with
  $\sigma_v \approx 0.196$ ($\mathcal{M}_\mathrm{rms} \approx 0.19$),
  large eddy time $T_L = L/(2\sigma_v) \approx 2.56$
- **phase 2**: restart at $t = 10$ and seed $2^{18}$ particles per population uniformly at random
  (initial velocity = local gas velocity), Stokes drag, one-way coupling, run for $5\,T_L$
  - $\mathrm{St}_L = \tau_p / T_L = 10^{-3}$ (tracer-like)
  - $\mathrm{St}_L = 100$ (ballistic)

**Expectation:** both populations stay uniformly distributed (Poisson statistics).
Small-St particles act as tracers in a nearly divergence-free flow, which preserves
volume (only $\delta\rho/\rho \sim \mathcal{M}^2 \sim 1\%$ modulation). Large-St particles barely
react to the flow over $5\,T_L$ and move quasi-ballistically.
Strong clustering would require $\mathrm{St}_\eta \sim 1$ (preferential concentration) or
strongly compressive flow (shock trapping, as in Burgers turbulence)."""
    ),
    code(
        """import glob
import sys

import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

sys.path.insert(
    1,
    "../../external/parthenon/scripts/python/packages/"
    "parthenon_tools/parthenon_tools",
)
import phdf

N = 64  # cells per dimension, box [0, 1]^3
# stopping-time unit (= T_L used for this run) and seeding time, read from the first snapshot
with h5py.File(sorted(glob.glob(f"phase2_particles/parthenon.parts.*.phdf"))[0], "r") as f0:
    T_L = float(f0["Params"].attrs["tracers/stokes_time_unit"])
    T_SEED = float(f0["Info"].attrs["Time"])
print(f"T_L = {T_L:g}, particles seeded at t = {T_SEED:g}")

# Categorical palette (fixed order). Identity is also given by legends/labels.
C_SMALL, C_LARGE, C_REF = "#2a78d6", "#eb6834", "#0b0b0b"
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e4e3df",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#8a8984",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "lines.linewidth": 2,
        "legend.frameon": False,
    }
)"""
    ),
    code(
        """def load_gas(d):
    \"\"\"Assembles prim (rho, vx, vy, vz, p) into global arrays indexed [var, z, y, x].\"\"\"
    p = d.Get("prim", flatten=False).astype(np.float64)
    nb = p.shape[-1]
    out = np.zeros((p.shape[1], N, N, N))
    for b, bb in enumerate(d.BlockBounds):
        i0, j0, k0 = (int(round((bb[2 * a] + 1e-8) * N)) for a in range(3))
        out[:, k0 : k0 + nb, j0 : j0 + nb, i0 : i0 + nb] = p[b]
    return out


def load_snapshot(fname):
    d = phdf.phdf(fname)
    s = d.GetSwarm("tracers")
    tau = s.Get("tau_p").astype(np.float64)
    pos = np.stack([s.x, s.y, s.z], axis=-1).astype(np.float64)
    vel = np.stack([s.Get(v) for v in ("vp_x", "vp_y", "vp_z")], axis=-1).astype(np.float64)
    taus = np.unique(tau)
    parts = {t: {"pos": pos[tau == t], "vel": vel[tau == t], "rho": s.Get("rho")[tau == t]}
             for t in taus}
    return {"t": d.Time, "gas": load_gas(d), "parts": parts}


files = sorted(glob.glob("phase2_particles/parthenon.parts.*.phdf"))
snaps = [load_snapshot(f) for f in files]
TAUS = sorted(snaps[0]["parts"].keys())
LABEL = {TAUS[0]: rf"St$_L$ = {TAUS[0] / T_L:.0e}", TAUS[1]: rf"St$_L$ = {TAUS[1] / T_L:.0f}"}
COLOR = {TAUS[0]: C_SMALL, TAUS[1]: C_LARGE}
times = np.array([s["t"] for s in snaps])
print(f"{len(snaps)} snapshots at t = {np.round(times, 2)}")
print({LABEL[t]: len(snaps[0]['parts'][t]['pos']) for t in TAUS})"""
    ),
    md("## 0. Gas: turbulence saturation and seeding time"),
    code(
        """h1 = np.loadtxt("phase1_gas/parthenon.out1.hst")
h2 = np.loadtxt("phase2_particles/parthenon.out1.hst")
h = np.concatenate([h1, h2[h2[:, 0] > h1[-1, 0]]])
sigma_v = np.sqrt(2 * h[:, 8] / h[:, 4])

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(h[:, 0], sigma_v, color=C_REF, lw=1.5, label=r"$\\sigma_v = \\sqrt{2\\,E_\\mathrm{kin}/M}$")
ax.plot(h[:, 0], h[:, 10], color="#8a8984", lw=1.5, ls="--", label=r"$\\langle |v|/c_s \\rangle$")
ax.axvline(T_SEED, color=C_SMALL, lw=1)
ax.text(T_SEED + 0.2, 0.02, "particles seeded", color="#52514e", fontsize=9)
ax.set_xlabel("t")
ax.set_ylim(0, None)
ax.legend(loc="lower right")
fig.tight_layout()
print(f"sigma_v during phase 2: {sigma_v[h[:, 0] >= T_SEED].mean():.3f}")"""
    ),
    md(
        r"""## 1. Particle positions in a thin slab

Particles with $|z - 0.5| < \Delta x/2$ (one cell thick, $\approx 4096$ per population),
over the gas vorticity magnitude $|\boldsymbol\omega|$ in that cell layer.
Top: at seeding; bottom: after $5\,T_L$."""
    ),
    code(
        """def vorticity_mag(gas):
    dx = 1.0 / N
    vx, vy, vz = gas[1], gas[2], gas[3]  # arrays [z, y, x]

    def d(f, axis):
        return (np.roll(f, -1, axis) - np.roll(f, 1, axis)) / (2 * dx)

    wx = d(vz, 1) - d(vy, 0)
    wy = d(vx, 0) - d(vz, 2)
    wz = d(vy, 2) - d(vx, 1)
    return np.sqrt(wx**2 + wy**2 + wz**2)


z0, k0 = 0.5, N // 2
fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
for row, snap in zip(axes, (snaps[0], snaps[-1])):
    w = vorticity_mag(snap["gas"])[k0]
    vmax = np.percentile(vorticity_mag(snaps[-1]["gas"]), 99.5)
    for ax, tau in zip(row, TAUS):
        im = ax.imshow(w, origin="lower", extent=(0, 1, 0, 1), cmap="Greys", vmin=0,
                       vmax=vmax, interpolation="nearest")
        p = snap["parts"][tau]["pos"]
        m = np.abs(p[:, 2] - z0) < 0.5 / N
        ax.plot(p[m, 0], p[m, 1], ".", ms=1.6, color=COLOR[tau], mec="none")
        ax.set_title(f"{LABEL[tau]},  t - t$_s$ = {(snap['t'] - T_SEED) / T_L:.1f} $T_L$"
                     f"  ({m.sum()} particles)", loc="left", fontsize=10)
        ax.grid(False)
for ax in axes[-1]:
    ax.set_xlabel("x")
for ax in axes[:, 0]:
    ax.set_ylabel("y")
fig.colorbar(im, ax=axes, shrink=0.6, label=r"$|\\omega|$ (gas, slab $z = 0.5$)")"""
    ),
    md(
        r"""## 2. Coarse-grained number density vs. Poisson

Counts $n$ in non-overlapping cubes of side $\ell$; for a uniform random (Poisson) distribution
$P(n)$ is Poissonian with mean $\bar n = N_p \ell^3$. Shown after $5\,T_L$ for $\ell = 4\,\Delta x$
($16^3$ cubes, $\bar n = 64$)."""
    ),
    code(
        """def box_counts(pos, nbox):
    idx = np.floor(pos * nbox).astype(int) % nbox
    flat = (idx[:, 2] * nbox + idx[:, 1]) * nbox + idx[:, 0]
    return np.bincount(flat, minlength=nbox**3)


nbox = 16
fig, ax = plt.subplots(figsize=(7, 4))
for tau in TAUS:
    c = box_counts(snaps[-1]["parts"][tau]["pos"], nbox)
    bins = np.arange(c.min() - 0.5, c.max() + 1.5)
    ax.hist(c, bins=bins, density=True, histtype="step", lw=2, color=COLOR[tau],
            label=f"{LABEL[tau]}  (Var/mean = {c.var() / c.mean():.3f})")
nbar = c.mean()
n = np.arange(int(nbar - 5 * np.sqrt(nbar)), int(nbar + 5 * np.sqrt(nbar)))
ax.plot(n, stats.poisson.pmf(n, nbar), color=C_REF, lw=1.5, ls="--",
        label=f"Poisson, mean {nbar:.0f}  (Var/mean = 1)")
ax.set_xlabel(rf"particles per cube of side {N // nbox}$\\Delta x$")
ax.set_ylabel("P(n)")
ax.legend(fontsize=9)
fig.tight_layout()"""
    ),
    md(
        r"""## 3. Clustering vs. time

Normalized variance excess $\mathrm{Var}(n)/\bar n - 1$ (= 0 for Poisson) at three coarse-graining
scales. Shaded: $\pm 2\sigma$ sampling noise of a Poisson field, $\approx 2\sqrt{2/N_\mathrm{cubes}}$.
For tracers, gas density fluctuations alone would give $\approx \bar n\,\mathrm{Var}(\rho)/\bar\rho^2$ (dotted)."""
    ),
    code(
        """fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, nbox in zip(axes, (32, 16, 8)):
    for tau in TAUS:
        exc = []
        for s in snaps:
            c = box_counts(s["parts"][tau]["pos"], nbox)
            exc.append(c.var() / c.mean() - 1)
        ax.plot((times - T_SEED) / T_L, exc, "o-", ms=5, lw=1.5, color=COLOR[tau],
                label=LABEL[tau])
    noise = 2 * np.sqrt(2 / nbox**3)
    ax.axhspan(-noise, noise, color="#e4e3df", alpha=0.6, lw=0)
    # expected excess from gas density fluctuations (tracers follow rho)
    nbar = len(snaps[0]["parts"][TAUS[0]]["pos"]) / nbox**3
    rho_exp = []
    for s in snaps:
        r = s["gas"][0].reshape(nbox, N // nbox, nbox, N // nbox, nbox, N // nbox).mean(axis=(1, 3, 5))
        rho_exp.append(nbar * r.var() / r.mean() ** 2)
    ax.plot((times - T_SEED) / T_L, rho_exp, ":", color=C_REF, lw=1.5,
            label=r"$\\bar n\\,\\mathrm{Var}(\\rho)/\\bar\\rho^2$")
    ax.axhline(0, color="#8a8984", lw=0.8)
    ax.set_title(rf"cube side {N // nbox}$\\Delta x$,  $\\bar n$ = {nbar:.0f}", loc="left",
                 fontsize=10)
    ax.set_xlabel(r"$(t - t_s)/T_L$")
axes[0].set_ylabel(r"Var$(n)/\\bar n - 1$")
axes[0].legend(fontsize=9)
fig.tight_layout()"""
    ),
    md(
        r"""## 4. Pair correlation function

$g(r) = $ (number of pairs at separation $r$) / (expected number for a uniform distribution),
periodic box, $10^5$ randomly subsampled particles per population, after $5\,T_L$.
Clustering at scale $r$ would show up as $g(r) > 1$ (and a power law $g \sim r^{D_2 - 3}$
for fractal clusters). Shaded band: $\pm 2\sigma$ Poisson counting noise."""
    ),
    code(
        """rng = np.random.default_rng(1)
edges = np.logspace(np.log10(0.25 / N), np.log10(0.25), 22)
rc = np.sqrt(edges[1:] * edges[:-1])
fig, ax = plt.subplots(figsize=(7, 4))
nsub = 100_000
for tau in TAUS:
    p = snaps[-1]["parts"][tau]["pos"] % 1.0
    p = p[rng.choice(len(p), nsub, replace=False)]
    tree = cKDTree(p, boxsize=1.0)
    cum = tree.count_neighbors(tree, edges).astype(float) - nsub  # remove self pairs
    dd = np.diff(cum)
    expected = nsub * (nsub - 1) * 4 / 3 * np.pi * np.diff(edges**3)
    g = dd / expected
    ax.plot(rc * N, g, "o-", ms=5, lw=1.5, color=COLOR[tau], label=LABEL[tau])
ax.fill_between(rc * N, 1 - 2 / np.sqrt(expected / 2), 1 + 2 / np.sqrt(expected / 2),
                color="#e4e3df", alpha=0.7, lw=0, label=r"Poisson $\\pm 2\\sigma$")
ax.axhline(1, color=C_REF, lw=1, ls="--")
ax.set_xscale("log")
ax.set_xlabel(r"$r / \\Delta x$")
ax.set_ylabel("g(r)")
ax.legend(fontsize=9)
fig.tight_layout()"""
    ),
    md(
        r"""## 5. Preferential sampling of the flow

Gas quantities sampled at the particle positions (nearest cell), normalized by their volume
average. A value of 1 means the particles sample the flow uniformly.
Particles with $\mathrm{St}_\eta \sim 1$ in incompressible turbulence would show
$\langle\omega^2\rangle_p/\langle\omega^2\rangle_V < 1$ (expelled from vortices)."""
    ),
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for tau in TAUS:
    r_w, r_rho = [], []
    for s in snaps:
        w2 = vorticity_mag(s["gas"]) ** 2
        rho = s["gas"][0]
        idx = np.floor(s["parts"][tau]["pos"] * N).astype(int) % N
        r_w.append(w2[idx[:, 2], idx[:, 1], idx[:, 0]].mean() / w2.mean())
        r_rho.append(rho[idx[:, 2], idx[:, 1], idx[:, 0]].mean() / rho.mean())
    x = (times - T_SEED) / T_L
    axes[0].plot(x, r_w, "o-", ms=5, lw=1.5, color=COLOR[tau], label=LABEL[tau])
    axes[1].plot(x, np.array(r_rho) - 1, "o-", ms=5, lw=1.5, color=COLOR[tau],
                 label=LABEL[tau])
axes[0].axhline(1, color=C_REF, lw=1, ls="--")
axes[0].set_ylabel(r"$\\langle \\omega^2 \\rangle_p / \\langle \\omega^2 \\rangle_V$")
axes[1].axhline(0, color=C_REF, lw=1, ls="--")
axes[1].set_ylabel(r"$\\langle \\rho \\rangle_p / \\langle \\rho \\rangle_V - 1$")
for ax in axes:
    ax.set_xlabel(r"$(t - t_s)/T_L$")
axes[0].legend(fontsize=9)
fig.tight_layout()"""
    ),
    md(
        r"""## 6. Particle velocities

rms particle speed relative to the gas $\sigma_v$. Small-St particles follow the gas;
large-St particles keep the velocity they were seeded with and slowly decay,
$v_\mathrm{rms}(t) \approx v_\mathrm{rms}(t_s)\,e^{-(t-t_s)/\tau_p}$ (dashed)."""
        if INIT_VEL == "gas"
        else r"""## 6. Particle velocities

rms particle speed relative to the gas $\sigma_v$ (log scale). Particles start at rest.
Small-St particles pick up the gas velocity within $\tau_p \ll$ snapshot spacing;
large-St particles are only slowly accelerated by drag. For $t - t_s \ll \tau_p$,
$\mathbf{v}_p \approx \tau_p^{-1}\int \mathbf{u}\,dt$, which grows $\propto t$ while
$t - t_s$ is shorter than the Lagrangian correlation time and $\propto \sqrt{t}$ after."""
    ),
    code(
        """fig, ax = plt.subplots(figsize=(7, 3.8))
sig = np.interp(times, h[:, 0], sigma_v)
x = (times - T_SEED) / T_L
for tau in TAUS:
    vrms = np.array([np.sqrt((s["parts"][tau]["vel"] ** 2).sum(axis=1).mean()) for s in snaps])
    pos = vrms > 0  # particles seeded at rest have v = 0 at t_s (not shown on log axis)
    ax.plot(x[pos], vrms[pos] / sig[pos], "o-", ms=5, lw=1.5, color=COLOR[tau],
            label=LABEL[tau])
    if tau == TAUS[1] and vrms[0] > 0:
        ax.plot(x, vrms[0] / sig * np.exp(-(times - T_SEED) / tau), "--", color=COLOR[tau],
                lw=1.2, label=r"$v_\\mathrm{rms}(t_s)\\,e^{-(t-t_s)/\\tau_p}/\\sigma_v(t)$")
if any(np.sqrt((snaps[0]["parts"][t]["vel"] ** 2).sum(axis=1).mean()) == 0 for t in TAUS):
    ax.set_yscale("log")
ax.axhline(1, color=C_REF, lw=1, ls=":")
ax.set_xlabel(r"$(t - t_s)/T_L$")
ax.set_ylabel(r"$v_{p,\\mathrm{rms}} / \\sigma_v$")
ax.legend(fontsize=9)
fig.tight_layout()"""
    ),
    md(
        r"""## 7. Time average over the last 5 snapshots ($t - t_s = 3$–$5\,T_L$)

Error bars: standard error of the mean over the 5 snapshots, $s/\sqrt{5}$.
Grey band: $\pm 2\sigma$ of a Poisson field averaged over 5 independent snapshots,
$2\sqrt{2/N_\mathrm{cubes}}/\sqrt{5}$. Snapshots are $0.5\,T_L$ apart and therefore not fully
independent, so both error estimates are on the optimistic side.
"Significance" = excess in units of that single-snapshot Poisson noise divided by $\sqrt 5$."""
    ),
    code(
        """late = snaps[-5:]
print("averaging over t - t_s =", np.round((np.array([s["t"] for s in late]) - T_SEED) / T_L, 2), "T_L")
NBOXES = (32, 16, 8, 4)  # cube sides 2, 4, 8, 16 dx

avg = {}
print(f"\\n{'population':<14}{'cube':>6}{'n_bar':>8}{'Var/mean':>12}{'+/- (SEM)':>11}"
      f"{'Poisson 1sig':>14}{'signif.':>9}{'rho-only':>10}")
for tau in TAUS:
    avg[tau] = {}
    for nbox in NBOXES:
        r = np.array([box_counts(s["parts"][tau]["pos"], nbox) for s in late])
        vm = r.var(axis=1) / r.mean(axis=1)
        nbar = r.mean()
        sig1 = np.sqrt(2 / nbox**3) / np.sqrt(len(late))
        # expectation if particles simply traced the gas density
        rho_only = np.mean([
            nbar * np.var(rg := s["gas"][0].reshape(nbox, N // nbox, nbox, N // nbox, nbox,
                                                  N // nbox).mean(axis=(1, 3, 5)))
            / rg.mean() ** 2 for s in late])
        avg[tau][nbox] = dict(vm=vm.mean(), sem=vm.std(ddof=1) / np.sqrt(len(vm)),
                              sig=sig1, nbar=nbar, rho_only=rho_only, counts=r)
        a = avg[tau][nbox]
        print(f"{LABEL[tau].replace('$', '').replace('_L', 'L'):<14}{N // nbox:>4}dx"
              f"{nbar:>8.0f}{a['vm']:>12.3f}{a['sem']:>11.3f}{sig1:>14.3f}"
              f"{(a['vm'] - 1) / sig1:>8.1f}s{1 + rho_only:>10.3f}")"""
    ),
    code(
        """fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

# (a) Var/mean - 1 vs coarse-graining scale
ax = axes[0]
sides = np.array([N // nb for nb in NBOXES])
band = np.array([2 * avg[TAUS[0]][nb]["sig"] for nb in NBOXES])
ax.fill_between(sides, -band, band, color="#e4e3df", alpha=0.8, lw=0,
                label=r"Poisson $\\pm 2\\sigma$ (5 snapshots)")
for k, tau in enumerate(TAUS):
    vm = np.array([avg[tau][nb]["vm"] - 1 for nb in NBOXES])
    sem = np.array([avg[tau][nb]["sem"] for nb in NBOXES])
    ax.errorbar(sides * (1 + 0.04 * (k - 0.5)), vm, yerr=sem, fmt="o-", ms=6, lw=1.5,
                capsize=3, color=COLOR[tau], label=LABEL[tau])
ax.plot(sides, [avg[TAUS[0]][nb]["rho_only"] for nb in NBOXES], ":", color=C_REF, lw=1.5,
        label=r"gas-density only, $\\bar n\\,\\mathrm{Var}(\\rho)/\\bar\\rho^2$")
ax.axhline(0, color="#8a8984", lw=0.8)
ax.set_xscale("log", base=2)
ax.set_xticks(sides, [f"{s}" for s in sides])
ax.set_xlabel(r"cube side $\\ell / \\Delta x$")
ax.set_ylabel(r"$\\langle \\mathrm{Var}(n)/\\bar n \\rangle_t - 1$")
ax.set_title("count variance excess (0 = uniform random)", loc="left", fontsize=10)
ax.legend(fontsize=8.5, loc="upper left")

# (b) time-averaged P(n) at 4 dx vs Poisson (log scale shows the tails)
ax = axes[1]
nbox = 16
for tau in TAUS:
    c = avg[tau][nbox]["counts"].ravel()
    bins = np.arange(c.min() - 0.5, c.max() + 1.5)
    ax.hist(c, bins=bins, density=True, histtype="step", lw=2, color=COLOR[tau],
            label=f"{LABEL[tau]}  (Var/mean = {avg[tau][nbox]['vm']:.3f})")
nbar = avg[TAUS[0]][nbox]["nbar"]
n = np.arange(15, 125)
ax.plot(n, stats.poisson.pmf(n, nbar), color=C_REF, lw=1.5, ls="--",
        label=f"Poisson, mean {nbar:.0f}")
ax.set_yscale("log")
ax.set_ylim(1e-6, 0.1)
ax.set_xlim(15, 125)
ax.set_xlabel(rf"particles per cube of side {N // nbox}$\\Delta x$")
ax.set_ylabel("P(n), 5 snapshots combined")
ax.legend(fontsize=8.5, loc="lower center")
fig.tight_layout()"""
    ),
    code(
        """fig, ax = plt.subplots(figsize=(7, 4))
nsub = 100_000
for tau in TAUS:
    gs = []
    for s in late:
        p = s["parts"][tau]["pos"] % 1.0
        p = p[rng.choice(len(p), nsub, replace=False)]
        tree = cKDTree(p, boxsize=1.0)
        dd = np.diff(tree.count_neighbors(tree, edges).astype(float))
        expected = nsub * (nsub - 1) * 4 / 3 * np.pi * np.diff(edges**3)
        gs.append(dd / expected)
    gs = np.array(gs)
    ax.errorbar(rc * N, gs.mean(axis=0), yerr=gs.std(axis=0, ddof=1) / np.sqrt(len(gs)),
                fmt="o-", ms=5, lw=1.5, capsize=2, color=COLOR[tau], label=LABEL[tau])
noise = 2 / np.sqrt(expected / 2) / np.sqrt(len(late))
ax.fill_between(rc * N, 1 - noise, 1 + noise, color="#e4e3df", alpha=0.8, lw=0,
                label=r"Poisson $\\pm 2\\sigma$ (5 snapshots)")
ax.axhline(1, color=C_REF, lw=1, ls="--")
ax.set_xscale("log")
ax.set_xlabel(r"$r / \\Delta x$")
ax.set_ylabel(r"$\\langle g(r) \\rangle_t$")
ax.set_title(r"pair correlation, averaged over $t - t_s = 3$–$5\\,T_L$", loc="left",
             fontsize=10)
ax.legend(fontsize=9)
fig.tight_layout()"""
    ),
]

nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
for cell in nb.cells:
    cell.source = cell.source.replace("phase2_particles", RUN_DIR)
    if INIT_VEL == "zero":
        cell.source = cell.source.replace(
            "(initial velocity = local gas velocity)",
            "(**initially at rest**, `initial_particle_velocity=zero`)",
        )
        cell.source = cell.source.replace(
            "# Inertial particles in subsonic, solenoidally driven turbulence",
            "# Inertial particles in subsonic, solenoidally driven turbulence "
            "(particles seeded at rest)",
        )
nbf.write(nb, OUT_NAME)
print("written", OUT_NAME, "from", RUN_DIR)
