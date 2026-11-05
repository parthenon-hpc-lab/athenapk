"""Builds particle_distribution.ipynb for the supersonic run (needs nbformat)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

nb.cells = [
    md(
        r"""# Inertial particles in supersonic, solenoidally driven turbulence

**Setup** (`turb_super.in`, `run_phase2.sh`)
- $64^3$, periodic unit box, Euler (HLLC/PLM/VL2, first-order flux fallback), $\gamma = 1.0001$
  (near isothermal, $c_s \approx 1$)
- purely solenoidal OU forcing at $k \approx 2$, `accel_rms = 40`, `corr_time = 0.1`
- **phase 1** (gas only): $t = 0 \to 1$, saturated from $t \approx 0.2$ with
  $\sigma_v \approx 3.66$ ($\mathcal{M}_\mathrm{rms} \approx 3.65$), $T_L = L/(2\sigma_v) \approx 0.137$
- **phase 2**: restart at $t = 1$, seed $2^{18}$ particles per population uniformly at random,
  **at rest**, Stokes drag, one-way coupling, run for $5\,T_L$
  - $\mathrm{St}_L = \tau_p/T_L = 0.01$, $1$, $100$
- **pure tracers** (`tracers/type=tracer`, massless, advected with the interpolated gas velocity
  by Heun's method): separate restart from the same state with the same seeding; the gas
  evolution is bit-identical (one-way coupling), so they are shown together as St = 0

**Baseline.** The gas density is strongly non-uniform here (log-normal-ish, $\sigma_{\ln\rho}\sim 1$),
so "uniform random" is no longer the reference for tracer-like particles. A perfect tracer
population is a Poisson sample of the *gas density*: its density PDF is the mass-weighted gas PDF and
$\mathrm{Var}(n)/\bar n = 1 + \bar n\,\mathrm{Var}(\rho_\ell)/\bar\rho^2$. Deviations from that
measure dust-gas decoupling."""
    ),
    code(
        r"""import glob
import sys

import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(
    1,
    "../../external/parthenon/scripts/python/packages/"
    "parthenon_tools/parthenon_tools",
)
import phdf

N = 64  # cells per dimension, box [0, 1]^3
RUN = "phase2_particles"
TRACER_RUN = "phase2_tracers"
# stopping-time unit (= T_L used for this run) and seeding time, read from the first snapshot
with h5py.File(sorted(glob.glob(f"{RUN}/parthenon.parts.*.phdf"))[0], "r") as f0:
    T_L = float(f0["Params"].attrs["tracers/stokes_time_unit"])
    T_SEED = float(f0["Info"].attrs["Time"])
print(f"T_L = {T_L:g}, particles seeded at t = {T_SEED:g}")

# Categorical palette (fixed order). Identity is also given by legends/labels.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
C_REF, C_GREY, C_GRID = "#0b0b0b", "#8a8984", "#e4e3df"
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": C_GRID,
        "grid.linewidth": 0.6,
        "axes.edgecolor": C_GREY,
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "lines.linewidth": 2,
        "legend.frameon": False,
    }
)
rng = np.random.default_rng(1)"""
    ),
    code(
        r"""def load_gas(d):
    "Assembles prim (rho, vx, vy, vz, p) into global arrays indexed [var, z, y, x]."
    p = d.Get("prim", flatten=False).astype(np.float64)
    nb = p.shape[-1]
    out = np.zeros((p.shape[1], N, N, N))
    for b, bb in enumerate(d.BlockBounds):
        i0, j0, k0 = (int(round((bb[2 * a] + 1e-8) * N)) for a in range(3))
        out[:, k0 : k0 + nb, j0 : j0 + nb, i0 : i0 + nb] = p[b]
    return out


def load_snapshot(fname, tracer_fname):
    d = phdf.phdf(fname)
    s = d.GetSwarm("tracers")
    tau = s.Get("tau_p").astype(np.float64)
    pos = np.stack([s.x, s.y, s.z], axis=-1).astype(np.float64) % 1.0
    vel = np.stack([s.Get(v) for v in ("vp_x", "vp_y", "vp_z")], axis=-1).astype(np.float64)
    # pure tracers (tracers/type=tracer, separate run with identical gas evolution):
    # stored as tau_p = 0, velocity = interpolated gas velocity at the tracer position
    dt_ = phdf.phdf(tracer_fname)
    assert abs(dt_.Time - d.Time) < 1e-10, (dt_.Time, d.Time)
    st = dt_.GetSwarm("tracers")
    tau = np.concatenate([np.zeros(len(st.x)), tau])
    pos = np.concatenate([np.stack([st.x, st.y, st.z], axis=-1).astype(np.float64) % 1.0, pos])
    vel = np.concatenate([np.stack([st.Get(v) for v in ("vel_x", "vel_y", "vel_z")],
                                   axis=-1).astype(np.float64), vel])
    parts = {}
    for t in np.unique(tau):
        m = tau == t
        idx = np.floor(pos[m] * N).astype(int) % N
        parts[t] = {"pos": pos[m], "vel": vel[m], "cell": (idx[:, 2], idx[:, 1], idx[:, 0])}
    return {"t": d.Time, "gas": load_gas(d), "parts": parts}


files = sorted(glob.glob(f"{RUN}/parthenon.parts.*.phdf"))
tfiles = sorted(glob.glob(f"{TRACER_RUN}/parthenon.parts.*.phdf"))
assert len(files) == len(tfiles)
snaps = [load_snapshot(f, ft) for f, ft in zip(files, tfiles)]
TAUS = sorted(snaps[0]["parts"].keys())  # 0 = pure tracer, then the inertial populations
LABEL = {t: (r"tracer (St = 0)" if t == 0 else rf"St$_L$ = {t / T_L:.3g}") for t in TAUS}
# fixed colors per population (inertial ones keep the colors of the earlier version)
COLOR = dict(zip(TAUS, [COLORS[3]] + COLORS[:3]))
times = np.array([s["t"] for s in snaps])
x_t = (times - T_SEED) / T_L
late = snaps[-5:]
print(f"{len(snaps)} snapshots at (t - t_s)/T_L = {np.round(x_t, 2)}")
print({LABEL[t]: len(snaps[0]["parts"][t]["pos"]) for t in TAUS})"""
    ),
    code(
        r"""def ddx(f, axis):
    return (np.roll(f, -1, axis) - np.roll(f, 1, axis)) * (N / 2)


def div_v(gas):  # arrays [z, y, x]
    return ddx(gas[1], 2) + ddx(gas[2], 1) + ddx(gas[3], 0)


def vort2(gas):
    vx, vy, vz = gas[1], gas[2], gas[3]
    return ((ddx(vz, 1) - ddx(vy, 0)) ** 2 + (ddx(vx, 0) - ddx(vz, 2)) ** 2
            + (ddx(vy, 2) - ddx(vx, 1)) ** 2)


def coarse(f, nbox):
    k = N // nbox
    return f.reshape(nbox, k, nbox, k, nbox, k).mean(axis=(1, 3, 5))


def box_counts(pos, nbox):
    idx = np.floor(pos * nbox).astype(int) % nbox
    return np.bincount((idx[:, 2] * nbox + idx[:, 1]) * nbox + idx[:, 0],
                       minlength=nbox**3).reshape(nbox, nbox, nbox)"""
    ),
    md("## 0. Gas: turbulence saturation and seeding time"),
    code(
        r"""h1 = np.loadtxt("phase1_gas/parthenon.out1.hst")
h2 = np.loadtxt(f"{RUN}/parthenon.out1.hst")
h = np.concatenate([h1, h2[h2[:, 0] > h1[-1, 0]]])
sigma_v = np.sqrt(2 * h[:, 8] / h[:, 4])

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(h[:, 0], sigma_v, color=C_REF, lw=1.5,
        label=r"$\sigma_v = \sqrt{2\,E_\mathrm{kin}/M}$")
ax.plot(h[:, 0], h[:, 10], color=C_GREY, lw=1.5, ls="--", label=r"$\langle |v|/c_s \rangle$")
ax.axvline(T_SEED, color=COLORS[0], lw=1)
ax.text(T_SEED + 0.01, 0.3, "particles seeded", color="#52514e", fontsize=9)
ax.set_xlabel("t")
ax.set_ylim(0, None)
ax.legend(loc="lower right")
fig.tight_layout()
sel = h[:, 0] >= T_SEED
print(f"phase 2: sigma_v = {sigma_v[sel].mean():.2f}, <|v|/c_s> = {h[sel, 10].mean():.2f}")"""
    ),
    md(
        r"""## 1. Particle positions in a thin slab

Particles with $|z - 0.5| < \Delta x/2$ (one cell thick) over the gas column
$\log_{10}\rho$ in that cell layer. Top: at seeding; bottom: after $5\,T_L$."""
    ),
    code(
        r"""z0, k0 = 0.5, N // 2
lr_all = np.log10(snaps[-1]["gas"][0][k0])
vmin, vmax = np.percentile(lr_all, [1, 99.5])
fig, axes = plt.subplots(2, 4, figsize=(19, 9.5), sharex=True, sharey=True)
for row, snap in zip(axes, (snaps[0], snaps[-1])):
    lr = np.log10(snap["gas"][0][k0])
    for ax, tau in zip(row, TAUS):
        im = ax.imshow(lr, origin="lower", extent=(0, 1, 0, 1), cmap="Greys", vmin=vmin,
                       vmax=vmax, interpolation="nearest")
        p = snap["parts"][tau]["pos"]
        m = np.abs(p[:, 2] - z0) < 0.5 / N
        ax.plot(p[m, 0], p[m, 1], ".", ms=1.5, color=COLOR[tau], mec="none")
        ax.set_title(f"{LABEL[tau]},  t - t$_s$ = {(snap['t'] - T_SEED) / T_L:.1f} $T_L$"
                     f"  ({m.sum()})", loc="left", fontsize=10)
        ax.grid(False)
for ax in axes[-1]:
    ax.set_xlabel("x")
for ax in axes[:, 0]:
    ax.set_ylabel("y")
fig.colorbar(im, ax=axes, shrink=0.6, label=r"$\log_{10}\rho$ (gas, slab $z = 0.5$)")"""
    ),
    md(
        r"""## 2. Gas density seen by the particles

PDF of $s = \ln(\rho/\bar\rho)$ in the particle's cell, averaged over the last 5 snapshots
($t - t_s = 3$–$5\,T_L$), compared with the volume-weighted (dashed grey) and
mass-weighted (dashed black) gas PDFs. Perfect tracers sample the mass-weighted PDF;
particles that ignore the gas sample the volume-weighted one."""
    ),
    code(
        r"""bins = np.linspace(-5, 3.5, 86)
bc = 0.5 * (bins[1:] + bins[:-1])
hv, hm = np.zeros(len(bc)), np.zeros(len(bc))
hp = {t: np.zeros(len(bc)) for t in TAUS}
s_vol = []
for s in late:
    rho = s["gas"][0]
    lns = np.log(rho / rho.mean())
    s_vol.append(lns.std())
    hv += np.histogram(lns, bins, density=True)[0] / len(late)
    hm += np.histogram(lns, bins, weights=rho, density=True)[0] / len(late)
    for t in TAUS:
        hp[t] += np.histogram(lns[s["parts"][t]["cell"]], bins, density=True)[0] / len(late)

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(bc, hv, "--", color=C_GREY, lw=1.8, label="gas, volume-weighted")
ax.plot(bc, hm, "--", color=C_REF, lw=1.8, label="gas, mass-weighted")
for t in TAUS:
    ax.plot(bc, hp[t], color=COLOR[t], lw=1.8, label=f"particles, {LABEL[t]}")
ax.set_yscale("log")
ax.set_ylim(1e-4, 1)
ax.set_xlabel(r"$s = \ln(\rho/\bar\rho)$ at particle / cell")
ax.set_ylabel("PDF")
ax.legend(fontsize=9, loc="lower center")
fig.tight_layout()

sig_s = np.mean(s_vol)
M = 3.65
print(f"volume-weighted sigma_s = {sig_s:.2f}  ->  b = sqrt(exp(sigma_s^2) - 1)/M = "
      f"{np.sqrt(np.exp(sig_s**2) - 1) / M:.2f}  (b ~ 1/3 expected for solenoidal driving)")"""
    ),
    md(
        r"""## 3. Count statistics vs. time

$\mathrm{Var}(n)/\bar n$ in non-overlapping cubes (log scale; 1 = uniform random).
Dotted: expectation for perfect tracers, $1 + \bar n\,\mathrm{Var}(\rho_\ell)/\bar\rho^2$
with $\rho_\ell$ the gas density averaged over the cube."""
    ),
    code(
        r"""fig, axes = plt.subplots(1, 3, figsize=(14, 3.9), sharey=True)
for ax, nbox in zip(axes, (32, 16, 8)):
    nbar = len(snaps[0]["parts"][TAUS[0]]["pos"]) / nbox**3
    for t in TAUS:
        vm = [(c := box_counts(s["parts"][t]["pos"], nbox)).var() / c.mean() for s in snaps]
        ax.plot(x_t, vm, "o-", ms=5, lw=1.5, color=COLOR[t], label=LABEL[t])
    tr = [1 + nbar * (r := coarse(s["gas"][0], nbox)).var() / r.mean() ** 2 for s in snaps]
    ax.plot(x_t, tr, ":", color=C_REF, lw=1.8, label="perfect tracers")
    ax.axhline(1, color=C_GREY, lw=0.8)
    ax.set_yscale("log")
    ax.set_title(rf"cube side {N // nbox}$\Delta x$,  $\bar n$ = {nbar:.0f}", loc="left",
                 fontsize=10)
    ax.set_xlabel(r"$(t - t_s)/T_L$")
axes[0].set_ylabel(r"Var$(n)/\bar n$")
axes[0].legend(fontsize=9)
fig.tight_layout()"""
    ),
    md(
        r"""## 4. Particle-to-gas ratio

In cubes of side $4\,\Delta x$ ($\bar n = 64$): $\epsilon = (n/\bar n)/(\rho_\ell/\bar\rho)$,
the local "dust-to-gas ratio" normalized to its mean. Perfect tracers give $\epsilon = 1$ up to
Poisson noise; the grey curve is a mock tracer population drawn as
$n \sim \mathrm{Poisson}(\bar n \rho_\ell/\bar\rho)$ on the same gas fields.
Averaged over the last 5 snapshots."""
    ),
    code(
        r"""nbox = 16
nbar = len(snaps[0]["parts"][TAUS[0]]["pos"]) / nbox**3
eps = {t: [] for t in TAUS}
eps_mock, rho_l = [], []
for s in late:
    r = coarse(s["gas"][0], nbox)
    r = r / r.mean()
    rho_l.append(r.ravel())
    eps_mock.append(rng.poisson(nbar * r).ravel() / nbar / r.ravel())
    for t in TAUS:
        eps[t].append((box_counts(s["parts"][t]["pos"], nbox) / nbar / r).ravel())
eps = {t: np.concatenate(v) for t, v in eps.items()}
eps_mock, rho_l = np.concatenate(eps_mock), np.concatenate(rho_l)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axes[0]
be = np.linspace(0, 8, 81)
ax.hist(eps_mock, be, density=True, histtype="stepfilled", color=C_GRID, label="mock perfect tracers")
for t in TAUS:
    ax.hist(eps[t], be, density=True, histtype="step", lw=1.8, color=COLOR[t],
            label=f"{LABEL[t]}  (std = {eps[t].std():.2f})")
ax.set_yscale("log")
ax.set_xlabel(r"$\epsilon = (n/\bar n)\,/\,(\rho_\ell/\bar\rho)$")
ax.set_ylabel("PDF (cubes)")
ax.legend(fontsize=9)

# (b) mean particle density vs gas density (conditional mean)
ax = axes[1]
lb = np.logspace(np.log10(rho_l.min()), np.log10(rho_l.max()), 25)
which = np.digitize(rho_l, lb)
xc = np.sqrt(lb[1:] * lb[:-1])
for t in TAUS:
    nn = eps[t] * rho_l
    y = np.array([nn[which == i].mean() if (which == i).sum() > 20 else np.nan
                  for i in range(1, len(lb))])
    ax.plot(xc, y, "o-", ms=4.5, lw=1.5, color=COLOR[t], label=LABEL[t])
ax.plot(xc, xc, ":", color=C_REF, lw=1.5, label=r"$n/\bar n = \rho_\ell/\bar\rho$ (tracers)")
ax.axhline(1, color=C_GREY, lw=1, ls="--", label="uniform")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"gas density in cube $\rho_\ell/\bar\rho$")
ax.set_ylabel(r"$\langle n \rangle / \bar n$ at that gas density")
ax.legend(fontsize=9)
fig.tight_layout()
print(f"std(eps): mock tracers {eps_mock.std():.3f}, "
      + ", ".join(f"{LABEL[t].replace('$', '')} {eps[t].std():.3f}" for t in TAUS))"""
    ),
    md(
        r"""## 5. Which flow regions do the particles sample?

Gas quantities in the particle's cell, averaged over particles, vs. time.
Dotted black: the same quantity mass-weighted over the gas (what perfect tracers would give);
dashed grey: volume average (what a uniform distribution would give).
Divergence is normalized by its volume rms; $\omega^2$ by its volume mean."""
    ),
    code(
        r"""fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
qs = [
    (r"$\langle\rho\rangle / \bar\rho$", lambda g: g[0] / g[0].mean()),
    (r"$\langle\nabla\cdot\mathbf{u}\rangle / \mathrm{rms}_V(\nabla\cdot\mathbf{u})$",
     lambda g: (d := div_v(g)) / d.std()),
    (r"$\langle\omega^2\rangle / \langle\omega^2\rangle_V$", lambda g: (w := vort2(g)) / w.mean()),
]
for ax, (lab, fq) in zip(axes, qs):
    fields = [fq(s["gas"]) for s in snaps]
    for t in TAUS:
        ax.plot(x_t, [f[s["parts"][t]["cell"]].mean() for f, s in zip(fields, snaps)], "o-",
                ms=5, lw=1.5, color=COLOR[t], label=LABEL[t])
    ax.plot(x_t, [(f * s["gas"][0]).sum() / s["gas"][0].sum() for f, s in zip(fields, snaps)],
            ":", color=C_REF, lw=1.8, label="mass-weighted gas")
    ax.plot(x_t, [f.mean() for f in fields], "--", color=C_GREY, lw=1.2, label="volume average")
    ax.set_ylabel(lab)
    ax.set_xlabel(r"$(t - t_s)/T_L$")
axes[0].set_yscale("log")
axes[0].legend(fontsize=9)
fig.tight_layout()"""
    ),
    md(
        r"""## 6. Particle velocities

Left: rms particle speed / $\sigma_v$ (log scale; all inertial particles start at rest, so $t = t_s$
is not shown). Right: rms slip velocity $|\mathbf{v}_p - \mathbf{u}(\mathbf{x}_p)|$ / $\sigma_v$ (gas
velocity of the particle's cell). For the pure tracers this is only the difference between the
interpolated gas velocity and the cell-centred value."""
    ),
    code(
        r"""sig = np.interp(times, h[:, 0], sigma_v)
fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), sharey=True)
for t in TAUS:
    vrms, slip = [], []
    for s in snaps:
        p = s["parts"][t]
        u = np.stack([s["gas"][k][p["cell"]] for k in (1, 2, 3)], axis=-1)
        vrms.append(np.sqrt((p["vel"] ** 2).sum(axis=1).mean()))
        slip.append(np.sqrt(((p["vel"] - u) ** 2).sum(axis=1).mean()))
    vrms, slip = np.array(vrms) / sig, np.array(slip) / sig
    m = vrms > 0
    axes[0].plot(x_t[m], vrms[m], "o-", ms=5, lw=1.5, color=COLOR[t], label=LABEL[t])
    axes[1].plot(x_t[m], slip[m], "o-", ms=5, lw=1.5, color=COLOR[t], label=LABEL[t])
for ax in axes:
    ax.axhline(1, color=C_REF, lw=1, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel(r"$(t - t_s)/T_L$")
axes[0].set_ylabel(r"$v_{p,\mathrm{rms}} / \sigma_v$")
axes[1].set_ylabel(r"$|\mathbf{v}_p - \mathbf{u}|_\mathrm{rms} / \sigma_v$")
axes[0].legend(fontsize=9)
fig.tight_layout()"""
    ),
    md(
        r"""## 7. Time average over the last 5 snapshots ($t - t_s = 3$–$5\,T_L$)

Table: $\mathrm{Var}(n)/\bar n$ (± standard error over the 5 snapshots) at four coarse-graining
scales, vs. the perfect-tracer expectation. Figure: pair correlation function $g(r)$ of the
particles ($10^5$ random subsample per population), compared with the gas-density
autocorrelation $\langle\rho(\mathbf{x})\rho(\mathbf{x}+\mathbf{r})\rangle/\bar\rho^2$ (what perfect
tracers would give for $r \gtrsim \Delta x$; computed by FFT on the grid)."""
    ),
    code(
        r"""NBOXES = (32, 16, 8, 4)
print(f"{'population':<14}" + "".join(f"{f'{N // nb}dx (nbar={N**3 // nb**3})':>26}" for nb in NBOXES))
for t in TAUS + ["tracer"]:
    row = []
    for nb in NBOXES:
        nbar = N**3 / nb**3
        if t == "tracer":
            v = np.array([1 + nbar * (r := coarse(s["gas"][0], nb)).var() / r.mean() ** 2
                          for s in late])
        else:
            v = np.array([(c := box_counts(s["parts"][t]["pos"], nb)).var() / c.mean()
                          for s in late])
        row.append(f"{v.mean():>15.2f} +/- {v.std(ddof=1) / np.sqrt(len(v)):<6.2f}")
    name = "perfect tracer" if t == "tracer" else LABEL[t].replace("$", "").replace("_L", "L")
    print(f"{name:<14}" + "".join(f"{x:>26}" for x in row))"""
    ),
    code(
        r"""edges = np.logspace(np.log10(0.25 / N), np.log10(0.25), 22)
rc = np.sqrt(edges[1:] * edges[:-1])
nsub = 100_000
shell = nsub * (nsub - 1) * 4 / 3 * np.pi * np.diff(edges**3)

# gas density autocorrelation on the grid (periodic, FFT), binned in |r|
k = np.fft.fftfreq(N, d=1.0 / N)
dist = np.sqrt(sum(np.meshgrid(*(np.minimum(np.abs(k), N - np.abs(k)),) * 3, indexing="ij")[a] ** 2
                   for a in range(3))) / N
acf = 0
for s in late:
    d = s["gas"][0] / s["gas"][0].mean()
    fd = np.fft.fftn(d)
    acf = acf + np.fft.ifftn(fd * np.conj(fd)).real / N**3 / len(late)
rb = np.arange(1, 17)  # integer cell separations
acf_r = [acf[(dist >= (r - 0.5) / N) & (dist < (r + 0.5) / N)].mean() for r in rb]

fig, ax = plt.subplots(figsize=(7.5, 4.4))
for t in TAUS:
    gs = []
    for s in late:
        p = s["parts"][t]["pos"]
        p = p[rng.choice(len(p), nsub, replace=False)]
        tree = cKDTree(p, boxsize=1.0)
        gs.append(np.diff(tree.count_neighbors(tree, edges).astype(float)) / shell)
    gs = np.array(gs)
    ax.errorbar(rc * N, gs.mean(axis=0), yerr=gs.std(axis=0, ddof=1) / np.sqrt(len(gs)),
                fmt="o-", ms=5, lw=1.5, capsize=2, color=COLOR[t], label=LABEL[t])
ax.plot(rb, acf_r, ":", color=C_REF, lw=1.8,
        label=r"gas $\langle\rho\rho\rangle(r)/\bar\rho^2$ (perfect tracers)")
ax.axhline(1, color=C_GREY, lw=1, ls="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$r / \Delta x$")
ax.set_ylabel(r"$\langle g(r) \rangle_t$")
ax.set_title(r"pair correlation, averaged over $t - t_s = 3$–$5\,T_L$", loc="left", fontsize=10)
ax.legend(fontsize=9)
fig.tight_layout()"""
    ),
]

nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "particle_distribution.ipynb")
print("written particle_distribution.ipynb")
