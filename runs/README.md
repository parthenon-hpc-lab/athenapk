# Inertial particles in driven turbulence: how to run

Two setups, each in two phases:

1. **Phase 1, gas only.** Drive turbulence from rest until it is statistically steady.
2. **Phase 2, particles.** Restart from the end of phase 1 and seed particles uniformly at random
   into the developed flow.

| Directory | Flow | Phase-2 populations |
|---|---|---|
| `supersonic/` | M_rms ≈ 3.7, solenoidal driving (`accel_rms = 40`, `corr_time = 0.1`) | inertial St_L = 0.01, 1, 100 (`particles.in`), pure tracers (`tracers.in`) |
| `subsonic/` | M_rms ≈ 0.19, solenoidal driving (`accel_rms = 0.08`, `corr_time = 1`) | inertial St_L = 0.001, 100 (`particles.in`) |

Both are 64³ in a periodic unit box, using Euler (HLLC/PLM/VL2) with γ = 1.0001 (near isothermal,
c_s ≈ 1). The box is 8 meshblocks of 32³, so use **up to 8 MPI ranks**.

```
runs/
├── README.md
├── tl.py                  # T_L from a phase-1 history file + the phase-2 values that depend on it
├── .jupyter/              # kernel spec for executing the notebooks (see "Analysis")
├── supersonic/
│   ├── gas.in             # phase 1: full input file
│   ├── particles.in       # phase 2: overrides applied on restart (inertial particles)
│   ├── tracers.in         # phase 2: overrides applied on restart (massless tracers)
│   └── make_notebook.py   # builds particle_distribution.ipynb
└── subsonic/
    ├── gas.in
    ├── particles.in
    └── make_notebook.py
```

Output goes into `phase1_gas/`, `phase2_particles/` and `phase2_tracers/` inside each setup
directory. `runs/.gitignore` keeps output and generated notebooks out of git.

## 0. Build

A working binary from the earlier runs is at `../test_run/build/bin/athenaPK`. To build a fresh
one from the repository root (`/Users/vk/Codes/athenapk`), use:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPARTHENON_DISABLE_OPENPMD=ON
cmake --build build -j 10 --target athenaPK
```

- `-DPARTHENON_DISABLE_OPENPMD=ON` is required here, because otherwise configuring fails looking
  for ADIOS2.
- MPI (Homebrew OpenMPI) and HDF5 (Homebrew) are picked up automatically.
- For the regression tests, add `-DPython3_EXECUTABLE=$HOME/miniforge3/envs/4apk/bin/python3`
  (an env with h5py). The tests are then run with
  `cd build && ctest -R inertial_particles --output-on-failure`.

Below, `BIN` is the binary:

```bash
export BIN=/Users/vk/Codes/athenapk/test_run/build/bin/athenaPK   # or .../build/bin/athenaPK
```

## 1. Phase 1: gas only

```bash
cd /Users/vk/Codes/athenapk/runs/supersonic        # or subsonic
mkdir -p phase1_gas && cd phase1_gas
mpirun -np 8 $BIN -i ../gas.in > run.log 2>&1
```

- **supersonic:** runs to t = 1 in about 3 min. The flow is saturated from t ≈ 0.2.
- **subsonic:** runs to t = 10 in about 2.5 min. The flow is saturated from t ≈ 5.

The run writes the history file `parthenon.out1.hst` and restart files, ending with
`parthenon.restart.final.rhdf`. No gas snapshots are written in this phase (`output2/dt = -1`).

Check saturation and compute the large eddy time T_L = L/(2σ_v) from the history. The second
argument starts the averaging window after the initial transient:

```bash
cd ..
python tl.py phase1_gas 0.3      # supersonic (use 5 for subsonic); needs numpy
```

The reference runs gave:

| | σ_v | T_L | restart time T0 |
|---|---|---|---|
| supersonic | 3.657 | 0.1367 | 1 |
| subsonic | 0.1956 | 2.556 | 10 |

If your T_L is different (for example, after changing `accel_rms`), update the three values marked
`[T_L]` in `particles.in` and `tracers.in`. You can also pass the overrides that `tl.py` prints on
the command line (see below).

## 2. Phase 2: seed particles into the developed flow

```bash
cd /Users/vk/Codes/athenapk/runs/supersonic
mkdir -p phase2_particles && cd phase2_particles
mpirun -np 8 $BIN -r ../phase1_gas/parthenon.restart.final.rhdf -i ../particles.in > run.log 2>&1
```

With `-r <restart> -i <file>`, the parameters in `<file>` override the ones stored in the restart
file. Command-line `block/param=value` arguments override both. Run times:

- **supersonic:** about 6 min (5 T_L, 3 × 2¹⁸ particles).
- **subsonic:** about 5 min.

Particles are seeded once, at the first cycle after the restart, and you get 11 snapshots
`parthenon.parts.*.phdf` at intervals of 0.5 T_L.

**Pure tracers (supersonic only).** Run these as a separate run from the same restart file:

```bash
cd /Users/vk/Codes/athenapk/runs/supersonic
mkdir -p phase2_tracers && cd phase2_tracers
mpirun -np 8 $BIN -r ../phase1_gas/parthenon.restart.final.rhdf -i ../tracers.in > run.log 2>&1
```

Particles don't act on the gas, so with the same restart file and the **same number of ranks**,
the gas evolution is bit-identical to the `phase2_particles` run. You can check that with
`diff phase2_tracers/parthenon.out1.hst phase2_particles/parthenon.out1.hst`. The notebook relies
on this and analyses both runs together.

### Changing things

Do either of the following:

- Edit `particles.in`.
- Append overrides to the `mpirun` line. For example, here is a different set of Stokes numbers, a
  longer run, and particles seeded with the local gas velocity instead of at rest:

  ```bash
  ... -i ../particles.in tracers/stokes_numbers=0.1,0.3,1.0 \
      parthenon/time/tlim=2.37 tracers/initial_particle_velocity=gas
  ```

| Parameter (`<tracers>`) | Meaning |
|---|---|
| `type` | `inertial` (with drag) or `tracer` (massless, follows the interpolated gas velocity) |
| `stokes_numbers` | list of St; one population of `initial_num_tracers_per_cell` particles per cell per entry |
| `stokes_time_unit` | τ_p = St × `stokes_time_unit`. Set it to T_L for St_L, or to τ_η for St_η (no code change needed) |
| `drag_law` | `stokes` (constant τ_p) or `epstein` (τ_s = τ_p ρ_ref c_ref / (ρ c_s); needs `epstein_rho_ref`, `epstein_cs_ref`) |
| `initial_particle_velocity` | `zero` (at rest) or `gas` (local gas velocity at seeding) |
| `initial_rng_seed` | random seed for the seeding positions |
| `seed_time` | for a fresh (non-restart) run: seed at the first cycle with t ≥ `seed_time` (default 0) |

See `docs/input.md` ("Inertial particles") for the equations and the integrator.

For other resolutions, set `parthenon/mesh/nx{1,2,3}` in `gas.in`. Phase 1 must be rerun, and the
mesh can't be changed on restart. Keep `meshblock/nx*` ≥ 16, and use at most (number of meshblocks)
MPI ranks. At 128³ with 32³ blocks there are 64 blocks; expect roughly 16× the cost of 64³.

## 3. Analysis notebooks

Analysis needs a Python env with numpy, scipy, matplotlib, h5py, nbformat and nbconvert. The
`d3` env has all of them. The default `python3` Jupyter kernel points at base miniforge (no h5py),
so the kernel spec in `runs/.jupyter` points it at `d3` instead. That's what `JUPYTER_PATH` below
is for.

```bash
PY=$HOME/miniforge3/envs/d3/bin/python3

# supersonic: needs phase1_gas, phase2_particles and phase2_tracers
cd /Users/vk/Codes/athenapk/runs/supersonic
$PY make_notebook.py
JUPYTER_PATH=$PWD/../.jupyter $PY -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 particle_distribution.ipynb

# subsonic: args = <run dir> <gas|zero (initial velocity used)> <notebook name>
cd /Users/vk/Codes/athenapk/runs/subsonic
$PY make_notebook.py phase2_particles zero particle_distribution.ipynb
JUPYTER_PATH=$PWD/../.jupyter $PY -m nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 particle_distribution.ipynb
```

Executing a notebook takes a few minutes, mostly for the pair-correlation sums. Alternatively,
open the notebook in Jupyter/VS Code with the `d3` env as the kernel.

- The notebooks read T_L (`tracers/stokes_time_unit`) and the seeding time from the first
  snapshot, so they adapt to your runs. The numbers quoted in the notebooks' introduction text
  (σ_v, Mach number, and so on) describe the reference runs; update them if your setup differs.
- **supersonic:** shows gas density PDFs seen by the particles, count statistics against the
  perfect-tracer expectation, the particle-to-gas ratio, preferential sampling, velocities and
  g(r).
- **subsonic:** shows count statistics against Poisson, g(r), ⟨ω²⟩ sampled by the particles, and
  velocities.

## Notes

- Restart files of phase 2 are disabled (`output3/dt = -1`). To restart phase 2 itself, set
  `parthenon/output3/dt` to a positive value; particles and their `vp`/`tau_p` are stored in the
  restart files, and they are not re-seeded on restart.
- The St_L = 100 populations barely move within 5 T_L (τ_p = 100 T_L). Their distribution is a
  transient, not a steady state; reaching one needs runs of order τ_p.
- In the supersonic flow, even massless tracers cluster more than the gas density at scales below
  a few cells. This is a known artifact of tracers that follow the interpolated velocity; it
  should shrink with resolution. Compare against the tracer population, not only against the gas.
