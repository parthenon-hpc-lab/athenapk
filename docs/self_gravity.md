# Self-gravity

AthenaPK can evolve the gas under its own gravity by solving the Poisson equation

$$\nabla^2 \phi = 4\pi G\,\rho$$

on every stage of the integrator and adding the resulting acceleration $-\nabla\phi$ as
an unsplit source term to the momentum and total-energy equations. The solver is a port of the
[Artemis](https://github.com/lanl/artemis) self-gravity module and is built on
Parthenon's geometric-multigrid (GMG) infrastructure.

## Algorithm

- The potential `grav.phi` and the right-hand side `grav.rhs` $= 4\pi G(\rho-\bar\rho)$
  are cell-centered fields registered by the `self_gravity` package. Both are written
  to output (`hdf5` as well as OpenPMD) when listed in a `variables` line.
- The Poisson equation is solved with a **BiCGSTAB Krylov solver preconditioned by
  geometric multigrid** (`parthenon::solvers::BiCGSTABSolver` +
  `MGSolver`), which converges robustly on the block-AMR hierarchy.
- The solve is **stage-consistent**: the Poisson equation is solved once per integrator
  stage from that stage's density, and the Artemis-style flux-weighted gravitational
  source term (`ApplyGravitySource`) carries the stage's $\beta\,\Delta t$ weight, exactly
  like the hydro flux update and the other unsplit sources. Both the predictor and the
  corrector of the VL2 integrator therefore feel gravity. The cost is one extra elliptic
  solve per step.
- **Second-order accuracy.** The momentum source is the one derived by
  [Mullen, Hanawa & Gammie (2021)](https://doi.org/10.3847/1538-4365/abcfbd), their
  Equations (43)-(45) applied per stage as their (63) and (67): the start-of-stage
  density multiplied by the face-averaged gravity of that same density,
  $\rho^{(\ell)}\,\tfrac{1}{2}(g_{i-1/2}+g_{i+1/2})$, weighted by the stage's
  $\beta\,\Delta t$. With one Poisson solve per stage this is second-order accurate in
  space and time, and needs only two solves per step for a two-stage integrator.

  Verified on their test problem (§4.2.1): a Jeans-*stable* linear wave with
  $\lambda/\lambda_J=1/2$ ($c_s=1/\pi$, $\rho_0=1$, $A=10^{-6}$, $\gamma=5/3$,
  $4\pi G=1$, so $\omega^2=3$), evolved a quarter period to $t=\tfrac{\pi}{2\omega}$ and
  compared against the analytic linear solution
  $\rho = \rho_0\,[1 + A\beta\sin kx]$ with
  $\beta = k^2c_s^2(1-\gamma^{-1})/\omega^2 = 8/15$:

  | $N$ | $L_1(\rho)$ | order | modal-amplitude error | order |
  |-----|------------|-------|-----------------------|-------|
  |  32 | 1.807e-09  |   —   | 1.947e-09 |   —   |
  |  64 | 4.354e-10  | 2.05  | 5.242e-10 | 1.89  |
  | 128 | 1.002e-10  | 2.12  | 1.325e-10 | 1.98  |
  | 256 | 2.289e-11  | 2.13  | 3.306e-11 | 2.00  |
  | 512 | 5.421e-12  | 2.08  | 8.233e-12 | 2.01  |

  A $\Delta t$-only refinement at *fixed* $\Delta x$ is **not** a valid way to measure
  this, because the PLM slope limiter is not a smooth function of the state: which cells
  it clips changes with $\Delta t$, so Richardson extrapolation over a CFL ladder does not
  return a truncation-error order. Measured on the same 128-cell mesh, with successive
  differences of the final modal amplitude over CFL = 0.1/0.05/0.025/0.0125:

  | reconstruction | gravity | measured "order" |
  |----------------|---------|------------------|
  | donor cell     | off     | 1.96, 1.98 |
  | PLM            | off     | 0.75, 0.88 |
  | PLM            | on (stable mode)   | 0.79, 0.90 |
  | PLM            | on (unstable mode) | 1.10, 1.06 |

  With the limiter removed the ladder recovers the expected second order, and with the
  limiter present it reads below 2 *whether or not gravity is enabled*. The artefact
  therefore belongs to the diagnostic and to the base scheme, not to the gravity
  coupling; the resolution ladder above is the meaningful measurement.
- **Known deviation from the reference scheme.** Mullen et al.'s energy source (their
  Equations 57, 64 and 68) dots the mass flux with the *time-averaged* gravity
  $\tfrac{1}{2}(g^{(0)}+g^{(\ell)})$, which makes the source exactly the divergence of a
  gravitational energy flux and conserves total energy to round-off. Like Athena++, which
  documents the same limitation in its own `src/hydro/srcterms/self_gravity.cpp`, this
  port uses the instantaneous start-of-stage gravity instead, so energy is not conserved to
  round-off. The effect is small — on the unstable Jeans mode, removing the energy source
  altogether shifts the final modal amplitude by 2.1e-9 out of 3.06e-3 — but implementing
  the conservative form would need the potential of the *end*-of-stage density and a
  stored $\phi^{(0)}$, i.e. a reordering of the solve.
- The source term writes interior cells only and runs after the stage's boundary
  exchange (the solve is global, so it cannot sit inside the stage task list), so the
  ghost zones are re-communicated before `FillDerived`. Without that, the next stage's
  reconstruction would use ghost values missing one gravitational kick.
- For fully periodic domains the **Jeans swindle** (`use_swindle`) subtracts the mean
  density so the periodic Poisson problem is well posed.

## Enabling self-gravity

Self-gravity is enabled from its own block by selecting a Poisson solver. The mesh must
have multigrid enabled.

```
<parthenon/mesh>
multigrid = true          # required by the GMG-preconditioned solver

<self_gravity>
solver         = multigrid  # "none" (default) disables self-gravity
four_pi_G      = 1.0        # value of 4*pi*G in code units
use_swindle    = true       # subtract mean density (default: true iff fully periodic)

# Gravity boundary conditions, per face (ix1_bc/ox1_bc/.../ix3_bc/ox3_bc).
# "default" inherits the hydro BC; "zero" imposes homogeneous Dirichlet (phi=0).
ix1_bc = zero
ox1_bc = zero
# ... etc

<self_gravity/multigrid_solver_params>
preconditioner               = Multigrid
max_iterations               = 200
residual_tolerance           = 1.0e-10
absolute_residual_tolerance  = 1.0e-10
relative_residual            = true
relative_residual_tolerance  = 1.0e-6
```

### The gravitational constant

The solver works entirely in code units: `self_gravity/four_pi_G` is the value of
$4\pi G$ in those units. The default `1.0` is the usual normalization of the Jeans and
Bonnor-Ebert setups; a problem posed in physical units simply passes the corresponding
code-unit value.

### Input parameters

| Block | Parameter | Default | Meaning |
|-------|-----------|---------|---------|
| `<self_gravity>` | `solver` | `none` | Poisson solver. `multigrid` enables self-gravity via the GMG infrastructure; `none` disables the package. |
| `<parthenon/mesh>` | `multigrid` | `false` | Must be `true`; enables the GMG hierarchy. |
| `<self_gravity>` | `four_pi_G` | `1.0` | Value of $4\pi G$ in code units. |
| `<self_gravity>` | `use_swindle` | periodic? | Subtract the mean density from the RHS. Defaults to `true` for a fully periodic domain, `false` otherwise. |
| `<self_gravity>` | `{i,o}x{1,2,3}_bc` | `default` | Per-face gravity BC. `default` follows the hydro BC; `zero` = homogeneous Dirichlet. |
| `<self_gravity>` | `packed_bc` | `true` | Apply the `zero`/`neumann` $\phi$ BCs to a whole `MeshData` in one kernel per face during the solve (large GPU launch-latency saving at deep AMR; bit-identical to the per-block path). Automatically falls back to the per-block path for any other face type. |
| `<self_gravity/multigrid_solver_params>` | `solver_type` | `BiCGSTAB` | `BiCGSTAB` = GMG-preconditioned Krylov solver (robust default). `MG` = pure geometric multigrid, which avoids the two global reductions per Krylov iteration (helps latency-bound multi-rank GPU runs) but requires at least the SRJ2 smoother (the Parthenon `MGParams` default) on AMR grids. |
| `<self_gravity/multigrid_solver_params>` | `preconditioner` | `Multigrid` | Krylov preconditioner. |
| `<self_gravity/multigrid_solver_params>` | `max_iterations` | — | Maximum Krylov iterations per solve. |
| `<self_gravity/multigrid_solver_params>` | `residual_tolerance`, `relative_residual_tolerance`, `absolute_residual_tolerance` | — | Convergence tolerances, see the [Parthenon solver documentation](https://parthenon-hpc-lab.github.io/parthenon/develop/src/solvers.html). |

> **Boundary-condition caveat.** Each face may be `zero` (homogeneous Dirichlet,
> $\phi=0$), `neumann` (homogeneous Neumann, $\partial\phi/\partial n=0$, e.g. a
> symmetry plane), or `default` (inherit the hydro BC; periodic faces give a periodic
> $\phi$). An **all**-Neumann domain is gauge-unfixed (the potential is then determined
> only up to an additive constant) and isolated-mass (multipole) BCs are **not**
> implemented — use `zero` Dirichlet faces for isolated collapse problems (as in
> `inputs/collapse_be.in`).

## Jeans-length refinement

The port adds a refinement criterion that keeps the local Jeans length resolved by at
least `njeans` cells (Truelove et al. 1997), which is essential for suppressing
artificial fragmentation during collapse:

```
<parthenon/mesh>
refinement = adaptive

<refinement>
type   = jeans
njeans = 8            # refine when lambda_J / dx < njeans (typically 8-16)
```

A block is flagged for refinement when $\lambda_J/\Delta x < N_{\rm Jeans}$ and for
derefinement when $\lambda_J/\Delta x > 2.5\,N_{\rm Jeans}$, with
$\lambda_J = 2\pi c_s/\sqrt{\rho}$ (hydro) or $2\pi(c_s+v_A)/\sqrt{\rho}$ (MHD) in
$4\pi G=1$ units.

## Example problems

Two problem generators exercise the solver and ship with matching input decks:

- **`jeans`** (`src/pgen/jeans.cpp`, `inputs/jeans.in`) — a uniform periodic box with a
  small sinusoidal density perturbation, used to validate the linear Jeans dispersion
  relation. The deck runs the Jeans-unstable regime; comment/uncomment `cs` for the
  stable one.
- **`collapse_be`** (`src/pgen/collapse_be.cpp`, `inputs/collapse_be.in`) — collapse of
  a marginally-stable Bonnor-Ebert sphere, driven to first-hydrostatic-core densities.
  The setup is posed entirely in code units, in the normalization of Tomida (2011):
  $4\pi G=1$, isothermal sound speed $c_s=1$, and central density of the *critical*
  Bonnor-Ebert sphere $=1$, which fixes its radius at $6.45$ and its mass at $197.561$.
  The only free parameters are then the central density `f`, the barotropic stiffening
  density `rhocrit`, the amplitude `amp` of an $m=2$ perturbation and the rotation rate
  `omegatff` in units of $1/t_{\rm ff}$; the shipped deck corresponds to a 6 M$_\odot$
  cloud at 10 K with $\rho_{\rm crit}=10^{-13}\,$g cm$^{-3}$.

  Note that despite `<hydro> eos = adiabatic`, this setup is **not** an ideal-gas run:
  its problem-specific unsplit source term overwrites the thermal energy every stage
  with a barotropic equation of state, $e_{\rm th} = \rho/(\gamma-1)\sqrt{1 +
  (\rho/\rho_{\rm crit})^{2(\gamma-1)}}$, which is exactly isothermal below
  $\rho_{\rm crit}$ and stiffens to an adiabat of index $\gamma$ above it. `gamma`
  therefore only sets the stiff branch. The same source also zeroes the momentum outside
  the sphere radius, i.e. it imposes a fixed-velocity boundary on the ambient medium.

## Validation

### Jeans dispersion relation

The `jeans` regression test (`tst/regression/test_suites/jeans`) seeds a small
sinusoidal perturbation with zero initial velocity and checks the evolution of the
total kinetic energy against linear theory,

$$\omega^2 = k^2 c_s^2 - 4\pi G\rho_0,$$

for both the stable ($\omega^2>0$, oscillating) and unstable ($\omega^2<0$, growing)
regimes. The measured oscillation frequency and growth rate agree with the analytic
dispersion relation to within a few percent.

![Jeans dispersion validation](self_gravity/jeans_dispersion.png)

### Bonnor-Ebert collapse and cross-code comparison

The `collapse_be` setup collapses to first-core densities and, in a matched-physics
comparison, tracks the reference CT-MHD code Athena++ on the same problem. The peak
density runs away by $>5$ orders of magnitude and crosses $\rho_{\rm crit}$ at nearly
the same time in both codes, while the Jeans-length AMR criterion progressively adds
refinement levels as the core forms.

![Bonnor-Ebert collapse: peak-density evolution vs Athena++](self_gravity/collapse_be_density_evolution.png)
