# Star particles: star formation and supernova feedback

The `stars` package forms star particles stochastically from dense gas, moves them,
and returns mass, energy and momentum to the gas through type II and type Ia
supernovae (SNe). Star formation follows SMUGGLE
([Marinacci et al. 2019](https://ui.adsabs.harvard.edu/abs/2019MNRAS.489.4233M)),
and the feedback coupling follows FIRE-2
([Hopkins et al. 2018](https://ui.adsabs.harvard.edu/abs/2018MNRAS.477.1578H)).

Each star particle represents a stellar population (with a Chabrier IMF), not a
single star.

- [Quick start](#quick-start)
- [Requirements](#requirements)
- [Star formation](#star-formation)
- [Star transport](#star-transport)
- [Supernova feedback](#supernova-feedback)
- [Particle IDs, restarts and AMR](#particle-ids-restarts-and-amr)
- [Outputs](#outputs)
- [Parameter reference](#parameter-reference)
- [Test problems and regression tests](#test-problems-and-regression-tests)
- [Known limitations](#known-limitations)

## Quick start

```ini
# A <units> block is required (see Requirements)
<units>
code_length_cgs = 3.085677580962325e+21   # kpc
code_mass_cgs   = 1.98841586e+33          # Msun
code_time_cgs   = 3.15576e+13             # Myr

<hydro>
He_mass_fraction = 0.25        # required for SN feedback and the Cen-Ostriker criterion
Tfloor           = 10.0        # recommended with SN feedback (internal-energy floor)

<parthenon/mesh>
nghost = 3                     # >= SN_injection_radius_cells + 1 with SN feedback

<stars>
enabled              = true
sf_density_threshold = 1.0e8   # required, code density units
transport_mode       = none    # required: gravity | advection | none
SN_II_enabled        = true
SN_Ia_enabled        = true
```

All other parameters have defaults, listed in the
[parameter reference](#parameter-reference).

## Requirements

The package checks these at startup and stops with a message if one is missing.

| Requirement | When | Why |
|---|---|---|
| A `<units>` block | always | star formation uses G in code units |
| `<units>` and `hydro/He_mass_fraction` | SN feedback, or `sf_virial_criterion = cenostriker` | hydrogen number density, gas temperature |
| `stars/sf_density_threshold` > 0 | always | star formation is always on once stars are enabled |
| `stars/transport_mode` set | always | no default: `advection` is for testing only |
| `parthenon/mesh/nghost` ≥ `SN_injection_radius_cells` + 1 | SN feedback | the SN kernel reads ghost cells |
| double-precision build | always | particle-ID offsets are stored bit-exactly in a `Real` field |

Only 2D and 3D meshes are supported (not checked).

## Star formation

Star formation runs once per cycle, after the hydro update, in every interior
cell with $\rho > \rho_\mathrm{th}$ (`sf_density_threshold`) that passes the
optional virial criterion.

**Rate.** A cell of gas mass $M = \rho V$ forms stars at
$\dot M_\star = \epsilon\, M / t_\mathrm{ff}$, with
$t_\mathrm{ff} = \sqrt{3\pi / (32 G \rho)}$ and $\epsilon$ = `sf_efficiency`
(efficiency per free-fall time).

**Stochastic draw.** A star takes the fraction $\epsilon_m$ = `sf_mass_efficiency`
of the cell's gas, so per step a cell forms a star with probability

$$P = \min\left(1, \frac{1 - e^{-\dot M_\star \Delta t / M}}{\epsilon_m}\right),$$

the partial-conversion form of the SMUGGLE probability
([Springel & Hernquist 2003](https://ui.adsabs.harvard.edu/abs/2003MNRAS.339..289S)).
The expected mass formed per step is then $\dot M_\star \Delta t$, matching the
`star_formation_rate` history output. A cell forms at most one star per step. The
draws use a deterministic hash keyed on cell, block, time and population, so a
run is reproducible.

**New star.** The star sits at the cell center, with mass $\epsilon_m \rho V$, the
cell's velocity (all three components, also in 2D), and its birth time and
position. The gas keeps its velocity and its passive-scalar concentrations.
`sf_energy_mode` sets what happens to the gas energy:

| `sf_energy_mode` | Gas energy after formation |
|---|---|
| `isobaric` (default) | removes only the kinetic energy of the converted mass; pressure is unchanged, so the remaining gas is hotter |
| `isothermal` | removes the converted fraction of kinetic and internal energy; temperature is unchanged |

The magnetic field is never modified.

### Virial criteria

With `sf_virial_criterion_enabled = true`, a cell must also pass one of these
gravitational-collapse criteria:

| `sf_virial_criterion` | Cell forms stars if |
|---|---|
| `hopkins` (default) | $\alpha = \dfrac{\lVert\nabla \mathbf v\rVert^2 + (c_s/\Delta x)^2}{8 \pi G \rho} \le \alpha_\mathrm{crit}$ ([Hopkins et al. 2013](https://ui.adsabs.harvard.edu/abs/2013MNRAS.432.2647H)) |
| `hopkinsalfven` | as `hopkins`, with $c_s^2 \to c_s^2 + v_A^2$, $v_A^2 = B^2/\rho$ |
| `cenostriker` | $\nabla\cdot\mathbf v < 0$, $T <$ `sf_temperature_threshold` and $M > M_J$ ([Cen & Ostriker 1992](https://ui.adsabs.harvard.edu/abs/1992ApJ...399L.113C)) |

Here $\lVert\nabla \mathbf v\rVert$ is the Frobenius norm of the velocity-gradient
tensor (central differences), $c_s$ the adiabatic sound speed, $\Delta x$ the
geometric-mean cell size, $\alpha_\mathrm{crit}$ = `sf_alpha_crit`, and
$M_J = \pi^{5/2} c_s^3 / (6\, G^{3/2} \rho^{1/2})$ the Jeans mass.

## Star transport

`transport_mode` sets how stars move:

| `transport_mode` | Motion |
|---|---|
| `gravity` | kick-drift-kick leapfrog in a static spherical potential registered by the problem generator as `gravity_field` (e.g. `cluster`); sub-steps resolve the local dynamical time $\sqrt{r/g}$ |
| `advection` | moves with the gas, like tracers (testing only) |
| `none` | stars stay at their birth position |

Stars do not feel the gas or each other.

## Supernova feedback

Enable the channels with `SN_II_enabled` and `SN_Ia_enabled`. Each step, every
star draws its number of SNe and couples their ejecta to the surrounding gas.

### Event rates

- **Type II.** Progenitors of 8–100 M☉ whose lifetime ends during the step, from
  the [Portinari et al. (1998)](https://ui.adsabs.harvard.edu/abs/1998A%26A...334..505P)
  lifetime and remnant-mass tables at solar metallicity and a
  [Chabrier (2003)](https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C) IMF
  normalized over 0.1–100 M☉ (about one SN II per 85 M☉ formed, between roughly
  3 and 40 Myr). The ejecta are the progenitor mass minus the remnant mass.
- **Type Ia.** The delay-time distribution of
  [Maoz, Mannucci & Brandt (2012)](https://ui.adsabs.harvard.edu/abs/2012MNRAS.426.3282M)
  as in SMUGGLE: $2.6\times10^{-3}$ SNe per M☉ after 40 Myr, with slope $-1.12$,
  and 1.37 M☉ of ejecta per event.

The number of SNe is a Poisson draw (exact for any mean) keyed on the star's birth
time and position and the current time, so it does not depend on particle IDs or
thread scheduling. When the ejecta would exceed the star's remaining mass, they are
reduced to that mass and the star is removed.

### Coupling to the gas

The ejecta are spread over the cells around the star with a cubic-spline kernel of
smoothing length $h = (r+1)\Delta x/2$, with $r$ = `SN_injection_radius_cells` and
$\Delta x$ the cell size of the star's host block (support $2h$). In the star's rest
frame, cell $b$ receives

| Quantity | Amount |
|---|---|
| mass | $\Delta m_b = s_b M_\mathrm{ej}$ |
| total energy | $\Delta E_b = s_b E_\mathrm{SN}$, with $E_\mathrm{SN} = N_\mathrm{SN}\,$`E_SN_per_event` |
| momentum | $\Delta \mathbf p_b = \bar{\mathbf w}_b\, p_\mathrm{SN} \min\left(\sqrt{1 + m_b/\Delta m_b},\ p_t/p_\mathrm{SN}\right)$ |

with $p_\mathrm{SN} = \sum_\mathrm{II,\,Ia} \sqrt{2 N E_\mathrm{SN,event} M_\mathrm{ej}}$
and the terminal momentum
$p_t = 4.8\times10^5\,\mathrm{M_\odot\,km\,s^{-1}}\, N_\mathrm{SN}^{13/14}\, \langle n_H\rangle^{-1/7}$,
where $\langle n_H\rangle$ (cm⁻³) is the kernel-averaged hydrogen density.

- **Weights.** $\bar{\mathbf w}_b$ are the FIRE-2 tensor-corrected vector weights,
  which sum to zero, and $s_b = |\bar{\mathbf w}_b|$. A star exactly at a cell center
  gives that cell a share of mass and energy only, i.e. heat.
- **Momentum boost.** The factor $\sqrt{1 + m_b/\Delta m_b}$ accounts for the
  unresolved Sedov–Taylor phase: it is the momentum gained by sweeping up the cell
  mass $m_b$ at constant energy, and $p_t$ caps it once radiative losses dominate.
  The boost does not add energy: the total energy coupled is always $E_\mathrm{SN}$.
- **Frame.** Mass, momentum and energy are boosted to the simulation frame with the
  star velocity. The split into kinetic and thermal energy follows from the hydro
  variables.
- **Gas state.** $m_b$ and $\langle n_H\rangle$ come from a density snapshot taken
  before the step's deposits, so overlapping events don't depend on their order.
- **Energy floor.** After all deposits, the gas internal energy is floored at the
  hydro `pfloor`/`Tfloor` (or the smallest positive number if neither is set). The
  floor can be needed where gas moves fast relative to the star. Set a floor when
  using SN feedback.

### Kernels crossing block boundaries

When a kernel reaches into neighboring blocks, the host block deposits its own
share and sends each neighbor's share as a temporary `ghost_stars` particle, which
the neighbor deposits after the swarm communication. At the same refinement level
the split is exact. Across refinement levels each region's totals are conserved,
but the distribution is approximate.

After star formation and feedback, the gas ghost cells are exchanged again and the
next timestep is estimated from the resulting state, so a SN-heated cell limits
the next timestep.

## Particle IDs, restarts and AMR

- **IDs.** On uniform and statically refined meshes every star gets a unique ID.
  With `parthenon/mesh/refinement = adaptive`, IDs cannot be kept unique, so every
  star gets the ID `UINT64_MAX` and a warning is printed at startup. IDs are labels
  only: nothing in the physics depends on them.
- **Restarts.** All star data are written to restart files. The one-time setup (ID
  ranges and optional initial seeding) runs only on the first start, so a restart
  neither duplicates seeded stars nor reuses IDs.

## Outputs

### History

| Column | Meaning (code units) |
|---|---|
| `star_formation_rate` | instantaneous $\sum \dot M_\star$ over cells passing the density and virial criteria, i.e. the expected formation rate |
| `sn_ii_power`, `sn_ia_power` | SN energy coupled during the last step divided by its timestep (only with the channel enabled) |

### Star particles

The `stars` swarm carries the standard positions and `id`, plus:

| Variable | Meaning |
|---|---|
| `mass` | current mass (decreases with SN ejecta) |
| `birth_mass` | mass at formation |
| `injection_time` | formation time |
| `birth_x`, `birth_y`, `birth_z` | formation position |
| `v_x`, `v_y`, `v_z` | velocity |

To write them to HDF5 outputs:

```ini
<parthenon/output1>
file_type = hdf5
dt        = 1.0
swarms    = stars
stars_variables = id, x, y, z, mass, birth_mass, injection_time
```

## Parameter reference

All parameters below go in the `<stars>` block.

| Parameter | Default | Description |
|---|---|---|
| `enabled` | `false` | enable the package |
| `sf_density_threshold` | — (required) | density above which cells can form stars, code units |
| `sf_efficiency` | `0.01` | star formation efficiency per free-fall time $\epsilon$ (> 0) |
| `sf_mass_efficiency` | `0.5` | fraction of the cell gas turned into a new star $\epsilon_m$, in (0, 1) |
| `sf_energy_mode` | `isobaric` | gas energy update on formation: `isobaric` or `isothermal` |
| `sf_virial_criterion_enabled` | `false` | apply a virial criterion |
| `sf_virial_criterion` | `hopkins` | `hopkins`, `hopkinsalfven` or `cenostriker` |
| `sf_alpha_crit` | `1.0` | critical virial parameter (`hopkins`, `hopkinsalfven`) |
| `sf_temperature_threshold` | `1.0e4` | temperature ceiling in K (`cenostriker`) |
| `transport_mode` | — (required) | `gravity`, `advection` (testing only) or `none` |
| `seed_stars` | `false` | call the problem generator's initial star seeding on the first start |
| `SN_II_enabled` | `false` | type II SN feedback |
| `SN_Ia_enabled` | `false` | type Ia SN feedback |
| `E_SN_per_event` | `1.0e51` | energy per SN, erg |
| `SN_injection_radius_cells` | `2` | kernel radius in host cells: $2h = (r+1)\Delta x$ |
| `SN_test_event_age` | `-1` (off) | testing only: replaces the SN II rate by exactly one event per star at this age (code time units), with the IMF-mean ejecta of 8–100 M$_\odot$ stars; requires `SN_II_enabled` |

## Test problems and regression tests

- **`star_formation` problem generator** (`problem_id = star_formation`): uniform gas
  of density `rho_bg` and temperature `T_bg` (K) with overdense cells of density
  `rho_peak` in pressure equilibrium. These are set in `<problem/star_formation>`:
  either one peak at (`x_peak`, `y_peak`, `z_peak`) with `ic_mode = single_peak`,
  or `n_peaks` random peaks per block with `ic_mode = multi_peak` (`rng_seed`).
  `ic_mode = uniform` sets the background only; with `stars/seed_stars = true` it
  then places one star of `star_mass` (M$_\odot$) at rest at (`star_x`, `star_y`,
  `star_z`). Combined with `SN_test_event_age = 0`, this gives a single supernova
  blast wave to compare with the Sedov–Taylor solution.
- **Cluster star seeding** (`stars/seed_stars = true` with `problem_id = cluster`):
  zero-mass stars on circular orbits within `r_max` of the cluster center, for
  testing `transport_mode = gravity`. Parameters `n_stars` (per MPI rank), `r_max`
  and `rng_seed` go in `<problem/cluster/seed_stars>`. For a custom seeding
  routine, see [user problem generators](user_pgen.md#star-particles).
- **Regression tests**, currently run serially:
  - `star_formation_sfr_stats`: formation-time statistics against the expected rate;
  - `sn_feedback_refinement`: mass/momentum/energy budgets of SN events in the
    interior, at block faces, edges and corners, and across a refinement boundary;
  - `cluster_star_orbits`: orbits under `transport_mode = gravity`.

## Known limitations

- **Metallicity.** The SN II lifetime and yield tables are for solar metallicity
  only; a warning is printed with tabular cooling.
- **`E_SN_per_event`.** The terminal momentum does not scale with
  `E_SN_per_event` (it assumes $10^{51}$ erg per SN).
- **Momentum residual.** Below the terminal-momentum cap, the per-cell boost leaves
  a small net momentum (about 1% of $p_\mathrm{SN}$) for a star that is off-center
  in its cell.
- **Large SN counts.** Very massive star particles with long timesteps (thousands
  of SNe per step) are handled correctly, but the feedback is then a coarse,
  near-continuous source and the Poisson draw becomes slow.
- **Not yet tested.** Stars with adaptive mesh refinement, and stars on GPU builds.
