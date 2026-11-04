# Plane-parallel precipitator

The `precipitator` problem generator initializes a plane-parallel atmosphere in
hydrostatic equilibrium. The atmosphere is symmetric about `x3 = 0`; `x1` and `x2`
are the horizontal directions and `x3` is the stratified vertical direction.

The setup supports a magnetic field read from the hydrostatic profile, isobaric
density perturbations, tabular cooling, proportional feedback heating, and
Ornstein-Uhlenbeck velocity driving. Gravity is applied with a well-balanced source
term constructed from the tabulated gravitational potential.

## Quick start

Start with the [MHD example input deck](../inputs/precipitator_mhd_lowres.in) and run
AthenaPK from the repository root:

```sh
cp inputs/precipitator_mhd_lowres.in precipitator.in
./build-host/bin/athenaPK -i precipitator.in
```

Before running, change `precipitator/hse_profile_filename` in `precipitator.in` to a
profile file that exists locally. The large profile tables named by some example
decks are external inputs and may not be present in a fresh checkout.

Paths in an AthenaPK input file are resolved relative to the process working
directory, not relative to the input file. The command above therefore makes paths
such as `inputs/precipitator.cooling` resolve from the repository root. An absolute
path is the least ambiguous choice on a cluster.

The current generator initializes magnetic components from the profile, so use the
GLM-MHD state layout. A hydrodynamic-like calculation can use `fluid = glmmhd` with
zero magnetic field in the profile.

## Mesh and solver setup

Use a domain centered on `x3 = 0`, periodic horizontal boundaries, and the custom
precipitator reflection condition on both vertical boundaries:

```ini
<job>
problem_id = precipitator

<parthenon/mesh>
nghost = 4

x1min  = -50.0
x1max  =  50.0
ix1_bc = periodic
ox1_bc = periodic

x2min  = -50.0
x2max  =  50.0
ix2_bc = periodic
ox2_bc = periodic

x3min  = -100.0
x3max  =  100.0
ix3_bc = precipitator_reflect_x3
ox3_bc = precipitator_reflect_x3

<hydro>
fluid = glmmhd
gamma = 1.6666666666666667
eos = adiabatic
riemann = lhlld
reconstruction = plm
first_order_flux_correct = true

He_mass_fraction = 0.25
```

The custom boundary condition reflects the normal momentum at the `x3` faces. Do
not copy the older `ix3_bc = user` spelling; use the registered
`precipitator_reflect_x3` name.

Choose `<units>` before preparing the remaining values. The reference decks use one
kpc, one solar mass, and one Myr as their code units:

```ini
<units>
code_length_cgs = 3.086e21
code_mass_cgs   = 1.98841586e33
code_time_cgs   = 3.15576e13
```

Mesh coordinates, times, velocities, and most runtime controls are in these code
units unless a parameter name or the tables below state otherwise.

## Hydrostatic profile table

`hse_profile_filename` names a whitespace-separated text file. Its first line is
always consumed as a header. Every subsequent line must contain at least eight
numeric columns; blank or comment lines are not accepted after the header.

| Column | Quantity | Expected units | Use in this problem |
| --- | --- | --- | --- |
| 1 | height `z` | cm | interpolation coordinate, queried at `abs(x3)` |
| 2 | mass density `rho` | g cm^-3 | initial density |
| 3 | thermal pressure `P` | erg cm^-3 | initial pressure |
| 4 | gravitational acceleration `g` | cm s^-2 | loaded by the shared profile reader |
| 5 | number density `n` | not interpreted | required placeholder, otherwise unused |
| 6 | entropy proxy `K` | not interpreted | required placeholder, otherwise unused |
| 7 | gravitational potential `phi` | cm^2 s^-2 | well-balanced gravity source |
| 8 | magnetic field `B` | gauss | initial `B1`; set to zero for a hydro-like run |

Provide at least three rows sorted by increasing height, with positive density and
pressure. The pressure, density, and potential should describe the same hydrostatic
solution. The current interpolator is not safe at or above the final tabulated
height. Make that height strictly greater than every `abs(x3)` queried at a cell
center or `x3` face, including ghost zones; ending the table exactly at the largest
query is not sufficient. With `uniform_init = 1`, it must instead be strictly
greater than `uniform_init_height`.

A typical header is:

```text
# z rho(z) P(z) g(z) n(z) K(z) phi(z) B(z)
```

The helper `scripts/plot_hse_gravity.py PROFILE` plots the tabulated acceleration
from column 4 against the coordinate in column 1. It does not read or differentiate
the potential column, so validate the consistency of `g` and `phi` separately.

## Core precipitator parameters

The following block contains every parameter read unconditionally by the generator:

```ini
<precipitator>
hse_profile_filename = /absolute/path/to/profile.txt

# 0: initialize the stratified profile; 1: use one profile sample everywhere
uniform_init = 0

enable_heating = none
thermostat_temperature = 1.0e7
thermostat_Kp = 10.0
h_smooth_heatcool_kpc = 5.0

# Set the amplitude to zero to disable initial density perturbations.
perturb_sin_drho_over_rho = 0.01
perturb_kx = 16
perturb_ky = 16
perturb_kz = 32
perturb_exponent = 1
```

`uniform_init = 0` samples the tabulated atmosphere at each cell and enables
gravity. `uniform_init = 1` disables gravity and initializes the whole box from
`uniform_init_height`; that conditional parameter is expressed in the profile's
first-column units (cm), not in code length.

The density perturbation is isobaric: density is multiplied by `1 + delta` while
the tabulated pressure is retained. `perturb_kx`, `perturb_ky`, and `perturb_kz`
set the largest integer mode index in each direction. `perturb_exponent` controls
the spectral weighting; larger values suppress small-scale modes more strongly.

`h_smooth_heatcool_kpc` is always specified in physical kpc. The factor
`tanh(abs(x3) / h_smooth)^4` tapers feedback heating and velocity driving and is
also included in the `tcool_over_tff` diagnostic. It does not taper the tabular
radiative source: `cooling/enable_cooling = tabular` cools every cell at the full
tabulated rate, including near `x3 = 0`.

Although `enable_heating` defaults to `none`, `thermostat_temperature`,
`thermostat_Kp`, and `h_smooth_heatcool_kpc` are currently required even when
feedback heating is disabled.

## Cooling and feedback heating

Tabular cooling is configured through AthenaPK's regular `<cooling>` block. The
table contains exactly two numeric columns, `log10(T / K)` and
`log10(Lambda / (erg cm^3 s^-1))`, apart from comment and blank lines.

```ini
<cooling>
enable_cooling = tabular
table_filename = inputs/precipitator.cooling
lambda_units_cgs = 1.0
integrator = rk12
cfl = 1.0
```

The supplied `inputs/precipitator.cooling` is a minimal constant-cooling example;
replace it with the intended physical cooling curve for production work. See
[Configuring solvers in the input file](input.md#cooling) for integrator and table
requirements.

Set `precipitator/enable_heating = magic` to enable proportional feedback. At each
height, the source term uses the horizontally averaged temperature error relative to
`thermostat_temperature` and scales it with `thermostat_Kp` and the background
cooling time. This mode requires tabular cooling because it uses the loaded cooling
curve even when evaluating the feedback term.

Set both of the following to disable radiative source terms for an initial smoke
test:

```ini
<cooling>
enable_cooling = none

<precipitator>
enable_heating = none
```

## Optional velocity driving

Velocity driving is disabled when `sigma_v <= 0`. When it is enabled, add:

```ini
<precipitator/driving>
sigma_v = 0.0001
k_peak = 2.0
num_modes = 40
sol_weight = 1.0
vertical_driving_only = false
rseed = 42
t_corr = 500.0
```

`sigma_v` is the target driving amplitude in code velocity, `t_corr` is in code
time, and `k_peak` is the normalized peak wavenumber. `sol_weight` ranges from zero
for compressive modes to one for solenoidal modes. With
`vertical_driving_only = true`, only the `x3` velocity is driven.

The older example parameters `thermostat_Ki`, `numHist`, and
`precipitator/driving/max_height` are not read by the current implementation.

## Outputs and first-run checks

Useful derived fields include:

- `temperature`, in kelvin;
- `entropy`, the code-unit proxy `P / rho^gamma`;
- `drho_over_rho`, `dP_over_P`, `dK_over_K`, and `dT_over_T`, relative to
  horizontal averages;
- `dv`, the velocity relative to the horizontal mean in km s^-1;
- `mach_sonic` and `plasma_beta`;
- `tcool_over_tff`, when tabular cooling is enabled;
- `accel` and `turbulent_heating`, when velocity driving is enabled.

For a new profile, first run only a few cycles with cooling, heating, density
perturbations, and velocity driving disabled. Check that density and pressure remain
close to `density_hse` and `pressure_hse`, then enable one optional component at a
time. A failure to open a profile or cooling table almost always means that its path
was interpreted relative to a different working directory.
