# Spherical precipitator

The `precipitator_spherical` problem generator initializes a spherical atmosphere
on a uniform Cartesian mesh. Density, pressure, and a fixed gravitational potential
come from a radial hydrostatic profile. Optional components add a force-free magnetic
field, Fourier-Bessel density perturbations, power-law cooling, proportional feedback
heating, a hydrostatic outer buffer, and an outer velocity sponge.

The center is fixed at the coordinate origin; there is no input parameter for moving
it. The intended setup is therefore a three-dimensional box symmetric about zero in
all directions.

## Requirements and quick start

This problem requires uniform Cartesian coordinates and the GLM-MHD fluid layout.
The generator checks both requirements at startup. A suitable hydro block is:

```ini
<job>
problem_id = precipitator_spherical

<hydro>
fluid = glmmhd
gamma = 1.6666666666666667
eos = adiabatic
riemann = lhlld
reconstruction = plm
first_order_flux_correct = true

He_mass_fraction = 0.25
```

Begin with the [small input deck](../inputs/precipitator_spherical_smoke.in), supply a
local hydrostatic profile, and launch from the repository root:

```sh
cp inputs/precipitator_spherical_smoke.in spherical_smoke.in
./build-host/bin/athenaPK -i spherical_smoke.in \
  precipitator/hse_profile_filename=/absolute/path/to/profile.txt \
  precipitator/force_free_bfield_gauss=0.0
```

The smoke deck runs five cycles without cooling, heating, a sponge, an outer buffer,
or density perturbations. It still reads `inputs/force_free_params.txt`; its negative
`force_free_bfield_gauss` value leaves the raw force-free amplitude enabled. The
command above overrides it so that the first test has no magnetic field.

The fuller `inputs/precipitator_spherical.in` deck enables cooling, heating, a buffer,
a sponge, perturbations, and openPMD output. OpenPMD is enabled by AthenaPK's default
configuration; if it was disabled at build time, change those output blocks to HDF5
or rebuild with openPMD support.

As with the plane-parallel setup, file paths are resolved relative to the directory
where AthenaPK is launched. The large profile tables named by the example decks are
external inputs and may not be present in a fresh checkout.

## Mesh and units

Use a centered Cartesian domain. The reference setup uses periodic boundaries and
stabilizes the outer part of the sphere with a buffer and sponge:

```ini
<parthenon/mesh>
nghost = 4

nx1    = 64
x1min  = -120.0
x1max  =  120.0
ix1_bc = periodic
ox1_bc = periodic

nx2    = 64
x2min  = -120.0
x2max  =  120.0
ix2_bc = periodic
ox2_bc = periodic

nx3    = 64
x3min  = -120.0
x3max  =  120.0
ix3_bc = periodic
ox3_bc = periodic
```

For a centered cube with half-width `L`, the problem's nominal outer radius is `L`,
the radius of the largest sphere inscribed in the box. Buffer and sponge radii are in
code length. The reference units make one code length a kpc and one code time a Myr:

```ini
<units>
code_length_cgs = 3.086e21
code_mass_cgs   = 1.98841586e33
code_time_cgs   = 3.15576e13
```

## Hydrostatic profile

`hse_profile_filename` uses the same eight-column reader described in the
[plane-parallel profile guide](precipitator.md#hydrostatic-profile-table). For this
problem, column 1 is spherical radius in cm. Density and pressure are in CGS, and the
gravitational potential is in cm^2 s^-2. The acceleration column is used to extend
hydrostatic balance through the optional outer buffer. The profile's magnetic-field
column is parsed but is not used; the spherical magnetic field is configured
separately.

The file must have one header line followed by at least three numeric rows sorted by
increasing radius. Do not put blank or comment lines after the header. Keep density
and pressure positive and make their hydrostatic balance consistent with the
potential.

The current interpolator is not safe at or above the final tabulated radius. Make
that radius strictly greater than every radius queried at a cell center or face,
including ghost zones. This coverage must reach the ghost-zone cube corners, not
just the nominal outer radius `L`; enabling the outer buffer does not remove the
requirement because the potential is still sampled throughout the mesh. For the
64^3, `nghost = 4`, `[-120, 120]^3` example above, the largest queried face radius
is about 231.7 code lengths; choose a final table radius above that, such as 232 code
lengths expressed in cm.

## Minimal problem block

The following block disables all optional evolution terms and explicitly removes the
magnetic field:

```ini
<cooling>
enable_cooling = none

<precipitator>
hse_profile_filename = /absolute/path/to/profile.txt

force_free_param_file = inputs/force_free_params.txt
force_free_bfield_gauss = 0.0

enable_powerlaw_cooling = 0
enable_heating = none
h_smooth_heatcool = 1.0

outer_sponge_inner_radius = 120.0
outer_sponge_tau = 0.0
enable_outer_buffer_halo = 0
outer_buffer_inner_radius = 120.0

enable_fourier_bessel_perturbations = 0
perturbation_sigma = 0.0

# A positive default would stop after a sufficiently nonlinear overdensity.
density_contrast_stop_threshold = -1.0
```

The spherical problem has its own power-law cooling source. Keep AthenaPK's generic
`cooling/enable_cooling = none` unless combining two independent cooling sources is
intentional.

## Force-free magnetic field

`force_free_param_file` is always opened, even when the requested magnetic field is
zero. It must contain an `alpha` assignment; comments beginning with `#` are allowed:

```text
# alpha is in inverse code length
alpha=4.493409457907e-02
```

Only `alpha` is currently read from this file. An `amplitude=` line may be present but
does not set the field normalization. Use `force_free_bfield_gauss` in the input deck:

- `0.0` disables the magnetic field;
- a positive value normalizes the small-radius field to that strength in gauss;
- a negative value retains an internal raw amplitude of one in code units and does
  **not** disable the field.

For a nonzero requested field, `alpha` must also be nonzero. With one code length equal
to one kpc, the supplied `alpha` is interpreted in kpc^-1.

## Power-law cooling and feedback heating

The spherical cooling source follows `n_H^2 Lambda`. A typical setup is:

```ini
<cooling>
enable_cooling = none

<precipitator>
enable_powerlaw_cooling = 1
powerlaw_lambda_cgs = 1.0e-22

enable_heating = magic
thermostat_temperature = 1.0e7
thermostat_Kp = 10.0
h_smooth_heatcool = 10.0
magic_heating_max_eint_fraction = 0.1
magic_heating_temp_ceiling_factor = 2.0
```

`powerlaw_lambda_cgs` is in erg cm^3 s^-1. Cooling can be disabled independently of
feedback heating with `enable_powerlaw_cooling = 0`; the feedback still uses the
power-law coefficient to define the background cooling time.

The feedback source compares the shell-averaged temperature with
`thermostat_temperature`. `thermostat_Kp` is the proportional gain. A zero gain
disables feedback even when `enable_heating = magic`.

`h_smooth_heatcool` is in code length and applies a central `tanh(r / h)^4` taper.
The two optional limiters are disabled when set to zero:

- `magic_heating_max_eint_fraction` limits positive heating in one step to a fraction
  of the current thermal energy;
- `magic_heating_temp_ceiling_factor` prevents heating above a multiple of the local
  hydrostatic temperature.

## Outer buffer and sponge

The buffer extends hydrostatic initial conditions beyond a matching radius and
smoothly turns off cooling and heating near the nominal outer radius:

```ini
<precipitator>
enable_outer_buffer_halo = 1
outer_buffer_inner_radius = 100.0
outer_buffer_entropy_slope = 0.0
```

At `outer_buffer_inner_radius`, density and pressure are matched to the input profile.
Outside that radius, the setup integrates hydrostatic balance with
`K(r) proportional to r^outer_buffer_entropy_slope`. The slope must be nonnegative.
Initial density perturbations are not applied in the buffer.

The independent sponge damps momentum outside its inner radius while preserving
thermal energy:

```ini
<precipitator>
outer_sponge_inner_radius = 100.0
outer_sponge_tau = 5.0
```

`outer_sponge_tau` is in code time. A nonpositive value disables the sponge. Both
features are most useful when their inner radius is smaller than the nominal outer
radius.

## Fourier-Bessel density perturbations

Enable the multiplicative density perturbation with:

```ini
<precipitator>
enable_fourier_bessel_perturbations = 1
perturbation_sigma = 0.01
perturbation_lmax = 12
perturbation_radial_modes = 16
perturbation_kmin = 1.0
perturbation_kmax = 16.0
perturbation_seed = 88172645463393265
```

`perturbation_lmax` controls angular complexity and must be between 0 and 32.
`perturbation_radial_modes` must be positive when perturbations are enabled. The
wavenumber interval is dimensionless on the normalized radial coordinate. Store a
large seed as a decimal string, as in the example, so it is not truncated to a
32-bit integer.

Setting either the enable flag or `perturbation_sigma` to zero disables the
perturbation. `perturbation_small_kr_threshold` defaults to `1.0` and only selects the
small-argument Bessel evaluation; most users should leave it unchanged.

## Diagnostics and stopping condition

The problem stops after the maximum density contrast relative to a shell average
exceeds `density_contrast_stop_threshold`. Its default is `10.0`; set it to a
nonpositive value to disable this automatic stopping condition.

Shell averages used by feedback, diagnostics, and the stopping condition use
`parthenon/mesh/nx1` radial bins from the origin through the largest cube-corner
radius. Increase `nx1` when these profiles need finer radial resolution.

For source-term debugging, set:

```ini
<precipitator>
source_diagnostics = 1
source_diag_min_hse_dv_kms = 100.0
```

This reports cells where the hydrostatic source produces a velocity increment above
the specified km s^-1 threshold.

Useful output fields include `density_hse`, `pressure_hse`, `grav_phi`, `divB`,
`temperature_K`, `tcool_myr`, `delta_rho_over_rho_bar`,
`delta_pressure_over_pressure_bar`, `delta_entropy_over_entropy_bar`,
`delta_temperature_over_temperature_bar`, `dv1_kms`, `dv2_kms`, `dv3_kms`,
`mach_sonic`, and `plasma_beta`. Despite its name, `entropy_K` is the code-unit proxy
`P / rho^gamma`.

For a new profile, validate in stages: first run the minimal setup, check hydrostatic
drift and `divB`, then enable the magnetic field, perturbations, buffer and sponge,
and finally cooling and heating. This makes unit or profile inconsistencies much
easier to isolate.
