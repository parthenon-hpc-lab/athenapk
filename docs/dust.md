The dust module is written by Fred Jennings at UNIMORE, funded by ERC Consolidator Grant 101086804 - BlackHoleWeather, PI Massimo Gaspari. To use the dust module, please contact F. Jennings before usage.

# Model

The model follows the grain-size evolution scheme of McKinnon et al. 2018 (MNRAS 478, 2851, [arXiv:1805.04521](https://arxiv.org/abs/1805.04521)), adapted to run as a hydrodynamically passive tracer on the AthenaPK mesh.

- Grain sizes are split into `num_grainsize_bins` log-spaced bins per composition (carbonaceous and/or silicate). Each bin carries two passive scalars, the grain number density and the grain mass density.
- Within a bin, dn/da is reconstructed either linearly (McKinnon+18 eq. 31, with optional slope limiting) or log-linearly, dn/da = beta a^kappa (new in this implementation).
- Grains grow by gas-phase metal accretion (Hirashita & Kuo 2011) and shrink by thermal sputtering (Tsai & Mathews 1995), with a size-independent da/dt as in McKinnon+18 eqs. 49-50. The bins are then remapped with eqs. 34-39; mass grown past the top edge goes into the top bin (eqs. 44-46). The metallicity is fixed (Z = 0.33 Zsun) and the metal reservoir is not depleted.
- If no log-linear profile can reproduce both the number and the mass of a bin (|kappa| would exceed 45, e.g. when all grains sit near one edge), `loglinear` keeps the mass and adjusts the number, as McKinnon's slope limiting does. `hybrid_loglinear` instead treats such a bin as a delta function at its mean grain size, which keeps both.
- Dust cools the gas through electron collisions (Dwek & Werner 1981), assuming fully ionised H and He: n_H / n_e = 2X / (2X + Y), with Y = `<hydro> He_mass_fraction`.
- Optionally, AGB winds inject dust following a stellar density profile, a Salpeter IMF, main-sequence lifetimes and the Dell'Agli et al. 2017 yields (see below).

# Time integration

- `subcycle = true` (default): dust is evolved inside the tabular cooling subcycles, and the dust cooling is part of de/dt. This needs `<cooling> enable_cooling = tabular` with the `rk12` or `rk45` integrator. The `townsend` integrator is not supported, since it needs one cooling function shared by all cells.
- `subcycle = false`: dust is updated once per step as a split source term, using da/dt at the start of the step. Dust cooling is only applied through the tabular cooling, so it is inactive without it.
- `time_integrator` only applies to `subcycle = true`. `euler` uses da/dt at the start of each subcycle; `heun` uses the average of da/dt at its start and end.
- With AGB winds, half of the step's AGB dust is injected before the size update and half after it.

# Parameters

Defaults marked "required" have no default: the run stops if they are missing.

## `<dust>`

| Parameter | Default | Description |
|---|---|---|
| `active` | `false` | Enable the dust model |
| `carbonaceous_grains`, `silicate_grains` | `false` | Grain compositions to evolve (at least one) |
| `carbonaceous_grain_density`, `silicate_grain_density` | required per active composition | Material density of the grains, in g/cm^3 (e.g. 2.2 and 3.3) |
| `num_grainsize_bins` | `2` | Number of size bins |
| `grainsize_bins_low_edge`, `grainsize_bins_high_edge` | `1e-5`, `1e-1` | Edges of the size range, in micron |
| `piecewise_method` | required | `linear`, `loglinear` or `hybrid_loglinear` |
| `slope_limiting` | `true` | Slope limiting of linear bins (only used with `linear`) |
| `subcycle` | `true` | See "Time integration" |
| `time_integrator` | required | `euler` or `heun` |
| `thermal_sputtering` | `false` | Grain shrinking by sputtering |
| `sputtering_suppresion_factor` | `1` | Multiplies the sputtering rate |
| `metal_accretion` | `false` | Grain growth by accretion of gas-phase metals |
| `AGB_winds` | `false` | Dust injection by AGB winds |
| `cooling` | required | `off`, `Dwek_Werner1981` (grains at the bin midpoint) or `Dwek_Werner1981_INTEGRATED` (integrated over the reconstructed distribution) |
| `dust_cool_table_N_Tbins` | required for `Dwek_Werner1981` | `<= 0`: rates computed on the fly. `N >= 2`: lookup table of N bins from log10(T/K) = 0 to 9(N-1)/N, held constant above it |
| `disable_all_gas_cooling_for_testing` | `false` | Testing only: keep the dust cooling and drop the gas cooling |
| `init_profile` | required | `const_dtg`, `vogelsberger_19` or `stellar_profile` (see below) |
| `init_dtg_mass_ratio` | required for `const_dtg` | Initial dust-to-gas mass ratio |
| `init_run_stellar_injection_time` | required for `stellar_profile` | Duration of AGB injection used to set the initial dust, in code time |
| `carbonaceous_grain_mass_fraction`, `silicate_grain_mass_fraction` | required with both compositions | Relative weights of the two compositions in the initial dust mass (normalised by their sum) |
| `init_grainsize_distribution` | required | `MRN` (dn/da ~ a^-3.5, Mathis, Rumpl & Nordsieck 1977), `MRN_inverse` (dn/da ~ a^4.5), `flat`, or `flat_in_range` |
| `flat_graindist_in_range_amin_microM`, `flat_graindist_in_range_amax_microM` | `1`, `1` | Size range for `flat_in_range`, in micron |

Initial dust profiles:
- `const_dtg`: constant dust-to-gas ratio `init_dtg_mass_ratio`. With `<problem/cluster/uniform_gas> init_uniform_gas = true`, the ratio is `uniform_gas_dust_to_gas` instead, whatever `init_profile` is.
- `vogelsberger_19`: radial dust-to-gas ratio fit of Vogelsberger et al. 2019, capped at 1e-4, with r200 from `<problem/cluster/gravity> m_nfw_200`.
- `stellar_profile`: the dust injected by the AGB winds over `init_run_stellar_injection_time`, with the size distribution set by `init_grainsize_distribution` rather than the AGB one. Requires `AGB_winds = true`.

`vogelsberger_19` and `stellar_profile` apply `dtgfloor` (below) everywhere at initialisation.

## `<dust/AGB_Winds>` (with `AGB_winds = true`)

The stellar density is either a power law, d log rho_star / d log r = `gamma_star`, or a Prugniel-Simien profile. It is normalised so that the mass between `R_lower_in_kpc` and `R_upper_in_kpc` is `Mstar_cent_in_Msun`. The injected dust follows dn/da = C a^-5 exp(-ln^2(a / a_AGB) / (2 sigma_AGB^2)), with a_AGB = 0.1 micron.

| Parameter | Default | Description |
|---|---|---|
| `Mstar_cent_in_Msun` | required | Stellar mass of the central galaxy between the two radii below |
| `R_lower_in_kpc`, `R_upper_in_kpc` | required | Radii used for the normalisation |
| `stellar_radial_profile` | required | `power_law` or `prugniel_simien` |
| `gamma_star` | `-2.2` | Power-law slope (Cappellari et al. 2015) |
| `sersic_n`, `sersic_Re` | required for `prugniel_simien` | Sersic index, and effective radius in code length |
| `sigma_AGB` | required | Width of the injected size distribution |
| `AGB_max_radius_in_kpc` | no limit | Inject AGB dust only within this radius |

## `<dust/AGB_data_table>` (with `AGB_winds = true`)

| Parameter | Description |
|---|---|
| `AGB_table_filename` | Yield table, e.g. `src/dust/DellAgli_SolarZ_AGB.txt`. Column 0 is the initial stellar mass in Msun, the others the dust yields per star in Msun. Lines starting with `#` are skipped |
| `silicate_column_indexes_to_sum` | Columns summed into the silicate yield (`2,3`: olivine and pyroxene in the Dell'Agli table) |
| `carbonaceous_column_index` | Column of the carbonaceous yield (`7` in the Dell'Agli table) |

The dust return per unit stellar mass and time is the integral over 1.5-8 Msun of IMF(M) m_dust(M) / t_MS(M), with a Salpeter IMF on 0.3-100 Msun and t_MS = 10^4 Myr (M/Msun)^-2.5. It is printed at startup.

## `<problem/cluster/clips>`

| Parameter | Default | Description |
|---|---|---|
| `dtgfloor` | `-1` (off) | Dust-to-gas floor. Within `clip_r`, cells with dust below the floor are rescaled up to it, keeping their size distribution. Cells without dust are left empty |

## `<problem/cluster/dust_history>`

Masses and cooling rates binned in radius and gas temperature, written only with tabular cooling. Since `write_to_file` defaults to `true`, the radial and temperature ranges must be set whenever dust is active, or the output switched off.

| Parameter | Default | Description |
|---|---|---|
| `write_to_file` | `true` | Write the history files |
| `min_radius`, `max_radius` | `0` | Radial range, in code length (must satisfy min < max when writing) |
| `num_radial_bins` | `10` | Number of radial bins |
| `log_space` | `true` | Log- or linearly spaced radial bins |
| `min_T_kelvin`, `max_T_kelvin` | `0` | Temperature range, in K, always log-spaced (must satisfy min < max when writing) |
| `num_temperature_bins` | `10` | Number of temperature bins |
| `dust_history_filename` | `dust_history.dat` | Base file name; tags for the rank, partition and quantity are inserted before `.dat` |

Output, appended every step under `./dust_history/` by rank 0 after summing over ranks:
- `dust_bin_NNN/`: log10 of the dust mass (Msun) and of the dust cooling loss rate (erg/s) for bin NNN = composition * `num_grainsize_bins` + size bin, one column per (radius, temperature) bin
- `Gas_Cooling/`: log10 of the gas cooling loss rate (erg/s), one column per radial bin
- `AGB_History/` (with AGB winds): log10 of the AGB dust injected over the step and of the stellar mass, per radial bin

Radii are measured from the origin. Empty bins appear as `-inf`. With more than one mesh-data partition per rank (`<parthenon/mesh> pack_size`), every rank must have the same number of partitions, since each partition is reduced separately.

At startup, rank 0 also writes `agb_normalised_*_number_distribution_array.txt` (grains per unit injected mass in each bin) and, with a lookup table, `dust_cool_table/dust_cooling_table.txt`.

# Example

```
<dust>
active                       = true
carbonaceous_grains          = true
silicate_grains              = true
carbonaceous_grain_density   = 2.2
silicate_grain_density       = 3.3
carbonaceous_grain_mass_fraction = 1
silicate_grain_mass_fraction = 1
num_grainsize_bins           = 8
grainsize_bins_low_edge      = 5e-3
grainsize_bins_high_edge     = 0.5
piecewise_method             = hybrid_loglinear
subcycle                     = true
time_integrator              = heun
thermal_sputtering           = true
metal_accretion              = true
cooling                      = Dwek_Werner1981_INTEGRATED
init_profile                 = const_dtg
init_dtg_mass_ratio          = 1e-2
init_grainsize_distribution  = MRN

<problem/cluster/dust_history>
min_radius           = 0.001
max_radius           = 0.1
num_radial_bins      = 5
min_T_kelvin         = 1000
max_T_kelvin         = 1e9
num_temperature_bins = 5
```

For AGB injection, add `AGB_winds = true` to `<dust>` and:

```
<dust/AGB_Winds>
stellar_radial_profile = power_law
gamma_star             = -2.2
sigma_AGB              = 0.47
Mstar_cent_in_Msun     = 1e11
R_lower_in_kpc         = 0.1
R_upper_in_kpc         = 30
AGB_max_radius_in_kpc  = 30

<dust/AGB_data_table>
AGB_table_filename             = /path/to/athenapk/src/dust/DellAgli_SolarZ_AGB.txt
silicate_column_indexes_to_sum = 2,3
carbonaceous_column_index      = 7
```

# TODOs
- Shattering and coagulation
- Temperature-dependent sticking efficiencies
- More sophisticated star formation / AGB injection prescription
- Supernovae
- Metal model with gas-phase depletion by accretion and enrichment by sputtering
