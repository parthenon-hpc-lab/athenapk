The dust module is written by Fred Jennings at UNIMORE, funded by ERC Consolidator Grant 101086804 - BlackHoleWeather, PI Massimo Gaspari. To use the dust module, please contact F. Jennings before usage.

# Model
The dust model is designed off of the implementation of dust introduced in the 2018 McKinnon Arepo paper https://ui.adsabs.harvard.edu/abs/2018MNRAS.478.2851M/.

The model has been adapted to run as a hydrodynamically-passive tracer fluid on the AthenaPK mesh, and a new log-linear reconstruction scheme has also been implemented. This scheme also has a "hybrid" mode where the reconstruction switches to a delta-function at one bin-edge if the slope gets too steep.

A number of initial conditions have been implemented, both for the dust-to-gas ratio and for the initial grain-size distribution. These can be selected in the parameter file.

# Parameters
A typical parameter file addition for the dust looks like (with possible options in square braces):

<dust> <br>
active                              = [true, false]                                         -- default = false  <br>
subcycle                            = [true, false]                                         -- default = true <br>
time_integrator                     = [heun, euler]                                          <br>
max_dM_in_bin                       = 0.1                                                   -- default = -1. <br>
piecewise_method                    = [loglinear, linear]                                    <br>
cooling                             = [Dwek_Werner1981_INTEGRATED, Dwek_Werner1981, off]      <br>
dust_cool_table_N_Tbins             = [-1,  or +/ve int] <br>
disable_all_gas_cooling_for_testing = [true, false]                                         -- default = false <br>
num_grainsize_bins                  = 8                                                     -- default = 2 <br>
grainsize_bins_low_edge             = 5e-3                                                  -- default = 1e-5 <br>
grainsize_bins_high_edge            = 0.5                                                   -- default = 1e-1 <br>
carbonaceous_grains                 = [true, false]                                         -- default = false  <br>
silicate_grains                     = [true, false]                                         -- default = false <br>
metal_accretion                     = [true, false]                                         -- default = false <br>
thermal_sputtering                  = [true, false]                                         -- default = false <br>
AGB_winds                           = [true, false]                                         -- default = false <br>
carbonaceous_grain_mass_fraction    = 1  <br>
silicate_grain_mass_fraction        = 1  <br>
carbonaceous_grain_density          = 2.2   <br>
silicate_grain_density              = 3.3   <br>
init_profile                        = [const_dtg, vogelsberger_19] <br>
init_dtg_mass_ratio                 = 1e-2  <br>
init_grainsize_distribution         = [MRN, MRN_inverse, flat_in_range]  <br>
flat_graindist_in_range_amin_microM = 2e-2 <br>
flat_graindist_in_range_amax_microM = 6e-2 <br>
slope_limiting                      = [true, false]                                         -- default = true <br>
init_profile                        = [const_dtg, stellar_profile]
init_run_stellar_injection_time     = 2e-3
init_dtg_mass_ratio                 = 1e-5 #1e-2 ##5e-6
sputtering_suppresion_factor        = 1.                                                    -- default = 1    <br>

<dust/AGB_Winds>  <br>
gamma_star                          = -2.5 <br>
sigma_AGB                           = 0.47 <br>
Mstar_cent_in_Msun                  = 1e11 <br>
R_upper_in_kpc                      = 30 <br>
AGB_max_radius_in_kpc               = 30  ### Only inject dust via AGB within this radius <br>

<dust/AGB_data_table> <br>
AGB_table_filename                  = $AthenaPK_root/src/dust/DellAgli_SolarZ_AGB.txt
silicate_column_indexes_to_sum      = 2,3
carbonaceous_column_index           = 7

<problem/cluster/dust_history> <br>
write_to_file                       = [true, false]                                         -- default = true  <br>
min_radius                          = 0.001 <br>
max_radius                          = 0.1 <br>
num_radial_bins                     = 5 <br>
min_T_kelvin                        = 1000 <br>
max_T_kelvin                        = 1e9 <br>
num_temperature_bins                = 5 <br>
dust_history_filename               = dust_history.dat                                      -- default = dust_history.dat <br>


# Parameter Descriptions
<dust> <br>
active                              : Whether the dust model should be activated <br>
subcycle                            : Include dust in the cooling subcycling term <br>
time_integrator                     : Which time-integration scheme to use for the dust distribution integration <br>
max_dM_in_bin                       : Reject subcycling timestep if any bin mass changed by more than a fraction max_dM_in_bin in a dt. -1. gives no constraint <br>
piecewise_method                    : Which bin-reconstruction method to use (to reconstruct N(a)) <br>
dust_cool_table_N_Tbins             : Set to -1 for on-the-fly calculation, or to a positive int N to use lookup table with N T bins <br>
cooling                             : Whether to add dust IR losses to the gas cooling <br>
disable_all_gas_cooling_for_testing : Testing option - ONLY consider IR losses and no gas-phase radiative losses <br>
num_grainsize_bins                  : The number of size-bins for the discretisation of the grain-size distribution <br>
grainsize_bins_low_edge             : Lower edge of grainsize distribution, in micro-meters <br>
grainsize_bins_high_edge            : Upper edge of grainsize distribution, in micro-meters <br>
carbonaceous_grains                 : Add and evolve carbonaceous grains <br>
silicate_grains                     : Add and evolve silicate grains <br>
metal_accretion                     : Whether to include grain growth from gas-phase metal accretion <br>
thermal_sputtering                  : Whether to include grain shrinkage from hot ion sputtering <br>
AGB_winds                           : Whether to create dusts via a simple AGB wind model <br>
carbonaceous_grain_mass_fraction    : Fraction of the total dust mass in initial conditions in form of carbonaceous dust <br>
silicate_grain_mass_fraction        : Fraction of the total dust mass in initial conditions in form of silicate dust <br>
carbonaceous_grain_density          : Grain density of carbonaceous dust in g/cm3 <br>
silicate_grain_density              : Grain density of silicate dust in g/cm3 <br>
init_profile                        : Initial radial profile for the DTG ratio <br>
init_dtg_mass_ratio                 : If init_profile_str == "const_dtg" then this parameter sets that dust-to-gas ratio value <br>
init_grainsize_distribution         : Shape of the initial grain-size/mass distribution <br>
flat_graindist_in_range_amin_microM : If init_grainsize_distribution_str == "flat_in_range" then this param sets the lower edge of that range, in micro-meters <br>
flat_graindist_in_range_amax_microM : If init_grainsize_distribution_str == "flat_in_range" then this param sets the upper edge of that range, in micro-meters <br>
slope_limiting                      : If linear reconstruction, decide to do slope limiting or not <br>
init_profile                        : Whether to set the initial profile as const DTG (using init_dtg_mass_ratio below), or let the AGB stellar profile run for a time (init_run_stellar_injection_time) to set init conds <br>
init_run_stellar_injection_time     : See above. <br>
init_dtg_mass_ratio                 : See above. <br>
sputtering_suppresion_factor        : The sputtering is multiplied by this factor - e.g. set to 0.5 to reduce sputtering by half. <br>

<dust/AGB_Winds> <br>
The AGB wind model is described by a radial stellar density profile, an IMF, and a stellar lifetime function.  <br>
The stellar profile is described by $\gamma'_\text{star} = \Delta \log \rho_\text{star} / \Delta \log r$ <br>
The grain-size distribution of the injected dust is computed using a common form: $\frac{\partial n}{\partial a} = \frac{C}{a^5} \exp \Bigg ( - \frac{\text{ln}^2 (a/a_\text{AGB})}{2 \sigma^2_\text{AGB}} \Bigg)$ <br>

gamma_star                          : $\gamma'_\text{star}$ <br>
sigma_AGB                           : $\sigma_\text{AGB}$ <br>
Mstar_cent_in_Msun                  : The stellar mass of the central galaxy, used to normalise the stellar density radial profile <br>
R_upper_in_kpc                      : The upper radius which to use to calculate the norm for the stellar density profile <br>
AGB_max_radius_in_kpc               : The cut-off for the region in which to inject AGB dust <br>

<dust/AGB_data_table> <br>
AGB_table_filename                  : Path to the yields table <br>
silicate_column_indexes_to_sum      : Sum these (possibly multiple) table columns in AGB_table_filename for total silicate dust yield <br>
carbonaceous_column_index           : Sum THIS __SINGLE__ table column in AGB_table_filename for total carbonaceous dust yield <br>

<problem/cluster/dust_history> <br>
User has the option to measure dust masses and cooling rates in radii and gas T bins and write to file <br>
If AGB winds are active we will also use these params to print injection history (in development) <br>
write_to_file                       : Whether to calculate the dust data and write to file <br>
min_radius                          : Minimum radius for the binning, in CODE LENGTH <br>
max_radius                          : Maximum radius for the binning, in CODE LENGTH <br>
num_radial_bins                     : Number of bins in radius <br>
min_T_kelvin                        : Minimum Temperature for the binning, in Kelvin <br>
max_T_kelvin                        : Maximum Temperature for the binning, in Kelvin <br>
num_temperature_bins                : Number of bins in Temperature <br>
dust_history_filename               : Filename for the dust history file <br>




# Important to-notes
- Dust only works with tabular cooling, since Townsend requires a specific form for the cooling function


# TODOs
- Adding in shattering and Coagulation
- Temperature-dependent sticking efficiencies
- More sophisticated SF/AGB injection prescription
- Supernovae
- Add metal model and gas-phase depletion due to accretion/enrichment due to sputtering
