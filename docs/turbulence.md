# Driven turbulence simulations

The turbulence problem generator uses explicit inverse Fourier transformations (iFTs)
on each meshblock in order to reduce communication during the iFT.
Thus, it is only efficient if comparatively few modes are used (say < 100).

Quite generally, driven turbulence simulations start from uniform initial conditions
(uniform density and pressure, some initial magnetic field configuration in case of an
MHD setup, and the fluid at rest) and reach a state of stationary, isotropic (or anisotropic
depending on the strength of the background magnetic field) turbulence after one to few
large eddy turnover times (again depending on the background magnetic field strength).
The large eddy turnover time is usually defined as `T = L/U` with `L` being the scale
of the largest eddies and `U` the root mean square Mach number in the stationary regime.

The current implementation uses the following forcing spectrum
`(k/k_peak)^2 * (2 - (k/k_peak)^2)`.
Here, `k_peak` is the peak wavenumber of the forcing spectrum. It is related the scales of the largest eddies as
`L = 1/k_f` given that a box size of 1 is currently assumed/hardcoded.

## Problem setup

An example parameter file can be found in `inputs/turbulence.in`.

A typical setup contains the following blocks in the input file:

```
<job>
problem_id = turbulence

<problem/turbulence>
rho0         = 1.0      # initial mean density
p0           = 1.0      # initial mean pressure
b0           = 0.01     # initial magnetic field strength
b_config     = 0        # 0 - net flux; 1 - no net flux uniform B; 2 - non net flux sin B; 4 - field loop
kpeak        = 2.0      # characteristic wavenumber
corr_time    = 1.0      # autocorrelation time of the OU forcing process
rseed        = 20190729 # random seed of the OU forcing process
sol_weight   = 1.0      # solenoidal weight of the acceleration field
accel_rms    = 0.5      # root mean square value of the acceleration field
num_modes    = 30       # number of wavemodes

<modes>
k_1_0	= +2
k_1_1	= -1
k_1_2	= +0
k_2_0	= +1
...
```

The following parameters can be changed to control both the initial state:

- `rho0` initial mean density
- `p0` initial mean thermal pressure
- `b0` initial mean magnetic field strength
- `b_config`
  - `0`: net flux case (uniform B_x)
  - `1`: no net flux case (uniform B_x with changing sign in each half of the box)
  - `2`: no net flux with initial sinosoidal B_x field
  - `3`: deprecated
  - `4`: closed field loop/cylinder in the box (in x-y plane) located at
    - `x0=0.5` (default)
    - `y0=0.5` (default)
    - `z0=0.5` (default)
    - and radius `loop_rad=0.25`

as well as the driving field:

- `kpeak` peak wavenumber of the forcing spectrum. Make sure to update the wavemodes to match `kpeak`, see below.
- `corr_time` autocorrelation time of the acceleration field (in code units).
Using delta-in-time forcing, i.e., a very low value, is discouraged, see [Grete et al. 2018 ApJL](https://iopscience.iop.org/article/10.3847/2041-8213/aac0f5).
- `rseed` random seed for the OU process. Only change for new simulation, but keep unchanged for restarting simulations.
- `sol_weight` solenoidal weight of the acceleration field. `1.0` is purely solenoidal/rotational and `0.0` is purely dilatational/compressive. Any value between `0.0` and `1.0` is possible. The parameter is related to the resulting rotational power in the 3D acceleration field as
`1. - ((1-sol_weight)^2/(1-2*sol_weight+3*sol_weight^2))`, see eq (9) in [Federrath et al. 2010 A&A](
https://doi.org/10.1051/0004-6361/200912437).
- `accel_rms` root mean square value of the acceleration (controls the "strength" of the forcing field)
- `num_modes` number of wavemodes that are specified in the `<modes>` section of the parameter file.
The modes are specified manually as an explicit inverse FT is performed and only modes set are included (all others are assumed to be 0).
This is done to make the global inverse FT possible without any
expensive communication between blocks but this becomes excessively
expensiv for large number of modes.
Typically using a few tens of modes is a good choice in practice.
In order to generate a set of modes run the `inputs/generate_fmturb_modes.py` script and replace
the corresponding parts of the parameter file with the output of the script.
Within the script, the top three variables (`k_peak`, `k_high`, and `k_low`) need to be adjusted in
order to generate a complete set (i.e., all) of wavemodes.
Important, the `k_peak` in the script should match the `k_peak` set
in the input file.
Alternatively, wavemodes can be chosen/defined manually, e.g., if not all wavemodes are desired or
only individual modes should be forced.

### Blob injection

In order to study the evolution of (cold) clouds in a turbulent environment,
blobs can be injected to the simulation.
The injection is a one off mechanism controlled by the following parameters

```
<problem/turbulence>
# only one of the following three conditions can be set at a given time
inject_once_at_time = -1.0
inject_once_at_cycle = -1
inject_once_on_restart = false # should not be set in the input file, but only via the command line upon restart

inject_n_blobs = -1 # number of blob to inject

# then for the given number of blobs follow parameters need to be given (starting to count with 0)
inject_blob_radius_0 = ... # float, in code length units, no default value
inject_blob_loc_0 = ...,...,... # location vector of three comma-separated floats, in code length units, no default value
inject_blob_chi_0 = ... # float, dimensionless, no default value, density ratio to existing value

inject_blob_radius_1 = ...
...
```

In practice, this will result in blobs being seeded at a given time, cycle, or upon restart
by adjusting the density within the blob's radius by a factor of $\chi$ and
at the same time adjusting the temperature by a factor of $1/\chi$ so that the
blob remain in pressure equilibrium.

While this is an action that is performed once, it can be repeated upon restart (or a later
time) by resetting the variables.

A current restriction is that the blobs cannot be seeded across a domain boundary (i.e.,
the periodicity of the box is not taken into account).

### Rescaling **not recommended*

*The rescaling described in the following is generally not recommended, as it result in a
state that is not naturally reached.
Moreover, given the artificial nature of a hard reset, some time after the rescaling is
required for the system to readjust.

For non-isothermal simulations, the plasma will eventually heat up over time due to dissipation.
One possibility to remove that extra heat (or add heat), is to rescale the temperature in the simulation.
This can be done via the following parameters:

```
<problem/turbulence>
# only one of the following three conditions can be set at a given time
rescale_once_at_time = -1.0
rescale_once_at_cycle = -1
rescale_once_on_restart = false # should not be set in the input file, but only via the command line upon restart

rescale_to_rms_Ms = -1.0
```

As the parameters suggest, rescaling is a one off action (though it can be repeated when
the parameters are set again for a subsequent restart).
The density and velocity field are not changed, only the specific internal energy is
adjusted so that the volume-weighted root mean squared sonic Mach number matches
the target value.

## Typical results

The results shown here are obtained from running simulations with the parameters given in the next section.

### High level temporal evolution
![image](img/turb_evol.png)

### Power spectra
![image](img/turb_spec.png)

### Consistency of acceleration field
As each meshblock does a full iFT of all modes the following slices from a run with 8 meshblocks 
illustrate that there is no discontinuities at the meshblock boundary.

Plot shows x-, y-, and z-acceleration (in rows top to bottom) slices in the x-, y-, and z-direction (in columns from left to right).

![image](img/turb_acc.png)
