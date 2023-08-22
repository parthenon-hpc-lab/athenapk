# Turbulence problem generator

The problem generator has been refactored from its implementation in K-Athena.
Until the documentation is fully adapted, please consult https://gitlab.com/pgrete/kathena/-/wikis/turbulence for general information.

A sample input file is provided in [turbulence.in](../inputs/turbulence.in)

## Blob injection

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

## Rescaling **not recommended*

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
