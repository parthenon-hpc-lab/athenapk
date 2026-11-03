<!-- This file was made in part with generative AI -->
# Standard problem generators

## CT-compatible MHD examples

CPAW, linear MHD waves, field-loop advection, Orszag--Tang, and current sheet
have standalone decks under `inputs/ct_glm_compatible/`.
They support `ctmhd`, `ucthlldmhd`, and `glmmhd` without duplicating pgen
implementations. The wave generators include 1D CT initialization; other
examples are multidimensional. See [the CT guide](constrained_transport.md)
for deck variants, initialization/refinement details, diagnostic definitions,
and reproducible commands. Legacy GLM field-loop, CPAW and OT input paths remain available.
See [validation](ct_validation.md) for builds, test commands and summaries.

## Hydro and MHD linear waves

`job/problem_id=linear_wave_hydro` selects the preserved Euler wave implementation.
Use `inputs/linear_wave1d_hydro.in` or `inputs/linear_wave3d_hydro.in`; hydro
parameters remain in `problem/linear_wave`, with hydro modes 0--4. Older decks
must update the problem selector. `job/problem_id=linear_wave_mhd` instead selects
the separate GLM/CT MHD implementation and `problem/linear_wave_mhd`, with modes
0--6. The GLM magnetic-wave and diffusion tests explicitly use the MHD generator.

## Multi cloud shattering (`job/problem_id=shattering`)

This problem generator implements the basic multi cloud shattering setup
following [Gronke & Oh 2020](https://doi.org/10.1093/mnrasl/slaa033G).

The initial conditions are currently hardcoded to a unit cube (though
physical dimensions have to be assigned via units) with initially
unit density (in code units) for the hot phase.

In addition to specifying units, the problem generator can be configured via

```
<problem/shattering>
chi_i = 100        # initial overdensity
T_cloud = 4e5      # initial temperature
#reset_times = true # set by default
```

which sets initial overdensity and temperature of the clouds, which
are initialized in pressure equilibrium with the hot phase.
By default, following the paper the simulation times
(i.e., `tlim` and the `dt` for outputs) are rescaled with respect to
the relevant dynamical timescale (e.g., cooling timescales or sound crossing timescale),
which is also reported on terminal upon simulation startup.

Moreover, following details are currently hardcoded (following the choices in
the paper), but can easily be modified in the code when needed:
- the individual cloud radii are fixed to 0.125 (i.e., 1/8 of the box)
- four clouds are seeded at positions  `{0.5, 0.5, 0.5}, {0.45, 0.5, 0.55}, {0.5, 0.47, 0.42}, {0.53, 0.56, 0.5}`
- initial perturpations are using a normal distribution with standard deviation of 0.01


# More complex problem generators

- [Galaxy Cluster and Cluster-like Problem Setup](cluster.md)
- [Driven turbulence](turbulence.md)
