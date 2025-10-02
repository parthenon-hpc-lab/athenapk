# AthenaPK units

## General unit system

Internally, all calculations are done in "code units" and there are no conversions between
code and physical units during runtime (with excetion like the temperature when cooling is
being used).
Therefore, in general no units need to be prescribed to run a simulation.

If units are required (e.g., if cooling is used and, thus, a conversion between internal energy
in code units and physical temperature is required) they are configured in the input block
as follows:

```
<units>
code_length_cgs = 3.085677580962325e+24 # 1 Mpc in cm
code_mass_cgs = 1.98841586e+47          # 1e14 Msun in g
code_time_cgs = 3.15576e+16             # 1 Gyr in s
```

This information will also be used by postprocessing tools (like yt) to convert between
code units and a physical unit system (like cgs).

Moreover, internally a set of factors from code to cgs units are available to process conversions
if required (e.g., from the input file).

For example, for an input parameter (in the input file) like

```
<problem/cloud>
r0_cgs = 3.085677580962325e+20          # 100 pc
```

the conversion should happen in the problem generator lik

```c++
  r_cloud = pin->GetReal("problem/cloud", "r0_cgs") / units.code_length_cgs();
```

so that the resulting quantity is internally in code units (here code length).

It is highly recommended to be *very* explicit/specific about units everywhere (as it is
a common source of confusion) like adding the `_cgs` suffix to the parameter in the
input file above.

## Magnetic units

Internally, AthenaPK (and almost all MHD codes) use
[Heaviside-Lorentz units](https://en.wikipedia.org/wiki/Heaviside%E2%80%93Lorentz_units),
where the magnetic field is transformed from $B \rightarrow B / \sqrt{4 \pi}$.
(See also the note in the
[Castro documentation](https://amrex-astro.github.io/Castro/docs/mhd.html) about this.)

So when converting from CGS-Gaussian units to code units, it is necessary to divide
by $\sqrt{4 \pi}$ (in addition to the base dimensional factors).
This is automatically handled by the `units.code_magnetic_cgs()` factors.


