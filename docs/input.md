# Input (parameter) files

## Hydro/MHD

`AthenaPK` currently supports compressible (magneto)hydrodynamic simulations.
Key parameters are controlled via the `<hydro>` block in the input file and include

Parameter: `fluid` (string)
- `euler` for hydrodynamics
- `glmmhd` for MHD using GLM based divergence cleaning

### Common options

#### Riemann solvers

Fluxes are calculated using one of the following (1D) Riemann solvers.

Parameter: `riemann` (string)
- `llf` : Local Lax Friedrichs or Rusanov's method[^LLF].
Most diffusive Riemann solver. Not recommended for applications.
In `AthenaPK` only supported in combination with `dc` reconstruction and mainly
used for first order flux correction.
- `hlle` : Harten-Lax-van-Leer[^HLL83] with using signal speeds as proposed by Einfeldt[^E91]. Very diffusive for contact discontinuities.
- `hllc` : (HD only) Similar to HLLE but captures the _C_ontact discontinuity and is less diffusive, see [^LLF]
- `hlld` : (MHD only) Similar to HLLE but captures more _D_iscontinuities and is less diffusive, see [^MK05]
- `none` : Disable calculation for (M)HD fluxes. Useful, e.g., for testing pure diffusion equations.
Requires `hydro/reconstruction=dc` (though reconstruction is not used in practice).

[^LLF]:
    E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed., Springer-Verlag, Berlin, (1999) chpt. 10.

[^HLL83]:
    A. Harten, P. D. Lax and B. van Leer, "On upstream differencing and Godunov-type schemes for hyperbolic conservation laws", SIAM Review 25, 35-61 (1983)

[^E91]:
    Einfeldt et al., "On Godunov-type methods near low densities", JCP, 92, 273 (1991)

[^MK05]:
    Miyoshi, T. and Kusano, K., "A multi-state HLL approximate Riemann solver for ideal magnetohydrodynamics", Journal of Computational Physics, vol. 208, no. 1, pp. 315–344, 2005. doi: https://dx.doi.org/10.1016/j.jcp.2005.02.017

#### Reconstruction
Primitive variables are reconstructed using one of the following methods.

Parameter: `reconstruction` (string)
- `dc` : donor cell/piecewise constant (first order)
- `plm` : piecewise linear (second order)
- `ppm` : piecewise parabolic (third order)
- `limo3` : LimO3 (third order)
- `weno3` : WENO3 (third order)
- `wenoz` : WENO-Z (third order but more accurate than WENO3)

Note, `ppm` and `wenoz` need at least three ghost zones (`parthenon/mesh/num_ghost`).

#### Floors

Three floors can be enforced.
In practice, this happens in the conserved to primitive variable converison.

By default, all floors are disabled (set to a negative value) and the code will crash
with an error message when negative values are encountered.

To control the floors, following parameters can be set in the `<hydro>` block:
- `dfloor` (float): density floor in code units
- `pfloor` (float): pressure floor in code units
- `Tfloor` (float): temperature floor in K (requires a set of units to be defined)

*Note* the pressure floor will take precedence over the temperature floor in the
conserved to primitive conversion if both are defined.

#### Units

See(here)[units.md].

#### Diffusive processes

Diffusive processes in AthenaPK can be configured in the `<diffusion>` block of the input file.
```
<diffusion>
integrator = unsplit       # alternatively: rkl2 (for rkl2 integrator (operator split integrator)
#rkl2_max_dt_ratio = 100.0 # limits the ratio between the parabolic and hyperbolic timesteps (only used for RKL2 operator split integrator)
#cfl = 1.0                 # Additional safety factor applied in the caluclation of the diffusive timestep (used in both unsplit and RKL2 integration schemes). Defaults to hyperbolic cfl.

conduction = anisotropic               # none (disabled), or isotropic, or anisotropic
conduction_coeff = fixed               # alternative: spitzer
thermal_diff_coeff_code = 0.01         # fixed coefficent in code units (code_length^2/code_time)
#spitzer_cond_in_erg_by_s_K_cm = 4.6e7 # spitzer coefficient in cgs units (requires definition of a unit system)
#conduction_sat_phi = 0.3              # fudge factor to account for uncertainties in saturated fluxes


viscosity = none            # none (disabled) or isotropic
viscosity_coeff = fixed
mom_diff_coeff_code = 0.25  # fixed coefficent of the kinetmatic viscosity in code units (code_length^2/code_time)

resistivity = none          # none (disabled) or ohmic
resistivity_coeff = fixed
ohm_diff_coeff_code = 0.25  # fixed coefficent of the magnetic (ohmic) diffusivity code units (code_length^2/code_time)
```
(An)isotropic thermal conduction (with fixed or Spitzer coefficient), and isotropic viscosity and
resistivity with fixed coefficient are currently implemented.
They can be integrated in an unsplit manner or operator split using a second-order accurate RKL2
supertimestepping algorithm.
More details are described in the following.

#### Integrators

Diffusive processes can be integrated in either an unsplit
fashion (`diffusion/integrator=unsplit`) or operator split using a second-order accurate
RKL2 super timestepping algorithm (`diffusion/integrator=rkl2`) following [^M+14].

In the unsplit case, the diffusive processes are included at the end of every stage in
the main integration loop and the global timestep is limited accordingly.
A separate CFL can be set for the diffusive processes via `diffusion/cfl=...`, which
defaults to the hyperbolic value if not set.

In the RKL2 case, the global timestep is not limited by the diffusive processes by default.
However, as reported by [^V+17] a large number of stages
($`s \approx \sqrt(\Delta t_{hyp}/\Delta t_{par}) \geq 20`$) in the supertimestepping
(in combination with anisotropic, limited diffusion) may lead to a loss in accuracy, which
is why the difference between hyperbolic and parabolic timesteps can be limited by
`diffusion/rkl2_max_dt_ratio=...` and a warning is shown if the ratio is above 400.

[^M+14]:
    C. D. Meyer, D. S. Balsara, and T. D. Aslam, “A stabilized Runge–Kutta–Legendre method for explicit super-time-stepping of parabolic and mixed equations,” Journal of Computational Physics, vol. 257, pp. 594–626, 2014, doi: https://doi.org/10.1016/j.jcp.2013.08.021.

[^V+17]:
    B. Vaidya, D. Prasad, A. Mignone, P. Sharma, and L. Rickler, “Scalable explicit implementation of anisotropic diffusion with Runge–Kutta–Legendre super-time stepping,” Monthly Notices of the Royal Astronomical Society, vol. 472, no. 3, pp. 3147–3160, 2017, doi: 10.1093/mnras/stx2176.


##### Isotropic (hydro and MHD) and anisotropic thermal conduction (only MHD)
In the presence of magnetic fields thermal conduction is becoming anisotropic with the flux along
the local magnetic field direction typically being much stronger than the flux perpendicular to the magnetic field.

From a theoretical point of view, thermal conduction is included in the system of MHD equations by an additional
term in the total energy equation:
```math
\delta_t E + \nabla \cdot (... + \mathbf{F}_\mathrm{c})
```
where the full thermal conduction flux $`\mathbf{F}_\mathrm{c}`$ contains both the classic thermal conduction
```math
\mathbf{F}_\mathrm{classic} = - \kappa \mathbf{\hat b} (\mathbf{\hat b \cdot \nabla T})
```
as well as the saturated flux (as introduced by [^CM77])
```math
\mathbf{F}_\mathrm{sat} = - 5 \phi \rho^{-1/2} p^{3/2} \mathrm{sgn}(\mathbf{\hat b \cdot \nabla T}) \mathbf{\hat b}
```

From an implementation point of view, two options implemented and can be configured within a `<diffusion>` block in the input file.
The diffusive fluxes are implemented in an unsplit fashion, i.e., they are added to the hyperbolic fluxes in each stage of
the integration step (before flux correction in case of AMR, and calculating the flux divergence).
Moreover, they are implemented explicitly, i.e., they add a (potentially very restrictive) constraint to the timestep due to the scaling with $`\propto \Delta_x^2`$.
Finally, we employ limiters for calculating the temperature gradients following Sharma & Hammett (2007)[^SH07].
This prevents unphysical conduction against the gradient, which may be introduced because the off-axis gradients are not centered on the interfaces.
Similarly, to account for the different nature of classic and saturated fluxes (parabolic and hyperbolic, respectively),
we follow [^M+12] and use a smooth transition
```math
\mathbf{F}_\mathrm{c} = \frac{q}{q + F_\mathrm{classic}} \mathbf{F}_\mathrm{classic} \quad \mathrm{with} \quad q = 5 \phi \rho^{-1/2} p^{3/2}
```
and upwinding of the hyperbolic, saturated fluxes.

To enable thermal conduction, set

Parameter: `conduction` (string)
- `none` : No thermal conduction
- `isotropic` : Isotropic thermal conduction
- `anisotropic` : Anisotropic thermal conduction

In addition the coefficient (or diffusivity) needs to be set

Parameter: `conduction_coeff` (string)
- `spitzer` : Anisotropic thermal conduction with a temperature dependent classic Spitzer thermal conductivity
  $`\kappa (T) = c_\kappa T^{5/2} \mathrm{erg/s/K/cm}`$ and
  $`c_\kappa`$ being constant prefactor (set via the additional `diffusion/spitzer_cond_in_erg_by_s_K_cm` parameter with a default value of $`4.6\times10^{-7}`$ which assumes a fully ionized hydrogen plasma [^CM77] with $`\ln \lambda = 40`$ approximating ICM conditions). Note, as indicated by the units in the input parameter name, this kind of thermal conductivity requires a full set of units
  to be defined for the simulation.
- `fixed` : Contrary to a temperature dependent conductivity, a simple thermal diffusivity can be used instead for which
the conduction flux is $`\mathbf{F} = - \chi \rho \mathbf{\hat b} (\mathbf{\hat b \cdot \nabla \frac{p_\mathrm{th}}{\rho}})`$
Here, the strength, $`\chi`$, is controlled via the additional `thermal_diff_coeff_code` parameter in code units.
Given the dimensions of $`L^2/T`$ it is referred to a thermal diffusivity rather than thermal conductivity.

Parameter: `conduction_sat_phi` (float)
- Default value 0.3\
Factor to account for the uncertainty in the estimated of saturated fluxes, see [^CM77].
Default value corresponds to the typical value used in literature and goes back to [^MMM80] and [^BM82].


[^SH07]:
    P. Sharma and G. W. Hammett, "Preserving monotonicity in anisotropic diffusion," Journal of Computational Physics, vol. 227, no. 1, Art. no. 1, 2007, doi: https://doi.org/10.1016/j.jcp.2007.07.026.

[^M+12]:
    A. Mignone, C. Zanni, P. Tzeferacos, B. van Straalen, P. Colella, and G. Bodo, “THE PLUTO CODE FOR ADAPTIVE MESH COMPUTATIONS IN ASTROPHYSICAL FLUID DYNAMICS,” The Astrophysical Journal Supplement Series, vol. 198, Art. no. 1, Dec. 2011, doi: https://doi.org/10.1088/0067-0049/198/1/7

[^CM77]:
    L. Cowie and C. F. McKee, “The evaporation of spherical clouds in a hot gas. I. Classical and saturated mass loss rates.,” , vol. 211, pp. 135–146, Jan. 1977, doi: https://doi.org/10.1086/154911

[^MMM80]:
    C. E. Max, C. F. McKee, and W. C. Mead, “A model for laser driven ablative implosions,” The Physics of Fluids, vol. 23, Art. no. 8, 1980, doi: https://doi.org/10.1063/1.863183

[^BM82]:
    S. A. Balbus and C. F. McKee, “The evaporation of spherical clouds in a hot gas. III - Suprathermal evaporation,” , vol. 252, pp. 529–552, Jan. 1982, doi: https://doi.org/10.1086/159581

#### Viscosity/Momentum diffusion

Only isotropic viscosity with a (spatially and temporally) fixed coefficient in code units
(`code_length`^2/`code_time`) is currently implemented.
To enable set (in the `<diffusion>` block)
```
viscosity = isotropic
viscosity_coeff = fixed
mom_diff_coeff_code = 0.25  # fixed coefficent of the kinetmatic viscosity in code units (code_length^2/code_time)
```

#### Resistivity/Ohmic diffusion

Only resistivity with a (spatially and temporally) fixed coefficient in code units
(`code_length`^2/`code_time`)is currently implemented.
To enable set (in the `<diffusion>` block)
```
resistivity = ohmic
resistivity_coeff = fixed
ohm_diff_coeff_code = 0.25  # fixed coefficent of the magnetic (ohmic) diffusivity code units (code_length^2/code_time)
```


### Additional MHD options in `<hydro>` block

Parameter: `glmmhd_source` (string)
- `dedner_plain` (default): Basic Dedner+2002[^D02] source function, i.e., right hand side is only defined for $`\delta_t \psi`$ with $`- (c_h^2/ c_p^2) \psi`$.
- `dedner_extended` :  Extended Dedner+2002[^D02] source function, i.e., additional terms on the right hand side that also modify the momentum and total energy density fields.
This formulation is not conservative any more, however, potentially results in
_"a more robust scheme in presence of strong discontinuity propagating through highly magnetized environment more robus solution"_[^MT10].

Parameter: `glmmhd_alpha` (float)
- Real value between 0 and 1. (default: 0.1)\
Corresponds to the ratio of diffusive, $`T_d`$ to advective $`T_a`$ timescales of the divergence cleaning.
Dedner+2002[^D02] suggested to use a grid independent value of $`c_p^2/c_h=0.18`$ to determine $`c_p`$ in the source term.
However, Mignone & Tzeferacos2010[^MT10] pointed out that this is not a dimensionless quantify.
For this reason, they defined $`\alpha = \Delta_x c_h / c_p^2`$ and update $`\psi`$ numerically with
```math
\psi^{n+1} = \psi^n \mathrm{exp}(- \alpha \Delta_t c_h / \Delta_x),
```
see their equation (27).
This result in a self consistent ratio of timescale: $`\alpha = T_d / T_a`$ with
$`T_d = \Delta_x^2 / c_p^2`$ and $`T_a = \Delta_x / c_h`$.
We implemented this apporoach.
In practice (for the choice of setting the divergence cleaning speed described below), the argument of the exponential reduces to $`-c_{cfl} \alpha`$ if $`\Delta_t`$ is the global timestep constrained by the hyperbolic timestep.
If, for example, the global timestep is further restricted by an explicit diffusive process, then this "shortcut" does not apply.
**Note**, the optimal choice for $`\alpha`$ is a problem dependent.

##### Side note on setting the divergence cleaning speed

The cleaning speed $`c_h`$ at which the divergence errors are transported can be defined locally and globally.
Typically the latter approach is chosen (e.g., [^D02],[^MT10], [^D18]) because of its simplicity: no additional handling of a spatially varying quantity is required.
We follow the same approach in our implementation.


We set the cleaning speed following the original paper by Dedner+2002[^D02] to
```math
c_h = c_{cfl} \mathrm{min}_\Omega (\Delta_{x_1}, \Delta_{x_2}, \Delta_{x_2}) / \Delta_t
```
where $`\mathrm{min}_\Omega`$ corresponds to the minimum cell extent in the entire simulation domain.
This definitions results in the maximum $`c_h`$ that is compatible with the hyperbolic timestep $`\Delta_t`$.
In other words, no new timestep restriction is introduced.

For uniform mesh simulations this $`c_h`$ directly corresponds to the global maxium signal speed
$`\lambda_{\mathrm{max},\Omega} = \mathrm{max}_\Omega (|u_{x_1}| + c_{f,1}, |u_{x_2}| + c_{f,2},|u_{x_3}| + c_{f,3})`$.
For static/adaptive mesh refinement simulations the dependency on the global, minimum cell extent is important
because $`\lambda_\mathrm{max}`$ could be larger on a coarse block than on a fine block so that using the global
$`c_h = \lambda_\mathrm{max}`$ could violate the CFL condition on finer level.

[^D02]:
    A. Dedner, F. Kemm, D. Kröner, C.-D. Munz, T. Schnitzer, and M. Wesenberg, “Hyperbolic Divergence Cleaning for the MHD Equations,” Journal of Computational Physics, vol. 175, no. 2, Art. no. 2, 2002, doi: https://doi.org/10.1006/jcph.2001.6961.

[^MT10]:
    A. Mignone and P. Tzeferacos, “A second-order unsplit Godunov scheme for cell-centered MHD: The CTU-GLM scheme,” Journal of Computational Physics, vol. 229, no. 6, Art. no. 6, 2010, doi: https://doi.org/10.1016/j.jcp.2009.11.026.

[^D18]:
    D. Derigs, A. R. Winters, G. J. Gassner, S. Walch, and M. Bohm, “Ideal GLM-MHD: About the entropy consistent nine-wave magnetic field divergence diminishing ideal magnetohydrodynamics equations,” Journal of Computational Physics, vol. 364, pp. 420–467, 2018, doi: https://doi.org/10.1016/j.jcp.2018.03.002.

### Debugging options

Following options are typically not used for productions runs but can
be useful for debugging and/or testing.

In the `<hydro>` block:

Parameter: `calc_dt_hyp` (bool)
- Enables calculation of the hyperbolic timestep constraint.
Is internally enabled by default if hyperbolic fluxes are calculated (and disabled if
not), but can be overridden by this "external"/input option, which takes precedence
over the internal default.

Parameter: `max_dt` (float)
- Maximum global timestep. Disabled (i.e., set to a negative value) by default.
If set to a positive value, it will limit the `dt` in the simulation if `max_dt` is lower
than any other timestep constraint (e.g., the hyperbolic one).

### Cooling

Tabular cooling (e.g., for optically thin cooling) is enabled through the `cooling` block in the input file.
The tabulated table itself is a text file containing (apart from comments) only two columns (two floating
point numbers per line with the first one being the log10 temperature and the second one the
log10 cooling rate scaled to a source function with $S_{cool} = n_H^2 \Lambda(T)$).

A possible block might look like:

```
<cooling>
enable_cooling = tabular                # To disable, set to `none`
table_filename = schure.cooling_1.0Z    # Path to the cooling table (in a text file)
lambda_units_cgs = 1                    # Conversion factor of the cooling rate relative to CGS units

integrator = townsend              # Other possible options are `rk12` and `rk45` for error bound subcycling
#max_iter = 100                    # Max number of iteration for subcycling. Unsued for Townsend integrator
cfl = 0.1                          # Restrict global timestep to `cfl*e / dedt`, i.e., some fraction of change per cycle in the specific internal energy (i.e., temperature)
d_log_temp_tol = 1e-8              # Tolerance in cooling table between subsequent entries. Both subcycling integrators and cfl restriction rely on a table lookup that assumes equally spaced (in log space) temperature values.
#d_e_tol = 1e-8                    # Tolerance for the relative error in the change of internal energy for the error bound subcyling integrators (rk12 and rk45). Unused for Townsend integrator.
```

*Note* several special cases for handling the lower end of the cooling table/low temperatures:
- Cooling is turned off once the temperature reaches the lower end of the cooling table. Within the cooling function, gas does not cool past the cooling table.
- If the global temperature floor `<hydro/Tfloor>` is higher than the lower end of the cooling table, then the global temperature floor takes precedence.
- The pressure floor if present is not considered in the cooling function, only the temperature floor `<hydro/Tfloor>`

Finally, a more comprehensive descriptions of various conventions used for cooling
functions (and the chosen implementation in AthenaPK) can be found in [Cooling Notes](cooling_notes.md)
and a notebook comparing various cooling tables (and their conversion) in [cooling/cooling.ipynb](cooling/cooling.ipynb).
