# Input (parameter) files

## Hydro/MHD

`AthenaPK` currently supports compressible (magneto)hydrodynamic simulations.
Key parameters are controlled via the `hydro` block in the input file and include

`hydro/fluid`
- `euler` for hydrodynamics
- `glmmhd` for MHD using GLM based divergence cleaning

### Common options

#### Riemann solvers

Riemann solvers are configured via the `hydro/riemann` parameter.
Following options are available:

- `llf`
- `hlle`
- `hllc`
- `hlld`
- `none` : Disable calculation for (M)HD fluxes. Useful, e.g., for testing pure diffusion equations.
Requires `hydro/reconstruction=dc` (though reconstruction is not used in practice).

#### Reconstruction
Primitive variables are reconstructed using one of the following methods.

`hydro/reconstruction`
- `dc` : donor cell/piecewise constant (first order)
- `plm` : piecewise linear (second order)
- `ppm` : piecewise parabolic (third order)
- `wenoz` : WENO-Z (third order)

Note, `ppm` and `wenoz` need at least three ghost zones (`parthenon/mesh/num_ghost`).

#### Diffusive processes

##### Anisotropic thermal conduction (required MHD)
In the presence of magnetic fields thermal conduction is becoming anisotropic with the flux along
the local magnetic field direction typically being much stronger than the flux perpendicular to the magnetic field.

From a theoretical point of view, thermal condution is included in the system of MHD equations by an additional
term in the total energy equation:
```math
\delta_t E + \nabla \cdot (... + \mathbf{F}) \quad \mathrm{with}\\
\mathbf{F} = - \kappa \mathbf{\hat b} (\mathbf{\hat b \cdot \nabla T})
```

From an implementation point of view, two options implemented and can be configured within a `<diffusion>` block in the input file.
The diffusive fluxes are implemented in an unplit fashion, i.e., they are added to the hyperbolic fluxes in each stage of
the integration step (before flux correction in case of AMR, and calculating the flux divergence).
Moreover, they are implemented explicitly, i.e., they add a (potentially very restrictive) constraint to the timestep due to the scaling with $`\propto \Delta_x^2`$.
Finally, we employ limiters for calculating the temperature gradients following Sharma & Hammett (2007)[^SH07].
This prevents unphysical conduction against the gradient, which may be introduced because the off-axis gradients are not centered on the interfaces.

To enable conduction, set
`diffusion/conduction`
- `none` : No thermal conduction
- `spitzer` : Anisotropic thermal conduction with a temperature dependent classic Spitzer thermal conductivty
  $`\kappa (T) = c_\kappa T^{5/2} \mathrm{erg/s/K/cm}`$ and
  $`c_\kappa`$ being constant prefactor (set via `diffusion/spitzer_cond_in_erg_by_s_K_cm` with a default value of $`4.6\times10^{-7}`$). Note, as indicated by the units in the input parameter name, this kind of thermal conductivity requires a full set of units
  to be defined for the simulation.
- `thermal_diff` : Contrary to a temperature dependent conductivity, a simple thermal diffusivity can be used instead for which
the conduction flux is $`\mathbf{F} = - \chi \rho \mathbf{\hat b} (\mathbf{\hat b \cdot \nabla \frac{p_\mathrm{th}}{\rho}})`$
Here, the strenth, $`\chi`$, is controlled via the `diffusion/thermal_diff_coeff_code` parameter in code units.
Given the dimensions of $`L^2/T`$ it is referred to a thermal diffusivity rather than thermal conductivity.

[^SH07]:
    P. Sharma and G. W. Hammett, “Preserving monotonicity in anisotropic diffusion,” Journal of Computational Physics, vol. 227, no. 1, Art. no. 1, 2007, doi: https://doi.org/10.1016/j.jcp.2007.07.026.



### Additional MHD options

`hydro/glmmhd_source`:
- `dedner_plain` (default): Basic Dedner+2002[^D02] source function, i.e., right hand side is only defined for $`\delta_t \psi`$ with $`- (c_h^2/ c_p^2) \psi`$.
- `dedner_extended` :  Extended Dedner+2002[^D02] source function, i.e., additional terms on the right hand side that also modify the momentum and total energy density fields.
This formulation is not conservative any more, however, potentially results in
_"a more robust scheme in presence of strong discontinuity propagating through highly magnetized environment more robus solution"_[^MT10].

`hydro/glmmhd_alpha`: Real value between 0 and 1. (default: 0.1)\
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
where $`\mathrm{min}_\Omega`$ corresponds to the minium cell extent in the entire simulation domain.
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
