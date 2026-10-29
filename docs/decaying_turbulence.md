# Decaying turbulence simulations / stochastic field generator

Problem generator for decaying turbulence simulations, the setup used in e.g. [Kasselmann et al. 2026](https://arxiv.org/pdf/2606.02617). 

The "decaying_turbulence" problem generator initializes a stochastic (gaussian random) magnetic field on a periodic domain with a specific magnetic energy power spectrum
$$E_M(k) := \frac{1}{2} \left<|\mathbf{B}(\mathbf{k})|^2\right>,$$
where $\mathbf{B}(\mathbf{k})$ is the magnetic field in Fourier-space and $\left<\right>$ indicates the average over all modes with the same magnitude $k$.

The currently implemented power-spectrum is given by:
$$E_k = k^{n1} \cdot \left( 1 + \left(\frac{k}{k_I}\right)^{\alpha} \right)^{-(n1+n2)/\alpha},$$
such that $E_k\propto k^{n1}$ for $k\ll k_I$ and $E_k\propto k^{-n2}$ for $k\gg k_I$. $k_I$ is the peak of the spectrum and $\alpha$ controls the sharpness of the transition between the two power-laws.

The magnetic helicity power spectrum is given by, 

$$\mathcal{H}(k) = \left<\mathbf{A}(\mathbf{k})\cdot \mathbf{B}^*(\mathbf{k})\right>,$$

where $\mathbf{A}$ is the vector potential of the magnetic field. (Note that while $\mathbf{A}$ is a gauge-dependent quantity, the gauge-dependent term in $\mathcal{H}(k)$ vanishes on a periodic domain.)

Realizability then requires that the fractional helicity spectrum $$h(k) = \frac{k\mathcal{H}(k)}{2 E_M(k)}$$

satisfies $-1 \leq h(k) \leq 1$ everywhere. In this implementation, the initial $h(k)$ is assumed to be uniform over $k$ and controlled by a single value (e.g. $h = 0$ for a field with vanishing average helicity, $h = 1$ for a field with maximally positive helicity). 

The fluid is initially at rest and pressure and density are uniform. 

## Problem setup

An example parameter file can be found in `inputs/decaying_turbulence.in`.

A typical setup contains the following blocks in the input file:

```
<job>
problem_id = decaying_turbulence

<problem/decaying_turbulence>
rho0 = 1.0 # initial (uniform) density
p0   = 1.0 # initial (uniform) fluid pressure

B_rms = 0.3 # initial root mean square of the magnetic field strength

# Double Power-law power  Spectrum params:
kI = 30.0	# Power Spectrum peak in units of 2*pi/L 
n1 = 4.0	# power law index such that P ~ k^(n1) for k << kI
n2 = 2.0	# Power law index such that P ~ k^(-n2) for k >> KI
alpha = 10.0	# Controls the sharpness of the transition between the two power laws

helicity = 0.0 # Helicity fraction of the magnetic field (between -1 and 1)
```

## Requirements

1. The simulation domain must be a uniform grid, adaptive or static mesh refinement are not supported. 
2. The simulation domain must be cubic: Same physical length and number of cells in all three dimensions. 
3. Since this problem generator makes use of [Parthenon's Fourier transform functionality](https://parthenon-hpc-lab.github.io/parthenon/develop/src/fourier_transforms.html), AthenaPK must be compiled with `-DPARTHENON_ENABLE_FFT=ON`

## Using a different power spectrum 

Modifying the problem generator to set up a field with a different power-spectrum than the one above is straightforward. In `src/pgen/decaying_turbulence.cpp`: 

1. Modify `double PowerSpectrum()` or add your own function alongside it. Its first argument must be the wavenumber magnitude in physical code units. Additional arguments can come after.

2. If your power spectrum has arguments that you want to tweak at runtime, add them to your input file (e.g.`inputs/decaying_turbulence.in`) and load them in inside C. 

3. In line 83: `// Define the power spectrum function
  auto P = [=](double k) -> Real {
    return PowerSpectrum(k, kI_phys, n1, n2, alpha);
  };`
  Modify the lambda to return your Power spectrum instead and make sure to pass its parameters in the correct order. 

## Implementation details

The main function for setting up a stochastic field is defined in `src/utils/generate_stochastic_b_field.cpp`. 

A solenoidal field in Fourier space, $\mathbf{B}(\mathbf{k})$, can locally be expressed in the basis
$$\mathbf{B}(\mathbf{k}) = B^+(\mathbf{k})\: \mathbf{e}^+(\mathbf{k}) + B^-(\mathbf{k})\: \mathbf{e}^-(\mathbf{k}),$$
where $\mathbf{e}^\pm(\mathbf{k})$ are local Eigenvectors of the curl operator,
$$i\mathbf{k}\times\mathbf{e}^\pm (\mathbf{k}) = \pm |\mathbf{k}|\mathbf{e}^\pm(\mathbf{k}).$$
$\mathbf{e}^+(\mathbf{k})$ and $\mathbf{e}^{-}(\mathbf{k})$ therefore carry positive and negative helicity respectively.  
$B^+(\mathbf{k})$ and $B^-(\mathbf{k})$ are complex numbers whose real and imaginary parts are sampled independently from a gaussian distribution with zero mean. The variances for each component, $\sigma^+(\mathbf{k}), \sigma^-(\mathbf{k})$ are determined as
$$\sigma^\pm(\mathbf{k}) = \sigma(\mathbf{k}) \sqrt{\frac{1\pm h}{2}},$$
such that 
$$\sigma^+(\mathbf{k})^2 + \sigma^-(\mathbf{k})^2 = \sigma(\mathbf{k})^2 = \langle |\mathbf{B}(\mathbf{k})|^2\rangle.$$
Here, $\langle |\mathbf{B}(\mathbf{k})|^2\rangle$ is set by the desired power spectrum and $h$ is the desired total helicity fraction. (E.g. for $h=1$, all power goes into $B^+(\mathbf{k})$ and for $h=0$, power is equally distributed between the circular modes.) 

The basis is constructed explicitely for each mode and the prefactors are sampled accordingly. 

Note that a field constructed in this way, the divergence-free condition holds precisely in Fourier space, since $\mathbf{e}^\pm$ are perpendicular to $\mathbf{k}$. 

After sampling the field in Fourier space, it is transformed into real space and normalized to the desired root mean square value. 
