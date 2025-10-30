# Stochastic B field

The "stochastic_B_field" problem generator initializes a stochastic magnetic field with a specific magnetic power spectrum
$$E_M(k) := 4\pi \left<|\hat{B}(\mathbf{k})|^2\right> k^2,$$
where $\hat{B}$ is B field in Fourier-space and $\left<\right>$ indicates the average over all modes with the same magnitude $k$.

The currently implemented power-spectrum is given by:
$$E_k = k^{n1} \cdot \left( 1 + \left(\frac{k}{k_I}\right)^{\alpha} \right)^{-(n1+n2)/\alpha},$$
such that $E_k\propto k^{n1}$ for $k\ll k_I$ and $E_k\propto k^{-n2}$ for $k\gg k_I$. $k_I$ is the peak of the spectrum and $\alpha$ controls the sharpness of the transition between the two power-laws.

Additionally, the fractional helicity of the field, defined by
$$h = \frac{k\: H(k)}{2\: E_M(k)}$$
can be controlled, where
$$H(k) = 4\pi k^2 \left< \hat{\mathbf{A}}(\mathbf{k}) \cdot \hat{\mathbf{B}}^*(\mathbf{k})\right>$$
is the isotropic helicity spectrum. $h\in [-1, 1]$, i.e. set h = 0 for a nonhelical field and h = 1 (-1) for a field with full right(left)-handed helicity. 

The fluid is initially at rest and pressure and density are uniform. 

## Problem setup

An example parameter file can be found in `inputs/stochastic_B_field.in`.

A typical setup contains the following blocks in the input file:

```
<job>
problem_id = turbulence

<problem/stochastic_B_field>
rho0 = 1.0
p0   = 1.0
vx   = 0.0
vy   = 0.0
vz   = 0.0
kmax = 512.0	# Largest allowed mode in units of 2*pi/L_x where L_x is the box size as defined under <mesh>
		# This should be <= min(nx1, nx2, nx3)
B_rms = 0.3

# Double Power-law power  Spectrum params:
kI = 80.0	# Power Spectrum peak in units of 2*pi/L 
n1 = 4.0	# power law index such that P ~ k^(n1) for k << kI
n2 = 1.6666666666666667	# Power law index such that P ~ k^(-n2) for k >> KI
alpha = 10.0	# Controls the sharpness of the transition between the two power laws

helicity = 0.0 # Helicity fraction (between -1 and 1)
```