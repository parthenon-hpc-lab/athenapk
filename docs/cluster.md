# Galaxy Cluster and Cluster-like Problem Setup

This problem generator initializes an isolated ideal galaxy cluster (or a
galaxy-cluster-like object). Simulations begin as a spherically symmetric set up of
gas in hydrostatic equilibrium with a fixed defined gravitational potential and
an ACCEPT-like entropy profile. An initial magnetic tower can also be included.

In addition to the fixed gravitational potential source term, the problem
generator also includes AGN feedback via any combination of the injection of a
magnetic tower, kinetic jet, and a flat volumetric thermal dump around the
fixed AGN. Feedback via the magnetic tower and jet can be set to precess around
the z-axis. The AGN feedback power can be set to a fixed power or triggered via
Boosted Bondi accretion, Bondi-Schaye accretion,  or cold gas near the AGN.

## Units

All parameters in `<problem/cluster>` are defined in code units. Code units can
be defined under `<units`. For example,
```
<units>
#Units parameters
code_length_cgs = 3.085677580962325e+24 # in cm
code_mass_cgs = 1.98841586e+47 # in g
code_time_cgs = 3.15576e+16 # in s
```
will set the code length-unit to 1 Mpc, the code mass to 10^14 solar masses,
and the code time to 1 Gyr.


## Fixed Gravitational Profile

A gravitational profile can be defined including components from an NFW dark
matter halo, a brightest cluster galaxy (BCG), and a point-source central
supermassive black hole. This gravitational potential is used to determine
initial conditions in hydrostatic equilbrium and by default as a source term
during evolution. Parameters for the gravitationl profile are placed into `<problem/cluster/gravity>`.


The toggles to include different components are as follows:
```
<problem/cluster/gravity>
include_nfw_g = True
which_bcg_g = HERNQUIST #or NONE
include_smbh_g = True
```
Where `include_nfw_g` for the NFW dark-matter halo ([Navarro
1997](doi.org/10.1086/304888)) is boolean; `which_bcg_g` for the BCG can be
`NONE` for no BCG, `HERNQUIST` for  Hernquist profile ([Hernquist
1990](doi.org/10.1086/168845)), and `include_smbh_g` for the SMBH is
boolean.

Parameters for the NFW profile are
```
<problem/cluster/gravity>
c_nfw = 6.0 # Unitless
m_nfw_200 = 10.0 # in code_mass
```
which adds a gravitational acceleration defined by

$$
g_{\text{NFW}}(r) = 
  \frac{G}{r^2}
  \frac{M_{NFW}  \left [ \ln{\left(1 + \frac{r}{R_{NFW}} \right )} - \frac{r}{r+R_{NFW}} \right ]}
        { \ln{\left(1 + c_{NFW}\right)} - \frac{ c_{NFW}}{1 + c_{NFW}} }
$$
The scale radius $R_{NFW}$ for the NFW profile is computed from
$$
R_{NFW} = \left ( \frac{M_{NFW}}{ 4 \pi \rho_{NFW} \left [ \ln{\left ( 1 + c_{NFW} \right )} - c_{NFW}/\left(1 + c_{NFW} \right ) \right ] }\right )^{1/3}
$$
where the scale density $\rho_{NFW}$ is computed from 
$$
\rho_{NFW} = \frac{200}{3} \rho_{crit} \frac{c_{NFW}^3}{\ln{\left ( 1 + c_{NFW} \right )} - c_{NFW}/\left(1 + c_{NFW} \right )}.
$$
The critical density $\rho_{crit}$ is computed from
$$
    \frac{3 H_0^2}{8 \pi G}.
$$

Parameters for the HERNQUIST BCG are controlled via:
```
<problem/cluster/gravity>

m_bcg_s = 0.001 # in code_mass
r_bcg_s = 0.004 # in code_length
```
where a HERNQUIST profile adds a gravitational acceleration defined by

$$
 g_{BCG}(r) = G \frac{ M_{BCG} }{R_{BCG}^2} \frac{1}{\left( 1 + \frac{r}{R_{BCG}}\right)^2}
$$

Gravitational acceleration from the SMBH is inserted as a point source defined solely by its mass
```
<problem/cluster/gravity>
m_smbh = 1.0e-06 # in code_mass
```

Some acceleration profiles may be ill-defined at the origin. For this, we provide a smoothing length parameter,
```
<problem/cluster/gravity>
g_smoothing_radius = 0.0 # in code_length
```
which works as a minimum r when the gravitational potential is applied. It
effectively modifies the gravitation acceleration to work as

$$
\tilde{g} (r) = g( max( r, r_{smooth}))
$$

By default, the gravitational profile used to create the initial conditions is
also used as an accelerating source term during evolution. This source term can
be turned off, letting this gravitational profile to only apply to
initialization, with the following parameter in `<problem/cluster/gravity>`.
```
<problem/cluster/gravity>
gravity_srcterm = False
```

## Entropy Profile

The `cluster` problem generator initializes a galaxy-cluster-like system with an entropy profile following the ACCEPT profile

$$
 K(r) = K_{0} + K_{100} \left ( r/ 100 \text{ kpc} \right )^{\alpha_K}
 $$
 
where we are using the entropy $K$ is defined as 

$$
 K \equiv \frac{ k_bT}{n_e^{2/3} }
 $$
 
This profile is determined by these parameters
```
<problem/cluster/entropy_profile>
k_0 = 8.851337676479303e-121 # in code_length**4*code_mass/code_time**2
k_100 = 1.3277006514718954e-119 # in code_length**4*code_mass/code_time**2
r_k = 0.1 # in code_length
alpha_k = 1.1 # unitless
```

## Defining Initial Hydrostatic Equilibrium

With a gravitational profile and initial entropy profile defined, the system of
equations for initial hydrostatic equilibrium are still not closed. In order to
close them, we fix the density of the cluster to a defined value at a defined
radius.  This radius and density is set by the parameters
```
<problem/cluster/hydrostatic_equilibrium>
r_fix = 2.0 # in code_length
rho_fix = 0.01477557589278723 # in code_mass/code_length**3
```
In each meshblock the equations for hydrostatic equilbirium and the entropy
profile are integrated to obtain density and pressure profiles inward and
outward from `r_fix` as needed to cover the meshblock. The parameter
`r_sampling` controls the resolution of the profiles created, where higher
values give higher resolution.
```
<problem/cluster/hydrostatic_equilibrium>
r_sampling = 4.0
```
Specifically, the resolution of the 1D profile for each meshblock is either
`min(dx,dy,dz)/r_sampling` or `r_k/r_sampling`, whichever is smaller.

## AGN Triggering

If AGN triggering is enabled, at the end of each time step, a mass accretion
rate `mdot` is determined from conditions around the AGN according to the
different triggering prescriptions. The accreted mass is removed from the gas
around the AGN, with details depending on each prescription explained below,
and is used as input for AGN feedback power.

The triggering prescriptions currently implemented are "boosted Bondi
accretion" ([Bondi 1952](doi.org/10.1093/mnras/112.2.195), [Meece
2017](doi.org/10.3847/1538-4357/aa6fb1)), "Bondi-Schaye accretion" ([Bondi and
Schaye 2009](doi.org/10.1111/j.1365-2966.2009.15043.x)),  and "cold gas"
([Meece 2017](doi.org/10.3847/1538-4357/aa6fb1)). These modes can be chosen via
```
<problem/cluster/agn_triggering>
triggering_mode = COLD_GAS # or NONE, BOOSTED_BONDI, BONDI_SCHAYE
```
where `triggering_mode=NONE` will disable AGN triggering. 

With BOOSTED_BONDI accretion, the mass rate of accretion follows

$$
\dot{M} = \alpha \frac { 2 \pi G^2 M^2_{SMBH} \hat {\rho} } {
\left ( \hat{v}^2 + \hat{c}_s^2 \right ) ^{3/2} }
$$

where $\hat{rho}$, $\hat{v}$, and $\hat{c}_s$ are respectively the mass weighted density,
velocity, and sound speed within the accretion region. The mass of the SMBH,
the radius of the sphere of accretion around the AGN, and the $\alpha$ parameter
can be set with
```
<problem/cluster/gravity>
m_smbh = 1.0e-06 # in code_mass

<problem/cluster/agn_triggering>
accretion_radius = 0.001 # in code_length
bondi_alpha= 100.0 # unitless
```
With BONDI_SCHAYE accretion, the `$\alpha$` used for BOOSTED_BONDI accretion is modified to depend on the number density following:

$$
\alpha =
 \begin{cases}
1 & n \leq n_0 \\\\
 ( n/n_0 ) ^\beta & n > n_0\\\\
\end{cases}
$$

where `n` is the mass weighted mean density within the accretion region and the parameter `n_0` and `beta` can be set with
```
<problem/cluster/agn_triggering>
bondi_n0= 2.9379989445851786e+72 # in 1/code_length**3
bondi_beta= 2.0 # unitless
```

With both BOOSTED_BONDI and BONDI_SCHAYE accretion, mass is removed from each
cell within the accretion zone at a mass weighted rate. E.g. the mass in each
cell within the accretion region changes by
```
new_cell_mass = cell_mass - cell_mass/total_mass*mdot*dt;
```
where `total_mass` is the total mass within the accretion zone.  The accreted
mass is removed from the gas which momentum density and energy density
unchanged. Thus velocities and temperatures will increase where mass is
removed.


With COLD_GAS accretion, the accretion rate becomes the total mass within the accretion zone equal to or
below a defined cold temperature threshold divided by a defined accretion
timescale. The temperature threshold and accretion timescale are defined by
```
<problem/cluster/agn_triggering>
cold_temp_thresh= 100000.0
cold_t_acc= 0.1
```
Mass is removed from each cell in the accretion zone on the accretion
timescale. E.g. for each cell in the accretion zone with cold gas
```
new_cell_mass = cell_mass - cell_mass/cold_t_acc*dt;
```
As with the Bondi-like accretion prescriptions, this mass is removed such that
the momentum and energy densities are unchanged.


## AGN Feedback

AGN feedback can be both triggered via the mechanisms in the section above and with a fixed power.
```
<problem/cluster/agn_feedback>
fixed_power = 0.0
efficiency = 0.001
```
Where and `mdot` calculated from AGN triggering will lead to an an AGN feedback
power of `agn_power = efficiency*mdot*c**2`. The parameter `efficiency` is
specifically the AGN's effiency converting in-falling mass into energy in the
jet. The fixed power and triggered power are not mutually exclusive; if both
`fixed_power` is defined and triggering is enabled with a non-zero
`efficiency`, then the `fixed_power` will be added to the triggered AGN power.


AGN feedback can be injected via any combination of an injected magnetic tower,
a thermal dump around the AGN, and a kinetic jet. The fraction deposited into
each mechansim can be controlled via
```
<problem/cluster/agn_feedback>
magnetic_fraction = 0.3333
thermal_fraction = 0.3333
kinetic_fraction = 0.3333
```
These values are automatically normalized to sum to 1.0 at run time.

Thermal feedback is deposited at a flat power density within a sphere of defined radius
```
<problem/cluster/agn_feedback>
thermal_radius = 0.0005
```
Mass is also injected into the sphere at a flat density rate with the existing
velocity and temperature to match the accreted mass proportioned to thermal
feedback, e.g.
```
thermal_injected_mass = mdot * (1 - efficiency) * normalized_thermal_fraction;
```

Kinetic feedback is deposited into two disks along the axis of the jet within a
defined radius, thickness of each disk, and an offset above and below the plane
of the AGN disk where each disk begins.
```
<problem/cluster/agn_feedback>
kinetic_jet_radius  = 0.0005
kinetic_jet_thickness  = 0.0005
kinetic_jet_offset  = 0.0005
```
Along the axis of the jet, kinetic energy will be deposited as far away as
`kinetic_jet_offset+kinetic_jet_thickness` in either direction. With a z-axis
aligned jet, `kinetic_jet_thickness` should be a multiple of the deposition
zone grid size, otherwise feedback will be lost due to systematic integration
error.

The axis of the jet can be set to precess with 
```
<problem/cluster/precessing_jet>
jet_theta= 0.15 # in radians
jet_phi0= 0.2  # in radians
jet_phi_dot= 628.3185307179587 # in radians/code_time
```
at defined precession angle off of the z-axis (`jet_theta`), an initial
azimuthal angle (`jet_phi0`), and an rate of azimuthal precession
(`jet_phi_dot`).

Kinetic jet feedback is injected  is injected as if disk of fixed temperature
and velocity and changing density to match the AGN triggering rate were added
to the existing ambient gas. Either or both the jet temperature $T_{jet}$ and
velocity $v_{jet}$ can be set via
```
<problem/cluster/agn_feedback>
#kinetic_jet_velocity  = 13.695710297774411 # code_length/code_time
kinetic_jet_temperature = 1e7 # K
```
However, $T_{jet}$ and $v_{jet}$ must be non-negative and fulfill
$$
v_{jet} = \sqrt{ 2 \left ( \epsilon c^2 - (1 - \epsilon) \frac{k_B T_{jet}}{ \mu m_h \left( \gamma - 1 \right} \right ) }
$$
to ensure that the sum of rest mass energy, thermal energy, and kinetic energy of the new gas sums to $\dot{M} c^2$. Note that these equations places limits  on $T_{jet}$ and $v_{jet}$, specifically
$$
v_{jet} \leq c \sqrt{ 2 \epsilon } \qquad \text{and} \qquad \frac{k_B T_{jet}}{ \mu m_h \left( \gamma - 1 \right} \leq c^2 \frac{ \epsilon}{1 - \epsilon}
$$
If the above equations are not satified then an exception will be thrown at
initialization. If neither $T_{jet}$ nor $v_{jet}$ are specified, then
$v_{jet}$ will be computed assuming $T_{jet}=0$ and a warning will be given
that the temperature of the jet is assumed to be 0 K.

The total mass injected with kinetic jet feedback at each time step is
```
kinetic_injected_mass = mdot * (1 - efficiency) * normalized_kinetic_fraction;
```
In each cell the added density, momentum, and energy are
```
kinetic_injected_density = kinetic_injected_mass/(2*kinetic_jet_thickness*pi*kinetic_jet_radius**2)
kinetic_injected_momentum_density = kinetc_injected_density*kinetic_jet_velocity**2
kinetic_injected_energy_density = mdot*efficiency*normalized_kinetic_fraction/(2*kinetic_jet_thickness*pi*kinetic_jet_radius**2
```
Note that this only guarentees a fixed change in momentum density and total
energy density; changes in kinetic energy density will depend on the velocity
of the ambient gas. Temperature will also change but should always increase
with kinetic jet feedback.

Magnetic feedback is injected following  ([Hui 2006](doi.org/10.1086/501499))
where the injected magnetic field follows 

$$
\begin{align}
\mathcal{B}_r      &=\mathcal{B}_0 2 \frac{h r}{\ell^2} \exp{ \left ( \frac{-r^2 - h^2}{\ell^2} \right )} \\\\
\mathcal{B}_\theta &=\mathcal{B}_0 \alpha \frac{r}{\ell} \exp{ \left ( \frac{-r^2 - h^2}{\ell^2} \right ) } \\\\
\mathcal{B}_h      &=\mathcal{B}_0 2 \left( 1 - \frac{r^2}{\ell^2} \right ) \exp{ \left ( \frac{-r^2 - h^2}{\ell^2} \right )} \\\\
\end{align}
$$

which has  the corresponding vector potential field

$$
\begin{align}
\mathcal{A}_r &= 0 \\\\
\mathcal{A}_{\theta} &= \mathcal{B}_0 \ell \frac{r}{\ell} \exp{ \left ( \frac{-r^2 - h^2}{\ell^2} \right )} \\\\
\mathcal{A}_h &= \mathcal{B}_0 \ell \frac{\alpha}{2}\exp{ \left ( \frac{-r^2 - h^2}{\ell^2} \right )}
\end{align}
$$

The parameters $\alpha$ and $\ell$ can be changed with
```
<problem/cluster/magnetic_tower>
alpha = 20
l_scale = 0.001
```
When injected as a fraction of 

Mass is also injected along with the magnetic field following

$$
\dot{\rho} = \dot{\rho}_B * \exp{ \frac{ -r^2 + -h^2}{\ell^2} }
$$

where $\dot{\rho}_B$ is set to

$$
\dot{\rho}_B = \frac{3 \pi}{2} \frac{\dot{M} \left ( 1 - \epsilon \right ) f_{magnetic}}{\ell^3}
$$

so that the total mass injected matches the accreted mass propotioned to magnetic feedback.

```
<problem/cluster/magnetic_tower>
l_mass_scale = 0.001
```

A magnetic tower can also be inserted at runtime and injected at a fixed
increase in magnetic field, and additional mass can be injected at a fixed
rate.
```
<problem/cluster/magnetic_tower>
initial_field = 0.12431560000204142
fixed_field_rate = 1.0
fixed_mass_rate = 1.0
```

## SNIA Feedback

Following [Prasad 2020](doi.org/10.1093/mnras/112.2.195), AthenaPK can inject
mass and energy from type Ia supernovae following the mass profile of the BCG.
This SNIA feedback can be configured with
```
<problem/cluster/snia_feedback>
power_per_bcg_mass = 0.0015780504379367209 # in units code_length**2/code_time**3
mass_rate_per_bcg_mass = 0.00315576 # in units 1/code_time
disabled = False
```
where `power_per_bcg_mass` and `mass_rate_per_bcg_mass` is the power and mass
per time respectively injected per BCG mass at a given radius. This SNIA
feedback is otherwise fixed in time, spherically symmetric, and dependant on
the BCG specified in `<problem/cluster/gravity>`.
