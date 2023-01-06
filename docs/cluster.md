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
which_bcg_g = HERNQUIST #or NONE or MATHEWS
include_smbh_g = True
```
Where `include_nfw_g` for the NFW dark-matter halo (CITEME) is boolean;
`which_bcg_g` for the BCG can be `NONE` for no BCG, `HERNQUIST` for  Hernquist
profile (CITEME), or `MATHEWS` for a Mathews (CITEME) profile, and
`include_smbh_g` for the SMBH is boolean.

Parameters for the NFW profile are
```
<problem/cluster/gravity>
c_nfw = 6.0 # Unitless
m_nfw_200 = 10.0 # in code_mass
```
which adds a gravitational acceleration defined by

(EQUATIONME)


Parameter for both a HERNQUIST and MATHEWS BCG are:
```
<problem/cluster/gravity>

m_bcg_s = 0.001 # in code_mass
r_bcg_s = 0.004 # in code_length
```
where a HERNQUIST profile adds a gravitational acceleration defined by

(EQUATIONME)

and a MATHEWS profile adds a gravitational acceleration defined by

(EQUATIONME)

Gravitational acceleration from the SMBH is defined solely by its mass
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

(EQUATIONME)

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

(EQUATIONME, DEFINE ENTROPY K)

This profile is determined by these parameters
```
<problem/cluster/entropy_profile>
k_0 = 8.851337676479303e-121 # in FIXME
k_100 = 1.3277006514718954e-119 # in FIXME
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

(FIXME) `r_sampling` and `max_dr` determine details about how the profile for
the initial hydrodynamic equilbrium is integrated. They can potentially be
removed.
```
<problem/cluster/hydrostatic_equilibrium>
r_fix = 2.0 # in code_length
r_sampling = 4.0
max_dr = 0.001
```

## AGN Triggering

If AGN triggering is enabled, at the end of each time step, a mass accretion
rate `mdot` is determined from conditions around the AGN according to the
different triggering prescriptions. The accreted mass is removed from the gas
around the AGN, with details depending on each prescription explained below,
and is used as input for AGN feedback power.

The triggering prescriptions currently implemented are "boosted Bondi accretion," "Bondi-Schaye accretion," and "cold gas." These modes can be chosen via
```
<problem/cluster/agn_triggering>
triggering_mode = COLD_GAS # or NONE, BOOSTED_BONDI, BONDI_SCHAYE
```
where `triggering_mode=NONE` will disable AGN triggering. 

With BOOSTED_BONDI accretion, the mass rate of accretion follows

(EQUATIONME)

where `rho`, `v`, and `cs` are respectively the mass weighted density,
velocity, and sound speed within the accretion region. The mass of the SMBH,
the radius of the sphere of accretion around the AGN, and the `alpha` parameter
can be set with
```
<problem/cluster/gravity>
m_smbh = 1.0e-06 # in code_mass

<problem/cluster/agn_triggering>
accretion_radius = 0.001 # in code_length
bondi_alpha= 100.0 # unitless
```

With BONDI_SCHAYE accretion, the `alpha` used for BOOSTED_BONDI accretion is modified to depend on the number density following:

(EQUATIONME)

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
where `total_mass` is the total mass within the accretion zone. This mass is
removed from the conserved variables such that the velocity and temperature of
the cell remains the same. Momentum and energy density thus will also change.

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
the velocity and temperature of the cell remains the same. Momentum and energy
density thus will also change.


## AGN Feedback

AGN feedback can be both triggered via the mechanisms in the section above and with a fixed power.
```
<problem/cluster/agn_feedback>
fixed_power = 0.0
efficiency = 0.001
```
Where and `mdot` calculated from AGN triggering will lead to an an AGN feedback
power of `agn_power = efficiency*mdot*c**2`. The fixed power and triggered
power are not mutually exclusive; if both `fixed_power` is defined and
triggering is enabled with a non-zero `efficiency`, then the `fixed_power` will
be added to the triggered AGN power.


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
thermal_injected_mass = mdot * normalized_thermal_fraction;
```

Kinetic feedback is deposited into cylinder along the axis of the jet within a
defined radius and height above and below the plane of the AGN disk.
```
<problem/cluster/agn_feedback>
kinetic_jet_radius  = 0.0005
kinetic_jet_height  = 0.0005
```
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

Kinetic jet energy is injected at a flat power density within the cynlinder of
injection as purely kinetic energy. The injected mass will match the
proportioned kinetic feedback, e.g.
```
kinetic_injected_mass = mdot * normalized_kinetic_fraction;
```
and the injected momentum will total the injected kinetic feedback energy. Gas
energy desnity will remain unchanged. As a result, the injected mass density  rate will be

(EQUATIONME)

and the velocity of the injected gas will be 

(EQUATIONME).

Magnetic feedback is injected following  (CITEME) where the injected magnetic field follows 

(EQUATIONME).

The parameters `alpha` and `l` 
```
<problem/cluster/magnetic_tower>
alpha = 20
l_scale = 0.001
```

Mass is also injected along with the magnetic field following

(EQUATIONME)

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

