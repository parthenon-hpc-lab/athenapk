
#### Tracers

Tracer particles can be enabled and configured via the `<tracers>` block in the input file.

---

##### Enabling Tracers

```ini
<tracers>
enabled = true
```

---

##### Advection Method

Specify the method for advecting tracers:

```ini
advection_method = fluxinterp   # options: fluxinterp (recommended), vinterp
```

---

##### Swarm Populations

Multiple tracer populations (swarms) can be defined and configured independently:

```ini
swarm_names = tracers0,tracers1 # examples, any name could work
```

Each swarm's parameters must be provided **within** the `<tracers>` section.

###### Example:

```ini
# tracers0: persistent population of tracers injected initially
tracers0/initial_num_tracers_per_cell = 1.0
tracers0/injection_enabled    = false
tracers0/removal_enabled      = false

# tracers1: dynamically injected tracers
tracers1/initial_num_tracers_per_cell = 0
tracers1/injection_enabled    = true
tracers1/injection_criteria   = density_above
tracers1/injection_threshold  = 8.0 # in code units
tracers1/injection_timescale  = 0.05
tracers1/injection_num_target = 1

tracers1/removal_enabled             = true
tracers1/removal_exception           = true
tracers1/removal_exception_criteria  = density_above
tracers1/removal_exception_threshold = 8.0 # in code units
tracers1/lifetime                    = 0.05
```

Note that to keep the population of dynamically injected tracers roughly stable in time, it is recommended to match `injection_timescale` and `lifetime`. If `lifetime` is much larger than the injection timescale (or if removal isn't even enabled), the particle's population can grow increasingly large in an uncontrolled fashion.

---

##### Initial Seeding

```ini
initial_seed_method = random_per_block   # alternative: user
```

Two seeding methods are supported:

- **`random_per_block`**
  - Seeds particles randomly in each mesh block.
  - **Note**: the random number generator seed uses the unique block id. Therefore, simulations with the mesh decomposition (mesh and meshblock sizes) are identical independent of the number of MPI ranks used, but if the meshblock size is changed for given mesh (and thus the total number of blocks) the initial state will be different.
  - Controlled by:
    - `tracers0/initial_num_tracers_per_cell`
    - Optional: `initial_rng_seed` to customize randomness

- **`user`**
  - Uses the `ProblemSeedInitialTracers` callback to seed particles manually.
  - See [callback documentation](https://github.com/parthenon-hpc-lab/athenapk/blob/main/docs/pgen.md#tracers)

---

##### Output Configuration

By default, swarm fields are written only to restart files.
If they are required for "standard" output files (like single precision `hdf5`),
they need to be added manually to the output block, e.g., (bottom two lines)

```ini
<parthenon/output2>
file_type  = hdf5
variables  = prim
dt         = 0.1
id         = prim
single_precision_output = true

swarms = tracers0,tracers1
tracers_variables = id, x, y, z, rho
# write_swarm_xdmf = true   # optional: enables xdmf file for Paraview/Visit
```

