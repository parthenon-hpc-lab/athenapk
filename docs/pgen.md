# Custom problem generators

Different simulation setups in AthenaPK are controlled via the so-called problem generators.
New problem generators can easily be added and we are happy to accept and merge contibuted problem
generators by any user via pull requests.

## Addding a new problem generator

In general, four small steps are required:

### 1. Add a new source file to the `src/pgen` folder

The file shoud includes at least the `void ProblemGenerator(Mesh *pmesh, ParameterInput *pin, MeshData<Real> *md)`
function, which is used to initialize the data at the beginning of the simulation.
Alternatively, the `MeshBlock` version (`void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin)`) can be used that
operates on a single block at a time rather than on a collection of blocks -- though this is not recommended from a performance point of view.

The function (and all other functions in that file) should be encapsulated in their own namespace (that, ideally, is named
identical to the file itself) so that the functions name are unique across different problem generators.

Tip: Do not write the problem generator file from scratch but mix and match from existing ones,
e.g., start with a simple one like [orszag_tang.cpp](../src/pgen/orszag_tang.cpp).

### 2. Add new function(s) to `pgen.hpp`

All callback functions, i.e., at least the `ProblemGenerator` plus additional optional ones (see step 5 below),
need to be added to [pgen.hpp](../src/pgen/pgen.hpp).
Again, just follow the existing pattern in the file and add the new function declarations with the appropriate namespace.

### 3. Add callbacks to `main.cpp`

All problem specific callback functions need to be enrolled in the [`main.cpp`](../src/main.cpp) file.
The selection (via the input file) is controlled by the `problem_id` string in the `<job>` block.
Again, for consistency it is recommended to pick a string that matches the namespace and problem generator file.

### 4. Ensure new problem generator is compiled

Add the new source file to [src/pgen/CMakeLists.txt](../src/pgen/CMakeLists.txt) so that it will be compiled
along all other problem generators.

### 5. (Optional) Add more additional callback

In addition to the `ProblemGenerator` that initializes the data, other callback functions exists
that allow to modify the behavior of AthenaPK on a problem specific basis.
See [Callback functions](#Callback-functions)] below for available options.

### 6. (Optional but most likely required) Write an input file

In theory, one can hardcode all paramters in the source file (like in the
[orszag_tang.cpp](../src/pgen/orszag_tang.cpp) problem generator) but it
prevents modification of the problem setup once the binary is compiled.

The more common usecase is to create an input file that contains a problem specific
input block.
The convention here is to have a block named `<problem/NAME>` where `NAME` is the name
of the problem generator (or namespace used).
For example, the Sod shock tube problem generator processes the input file with lines like
```
  Real rho_l = pin->GetOrAddReal("problem/sod", "rho_l", 1.0);
```
to set the density on the left hand side of the domain.

## Callback functions

### Source functions

Additional (physical) source terms (e.g., the ones typically on the right hand side of
system of equations) can be added in various ways from an algorithmic point of view.

#### Unsplit

Unsplit sources are added at each stage of the (multi-stage) integration after the
(conserved) fluxes are calculated.
Therefore, the "conserved" variables should be updated using the "primitive" ones
and by taking into account the corresponding `dt` (more specifically the `beta*dt`
of the particular stage), see, for example, the `DednerSource` function in
[dedner_source.cpp](../src/hydro/glmmhd/dedner_source.cpp).

Users can enroll a custom source function
```c++
void MyUnsplitSource(MeshData<Real> *md, const Real beta_dt);
```
by implementing a function with that signature and assigning it to the
`Hydro::ProblemSourceUnsplit` callback (currently in `main.cpp`).

Note, there is no requirement to call the function `MyUnsplitSource`.

#### Split first order (generally not recommended)

If for special circumstances sources need to be added in a fully operator split way,
i.e., the source function is only called *after* the full hydro (or MHD) integration,
a custom function
```c++
void MyFirstOrderSource(MeshData<Real> *md, const parthenon::SimTime &tm);
```
can be enrolled to the `Hydro::ProblemSourceFirstOrder` callback (currently in `main.cpp`).
The current `dt` can be accessed through `tm.dt`.

Note, as the name suggests, this method is only first order accurate (while there
is no requirement to call the function `MyFirstOrderSource`).


#### Split second order (Strang splitting)

Strang splitting achieves second order by interleaving the main hydro/MHD update
with a source update.
In practice, AthenaPK calls Strang split source with `0.5*dt` before the first stage of
each integrator and with `0.5*dt` after the last stage of the integrator.
Note that for consistency, a Strang split source terms should update both conserved
and primitive variables in all zones (i.e., also in the ghost zones as those
are currently not updated after calling a source function).

Users can enroll a custom source function
```c++
void MyStrangSplitSource(MeshData<Real> *md, const Real beta_dt);
```
by implementing a function with that signature and assigning it to the
`Hydro::ProblemSourceStrangSplit` callback (currently in `main.cpp`).

Note, there is no requirement to call the function `MyStrangSplitSource`.


### Timestep restrictions

If additional problem specific physics are implemented (e.g., through the source
functions above) that require a custom timestep, it can be added via the
`ProblemEstimateTimestep` callback with the following signature
```c++
Real ProblemEstimateTimestep(MeshData<Real> *md);
```

The return value is expected to be the minimum value over all blocks in the
contained in the `MeshData` container, cf., the hydro/mhd `EstimateTimestep` function.
Note, that the hyperbolic CFL value is currently applied after the function call, i.e.,
it is also applied to the problem specific function.

### Additional initialization on startup (adding variables/fields, custom history output, ...)

Sometimes a problem generator requires a more general processing of the input file
and/or needs to make certain global adjustments (e.g., adding a custom callback function
for the history output or adding a new field).
This can be achieved via modifying the AthenaPK "package" (currently called
`parthenon::StateDescriptor`) through the following function.
```c++
void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg)
```

For example, the [src/pgen/turbulence.cpp](../[src/pgen/turbulence.cpp]) problem generator
add an additional field (here for an acceleration field) by calling
```c++
  Metadata m({Metadata::Cell, Metadata::Derived, Metadata::OneCopy},
             std::vector<int>({3}));
  pkg->AddField("acc", m);
```
in the `ProblemInitPackageData`.

### Additional initialization on mesh creation/remeshing/load balancing

For some problem generators it is required to initialize data on "new" blocks.
These new blocks can, for example, be created during mesh refinement
(or derefinement) and this data is typically not active data (like
conserved or primitive variables as those are handled automatically)
but more general data.
Once example is the phase information in the turbulence driver that
does not vary over time but spatially (and therefore at a per block level).

The appropriate callback to enroll is
```c++
void InitMeshBlockUserData(MeshBlock *pmb, ParameterInput *pin)
```

### UserWorkAfterLoop

If additional work is required once the main loop finishes (i.e., once the
simulation reaches its final time) computations can be done inside the
```c++
void UserWorkAfterLoop(Mesh *mesh, ParameterInput *pin, parthenon::SimTime &tm)
```
callback function.
This is, for example, done in the linear wave problem generator to calculate the
error norms for convergence tests.

### MeshBlockUserWorkBeforeOutput

Sometimes it is desirable to further process data before an output file is written.
For this the
```c++
void UserWorkBeforeOutput(MeshBlock *pmb, ParameterInput *pin)
```
callback is available, that is called once every time right before a data output
(hdf5 or restart) is being written.
