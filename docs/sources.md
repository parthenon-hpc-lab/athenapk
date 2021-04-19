# Source functions

Additional (physical) source terms (e.g., the ones typically on the right hand side of
system of equations) can be added in various ways from an algorithmic point of view.

## Unsplit 

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

## Split first order (generally not recommended)

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


## Split second order (Strang splitting)

Not implemented yet.

