# AthenaPK

AthenaPK: a performance portable version based on [Athena++](https://github.com/PrincetonUniversity/athena),  [Parthenon](https://github.com/parthenon-hpc-lab/parthenon) and [Kokkos](https://github.com/kokkos/kokkos).

## Overview

It is highly recommended to only use AthenaPK with the Kokkos and Parthenon versions that are provided by the submodules (see [building](#building)) and to build everything (AthenaPK, Parthenon, and Kokkos) together from source.
Neither other versions or nor using preinstalled Parthenon/Kokkos libraries have been tested.

Current features include
- first, second, and third order (magneto)hydrodynamics with
  - RK1, RK2, RK3, VL2 integrators
  - piecewise constant (DC), piecewise linear (PLM), piecewise parabolic (PPM), WENO3, LimO3, and WENOZ reconstruction
  - HLLE (hydro and MHD), HLLC (hydro), and HLLD (MHD) Riemann solvers
  - adiabatic equation of state
  - MHD based on hyperbolic divergence cleaning following Dedner+ 2002
  - diffusion processes
    - isotropic and anisotropic thermal conduction
    - viscosity
    - resistivity
  - diffusion integrator
    - unsplit
    - operator-split, second-order RKL2 supertimestepping
  - optically thin cooling based on tabulated cooling tables with either Townsend 2009 exact integration or operator-split subcycling
- static and adaptive mesh refinement
- problem generators for
  - linear waves
  - circularly polarized Alfven wave
  - blast wave
  - Kelvin-Helmholtz instability
  - field loop advection
  - Orszag Tang vortex
  - cloud-in-wind/cloud crushing
  - turbulence (with stochastic forcing via an Ornstein-Uhlenbeck process)

Latest performance results for various methods on a single Nvidia Ampere A100 can be found [here](https://github.com/parthenon-hpc-lab/athenapk/actions/workflows/ci.yml).

## Getting in touch

If you
* encounter a bug or problem,
* have a feature request,
* would like to contribute, or
* have a general question or comment

please either
- open an issue/merge request, or
- contact us in the AthenaPK channel on matrix.org [#AthenaPK:matrix.org](https://app.element.io/#/room/#AthenaPK:matrix.org)

## Getting started

### Installation

#### Dependencies

##### Required

* CMake 3.13 or greater
* C++17 compatible compiler
* Parthenon (using the submodule version provided by AthenaPK)
* Kokkos (using the submodule version provided by AthenaPK)

##### Optional

* MPI
* OpenMP (for host parallelism. Note that MPI is the recommended option for on-node parallelism.)
* HDF5 (for outputs)
* Python3 (for regressions tests with numpy, scipy, matplotlib, unyt, and h5py modules)
* Ascent (for in situ visualization and analysis)

#### Building AthenaPK

Obtain all (AthenaPK, Parthenon, and Kokkos) sources

    git clone https://github.com/parthenon-hpc-lab/athenapk.git athenapk
    cd athenapk

    # get submodules (mainly Kokkos and Parthenon)
    git submodule init
    git submodule update

Most of the general build instructions and options for Parthenon (see [here](https://parthenon-hpc-lab.github.io/parthenon/develop/src/building.html)) also apply to AthenaPK.
The following examples are a few standard cases.

Most simple configuration (only CPU, no MPI).
The `Kokkos_ARCH_...` parameter should be adjusted to match the target machine where AthenaPK will be executed.
A full list of architecture keywords is available on the [Kokkos wiki](https://kokkos.github.io/kokkos-core-wiki/keywords.html#architecture-keywords).

    # configure with enabling Intel Broadwell or similar architecture (AVX2) instructions
    cmake -S. -Bbuild-host -DKokkos_ARCH_BDW=ON -DPARTHENON_DISABLE_MPI=ON
    # now build with
    cd build-host && make
    # or alternatively
    cmake --build build-host

If `cmake` has troubling finding the HDF5 library (which is required for writing analysis outputs or
restartings simulation) an additional hint to the location of the library can be provided via
`-DHDF5_ROOT=/path/to/local/hdf5` on the first `cmake` command for configuration.

An Intel Skylake system (AVX512 instructions) with NVidia Volta V100 GPUs and with MPI enabled (the latter is the default option, so they don't need to be specified)

    cmake -S. -Bbuild-gpu -DKokkos_ARCH_SKX=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON
    # now build with
    cd build-gpu && make
    # or alternatively build with
    cmake --build build-gpu

#### Run AthenaPK

Some example input files are provided in the [inputs](inputs/) folder.

    # for a simple linear wave test run
    ./bin/athenaPK -i ../inputs/linear_wave3d.in

    # to run a convergence test:
    for M in 16 32 64 128; do
      export N=$M;
      ./bin/athenaPK -i ../inputs/linear_wave3d.in parthenon/meshblock/nx1=$((2*N)) parthenon/meshblock/nx2=$N parthenon/meshblock/nx3=$N parthenon/mesh/nx1=$((2*M)) parthenon/mesh/nx2=$M parthenon/mesh/nx3=$M
    done

    # and check the resulting errors
    cat linearwave-errors.dat

#### Data Analysis

There exit several options to read/process data written by AthenaPK -- specifically in
the `file_type = hdf5` format, see
[Parthenon doc](https://parthenon-hpc-lab.github.io/parthenon/develop/src/outputs.html):

1. With [ParaView](https://www.paraview.org/) and
[VisIt](https://wci.llnl.gov/simulation/computer-codes/visit/).
In ParaView, select the "XDMF Reader" when prompted.

2. With [yt](https://yt-project.org/)
As of versions >=4.4 `*.phdf` files can be read as usual with `yt.load()`.

3. Using [Ascent](https://github.com/Alpine-DAV/ascent) (for in situ visualization and analysis).
This requires Ascent to be installed/available at compile time of AthenaPK.
To enable set `PARTHENON_ENABLE_ASCENT=ON`.

4. (Not recommended) Using the integrated Python script called "`phdf`" provided by Parthenon,
i.e., the either install `parthenon_tools`
(located in `external/parthenon/scripts/python/packages/parthenon/tools`) or add
that directory to your Python path.
Afterwards data can be read, e.g., as follows
```Python
data_file = phdf.phdf(data_filename)
prim = data_file.Get("prim")
```
see also an internal regression test that uses this interface [here](tst/regression/test_suites/aniso_therm_cond_ring_conv/aniso_therm_cond_ring_conv.py).
