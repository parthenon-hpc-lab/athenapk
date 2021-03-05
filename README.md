# AthenaPK

AthenaPK: a performance portable version of Athena++ built on [Parthenon](https://github.com/lanl/parthenon) and [Kokkos](https://github.com/kokkos/kokkos).

## Current state of the code

**WARNING** This code is under active development, i.e., the interfaces are not considered stable at this point.

For this reason, it is highly recommended to only use AthenaPK with the Kokkos and Parthenon versions that are provided by the submodules (see [building](#building)) and to build everything (AthenaPK, Parthenon, and Kokkos) together from source.
Neither other versions or nor using preinstalled Parthenon/Kokkos libraries have been tested.

Current features include
- first and second order hydrodynamics with
  - RK1, RK2, RK3, VL2 integrators
  - piecewise constant (DC), piecewise linear (PLM), piecewise parabolic (PPM), and WENOZ reconstruction
  - HLLE Riemann solver
  - Adiabatic equation of state
- static and adaptive mesh refinement
- problem generators for
  - a linear wave
  - blast wave
  - Kelvin-Helmholtz instability

Latest performance results for various methods on a single Nvidia Volta V100 can be found [here](https://gitlab.com/theias/hpc/jmstone/athena-parthenon/athenapk/-/jobs/artifacts/main/file/build-cuda/tst/regression/outputs/performance/performance.png?job=cuda-regression).

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
* C++14 compatible compiler
* Parthenon (using the submodule version provided by AthenaPK)
* Kokkos (using the submodule version provided by AthenaPK)

##### Optional

* MPI
* OpenMP (for host parallelism. Note that MPI is the recommended option for on-node parallelism.)
* HDF5 (for outputs)

#### Building AthenaPK

Obtain all (AthenaPK, Parthenon, and Kokkos) sources

    git clone https://gitlab.com/theias/hpc/jmstone/athena-parthenon/athenapk.git athenaPK
    cd athenaPK

    # get submodules (mainly Kokkos and Parthenon)
    git submodule init
    git submodule update

Most of the general build instructions and options for Parthenon (see [here](https://github.com/lanl/parthenon/blob/develop/docs/building.md)) also apply to AthenaPK.
The following examples are a few standard cases.

Most simple configuration (only CPU, no MPI, no HDF5)

    # enabling Broadwell architecture (AVX2) instructions
    cmake -S. -Bbuild-host -DKokkos_ARCH_BDW=ON -DPARTHENON_DISABLE_MPI=ON -DPARTHENON_DISABLE_HDF5=ON ../
    cd build-host && make

An Intel Skylake system (AVX512 instructions) with NVidia Volta V100 GPUs and with MPI and HDF5 enabled (the latter is the default option, so they don't need to be specified)

    cmake -S. -Bbuild-gpu -DKokkos_ARCH_SKX=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_VOLTA70=ON ../
    cd build-gpu && make

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

