# AthenaPK

AthenaPK: a performance portable version of Athena++ built on Parthenon and Kokkos

## How to build

    # code resides in a private GitLab repo and can only be accessed with two-factor authentication
    # So first you must create a personal access token for your GitLab account, then
    git clone https://TOKEN_NAME:TOKEN@gitlab.com/theias/hpc/jmstone/athenaPK.git
    cd athenaPK

    # get submodules (mainly Kokkos and Parthenon)
    git submodule init
    git submodule update

    # now build
    mkdir build
    cd build
    # enabling Broadwell architecture (AVX2) instructions with OpenMP and HDF5
    cmake -DKokkos_ARCH_BDW=True -DKokkos_ENABLE_OPENMP=True  -DCMAKE_CXX_FLAGS=-O3 -DPROBLEM=SOD ../
    make

    # OR ALTERNATIVELY, disable OpenMP, MPI, and HDF5
    cmake -DKokkos_ARCH_BDW=True -DDISABLE_OPENMP=ON -DDISABLE_MPI=ON -DDISABLE_HDF5=ON -DCMAKE_CXX_FLAGS=-O3 -DPROBLEM=SOD ../
    make

    # run test (doesn't work at the moment)
    ./src/athenaPK -i ../inputs/sod.in

    # now plot results ("cons" are the conserved variables defined as graphics output in parthinput.sod)
    python ../scripts/python/movie1d.py cons *.phdf

### Choosing problem types

The following problem types are currently implemented and availabed through the cmake PROBLEM variable, e.g., `-DPROBLEM=SOD`

- `LINWAVE`     Linear wave (default)
- `SOD`         Shock tube


