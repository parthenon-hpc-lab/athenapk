# Modules for building/running AthenaPK on JUPITER (Stages/2025, GCC + CUDA 12.6)
module --force purge
module load Stages/2025
module load GCC CUDA OpenMPI HDF5 CMake Ninja SciPy-bundle/2024.05 h5py matplotlib
export KT=${KT:-/e/scratch/jureap19/fournier/stars_test/athenapk-sfeed-profiling/kokkos-tools}
