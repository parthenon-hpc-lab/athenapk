
## Getting AthenaPK


The AthenaPK contains git submodules for Parthenon and Kokkos which are
required for compiling. Make sure to checkout out all submodules with
`--recursive`.

It's also preferrable to build AthenaPK outside of the source directory and to
keep separate builds for CPUs/GPUs and release/debugging builds.
```
#Make a directory to hold the source code and builds for AthenaPK
mkdir athenapk-project
cd athenapk-project
export ATHENAPK_HOME=$(pwd)

# Need proxies to get submodules
export http_proxy=proxyout.lanl.gov:8080
export https_proxy=proxyout.lanl.gov:8080

# Clone the project with all submodules
git clone --recursive git@github.com:parthenon-hpc-lab/athenapk.git
# Without --recursive you can also run
# git submodule --init
# git submodule update --recursive

# Make an out-of-source directory to store builds
mkdir builds

```


##  Compiling AthenaPK for CPUs on Chicoma
Developing and debugging AthenaPK is often easier on CPUs since we can run with
a single thread of execution. Note however that code works on CPUs will not
necessarily work correctly on GPUs since GPUs introduce additional memory and execution spaces.

Compiling AthenaPK for CPUs on Chicoma is best accomplished with the Cray clang environment.
```
# Get a 1 node job for compiling
salloc -N 1 --tasks-per-node=128 -A MY_ALLOCATION -t 4:00:00
```

AthenaPK needs `cmake` to compile and a parallel capable `hdf5` to make full
use of the code. Here's the modules and environment needed to compile AthenaPK
with clang. This same environment is also needed to run the CPU build.
```
# Load the appropriate Cray/Clang environtment
module purge
module load PrgEnv-cray
module load cmake/3.22.3 python/3.10-anaconda-2023.03  cray-hdf5-parallel/1.12.2.1
export CRAY_CPU_TARGET=x86-64
```

For this tutorial we  build AthenaPK for CPUs with Debugging flags, building in
a separate directory just for this build.
```
# Make a directory for a clang  build for CPUs in Debug
mkdir ${ATHENAPK_HOME}/builds/chicoma-clang-Debug/
cd ${ATHENAPK_HOME}/builds/chicoma-clang-Debug/

# Configure the build wtih AMD AVX instructions
cmake -D CMAKE_CXX_COMPILER=CC \
  -D CMAKE_BUILD_TYPE=Debug \
  -D Kokkos_ARCH_AMDAVX=ON \
  -D PARTHENON_ENABLE_PYTHON_MODULE_CHECK=OFF \ # Need unyt for all tests
  -S ../../athenapk -B.

#Build AthenaPK
make -j 32
```

AthenaPK includes a number of regression tests to check correctness of the
code. Note however that the Python module `unyt` is requied for the full test
suite. Additionally, our CPU Debug build is quite slow and so the full test
suite might take hours.
```
# Run the Riemann problem test suite with MPI
# (This will take a very long time!)
ctest -R regression_mpi_test:riemann_hydro
```



##  Compiling AthenaPK for GPUs on Chicoma
Compiling AthenaPK for GPUs on Chicoma is slightly more complicated and
requires very specific modules to get working. 
```
# Get an allocation on the GPU nodes
salloc -N 1 --tasks-per-node=64 -A MY_ALLOCATION_g -p gpu -t 4:00:00
```

We use these modules to build and run AthenaPK for Chicoma on GPUs. Note that
this MPI library supports GPUDirect so that AthenaPK can make MPI calls
directly from GPU memory.
```
# Setup an environment for CUDA
module purge
module load PrgEnv-gnu
module load cpe-cuda cudatoolkit craype-accel-nvidia80
module load cray-mpich/8.1.21 craype-accel-nvidia80
module load cmake/3.22.3 cray-hdf5-parallel/1.12.2.1 python/3.10-anaconda-2023.03
export MPICH_GPU_SUPPORT_ENABLED=1
```

Like the CPU build, we build the GPU build in a separate directory, this time
with release optimization but still with debugging information. Note that we
need to specify `gcc` as the C compiler and the `nvcc_wrapper` included with
Kokkos. We also need additional Kokkos flags to compile for CUDA and the NVIDIA
A100 GPUs on Chicoma.
```
# Make a directory for a CUDA build for A100 GPUs
mkdir ${ATHENAPK_HOME}/builds/cuda-A100-RelWithDebInfo/
cd ${ATHENAPK_HOME}/builds/cuda-A100-RelWithDebInfo/

export KOKKOS_DIR=${ATHENAPK_HOME}/athenapk/external/Kokkos

# Configure the build with CUDA Compute Capability 8.0
cmake \
  -D CMAKE_BUILD_TYPE=RelWithDebInfo \
  -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=$KOKKOS_DIR/bin/nvcc_wrapper \
  -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_CUDA_LAMBDA=ON -D Kokkos_ARCH_AMPERE80=ON \
  -D SERIAL_WITH_MPIEXEC=ON -D TEST_MPIEXEC=srun \
  -D PARTHENON_ENABLE_PYTHON_MODULE_CHECK=OFF \
  -D CMAKE_CXX_FLAGS="${PE_MPICH_GTL_DIR_nvidia80} ${PE_MPICH_GTL_LIBS_nvidia80}" \ 
  -S ../../athenapk -B.

# Build it!
make -j 32
```

We can now run the full test suite with GPUs. However, tests requiring `unyt`
will fail without installing the `unyt` Python module.
```
# Test it!
ctest
```

## Running  AthenaPK on Chicoma GPUs

AthenaPK runs similarly to Athena++. We run with one MPI rank per GPU in our allocation and provide an input file with `-i`. The AthenaPK source has a few example input files
```
#Make sure you have a job with GPUs and that your're running on scratch

#Reports GPUs on this node
nvidia-smi

#Run on 4 GPUs
srun -n 4 ${ATHENAPK_PROJECT}/builds/cuda-A100-RelWithDebInfo/bin/athenaPK  \
  -i ${ATHENAPK_PROJECT}/athenapk/inputs/orszag_tang.in
```

We can also change input parameters from the command line. Here we change the
output type from outputing only the primitives to outputing restart files
```
#Change input parameters from command line
#Rerun making restart outputs
srun -n 4 ${ATHENAPK_PROJECT}/builds/cuda-A100-RelWithDebInfo/bin/athenaPK  \
  -i ${ATHENAPK_PROJECT}/athenapk/inputs/orszag_tang.in parthenon/output0/file_type=rst
```

We can also continue the simulation (a restart) and change paramters. Here we
restart from the last output and run for longer.
```
#Restart simulation and evolve for longer
srun -n 2 ${ATHENAPK_PROJECT}/builds/cuda-A100-RelWithDebInfo/bin/athenaPK  \
 -r parthenon.prim.final.rhdf parthenon/time/tlim=2.0
```

## Analysis with yt (using custom parthenon-frontend fork)

There are several options for analyzing AthenaPK outputs including Paraview,
Visit, and yt. To read AthenaPK outputs with `yt`, we need the
`parthenon-frontend` fork of `yt`. We can install this module into a new
virtual environment if desired.

```
# Need proxies to get source
export http_proxy=proxyout.lanl.gov:8080
export https_proxy=proxyout.lanl.gov:8080

# Get the yt fork with the Parthenon Frontend and change to the parthenon branch
git clone git@github.com:forrestglines/yt.git
cd yt/
git checkout -b parthenon-frontend origin/parthenon-frontend

## OPTIONAL ##
# Setup a virtual python environment to install yt
module load python/3.10-anaconda-2023.03
python -m venv python3.10-yt
#Activate the new environment
source ${ATHENAPK_PROJECT}/python3.10-yt/bin/activate
## END OPTIONAL ##

# Install yt into the virtual environment
pip install -e .
```

We can plot outputs with yt like any other code supported by `yt`
```python
# plotting_with_yt.py
import yt

sim_dir = "/PATH/TO/SIMULATION/"

ds = yt.load(f"{sim_dir}/parthenon.prim.final.rhdf")

slc = yt.SlicePlot(ds,"z",fields=("gas","vorticity_z") )
slc.save()
slc.show()
```
