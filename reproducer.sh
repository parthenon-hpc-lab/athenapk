#!/bin/sh

set -x

NRANKS=8
BUILD_TYPE=Release

# compile
rm -rf build
mkdir build
cd build
cmake .. -DPARTHENON_ENABLE_PYTHON_MODULE_CHECK=OFF -DCMAKE_BUILD_TYPE=$BUILD_TYPE
cmake --build .
cd ..

# first run
mpirun -np $NRANKS ./build/bin/athenaPK -i inputs/restart_reproducer.in

# move outputs to avoid being overwritten
mkdir first_run
mv parthenon.* first_run

# restart run
mpirun -np $NRANKS ./build/bin/athenaPK -r first_run/parthenon.restart.00001.rhdf

## compare first outputs after restart

echo "\nComparing second outputs after restart..."
h5diff first_run/parthenon.prim.00002.phdf parthenon.prim.00002.phdf
h5diff first_run/parthenon.restart.00002.rhdf parthenon.restart.00002.rhdf

## compare with internal tool
uv run external/parthenon/scripts/python/packages/parthenon_tools/parthenon_tools/phdf_diff.py parthenon.restart.00002.rhdf first_run/parthenon.restart.00002.rhdf
