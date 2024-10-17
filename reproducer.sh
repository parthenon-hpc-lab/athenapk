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
mpirun -np $NRANKS ./build/bin/athenaPK -r first_run/parthenon.restart.00000.rhdf

## compare first outputs after restart
# NOTE: This comparison will fail with Parthenon 24.08 --> current develop (as of 17 Oct 24) !!
# (Restarting from 00000.rhdf will produce an output numbered 00001.rhdf that is identical to the original 00000 rhdf output.)

echo "Comparing first outputs post-restart..."
h5diff first_run/parthenon.prim.00001.phdf parthenon.prim.00001.phdf
h5diff first_run/parthenon.restart.00001.rhdf parthenon.restart.00001.rhdf

## compare second outputs after restart

echo "\nComparing second outputs after restart..."
h5diff first_run/parthenon.prim.00002.phdf parthenon.prim.00002.phdf
h5diff first_run/parthenon.restart.00002.rhdf parthenon.restart.00002.rhdf

