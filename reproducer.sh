#!/bin/sh

set -x

# compile
mkdir build
cd build
cmake .. -DPARTHENON_ENABLE_PYTHON_MODULE_CHECK=OFF
cmake --build .
cd ..

# first run
mpirun -np 8 ./build/bin/athenaPK -i inputs/precipitator_restart_reproducer.in

# move outputs to avoid being overwritten
mkdir first_run
mv parthenon.* first_run
rm *.csv

# restart run
mpirun -np 8 ./build/bin/athenaPK -r first_run/parthenon.restart.00000.rhdf

# clean up
rm *.csv

# compare restart outputs
h5diff first_run/parthenon.restart.00000.phdf parthenon.restart.00001.phdf
h5diff first_run/parthenon.restart.00001.phdf parthenon.restart.00002.phdf

# compare snapshot outputs
h5diff first_run/parthenon.prim.00000.phdf parthenon.prim.00001.phdf
h5diff first_run/parthenon.prim.00001.phdf parthenon.prim.00002.phdf
