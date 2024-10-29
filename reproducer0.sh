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
mpirun -np 8 ./build/bin/athenaPK -r first_run/parthenon.restart.00002.rhdf

# clean up
rm *.csv

# compare t=0 restart outputs
h5diff first_run/parthenon.restart.00002.rhdf parthenon.restart.now.rhdf

# compare with phdf_diff
uv run external/parthenon/scripts/python/packages/parthenon_tools/parthenon_tools/phdf_diff.py --tol 0 first_run/parthenon.restart.00002.rhdf parthenon.restart.now.rhdf
