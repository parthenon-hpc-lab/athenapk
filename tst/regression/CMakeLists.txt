# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2020-2023, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

# Any external modules that are required by python can be added to REQUIRED_PYTHON_MODULES
# list variable, before including TestSetup.cmake.
list(APPEND REQUIRED_PYTHON_MODULES numpy unyt matplotlib h5py scipy)
message(STATUS
  "Checking for Python modules (${REQUIRED_PYTHON_MODULES}) required for regression tests.\n"
  "   Check can be disabled by setting PARTHENON_ENABLE_PYTHON_MODULE_CHECK=OFF but then\n"
  "   tests are not guaranteed to work anymore.")

# import Parthenon setup_test_serial and setup_test_parallel
include(${PROJECT_SOURCE_DIR}/external/parthenon/cmake/TestSetup.cmake)

function(setup_test_both dir arg test_labels)
  setup_test_serial(${dir} "${arg}" "${test_labels}")
  # just using 2 ranks for MPI tests
  setup_test_parallel(2 ${dir} "${arg}" "${test_labels}")
endfunction()

setup_test_both("riemann_hydro" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/sod.in --num_steps 39" "other")

setup_test_both("turbulence" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/turbulence.in --num_steps 1" "other")

setup_test_both("convergence" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/linear_wave3d.in --num_steps 40" "convergence")

setup_test_both("mhd_convergence" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/linear_wave3d.in --num_steps 48" "convergence")

setup_test_serial("performance" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/linear_wave3d.in --num_steps 21" "performance")

setup_test_both("cluster_hse" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/cluster/hse.in --num_steps 2" "convergence")

setup_test_serial("cluster_tabular_cooling" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/cluster/cooling.in --num_steps 11" "convergence")

setup_test_both("aniso_therm_cond_ring_conv" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/diffusion.in --num_steps 4" "convergence")
  
setup_test_both("aniso_therm_cond_ring_multid" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/diffusion.in --num_steps 4" "convergence")

setup_test_both("field_loop" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/field_loop.in --num_steps 12" "convergence")

setup_test_both("cluster_magnetic_tower" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/cluster/magnetic_tower.in --num_steps 4" "convergence")

setup_test_both("cluster_hydro_agn_feedback" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/cluster/hydro_agn_feedback.in --num_steps 5" "convergence")

setup_test_both("cluster_agn_triggering" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/cluster/agn_triggering.in --num_steps 3" "convergence")
