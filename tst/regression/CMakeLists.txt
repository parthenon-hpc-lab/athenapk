# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2020-2022, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

# import Parthenon setup_test_serial and setup_test_parallel
include(${PROJECT_SOURCE_DIR}/external/parthenon/cmake/TestSetup.cmake)

setup_test_serial("convergence" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/linear_wave3d.in --num_steps 36" "convergence")

setup_test_serial("mhd_convergence" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/linear_wave3d.in --num_steps 40" "convergence")

setup_test_serial("performance" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/linear_wave3d.in --num_steps 20" "performance")

setup_test_serial("cluster_hse" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/cluster/hse.in --num_steps 1" "convergence")

setup_test_serial("cluster_tabular_cooling" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/cluster/cooling.in --num_steps 10" "convergence")

  setup_test_serial("aniso_therm_cond_ring_conv" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/diffusion.in --num_steps 4" "convergence")
  
  setup_test_serial("aniso_therm_cond_ring_multid" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/diffusion.in --num_steps 4" "convergence")

  setup_test_serial("field_loop" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/field_loop.in --num_steps 12" "convergence")
