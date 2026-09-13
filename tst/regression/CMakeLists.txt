# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2020-2025, Athena Parthenon Collaboration. All rights reserved.
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
 
setup_test_both("aniso_therm_cond_gauss_conv" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/diffusion.in --num_steps 24" "convergence")

setup_test_both("diffusion" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/diffusion.in --num_steps 12" "convergence")

  setup_test_both("diffusion_linwave3d" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/linear_wave3d.in --num_steps 2" "convergence")

setup_test_both("field_loop" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/field_loop.in --num_steps 12" "convergence")

setup_test_both("cluster_magnetic_tower" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/cluster/magnetic_tower.in --num_steps 4" "convergence")

setup_test_both("cluster_hydro_agn_feedback" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/cluster/hydro_agn_feedback.in --num_steps 5" "convergence")

setup_test_both("cluster_agn_triggering" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/cluster/agn_triggering.in --num_steps 3" "convergence")

  setup_test_both("cluster_tracers" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/cluster/hydro_agn_feedback_tracers.in --num_steps 1" "regression")

  # Serial only for now: the 2-rank variant of setup_test_both crashes at cycle 0 for
  # *any* particle-swarm-enabled setup on this build, not just these two -- confirmed
  # by independently reproducing two different crashes (an MPI send-request assertion
  # in bvals.cpp, and a std::out_of_range in a neighbor/rank map) on the pre-existing,
  # untouched tracers-based particle_advection_vinterp test as well, while a pure-hydro
  # MPI run (sod.in) completes cleanly. So this looks like a pre-existing swarm/MPI
  # issue in the shared particle infrastructure, not something specific to either test
  # below -- switch back to setup_test_both once that's fixed.
  setup_test_serial("star_formation_sfr_stats" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/star_formation_sfr_stats.in --num_steps 1" "regression")

  setup_test_serial("cluster_star_orbits" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/cluster/star_orbits.in --num_steps 1" "regression")

  # Also serial-only for now -- see the note above; this one exercises the exact same
  # SwarmContainer::Send/Receive ghost-particle path (more so, in fact: it's what that
  # path exists for) so it should get the same MPI re-check once the fix lands.
  setup_test_serial("sn_feedback_refinement" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/sn_feedback_refinement.in --num_steps 5" "regression")

  setup_test_both("particle_advection_fluxinterp" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/turbulence.in --num_steps 2" "regression")

  setup_test_both("particle_advection_vinterp" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/turbulence.in --num_steps 2" "regression")
  
  setup_test_serial("lw_implode_symmetry" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/lw_implode.in --num_steps 1" "regression")
