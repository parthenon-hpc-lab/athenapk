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

setup_test_both("jeans" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/jeans.in --num_steps 2" "convergence")

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

  # Serial only: dynamic tracer injection seeds its RNG from (cell indices, block gid,
  # time), and block gid assignment is decomposition-dependent, so the exact set of
  # injected tracers -- and thus this test's exact-match comparison against
  # ref_data.pkl -- isn't invariant across different rank counts.
  setup_test_serial("cluster_tracers" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/cluster/hydro_agn_feedback_tracers.in --num_steps 2" "regression")

  # Serial-only for now. The 2-rank crashes at cycle 0 an earlier note reported here
  # for every swarm setup (an MPI send-request assertion in bvals.cpp, a
  # std::out_of_range in a neighbor/rank map) no longer reproduce: particle_advection
  # runs with setup_test_both, and every non-restart step of the star tests below ran
  # on 1 and 2 ranks (Oct 2026, outputs switched to history files, no parallel HDF5
  # locally) without crashes or hangs, with the same history up to round-off in
  # near-zero momentum sums. Still to do before setup_test_both: run these tests
  # through the regression framework with MPI and parallel HDF5.
  setup_test_serial("star_formation_sfr_stats" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/star_formation_sfr_stats.in --num_steps 1" "regression")

  # The same setup reduced to 32^3, run through and restarted halfway
  setup_test_serial("star_formation_restart" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/star_formation_sfr_stats.in --num_steps 2" "regression")

  setup_test_serial("cluster_star_orbits" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/cluster/star_orbits.in --num_steps 1" "regression")

  # Serial-only for now (see the note above); of the star tests, this one exercises
  # the ghost-particle exchange the most.
  setup_test_serial("sn_feedback_refinement" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/sn_feedback_refinement.in --num_steps 7" "regression")

  # Single forced SN on uniform, static and adaptive meshes, plus Sedov-Taylor;
  # serial-only (see the note above). Step 2 uses a single meshblock, so it can't run
  # on 2 ranks: an MPI variant needs it skipped or changed.
  setup_test_serial("sn_blast_wave" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/sn_blast_wave.in --num_steps 7" "regression")

  # Natural SN II/Ia rates of one massive star against the model, then a restart;
  # serial-only for the same reason as above.
  setup_test_serial("sn_rates" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/sn_rates.in --num_steps 2" "regression")

  setup_test_both("particle_advection" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/turbulence.in --num_steps 2" "regression")
  
  setup_test_serial("lw_implode_symmetry" "--driver ${PROJECT_BINARY_DIR}/bin/athenaPK \
  --driver_input ${PROJECT_SOURCE_DIR}/inputs/lw_implode.in --num_steps 1" "regression")
