# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2026, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

# Reuse Parthenon's Catch2 setup (Parthenon's own tests are disabled in AthenaPK, so its
# catch2_define target does not exist here).
if (NOT TARGET Catch2::Catch2)
  find_package(Catch2 2.13.8 QUIET)
  if (NOT Catch2_FOUND)
    set(PARTHENON_CATCH2_DIR ${PROJECT_SOURCE_DIR}/external/parthenon/external/Catch2)
    if (NOT EXISTS ${PARTHENON_CATCH2_DIR}/single_include)
      message(STATUS "Downloading Catch2 submodule.")
      execute_process(COMMAND git submodule update --init --force -- external/Catch2
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/external/parthenon)
    endif()
    add_subdirectory(${PARTHENON_CATCH2_DIR} Catch2)
    list(APPEND CMAKE_MODULE_PATH "${PARTHENON_CATCH2_DIR}/contrib")
  endif()
endif()
include(Catch)

add_executable(athenapk_unit_tests
  ${PROJECT_SOURCE_DIR}/external/parthenon/tst/catch2_define.cpp
  test_generate_stochastic_b_field.cpp
  ${PROJECT_SOURCE_DIR}/src/utils/generate_stochastic_b_field.cpp
)
target_include_directories(athenapk_unit_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(athenapk_unit_tests PRIVATE parthenon Catch2::Catch2 MPI::MPI_CXX)
# Lets catch2_define.cpp initialize MPI for tests tagged [MPI]
target_compile_definitions(athenapk_unit_tests PRIVATE CATCH2_MPI_PARALLEL)

catch_discover_tests(athenapk_unit_tests PROPERTIES LABELS "unit")

add_test(NAME athenapk_unit_tests_mpi
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NUM_MPI_PROC_TESTING}
    ${MPIEXEC_PREFLAGS} $<TARGET_FILE:athenapk_unit_tests> "[MPI]" ${MPIEXEC_POSTFLAGS})
set_tests_properties(athenapk_unit_tests_mpi PROPERTIES LABELS "unit;mpi")
