#=========================================================================================
# AthenaPK - a performance portable block structured AMR MHD code
# Copyright (c) 2021, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");
#=========================================================================================
# (C) (or copyright) 2020. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department
# of Energy/National Nuclear Security Administration. The Government is granted for
# itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works, distribute copies to
# the public, perform publicly and display publicly, and to permit others to do so.
#=========================================================================================

find_program(
    CLANG_FORMAT
    NAMES
        clang-format-9 # Debian package manager, among others, provide this name
        clang-format-mp-9.0 # MacPorts
        clang-format # Default name
    )

if (CLANG_FORMAT AND NOT CLANG_FORMAT_VERSION)
    # Get clang-format --version
    execute_process(
        COMMAND ${CLANG_FORMAT} --version
        OUTPUT_VARIABLE CLANG_FORMAT_VERSION_OUTPUT)

    if (CLANG_FORMAT_VERSION_OUTPUT MATCHES "clang-format version ([0-9]+\.[0-9]+\.[0-9]+)")
        set(CLANG_FORMAT_VERSION ${CMAKE_MATCH_1})
        message(STATUS "clang-format --version: " ${CLANG_FORMAT_VERSION})

        set(CLANG_FORMAT_VERSION ${CLANG_FORMAT_VERSION} CACHE STRING "clang-format version")
    endif()
endif()

if (NOT CLANG_FORMAT_VERSION)
    message(
        WARNING
        "Couldn't determine clang-format version. clang-format 9.0 is \
        required - results on other versions may not be stable")

    set(CLANG_FORMAT_VERSION "0.0.0" CACHE STRING "clang-format version not found")
elseif (NOT (CLANG_FORMAT_VERSION VERSION_GREATER_EQUAL "9.0" AND
         CLANG_FORMAT_VERSION VERSION_LESS "10.0"))
    message(
        WARNING
        "clang-format version 9.0 is required - results on other \
        versions may not be stable")
endif()

# Specifically trying to exclude external here - I'm not sure if there's a better way
set(
    GLOBS
    ${PROJECT_SOURCE_DIR}/src/[^\.]*.cpp     ${PROJECT_SOURCE_DIR}/src/[^\.]*.hpp
    ${PROJECT_SOURCE_DIR}/tst/[^\.]*.cpp     ${PROJECT_SOURCE_DIR}/tst/[^\.]*.hpp
)

file(GLOB_RECURSE FORMAT_SOURCES CONFIGURE_DEPENDS ${GLOBS})

if (CLANG_FORMAT)
  add_custom_target(clang-format-apply
    COMMAND echo "Formatting C++ files with clang-format"
    COMMAND ${CLANG_FORMAT} -style=file -i ${FORMAT_SOURCES}
    VERBATIM)
endif()
