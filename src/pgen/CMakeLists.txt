# Copyright (c) 2020, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

set(PROBLEM "LINWAVE" CACHE STRING
    "Default problem type")

set(PROBLEM_VALUES "LINWAVE;SOD"
    CACHE STRING "Possible problem types.")

set_property(CACHE PROBLEM PROPERTY STRINGS ${PROBLEM_VALUES})

message(STATUS "PROBLEM='${PROBLEM}' (problem type)")

if (${PROBLEM} STREQUAL "SOD")
    target_sources(athenaPK PRIVATE pgen/sod.cpp)
else()
    target_sources(athenaPK PRIVATE pgen/linear_wave.cpp)
endif()
