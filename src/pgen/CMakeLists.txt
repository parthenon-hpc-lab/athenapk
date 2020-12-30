# Copyright (c) 2020, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

set(PROBLEM "LINWAVE" CACHE STRING
    "Default problem type")

set(PROBLEM_VALUES "LINWAVE;SOD;BLAST"
    CACHE STRING "Possible problem types.")

set_property(CACHE PROBLEM PROPERTY STRINGS ${PROBLEM_VALUES})

message(STATUS "PROBLEM='${PROBLEM}' (problem type)")

if (${PROBLEM} STREQUAL "SOD")
    target_sources(athenaPK PRIVATE sod.cpp)
elseif (${PROBLEM} STREQUAL "BLAST")
    target_sources(athenaPK PRIVATE blast.cpp)
else()
    target_sources(athenaPK PRIVATE linear_wave.cpp)
endif()
