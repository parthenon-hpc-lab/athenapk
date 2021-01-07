# Copyright (c) 2020, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

if (${PROBLEM} STREQUAL "SOD")
    target_sources(athenaPK PRIVATE sod.cpp)
elseif (${PROBLEM} STREQUAL "BLAST")
    target_sources(athenaPK PRIVATE blast.cpp)
else()
    target_sources(athenaPK PRIVATE linear_wave.cpp)
endif()
