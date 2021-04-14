# Copyright (c) 2020, AthenaPK Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

add_executable(
    athenaPK
        main.cpp
        eos/adiabatic_glmmhd.cpp
        eos/adiabatic_hydro.cpp
        hydro/hydro_driver.cpp
        hydro/hydro.cpp
        refinement/gradient.cpp
        refinement/other.cpp
)

add_subdirectory(pgen)

target_link_libraries(athenaPK PRIVATE parthenon)
