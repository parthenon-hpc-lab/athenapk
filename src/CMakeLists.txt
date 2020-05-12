# Copyright (c) 2020, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

add_executable(
    athena-parthenon
        main.cpp
        eos/adiabatic_hydro.cpp
        hydro/hydro_driver.cpp
        hydro/hydro.cpp
        hydro/rsolvers/hydro_hlle.cpp
)

add_subdirectory(pgen)

target_link_libraries(athena-parthenon PRIVATE parthenon)
