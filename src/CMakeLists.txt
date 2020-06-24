# Copyright (c) 2020, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

add_executable(
    athenaPK
        main.cpp
        eos/adiabatic_hydro.cpp
        hydro/hydro_driver.cpp
        hydro/hydro.cpp
        hydro/rsolvers/hydro_hlle.cpp
        hydro/rsolvers/hydro_hlle_kji.cpp
        recon/dc.cpp
        recon/plm.cpp
)

add_subdirectory(pgen)

target_link_libraries(athenaPK PRIVATE parthenon)
