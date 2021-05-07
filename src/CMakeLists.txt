# Copyright (c) 2020-2021, AthenaPK Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

add_executable(
    athenaPK
        main.cpp
        eos/adiabatic_glmmhd.cpp
        physical_constants.hpp
        eos/adiabatic_hydro.cpp
        hydro/hydro_driver.cpp
        hydro/hydro.cpp
        hydro/glmmhd/dedner_source.cpp
        hydro/glmmhd/cleaning_speed.cpp
        hydro/srcterms/gravitational_field.hpp
        hydro/srcterms/tabular_cooling.hpp
        hydro/srcterms/tabular_cooling.cpp
        refinement/gradient.cpp
        refinement/other.cpp
)

add_subdirectory(pgen)

target_link_libraries(athenaPK PRIVATE parthenon)
