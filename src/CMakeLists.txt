# Copyright (c) 2020-2021, AthenaPK Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

add_executable(
    athenaPK
        main.cpp
        eos/adiabatic_glmmhd.cpp
        units.hpp
        eos/adiabatic_hydro.cpp
        hydro/diffusion/diffusion.hpp
        hydro/diffusion/conduction.cpp
        hydro/hydro_driver.cpp
        hydro/hydro.cpp
        hydro/glmmhd/dedner_source.cpp
        hydro/prolongation/custom_ops.hpp
        hydro/srcterms/gravitational_field.hpp
        hydro/srcterms/tabular_cooling.hpp
        hydro/srcterms/tabular_cooling.cpp
        refinement/gradient.cpp
        refinement/other.cpp
        tracers/tracers.cpp
        tracers/tracers.hpp
        utils/few_modes_ft.cpp
        utils/few_modes_ft.hpp
        utils/interpolation.hpp
)

add_subdirectory(pgen)

target_link_libraries(athenaPK PRIVATE parthenon)
