# This file was made in part with generative AI
# Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

target_sources(athenaPK PRIVATE
    advection.cpp
    blast.cpp
    cloud.cpp
    cluster.cpp
    cluster/agn_feedback.cpp
    cluster/agn_triggering.cpp
    cluster/cluster_clips.cpp
    cluster/cluster_reductions.cpp
    cluster/hydrostatic_equilibrium_sphere.cpp
    cluster/magnetic_tower.cpp
    cluster/snia_feedback.cpp
    cluster/stellar_feedback.cpp
    ct_glm_compatible/current_sheet.cpp
    ct_glm_compatible/field_loop.cpp
    ct_glm_compatible/linear_wave_mhd.cpp
    ct_glm_compatible/orszag_tang.cpp
    ct_glm_compatible/cpaw.cpp
    diffusion.cpp
    kh.cpp
    linear_wave_hydro.cpp
    lw_implode.cpp
    rand_blast.cpp
    sod.cpp
    shattering.cpp
    turbulence.cpp
    )
