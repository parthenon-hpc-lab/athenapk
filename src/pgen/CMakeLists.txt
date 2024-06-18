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
    cpaw.cpp
    diffusion.cpp
    field_loop.cpp
    kh.cpp
    linear_wave.cpp
    linear_wave_mhd.cpp
    orszag_tang.cpp
    precipitator.cpp
    rand_blast.cpp
    sod.cpp
    turbulence.cpp
    )
