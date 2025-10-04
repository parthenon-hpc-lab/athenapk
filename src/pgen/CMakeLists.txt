# Copyright (c) 2020-2021, Athena Parthenon Collaboration. All rights reserved.
# Licensed under the 3-Clause License (the "LICENSE");

# © 2024. Triad National Security, LLC. All rights reserved.
#
# This program was produced under U.S. Government contract 89233218CNA000001 for Los
# Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC
# for the U.S. Department of Energy/National Nuclear Security Administration. All rights
# in the program are reserved by Triad National Security, LLC, and the U.S. Department of
# Energy/National Nuclear Security Administration. The Government is granted for itself
# and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license
# in this material to reproduce, prepare. derivative works, distribute copies to the
# public, perform publicly and display publicly, and to permit others to do so.

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
    lw_implode.cpp
    orszag_tang.cpp
    precipitator.cpp
    precipitator_srcterms.cpp
    rand_blast.cpp
    sod.cpp
    turbulence.cpp
    disk.cpp
    )
