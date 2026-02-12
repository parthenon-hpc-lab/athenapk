# Changelog

## Current develop (i.e., `main` branch)

### General notes

Particle ids have been updated in Parthenon to be `uint64` and added by default.
Thus, the original "`id`" tracer variable has been removed in favor of the Parthenon default version.
Access to that `id` is done via `auto &id = swarm->Get<std::uint64_t>(swarm_position::id::name()).Get();`
and via `swarm.id` in the `phdf` python tools.
Other postprocessing tools, like VisIt will automatically identify the new field.

From updated Parthenon submodule:

- `packs_per_rank` can now be used instead of `pack_size` in the `<parthenon/mesh>` input block.
It is the new default (i.e., it's set automatically when it's not present in the input file)
because it result in better load balance.
- For simulation on (AMD) GPUs with AMR and many blocks per rank, the number of MPI messages in flight
(especially during mesh refinement) could sometimes cause "Memory access fault by GPU".
This is related to how the MPI library and hardware manage handles to communication buffer in device memory.
To circumvent this issue, Parthenon now supports [coalesced communication](https://parthenon-hpc-lab.github.io/parthenon/develop/src/boundary_communication.html#coalesced-mpi-communication)
where multiple messages between ranks are combined.
This comes at a small performance cost (due to the additional packing and unpacking of messages).
To enable, set `do_coalesced_comms=true` in the `<parthenon/mesh>` block of the input file.
- Input parameters can now be [automatically documented](https://github.com/parthenon-hpc-lab/parthenon/pull/1283)
by adding an optional string as last argument to any `ParameterInput` `Get` or `GetOrAdd` call.

### Added (new features/APIs/variables/...)
- [[PR 158]](https://github.com/parthenon-hpc-lab/athenapk/pull/158) Update particle id handling (now automated `uint64`). Extend particle history lookback in turbulence pgen and include in turbulence test
- [[PR 157]](https://github.com/parthenon-hpc-lab/athenapk/pull/157) Support injection of blobs with density/temp contrast in turbulence simulations

### Changed (changing behavior/API/variables/...)

### Fixed (not changing behavior/API/variables/...)

### Infrastructure
- [[PR 149]](https://github.com/parthenon-hpc-lab/athenapk/pull/149) Allow triggering of pipelines manually
- [[PR 156]](https://github.com/parthenon-hpc-lab/athenapk/pull/156) Bump formatters to clang-format-20 and black 25.12
- [[PR 146]](https://github.com/parthenon-hpc-lab/athenapk/pull/146) Bump Parthenon 25.12 and Kokkos 4.7.02

### Removed (removing behavior/API/varaibles/...)

### Incompatibilities (i.e. breaking changes)
- [[PR 146]](https://github.com/parthenon-hpc-lab/athenapk/pull/146) `pmesh->is_restart` removed. Use `arthenon::Globals::is_restart` instead.


## Release 25.05

### IMPORTANT

If you pulled from `main` after 11 Nov 24 ([[PR 124]](https://github.com/parthenon-hpc-lab/athenapk/pull/124))
please updated immediate to a version after 18 Mar 24 ([[PR 136]](https://github.com/parthenon-hpc-lab/athenapk/pull/136)).
In between a subtle bug was introduced that resulted in inconsistent divergence cleaning speeds in MHD simulation with mesh
refinement.

### Added (new features/APIs/variables/...)
- [[PR 140]](https://github.com/parthenon-hpc-lab/athenapk/pull/140) Add hydro reflecting boundary conditions
- [[PR 102]](https://github.com/parthenon-hpc-lab/athenapk/pull/102) Add support for tracer particles
- [[PR 89]](https://github.com/parthenon-hpc-lab/athenapk/pull/89) Add viscosity and resistivity
- [[PR 1]](https://github.com/parthenon-hpc-lab/athenapk/pull/1) Add isotropic thermal conduction and RKL2 supertimestepping

### Changed (changing behavior/API/variables/...)
- [[PR 122]](https://github.com/parthenon-hpc-lab/athenapk/pull/122) Fixed sqrt(4pi) factor in CGS Gauss unit and add unit doc
- [[PR 119]](https://github.com/parthenon-hpc-lab/athenapk/pull/119) Fixed Athena++ paper test case for KHI pgen. Added turbulence pgen doc.
- [[PR 97]](https://github.com/parthenon-hpc-lab/athenapk/pull/97) Fixed Schure cooling curve. Removed SD one. Added description of cooling function conventions.
- [[PR 84]](https://github.com/parthenon-hpc-lab/athenapk/pull/84) Bump Parthenon to latest develop (2024-02-15)

### Fixed (not changing behavior/API/variables/...)
- [[PR 136]](https://github.com/parthenon-hpc-lab/athenapk/pull/136) Fix using MPI reduced mindx
- [[PR 128]](https://github.com/parthenon-hpc-lab/athenapk/pull/128) Fixed `dt_diff` in RKL2

### Infrastructure
- [[PR 150]](https://github.com/parthenon-hpc-lab/athenapk/pull/150) Introduce CalVer and add CONTRIBUTING.md
- [[PR 142]](https://github.com/parthenon-hpc-lab/athenapk/pull/142) Bump Kokkos 4.6.1 and Parthenon 25.05
- [[PR 136]](https://github.com/parthenon-hpc-lab/athenapk/pull/136) Bump Kokkos 4.5.1 (for support of AMD APUs)
- [[PR 129]](https://github.com/parthenon-hpc-lab/athenapk/pull/129) Bump Parthenon to support `dn` based outputs
- [[PR 124]](https://github.com/parthenon-hpc-lab/athenapk/pull/124) Bump Kokkos 4.4.1 (and Parthenon to include view-of-view fix)
- [[PR 117]](https://github.com/parthenon-hpc-lab/athenapk/pull/117) Update devcontainer.json to latest CI container
- [[PR 114]](https://github.com/parthenon-hpc-lab/athenapk/pull/114) Bump Parthenon 24.08 and Kokkos to 4.4.00
- [[PR 112]](https://github.com/parthenon-hpc-lab/athenapk/pull/112) Add dev container configuration
- [[PR 105]](https://github.com/parthenon-hpc-lab/athenapk/pull/105) Bump Parthenon to latest develop (2024-03-13)
- [[PR 84]](https://github.com/parthenon-hpc-lab/athenapk/pull/84) Added `CHANGELOG.md`

### Removed (removing behavior/API/varaibles/...)

### Incompatibilities (i.e. breaking changes)
- [[PR 142]](https://github.com/parthenon-hpc-lab/athenapk/pull/142) Removed `coords...FA<>()` interface in Parthenon
- [[PR 124]](https://github.com/parthenon-hpc-lab/athenapk/pull/124) Enrolling custom boundary conditions changed
  - Boundary conditions can now be enrolled using a string that can be subsequently be used in the input file (see, e.g., cloud problem generator)
- [[PR 114]](https://github.com/parthenon-hpc-lab/athenapk/pull/114) Bump Parthenon 24.08 and Kokkos to 4.4.00
  - Changed signature of `UserWorkBeforeOutput` to include `SimTime` as last paramter
  - Fixes bitwise idential restarts for AMR simulations (the derefinement counter is now included)
  - Order of operations in flux-correction has changed (expect round-off error differences to previous results for AMR sims)
  - History outputs now carry the output block number, i.e., a file previously called parthenon.hst might now be called parthenon.out1.hst
  - History outputs now contain two additional columns (cycle number and meshblock counts), which changes/shifts the column indices (hint: use the column headers to parse the contents and do not rely on fixed indices as they may also vary between different pgen due to custom/pgen-dependent content in the history file)
  - Given the introduction of a forest of tree (rather than a single tree), the logical locations are each meshblock (`pmb->loc`) are now local to the tree and not global any more. To recover the original global index use `auto loc = pmb->pmy_mesh->Forest().GetLegacyTreeLocation(pmb->loc);`
- [[PR 97]](https://github.com/parthenon-hpc-lab/athenapk/pull/97)
  - Removes original `schure.cooling` cooling curve as it had unknown origin.
  - To avoid confusion, only cooling table for a single solar metallicity are supported
    from now on (i.e., the parameters to specify temperature and lambda columns have been removed).
  - Added `schure.cooling_#Z` curves (and associated notebook to calculate it from the paper tables).
- [[PR 84]](https://github.com/parthenon-hpc-lab/athenapk/pull/84) Bump Parthenon to latest develop (2024-02-15)
  - Updated access to block dimension: `pmb->block_size.nx1` -> `pmb->block_size.nx(X1DIR)` (and similarly x2 and x3)
  - Update access to mesh size: `pmesh->mesh_size.x1max` -> `pmesh->mesh_size.xmax(X1DIR)` (and similarly x2, x3, and min)
  - Updated Parthenon `GradMinMod` signature for custom prolongation ops
  - `GetBlockPointer` returns a raw pointer not a shared one (and updated interfaces to use raw pointers rather than shared ones)

