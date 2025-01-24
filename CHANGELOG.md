# Changelog

## Current develop (i.e., `main` branch)

### Added (new features/APIs/variables/...)
- [[PR 89]](https://github.com/parthenon-hpc-lab/athenapk/pull/89) Add viscosity and resistivity
- [[PR 1]](https://github.com/parthenon-hpc-lab/athenapk/pull/1) Add isotropic thermal conduction and RKL2 supertimestepping

### Changed (changing behavior/API/variables/...)
- [[PR 122]](https://github.com/parthenon-hpc-lab/athenapk/pull/122) Fixed sqrt(4pi) factor in CGS Gauss unit and add unit doc
- [[PR 119]](https://github.com/parthenon-hpc-lab/athenapk/pull/119) Fixed Athena++ paper test case for KHI pgen. Added turbulence pgen doc.
- [[PR 97]](https://github.com/parthenon-hpc-lab/athenapk/pull/97) Fixed Schure cooling curve. Removed SD one. Added description of cooling function conventions.
- [[PR 84]](https://github.com/parthenon-hpc-lab/athenapk/pull/84) Bump Parthenon to latest develop (2024-02-15)

### Fixed (not changing behavior/API/variables/...)
- [[PR 128]](https://github.com/parthenon-hpc-lab/athenapk/pull/128) Fixed `dt_diff` in RKL2

### Infrastructure
- [[PR 128]](https://github.com/parthenon-hpc-lab/athenapk/pull/128) Bump Parthenon to support `dn` based outputs
- [[PR 124]](https://github.com/parthenon-hpc-lab/athenapk/pull/124) Bump Kokkos 4.4.1 (and Parthenon to include view-of-view fix)
- [[PR 117]](https://github.com/parthenon-hpc-lab/athenapk/pull/117) Update devcontainer.json to latest CI container
- [[PR 114]](https://github.com/parthenon-hpc-lab/athenapk/pull/114) Bump Parthenon 24.08 and Kokkos to 4.4.00
- [[PR 112]](https://github.com/parthenon-hpc-lab/athenapk/pull/112) Add dev container configuration
- [[PR 105]](https://github.com/parthenon-hpc-lab/athenapk/pull/105) Bump Parthenon to latest develop (2024-03-13)
- [[PR 84]](https://github.com/parthenon-hpc-lab/athenapk/pull/84) Added `CHANGELOG.md`

### Removed (removing behavior/API/varaibles/...)

### Incompatibilities (i.e. breaking changes)
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

