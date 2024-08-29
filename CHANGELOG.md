# Changelog

## Current develop (i.e., `main` branch)

### Added (new features/APIs/variables/...)
- [[PR 1]](https://github.com/parthenon-hpc-lab/athenapk/pull/1) Add isotropic thermal conduction and RKL2 supertimestepping

### Changed (changing behavior/API/variables/...)
- [[PR 84]](https://github.com/parthenon-hpc-lab/athenapk/pull/84) Bump Parthenon to latest develop (2024-02-15)

### Fixed (not changing behavior/API/variables/...)
- [[PR 97]](https://github.com/parthenon-hpc-lab/athenapk/pull/97) Fixed Schure cooling curve. Removed SD one. Added description of cooling function conventions.

### Infrastructure
- [[PR 112]](https://github.com/parthenon-hpc-lab/athenapk/pull/112) Add dev container configuration
- [[PR 105]](https://github.com/parthenon-hpc-lab/athenapk/pull/105) Bump Parthenon to latest develop (2024-03-13)
- [[PR 84]](https://github.com/parthenon-hpc-lab/athenapk/pull/84) Added `CHANGELOG.md`

### Removed (removing behavior/API/varaibles/...)

### Incompatibilities (i.e. breaking changes)
- [[PR 97]](https://github.com/parthenon-hpc-lab/athenapk/pull/97)
  - Removes original `schure.cooling` cooling curve as it had unknown origin.
  - Added `updated_schure.cooling` curve (and associated notebook to calculate it from the paper tables).
- [[PR 84]](https://github.com/parthenon-hpc-lab/athenapk/pull/84) Bump Parthenon to latest develop (2024-02-15)
  - Updated access to block dimension: `pmb->block_size.nx1` -> `pmb->block_size.nx(X1DIR)` (and similarly x2 and x3)
  - Update access to mesh size: `pmesh->mesh_size.x1max` -> `pmesh->mesh_size.xmax(X1DIR)` (and similarly x2, x3, and min)
  - Updated Parthenon `GradMinMod` signature for custom prolongation ops
  - `GetBlockPointer` returns a raw pointer not a shared one (and updated interfaces to use raw pointers rather than shared ones)

