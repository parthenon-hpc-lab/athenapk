# Changelog

## Current develop (i.e., `main` branch)

### Added (new features/APIs/variables/...)

### Changed (changing behavior/API/variables/...)
- [[PR 84]](https://github.com/parthenon-hpc-lab/athenapk/pull/84) Bump Parthenon to latest develop (2024-02-15)

### Fixed (not changing behavior/API/variables/...)

### Infrastructure
- [[PR 84]](https://github.com/parthenon-hpc-lab/athenapk/pull/84) Added `CHANGELOG.md`

### Removed (removing behavior/API/varaibles/...)

### Incompatibilities (i.e. breaking changes)
- [[PR 84]](https://github.com/parthenon-hpc-lab/athenapk/pull/84) Bump Parthenon to latest develop (2024-02-15)
  - Updated access to block dimension: `pmb->block_size.nx1` -> `pmb->block_size.nx(X1DIR)` (and similarly x2 and x3)
  - Update access to mesh size: `pmesh->mesh_size.x1max` -> `pmesh->mesh_size.xmax(X1DIR)` (and similarly x2, x3, and min)
  - Updated Parthenon `GradMinMod` signature for custom prolongation ops
  - `GetBlockPointer` returns a raw pointer not a shared one (and updated interfaces to use raw pointers rather than shared ones)

