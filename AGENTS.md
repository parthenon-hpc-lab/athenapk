# Repository Guidelines

## Project Structure & Module Organization
AthenaPK extends Parthenon and Kokkos; core solvers and physics packages live in `src/`. Problem generators sit under `src/pgen/`, while hydro drivers and plumbing live in `src/hydro/`. Build logic is centralized in `CMakeLists.txt` plus helper modules in `cmake/`. Reference input decks reside in `inputs/`, documentation and usage notes in `docs/`, and helper scripts in `scripts/`. Tests are split between `tst/regression` for Python-driven integration suites and `tst/unit` for focused checks; keep generated artifacts under a dedicated `build/` tree.

## Build, Test, and Development Commands
Configure a host build with `cmake -S . -B build-host -DKokkos_ARCH_BDW=ON -DPARTHENON_DISABLE_MPI=ON`, adjusting architecture flags as needed. Compile via `cmake --build build-host -j` or by calling `ninja` in the build directory. Run targeted tests with `ctest --output-on-failure` from `build-host`, using labels such as `ctest -L regression` or `ctest -L unit`. Launch a sample problem for sanity checks with `build-host/bin/athenaPK -i ../inputs/linear_wave3d.in`.

## Coding Style & Naming Conventions
C++ code targets C++17 and follows the repository `.clang-format` (two-space indentation, Allman braces). Python tooling is formatted with `black`. Favor expressive identifiers; keep functions camelCase, files snake_case, and constants SCREAMING_SNAKE_CASE. Public headers should remain minimal and rely on namespaces over macros; run `cmake --build build-host --target format-athenapk` before committing.

## Testing Guidelines
Regression suites live in `tst/regression/test_suites/<suite>/<suite>.py`; unit tests in `tst/unit`. Register new regression suites in `tst/regression/CMakeLists.txt` and add novel input decks under `inputs/`. Execute regression coverage with `ctest -L regression` and unit checks with `ctest -L unit`; collect logs when diagnosing failures.

## Commit & Pull Request Guidelines
Write imperative commit subjects (e.g., `Add cooling table loader`) with bodies wrapped near 72 characters. Ensure commits compile, pass formatting, and satisfy relevant `ctest` labels. Pull requests should summarize problem, solution, validation, and link issues; include plots or tables for physics changes and call out follow-up tasks.

## Submodules & Dependencies
Parthenon and Kokkos are vendored via `external/` submodules. After pulling updates, run `git submodule update --init --recursive`. When bumping versions, document the motivation and rerun full regression coverage on target hardware.
