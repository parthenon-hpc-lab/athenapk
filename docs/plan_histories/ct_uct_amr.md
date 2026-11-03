# Second-order constrained transport PR preparation

<!-- This file was made in part with generative AI -->

Temporary descriptive filename: rename this document to the assigned PR number
after opening the pull request, as requested by CONTRIBUTING.md.

## Intent

Add CT-contact and UCT-HLLD as second-order ideal-MHD options without replacing
GLM, using the pinned Parthenon and Kokkos submodules. Keep review focused on
staggered integration/refinement, compatible initialization, examples and tests.

## Implementation

- Add CT EOS/runtime dispatch, shared face-normal Riemann states, edge EMFs,
  staggered RK updates and pre-primitive magnetic centering.
- Preserve GLM paths and legacy input decks. Separate CT examples default to
  UCT-HLLD/PLM/HLLD/RK2 with two ghost zones.
- Use the internal AthenaPK UpdateWithFluxEmfCurl. Its 1D RHS is zero for Bx;
  transverse cell magnetic fields remain evolved. Do not depend on experimental
  Parthenon curl utilities.
- Register shared-face minmod prolongation, average restriction and Toth--Roe
  internal prolongation; correct EMFs before the curl update.
- Match midpoint coarse-edge potentials to two fine-edge midpoint samples in
  CPAW, linear waves and rotated field loop, using a shared pgen-only neighbor
  matcher. Keep this separate from runtime remeshing.
- Preserve optional nonconservative magnetic-energy correction and diagnostic
  names/normalizations. Reject unsupported CT/VL2, flux correction, resistivity,
  RKL2 and field-loop configurations.
- Provide standalone uniform/SMR/AMR decks
  and a portable singleton-GPU convergence script plus its Python companion.
- Add feature/input/pgen documentation, changelog entries and AI disclosure.
- Author native CT regression definitions covering dimensions, convergence,
  layouts, refinement, shock-problem smoke cases and expected configuration
  failures. Do not execute them during preparation.

## User validation gate

The submitter rebuilt with the pinned submodules and completed fresh CT, Euler
and selected GLM regression suites on 2026-11-03. All nine automated summaries
report PASSED; configurations, measured checks and runtimes are recorded in
[the validation guide](../ct_validation.md). The optional CPU serial CPAW restart
exercise passed: final time and hierarchy agree, and conserved, primitive and
physical face-field values match bitwise after matching logical block locations.
Other restart and standalone OT refinement exercises
are optional and not reported as performed. Formatting/lint and final review
remain outstanding.

## Repository boundaries

Changes are prepared in a dedicated PR worktree. Original development results and Parthenon experiments are not touched. No staging, commits, remote
Git operations, job submission or GitHub PR creation are performed.

## Validation-reporting and wave-routing follow-up

- Preserve Euler initialization under the explicit linear_wave_hydro selector and
  _hydro source/deck filenames; preserve hydro parameter keys and numerical formulas.
- Route GLM convergence/diffusion and GLM performance cases through linear_wave_mhd,
  using canonical MHD parameters and named magnetic error/history columns.
- Write validation_summary.txt during fresh native runs, separating acceptance gates
  from informational measurements and leaving killed/timed-out runs incomplete.
- Add step-by-step validation documentation, historical timing evidence, and the
  reference-machine GPU MPI transport settings; assign four-hour suite deadlines.
- Remove the unvalidated contributed example and obsolete implementation-history
  discussion. Keep unrelated upstream problems and CT numerical gates unchanged.
- The submitter reran the automated suites after this update. No build or test
  execution was performed by the agent while authoring the changes.

## Validation-record and build-guide follow-up

- Record the fresh automated passes and their actual case counts and wall times;
  distinguish the optional CPAW restart exercise from strict equivalence validation.
- Label AMR restart and standalone OT refinement exercises as optional extensions.
- Select the parallel HDF5 prefix explicitly for CPU MPI and GPU builds, including
  recovery from cached serial Conda HDF5 discovery without changing other build flags.
- Stop before formatting/lint execution; the submitter receives instructions for
  those checks. No numerical code, result files, Git history or submodules change.

## Formatting and targeted lint review

- Verify clang-format 20 and Black on the contributed/modified files; both pass.
- Add repository license notices to the contributed headers, retain Athena++
  attribution for the adapted edge checks, supply direct standard-library includes,
  and remove contributed redundant blank lines. Preserve numerical expressions
  and task ordering.
- Whole-file cpplint remains nonzero: 237 findings concern existing conventions,
  inherited lines, or an already-braced `if constexpr` chain. Record the category
  breakdown in the local PR checklist; do not suppress findings or claim a pass.
- Leave compilation, regression execution, final review and submission to the user.
