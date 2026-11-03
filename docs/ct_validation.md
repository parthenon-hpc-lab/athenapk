# Building and validating constrained transport

<!-- This file was made in part with generative AI -->

This guide covers CT-contact (`ctmhd`), UCT-HLLD (`ucthlldmhd`), the Euler wave
problem, and selected GLM compatibility tests. For the equations, initialization,
supported configurations and limitations, see [constrained transport](constrained_transport.md).
Commands are run from the repository root unless indicated otherwise.

Fresh runs of the reporting and corrected GLM routing passed on the reference
machine on 2026-11-03; the recorded configurations and measurements are below.
Other checkouts and environments still need their own validation. Review the
summaries and report failures rather than changing tolerances to obtain a pass.

## 1. Prepare dependencies and the Python environment

Use the submodule commits recorded by this checkout:

```bash
git submodule update --init --recursive
git submodule status
git diff --submodule -- external
```

Do not use `--remote` or substitute a separately modified Parthenon checkout.
The last command should show no dependency changes. If initialization requires
network access, arrange that before starting a compute allocation.

Activate a Python environment containing the regression dependencies:

```bash
python -c 'import numpy, unyt, matplotlib, h5py, scipy'
```

Choose matching compiler, MPI and HDF5 installations. The MPI build requires
parallel HDF5; the GPU build additionally requires CUDA, a compatible host compiler,
and a CUDA-aware MPI communication path. Loading CUDA alone does not make an
existing MPI/UCX installation CUDA-aware.

### MSU HPCC reference environment (not portable defaults)

These modules were used during development. They are examples for that machine,
not dependencies hard-coded into AthenaPK. Other systems may use different modules,
compilers, GPU architectures, launchers and HDF5 locations.

```bash
conda activate mhd
module purge
module load foss/2023a
module load HDF5/1.14.0-gompi-2023a
export LD_LIBRARY_PATH="$EBROOTHDF5/lib:${LD_LIBRARY_PATH:-}"
export CT_HDF5_ROOT="$EBROOTHDF5"
```

For GPU compilation and execution, additionally load:

```bash
module load CUDA/12.3.0
export NVCC_WRAPPER_DEFAULT_COMPILER="$(command -v g++)"
```

`mhd` is the development environment's name; substitute your own. The MPI build
commands below use `CT_HDF5_ROOT` to select the **parallel** HDF5 installation.
The reference module block sets it automatically. On other systems, set it to the
installation prefix before configuring, for example
`export CT_HDF5_ROOT=/path/to/parallel-hdf5`. Reload the runtime libraries in each
terminal or batch allocation used for tests.

The reference Conda environment also contains a serial HDF5 installation. Without
explicit selection, CMake found its HDF5 2.1.0 package rather than the loaded
parallel HDF5 1.14.0 module, and correctly rejected the MPI build. Both MPI build
commands therefore set `HDF5_ROOT`, prefer parallel wrapper compilers and disable
package-config discovery. These options select the intended library; they do not
disable HDF5 output or MPI.

## 2. Configure separate builds

Never reuse one build directory for different compilers or CPU/CUDA backends.
Testing is enabled explicitly and Python is selected explicitly to avoid finding
a different interpreter without the required modules.
After this update, reconfigure and rebuild each build before CTest: the renamed
hydro sources/decks and corrected MHD registrations must not be used with a stale
executable or cached test configuration.

### Serial CPU build

```bash
cmake -S . -B build-host \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DAthenaPK_ENABLE_TESTING=ON \
  -DPARTHENON_DISABLE_MPI=ON \
  -DPARTHENON_DISABLE_HDF5=OFF \
  -DPARTHENON_DISABLE_OPENPMD=ON \
  -DPARTHENON_DISABLE_OPENMP=ON \
  -DPARTHENON_DISABLE_SPARSE=ON
cmake --build build-host -j8
```

### CPU MPI build

```bash
: "${CT_HDF5_ROOT:?Set CT_HDF5_ROOT to your parallel HDF5 prefix}"
cmake -S . -B build-mpi \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="$(command -v mpicc)" \
  -DCMAKE_CXX_COMPILER="$(command -v mpicxx)" \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DTEST_MPIEXEC="$(command -v mpirun)" \
  -DTEST_NUMPROC_FLAG=-n \
  -DHDF5_ROOT="$CT_HDF5_ROOT" \
  -DHDF5_PREFER_PARALLEL=ON \
  -DHDF5_NO_FIND_PACKAGE_CONFIG_FILE=ON \
  -DAthenaPK_ENABLE_TESTING=ON \
  -DPARTHENON_DISABLE_MPI=OFF \
  -DPARTHENON_DISABLE_HDF5=OFF \
  -DPARTHENON_DISABLE_OPENPMD=ON \
  -DPARTHENON_DISABLE_OPENMP=ON \
  -DPARTHENON_DISABLE_SPARSE=ON
cmake --build build-mpi -j8
```

The launcher must match your MPI installation and scheduler allocation. Replace
`TEST_MPIEXEC` and `TEST_NUMPROC_FLAG` as required by your site; CTest itself must
not be prefixed with `mpirun -n 2`.

### CUDA V100 build

Use this only with a compatible CUDA environment. Replace `VOLTA70` for a
different GPU architecture.

```bash
: "${CT_HDF5_ROOT:?Set CT_HDF5_ROOT to your parallel HDF5 prefix}"
cmake -S . -B build-gpu-v100 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$PWD/external/Kokkos/bin/nvcc_wrapper" \
  -DCMAKE_C_COMPILER="$(command -v mpicc)" \
  -DMPI_CXX_COMPILER="$(command -v mpicxx)" \
  -DHDF5_ROOT="$CT_HDF5_ROOT" \
  -DHDF5_PREFER_PARALLEL=ON \
  -DHDF5_NO_FIND_PACKAGE_CONFIG_FILE=ON \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DTEST_MPIEXEC="$(command -v mpirun)" \
  -DTEST_NUMPROC_FLAG=-n \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_VOLTA70=ON \
  -DKokkos_ARCH_NATIVE=OFF \
  -DAthenaPK_ENABLE_TESTING=ON \
  -DPARTHENON_DISABLE_MPI=OFF \
  -DPARTHENON_DISABLE_HDF5=OFF \
  -DPARTHENON_DISABLE_OPENPMD=ON \
  -DPARTHENON_DISABLE_OPENMP=ON \
  -DPARTHENON_DISABLE_SPARSE=ON
cmake --build build-gpu-v100 -j8
```

Compilation does not reserve a GPU for execution. Obtain the appropriate resources
before running tests. On a shared development node, follow its usage policy.

### Recover from a cached serial-HDF5 selection

If configuration already found the wrong HDF5, clear only its cached discovery
settings and select the parallel installation again:

```bash
: "${CT_HDF5_ROOT:?Set CT_HDF5_ROOT to your parallel HDF5 prefix}"
cmake -S . -B build-mpi \
  -U 'HDF5_*' -U 'hdf5_*' \
  -DHDF5_ROOT="$CT_HDF5_ROOT" \
  -DHDF5_PREFER_PARALLEL=ON \
  -DHDF5_NO_FIND_PACKAGE_CONFIG_FILE=ON
```

For the GPU build, repeat with `-B build-gpu-v100`; its cached CUDA/compiler flags
are retained. Check that configuration succeeds before invoking the build. A
subsequent "No Makefile" error is a consequence of failed configuration, not a
separate compiler error. Do not disable MPI or HDF5 to hide the discovery problem.

## 3. Understand and run the native tests

CMake registers named CTest entries using Parthenon's serial/MPI helper functions.
CTest launches Parthenon's Python regression runner. AthenaPK's suite definitions
then prepare simulation arguments, launch the executable, and analyze its output.

`ct_mhd` is one CTest entry containing 56 simulations, not 56 independent CTest
entries. `regression_test:` is a naming convention; `-R` selects test names by
regular expression. `^` and `$` anchor an exact name. `-N` lists without running.

```bash
ctest --test-dir build-host -N
ctest --test-dir build-gpu-v100 -N -R 'ct_mhd'
```

### CPU CT validation

```bash
ctest --test-dir build-host \
  -R '^regression_test:ct_invalid_config$' --output-on-failure
ctest --test-dir build-host \
  -R '^regression_test:ct_mhd$' --timeout 14400 --output-on-failure
ctest --test-dir build-mpi \
  -R '^regression_mpi_test:ct_mhd$' --timeout 14400 --output-on-failure
```

The invalid-configuration test expects 14 rejected executions and verifies the
diagnostic messages. CTest should report this suite as passed, not failed.

The MPI registration launches two ranks. Reserve at least two CPU slots. A
"not enough slots" failure occurs before numerical validation. On a permitted
development-node run, two hardware threads may be counted as one physical core
after CPU affinity is applied. For correctness testing only, an invocation can use:

```bash
env OMPI_MCA_rmaps_base_oversubscribe=1 \
    OMPI_MCA_hwloc_base_binding_policy=none \
    ctest --test-dir build-mpi -R '^regression_mpi_test:ct_mhd$' \
      --timeout 14400 --output-on-failure
```

Oversubscription does not grant additional resources and is not a performance
benchmark. Prefer a correctly sized allocation.

### GPU CT validation and reference-machine MPI settings

Reserve one GPU for the serial entry or two for the MPI entry. The GPU MPI
registration supplies `--kokkos-map-device-id-by=mpi_rank`, so each rank selects a
different visible device. Preserve the scheduler's `CUDA_VISIBLE_DEVICES`; setting
that variable yourself does not reserve devices.

On our reference system, the selected UCX installation did not expose CUDA memory
support, although Open MPI was CUDA-aware. Bypassing UCX avoided a GPU-buffer
segmentation fault. The two-GPU 3D AMR initialization also required disabling
CUDA IPC; an isolated IPC-on/off comparison reproduced and avoided the failure.
These are site/runtime requirements, not changes to the CT scheme:

```bash
env OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,smcuda,tcp \
    OMPI_MCA_btl_smcuda_use_cuda_ipc=0 \
    ctest --test-dir build-gpu-v100 -R '^regression_test:ct_mhd$' \
      --timeout 14400 --output-on-failure

env OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,smcuda,tcp \
    OMPI_MCA_btl_smcuda_use_cuda_ipc=0 \
    ctest --test-dir build-gpu-v100 -R '^regression_mpi_test:ct_mhd$' \
      --timeout 14400 --output-on-failure
```

Other installations may work with CUDA-aware UCX or different transports; do not
blindly apply these settings everywhere. See [Open MPI CUDA support](https://www.open-mpi.org/faq/?category=runcuda).

### Euler preservation and GLM compatibility

The original hydro wave implementation is now `linear_wave_hydro`, using
`inputs/linear_wave1d_hydro.in` and `inputs/linear_wave3d_hydro.in`. Its existing
`problem/linear_wave` parameter keys, hydro mode numbering and error-file format
are retained. Older decks need the renamed `job/problem_id`; no MHD eigenvectors
are substituted for Euler waves.

```bash
ctest --test-dir build-host -R '^regression_test:convergence$' \
  --timeout 14400 --output-on-failure
ctest --test-dir build-host \
  -R '^regression_test:(field_loop|mhd_convergence|diffusion_linwave3d)$' \
  --timeout 14400 --output-on-failure
```

The corrected GLM magnetic-wave suite contains 48 cases, uses the CT-compatible
MHD pgen and includes magnetic-component errors. It is not the former hydro wave
run with a GLM fluid selection. The diffusion test explicitly uses MHD slow mode 2
and `test=false`, so its final time 3 is code time rather than three periods.

For the preferred two-GPU GLM validation:

```bash
env OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,smcuda,tcp \
    OMPI_MCA_btl_smcuda_use_cuda_ipc=0 \
    ctest --test-dir build-gpu-v100 \
      -R '^regression_mpi_test:(field_loop|mhd_convergence|diffusion_linwave3d)$' \
      --timeout 14400 --output-on-failure
```

These are three suites run sequentially, each with its own four-hour deadline.
Do not run multiple CTest invocations concurrently in one build directory: their
top-level logs overlap, and rerunning a suite replaces its previous output tree.
Different build directories have isolated results, but still compete for hardware.

## 4. Read results and interpret acceptance

Each of the six validation suites writes a persistent `validation_summary.txt`:

```text
build-host/tst/regression/outputs/ct_mhd/validation_summary.txt
build-mpi/tst/regression/outputs/ct_mhd_mpi/validation_summary.txt
build-gpu-v100/tst/regression/outputs/mhd_convergence_mpi/validation_summary.txt
```

The same pattern applies to `ct_invalid_config`, `convergence`, `field_loop` and
`diffusion_linwave3d`. Read a report with, for example:

```bash
less build-gpu-v100/tst/regression/outputs/ct_mhd_mpi/validation_summary.txt
```

Reports start **INCOMPLETE** before the driver runs and finish **PASSED** only after
suite analysis succeeds. A driver or analysis failure is **FAILED**. A timeout,
kill, or unhandled runner failure can leave the last **INCOMPLETE** checkpoint;
it is never a pass. Elapsed time in an incomplete report is the time at the last
write. `PASS`/`FAIL` lines identify gates and thresholds; `INFO` lines are measured
diagnostics without pass/fail thresholds. Files are written during fresh runs;
there is no separate report-regeneration command for older results.

CTest's overall log is `build-*/Testing/Temporary/LastTest.log`, or its `.tmp` file
while active. A runner that captures subprocess output may not update the log
until a simulation returns. Subsequent CTest invocations replace that log. The
suite summaries remain until their output directories are replaced by a rerun.
Use `-V` for verbose CTest output when diagnosing a failed suite.

### What the checks establish

- CT smooth waves: volume-normalized component L1 errors and RMS-L1 (the norm of
  component L1 errors, not a pointwise RMS). Require decreasing errors and finest
  order >1.5. Report component orders too; zero-error components have undefined
  orders. Inspect approach to second order rather than treating 1.5 as the claim.
- CT layout checks: RMS-L1 difference divided by the larger error <5%, not bitwise
  identity. The minimal layout is one block serial and two blocks on two ranks.
- CT divergence: calculate the active-dimension discrete divergence directly from
  raw PHDF face fields, not from a cell-centered visualization. Require
  `max(dx1)*max(abs(div_face(B)))/B_ref < 2e-11`, using `B_ref=1e-3` for field loops
  and 1 otherwise. Reports include absolute/normalized initial and worst saved
  values, times and coordinates. History definitions are documented separately
  in the feature guide and must not be interchanged with this normalization.
- Coarse/fine initialization: compare each coarse face-cell magnetic flux to the
  sum over fine subfaces. Area coverage uses rtol `2e-12`, atol `1e-15`; normalized
  flux mismatch must be <`2e-11`. The summary states the scale and absolute mismatch.
- Divergence preservation: report the increase in domain-maximum absolute
  divergence from initialization. This is informational and is not a pointwise
  difference between cells on changing AMR hierarchies. Only saved times are checked.
- CT conservation: drift `max(abs(Q(t)-Q(0)))/max(abs(Q(0)),1) < 2e-11` for periodic
  mass/energy cases. With energy correction enabled, total-energy drift is reported
  but not required to vanish. Positivity and mass conservation remain gates.
- CT 1D, SMR/AMR and smoke cases: require constant normal B, changing transverse B,
  exercised coarse/fine interfaces, startup refinement, remeshing during advection,
  and finite positive density/pressure, as applicable to each group.
- Euler convergence: retain the legacy VL2/PLM N64 RMS-L1 limit `1.547584e-8`
  and report component errors and observed orders.
- GLM waves: require finite component errors, decreasing conserved and magnetic
  aggregate errors, finest order >0.75 for RK1/DC and >1.5 otherwise. These are
  convergence gates, not a new absolute-error calibration or a higher-order claim.
- GLM field loop: require finite histories/primitive states, positive density,
  pressure and magnetic energy. Energy retention and GLM `relDivB` are informational;
  no CT roundoff-divergence tolerance is imposed.
- GLM diffusion: fit the positive `MaxAbsV2` history to the existing analytic slow-
  mode decay model. Preserve relative rate-error limits 0.22 at N16 and 0.05 at N32.

### Recorded validation: 2026-11-03

The submitter rebuilt and reran the updated suites, including the persistent
reporters and corrected GLM wave routing. All nine saved summaries contain the
expected case counts and report **PASSED**, consistent with CTest's recorded
results. CPU runs used HPCC development/compute CPUs; GPU runs used V100-class
GPUs, Release builds and the module stack above. These are not a controlled
scaling comparison. CPU affinity, oversubscription, filesystem load, AMR block
counts and available GPU resources affect runtime.

| Suite and configuration | Cases | CTest wall time | Outcome |
|---|---:|---:|---|
| CT, CPU serial (latest rerun) | 56 | 42m25s | Passed |
| CT, CPU two-rank MPI | 56 | 1h10m51s | Passed |
| CT, one GPU | 56 | 14m07s | Passed |
| CT, two GPUs | 56 | 23m04s | Passed |
| Unsupported configurations, CPU serial | 14 | 1.43s | Passed |
| Euler convergence, one GPU | 40 | 2m02s | Passed |
| GLM magnetic-wave convergence, two GPUs | 48 | 3m52s | Passed |
| GLM slow-wave diffusion, two GPUs | 2 | 6.67s | Passed |
| GLM field loop, two GPUs | 12 | 1m12s | Passed |

The GPU runs used the `ob1` and `self,smcuda,tcp` transport configuration with
CUDA IPC disabled, as shown above. These outcomes validate the selected suites
and configurations, not every possible GLM or CT configuration.

For example, UCT-HLLD CPAW RMS-L1 errors at N16,32,64 were approximately
`7.022079e-3, 1.960908e-3, 4.983429e-4` in 1D, giving orders `1.8404, 1.9763`;
the 3D errors were `1.762021e-2, 5.382115e-3, 1.571640e-3`, giving `1.7110, 1.7759`.
CPU/GPU reports agreed to the printed error precision. This does not imply
bitwise-identical fields or fully asymptotic multidimensional convergence.
Across the CT reports, the largest normalized face divergence was approximately
`1.16e-13`, the largest normalized conservation drift approximately `3.00e-12`,
and the largest normalized coarse/fine magnetic-flux mismatch approximately
`6.58e-15`, all below the respective `2e-11` gates defined above.

The corrected GLM magnetic-wave checks passed for both conserved and magnetic
norms. Finest-interval orders were about 0.90 for RK1/DC and 1.81--2.35 across
the remaining measured norms. GLM diffusion rate errors were 21.42% at N16
(limit 22%) and 3.04% at N32 (limit 5%). These are observed results, not newly
calibrated tolerances.

The GLM field-loop pass is a finite/positive-state check, not a divergence or
energy-retention accuracy certificate. Its reported locally normalized
`relDivB` can become very large near magnetic nulls; the largest recorded value
was about `3.13e11` in the finest RK1/DC case. Do not interpret this number as
absolute divergence or compare it directly with CT's fixed-reference gate.

The original 1500-second CTest default interrupted the CPU CT suite. These six
validation suites now register a 14400-second timeout; the examples also specify
it explicitly. Parthenon's default 300-second task-collection timeout is a
different safeguard: a stalled boundary exchange is not repaired by increasing
CTest's deadline. Investigate transport, synchronization or communication errors.
PMIx cleanup warnings were also observed; they should not be confused with a
successful numerical analysis or ignored if accompanied by an actual failure.

## 5. Optional extended manual checks

The automated suites above are the documented regression validation. The following
manual exercises extend that coverage; they are optional and are not prerequisites
for recording those automated suites as passed. Do not mark an unperformed
optional exercise as passed, and disclose discrepancies in any exercise you run.
The mixed hydro/GLM performance suite is also an optional benchmark, not an
accuracy gate.

### Optional CPAW restart-continuity exercise

Compare uninterrupted versus restarted SMR-wave evolution, for example:

```bash
APK="$PWD/build-host/bin/athenaPK"
DECK=inputs/ct_glm_compatible/cpaw_smr_3d.in
"$APK" -i "$DECK" -d runs/cpaw-full parthenon/job/problem_id=ct \
  parthenon/time/tlim=100 parthenon/time/nlim=20 \
  parthenon/output0/dt=100 parthenon/output0/variables=cons,prim,Bface
"$APK" -i "$DECK" -d runs/cpaw-half parthenon/job/problem_id=ct \
  parthenon/time/tlim=100 parthenon/time/nlim=10 \
  parthenon/output2/file_type=rst parthenon/output2/id=restart \
  parthenon/output2/dt=100
"$APK" -r runs/cpaw-half/ct.restart.final.rhdf -d runs/cpaw-resumed \
  parthenon/time/nlim=20 parthenon/output0/dt=100 \
  parthenon/output0/variables=cons,prim,Bface
```

Verify the checkpoint filename before restarting. Compare final cycle/time,
hierarchy and cons/prim/Bface fields keyed by logical block location rather than
storage order. Inspect history divergence and conservation with energy correction
off. No automatic restart-equivalence tolerance is imposed by this manual example.

The CPU serial CPAW exercise passed on 2026-11-03. The uninterrupted and restarted
outputs both reach cycle 20 at time `0.03902305161190531` with the same 240-block
hierarchy. After matching blocks by refinement level and logical location, all
saved `cons`, `prim`, and physical `Bface` values are finite and bitwise identical
(maximum absolute difference zero). Final history `maxFaceDivB` agrees at
approximately `7.37e-14`. This establishes restart continuity for this specific
configuration, not for every problem, backend or refinement layout.

### Other optional extensions

- Repeat the restart exercise with `field_loop_amr_3d.in`, separate output
  directories, and the CT solver(s) of interest.
- Run the standalone OT AMR/SMR examples for additional refined shock-problem
  coverage beyond the automated uniform-grid smoke cases.

These extensions have not been reported as performed in this validation record.

## 6. Before submission

For the recorded validation, clang-format 20, Black, shell/Python syntax and
whitespace checks passed. License notices, AI-assistance disclosure, documentation
links, submodule references and the locally available upstream-relative diff were
reviewed. The submitter subsequently compiled the CPU serial, CPU MPI and GPU
builds after the formatting/include cleanup; no additional regression execution
is claimed after that cleanup.

Whole-file cpplint still reports style findings, including existing repository
conventions and inherited lines. It is not recorded as passed. Resolve the
remaining findings or obtain maintainer agreement on their treatment before
claiming merge readiness.

Before submission, review the complete staged diff, including new files, and
confirm that build artifacts, results and machine-specific files are excluded.
After opening the PR, add its assigned number to the changelog links and rename
the feature-plan history accordingly. Other checkouts need their own checks;
these recorded results do not establish validation for later numerical changes.
Rerun affected validation after substantive changes rather than changing
resolution, floors or thresholds merely to obtain a pass.
