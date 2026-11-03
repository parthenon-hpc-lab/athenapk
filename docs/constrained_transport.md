# Second-order constrained transport

<!-- This file was made in part with generative AI -->

AthenaPK provides two ideal-MHD constrained-transport (CT) choices alongside the
existing GLM divergence-cleaning solver. Both use face-centered magnetic fields
and edge electric fields. No modified Parthenon checkout is required.

## Choosing a solver

| Parameter | CT choices and limits |
| --- | --- |
| hydro/fluid | ctmhd (CT-contact) or ucthlldmhd (UCT-HLLD) |
| hydro/eos | adiabatic |
| hydro/reconstruction | plm only |
| hydro/riemann | hlld only |
| parthenon/time/integrator | rk1, rk2, rk3; rk2 is the example default |
| parthenon/mesh/nghost | at least 2 |
| hydro/first_order_flux_correct | false; native correction is not CT-compatible |
| hydro/ct_energy_correction | false by default; see below |
| hydro/scratch_level | examples use 1 to avoid small GPU shared-memory limits |
| Mesh | uniform Cartesian cells within blocks; uniform, static or adaptive hierarchy |

RK1 remains first-order in time. Selecting RK3 does not make PLM third-order in
space. VL2 is rejected: its donor-cell predictor specialization is not provided
for these CT solvers. Ohmic resistivity and RKL2 are also rejected; their native
GLM implementations do not provide the CT staggered-field integration needed
here. This is an ideal-MHD addition, not a non-ideal CT implementation. Other
source/diffusion combinations are not validated by these examples.

The GLM solver retains its existing reconstruction, Riemann, and integration
choices. A pgen must explicitly initialize Bface before it can be used with CT;
changing hydro/fluid does not make an arbitrary existing pgen CT-compatible.
Reflecting magnetic boundaries are not supplied. The examples use periodic
boundaries; outflow/custom magnetic boundaries require problem-specific review.

## Storage and the RK stage

- cons and prim hold the usual cell-centered variables. CT has eight fluid
  variables and no GLM psi variable.
- Bface is one face-centered scalar per orientation. It is independent,
  conserved, ghost-filled, and has associated edge flux storage.
- Bface.flux stores the physical electric field E = -v cross B on edges.
- uct_hlld stores six derived auxiliary values per face: AL, AR, DL, DR,
  VBART1, VBART2. They are recomputed, not independently evolved.

The normal magnetic component in the left and right reconstructed Riemann states
is replaced by the same evolved face value. Cell-centered Godunov magnetic
updates still supply transverse fields in inactive dimensions and the magnetic
estimate used by optional energy correction.

Each stage executes:

    cell reconstruction and HLLD Riemann solves
      -> CT-contact or UCT-HLLD edge EMFs
      -> Parthenon flux/EMF communication and refinement corrections
      -> finite-volume update of cons and internal UpdateWithFluxEmfCurl of Bface
      -> source terms and boundary exchange
      -> center_Mag_Field (PreFillDerivedMesh)
      -> conserved-to-primitive conversion

CT-contact follows the Gardiner--Stone upwind construction, using face electric
fluxes, cell electric fields, and mass-flux signs. UCT-HLLD follows the systematic
Mignone--Del Zanna construction, including PLM transverse reconstruction and the
HLLD-dependent transport/diffusion coefficients. The HLLD face solve and edge
assembly remain distinct operations. These are method-of-lines RK paths, not
an implementation of the CTU predictor/corrector integrator.

In 1D, only face Bx is updated, with zero curl RHS; By/Bz evolve in cons.
In 2D, Bx/By are face-evolved and Bz remains cell-evolved. All three are
face-evolved in 3D. Inactive components must not be overwritten from unused faces.
The internal curl task handles these cases without reading nonexistent 1D EMFs.

## Initializing a CT problem

For nontrivial analytic fields, evaluate each vector-potential component at its
component-aligned edge center. Take its compatible discrete curl to form face
fields, center active face components into cons, and use those magnetic values
when initializing total energy. An analytically divergence-free point field,
sampled independently onto faces, is not by itself a discretely solenoidal field.
Constant normal fields can instead be initialized directly, as in the 1D waves.

This PR uses second-order midpoint representations. It does not introduce
higher-order initialization or Gaussian quadrature. CPAW/linear-wave 1D setups
require zero rotation angles; their angle defaults are dimension-aware.
The field-loop CT setup supports iprob=1 in 2D/3D and the rotated cylinder
iprob=4 in 3D. Other CT orientations are rejected.

### Matching initial coarse and fine faces

Parthenon can regenerate analytic initial conditions while discovering the
initial AMR hierarchy. Independently sampling coarse and fine vector potentials
can disagree when a coarse edge is the union of two fine edges. Even if each
block's curl is initially divergence-free, replacing a shared face during
communication/restriction can then introduce divergence in adjacent cells.

The CPAW, linear-wave, and rotated-field-loop initializers share a host-side
neighbor matcher adapted from Athena++'s nblevel checks. It reads GetNeighbors(),
neighbor offsets and logical refinement levels. Proper nesting supplies a 2:1
coarse/fine ratio. On a coarse edge that touches finer neighbors, the potential
component along the edge is replaced by

    A_coarse = [A(x_center - dx/4) + A(x_center + dx/4)] / 2.

Those are the two fine-edge centers, not arbitrary offsets in global coordinates.
The edge width comes from Dxf, not from multiplying a coordinate by 0.25.
Only transverse block boundaries touching that edge matter. A finer neighbor
elsewhere on the block must not trigger the correction.

This matches the midpoint edge integrals and hence the restricted fine-face
magnetic flux. It is applied before taking the initial curl. In 2D the relevant
potential is nodal in the inactive direction, so no active-edge subdivision is
needed. Analytically trivial potential components also need no correction.

This analytic initialization correction does not reinitialize the evolving solution.
Subsequent remeshing uses Parthenon's registered operators:

- ProlongateSharedMinMod for shared faces;
- RestrictAverage for restriction;
- ProlongateInternalTothAndRoe for internal fine faces.

Edge-EMF correction and communication occur before the face curl update. Compatible
initialization, divergence-preserving transfer, and synchronized EMFs serve
different purposes; none is a substitute for the others.

## Optional magnetic-energy correction

When centering Bface into cons, hydro/ct_energy_correction=true applies

    E_new = E_old + (|B_CT|^2 - |B_Godunov|^2)/2.

This preserves internal energy through that replacement while leaving momentum
unchanged. It sacrifices strict total-energy conservation and may help
magnetically dominated problems, but is neither a positivity guarantee nor a
pressure floor. With false, the replacement does not alter conserved total
energy. The examples explicitly disable it except for current sheet, which
enables it. Standard density/pressure floors remain separate options.

## Standalone examples

All new decks are in inputs/ct_glm_compatible/. The unqualified filenames are
2D baselines. The original inputs/field_loop.in, inputs/cpaw.in and
inputs/orszag_tang.in remain available as GLM examples.

| Problem | Additional deck suffixes |
| --- | --- |
| cpaw | _1d, _3d, _smr_3d |
| linear_mhd_wave | _1d, _3d, _smr_3d |
| field_loop | _3d, _amr_2d, _amr_3d, _smr_3d |
| orszag_tang | _amr_2d, _smr_2d |
| current_sheet | 2D baseline |

Run from the repository root, using separate output directories:

    ./build-host/bin/athenaPK -i inputs/ct_glm_compatible/cpaw.in -d runs/cpaw
    ./build-host/bin/athenaPK -i inputs/ct_glm_compatible/field_loop_amr_3d.in -d runs/loop
    ./build-host/bin/athenaPK -i inputs/ct_glm_compatible/cpaw_smr_3d.in -d runs/smr

Change only hydro/fluid=ctmhd or hydro/fluid=glmmhd to compare solvers on the same
deck. GLM does not use Bface or ct_energy_correction. Keep initial conditions,
mesh, reconstruction and integration fixed for a controlled comparison.

job/problem_id selects the pgen. parthenon/job/problem_id selects the output
basename; changing only the latter must not change the generator. Linear-wave
decks use wave_flag=0 by default; 0/1/2 and 4/5/6 are the left/right fast,
Alfven and slow families, with 3 the entropy/contact mode. For a mode-6 test, set
problem/linear_wave_mhd/wave_flag=6.

CPAW interprets tlim as wave periods when its propagation speed is nonzero.
Linear waves do likewise with problem/linear_wave_mhd/test=true; avoid the
zero-speed contact mode with that option. Error routines compare against the
initial reference and therefore expect a full number of return periods, not an
arbitrary final time. Negative PHDF output dt disables that output; history
output remains enabled. Set a positive output0/dt to obtain snapshots.

Wave baseline N=32 means 64x32 in 2D or 64x32x32 in 3D. Change all active
resolutions and matching block sizes together. There must be at least as many
leaf MeshBlocks as MPI ranks. A single-block run should use one rank; more
blocks do not imply more ranks. The 1D decks contain two blocks; the 3D uniform
wave decks contain sixteen.

The field-loop vflow1/2/3 overrides are signed domain-crossing rates: the pgen
multiplies them by the respective global domain lengths to obtain velocities.
Omitted components inherit vflow. In 2D the third velocity is zero.

The refined decks preserve the comparison setup: nested level-1/2 regions for
waves; centered level-1 field-loop SMR; field-magnitude AMR thresholds
0.0005/0.00025 with three total levels; OT pressure-gradient threshold 0.2;
and the four original OT SMR regions. Field-loop runs end at 2, OT at 1,
current sheet at 10 in code time.

### GPU convergence utility

The portable script requires Bash, awk, Python 3, a GPU build, and an already
configured CUDA/MPI/HDF5 runtime environment. It does not load site modules,
reserve a GPU, launch Slurm jobs or compile. Run on an allocated or permitted
development GPU. It launches one singleton MPI process; sites requiring a
launcher should use their approved execution mechanism instead.

    ATHENAPK="$PWD/build-gpu-v100/bin/athenaPK" PYTHON=python3 \
      bash scripts/ct_convergence/gpu_convergence_test.sh

Optional OUTPUT_DIR and RESOLUTIONS overrides select a different result
directory and a whitespace-separated increasing list of even resolutions >=8.
Defaults are gpu-convergence/ and N=8,16,32. The mesh is 2N x N x N, eight
blocks, oblique 3D mode 6, UCT-HLLD/PLM/HLLD/RK2.

Only generated per-resolution error files, the combined table, and the old order
report are cleared on reruns. Run logs are overwritten by tee; other files are
not recursively removed.
Each solve must succeed and write exactly one row. The companion Python script
writes order.txt from linearwave-errors-6.dat.
Use separate result directories when changing solver or setup.

## Reading divergence and convergence diagnostics

For uniform Cartesian cells define

    D_face = (Bx[i+1]-Bx[i])/dx + (By[j+1]-By[j])/dy
             + (Bz[k+1]-Bz[k])/dz,
    L_active = sqrt(sum of squared active cell widths).

Only active directional differences are included. Package faceDivB is the
domain-volume-weighted mean of L_active*abs(D_face)/abs(B_cell).
maxFaceDivB is its cellwise maximum, reduced with MPI max. They are
dimensionless, not raw divergence in inverse-length units. Cells with exactly
zero local magnetic magnitude contribute zero to these package diagnostics,
so a fixed-reference measure is preferable near magnetic nulls.

For roundoff-level conservation checks set output1/data_format=%.16e; the
default five-decimal history precision cannot resolve such small changes.
The new CT regressions request the higher precision explicitly.

For GLM in 2D/3D, relDivB uses centered cell-field differences (half the
two-cell difference divided by dx) and the corresponding package cell-width
normalization. It is not the CT divergence operator.

Field loop retains legacy UserRelDivB and UserFaceDivB.
Let L_all=sqrt(dx^2+dy^2+dz^2), including the inactive-direction width in 2D,
and let B0 be the loop amplitude. Their definitions
are volume integrals, not domain-volume-normalized means:

    UserRelDivB = integral L_all*abs(D_centered)/B0 dV,
    UserFaceDivB = integral (L_all/2)*abs(D_face)/B0 dV.

Field-loop MaxFaceDivB (capital M) and OTMaxFaceDivB use
max(L_active*abs(D_face)/B0), with B0=amp for the loop and
B0=1/sqrt(4*pi) for OT. The capitalization and extra factors matter: these
columns are not interchangeable with maxFaceDivB or with each other.

CPAW and linear wave write volume-weighted L1 errors over leaf
cells, normalized by the total domain volume. Their RMS-L1 column combines
the component L1 errors; it is not the RMS of pointwise errors. CT magnetic
references use the corresponding staggered initialization/centering convention,
not an independently sampled analytic point field. Error files append a row on
every run: remove or isolate the exact error file before a convergence sequence.
Formal convergence is expected for smooth waves, not for the nonsmooth loop edge.

## Building and validation

See [the validation guide](ct_validation.md) for step-by-step CPU/MPI/GPU builds,
native test execution, persistent summaries, interpretation, measured runtimes,
GLM compatibility checks, and optional extended manual checks. The guide records
the fresh automated results from 2026-11-03 and separates them from the optional
CPAW restart exercise, which matched final time, hierarchy and fields exactly.
Other checkouts and
environments need their own validation; a failed or incomplete automated check
must not be presented as passed.

## References and provenance

- Gardiner and Stone (2005), An unsplit Godunov method for ideal MHD via
  constrained transport: upwind contact-based EMF construction.
- Mignone and Del Zanna (2021), A systematic construction of upwind constrained
  transport schemes for MHD: UCT-HLLD construction and magnetic-energy correction.
- Miyoshi and Kusano (2005): HLLD Riemann solver.
- Toth and Roe (2002): divergence-preserving mesh refinement.
- Athena++ CPAW initialization: neighbor-aware subdivision of edge potentials.
- The pinned Parthenon fine-advection example and refinement/flux communication
  infrastructure: staggered metadata and divergence-preserving transfer.

Original Athena++/AthenaPK copyright and BSD license notices are retained.
AI assistance was used for adaptation, cleanup, examples, documentation and
regression tooling; the submitter remains responsible for review and validation.
