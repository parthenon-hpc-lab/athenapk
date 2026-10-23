# Self-gravity scaling test on JUPITER

Goal: measure what self-gravity costs on GPU nodes, and how that cost scales with the
number of nodes. The test problem is driven turbulence with two nested levels of static
mesh refinement. Each case runs twice from the same developed-turbulence restart: once
without self-gravity and once with the optimized solver settings.

## Files

| File | What it is |
|---|---|
| `inputs/benchmarks/turb_sg_smr.in` | Test deck: periodic box, 64³ blocks, refinement level 1 in the central half of the box, level 2 in the central quarter. Optimized solver settings. |
| `jobs/sg_scaling.sbatch` | Job script with three modes: `PREP`, `CHECK` and timing. |
| `jobs/sg_scaling_summary.py` | Reads the logs; prints seconds per cycle, zone-cycles/s, throughput per GPU and the self-gravity cost ratio. |
| `jobs/parthenon_bicgstab_zero_rhs.patch` | Parthenon fix for the 0/0 in BiCGSTAB when the Poisson right-hand side is zero. Needed because the turbulence starts from a uniform density. |

These four files are not committed or pushed yet. Copy them over with the checkout, or
commit them first.

Optimized solver settings in the deck (`<self_gravity/multigrid_solver_params>`):

```
block_interior_prolongation  = MetadataDefault
relative_residual_tolerance  = 1.0e-4
absolute_residual_tolerance  = 1.0e-6
```

On one CPU core at 64³, these settings gave self-gravity at about 2.6× the cost of
hydro alone, against 10.4× with the defaults. The fields stayed within ~1e-8 of a fully
converged solve.

Mesh size: each refinement level holds about as many cells as the root grid, so the
mesh has about 2.8 × NROOT³ cells. With 64³ blocks:

| NROOT | Cells | Blocks | Finest Δx |
|---|---|---|---|
| 256 | 4.6e7 | 176 | 1/1024 |
| 512 | 3.7e8 | 1408 | 1/2048 |

## Plan: 1, 2 and 4 nodes

With only 1, 2 and 4 nodes, a clean weak-scaling series isn't possible: cubic root grids
grow the cell count in steps of 8×, not 2×. Run two strong-scaling series instead:

| Series | Mesh | Nodes | Cells per GPU | Blocks per GPU |
|---|---|---|---|---|
| A | NROOT = 256 | 1, 2, 4 | 1.2e7 → 2.9e6 | 44 → 11 |
| B | NROOT = 512 | 2, 4 (1 if it fits) | 4.6e7 → 2.3e7 | 176 → 88 |

Series A shows how self-gravity's cost grows as the GPUs run out of work. Series B keeps
the GPUs busy. Comparing A at 1 node with B at 4 nodes (1.2e7 vs 2.3e7 cells per GPU)
gives a rough weak-scaling point.

## How long it should take

These are estimates, not measurements. They assume ~5e7–1e8 zone-cycles/s per GPU for
this MHD setup without self-gravity, and a self-gravity cost of 3–6× on GPUs (2.6× on one
CPU core; more communication-bound on GPUs). The time step at the finest level is about
0.3·Δx/1.5, so ~2e-4 for NROOT = 256 and ~1e-4 for NROOT = 512.

| Job | Nodes | Cycles | Wall time | Node-hours |
|---|---|---|---|---|
| PREP, NROOT = 256 (to t = 0.5) | 1 | ~2500 | 5–10 min | 0.2 |
| PREP, NROOT = 512 (to t = 0.5) | 4 | ~5000 | 15–30 min | 1–2 |
| CHECK | 1 | 10 | 2–3 min | < 0.1 |
| Timing, series A | 1 / 2 / 4 | 2 × 200 | ~10 / 7 / 5 min | ~0.8 |
| Timing, series B | 2 / 4 | 2 × 200 | ~25 / 15 min | ~1.8 |

Total: about 4–5 node-hours, and about 1.5 hours of wall time if the jobs start
straight away. Without self-gravity a cycle should take ~0.1–0.2 s for series A on 1
node, and ~0.3–0.6 s with it. The first CHECK and series-A jobs calibrate these
numbers; if they are far off, scale the `--time` limits accordingly. Memory is not a
concern: roughly 0.7 kB per cell with the solver's work fields, so ~30 GB per GPU for
series B on 2 nodes, and ~60 GB on 1 node (of 96 GB).

## 1. Check out and build

```bash
export ROOT=/e/scratch/jureap19/fournier/self_gravity/athenapk-sfeed
git clone --recurse-submodules -b mfournier01/self-gravity \
  https://github.com/parthenon-hpc-lab/athenapk.git $ROOT
# copy the four files above into $ROOT, then:
cd $ROOT
git -C external/parthenon apply ../../jobs/parthenon_bicgstab_zero_rhs.patch
cp /e/scratch/jureap19/fournier/stars_test/athenapk-sfeed-profiling/jupiter_env.sh .  # if missing
source jupiter_env.sh
```

Reuse the GPU configuration of the existing profiling build:

```bash
grep -E "Kokkos_ENABLE_CUDA|Kokkos_ARCH_|CMAKE_BUILD_TYPE|PARTHENON_DISABLE" \
  /e/scratch/jureap19/fournier/stars_test/athenapk-sfeed-profiling/build-gpu/CMakeCache.txt
```

Then configure `build-gpu` with the same `-D...` flags, and add
`-DAthenaPK_ENABLE_TESTING=OFF` (the regression tests need Python modules at configure
time):

```bash
cmake -S . -B build-gpu -G Ninja <flags from above> -DAthenaPK_ENABLE_TESTING=OFF
cmake --build build-gpu -j 32
```

## 2. Make the restart files (`PREP`)

Evolve the turbulence without self-gravity to t = 0.5, so the timing runs start from
developed turbulence. The density is then far from uniform and the solver needs a
realistic number of iterations. Starting from t = 0 would make self-gravity look
cheaper than it is. Do this once per root grid size:

```bash
sbatch --nodes=1 --time=00:30:00 --export=ALL,PREP=1,NROOT=256 jobs/sg_scaling.sbatch
sbatch --nodes=4 --time=01:00:00 --export=ALL,PREP=1,NROOT=512 jobs/sg_scaling.sbatch
```

The restart lands in `$ROOT/results/sg_scaling/prep_root<NROOT>/`, and the job log
prints its path. The time step is set by the finest level (Δx = 1/(4·NROOT)), so a run
to t = 0.5 takes a few thousand cycles. If it doesn't finish, lower `TPREP` (e.g.
`TPREP=0.3`) or raise `--time`.

## 3. Check the solver on the refined mesh (`CHECK`)

`MetadataDefault` has only been tested on a uniform grid. Before timing anything, run 10
cycles with the solver residuals printed:

```bash
RST=$ROOT/results/sg_scaling/prep_root256/parthenon.restart.final.rhdf
sbatch --nodes=1 --export=ALL,CHECK=1,RST=$RST jobs/sg_scaling.sbatch
```

In `.../check/athenapk.log`, each solve prints `0 <r0>`, then two lines per iteration.
Expect 1–3 iterations per solve; the first solve after the restart starts from φ = 0 and
needs more. If solves reach `max_iterations` (100), the refined mesh doesn't suit
`MetadataDefault`. In that case set `block_interior_prolongation = Linear` in the deck,
which is slower but robust, and note it in the results.

## 4. Timing runs

Each timing job runs 200 cycles from the restart, first without and then with
self-gravity, and writes `summary.txt`. Pass `NCYC=...` for a different number of cycles.

**Series A** (NROOT = 256):

```bash
RST=$ROOT/results/sg_scaling/prep_root256/parthenon.restart.final.rhdf
for n in 1 2 4; do
  sbatch --nodes=$n --time=00:20:00 --export=ALL,RST=$RST jobs/sg_scaling.sbatch
done
```

**Series B** (NROOT = 512):

```bash
RST=$ROOT/results/sg_scaling/prep_root512/parthenon.restart.final.rhdf
for n in 2 4; do
  sbatch --nodes=$n --time=00:45:00 --export=ALL,RST=$RST jobs/sg_scaling.sbatch
done
# optional, if it fits in memory:
sbatch --nodes=1 --time=01:00:00 --export=ALL,RST=$RST jobs/sg_scaling.sbatch
```

At 4 nodes, series A has only 11 blocks per GPU, so expect to approach the
latency-bound regime there; that is part of what the test should show.

## 5. Collect the results

```bash
cat $ROOT/results/sg_scaling/*/summary.txt
```

Each `summary.txt` gives, for `nosg` and `sg`: ranks, cells, median seconds per cycle,
zone-cycles/s, zone-cycles/s per GPU, and the cost ratio `sg/nosg`. The first 10 cycles
are skipped, because they include setup and the first Poisson solve.

What to plot:

- **Strong scaling:** seconds per cycle against nodes, for `nosg` and `sg`, one line per
  series. The ideal is a slope of −1 on log-log axes.
- **Per-GPU throughput:** zone-cycles/s per GPU for series A at 1 node and series B at 4
  nodes. Similar values mean the code would weak-scale well over this range.
- **Cost of self-gravity:** the `sg/nosg` ratio against nodes. This is the main number.
  On one CPU core it was 2.6×. On GPUs it will likely be higher, because the multigrid
  solve does many small kernels, several hundred ghost-cell exchanges per step and 3
  global reductions per BiCGSTAB iteration. Expect it to grow with node count.

If the ratio is much higher than on the CPU, profile one self-gravity case with Nsight
Systems (as in `jobs/profile_nsys_ncu.sbatch`) to see whether communication or kernel
launches dominate.

## Caveats

- The job script and deck are untested on JUPITER. On a laptop, the deck ran from t = 0
  at reduced size (64³ root, 16³ blocks), with and without self-gravity. The
  `PREP` → restart → timing path was not run end to end.
- Self-gravity with a fully periodic box subtracts the mean density, which requires one
  mesh partition per rank. Don't set `parthenon/mesh/pack_size` or
  `parthenon/mesh/packs_per_rank`.
- The absolute tolerance 1e-6 is in code units of 4πGρ; it suits this deck (RHS of order
  1). For the cluster setup it has to be rescaled.
