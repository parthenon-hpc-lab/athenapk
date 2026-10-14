#!/bin/bash
# Timeline (nsys), region tree (space-time-stack) and per-kernel metrics (ncu) for one B run.
#SBATCH --job-name=stars_prof
#SBATCH --account=coldcluster
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

source /e/scratch/jureap19/fournier/stars_test/athenapk-sfeed-profiling/jobs/common.sh
OUT=$OUT/PROF_$SLURM_JOB_ID; mkdir -p $OUT
NS=${NS:-1000}
ARGS="problem/star_formation/n_stars_per_block=$NS"
# region totals + memory high-water marks
mkdir -p $OUT/stack && cd $OUT/stack
KOKKOS_TOOLS_LIBS=$KP_STACK srun -n 4 --cpu-bind=socket $WRAP $EXE -i $IN $ARGS > athenapk.log 2>&1
# nsys timeline, NVTX ranges from PARTHENON_INSTRUMENT
mkdir -p $OUT/nsys && cd $OUT/nsys
srun -n 4 --cpu-bind=socket $WRAP nsys profile -t cuda,nvtx,mpi -o stars_%q{SLURM_PROCID} \
  $EXE -i $IN $ARGS parthenon/time/nlim=60 > athenapk.log 2>&1
# ncu on one rank (1 GPU), top stars kernels
mkdir -p $OUT/ncu && cd $OUT/ncu
CUDA_VISIBLE_DEVICES=0 ncu --kernel-name-base demangled --kernel-name regex:"ApplyStellarFeedback|ApplyGhostFeedback" \
  --launch-skip 40 --set full -c 5 -o stars_ncu \
  $EXE -i $IN $ARGS parthenon/time/nlim=30 parthenon/mesh/nx1=128 parthenon/mesh/nx2=128 \
  parthenon/mesh/nx3=128 > athenapk.log 2>&1
