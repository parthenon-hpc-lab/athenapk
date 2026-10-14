#!/bin/bash
# Series E: CPU reference (build-cpu), 1 node, 4 ranks x 72 OpenMP threads.
#SBATCH --job-name=stars_E
#SBATCH --account=coldcluster
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --time=01:30:00
#SBATCH --output=%x-%j.out

source /e/scratch/jureap19/fournier/stars_test/athenapk-sfeed-profiling/jobs/common.sh
OUT=$OUT/E_$SLURM_JOB_ID
EXE=$ROOT/build-cpu/bin/athenaPK
export OMP_NUM_THREADS=72 OMP_PROC_BIND=spread OMP_PLACES=cores
cpu_case() {
  local label=$1; shift
  mkdir -p $OUT/$label && cd $OUT/$label
  KOKKOS_TOOLS_LIBS=$KP_TIMER/libkp_kernel_timer.so \
    srun -n 4 --cpu-bind=socket $EXE -i $IN parthenon/time/nlim=60 "$@" > athenapk.log 2>&1
  $KP_TIMER/kp_reader *.dat > kernel_times.txt 2>/dev/null
}
cpu_case E_off stars/enabled=false
for NS in 10 100 1000 10000; do cpu_case E_N$NS problem/star_formation/n_stars_per_block=$NS; done
