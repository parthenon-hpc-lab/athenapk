#!/bin/bash
# Series A-C re-timed with NO Kokkos tools loaded, 3 repeats (cases interleaved per repeat).
#SBATCH --job-name=stars_notool
#SBATCH --account=coldcluster
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

source /e/scratch/jureap19/fournier/stars_test/athenapk-sfeed-profiling/jobs/common.sh
unset KOKKOS_TOOLS_LIBS KOKKOS_TOOLS_TIMER_BINARY
OUT=$OUT/NT_$SLURM_JOB_ID
P=problem/star_formation/n_stars_per_block
run_nt() {  # run_nt <rep> <label> <overrides...>
  local rep=$1 label=$2; shift 2
  mkdir -p $OUT/rep$rep/$label && cd $OUT/rep$rep/$label
  srun -n 4 --cpu-bind=socket $WRAP $EXE -i $IN "$@" > athenapk.log 2>&1
}
for rep in 1 2 3; do
  run_nt $rep A_off stars/enabled=false
  run_nt $rep A_on_N0 $P=0
  for NS in 10 100 1000 10000; do run_nt $rep B_N$NS $P=$NS; done
  for NS in 1000 10000; do
    run_nt $rep C_noSN_N$NS $P=$NS stars/SN_II_enabled=false stars/SN_Ia_enabled=false
  done
done
