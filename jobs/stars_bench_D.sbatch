#!/bin/bash
# Series D: node scaling, stars on/off. Submit with --nodes=N and NODES=N exported:
#   sbatch --nodes=1 --export=ALL,NODES=1 ...; same for 2 and 4.
# NODES=4 also runs the 512^3 weak-scaling case (blocks per GPU kept at 4).
#SBATCH --job-name=stars_D
#SBATCH --account=coldcluster
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

source /e/scratch/jureap19/fournier/stars_test/athenapk-sfeed-profiling/jobs/common.sh
NODES=${NODES:-$SLURM_NNODES}
NR=$((4 * NODES))
OUT=$OUT/D_$SLURM_JOB_ID
# strong scaling: 256^3 on all node counts
run_case D_strong_n${NODES}_on $NR problem/star_formation/n_stars_per_block=1000
run_case D_strong_n${NODES}_off $NR stars/enabled=false
# weak scaling: 256^3 per node, i.e. 4 blocks/GPU, only for NODES=4 (512^3 = 512 blocks)
if [ $NODES -eq 4 ]; then
  M="parthenon/mesh/nx1=512 parthenon/mesh/nx2=512 parthenon/mesh/nx3=512"
  run_case D_weak_n4_on $NR $M problem/star_formation/n_stars_per_block=1000
  run_case D_weak_n4_off $NR $M stars/enabled=false
fi
