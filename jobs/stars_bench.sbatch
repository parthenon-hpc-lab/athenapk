#!/bin/bash
# Series A (baseline) and B (star count), 1 node, SNe on.
#SBATCH --job-name=stars_AB
#SBATCH --account=coldcluster
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:45:00
#SBATCH --output=%x-%j.out

source /e/scratch/jureap19/fournier/stars_test/athenapk-sfeed-profiling/jobs/common.sh
OUT=$OUT/AB_$SLURM_JOB_ID
# A: stars module off; stars on with no stars
run_case A_off 4 stars/enabled=false
run_case A_on_N0 4 problem/star_formation/n_stars_per_block=0
# B: star count
for NS in 10 100 1000 10000; do
  run_case B_N$NS 4 problem/star_formation/n_stars_per_block=$NS
done
