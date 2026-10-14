#!/bin/bash
# Series C: same star counts as B with SN II and SN Ia disabled (1 node).
#SBATCH --job-name=stars_C
#SBATCH --account=coldcluster
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

source /e/scratch/jureap19/fournier/stars_test/athenapk-sfeed-profiling/jobs/common.sh
OUT=$OUT/C_$SLURM_JOB_ID
for NS in 1000 10000; do
  run_case C_noSN_N$NS 4 problem/star_formation/n_stars_per_block=$NS \
    stars/SN_II_enabled=false stars/SN_Ia_enabled=false
done
