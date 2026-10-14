#!/bin/bash
# Section 3: star regression tests on 1 GPU, then sn_blast_wave on 1 vs 4 GPUs.
#SBATCH --job-name=stars_test
#SBATCH --account=coldcluster
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

source /e/scratch/jureap19/fournier/stars_test/athenapk-sfeed-profiling/jobs/common.sh
export KOKKOS_PRINT_CONFIGURATION=true
cd $ROOT/build-gpu
export CUDA_VISIBLE_DEVICES=0
ctest -R "star_formation_sfr_stats|cluster_star_orbits|sn_feedback_refinement|sn_blast_wave" --output-on-failure

# 1 vs 4 GPU comparison on the sn_blast_wave input (6 steps)
unset CUDA_VISIBLE_DEVICES
W=$OUT/test_$SLURM_JOB_ID; mkdir -p $W
for NR in 1 4; do
  mkdir -p $W/np$NR && cd $W/np$NR
  srun -n $NR --cpu-bind=socket $WRAP $EXE -i $ROOT/inputs/sn_blast_wave.in \
    parthenon/time/nlim=6 parthenon/output1/file_type=hst parthenon/output1/dn=1 \
    > athenapk.log 2>&1
done
echo "compare: python $ROOT/analysis/compare_hst.py $W/np1 $W/np4"
