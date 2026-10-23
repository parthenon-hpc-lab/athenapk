#!/bin/bash
# Self-gravity scaling test: driven turbulence with two levels of static refinement
# (inputs/benchmarks/turb_sg_smr.in), each case run without and with self-gravity.
# One GPU per rank, 4 ranks per node. Modes, selected with --export=ALL,...:
#   PREP=1,NROOT=256   evolve the turbulence without self-gravity to t = TPREP (default
#                      0.5) and write a restart file; once per root grid size
#   CHECK=1,RST=<file> 10 cycles with the solver residuals printed, to check the
#                      iteration count on the refined mesh before timing anything
#   RST=<file>         timing: NCYC (default 200) cycles from the restart, without and
#                      with self-gravity, then a summary
# The restart file fixes the mesh, so a timing job can use any number of nodes.
#SBATCH --job-name=sg_scaling
#SBATCH --account=coldcluster
#SBATCH --partition=booster
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=72
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out

ROOT=${ROOT:-/e/scratch/jureap19/fournier/self_gravity/athenapk-sfeed}
source $ROOT/jupiter_env.sh
export OMP_NUM_THREADS=1
EXE=$ROOT/build-gpu/bin/athenaPK
IN=$ROOT/inputs/benchmarks/turb_sg_smr.in
WRAP=$ROOT/jobs/gpu_wrapper.sh
NRANKS=$((SLURM_NNODES * 4))
RESULTS=${RESULTS:-$ROOT/results/sg_scaling}

run_case() { # directory, athenaPK arguments...
  local dir=$1; shift
  mkdir -p $dir && cd $dir
  srun -n $NRANKS --cpu-bind=socket $WRAP $EXE "$@" > athenapk.log 2>&1
  echo "$dir exit=$?"
}

if [[ ${PREP:-0} == 1 ]]; then
  NROOT=${NROOT:-256}
  # Without self-gravity (cheaper); the self-gravity runs solve for the potential on
  # their first step after the restart.
  run_case $RESULTS/prep_root$NROOT -i $IN \
    parthenon/mesh/nx1=$NROOT parthenon/mesh/nx2=$NROOT parthenon/mesh/nx3=$NROOT \
    parthenon/time/tlim=${TPREP:-0.5} parthenon/time/nlim=-1 parthenon/output3/dt=100 \
    self_gravity/solver=none parthenon/mesh/multigrid=false
  echo "restart: $(ls $RESULTS/prep_root$NROOT/*.restart.final.rhdf)"
  exit
fi

[[ -f $RST ]] || { echo "Set RST to a restart file (make one with PREP=1)"; exit 1; }
CYC0=$(python3 -c "import h5py; print(int(h5py.File('$RST', 'r')['Info'].attrs['NCycle']))")
NLIM="parthenon/time/nlim=$((CYC0 + ${NCYC:-200})) parthenon/time/tlim=100"
OUT=$RESULTS/$(basename $(dirname $RST))_n${SLURM_NNODES}_$SLURM_JOB_ID

if [[ ${CHECK:-0} == 1 ]]; then
  run_case $OUT/check -r $RST parthenon/time/nlim=$((CYC0 + 10)) parthenon/time/tlim=100 \
    self_gravity/multigrid_solver_params/print_per_step=true
  exit
fi

run_case $OUT/nosg -r $RST $NLIM self_gravity/solver=none parthenon/mesh/multigrid=false
run_case $OUT/sg -r $RST $NLIM
SLURM_NTASKS=$NRANKS python3 $ROOT/jobs/sg_scaling_summary.py $OUT/nosg $OUT/sg \
  | tee $OUT/summary.txt
