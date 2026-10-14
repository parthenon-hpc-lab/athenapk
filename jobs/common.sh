# Shared settings for the stars benchmark jobs (sourced by the sbatch scripts).
ROOT=/e/scratch/jureap19/fournier/stars_test/athenapk-sfeed-profiling
source $ROOT/jupiter_env.sh
export KT=$ROOT/kokkos-tools
export KP_TIMER=$KT/build/profiling/simple-kernel-timer
export KP_STACK=$KT/build/profiling/space-time-stack/libkp_space_time_stack.so
export OMP_NUM_THREADS=1
export KOKKOS_TOOLS_TIMER_BINARY=true  # write <host>-<pid>.dat, not ASCII to stdout
EXE=$ROOT/build-gpu/bin/athenaPK
IN=$ROOT/inputs/benchmarks/stars_bench.in
WRAP=$ROOT/jobs/gpu_wrapper.sh
OUT=${OUT:-$ROOT/results}

# run_case <label> <nranks> [parthenon overrides...]: kernel-timer run in $OUT/<label>
run_case() {
  local label=$1 nranks=$2; shift 2
  mkdir -p $OUT/$label && cd $OUT/$label
  KOKKOS_TOOLS_LIBS=$KP_TIMER/libkp_kernel_timer.so \
    srun -n $nranks --cpu-bind=socket $WRAP $EXE -i $IN "$@" > athenapk.log 2>&1
  $KP_TIMER/kp_reader *.dat > kernel_times.txt 2>/dev/null
  cd $OUT
}
