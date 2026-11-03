#!/usr/bin/env bash
# This file was made in part with generative AI
# The caller loads their CUDA/MPI/HDF5 environment and provides one visible GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ATHENAPK="${ATHENAPK:-$REPO_ROOT/build-gpu-v100/bin/athenaPK}"
PYTHON="${PYTHON:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/gpu-convergence}"
RESOLUTIONS="${RESOLUTIONS:-8 16 32}"

[[ -x "$ATHENAPK" ]] || { echo "Executable not found: $ATHENAPK" >&2; exit 1; }
# Resolve before changing directory. No MPI launcher: a singleton MPI GPU process.
ATHENAPK="$(cd "$(dirname "$ATHENAPK")" && pwd)/$(basename "$ATHENAPK")"
command -v "$PYTHON" >/dev/null || { echo "Python not found: $PYTHON" >&2; exit 1; }
read -r -a sizes <<< "$RESOLUTIONS"
(( ${#sizes[@]} >= 2 )) || { echo "Need at least two resolutions." >&2; exit 1; }
previous=0
for n in "${sizes[@]}"; do
    [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid resolution: $n" >&2; exit 1; }
    (( n >= 8 && n % 2 == 0 && n > previous )) || {
        echo "Resolutions must be increasing even integers >=8." >&2; exit 1;
    }
    previous=$n
done

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
COMBINED="$OUTPUT_DIR/linearwave-errors-6.dat"
rm -f -- "$OUTPUT_DIR/order.txt"
: > "$COMBINED"
for n in "${sizes[@]}"; do
    rundir="$OUTPUT_DIR/N$n"
    mkdir -p "$rundir"
    # The pgen appends errors; remove only this generated file, not the run directory.
    rm -f -- "$rundir/linearwave-errors-6.dat"
    (
        cd "$rundir"
        "$ATHENAPK" -i "$REPO_ROOT/inputs/ct_glm_compatible/linear_mhd_wave_3d.in" \
            parthenon/job/problem_id="linear_wave_gpu_N$n" \
            problem/linear_wave_mhd/wave_flag=6 \
            problem/linear_wave_mhd/compute_error=true \
            problem/linear_wave_mhd/ang_2=-999.9 \
            problem/linear_wave_mhd/ang_3=-999.9 \
            parthenon/mesh/nx1="$((2*n))" \
            parthenon/mesh/nx2="$n" parthenon/mesh/nx3="$n" \
            parthenon/meshblock/nx1="$n" \
            parthenon/meshblock/nx2="$((n/2))" \
            parthenon/meshblock/nx3="$((n/2))" \
            hydro/fluid=ucthlldmhd
    ) |& tee "$rundir/run.out"
    rows=$(awk '!/^#/ && NF {n++} END {print n+0}' "$rundir/linearwave-errors-6.dat")
    [[ "$rows" == 1 ]] || { echo "Expected one error row in $rundir, got $rows" >&2; exit 1; }
    cat "$rundir/linearwave-errors-6.dat" >> "$COMBINED"
done

"$PYTHON" "$SCRIPT_DIR/linear_mhd_wave_orders.py" "$COMBINED" 6 > "$OUTPUT_DIR/order.txt"
echo "Combined errors: $COMBINED"
echo "Convergence report: $OUTPUT_DIR/order.txt"
