#!/usr/bin/env bash
#
# This file was made in part with generative AI

set -euo pipefail

# Always operate from the repository root, regardless of the caller's
# current directory.
cd "$(git rev-parse --show-toplevel)"

CLANG_FORMAT="${CLANG_FORMAT:-clang-format}"
BLACK="${BLACK:-black}"

if ! command -v "${CLANG_FORMAT}" >/dev/null 2>&1; then
  echo "error: ${CLANG_FORMAT} was not found" >&2
  echo "Install the formatting tools with:" >&2
  echo "  python -m pip install -r requirements-format.txt" >&2
  exit 1
fi

if ! command -v "${BLACK}" >/dev/null 2>&1; then
  echo "error: ${BLACK} was not found" >&2
  echo "Install the formatting tools with:" >&2
  echo "  python -m pip install -r requirements-format.txt" >&2
  exit 1
fi

echo "Using $("${CLANG_FORMAT}" --version)"
echo "Using $("${BLACK}" --version)"

# Format tracked files only. Files inside Git submodules are therefore
# not included.
cpp_files=()
while IFS= read -r -d '' file; do
  cpp_files+=("${file}")
done < <(
  git ls-files -z -- \
    '*.c' '*.h' \
    '*.C' '*.H' \
    '*.cc' '*.hh' \
    '*.cpp' '*.hpp' \
    '*.cxx' '*.hxx' \
    '*.c++' '*.h++'
)

python_files=()
while IFS= read -r -d '' file; do
  python_files+=("${file}")
done < <(
  git ls-files -z -- '*.py'
)

if ((${#cpp_files[@]} > 0)); then
  echo "Formatting ${#cpp_files[@]} C/C++ files..."
  "${CLANG_FORMAT}" \
    --style=file \
    -i \
    "${cpp_files[@]}"
fi

if ((${#python_files[@]} > 0)); then
  echo "Formatting ${#python_files[@]} Python files..."
  "${BLACK}" "${python_files[@]}"
fi
