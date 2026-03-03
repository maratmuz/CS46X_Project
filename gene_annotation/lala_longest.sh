#!/usr/bin/env bash
set -euo pipefail

# Activate virtual environment
source /nfs/stak/users/limjar/hpc-share/myVenv/bin/activate

echo "=== Job started at $(date) ==="

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <base_directory> [path/to/lala_longest.py]" >&2
  echo "Example: $0 three_algae" >&2
  exit 1
fi

BASE_DIR=$1
SCRIPT_PATH=${2:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lala_longest.py"}

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Error: base directory not found: $BASE_DIR" >&2
  exit 1
fi

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Error: lala_longest.py not found: $SCRIPT_PATH" >&2
  exit 1
fi

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN=python
else
  echo "Error: python interpreter not found (expected python3 or python)." >&2
  exit 1
fi

count=0
while IFS= read -r -d '' gff_file; do
  ext=${gff_file##*.}
  out_file="${gff_file%.*}.reduced.${ext}"
  printf 'Filtering %s -> %s\n' "$gff_file" "$out_file"
  "$PYTHON_BIN" "$SCRIPT_PATH" --gff-file "$gff_file" > "$out_file"
  count=$((count + 1))
done < <(
  find "$BASE_DIR" -type f \
    \( -iname "*.gff" -o -iname "*.gff3" \) \
    ! -iname "*.reduced.gff" \
    ! -iname "*.reduced.gff3" \
    -print0
)

if [[ $count -eq 0 ]]; then
  echo "No .gff or .gff3 files found under: $BASE_DIR"
  exit 0
fi

echo "Finished. Reduced $count file(s)."
