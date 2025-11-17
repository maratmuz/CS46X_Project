#!/usr/bin/env bash
module load slurm

set -euo pipefail

# Default settings
PARTITION="dgxh"
GPUS=1
MEM="64g"
TIME="12:00:00"
VRAM=""
EXTRA_OPTS=""

usage() {
  cat <<USAGE
Usage: $0 [options]

Options:
  -p, --partition   Slurm partition          (default: $PARTITION)
  -g, --gpus        Number of GPUs           (default: $GPUS)
  -m, --mem         Memory per node          (default: $MEM)
  -t, --time        Walltime                 (default: $TIME)
      --vram        VRAM constraint, e.g. 80g -> --constraint=vram80g
      --extra       Extra salloc options (quoted string)
  -h, --help        Show this help message

Examples:
  $0
  $0 -g 2 -m 80g -t 24:00:00 --vram 80g
  $0 --extra "--cpus-per-task=8"
USAGE
}

# Simple flag parser
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--partition)
      PARTITION="$2"; shift 2;;
    -g|--gpus)
      GPUS="$2"; shift 2;;
    -m|--mem)
      MEM="$2"; shift 2;;
    -t|--time)
      TIME="$2"; shift 2;;
    --vram)
      VRAM="$2"; shift 2;;
    --extra)
      EXTRA_OPTS="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1;;
  esac
done

# Build salloc command
SALLOC_CMD=(salloc
  --partition="$PARTITION"
  --gres="gpu:${GPUS}"
  --mem="$MEM"
  --time="$TIME"
)

if [[ -n "$VRAM" ]]; then
  SALLOC_CMD+=(--constraint="vram${VRAM}")
fi

if [[ -n "$EXTRA_OPTS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_OPTS)
  SALLOC_CMD+=("${EXTRA_ARR[@]}")
fi

echo "Requesting allocation with:"
printf '  %q ' "${SALLOC_CMD[@]}"
echo
echo

# Precompute CUDA_VISIBLE_DEVICES string based on #GPUs
CUDA_DEVICES=""
if [[ "$GPUS" -gt 0 ]]; then
  # 0,1,2,...,(GPUS-1)
  CUDA_DEVICES=$(seq -s, 0 "$((GPUS - 1))")
fi

# Start interactive allocation; inside, grab node and SSH into it
"${SALLOC_CMD[@]}" <<EOF
echo "Allocated on node(s): \$SLURM_NODELIST"

HOST=\${SLURMD_NODENAME:-\${SLURM_NODELIST%%,*}}
echo "Connecting to host: \$HOST"
echo

ssh -t "\$HOST" 'export CUDA_VISIBLE_DEVICES='"$CUDA_DEVICES"'; \
  echo "CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"; \
  exec bash'
EOF
