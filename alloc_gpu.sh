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
      --extra       Extra srun options (quoted string)
  -h, --help        Show this help message
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

SRUN_CMD=(srun
  --partition="$PARTITION"
  --gres="gpu:${GPUS}"
  --mem="$MEM"
  --time="$TIME"
  --pty bash
)

if [[ -n "$VRAM" ]]; then
  SRUN_CMD+=(--constraint="vram${VRAM}")
fi

if [[ -n "$EXTRA_OPTS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_OPTS)
  SRUN_CMD+=("${EXTRA_ARR[@]}")
fi

echo "Requesting interactive GPU shell with:"
printf '  %q ' "${SRUN_CMD[@]}"
echo
echo

# This will block until a GPU node is available,
# then drop you into a shell on that node.
"${SRUN_CMD[@]}"
