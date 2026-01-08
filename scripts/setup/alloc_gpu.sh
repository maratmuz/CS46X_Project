#!/usr/bin/env bash
module load slurm

set -euo pipefail

# Default settings (align with working H200 request pattern)
LAUNCHER="srun"  # or "salloc"
PARTITION="dgxh"
GPUS=1
MEM="64g"
TIME="12:00:00"
VRAM="140"  # vram constraint; override with --vram
EXTRA_OPTS=""

usage() {
  cat <<USAGE
Usage: $0 [options]

Options:
  -p, --partition   Slurm partition          (default: $PARTITION)
  -g, --gpus        Number of GPUs           (default: $GPUS)
  -m, --mem         Memory per node          (default: $MEM)
  -t, --time        Walltime                 (default: $TIME)
      --vram        VRAM constraint, e.g. 80 -> --constraint=vram80g (default: $VRAM)
      --launcher    srun (default) or salloc
      --extra       Extra srun/salloc options (quoted string)
  -h, --help        Show this help message
USAGE
}

# Simple flag parser
while [[ $# -gt 0 ]]; do
  case "$1" in
    --launcher)
      LAUNCHER="$2"; shift 2;;
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

CMD=("$LAUNCHER"
  --partition="$PARTITION"
  --gres="gpu:${GPUS}"
  --mem="$MEM"
  --time="$TIME"
)

if [[ -n "$VRAM" ]]; then
  CMD+=(--constraint="vram${VRAM}g")
fi

if [[ -n "$EXTRA_OPTS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_OPTS)
  CMD+=("${EXTRA_ARR[@]}")
fi

# Append shell as the final args so constraints stay with srun/salloc
CMD+=(--pty bash)

echo "Requesting interactive GPU shell with:"
printf '  %q ' "${CMD[@]}"
echo
echo

# This will block until a GPU node is available,
# then drop you into a shell on that node.
"${CMD[@]}"
