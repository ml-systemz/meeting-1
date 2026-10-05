#!/usr/bin/env bash
# Nsight Compute deep dive on a single kernel. Use after nsys tells you which one.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACES="$ROOT/traces"
KERNEL=""
COUNT=1
LABEL=""

usage() {
  cat <<'USAGE'
usage: tools/ncu_kernel.sh --kernel REGEX [--count N] [--label NAME] -- <command>

Profiles matching kernel launches with Nsight Compute and writes
traces/<label>.ncu-rep. ncu serialises and replays each launch, so it is far
slower than the real run — always bound it with --count.

  tools/ncu_kernel.sh --kernel 'fused_mlp' --count 3 -- python bench.py

If this fails with "ERR_NVGPUCTRPERM", the driver is restricting counters:
run as root, or set NVreg_RestrictProfilingToAdminUsers=0 and reload nvidia.ko.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel) KERNEL="$2"; shift 2 ;;
    --count) COUNT="$2"; shift 2 ;;
    --label) LABEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "unexpected argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$KERNEL" || $# -eq 0 ]]; then
  echo "error: --kernel and a command are both required" >&2
  usage
  exit 2
fi

command -v ncu >/dev/null 2>&1 || { echo "error: ncu not on PATH" >&2; exit 1; }

[[ -n "$LABEL" ]] || LABEL="ncu-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$TRACES"

ncu \
  --set full \
  --kernel-name "regex:$KERNEL" \
  --launch-count "$COUNT" \
  --export "$TRACES/$LABEL" \
  --force-overwrite \
  -- "$@"

echo
echo "wrote traces/$LABEL.ncu-rep"
echo "open it in the Nsight Compute UI, or: ncu --import traces/$LABEL.ncu-rep --page details"
