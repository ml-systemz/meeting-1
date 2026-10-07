#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: bench/run.sh <1|2|3> [bench.py args]"
  echo
  echo "  bench/run.sh 1 --m 8192 --n 4096"
  echo "  bench/run.sh 2 --m 4096 --k 4096 --n 4096"
  echo "  bench/run.sh 3 --b 4 --s 4096 --h 16 --d 128"
  echo "  bench/run.sh 2 --m 8192 --k 8192 --n 8192 --kw block_m=128 --kw block_k=64"
  echo
  echo "  node name comes from \$MLSYS_NODE, default \$(whoami)-pallas"
  exit 2
}

[[ $# -ge 1 ]] || usage
case "$1" in 1|2|3) WHICH="$1"; shift ;; *) usage ;; esac

NODE="${MLSYS_NODE:-$(whoami)-pallas}"
REMOTE="/home/dev/meeting-1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v autor >/dev/null 2>&1 || { echo "autor is not installed - see help/setup.md"; exit 1; }
autor whoami >/dev/null 2>&1 || { echo "not signed in - run: autor login"; exit 1; }

if ! autor node ls 2>/dev/null | awk '{print $1}' | grep -qx "$NODE"; then
  echo "creating $NODE (H100, JAX preinstalled) - a minute or two"
  autor node create --chip h100-1 --name "$NODE" \
    --image jax-0.11-cuda12.9 --clock-lock --wait
fi

state() {
  autor node get "$NODE" --json 2>/dev/null \
    | python3 -c 'import sys,json; print(json.load(sys.stdin).get("status",""))' 2>/dev/null
}

if [[ "$(state)" != running* ]]; then
  echo "waking $NODE"
  autor run "$NODE" -- true >/dev/null 2>&1 || true
  for _ in $(seq 1 60); do
    [[ "$(state)" == running* ]] && break
    sleep 10
  done
  [[ "$(state)" == running* ]] || { echo "$NODE did not come up"; exit 1; }
fi

autor run "$NODE" -- mkdir -p "$REMOTE/kernel" "$REMOTE/bench" >/dev/null
autor cp "$ROOT/kernel/kernel${WHICH}.py" "$NODE:$REMOTE/kernel/kernel${WHICH}.py" >/dev/null
autor cp "$ROOT/bench/bench.py" "$NODE:$REMOTE/bench/bench.py" >/dev/null

autor run "$NODE" -- bash -lc \
  "cd $REMOTE && PYTHONUNBUFFERED=1 python3 bench/bench.py $WHICH $* 2>&1 \
   | grep -vE '^(W[0-9]|E[0-9]|[0-9]{4}-)|bfc_allocator|hlo_rematerialization|TF_GPU_ALLOCATOR|Allocator \(GPU|Current allocation summary'"

echo "stop the meter when you are done:  autor node stop $NODE -y"
