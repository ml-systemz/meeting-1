#!/usr/bin/env bash
set -uo pipefail

NODE="${MLSYS_NODE:-$(whoami)-pallas}"
ok()  { echo "  ok    $1"; }
bad() { echo "  BAD   $1"; }

echo
echo "your laptop"
command -v autor >/dev/null 2>&1 && ok "autor installed" || {
  bad "autor missing - see help/setup.md"; exit 1; }
autor whoami >/dev/null 2>&1 && ok "signed in" || {
  bad "not signed in - run: autor login"; exit 1; }
autor billing >/dev/null 2>&1 && ok "on the team" \
  || echo "  ?     cannot read billing - either you are not on the team, or your
        token has no org:read scope. Harmless if the gpu check below passes."

echo
echo "your machine"
STATE=$(autor node get "$NODE" --json 2>/dev/null \
  | python3 -c 'import sys,json; print(json.load(sys.stdin).get("status",""))' 2>/dev/null)
if [[ -z "$STATE" ]]; then
  bad "no node called $NODE - bench/run.sh will create one"
  exit 1
fi
ok "$NODE is $STATE"
if [[ "$STATE" != running* ]]; then
  echo "  waking it"
  autor run "$NODE" -- true >/dev/null 2>&1 || true
fi

echo
echo "the gpu"
autor run "$NODE" -- python3 -c '
import jax
d = jax.devices()[0]
print("  ok    jax", jax.__version__)
if d.platform != "gpu":
    print("  BAD   jax fell back to CPU - every timing is meaningless")
else:
    print("  ok    ", d, "cc", d.compute_capability)
from jax.experimental import pallas
from jax.experimental.pallas import triton
print("  ok    pallas + triton backend import")
' 2>&1 | grep -E '^\s+(ok|BAD)' || bad "could not reach the node"

echo
echo "if that is all ok, the problem is your kernel - see help/debug.md"
echo
