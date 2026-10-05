#!/usr/bin/env bash
# Capture a GPU timeline for a command, then summarise the gaps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACES="$ROOT/traces"
LOG="$ROOT/traces/runs.tsv"
LABEL=""

usage() {
  cat <<'USAGE'
usage: tools/profile.sh [--label NAME] -- <command> [args...]

Runs <command> under Nsight Systems, writes traces/<label>.nsys-rep, prints a
gap summary, and appends the run to traces/runs.tsv.

If nsys is not installed the command still runs, untraced; instrument it with
tools/torchprof.py instead.

  tools/profile.sh --label baseline -- python baseline/decode.py
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "unexpected argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "error: no command given" >&2
  usage
  exit 2
fi

[[ -n "$LABEL" ]] || LABEL="run-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$TRACES"

START=$(date +%s)
OUT="$TRACES/$LABEL"

if command -v nsys >/dev/null 2>&1; then
  echo "tracing with nsys -> $OUT.nsys-rep"
  nsys profile \
    --output "$OUT" \
    --force-overwrite true \
    --trace cuda,nvtx,osrt \
    --cuda-memory-usage false \
    -- "$@"
  STATUS=$?
else
  echo "nsys not found; running untraced. Use tools/torchprof.py to capture a trace." >&2
  "$@"
  STATUS=$?
fi

END=$(date +%s)
ELAPSED=$((END - START))

mkdir -p "$(dirname "$LOG")"
if [[ ! -f "$LOG" ]]; then
  printf 'timestamp\tuser\tlabel\tseconds\tstatus\n' > "$LOG"
fi
printf '%s\t%s\t%s\t%s\t%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${USER:-unknown}" "$LABEL" "$ELAPSED" "$STATUS" >> "$LOG"

echo
echo "elapsed ${ELAPSED}s  (logged to traces/runs.tsv)"

if [[ -f "$OUT.nsys-rep" ]]; then
  python3 "$ROOT/tools/trace_stats.py" "$OUT.nsys-rep" || true
fi

exit $STATUS
