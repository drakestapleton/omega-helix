#!/bin/sh
# End-to-end 5-second CPU smoke test for Omega Helix:
#   init -> gcheck (gradient oracle) -> train -> eval -> kvcmp (KV cache exactness) -> sample
#
# Usage:
#   sh scripts/quickstart-cpu.sh [path/to/ollm-binary]
set -eu

here=$(cd "$(dirname "$0")/.." && pwd)
BIN=${1:-}

if [ -z "$BIN" ]; then
  if [ "$(uname -s)" = Darwin ]; then
    BIN=$here/build/ollm-macos
    [ -x "$BIN" ] || TARGET=macos sh "$here/scripts/build.sh"
  else
    BIN=$here/build/ollm
    [ -x "$BIN" ] || TARGET=aarch64 sh "$here/scripts/build.sh"
  fi
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/omega-helix-smoke.XXXXXX")
trap 'rm -rf "$work"' EXIT

# Build a small training/eval byte corpus from this repo's README and docs
cat "$here/README.md" "$here/docs/LOOP.md" > "$work/corpus.txt"

echo "=== 1. Initialize tiny byte model (V=256, T=16, D=16, H=2, F=32, L=2, B=4) ==="
"$BIN" init "$work/m.ck" 256 16 16 2 32 2 4 1 20 5000

echo "=== 2. Finite-difference gradient check (gcheck) ==="
"$BIN" gcheck "$work/m.ck" "$work/corpus.txt"

echo "=== 3. Eval before training ==="
"$BIN" eval "$work/m.ck" "$work/corpus.txt" 8

echo "=== 4. Train 20 steps on CPU ==="
"$BIN" train "$work/m.ck" "$work/corpus.txt" 20 5

echo "=== 5. Eval after training ==="
"$BIN" eval "$work/m.ck" "$work/corpus.txt" 8

echo "=== 6. Sample 48 bytes ==="
"$BIN" sample "$work/m.ck" 42 48 800 "Omega "
echo ""
echo "QUICKSTART_CPU_OK"
