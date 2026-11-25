#!/bin/sh
# End-to-end CPU smoke test for Omega Helix:
#   adamck -> init -> gcheck -> train -> eval -> sample ->
#   BPE roundtrip -> kvcmp (bit-exact KV cache) ->
#   DNA specialist chain (growpro/trainf/spext/spchainsamp) -> looped core (OLLMCK03)
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
cat "$here/README.md" "$here/docs/LOOP.md" "$here/docs/CONTRACT.md" > "$work/corpus.txt"

echo "=== 1. Adam self-check (adamck) ==="
"$BIN" adamck

echo "=== 2. Initialize tiny byte model (V=256, T=16, D=16, H=2, F=32, L=2, B=4) ==="
"$BIN" init "$work/m.ck" 256 16 16 2 32 2 4 1 20 5000

echo "=== 3. Finite-difference gradient check (gcheck) ==="
"$BIN" gcheck "$work/m.ck" "$work/corpus.txt"

echo "=== 4. Eval before training ==="
"$BIN" eval "$work/m.ck" "$work/corpus.txt" 8

echo "=== 5. Train 20 steps on CPU ==="
"$BIN" train "$work/m.ck" "$work/corpus.txt" 20 5

echo "=== 6. Eval after training ==="
"$BIN" eval "$work/m.ck" "$work/corpus.txt" 8

echo "=== 7. Sample 48 bytes ==="
"$BIN" sample "$work/m.ck" 42 48 800 "Omega "
echo ""

echo "=== 8. BPE fit / encode / decode byte-exact roundtrip (V=300) ==="
"$BIN" bpefit "$work/corpus.txt" "$work/m300.bin" 300
"$BIN" bpeenc "$work/m300.bin" 300 "$work/corpus.txt" "$work/corpus.tok"
"$BIN" bpedec "$work/m300.bin" "$work/corpus.tok" "$work/corpus.dec"
cmp "$work/corpus.txt" "$work/corpus.dec"
echo "BPE_ROUNDTRIP_OK"

echo "=== 9. Bit-exact KV-cache parity check (kvcmp) ==="
"$BIN" init "$work/bpe.ck" 300 16 16 2 32 2 4 1 20 5000 1
"$BIN" kvcmp "$work/bpe.ck" 8 "Omega" "$work/m300.bin"

echo "=== 10. DNA Merkle Specialist Chain (growpro -> trainf -> spext -> spchainsamp) ==="
"$BIN" growpro "$work/bpe.ck" "$work/spec.ck" 3 10 1000
"$BIN" trainf "$work/spec.ck" "$work/corpus.tok" 5 5 2
"$BIN" spext "$work/bpe.ck" "$work/spec.ck" "$work/skill.sp"
"$BIN" spchainsamp "$work/bpe.ck" 16 800 "Omega" 42 "$work/m300.bin" 1 "$work/skill.sp" >/dev/null
echo "DNA_SPEC_CHAIN_OK"

echo "=== 11. Looped-Core OLLMCK03 init & train (LP=1, LC=2, NP=2) ==="
"$BIN" init "$work/loop.ck" 300 16 16 2 32 4 4 1 10 5000 1 1 2 2
"$BIN" train "$work/loop.ck" "$work/corpus.tok" 2 2

echo "QUICKSTART_CPU_OK"
