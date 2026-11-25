#!/bin/sh
# 15-Second Unified 8-Pillar Pure-Omega Demo (Zero GPU, Zero Python, Zero CUDA).
# Runs: OMG V=256 pack -> neural train -> OMG-masked decode -> Arboris 3-layer guided
# decode (bypass + OMG + LTP0) -> closed-form spforge -> reversible Q32 train (ARM64)
# and compiles every generated .om program with seeded omc.
set -eu

here=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$here/build/demo-8-pillars}
mkdir -p "$OUT"

OS=$(uname -s)
ARCH=$(uname -m)
OMC=${OMC:-$(command -v omc 2>/dev/null || printf '%s' "$HOME/.local/bin/omc")}
if [ ! -x "$OMC" ] && [ -x "$HOME/canon/omega-mv/seed/out/omc" ]; then
  OMC="$HOME/canon/omega-mv/seed/out/omc"
fi

if [ "$OS" = Darwin ]; then
  OUT="$OUT" TARGET=macos sh "$here/scripts/build.sh"
  BIN="$OUT/ollm-macos"
  OMC_HDR="target a64macos
"
elif [ "$ARCH" = x86_64 ] || [ "$ARCH" = amd64 ]; then
  OUT="$OUT" TARGET=x86linux sh "$here/scripts/build.sh"
  BIN="$OUT/ollm-x86linux"
  OMC_HDR=""
else
  OUT="$OUT" TARGET=aarch64 sh "$here/scripts/build.sh"
  BIN="$OUT/ollm"
  OMC_HDR=""
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/omega-helix-8p-demo.XXXXXX")
trap 'rm -rf "$work"' EXIT

echo "=== 1. Pillar 8 (OMG): Pack .om source -> V=256 canonical graph tokens ==="
cat > "$work/seed.om" <<'EOF'
om2
fn scale x ulong k ulong to ulong
  res = mul x k
  res = add res 1
  ret res
end
fn main to int
  a = ulong 7
  b = scale a 3
  if gt b 10
    ret int sub b 10
  end
  ret 0
end
EOF
"$BIN" omg pack "$work/seed.om" "$work/seed.tok"
cat "$work/seed.tok" "$work/seed.tok" "$work/seed.tok" "$work/seed.tok" > "$work/corpus.tok"

echo "=== 2. Train tiny V=256 Helix trunk on OMG tokens ==="
"$BIN" init "$work/base.ck" 256 16 16 2 32 2 4 1 10 4000
"$BIN" train "$work/base.ck" "$work/corpus.tok" 5 5

echo "=== 3. Pillar 8: Neural OMG-masked decode (100% omc-valid .om) ==="
"$BIN" omg samp "$work/base.ck" 42 64 0 "$work/samp.om"
{ printf '%s' "$OMC_HDR"; cat "$work/samp.om"; } | "$OMC" > "$work/samp.bin"
echo "OMGSAMP_COMPILED_OK ($(wc -c < "$work/samp.bin" | tr -d ' ') bytes)"

echo "=== 4. Pillar 5 + 8: Arboris 3-layer guided decode (Bypass + OMG wall + LTP0) ==="
"$BIN" arboris guide "$work/base.ck" "$work/corpus.tok" 42 64 0 "$work/guide.om"
{ printf '%s' "$OMC_HDR"; cat "$work/guide.om"; } | "$OMC" > "$work/guide.bin"
echo "ARBGUIDE_COMPILED_OK ($(wc -c < "$work/guide.bin" | tr -d ' ') bytes)"

echo "=== 5. Pillar 7: Closed-Form Ridge/Gauss-Newton Specialist Forge (spforge) ==="
cat "$here/README.md" "$here/docs/LOOP.md" > "$work/text.txt"
"$BIN" fcache "$work/base.ck" "$work/text.txt" 8 "$work/fc.hc"
"$BIN" spforge "$work/base.ck" "$work/fc.hc" "$work/closed.sp" 10000 1
"$BIN" spapp "$work/base.ck" "$work/closed.sp" "$work/spec.ck"
"$BIN" speval "$work/base.ck" "$work/closed.sp" "$work/text.txt" 8

if [ "$OS" = Darwin ] || [ "$ARCH" = aarch64 ]; then
  echo "=== 6. Pillar 2 + Lock 4: Reversible Q32 zero-activation training ==="
  "$BIN" rev train 8 10
fi

echo "DEMO_8_PILLARS_PASS"
