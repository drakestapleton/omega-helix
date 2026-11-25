#!/bin/sh
# Pure-Omega, zero-Python verification gate for Omega Helix.
# Builds the static pure-Omega binaries with `omc` and runs the full CPU smoke
# suite (adamck, init, gcheck, train, eval, sample, BPE, kvcmp, DNA, loop) plus
# the new native pure-Omega pillar self-tests (omg, arboris, rev, fcachem->spforge).
set -eu

here=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$here/build/verify-om}
mkdir -p "$OUT"

OS=$(uname -s)
ARCH=$(uname -m)
echo "=== Omega Helix Pure-Omega Zero-Python Gate ($OS $ARCH) ==="

if [ "$OS" = Darwin ]; then
  OUT="$OUT" TARGET=macos sh "$here/scripts/build.sh"
  BIN="$OUT/ollm-macos"
  BT_HDR='target a64macos
om2'
  BT_PLAT="mio.om"
  BT_REV="rev.om"
elif [ "$ARCH" = x86_64 ] || [ "$ARCH" = amd64 ]; then
  OUT="$OUT" TARGET=x86linux sh "$here/scripts/build.sh"
  BIN="$OUT/ollm-x86linux"
  BT_HDR='target x86linux
om2'
  BT_PLAT="lio.om"
  BT_REV="norev.om"
else
  OUT="$OUT" TARGET=aarch64 sh "$here/scripts/build.sh"
  OUT="$OUT" TARGET=gb10 sh "$here/scripts/build.sh"
  BIN="$OUT/ollm"
  BT_HDR='om2'
  BT_PLAT="lio.om"
  BT_REV="rev.om"
fi

OMC=${OMC:-$(command -v omc 2>/dev/null || printf '%s' "$HOME/.local/bin/omc")}
if [ ! -x "$OMC" ] && [ -x "$HOME/canon/omega-mv/seed/out/omc" ]; then
  OMC="$HOME/canon/omega-mv/seed/out/omc"
fi

sh "$here/scripts/quickstart-cpu.sh" "$BIN"

echo "=== 9. Pure-Omega 8-Pillar + 4 Byte-Traceability Locks (zero Python) ==="
"$BIN" omg
"$BIN" arboris
if [ "$OS" = Darwin ] || [ "$ARCH" = aarch64 ]; then
  "$BIN" rev selftest
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/omega-helix-omgate.XXXXXX")
trap 'rm -rf "$work"' EXIT

cat > "$work/sim_main.om" <<'EOF'
global gkcode uchar 1048576
global gkthr uint 64
global gkgpr uint 64
global gksm uint 64
global gkarg uint 32
global gkna uint 1
global gkt long 4

fn gklaunch id uint gx uint gy uint gz uint to bool
put gkt 0 add get gkt 0 1
ret true
end
EOF

cat > "$work/bt_driver.om" <<'EOF'
fn main to int
r1 = gm_selftest
if ne r1 0
  oputc 76
  oputc 49
  oputc 32
  oputl long r1
  oputc 10
  oflush
  ret r1
end
r2 = grg_epoch_selftest
if ne r2 0
  oputc 76
  oputc 50
  oputc 32
  oputl long r2
  oputc 10
  oflush
  ret r2
end
r3 = ejit_selftest
if ne r3 0
  oputc 76
  oputc 51
  oputc 32
  oputl long r3
  oputc 10
  oflush
  ret r3
end
oputc 66
oputc 84
oputc 76
oputc 79
oputc 67
oputc 75
oputc 95
oputc 79
oputc 75
oputc 10
oflush
ret 0
end
EOF

src=$here/native
{
  printf '%s\n' "$BT_HDR"
  for u in math.om data.om lpool.om model.om adam.om ckpt.om bpe.om spec.om "$BT_REV" arboris.om sp_solve.om omg.om cli.om "$BT_PLAT" noserve.om nogpu.om; do
    sed 's/^fn main to int$/fn ocpumain to int/' "$src/$u"
  done
  cat "$work/sim_main.om" "$src/gpu/gm.om" "$src/gpu/ring.om" "$src/gpu/ejit.om" "$work/bt_driver.om"
} > "$work/btlock.om"
"$OMC" < "$work/btlock.om" > "$work/btlock"
chmod +x "$work/btlock"
if [ "$OS" = Darwin ] && command -v codesign >/dev/null 2>&1; then
  codesign -s - -f "$work/btlock" >/dev/null 2>&1
fi
"$work/btlock"
cat "$here/README.md" "$here/docs/LOOP.md" > "$work/corpus.txt"
"$BIN" init "$work/base.ck" 256 16 16 2 32 2 4 1 10 4000
"$BIN" train "$work/base.ck" "$work/corpus.txt" 5 5
"$BIN" fcache "$work/base.ck" "$work/corpus.txt" 8 "$work/fc.hc"
"$BIN" spforge "$work/base.ck" "$work/fc.hc" "$work/closed.sp" 10000 1
"$BIN" spapp "$work/base.ck" "$work/closed.sp" "$work/spec.ck"
"$BIN" speval "$work/base.ck" "$work/closed.sp" "$work/corpus.txt" 8

echo "=== 10. Unified 8-Pillar Loop (omg pack/decode/samp -> omc, arboris guide -> omc, rev train) ==="
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
"$BIN" omg decode "$work/seed.tok" "$work/dec.om"
cat "$work/seed.tok" "$work/seed.tok" "$work/seed.tok" "$work/seed.tok" > "$work/omg_corpus.tok"
"$BIN" init "$work/omg.ck" 256 16 16 2 32 2 4 1 10 4000
"$BIN" train "$work/omg.ck" "$work/omg_corpus.tok" 5 5
"$BIN" omg samp "$work/omg.ck" 42 64 0 "$work/samp.om"
"$BIN" arboris guide "$work/omg.ck" "$work/omg_corpus.tok" 42 64 0 "$work/guide.om"
OMC_HDR=""
if [ "$OS" = Darwin ]; then
  OMC_HDR="target a64macos
"
fi
for omf in "$work/dec.om" "$work/samp.om" "$work/guide.om"; do
  { printf '%s' "$OMC_HDR"; cat "$omf"; } | "$OMC" > "$omf.bin"
done
if [ "$OS" = Darwin ] || [ "$ARCH" = aarch64 ]; then
  "$BIN" rev train 8 10
fi

echo "OMEGA_HELIX_PURE_OM_PASS"
