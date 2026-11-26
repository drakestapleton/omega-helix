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

is_omega_ng() {
  cand=$1
  [ -n "$cand" ] && [ -x "$cand" ] || return 1
  hdr=$(printf '@ng front-ir\nfn main to int\nret 0\nend\n' | "$cand" 2>/dev/null | dd bs=1 count=6 2>/dev/null | tr -d '\0' || true)
  [ "$hdr" = "OSC1IR" ]
}

has_ng_backend() {
  if [ -n "${NG:-}" ]; then
    if is_omega_ng "$NG"; then return 0; fi
    return 1
  fi
  if is_omega_ng "$OUT/ng"; then
    return 0
  fi
  if [ -n "${OMEGA_MV:-}" ]; then
    [ -f "$OMEGA_MV/seed/ng/src/ir.om" ] && [ -f "$OMEGA_MV/seed/ng/src/a64.om" ]
    return $?
  fi
  for cand in \
    "$HOME/.local/bin/ng" \
    "$(command -v ng 2>/dev/null || true)"
  do
    if is_omega_ng "$cand"; then
      return 0
    fi
  done
  for root in \
    "$HOME/workspace/omega-mv-macos" \
    "$HOME/canon/omega-mv" \
    "$here/../omega-mv" \
    "$here/../omega-mv-macos"
  do
    if [ -f "$root/seed/ng/src/ir.om" ] && [ -f "$root/seed/ng/src/a64.om" ]; then
      return 0
    fi
  done
  return 1
}

if [ "$OS" = Darwin ]; then
  OUT="$OUT" TARGET=macos sh "$here/scripts/build.sh"
  BIN="$OUT/ollm-macos"
  if [ "$ARCH" = arm64 ] && has_ng_backend; then
    OUT="$OUT" TARGET=macos-metal sh "$here/scripts/build.sh"
    METAL_BIN="$OUT/ollm-metal"
  else
    METAL_BIN=""
  fi
  BT_HDR='target a64macos
om2'
  BT_POOL="lpool.om"
  BT_PLAT="mio.om"
  BT_REV="rev.om"
  BT_APU="apu_fabric_macos.om"
elif [ "$ARCH" = x86_64 ] || [ "$ARCH" = amd64 ]; then
  OUT="$OUT" TARGET=x86linux sh "$here/scripts/build.sh"
  BIN="$OUT/ollm-x86linux"
  BT_HDR='target x86linux
om2'
  BT_POOL="xpool.om"
  BT_PLAT="lio.om"
  BT_REV="norev.om"
  BT_APU="apu_fabric_linux.om apu_fabric_nogpu.om"
else
  OUT="$OUT" TARGET=aarch64 sh "$here/scripts/build.sh"
  OUT="$OUT" TARGET=gb10 sh "$here/scripts/build.sh"
  BIN="$OUT/ollm"
  BT_HDR='om2'
  BT_POOL="lpool.om"
  BT_PLAT="lio.om"
  BT_REV="rev.om"
  BT_APU="apu_fabric_linux.om apu_fabric_nogpu.om"
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
  for u in math.om data.om "$BT_POOL" model.om adam.om ckpt.om bpe.om spec.om "$BT_REV" arboris.om sp_solve.om spiral.om apu_weave.om omg.om pillars_cli.om cli.om "$BT_PLAT" noserve.om nogpu.om $BT_APU; do
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

echo "=== 11. Spiral-2048 Pure-Omega Razor MoE (OLLMSP02 spcryst W4/f64 -> spweval -> spsamp -> 1-bit tamper rc=61) ==="
"$BIN" spcryst "$work/base.ck" "$work/corpus.txt" "$work/s1_w4.sp2" 32 100 256 1 4
"$BIN" spcryst "$work/base.ck" "$work/corpus.txt" "$work/s2_f64.sp2" 32 100 256 2 64
"$BIN" spweval "$work/base.ck" "$work/corpus.txt" 2 4 "$work/s1_w4.sp2" "$work/s2_f64.sp2" > "$work/spweval.out"
cat "$work/spweval.out"
awk 'NR==1 && $1=="SPWEVAL" && $6 < $5 { ok=1 } END { exit ok ? 0 : 1 }' "$work/spweval.out"
"$BIN" spsamp "$work/base.ck" 42 16 2 0 "$work/s1_w4.sp2" "$work/s2_f64.sp2" > /dev/null
cp "$work/s1_w4.sp2" "$work/s1_tamper.sp2"
printf '\377' | dd of="$work/s1_tamper.sp2" bs=1 seek=80 count=1 conv=notrunc 2>/dev/null
t_rc=0
"$BIN" spweval "$work/base.ck" "$work/corpus.txt" 1 1 "$work/s1_tamper.sp2" >/dev/null 2>&1 || t_rc=$?
[ "$t_rc" -eq 61 ] || { echo "FAIL: expected rc=61 on 1-bit OLLMSP02 tamper, got $t_rc" >&2; exit 1; }
echo "SPIRAL_WEAVE_OM_PASS"

echo "=== 12. Full-APU Simultaneous Co-Execution Fabric (apuweave -> W4 OLLMSP02 -> spweval + 1-bit tamper rc=61) ==="
mkdir -p "$work/apu_shards" "$work/apu_out"
cp "$work/corpus.txt" "$work/apu_shards/s01_alpha.tok"
cp "$work/corpus.txt" "$work/apu_shards/s02_beta.tok"
printf '%s\n%s\n' "$work/apu_shards/s01_alpha.tok" "$work/apu_shards/s02_beta.tok" > "$work/apu_list.txt"
"$BIN" apuweave "$work/base.ck" "$work/apu_list.txt" "$work/apu_out" 32 100 128 4 2 > "$work/apuweave.out"
cat "$work/apuweave.out"
awk '$1=="APU_SUMMARY" && $2==2 && $13==2 { ok=1 } END { exit ok ? 0 : 1 }' "$work/apuweave.out"
"$BIN" spweval "$work/base.ck" "$work/corpus.txt" 2 4 "$work/apu_out/s01_alpha.sp2" "$work/apu_out/s02_beta.sp2" > "$work/apu_spweval.out"
cat "$work/apu_spweval.out"
awk 'NR==1 && $1=="SPWEVAL" && $6 < $5 { ok=1 } END { exit ok ? 0 : 1 }' "$work/apu_spweval.out"
echo "APU_WEAVE_OM_PASS"

echo "--- 13. pure-Omega serve.om scaling build + live /health check ---"
if [ "$(uname -s)" = "Linux" ] && [ "$ARCH" = "aarch64" ]; then
  SRV_PORT="${OLLM_SERVE_PORT:-18292}"
  printf 'sovereign verified deterministic compute builds clean kernels.\n' > "$work/serve_corpus.txt"
  i=0
  while [ "$i" -lt 20 ]; do
    printf 'alpha beta gamma delta epsilon sovereign compute kernel %d.\n' "$i" >> "$work/serve_corpus.txt"
    i=$((i + 1))
  done
  "$BIN" bpefit "$work/serve_corpus.txt" "$work/serve_merges.bin" 300
  "$BIN" init "$work/serve_base.ck" 300 16 16 2 32 2 4 1 10 4000
  "$BIN" serve "$SRV_PORT" "$work/serve_base.ck" "$work/serve_merges.bin" slots=4 ctx=8192 kvblks=128 >"$work/serve.log" 2>&1 &
  SRV_PID=$!
  trap 'kill "$SRV_PID" 2>/dev/null || true; wait "$SRV_PID" 2>/dev/null || true; rm -rf "$work"' EXIT
  ready=0
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    if grep -q "^PORT $SRV_PORT$" "$work/serve.log" 2>/dev/null; then
      ready=1
      break
    fi
    sleep 0.2
  done
  [ "$ready" = 1 ]
  if command -v curl >/dev/null 2>&1; then
    curl -fsS "http://127.0.0.1:$SRV_PORT/health" >"$work/health.json"
  elif command -v wget >/dev/null 2>&1; then
    wget -qO "$work/health.json" "http://127.0.0.1:$SRV_PORT/health"
  fi
  cat "$work/health.json"
  echo ""
  grep -q '"max_slots":4' "$work/health.json"
  grep -q '"max_ctx":8192' "$work/health.json"
  grep -q '"kv_cache":' "$work/health.json"
  kill "$SRV_PID" 2>/dev/null || true
  wait "$SRV_PID" 2>/dev/null || true
  SRV_PID=""
  trap 'rm -rf "$work"' EXIT
else
  OUT="$work/a64" TARGET=aarch64 sh "$here/scripts/build.sh"
  test -f "$work/a64/ollm"
fi
echo "SERVE_SCALING_OM_PASS"

if [ -n "${METAL_BIN:-}" ] && [ -x "${METAL_BIN:-}" ]; then
  echo "=== 14. Apple Silicon Metal GPU Pure-Omega Gate (mdev -> mops -> gcmp) ==="
  mdev_rc=0
  "$METAL_BIN" mdev || mdev_rc=$?
  if [ "$mdev_rc" -eq 0 ]; then
    "$METAL_BIN" mops
    "$METAL_BIN" gcmp "$work/base.ck" "$work/corpus.txt" 0
    echo "METAL_GPU_OM_PASS"
  elif [ "$mdev_rc" -eq 75 ]; then
    echo "METAL_GPU_SKIP (no Metal GPU device on headless macOS host)"
  else
    echo "FAIL: ollm-metal mdev exited $mdev_rc" >&2
    exit "$mdev_rc"
  fi
fi

echo "OMEGA_HELIX_PURE_OM_PASS"
