#!/bin/sh
# Verify Omega Helix on your own laptop in ~30 seconds (macOS arm64 or Linux, CPU-only, zero GPU).
# Builds the native static binary, proves zero CUDA / dynamic libs, cross-checks against Python f64
# oracle.py, and trips two live negative controls (tampered OLLMSP01 rc=61 + oracle mutant).
set -eu

here=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$here/build/verify-30s}
mkdir -p "$OUT"

OS=$(uname -s)
ARCH=$(uname -m)
if [ "$OS" = Darwin ]; then
  TARGET=macos
  BIN="$OUT/ollm-macos"
elif [ "$ARCH" = x86_64 ] || [ "$ARCH" = amd64 ]; then
  TARGET=x86linux
  BIN="$OUT/ollm-x86linux"
else
  TARGET=aarch64
  BIN="$OUT/ollm"
fi

echo "=== 1. Build pure-Omega CPU engine ($TARGET) ==="
OUT="$OUT" TARGET="$TARGET" sh "$here/scripts/build.sh"

echo "=== 2. Binary & linkage proof (zero CUDA, zero external deps) ==="
file "$BIN"
if [ "$OS" = Darwin ]; then
  otool -L "$BIN"
else
  ldd "$BIN" 2>&1 || true
  if ldd "$BIN" 2>&1 | grep -qiE 'libc\.so|libcuda|libcublas'; then
    echo "ERROR: dynamic libc/CUDA dependency detected" >&2
    exit 1
  fi
fi

work=$(mktemp -d "${TMPDIR:-/tmp}/omega-helix-30s.XXXXXX")
trap 'rm -rf "$work"' EXIT

echo "=== 3. Tiny train/eval + independent Python f64 oracle parity ==="
python3 - "$work/raw.txt" <<'PY'
import sys
s = "Verify Omega Helix against independent IEEE-754 f64 arithmetic.\n<|endoftext|>\n" * 30
open(sys.argv[1], "w", encoding="utf-8").write(s)
PY
python3 "$here/tests/oracle.py" xprep --ollm "$BIN" --data "$work/raw.txt" --outdir "$work"
python3 "$here/tests/oracle.py" xinit --ollm "$BIN" --cfg small
python3 "$here/tests/oracle.py" xgrad --ollm "$BIN" --data "$work/ollm-train.bin" --cfg small
python3 "$here/tests/oracle.py" xstep --ollm "$BIN" --data "$work/ollm-train.bin" --cfg small
python3 "$here/tests/oracle.py" xeval --ollm "$BIN" --held "$work/ollm-held.bin" --cfg small

echo "=== 4. Negative controls (tampered OLLMSP01 rc=61 + OLLM_ORACLE_MUTANT) ==="
"$BIN" init "$work/base.ck" 256 16 16 2 32 2 2 42 4 3000 >/dev/null
"$BIN" growpro "$work/base.ck" "$work/spec.ck" 3 4 2000 >/dev/null
"$BIN" spext "$work/base.ck" "$work/spec.ck" "$work/skill.sp" >/dev/null
python3 - "$work/skill.sp" "$work/tampered.sp" <<'PY'
import sys
b = bytearray(open(sys.argv[1], "rb").read())
b[48] ^= 0x01
open(sys.argv[2], "wb").write(b)
PY
set +e
"$BIN" spapp "$work/base.ck" "$work/tampered.sp" "$work/out.ck" >/dev/null 2>&1
rc_sp=$?
OLLM_ORACLE_MUTANT=tiemaxa python3 "$here/tests/oracle.py" selftest-v1 >/dev/null 2>&1
rc_mut=$?
set -e
echo "Negative control 1 (1-bit tamper in OLLMSP01 .sp): rc=$rc_sp (expected 61)"
[ "$rc_sp" -eq 61 ] || { echo "ERROR: expected rc=61, got $rc_sp" >&2; exit 1; }
echo "Negative control 2 (OLLM_ORACLE_MUTANT=tiemaxa selftest-v1): rc=$rc_mut (fault injection caught)"
[ "$rc_mut" -ne 0 ] || { echo "ERROR: expected mutant check to fail" >&2; exit 1; }

echo "VERIFY_30S_PASS binary=$BIN sha256=$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$BIN")"
