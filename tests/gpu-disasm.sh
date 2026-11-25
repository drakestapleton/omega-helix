#!/bin/sh
# Disassemble the GB10 kernels that native/gpu/ emits, with nvdisasm as a read-only
# oracle. Builds the gb10 sources plus tests/gpu-dump.om, runs the dump on the CPU (no GPU),
# cuts slot N (16 KiB at N * 16384; two slots for 64x64 GEMM ids 40-58), trims trailing zero
# bytes to a 16-byte multiple, disassembles with `nvdisasm -b SM121`, checks for illegal
# instructions, and verifies declared GPR count gkgpr >= max_gpr + 3 (2 hardware-reserved
# PC GPRs). Any LOW or illegal instruction exits 4.
#
# Usage:
#   sh tests/gpu-disasm.sh --all         # check all 43 non-empty SM121 kernel slots
#   sh tests/gpu-disasm.sh SLOT...       # check specific slot ids (e.g. 0 27 40)
#
# Env:
#   OMC        Path to seeded omc (auto-detected if unset)
#   OMC_PIN    Optional expected sha256 of OMC (enforced only when set)
#   OUT        Scratch/output dir (default: ./build/gpu-disasm)
#   NVDISASM   Path to nvdisasm (default: /usr/local/cuda/bin/nvdisasm or PATH)
#   SHAPE      grset V T D H F B (default: "256 32 64 2 128 2")
#   VERBOSE    Set to 1 to print full SASS listings (default: 0 for --all, 1 for explicit slots)
set -eu

here=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$here/build/gpu-disasm}
SHAPE=${SHAPE:-256 32 64 2 128 2}
NVDISASM=${NVDISASM:-$(command -v nvdisasm 2>/dev/null || printf '/usr/local/cuda/bin/nvdisasm')}

find_omc() {
  if [ -n "${OMC:-}" ] && [ -x "$OMC" ]; then
    printf '%s\n' "$OMC"
    return 0
  fi
  for cand in \
    "$(command -v omc 2>/dev/null || true)" \
    "$HOME/.local/bin/omc" \
    "$HOME/canon/omega-mv/seed/out/omc" \
    "$here/../omega-mv/seed/out/omc"
  do
    if [ -n "$cand" ] && [ -x "$cand" ]; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  echo "error: seeded 'omc' compiler not found (set OMC=/path/to/omc)" >&2
  exit 1
}

[ $# -ge 1 ] || { echo "usage: gpu-disasm.sh --all | SLOT..." >&2; exit 2; }
[ -x "$NVDISASM" ] || { echo "error: nvdisasm not found at '$NVDISASM'" >&2; exit 3; }

OMC=$(find_omc)
if [ -n "${OMC_PIN:-}" ]; then
  got=$(sha256sum "$OMC" | cut -d' ' -f1)
  [ "$got" = "$OMC_PIN" ] || { echo "OMC_PIN_MISMATCH $got" >&2; exit 3; }
fi

if [ "$1" = "--all" ]; then
  VERBOSE=${VERBOSE:-0}
  set -- 0 1 2 3 4 5 6 7 9 10 11 13 14 15 16 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 40 42 44 46 48 50 52 54 56 58
else
  VERBOSE=${VERBOSE:-1}
fi

mkdir -p "$OUT"
src=$here/native
g=$src/gpu

{
  printf 'om2\n'
  cat "$src/math.om" "$src/data.om" "$src/lpool.om" "$src/model.om" "$src/adam.om" "$src/ckpt.om" "$src/bpe.om" "$src/spec.om" "$src/rev.om" "$src/arboris.om" "$src/sp_solve.om" "$src/omg.om" "$src/pillars_cli.om" "$src/cli.om" "$src/lio.om" "$src/noserve.om" |
    sed 's/^fn main to int$/fn ocpumain to int/'
  cat "$g/context.om" "$g/vector.om" "$g/launch.om" "$g/batch.om" "$g/gk.om" "$g/gr.om" "$g/gb.om" "$g/gg.om" "$g/gt.om" "$g/gm.om" "$g/ring.om" "$g/ejit.om"
  sed "s/^grset 256 32 64 2 128 2\$/grset $SHAPE/" "$here/tests/gpu-dump.om"
} > "$OUT/gpu-dump.om"
grep -qx "grset $SHAPE" "$OUT/gpu-dump.om" || { echo "SHAPE_NOT_SET $SHAPE" >&2; exit 2; }

"$OMC" < "$OUT/gpu-dump.om" > "$OUT/gpu-dump.tmp"
chmod +x "$OUT/gpu-dump.tmp"
mv -f "$OUT/gpu-dump.tmp" "$OUT/gpu-dump"

"$OUT/gpu-dump" > "$OUT/gpu-code.bin" 2> "$OUT/gpu-gpr.bin"
[ "$(wc -c < "$OUT/gpu-gpr.bin")" -eq 128 ] || { echo "GPR_TABLE_SIZE $(wc -c < "$OUT/gpu-gpr.bin")" >&2; exit 2; }
echo "DUMP $(sha256sum "$OUT/gpu-dump" | cut -d' ' -f1) source_sha256 $(sha256sum "$OUT/gpu-dump.om" | cut -d' ' -f1) code_sha256 $(sha256sum "$OUT/gpu-code.bin" | cut -d' ' -f1)"

bad=0
ok_slots=0
for s in "$@"; do
  ns=1
  [ "$s" -ge 40 ] && [ "$s" -le 58 ] && ns=2
  dd if="$OUT/gpu-code.bin" of="$OUT/gpu-slot$s.bin" bs=16384 skip="$s" count="$ns" status=none
  last=$(od -An -v -tu1 "$OUT/gpu-slot$s.bin" | tr -s ' ' '\n' | awk 'NF { n++; if ($1 != 0) l = n } END { print l + 0 }')
  n=$(( (last + 15) / 16 * 16 ))
  head -c "$n" "$OUT/gpu-slot$s.bin" > "$OUT/gpu-slot$s.t.bin"
  [ "$VERBOSE" = 1 ] && echo "SLOT $s bytes $n sha256 $(sha256sum "$OUT/gpu-slot$s.t.bin" | cut -d' ' -f1)"
  [ "$n" -eq 0 ] && { echo "SLOT $s empty"; continue; }
  "$NVDISASM" -b SM121 "$OUT/gpu-slot$s.t.bin" > "$OUT/gpu-slot$s.sass"
  [ "$VERBOSE" = 1 ] && cat "$OUT/gpu-slot$s.sass"
  if grep -qi 'illegal\|unknown' "$OUT/gpu-slot$s.sass"; then
    echo "SLOT $s ILLEGAL_INSN" >&2
    bad=1
    continue
  fi
  d=$(od -An -tu1 -j "$s" -N1 "$OUT/gpu-gpr.bin" | tr -d ' ')
  awk -v s="$s" -v d="$d" -v v="$VERBOSE" '
    function scan(x, ext,   t, e, v2) {
      x = " " x
      while (match(x, /[^A-Za-z0-9_.]R[0-9]+(\.64)?/)) {
        t = substr(x, RSTART + 2, RLENGTH - 2); e = ext
        if (t ~ /\.64$/) { e = 1; sub(/\.64$/, "", t) }
        v2 = t + e; if (v2 > mx) mx = v2
        x = substr(x, RSTART + RLENGTH)
      }
    }
    BEGIN { mx = -1 }
    match($0, /\/\*[0-9a-f]+\*\//) {
      r = substr($0, RSTART + RLENGTH); sub(/;.*/, "", r); sub(/^[ \t]+/, "", r)
      if (r ~ /^@/) sub(/^[^ \t]+[ \t]+/, "", r)
      op = r; sub(/[ \t].*/, "", op); sub(/^[^ \t]+/, "", r)
      w = 1; if (op ~ /\.128/) w = 4; else if (op ~ /\.64|\.WIDE/) w = 2
      while (match(r, /\[[^]]*\]/)) {
        p = RSTART; l = RLENGTH
        scan(substr(r, p, l), 0)
        r = substr(r, 1, p - 1) " " substr(r, p + l)
      }
      scan(r, w - 1)
    }
    END {
      k = mx + 3
      if (v == 1 || d < k) printf "GPR %s max R%d need %d declared %d %s\n", s, mx, k, d, (d >= k ? "OK" : "LOW")
      exit (d >= k ? 0 : 4)
    }
  ' "$OUT/gpu-slot$s.sass" && ok_slots=$((ok_slots + 1)) || bad=1
  [ "$VERBOSE" = 1 ] && echo "SMEM $s target $(od -An -tu1 -j $((64 + s)) -N1 "$OUT/gpu-gpr.bin" | tr -d ' ')"
done
echo "GPU_DISASM_OK slots=$ok_slots bad=$bad"
[ "$bad" -eq 0 ] || { echo "GPU_DISASM_FAIL" >&2; exit 4; }
