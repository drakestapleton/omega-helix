#!/bin/sh
# Build Omega Helix (ollm) binaries from pure seeded Omega (.om) sources.
#
# Usage:
#   sh scripts/build.sh                 # default: TARGET=gb10 (Linux AArch64 + GB10 SM121 GPU)
#   TARGET=aarch64 sh scripts/build.sh  # Linux AArch64 CPU + OpenAI HTTP server
#   TARGET=macos   sh scripts/build.sh  # Apple Silicon macOS Mach-O CPU binary
#   TARGET=x86linux sh scripts/build.sh # Linux x86-64 CPU binary
#   TARGET=x86efi  sh scripts/build.sh  # Bare-metal x86-64 UEFI application
#   TARGET=a64efi  sh scripts/build.sh  # Bare-metal AArch64 UEFI application
#   TARGET=tok     sh scripts/build.sh  # Standalone pure-Omega BPE tokenizer (ollm-tok)
#   TARGET=wtrain  sh scripts/build.sh  # Pure-Omega fine-tune & eval driver (wtrain)
#
# Env:
#   OMC   Path to seeded omc compiler (auto-detected if unset)
#   OUT   Output directory (default: ./build)
set -eu

here=$(cd "$(dirname "$0")/.." && pwd)
TARGET=${TARGET:-gb10}
OUT=${OUT:-$here/build}
src=$here/native

find_omc() {
  if [ -n "${OMC:-}" ] && [ -x "$OMC" ]; then
    printf '%s\n' "$OMC"
    return 0
  fi
  for cand in \
    "$(command -v omc 2>/dev/null || true)" \
    "$HOME/.local/bin/omc" \
    "$here/../omega-mv/seed/out/omc" \
    "$here/../omega-mv-macos/seed/out/omc"
  do
    if [ -n "$cand" ] && [ -x "$cand" ]; then
      printf '%s\n' "$cand"
      return 0
    fi
  done
  echo "error: seeded 'omc' compiler not found." >&2
  echo "  Install omc from https://interplane.io/omega/ or set OMC=/path/to/omc" >&2
  exit 1
}

OMC=$(find_omc)
mkdir -p "$OUT"

if [ "$TARGET" = tok ]; then
  cat "$src/tok/tkcore.om" "$src/tok/tkfit.om" "$src/tok/tkio.om" "$src/tok/tkcli.om" > "$OUT/ollm-tok.om"
  "$OMC" < "$OUT/ollm-tok.om" > "$OUT/ollm-tok.tmp"
  chmod +x "$OUT/ollm-tok.tmp"
  mv -f "$OUT/ollm-tok.tmp" "$OUT/ollm-tok"
  printf 'built %s (%s bytes)\n' "$OUT/ollm-tok" "$(wc -c < "$OUT/ollm-tok" | tr -d ' ')"
  exit 0
fi

if [ "$TARGET" = wtrain ]; then
  drv=$here/tools/wdrive
  { sed 's/^fn main to int$/fn wbmain to int/' "$drv/wbuild.om"; cat "$drv/wtrain.om"; } > "$OUT/wtrain.om"
  "$OMC" < "$OUT/wtrain.om" > "$OUT/wtrain.tmp"
  chmod +x "$OUT/wtrain.tmp"
  mv -f "$OUT/wtrain.tmp" "$OUT/wtrain"
  printf 'built %s (%s bytes)\n' "$OUT/wtrain" "$(wc -c < "$OUT/wtrain" | tr -d ' ')"
  exit 0
fi

lead=
pre=
pool=lpool.om
plat=lio.om
post=$src/noserve.om
bin=ollm

case "$TARGET" in
  aarch64)
    lead=om2
    pre="$src/net/http.om $src/net/json.om $src/net/jsonw.om"
    post=$src/serve.om
    bin=ollm
    ;;
  gb10)
    lead=om2
    pre="$src/net/http.om $src/net/json.om $src/net/jsonw.om"
    post=$src/serve.om
    bin=ollm-gb10
    ;;
  macos)
    lead='target a64macos'
    plat=mio.om
    post=$src/noserve.om
    bin=ollm-macos
    ;;
  x86linux)
    lead='target x86linux'
    plat=lio.om
    post=$src/noserve.om
    bin=ollm-x86linux
    ;;
  x86efi)
    lead='target x86efi'
    pool=epool.om
    plat=eio.om
    post=$src/noserve.om
    bin=ollm.efi
    ;;
  a64efi)
    lead='target a64efi'
    pool=epool.om
    plat=eio.om
    post=$src/noserve.om
    bin=ollm-aa64.efi
    EFIRT=${EFIRT:-$(dirname "$OMC")/../a64efi/efirt.om}
    if [ ! -f "$EFIRT" ]; then
      echo "error: a64efi requires efirt.om (set EFIRT=/path/to/efirt.om)" >&2
      exit 1
    fi
    ;;
  *)
    echo "error: unknown TARGET '$TARGET' (expected: gb10, aarch64, macos, x86linux, x86efi, a64efi, tok, wtrain)" >&2
    exit 2
    ;;
esac

om_file="$OUT/$bin.om"
{
  [ -z "$lead" ] || printf '%s\n' "$lead"
  [ "$TARGET" != a64efi ] || cat "$EFIRT"
  # shellcheck disable=SC2086
  [ -z "$pre" ] || cat $pre
  cat \
    "$src/math.om" \
    "$src/data.om" \
    "$src/$pool" \
    "$src/model.om" \
    "$src/adam.om" \
    "$src/ckpt.om" \
    "$src/bpe.om" \
    "$src/spec.om" \
    "$src/cli.om" \
    "$src/$plat" \
    "$post"
  [ "$TARGET" = gb10 ] || cat "$src/nogpu.om"
} > "$om_file"

if [ "$TARGET" = gb10 ]; then
  g=$src/gpu
  cat \
    "$g/context.om" \
    "$g/vector.om" \
    "$g/launch.om" \
    "$g/batch.om" \
    "$g/gk.om" \
    "$g/gr.om" \
    "$g/gb.om" \
    "$g/gg.om" >> "$om_file"
fi

"$OMC" < "$om_file" > "$OUT/$bin.tmp"
chmod +x "$OUT/$bin.tmp"
mv -f "$OUT/$bin.tmp" "$OUT/$bin"

if [ "$TARGET" = macos ] && command -v codesign >/dev/null 2>&1; then
  codesign -s - -f "$OUT/$bin" >/dev/null 2>&1
fi

printf 'built %s (%s bytes, target=%s, omc=%s)\n' \
  "$OUT/$bin" "$(wc -c < "$OUT/$bin" | tr -d ' ')" "$TARGET" "$OMC"
