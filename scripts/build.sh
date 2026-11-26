#!/bin/sh
# Build Omega Helix (ollm) binaries from pure seeded Omega (.om) sources.
#
# Usage:
#   sh scripts/build.sh                     # default: TARGET=gb10 (Linux AArch64 + GB10 SM121 GPU)
#   TARGET=aarch64     sh scripts/build.sh  # Linux AArch64 CPU + OpenAI HTTP server
#   TARGET=macos       sh scripts/build.sh  # Apple Silicon macOS Mach-O CPU binary
#   TARGET=macos-metal sh scripts/build.sh  # Apple Silicon macOS Mach-O + Metal GPU binary
#   TARGET=x86linux    sh scripts/build.sh  # Linux x86-64 CPU binary
#   TARGET=x86efi      sh scripts/build.sh  # Bare-metal x86-64 UEFI application
#   TARGET=a64efi      sh scripts/build.sh  # Bare-metal AArch64 UEFI application
#   TARGET=tok         sh scripts/build.sh  # Standalone pure-Omega BPE tokenizer (ollm-tok)
#   TARGET=wtrain      sh scripts/build.sh  # Pure-Omega fine-tune & eval driver (wtrain)
#   TARGET=openwaldo   sh scripts/build.sh  # Pure-Omega OpenWALDO JSONL ingestor (ollm-waldo)
#   TARGET=univ        sh scripts/build.sh  # Pure-Omega Universal zero-cap LLM engine (ollm-univ)
#   TARGET=interchange sh scripts/build.sh  # Pure-Omega Safetensors/HF/GGUF converter (ollm-interchange)
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
    "$HOME/canon/omega-mv/seed/out/omc" \
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

host_target_hdr() {
  case "$(uname -s)/$(uname -m)" in
    Darwin/arm64) printf 'target a64macos\n' ;;
    Linux/x86_64|Linux/amd64) printf 'target x86linux\n' ;;
  esac
}

is_omega_ng() {
  cand=$1
  [ -n "$cand" ] && [ -x "$cand" ] || return 1
  hdr=$(printf '@ng front-ir\nfn main to int\nret 0\nend\n' | "$cand" 2>/dev/null | dd bs=1 count=6 2>/dev/null | tr -d '\0' || true)
  [ "$hdr" = "OSC1IR" ]
}

build_ng_from_src() {
  ng_src=$1
  {
    host_target_hdr
    cat "$ng_src/ir.om" "$ng_src/front.om" "$ng_src/opt.om" "$ng_src/a64.om" "$ng_src/main.om"
  } > "$OUT/ng.om"
  "$OMC" < "$OUT/ng.om" > "$OUT/ng.tmp"
  chmod +x "$OUT/ng.tmp"
  mv -f "$OUT/ng.tmp" "$OUT/ng"
  if [ "$(uname -s)" = Darwin ] && command -v codesign >/dev/null 2>&1; then
    codesign -s - -f "$OUT/ng" >/dev/null 2>&1
  fi
  printf '%s\n' "$OUT/ng"
}

find_ng() {
  if [ -n "${NG:-}" ]; then
    if is_omega_ng "$NG"; then
      printf '%s\n' "$NG"
      return 0
    fi
    echo "error: NG='$NG' is not an executable Omega ng compiler." >&2
    exit 1
  fi
  if is_omega_ng "$OUT/ng"; then
    printf '%s\n' "$OUT/ng"
    return 0
  fi
  if [ -n "${OMEGA_MV:-}" ]; then
    if [ -f "$OMEGA_MV/seed/ng/src/ir.om" ] && [ -f "$OMEGA_MV/seed/ng/src/a64.om" ]; then
      build_ng_from_src "$OMEGA_MV/seed/ng/src"
      return 0
    fi
    echo "error: OMEGA_MV='$OMEGA_MV' missing seed/ng/src." >&2
    exit 1
  fi
  for cand in \
    "$HOME/.local/bin/ng" \
    "$(command -v ng 2>/dev/null || true)"
  do
    if is_omega_ng "$cand"; then
      printf '%s\n' "$cand"
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
      build_ng_from_src "$root/seed/ng/src"
      return 0
    fi
  done
  echo "error: 'ng' compiler or omega-mv seed/ng/src not found (set NG=/path/to/ng or OMEGA_MV=/path/to/omega-mv)." >&2
  exit 1
}

if [ "$TARGET" = macos-metal ]; then
  NG=$(find_ng)
  om_file="$OUT/ollm-metal.om"
  {
    printf 'target macos\nom2\n'
    cat \
      "$src/math.om" \
      "$src/data.om" \
      "$src/gpu/mpool.om" \
      "$src/model.om" \
      "$src/adam.om" \
      "$src/ckpt.om" \
      "$src/bpe.om" \
      "$src/spec.om" \
      "$src/norev.om" \
      "$src/cli.om" \
      "$src/noserve.om" \
      "$src/gpu/metal_io.om" \
      "$src/gpu/metal_msl.om" \
      "$src/gpu/metal_macos.om"
  } > "$om_file"
  { printf '@ng front-ir\n'; cat "$om_file"; } | "$NG" > "$OUT/ollm-metal.ir"
  { printf '@ng opt\n'; cat "$OUT/ollm-metal.ir"; } | "$NG" > "$OUT/ollm-metal.opt.ir"
  { printf '@ng a64 macho\n'; cat "$OUT/ollm-metal.opt.ir"; } | "$NG" > "$OUT/ollm-metal.tmp"
  chmod +x "$OUT/ollm-metal.tmp"
  mv -f "$OUT/ollm-metal.tmp" "$OUT/ollm-metal"
  if [ "$(uname -s)" = Darwin ] && command -v codesign >/dev/null 2>&1; then
    codesign -s - -f "$OUT/ollm-metal" >/dev/null 2>&1
  fi
  printf 'built %s (%s bytes, target=%s, ng=%s)\n' \
    "$OUT/ollm-metal" "$(wc -c < "$OUT/ollm-metal" | tr -d ' ')" "$TARGET" "$NG"
  exit 0
fi

if [ "$TARGET" = tok ]; then
  {
    host_target_hdr
    cat "$src/tok/tkcore.om" "$src/tok/tkfit.om" "$src/tok/tkio.om" "$src/tok/tkcli.om"
  } > "$OUT/ollm-tok.om"
  "$OMC" < "$OUT/ollm-tok.om" > "$OUT/ollm-tok.tmp"
  chmod +x "$OUT/ollm-tok.tmp"
  mv -f "$OUT/ollm-tok.tmp" "$OUT/ollm-tok"
  if [ "$(uname -s)" = Darwin ] && command -v codesign >/dev/null 2>&1; then
    codesign -s - -f "$OUT/ollm-tok" >/dev/null 2>&1
  fi
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

if [ "$TARGET" = openwaldo ]; then
  {
    host_target_hdr
    printf 'om2\n'
    cat "$src/net/json.om" "$here/tools/openwaldo/waldo_ingest.om"
  } > "$OUT/ollm-waldo.om"
  "$OMC" < "$OUT/ollm-waldo.om" > "$OUT/ollm-waldo.tmp"
  chmod +x "$OUT/ollm-waldo.tmp"
  mv -f "$OUT/ollm-waldo.tmp" "$OUT/ollm-waldo"
  if [ "$(uname -s)" = Darwin ] && command -v codesign >/dev/null 2>&1; then
    codesign -s - -f "$OUT/ollm-waldo" >/dev/null 2>&1
  fi
  printf 'built %s (%s bytes, target=%s, omc=%s)\n' \
    "$OUT/ollm-waldo" "$(wc -c < "$OUT/ollm-waldo" | tr -d ' ')" "$TARGET" "$OMC"
  exit 0
fi

if [ "$TARGET" = univ ]; then
  u=$src/univ
  {
    host_target_hdr
    printf 'om2\n'
    cat "$u/umath.om" "$u/uarena.om" "$u/uadam.om" "$u/uckpt.om" "$u/umodel.om" "$u/udistill.om" "$u/ugrammar.om" "$u/ucli.om"
  } > "$OUT/ollm-univ.om"
  "$OMC" < "$OUT/ollm-univ.om" > "$OUT/ollm-univ.tmp"
  chmod +x "$OUT/ollm-univ.tmp"
  mv -f "$OUT/ollm-univ.tmp" "$OUT/ollm-univ"
  if [ "$(uname -s)" = Darwin ] && command -v codesign >/dev/null 2>&1; then
    codesign -s - -f "$OUT/ollm-univ" >/dev/null 2>&1
  fi
  printf 'built %s (%s bytes, target=%s, omc=%s)\n' \
    "$OUT/ollm-univ" "$(wc -c < "$OUT/ollm-univ" | tr -d ' ')" "$TARGET" "$OMC"
  exit 0
fi

if [ "$TARGET" = interchange ]; then
  u=$src/univ
  {
    host_target_hdr
    printf 'om2\n'
    cat "$src/net/json.om" "$src/net/jsonw.om" "$u/umath.om" "$u/uarena.om" "$u/uckpt.om" "$u/interchange.om"
  } > "$OUT/ollm-interchange.om"
  "$OMC" < "$OUT/ollm-interchange.om" > "$OUT/ollm-interchange.tmp"
  chmod +x "$OUT/ollm-interchange.tmp"
  mv -f "$OUT/ollm-interchange.tmp" "$OUT/ollm-interchange"
  if [ "$(uname -s)" = Darwin ] && command -v codesign >/dev/null 2>&1; then
    codesign -s - -f "$OUT/ollm-interchange" >/dev/null 2>&1
  fi
  printf 'built %s (%s bytes, target=%s, omc=%s)\n' \
    "$OUT/ollm-interchange" "$(wc -c < "$OUT/ollm-interchange" | tr -d ' ')" "$TARGET" "$OMC"
  exit 0
fi

lead=
pre=
pool=lpool.om
plat=lio.om
rev_mod=rev.om
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
    lead='target a64macos
om2'
    plat=mio.om
    post=$src/noserve.om
    bin=ollm-macos
    ;;
  x86linux)
    lead='target x86linux
om2'
    pool=xpool.om
    plat=lio.om
    rev_mod=norev.om
    post=$src/noserve.om
    bin=ollm-x86linux
    ;;
  x86efi)
    lead='target x86efi'
    pool=epool.om
    plat=eio.om
    rev_mod=norev.om
    post=$src/noserve.om
    bin=ollm.efi
    ;;
  a64efi)
    lead='target a64efi'
    pool=epool.om
    plat=eio.om
    rev_mod=norev.om
    post=$src/noserve.om
    bin=ollm-aa64.efi
    EFIRT=${EFIRT:-$(dirname "$OMC")/../a64efi/efirt.om}
    if [ ! -f "$EFIRT" ]; then
      echo "error: a64efi requires efirt.om (set EFIRT=/path/to/efirt.om)" >&2
      exit 1
    fi
    ;;
  *)
    echo "error: unknown TARGET '$TARGET' (expected: gb10, aarch64, macos, macos-metal, x86linux, x86efi, a64efi, tok, wtrain, openwaldo, univ, interchange)" >&2
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
    "$src/spec.om"
  if [ "$TARGET" != x86efi ] && [ "$TARGET" != a64efi ]; then
    cat \
      "$src/$rev_mod" \
      "$src/arboris.om" \
      "$src/sp_solve.om" \
      "$src/spiral.om" \
      "$src/apu_weave.om" \
      "$src/omg.om" \
      "$src/pillars_cli.om"
  else
    cat "$src/$rev_mod"
    cat <<'EOF'
fn cmdspforge to void
exit 2
end
fn cmdspcryst to void
exit 2
end
fn cmdspweval to void
exit 2
end
fn cmdspwsamp to void
exit 2
end
fn cmdapuweave to void
exit 2
end
fn sp2_is_sp2 ai ulong to bool
ret false
end
fn cmdomg to void
exit 2
end
fn cmdomg_cli to void
exit 2
end
fn cmdarboris to void
exit 2
end
fn cmdarboris_cli to void
exit 2
end
EOF
  fi
  cat \
    "$src/cli.om" \
    "$src/$plat" \
    "$post"
  [ "$TARGET" = gb10 ] || cat "$src/nogpu.om"
  if [ "$TARGET" = macos ]; then
    cat "$src/apu_fabric_macos.om"
  elif [ "$TARGET" = aarch64 ] || [ "$TARGET" = x86linux ]; then
    cat "$src/apu_fabric_linux.om" "$src/apu_fabric_nogpu.om"
  fi
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
    "$g/gg.om" \
    "$g/gt.om" \
    "$g/gm.om" \
    "$g/ring.om" \
    "$g/ejit.om" \
    "$src/apu_fabric_linux.om" \
    "$src/apu_fabric_gb10.om" >> "$om_file"
fi

"$OMC" < "$om_file" > "$OUT/$bin.tmp"
chmod +x "$OUT/$bin.tmp"
mv -f "$OUT/$bin.tmp" "$OUT/$bin"

if [ "$TARGET" = macos ] && command -v codesign >/dev/null 2>&1; then
  codesign -s - -f "$OUT/$bin" >/dev/null 2>&1
fi

printf 'built %s (%s bytes, target=%s, omc=%s)\n' \
  "$OUT/$bin" "$(wc -c < "$OUT/$bin" | tr -d ' ')" "$TARGET" "$OMC"
