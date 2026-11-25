#!/bin/sh
# Omega Helix installer: pure-Omega binaries + Studio CLI, zero dependencies
# beyond a seeded `omc`, `python3` (for the Studio) and POSIX tools.
#
#   sh install.sh [--prefix DIR] [--skip-build] [--no-download] [--help]
#
# Installs into <PREFIX>/opt/omega-helix and links <PREFIX>/bin/omega-helix.
set -eu

PREFIX=${HOME:-/tmp}/.local
SKIP_BUILD=0
NO_DOWNLOAD=0

usage() {
  cat <<'EOF'
Omega Helix installer

Usage: sh install.sh [options]

Options:
  --prefix DIR     Install under DIR (default: $HOME/.local)
                     DIR/opt/omega-helix   code, docs, tools, binaries
                     DIR/bin/omega-helix   CLI entrypoint (+ ollm-univ,
                                           ollm-interchange, ollm-waldo links)
  --skip-build     Copy scripts, tools and the Studio only; compile nothing
                   (no omc needed)
  --no-download    Never try to install a missing omc (print guidance instead)
  -h, --help       Show this help

Environment:
  OMC   Path to a seeded omc compiler (auto-detected when unset)

Binaries built (pure Omega, no C, no CUDA):
  all platforms    ollm-univ  ollm-interchange  ollm-waldo  ollm-tok
  macOS arm64      ollm-macos
  Linux aarch64    ollm  ollm-wtrain  (+ ollm-gb10 when the GB10 target builds)
  Linux x86_64     ollm-x86linux
EOF
}

while [ $# -gt 0 ]; do
  case $1 in
    --prefix)
      [ $# -ge 2 ] || { echo "error: --prefix needs a directory" >&2; exit 2; }
      PREFIX=$2; shift 2 ;;
    --prefix=*) PREFIX=${1#--prefix=}; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --no-download) NO_DOWNLOAD=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option '$1' (try --help)" >&2; exit 2 ;;
  esac
done

SRC=$(cd "$(dirname "$0")" && pwd)
[ -f "$SRC/scripts/build.sh" ] && [ -d "$SRC/native" ] || {
  echo "error: run install.sh from an omega-helix checkout ($SRC has no scripts/build.sh)" >&2
  exit 1
}
mkdir -p "$PREFIX"
PREFIX=$(cd "$PREFIX" && pwd)

# --- presentation -----------------------------------------------------------
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != dumb ]; then
  ESC=$(printf '\033')
  C1="${ESC}[38;5;51m"; C2="${ESC}[38;5;141m"; C3="${ESC}[38;5;42m"; C4="${ESC}[38;5;214m"
  DIM="${ESC}[2m"; BOLD="${ESC}[1m"; RED="${ESC}[38;5;203m"; OFF="${ESC}[0m"
else
  C1=; C2=; C3=; C4=; DIM=; BOLD=; RED=; OFF=
fi

say()  { printf '%s\n' "$*"; }
step() { printf '%s==>%s %s%s%s\n' "$C3" "$OFF" "$BOLD" "$*" "$OFF"; }
note() { printf '    %s%s%s\n' "$DIM" "$*" "$OFF"; }
warn() { printf '%swarn:%s %s\n' "$C4" "$OFF" "$*" >&2; }
die()  { printf '%serror:%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

banner() {
  printf '\n'
  printf '  %s━━━━━━━━%s%s━━━━━━━━%s%s━━━━━━━━%s%s━━━━━━━━%s\n' "$C1" "$OFF" "$C2" "$OFF" "$C3" "$OFF" "$C4" "$OFF"
  printf '  %s%sOMEGA HELIX%s  %spure-Omega LLM training, four strands braided%s\n' "$BOLD" "$C1" "$OFF" "$DIM" "$OFF"
  printf '  %s━━━━━━━━%s%s━━━━━━━━%s%s━━━━━━━━%s%s━━━━━━━━%s\n\n' "$C4" "$OFF" "$C3" "$OFF" "$C2" "$OFF" "$C1" "$OFF"
}
banner

# --- platform ---------------------------------------------------------------
OS=$(uname -s); ARCH=$(uname -m)
case "$OS/$ARCH" in
  Darwin/arm64)
    PLAT="macOS arm64"
    TARGETS="univ interchange openwaldo tok macos" ;;
  Linux/aarch64|Linux/arm64)
    PLAT="Linux aarch64"
    TARGETS="univ interchange openwaldo tok aarch64 wtrain" ;;
  Linux/x86_64|Linux/amd64)
    PLAT="Linux x86_64"
    TARGETS="univ interchange openwaldo tok x86linux" ;;
  *) die "unsupported platform $OS/$ARCH (supported: Darwin arm64, Linux aarch64, Linux x86_64)" ;;
esac
GB10=0
if [ "$OS/$ARCH" != Darwin/arm64 ] && [ -e /dev/nvidiactl ] && [ "$ARCH" != x86_64 ]; then
  GB10=1
fi
step "Platform: $PLAT"
if [ "$GB10" = 1 ]; then note "NVIDIA GPU detected (/dev/nvidiactl): will also try the GB10 target"
else note "no GB10 GPU detected: CPU targets only"; fi
note "prefix: $PREFIX"

# --- omc --------------------------------------------------------------------
find_omc() {
  if [ -n "${OMC:-}" ] && [ -x "$OMC" ]; then printf '%s\n' "$OMC"; return 0; fi
  c=$(command -v omc 2>/dev/null || true)
  for cand in "$c" "${HOME:-}/.local/bin/omc" "${HOME:-}/canon/omega-mv/seed/out/omc"; do
    if [ -n "$cand" ] && [ -x "$cand" ]; then printf '%s\n' "$cand"; return 0; fi
  done
  return 1
}

OMC_BIN=
if [ "$SKIP_BUILD" = 0 ]; then
  OMC_BIN=$(find_omc || true)
  if [ -z "$OMC_BIN" ] && [ "$NO_DOWNLOAD" = 0 ] && [ "$OS" = Darwin ] && command -v brew >/dev/null 2>&1; then
    step "omc not found: installing via Homebrew"
    if brew tap drakestapleton/omega https://interplane.io/drakestapleton/homebrew-omega.git \
       && brew install drakestapleton/omega/omc; then
      OMC_BIN=$(find_omc || true)
    else
      warn "Homebrew install of omc failed"
    fi
  fi
  if [ -z "$OMC_BIN" ]; then
    printf '%serror:%s no seeded omc compiler found.\n' "$RED" "$OFF" >&2
    if [ "$OS" = Darwin ]; then
      say "  Install it:  brew tap drakestapleton/omega https://interplane.io/drakestapleton/homebrew-omega.git && brew install drakestapleton/omega/omc" >&2
    else
      say "  Install it from https://interplane.io/omega/ (static omc archive + sha256), or set OMC=/path/to/omc" >&2
    fi
    say "  Then re-run: sh install.sh --prefix $PREFIX    (or --skip-build to install the CLI only)" >&2
    exit 1
  fi
  step "Compiler: $OMC_BIN"
fi

# --- stage ------------------------------------------------------------------
OPT=$PREFIX/opt/omega-helix
mkdir -p "$PREFIX/opt" "$PREFIX/bin"
STAGE=$PREFIX/opt/.omega-helix.stage.$$
LOG=$STAGE/.build.log
trap 'rm -rf "$STAGE" "$PREFIX/opt/.omega-helix.old.$$"' EXIT
trap 'exit 130' INT TERM
rm -rf "$STAGE"
mkdir -p "$STAGE/bin"

step "Copying sources"
(cd "$SRC" && tar -cf - --exclude=__pycache__ --exclude='*.pyc' --exclude=.DS_Store \
   native scripts tools docs bin README.md LICENSE 2>/dev/null) | tar -C "$STAGE" -xf -
chmod +x "$STAGE/bin/omega-helix" "$STAGE/scripts/"*.sh 2>/dev/null || true

# --- build ------------------------------------------------------------------
BUILT=
FAILED=
build_one() { # target outname [optional]
  t=$1; out=$2; opt=${3:-0}
  tmpd=$STAGE/.build-$t
  mkdir -p "$tmpd"
  if TARGET=$t OUT=$tmpd OMC=$OMC_BIN sh "$SRC/scripts/build.sh" >"$LOG" 2>&1; then
    # build.sh names most binaries like $out; wtrain is emitted as plain "wtrain"
    prod=$tmpd/$out
    [ -f "$prod" ] || prod=$tmpd/$t
    if [ -f "$prod" ]; then
      mv -f "$prod" "$STAGE/bin/$out"
      chmod +x "$STAGE/bin/$out"
      sz=$(wc -c <"$STAGE/bin/$out" | tr -d ' ')
      printf '    %s✓%s %-16s %s%s bytes  (TARGET=%s)%s\n' "$C3" "$OFF" "$out" "$DIM" "$sz" "$t" "$OFF"
      BUILT="$BUILT $out"
      rm -rf "$tmpd"
      return 0
    fi
  fi
  if [ "$opt" = 1 ]; then
    printf '    %s–%s %-16s %sskipped (TARGET=%s did not build here)%s\n' "$C4" "$OFF" "$out" "$DIM" "$t" "$OFF"
  else
    printf '    %s✗%s %-16s (TARGET=%s)\n' "$RED" "$OFF" "$out" "$t"
    sed 's/^/        /' "$LOG" | tail -n 8
    FAILED="$FAILED $out"
  fi
  rm -rf "$tmpd"
  return 0
}

if [ "$SKIP_BUILD" = 1 ]; then
  step "Skipping build (--skip-build): no binaries compiled"
else
  step "Compiling pure-Omega binaries"
  for t in $TARGETS; do
    case $t in
      univ) o=ollm-univ ;;
      interchange) o=ollm-interchange ;;
      openwaldo) o=ollm-waldo ;;
      tok) o=ollm-tok ;;
      macos) o=ollm-macos ;;
      aarch64) o=ollm ;;
      wtrain) o=ollm-wtrain ;;
      x86linux) o=ollm-x86linux ;;
    esac
    # wtrain's build.sh output is "wtrain": build_one falls back to $TARGET
    build_one "$t" "$o"
  done
  [ "$GB10" = 1 ] && build_one gb10 ollm-gb10 1
  rm -f "$LOG"
  [ -z "$FAILED" ] || die "build failed for:$FAILED (nothing was installed)"
fi

# --- sign (macOS) -----------------------------------------------------------
if [ "$OS" = Darwin ] && [ "$SKIP_BUILD" = 0 ] && command -v codesign >/dev/null 2>&1; then
  step "Ad-hoc signing"
  for f in "$STAGE"/bin/ollm*; do
    [ -f "$f" ] || continue
    codesign -s - -f "$f" >/dev/null 2>&1 || die "codesign failed for $f"
    codesign -v "$f" >/dev/null 2>&1 || die "codesign verify failed for $f"
    note "signed $(basename "$f")"
  done
fi

# --- commit (atomic swap) ---------------------------------------------------
step "Installing to $OPT"
rm -f "$STAGE/.build.log"
if [ "$SKIP_BUILD" = 1 ]; then # keep binaries from a previous full install
  for f in "$OPT"/bin/ollm*; do
    [ -f "$f" ] && cp -p "$f" "$STAGE/bin/" && note "kept existing $(basename "$f")"
  done
fi
if [ -e "$OPT" ]; then mv "$OPT" "$PREFIX/opt/.omega-helix.old.$$"; fi
mv "$STAGE" "$OPT"
rm -rf "$PREFIX/opt/.omega-helix.old.$$"

link() { # target linkname
  ln -sfn "$1" "$PREFIX/bin/$2.tmp.$$" && mv -f "$PREFIX/bin/$2.tmp.$$" "$PREFIX/bin/$2"
}
link "$OPT/bin/omega-helix" omega-helix
for l in ollm-univ ollm-interchange ollm-waldo; do
  if [ -e "$OPT/bin/$l" ]; then link "$OPT/bin/$l" "$l"; fi
done
note "linked $PREFIX/bin/omega-helix"

# --- finish -----------------------------------------------------------------
printf '\n  %s━━━━━━━━%s%s━━━━━━━━%s%s━━━━━━━━%s%s━━━━━━━━%s\n' "$C1" "$OFF" "$C2" "$OFF" "$C3" "$OFF" "$C4" "$OFF"
printf '  %sOmega Helix installed.%s\n\n' "$BOLD" "$OFF"
case ":${PATH:-}:" in
  *":$PREFIX/bin:"*) ;;
  *) printf '  %sAdd to PATH:%s  export PATH="%s/bin:$PATH"\n\n' "$C4" "$OFF" "$PREFIX" ;;
esac
cat <<EOF
  Quick start:
    omega-helix            open the Studio (local web UI)
    omega-helix setup      guided first-run setup
    omega-helix catalog    browse OpenWALDO datasets
    omega-helix doctor     check binaries, python and hardware
EOF
if ! command -v python3 >/dev/null 2>&1; then
  warn "python3 not found: the omega-helix Studio CLI needs python3 (stdlib only)"
fi
exit 0
