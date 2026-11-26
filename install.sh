#!/bin/sh
# Omega Helix installer: pure-Omega binaries + Studio CLI, zero dependencies
# beyond a seeded `omc`, `python3` (for the Studio) and POSIX tools.
#
#   sh install.sh [--prefix DIR] [--skip-build] [--no-download] [--dry-run] [--help]
#
# Installs into <PREFIX>/opt/omega-helix and links <PREFIX>/bin/omega-helix.
# Auto-detects host topology: Direct Spark/Linux (desktop or headless),
# macOS (Apple Silicon native a64macos + Intel remote mode), and Windows WSL 2.
set -eu

PREFIX=${HOME:-/tmp}/.local
SKIP_BUILD=0
NO_DOWNLOAD=0
DRY_RUN=0

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
  --dry-run        Print auto-detected machine topology, targets & launch plan
                   without modifying disk
  -h, --help       Show this help

Environment:
  OMC   Path to a seeded omc compiler (auto-detected when unset)

Supported Topologies (auto-detected):
  Linux / DGX Spark (desktop)   Native aarch64/x86_64 (+ GB10), localhost, xdg-open (no SSH tunnel)
  Linux / DGX Spark (headless)  Native aarch64/x86_64 (+ GB10), localhost, clickable URL (no SSH tunnel)
  macOS Apple Silicon (arm64)   Native a64macos Mach-O (SHA-256 + ad-hoc codesign, zero notarization fees),
                                local or optional --remote <host> Spark SSH tunnel, opens via `open`
  macOS Intel (x86_64)          Studio CLI/UI driving a remote Spark/Linux host via SSH tunnel (`open`)
  Windows 10/11 (WSL 2)         Static x86linux inside WSL 2 (no native .exe), local or optional
                                --remote <host> SSH tunnel, opens via wslview/explorer.exe/cmd.exe
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
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown option '$1' (try --help)" >&2; exit 2 ;;
  esac
done

SRC=$(cd "$(dirname "$0")" && pwd)
[ -f "$SRC/scripts/build.sh" ] && [ -d "$SRC/native" ] || {
  echo "error: run install.sh from an omega-helix checkout ($SRC has no scripts/build.sh)" >&2
  exit 1
}

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

# --- topology auto-detection ------------------------------------------------
OS=$(uname -s)
ARCH=$(uname -m)
DISP=${DISPLAY:-}
WAYL=${WAYLAND_DISPLAY:-}
SSH_ENV=${SSH_CONNECTION:-${SSH_CLIENT:-${SSH_TTY:-}}}
IS_WSL=0
GB10=0
TEST_BROWSER=
PROC_VER_FILE=/proc/version

if [ "$DRY_RUN" = 1 ]; then
  [ -n "${OMEGA_INSTALL_TEST_OS:-}" ] && OS=$OMEGA_INSTALL_TEST_OS
  [ -n "${OMEGA_INSTALL_TEST_ARCH:-}" ] && ARCH=$OMEGA_INSTALL_TEST_ARCH
  if [ "${OMEGA_INSTALL_TEST_DISPLAY+set}" = set ]; then DISP=$OMEGA_INSTALL_TEST_DISPLAY; fi
  if [ "${OMEGA_INSTALL_TEST_WAYLAND+set}" = set ]; then WAYL=$OMEGA_INSTALL_TEST_WAYLAND; fi
  if [ "${OMEGA_INSTALL_TEST_SSH+set}" = set ]; then SSH_ENV=$OMEGA_INSTALL_TEST_SSH; fi
  if [ "${OMEGA_INSTALL_TEST_BROWSER+set}" = set ]; then TEST_BROWSER=$OMEGA_INSTALL_TEST_BROWSER; fi
  if [ "${OMEGA_INSTALL_TEST_PROC_VERSION+set}" = set ]; then PROC_VER_FILE=$OMEGA_INSTALL_TEST_PROC_VERSION; fi
fi

if [ "$OS" = Linux ]; then
  if [ "$DRY_RUN" = 1 ] && [ "${OMEGA_INSTALL_TEST_WSL+set}" = set ]; then
    IS_WSL=$OMEGA_INSTALL_TEST_WSL
  elif [ -n "${WSL_DISTRO_NAME:-}" ] || [ -n "${WSL_INTEROP:-}" ] || { [ -r "$PROC_VER_FILE" ] && grep -qi microsoft "$PROC_VER_FILE" 2>/dev/null; }; then
    IS_WSL=1
  fi
  if [ "$DRY_RUN" = 1 ] && [ "${OMEGA_INSTALL_TEST_GPU+set}" = set ]; then
    GB10=$OMEGA_INSTALL_TEST_GPU
  elif [ "$IS_WSL" = 0 ] && { [ "$ARCH" = aarch64 ] || [ "$ARCH" = arm64 ]; }; then
    if [ -e /dev/nvidiactl ] || [ -e /dev/aien-gpu ]; then
      GB10=1
    fi
  fi
fi

TOPOLOGY=unknown
PLAT="$OS $ARCH"
TARGETS=
OPTIONAL_TARGETS=
SIGN_MODE=sha256
BROWSER_CMD=print-url
LAUNCH_MODE=local
SKIP_SSH_TUNNEL=1
NATIVE_EXE=0

detect_wsl_browser() {
  if [ -n "$TEST_BROWSER" ]; then
    printf '%s\n' "$TEST_BROWSER"
    return 0
  fi
  if command -v wslview >/dev/null 2>&1; then
    printf 'wslview\n'
  elif command -v explorer.exe >/dev/null 2>&1; then
    printf 'explorer.exe\n'
  elif command -v cmd.exe >/dev/null 2>&1; then
    printf 'cmd.exe /c start\n'
  else
    printf 'print-url\n'
  fi
}

case "$OS/$ARCH" in
  Darwin/arm64)
    TOPOLOGY="macos-arm64"
    PLAT="macOS Apple Silicon (arm64)"
    TARGETS="univ interchange openwaldo tok macos"
    SIGN_MODE="sha256+adhoc-codesign"
    LAUNCH_MODE="local-or-remote"
    SKIP_SSH_TUNNEL=1
    if [ -n "$SSH_ENV" ]; then BROWSER_CMD="print-url"; else BROWSER_CMD="open"; fi
    ;;
  Darwin/x86_64)
    TOPOLOGY="macos-x86_64"
    PLAT="macOS Intel (x86_64)"
    TARGETS=""
    SKIP_BUILD=1
    SIGN_MODE="none"
    LAUNCH_MODE="remote-tunnel"
    SKIP_SSH_TUNNEL=0
    if [ -n "$SSH_ENV" ]; then BROWSER_CMD="print-url"; else BROWSER_CMD="open"; fi
    ;;
  Linux/aarch64|Linux/arm64)
    if [ "$IS_WSL" = 1 ]; then
      TOPOLOGY="wsl2"
      PLAT="Windows WSL 2 (aarch64)"
      TARGETS="univ interchange openwaldo tok aarch64"
      SIGN_MODE="sha256"
      LAUNCH_MODE="local-or-remote"
      SKIP_SSH_TUNNEL=1
      if [ -n "$SSH_ENV" ]; then BROWSER_CMD="print-url"; else BROWSER_CMD=$(detect_wsl_browser); fi
    else
      PLAT="Linux aarch64"
      if [ "$GB10" = 1 ]; then
        PLAT="DGX Spark Linux (aarch64, GB10)"
        OPTIONAL_TARGETS="gb10"
      fi
      TARGETS="univ interchange openwaldo tok aarch64 wtrain"
      SIGN_MODE="sha256"
      SKIP_SSH_TUNNEL=1
      if { [ -n "$DISP" ] || [ -n "$WAYL" ]; } && [ -z "$SSH_ENV" ]; then
        if [ "$GB10" = 1 ]; then TOPOLOGY="spark-desktop"; else TOPOLOGY="linux-desktop"; fi
        LAUNCH_MODE="local-desktop"
        if [ -n "$TEST_BROWSER" ]; then
          BROWSER_CMD=$TEST_BROWSER
        elif command -v xdg-open >/dev/null 2>&1; then
          BROWSER_CMD="xdg-open"
        else
          BROWSER_CMD="print-url"
        fi
      else
        if [ "$GB10" = 1 ]; then TOPOLOGY="spark-headless"; else TOPOLOGY="linux-headless"; fi
        LAUNCH_MODE="local-headless"
        BROWSER_CMD="print-url"
      fi
    fi
    ;;
  Linux/x86_64|Linux/amd64)
    if [ "$IS_WSL" = 1 ]; then
      TOPOLOGY="wsl2"
      PLAT="Windows 10/11 WSL 2 (x86_64, static x86linux)"
      TARGETS="univ interchange openwaldo tok x86linux"
      SIGN_MODE="sha256"
      LAUNCH_MODE="local-or-remote"
      SKIP_SSH_TUNNEL=1
      if [ -n "$SSH_ENV" ]; then BROWSER_CMD="print-url"; else BROWSER_CMD=$(detect_wsl_browser); fi
    else
      PLAT="Linux x86_64"
      TARGETS="univ interchange openwaldo tok x86linux"
      SIGN_MODE="sha256"
      SKIP_SSH_TUNNEL=1
      if { [ -n "$DISP" ] || [ -n "$WAYL" ]; } && [ -z "$SSH_ENV" ]; then
        TOPOLOGY="linux-desktop"
        LAUNCH_MODE="local-desktop"
        if [ -n "$TEST_BROWSER" ]; then
          BROWSER_CMD=$TEST_BROWSER
        elif command -v xdg-open >/dev/null 2>&1; then
          BROWSER_CMD="xdg-open"
        else
          BROWSER_CMD="print-url"
        fi
      else
        TOPOLOGY="linux-headless"
        LAUNCH_MODE="local-headless"
        BROWSER_CMD="print-url"
      fi
    fi
    ;;
  MINGW*|MSYS*|CYGWIN*|Windows_NT/*|Windows/*)
    TOPOLOGY="windows-powershell"
    PLAT="Windows ($OS/$ARCH)"
    TARGETS=""
    SIGN_MODE="none"
    LAUNCH_MODE="wsl2-required"
    SKIP_SSH_TUNNEL=1
    BROWSER_CMD="print-url"
    if [ "$DRY_RUN" = 0 ]; then
      die "no native Windows .exe exists. Run Omega Helix inside WSL 2 (wsl --install) using the static x86linux path, or use https://interplane.io/omega/compile/"
    fi
    ;;
  *)
    die "unsupported platform $OS/$ARCH (supported: Darwin arm64/x86_64, Linux aarch64/x86_64, Windows WSL 2)"
    ;;
esac

if [ "$DRY_RUN" = 1 ]; then
  printf 'TOPOLOGY=%s\n' "$TOPOLOGY"
  printf 'PLATFORM=%s\n' "$PLAT"
  printf 'OS=%s\n' "$OS"
  printf 'ARCH=%s\n' "$ARCH"
  printf 'IS_WSL=%s\n' "$IS_WSL"
  printf 'GB10=%s\n' "$GB10"
  printf 'TARGETS=%s\n' "$TARGETS"
  printf 'OPTIONAL_TARGETS=%s\n' "$OPTIONAL_TARGETS"
  printf 'SKIP_BUILD=%s\n' "$SKIP_BUILD"
  printf 'SIGN_MODE=%s\n' "$SIGN_MODE"
  printf 'LAUNCH_MODE=%s\n' "$LAUNCH_MODE"
  printf 'SKIP_SSH_TUNNEL=%s\n' "$SKIP_SSH_TUNNEL"
  printf 'BROWSER_CMD=%s\n' "$BROWSER_CMD"
  printf 'NATIVE_EXE=%s\n' "$NATIVE_EXE"
  printf 'BIND=127.0.0.1:18420\n'
  exit 0
fi

mkdir -p "$PREFIX"
PREFIX=$(cd "$PREFIX" && pwd)

banner() {
  printf '\n'
  printf '  %s━━━━━━━━%s%s━━━━━━━━%s%s━━━━━━━━%s%s━━━━━━━━%s\n' "$C1" "$OFF" "$C2" "$OFF" "$C3" "$OFF" "$C4" "$OFF"
  printf '  %s%sOMEGA HELIX%s  %spure-Omega LLM training, four strands braided%s\n' "$BOLD" "$C1" "$OFF" "$DIM" "$OFF"
  printf '  %s━━━━━━━━%s%s━━━━━━━━%s%s━━━━━━━━%s%s━━━━━━━━%s\n\n' "$C4" "$OFF" "$C3" "$OFF" "$C2" "$OFF" "$C1" "$OFF"
}
banner

step "Topology: $TOPOLOGY ($PLAT)"
if [ "$GB10" = 1 ]; then
  note "GB10 GPU detected (/dev/nvidiactl or /dev/aien-gpu): will also build ollm-gb10"
elif [ "$TOPOLOGY" = wsl2 ]; then
  note "WSL 2 detected: building static x86linux binaries (no native Windows .exe); Windows browser via $BROWSER_CMD"
elif [ "$TOPOLOGY" = macos-arm64 ]; then
  note "Apple Silicon native Mach-O: SHA-256 verified + ad-hoc codesign -s - -f (zero Apple notarization fees)"
elif [ "$TOPOLOGY" = macos-x86_64 ]; then
  note "Intel macOS: installing Studio CLI/UI for remote Spark SSH tunnel mode (native a64macos omc targets Apple Silicon)"
else
  note "no GB10 GPU detected: CPU targets only"
fi
note "launch mode: $LAUNCH_MODE · browser: $BROWSER_CMD · ssh tunnel: $([ "$SKIP_SSH_TUNNEL" = 1 ] && printf 'skipped (local localhost)' || printf 'remote mode')"
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
    elif [ "$IS_WSL" = 1 ]; then
      say "  WSL 2 (static x86linux): install omc from https://interplane.io/omega/ (verify SHA-256) or set OMC=/path/to/omc" >&2
    else
      say "  Install it from https://interplane.io/omega/ (static omc archive + sha256), or set OMC=/path/to/omc" >&2
    fi
    say "  Then re-run: sh install.sh --prefix $PREFIX    (or --skip-build to install the Studio CLI only)" >&2
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
(cd "$SRC" && COPYFILE_DISABLE=1 tar -cf - --exclude=__pycache__ --exclude='*.pyc' --exclude=.DS_Store --exclude='._*' \
   native scripts tools docs bin tests/oracle.py README.md LICENSE llms.txt 2>/dev/null) | tar -C "$STAGE" -xf -
chmod +x "$STAGE/bin/omega-helix" "$STAGE/scripts/"*.sh 2>/dev/null || true

# --- build ------------------------------------------------------------------
BUILT=
FAILED=
build_one() { # target outname [optional]
  t=$1; out=$2; opt=${3:-0}
  tmpd=$STAGE/.build-$t
  mkdir -p "$tmpd"
  if TARGET=$t OUT=$tmpd OMC=$OMC_BIN sh "$SRC/scripts/build.sh" >"$LOG" 2>&1; then
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
  step "Skipping build (--skip-build or remote-only platform): no binaries compiled"
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
    build_one "$t" "$o"
  done
  [ "$GB10" = 1 ] && build_one gb10 ollm-gb10 1
  rm -f "$LOG"
  [ -z "$FAILED" ] || die "build failed for:$FAILED (nothing was installed)"
fi

# --- sign (macOS) & SHA-256 verification ------------------------------------
if [ "$OS" = Darwin ] && [ "$SKIP_BUILD" = 0 ] && command -v codesign >/dev/null 2>&1; then
  step "Ad-hoc signing (zero Apple notarization fees)"
  for f in "$STAGE"/bin/ollm*; do
    [ -f "$f" ] || continue
    codesign -s - -f "$f" >/dev/null 2>&1 || die "codesign failed for $f"
    codesign -v "$f" >/dev/null 2>&1 || die "codesign verify failed for $f"
    note "signed $(basename "$f")"
  done
fi

if [ "$SKIP_BUILD" = 0 ]; then
  step "Verifying SHA-256 checksums"
  if command -v sha256sum >/dev/null 2>&1; then
    (cd "$STAGE/bin" && sha256sum ollm* > SHA256SUMS && sha256sum -c SHA256SUMS >/dev/null) || die "SHA-256 verification failed"
    note "recorded and verified $OPT/bin/SHA256SUMS (sha256sum)"
  elif command -v shasum >/dev/null 2>&1; then
    (cd "$STAGE/bin" && shasum -a 256 ollm* > SHA256SUMS && shasum -a 256 -c SHA256SUMS >/dev/null) || die "SHA-256 verification failed"
    note "recorded and verified $OPT/bin/SHA256SUMS (shasum -a 256)"
  fi
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
printf '  %sOmega Helix installed (%s).%s\n\n' "$BOLD" "$TOPOLOGY" "$OFF"
case ":${PATH:-}:" in
  *":$PREFIX/bin:"*) ;;
  *) printf '  %sAdd to PATH:%s  export PATH="%s/bin:$PATH"\n\n' "$C4" "$OFF" "$PREFIX" ;;
esac
case "$TOPOLOGY" in
  spark-desktop|linux-desktop)
    cat <<EOF
  Quick start (Direct local Linux/Spark with monitor — no SSH tunnel needed):
    omega-helix            bind localhost & open Studio via $BROWSER_CMD
    omega-helix doctor     check topology, binaries, GPU and OpenWALDO
    omega-helix setup      guided first-run setup
EOF
    ;;
  spark-headless|linux-headless)
    cat <<EOF
  Quick start (Headless Linux/Spark — prints clickable localhost URL):
    omega-helix            bind 127.0.0.1:18420 & print URL (no browser attempt)
    omega-helix doctor     check topology, binaries, GPU and OpenWALDO
EOF
    ;;
  macos-arm64|wsl2)
    cat <<EOF
  Quick start ($TOPOLOGY — local mode OR optional remote Spark SSH tunnel):
    omega-helix                         launch local Studio (opens via $BROWSER_CMD)
    omega-helix studio --remote spark   connect to a remote DGX Spark over SSH tunnel
    omega-helix doctor                  check topology, binaries and hardware
EOF
    ;;
  macos-x86_64)
    cat <<EOF
  Quick start (Intel macOS — remote Spark SSH tunnel mode):
    omega-helix studio --remote spark   connect to a remote DGX Spark over SSH tunnel (opens via open)
    omega-helix host list               list SSH hosts from ~/.ssh/config
EOF
    ;;
esac
if ! command -v python3 >/dev/null 2>&1; then
  warn "python3 not found: the omega-helix Studio CLI needs python3 (stdlib only)"
fi
exit 0
