#!/bin/sh
# 5-minute external reproducer & GitHub Issue #5 receipt generator for Omega Helix.
#
# POSIX sh (set -eu), zero external dependencies beyond sh, python3 (stdlib), and omc.
# Builds the native pure-Omega engines for the host OS/architecture, verifies linkage
# (Mach-O libSystem-only on macOS; static ELF zero libc/CUDA on Linux), runs available
# accelerator checks (Apple Silicon Metal or DGX Spark GB10 NVFP4), cross-checks bit-exact
# IEEE-754 f64 parity via tests/oracle.py, trips negative controls, and writes a
# deterministic GitHub Issue #5 receipt (<OUT>/repro-5min-receipt.txt) + Receipt SHA-256.
#
# Usage:
#   sh scripts/repro-5min.sh [--out <dir>] [--cpu-only] [--live-gpu] [--json]
set -eu

here=$(cd "$(dirname "$0")/.." && pwd)
caller_pwd=$(pwd)

OUT_ARG=""
CPU_ONLY=0
LIVE_GPU=0
JSON_MODE=0
PARSE_ERR=""

while [ $# -gt 0 ]; do
  case "$1" in
    --out)
      shift
      if [ $# -eq 0 ] || [ -z "${1:-}" ]; then
        PARSE_ERR="--out requires a directory argument"
        break
      fi
      OUT_ARG=$1
      ;;
    --out=*)
      OUT_ARG=${1#--out=}
      if [ -z "$OUT_ARG" ]; then
        PARSE_ERR="--out requires a non-empty directory argument"
        break
      fi
      ;;
    --cpu-only)
      CPU_ONLY=1
      ;;
    --live-gpu)
      LIVE_GPU=1
      ;;
    --json)
      JSON_MODE=1
      ;;
    -h|--help)
      cat <<'EOF'
Usage: sh scripts/repro-5min.sh [OPTIONS]

Options:
  --out <dir>   Output directory for builds, logs, and receipt (default: ./build/repro-5min)
  --cpu-only    Skip live Metal / GB10 GPU hardware checks
  --live-gpu    Require live Metal / GB10 GPU checks (fail closed if unavailable)
  --json        Emit valid JSON summary to stdout on both success and failure
  -h, --help    Show this help message
EOF
      exit 0
      ;;
    *)
      PARSE_ERR="unknown flag: $1"
      break
      ;;
  esac
  shift
done

if [ -z "$PARSE_ERR" ] && [ "$CPU_ONLY" -eq 1 ] && [ "$LIVE_GPU" -eq 1 ]; then
  PARSE_ERR="--cpu-only and --live-gpu are mutually exclusive"
fi

if [ -n "$OUT_ARG" ]; then
  case "$OUT_ARG" in
    /*) OUT=$OUT_ARG ;;
    *)  OUT="$caller_pwd/$OUT_ARG" ;;
  esac
elif [ -n "${OUT:-}" ]; then
  case "$OUT" in
    /*) : ;;
    *)  OUT="$caller_pwd/$OUT" ;;
  esac
else
  OUT="$here/build/repro-5min"
fi

mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

START_TS=$(date -u +%s)
DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
OS=$(uname -s)
ARCH=$(uname -m)

RECEIPT_FILE="$OUT/repro-5min-receipt.txt"
LOG_FILE="$OUT/repro-5min.log"
: > "$LOG_FILE"

work=$(mktemp -d "${TMPDIR:-/tmp}/omega-helix-repro5.XXXXXX")
STEP_SEQ=0
RECEIPT_WRITTEN=0
JSON_EMITTED=0
VERDICT="FAIL"
FAIL_REASON=""

OMC_PATH=""
OMC_SHA256="none"
BIN_LINES=""
BIN_JSON_LINES=""
LINKAGE_LINES=""
ACCEL_STATUS="NOT_RUN"
ACCEL_DETAILS="not run"
DISASM_STATUS="NOT_APPLICABLE"
DISASM_DETAILS="not applicable on $OS/$ARCH"
ORACLE_STATUS="NOT_RUN"
ORACLE_DETAILS="not run"
NEG_STATUS="NOT_RUN"
NEG_DETAILS="not run"
RECEIPT_SHA256=""

detect_machine_os() {
  if [ "$OS" = Darwin ]; then
    hw_model=$(sysctl -n hw.model 2>/dev/null || printf 'Apple')
    cpu_brand=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || printf 'Apple Silicon')
    os_ver=$(sw_vers -productVersion 2>/dev/null || uname -r)
    printf 'macOS %s (%s / %s / %s)' "$os_ver" "$ARCH" "$hw_model" "$cpu_brand"
  else
    k_rel=$(uname -r)
    if [ -r /proc/device-tree/model ]; then
      dt_model=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || true)
    else
      dt_model=""
    fi
    if [ -r /etc/os-release ]; then
      pretty=$(sed -n 's/^PRETTY_NAME=//p' /etc/os-release | head -n 1 | tr -d '"')
    else
      pretty=""
    fi
    if [ -n "$dt_model" ] && [ -n "$pretty" ]; then
      printf '%s (%s / %s / %s)' "$pretty" "$ARCH" "$dt_model" "$k_rel"
    elif [ -n "$pretty" ]; then
      printf '%s (%s / %s)' "$pretty" "$ARCH" "$k_rel"
    else
      printf '%s %s (%s)' "$OS" "$ARCH" "$k_rel"
    fi
  fi
}

detect_git_commit() {
  if command -v git >/dev/null 2>&1 && git -C "$here" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    sha=$(git -C "$here" rev-parse --short=12 HEAD 2>/dev/null || printf 'unknown')
    if [ -n "$(git -C "$here" status --porcelain 2>/dev/null || true)" ]; then
      printf '%s-dirty' "$sha"
    else
      printf '%s' "$sha"
    fi
  else
    printf 'unknown'
  fi
}

MACHINE_OS=$(detect_machine_os)
GIT_COMMIT=$(detect_git_commit)

sha256_file() {
  f=$1
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$f" | awk '{print $1}'
  else
    python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$f"
  fi
}

log_msg() {
  printf '%s\n' "$1" >> "$LOG_FILE"
  if [ "$JSON_MODE" -eq 0 ]; then
    printf '%s\n' "$1"
  fi
}

append_line() {
  cur=$1
  add=$2
  if [ -z "$cur" ]; then
    printf '%s' "$add"
  else
    printf '%s\n%s' "$cur" "$add"
  fi
}

record_binary() {
  bname=$1
  bpath=$2
  bbytes=$(wc -c < "$bpath" | tr -d ' ')
  bsha=$(sha256_file "$bpath")
  BIN_LINES=$(append_line "$BIN_LINES" "  $bname: sha256=$bsha bytes=$bbytes")
  BIN_JSON_LINES=$(append_line "$BIN_JSON_LINES" "$bname|$bsha|$bbytes|$bpath")
}

# Run command without masking its exit code (no pipes).
# Sets LAST_RC and LAST_OUT_FILE. Pass QUIET_RC=1 for expected-fail negative controls.
run_logged() {
  label=$1
  shift
  quiet_rc=${QUIET_RC:-0}
  QUIET_RC=0
  STEP_SEQ=$((STEP_SEQ + 1))
  LAST_OUT_FILE="$work/step_${STEP_SEQ}.out"
  set +e
  "$@" > "$LAST_OUT_FILE" 2>&1
  LAST_RC=$?
  set -e
  {
    printf '=== %s (rc=%s) ===\n' "$label" "$LAST_RC"
    cat "$LAST_OUT_FILE"
    printf '\n'
  } >> "$LOG_FILE"
  if [ "$quiet_rc" -eq 0 ] && [ "$JSON_MODE" -eq 0 ] && [ "$LAST_RC" -ne 0 ]; then
    cat "$LAST_OUT_FILE" >&2
  fi
  return 0
}

write_receipt() {
  end_ts=$(date -u +%s)
  ELAPSED_SEC=$((end_ts - START_TS))
  verdict_line="VERDICT=FAIL"
  if [ "$VERDICT" = "PASS" ]; then
    verdict_line="VERDICT=REPRO_5MIN_PASS"
  fi
  omc_disp=$OMC_PATH
  if [ -n "${HOME:-}" ]; then
    case "$omc_disp" in
      "$HOME"/*) omc_disp="~${omc_disp#"$HOME"}" ;;
    esac
  fi
  tmp_rec="$RECEIPT_FILE.tmp"
  {
    printf '=== Omega Helix 5-Minute External Reproducer Receipt ===\n'
    printf 'Date (UTC): %s\n' "$DATE_UTC"
    printf 'Machine / OS: %s\n' "$MACHINE_OS"
    printf 'Git Commit: %s\n' "$GIT_COMMIT"
    if [ -n "$OMC_PATH" ]; then
      printf 'Compiler (omc): %s (sha256=%s)\n' "$omc_disp" "$OMC_SHA256"
    else
      printf 'Compiler (omc): NOT_FOUND\n'
    fi
    printf 'Binary SHA-256s:\n'
    if [ -n "$BIN_LINES" ]; then
      printf '%s\n' "$BIN_LINES"
    else
      printf '  none\n'
    fi
    printf 'Linkage Proof:\n'
    if [ -n "$LINKAGE_LINES" ]; then
      printf '%s\n' "$LINKAGE_LINES"
    else
      printf '  none\n'
    fi
    printf 'Accelerator Checks: %s (%s)\n' "$ACCEL_STATUS" "$ACCEL_DETAILS"
    if [ "$DISASM_STATUS" != "NOT_APPLICABLE" ]; then
      printf 'SM121 Disassembly: %s (%s)\n' "$DISASM_STATUS" "$DISASM_DETAILS"
    fi
    printf 'Oracle Parity: %s (%s)\n' "$ORACLE_STATUS" "$ORACLE_DETAILS"
    printf 'Negative Controls: %s (%s)\n' "$NEG_STATUS" "$NEG_DETAILS"
    if [ -n "$FAIL_REASON" ]; then
      printf 'Failure Reason: %s\n' "$FAIL_REASON"
    fi
    printf 'Elapsed Seconds: %s\n' "$ELAPSED_SEC"
    printf '%s\n' "$verdict_line"
  } > "$tmp_rec"
  mv -f "$tmp_rec" "$RECEIPT_FILE"
  RECEIPT_SHA256=$(sha256_file "$RECEIPT_FILE")
  RECEIPT_WRITTEN=1
}

emit_json_summary() {
  if [ "$JSON_EMITTED" -eq 1 ]; then
    return 0
  fi
  JSON_EMITTED=1
  verdict_str="FAIL"
  ok_flag="0"
  if [ "$VERDICT" = "PASS" ]; then
    verdict_str="REPRO_5MIN_PASS"
    ok_flag="1"
  fi
  R_OK="$ok_flag" \
  R_VERDICT="$verdict_str" \
  R_FAIL_REASON="$FAIL_REASON" \
  R_DATE_UTC="$DATE_UTC" \
  R_MACHINE_OS="$MACHINE_OS" \
  R_OS="$OS" \
  R_ARCH="$ARCH" \
  R_GIT_COMMIT="$GIT_COMMIT" \
  R_OMC_PATH="$OMC_PATH" \
  R_OMC_SHA256="$OMC_SHA256" \
  R_BINARIES="$BIN_JSON_LINES" \
  R_LINKAGE="$LINKAGE_LINES" \
  R_ACCEL_STATUS="$ACCEL_STATUS" \
  R_ACCEL_DETAILS="$ACCEL_DETAILS" \
  R_DISASM_STATUS="$DISASM_STATUS" \
  R_DISASM_DETAILS="$DISASM_DETAILS" \
  R_ORACLE_STATUS="$ORACLE_STATUS" \
  R_ORACLE_DETAILS="$ORACLE_DETAILS" \
  R_NEG_STATUS="$NEG_STATUS" \
  R_NEG_DETAILS="$NEG_DETAILS" \
  R_ELAPSED="$ELAPSED_SEC" \
  R_RECEIPT_PATH="$RECEIPT_FILE" \
  R_RECEIPT_SHA256="$RECEIPT_SHA256" \
  R_LOG_PATH="$LOG_FILE" \
  python3 - <<'PY'
import json
import os

bins = {}
for line in os.environ.get("R_BINARIES", "").splitlines():
    if not line.strip():
        continue
    name, sha, size_s, path = line.split("|", 3)
    bins[name] = {
        "sha256": sha,
        "bytes": int(size_s),
        "path": path,
    }

linkage = [ln.strip() for ln in os.environ.get("R_LINKAGE", "").splitlines() if ln.strip()]

payload = {
    "ok": os.environ["R_OK"] == "1",
    "verdict": os.environ["R_VERDICT"],
    "fail_reason": os.environ["R_FAIL_REASON"] or None,
    "date_utc": os.environ["R_DATE_UTC"],
    "machine_os": os.environ["R_MACHINE_OS"],
    "os": os.environ["R_OS"],
    "arch": os.environ["R_ARCH"],
    "git_commit": os.environ["R_GIT_COMMIT"],
    "omc": os.environ["R_OMC_PATH"] or None,
    "omc_sha256": os.environ["R_OMC_SHA256"],
    "binaries": bins,
    "linkage_proof": linkage,
    "accelerator": {
        "status": os.environ["R_ACCEL_STATUS"],
        "details": os.environ["R_ACCEL_DETAILS"],
    },
    "gpu_disasm": {
        "status": os.environ["R_DISASM_STATUS"],
        "details": os.environ["R_DISASM_DETAILS"],
    },
    "oracle_parity": {
        "status": os.environ["R_ORACLE_STATUS"],
        "details": os.environ["R_ORACLE_DETAILS"],
    },
    "negative_controls": {
        "status": os.environ["R_NEG_STATUS"],
        "details": os.environ["R_NEG_DETAILS"],
    },
    "elapsed_seconds": int(os.environ["R_ELAPSED"]),
    "receipt_path": os.environ["R_RECEIPT_PATH"],
    "receipt_sha256": os.environ["R_RECEIPT_SHA256"],
    "log_path": os.environ["R_LOG_PATH"],
}
print(json.dumps(payload, indent=2))
PY
}

fail_exit() {
  reason=$1
  code=${2:-1}
  VERDICT="FAIL"
  FAIL_REASON=$reason
  write_receipt
  if [ "$JSON_MODE" -eq 1 ]; then
    emit_json_summary
  else
    printf 'ERROR: %s\n' "$reason" >&2
    cat "$RECEIPT_FILE"
    printf 'Receipt File: %s\n' "$RECEIPT_FILE"
    printf 'Receipt SHA-256: %s\n' "$RECEIPT_SHA256"
  fi
  exit "$code"
}

cleanup_on_exit() {
  exit_rc=$?
  if [ "$RECEIPT_WRITTEN" -eq 0 ]; then
    VERDICT="FAIL"
    if [ -z "$FAIL_REASON" ]; then
      FAIL_REASON="aborted unexpectedly (exit=$exit_rc)"
    fi
    write_receipt
    if [ "$JSON_MODE" -eq 1 ]; then
      emit_json_summary
    fi
  fi
  rm -rf "$work"
}
trap cleanup_on_exit EXIT

if [ -n "$PARSE_ERR" ]; then
  fail_exit "$PARSE_ERR" 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  fail_exit "python3 (stdlib) is required for independent oracle verification" 1
fi

find_omc() {
  if [ -n "${OMC:-}" ]; then
    if [ -x "$OMC" ]; then
      printf '%s\n' "$OMC"
      return 0
    fi
    return 1
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
  return 1
}

if ! OMC_PATH=$(find_omc); then
  if [ -n "${OMC:-}" ]; then
    fail_exit "OMC='$OMC' is not executable" 1
  fi
  fail_exit "seeded 'omc' compiler not found (checked OMC, PATH, ~/.local/bin/omc, ~/canon/omega-mv/seed/out/omc)" 1
fi
export OMC="$OMC_PATH"
OMC_SHA256=$(sha256_file "$OMC_PATH")

is_omega_ng() {
  cand=$1
  [ -n "$cand" ] && [ -x "$cand" ] || return 1
  hdr=$(printf '@ng front-ir\nfn main to int\nret 0\nend\n' | "$cand" 2>/dev/null | dd bs=1 count=6 2>/dev/null | tr -d '\0' || true)
  [ "$hdr" = "OSC1IR" ]
}

has_ng_backend() {
  if [ -n "${NG:-}" ]; then
    if is_omega_ng "$NG"; then
      return 0
    fi
    return 2
  fi
  if is_omega_ng "$OUT/ng"; then
    return 0
  fi
  if [ -n "${OMEGA_MV:-}" ]; then
    if [ -f "$OMEGA_MV/seed/ng/src/ir.om" ] && [ -f "$OMEGA_MV/seed/ng/src/a64.om" ]; then
      return 0
    fi
    return 2
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

build_target() {
  tgt=$1
  log_msg "=== Building TARGET=$tgt ==="
  run_logged "build:$tgt" env OUT="$OUT" TARGET="$tgt" OMC="$OMC_PATH" sh "$here/scripts/build.sh"
  if [ "$LAST_RC" -ne 0 ]; then
    fail_exit "build failed for TARGET=$tgt (rc=$LAST_RC)" "$LAST_RC"
  fi
}

check_darwin_linkage() {
  bname=$1
  bpath=$2
  run_logged "file:$bname" file "$bpath"
  if [ "$LAST_RC" -ne 0 ]; then
    fail_exit "file command failed on $bpath" "$LAST_RC"
  fi
  f_line=$(tr '\n' ' ' < "$LAST_OUT_FILE")
  if ! printf '%s' "$f_line" | grep -q 'Mach-O' || ! printf '%s' "$f_line" | grep -q 'arm64'; then
    fail_exit "unexpected file output for $bname: $f_line" 1
  fi
  run_logged "otool:$bname" otool -L "$bpath"
  if [ "$LAST_RC" -ne 0 ]; then
    fail_exit "otool -L failed on $bpath" "$LAST_RC"
  fi
  if grep -qiE 'cuda|cublas|cudnn' "$LAST_OUT_FILE"; then
    fail_exit "CUDA dependency found in $bname" 1
  fi
  deps=$(awk 'NR > 1 && NF > 0 { print $1 }' "$LAST_OUT_FILE" | tr '\n' ',' | sed 's/,$//')
  if [ "$deps" != "/usr/lib/libSystem.B.dylib" ]; then
    fail_exit "unexpected dynamic dependencies in $bname: $deps (expected /usr/lib/libSystem.B.dylib only)" 1
  fi
  LINKAGE_LINES=$(append_line "$LINKAGE_LINES" "  $bname: Mach-O arm64, deps=[$deps], zero CUDA")
}

check_linux_static_linkage() {
  bname=$1
  bpath=$2
  expected_arch=$3
  run_logged "file:$bname" file "$bpath"
  if [ "$LAST_RC" -ne 0 ]; then
    fail_exit "file command failed on $bpath" "$LAST_RC"
  fi
  f_line=$(tr '\n' ' ' < "$LAST_OUT_FILE")
  if ! printf '%s' "$f_line" | grep -q 'ELF' || ! printf '%s' "$f_line" | grep -qi "$expected_arch"; then
    fail_exit "unexpected ELF header for $bname: $f_line" 1
  fi
  if ! command -v ldd >/dev/null 2>&1; then
    fail_exit "ldd not found on Linux host to verify static ELF linkage" 1
  fi
  set +e
  ldd "$bpath" > "$work/ldd_${bname}.out" 2>&1
  ldd_rc=$?
  set -e
  {
    printf '=== ldd:%s (rc=%s) ===\n' "$bname" "$ldd_rc"
    cat "$work/ldd_${bname}.out"
    printf '\n'
  } >> "$LOG_FILE"
  if grep -qiE 'libc\.so|libcuda|libcublas|libcudnn' "$work/ldd_${bname}.out"; then
    fail_exit "dynamic libc/CUDA dependency detected in $bname" 1
  fi
  ldd_msg=$(tr '\n' ' ' < "$work/ldd_${bname}.out" | sed 's/[[:space:]]\+/ /g; s/ $//')
  LINKAGE_LINES=$(append_line "$LINKAGE_LINES" "  $bname: static ELF ($expected_arch), ldd='$ldd_msg', zero libc/CUDA")
}

run_oracle_parity() {
  bin_cpu=$1
  ix_bin=$2
  log_msg "=== Running independent Python f64 oracle parity (tests/oracle.py) ==="
  python3 - "$work/raw.txt" <<'PY'
import sys
s = "Verify Omega Helix against independent IEEE-754 f64 arithmetic.\n<|endoftext|>\n" * 30
open(sys.argv[1], "w", encoding="utf-8").write(s)
PY
  for step_cmd in xprep xinit xgrad xstep xeval; do
    case "$step_cmd" in
      xprep)
        run_logged "oracle:xprep" python3 "$here/tests/oracle.py" xprep --ollm "$bin_cpu" --data "$work/raw.txt" --outdir "$work"
        ;;
      xinit)
        run_logged "oracle:xinit" python3 "$here/tests/oracle.py" xinit --ollm "$bin_cpu" --cfg small
        ;;
      xgrad)
        run_logged "oracle:xgrad" python3 "$here/tests/oracle.py" xgrad --ollm "$bin_cpu" --data "$work/ollm-train.bin" --cfg small
        ;;
      xstep)
        run_logged "oracle:xstep" python3 "$here/tests/oracle.py" xstep --ollm "$bin_cpu" --data "$work/ollm-train.bin" --cfg small
        ;;
      xeval)
        run_logged "oracle:xeval" python3 "$here/tests/oracle.py" xeval --ollm "$bin_cpu" --held "$work/ollm-held.bin" --cfg small
        ;;
    esac
    if [ "$LAST_RC" -ne 0 ]; then
      fail_exit "tests/oracle.py $step_cmd failed (rc=$LAST_RC)" "$LAST_RC"
    fi
  done

  # Exercise interchange upgrade path on a fresh checkpoint
  run_logged "init:base" "$bin_cpu" init "$work/base.ck" 256 16 16 2 32 2 2 42 4 3000
  if [ "$LAST_RC" -ne 0 ]; then
    fail_exit "ollm init failed (rc=$LAST_RC)" "$LAST_RC"
  fi
  run_logged "interchange:upgrade" "$ix_bin" upgrade "$work/base.ck" "$work/base.ck4"
  if [ "$LAST_RC" -ne 0 ]; then
    fail_exit "ollm-interchange upgrade failed (rc=$LAST_RC)" "$LAST_RC"
  fi
  ORACLE_STATUS="PASS"
  ORACLE_DETAILS="xprep, xinit, xgrad, xstep, xeval PASS (f64 rtol<=1e-9); interchange upgrade PASS"
}

run_negative_controls() {
  bin_cpu=$1
  ix_bin=$2
  dna_bin=${3:-}
  log_msg "=== Running negative controls ==="

  # 1. OLLMSP01 1-bit tamper -> rc=61
  run_logged "neg:growpro" "$bin_cpu" growpro "$work/base.ck" "$work/spec.ck" 3 4 2000
  if [ "$LAST_RC" -ne 0 ]; then
    fail_exit "growpro setup failed (rc=$LAST_RC)" "$LAST_RC"
  fi
  run_logged "neg:spext" "$bin_cpu" spext "$work/base.ck" "$work/spec.ck" "$work/skill.sp"
  if [ "$LAST_RC" -ne 0 ]; then
    fail_exit "spext setup failed (rc=$LAST_RC)" "$LAST_RC"
  fi
  python3 - "$work/skill.sp" "$work/tampered.sp" <<'PY'
import sys
b = bytearray(open(sys.argv[1], "rb").read())
b[48] ^= 0x01
open(sys.argv[2], "wb").write(b)
PY
  QUIET_RC=1 run_logged "neg:ollmsp01_tamper" "$bin_cpu" spapp "$work/base.ck" "$work/tampered.sp" "$work/out.ck"
  rc_sp=$LAST_RC
  if [ "$rc_sp" -ne 61 ]; then
    fail_exit "negative control OLLMSP01 1-bit tamper expected rc=61, got rc=$rc_sp" 1
  fi

  # 2. OLLM_ORACLE_MUTANT=tiemaxa -> rc!=0
  QUIET_RC=1 run_logged "neg:oracle_mutant" env OLLM_ORACLE_MUTANT=tiemaxa python3 "$here/tests/oracle.py" selftest-v1
  rc_mut=$LAST_RC
  if [ "$rc_mut" -eq 0 ]; then
    fail_exit "negative control OLLM_ORACLE_MUTANT=tiemaxa unexpectedly passed (rc=0)" 1
  fi
  neg_summary="OLLMSP01_tamper=rc:$rc_sp(PASS); OLLM_ORACLE_MUTANT=tiemaxa:rc:$rc_mut(PASS)"

  # 3. OLLMBF01 1-bit tamper -> rc=61 (when present in checkout)
  if [ -n "$dna_bin" ] && [ -x "$dna_bin" ] && grep -q 'import-hf-bf16' "$here/native/univ/interchange.om" 2>/dev/null; then
    mkdir -p "$work/hf_bf16"
    python3 - "$work/hf_bf16" <<'PY'
import json
import os
import struct
import sys

out_dir = sys.argv[1]
V, D, H, H_KV, F, L = 16, 16, 2, 2, 32, 1
cfg = {
    "model_type": "llama",
    "architectures": ["LlamaForCausalLM"],
    "vocab_size": V,
    "hidden_size": D,
    "num_attention_heads": H,
    "num_key_value_heads": H_KV,
    "intermediate_size": F,
    "num_hidden_layers": L,
    "max_position_embeddings": 16,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
    "hidden_act": "silu",
    "torch_dtype": "bfloat16",
}
with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump(cfg, f)

def bf16_ones(n):
    return b"\x80\x3f" * n

def bf16_zeros(n):
    return b"\x00\x00" * n

tensors = [
    ("model.embed_tokens.weight", [V, D], bf16_ones(V * D)),
    ("model.norm.weight", [D], bf16_ones(D)),
    ("model.layers.0.input_layernorm.weight", [D], bf16_ones(D)),
    ("model.layers.0.self_attn.q_proj.weight", [D, D], bf16_zeros(D * D)),
    ("model.layers.0.self_attn.k_proj.weight", [D, D], bf16_zeros(D * D)),
    ("model.layers.0.self_attn.v_proj.weight", [D, D], bf16_zeros(D * D)),
    ("model.layers.0.self_attn.o_proj.weight", [D, D], bf16_zeros(D * D)),
    ("model.layers.0.post_attention_layernorm.weight", [D], bf16_ones(D)),
    ("model.layers.0.mlp.gate_proj.weight", [F, D], bf16_zeros(F * D)),
    ("model.layers.0.mlp.up_proj.weight", [F, D], bf16_zeros(F * D)),
    ("model.layers.0.mlp.down_proj.weight", [D, F], bf16_zeros(D * F)),
]
hdr = {}
payload = bytearray()
off = 0
for name, shape, data in tensors:
    hdr[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + len(data)]}
    payload.extend(data)
    off += len(data)
hdr_b = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
while len(hdr_b) % 8 != 0:
    hdr_b += b" "
with open(os.path.join(out_dir, "model.safetensors"), "wb") as f:
    f.write(struct.pack("<Q", len(hdr_b)))
    f.write(hdr_b)
    f.write(payload)
PY
    run_logged "neg:import_hf_bf16" "$ix_bin" import-hf-bf16 "$work/hf_bf16" "$work/model.bf16"
    if [ "$LAST_RC" -ne 0 ]; then
      fail_exit "OLLMBF01 negative control setup import-hf-bf16 failed (rc=$LAST_RC)" "$LAST_RC"
    fi
    python3 - "$work/model.bf16" "$work/tampered.bf16" <<'PY'
import sys
b = bytearray(open(sys.argv[1], "rb").read())
idx = min(300, len(b) - 1)
b[idx] ^= 0x01
open(sys.argv[2], "wb").write(b)
PY
    QUIET_RC=1 run_logged "neg:ollmbf01_info_tamper" "$ix_bin" info-bf16 "$work/tampered.bf16"
    rc_bf_ix=$LAST_RC
    QUIET_RC=1 run_logged "neg:ollmbf01_dna_tamper" "$dna_bin" info "$work/tampered.bf16"
    rc_bf_dna=$LAST_RC
    if [ "$rc_bf_ix" -ne 61 ] || [ "$rc_bf_dna" -ne 61 ]; then
      fail_exit "negative control OLLMBF01 tamper expected rc=61, got info-bf16=$rc_bf_ix dna-info=$rc_bf_dna" 1
    fi
    neg_summary="$neg_summary; OLLMBF01_tamper=rc:61(PASS)"
  else
    neg_summary="$neg_summary; OLLMBF01_tamper=SKIPPED_NOT_IN_CHECKOUT"
  fi

  # 4. OLLMCK05 1-bit tamper -> rc=14 (when ckfast is present in checkout)
  if grep -q 'ckfast' "$here/native/cli.om" 2>/dev/null; then
    run_logged "neg:ckfast_create" "$bin_cpu" ckfast "$work/base.ck" "$work/base.ck5" f32
    if [ "$LAST_RC" -ne 0 ]; then
      fail_exit "OLLMCK05 negative control setup ckfast failed (rc=$LAST_RC)" "$LAST_RC"
    fi
    python3 - "$work/base.ck5" "$work/tampered.ck5" <<'PY'
import sys
b = bytearray(open(sys.argv[1], "rb").read())
idx = min(260, len(b) - 1)
b[idx] ^= 0x01
open(sys.argv[2], "wb").write(b)
PY
    QUIET_RC=1 run_logged "neg:ollmck05_tamper" "$bin_cpu" ckf64 "$work/tampered.ck5" "$work/tampered_ck5.ck"
    rc_ck5=$LAST_RC
    if [ "$rc_ck5" -ne 14 ]; then
      fail_exit "negative control OLLMCK05 tamper expected rc=14, got rc=$rc_ck5" 1
    fi
    neg_summary="$neg_summary; OLLMCK05_tamper=rc:$rc_ck5(PASS)"
  else
    neg_summary="$neg_summary; OLLMCK05_tamper=SKIPPED_NOT_IN_CHECKOUT"
  fi

  NEG_STATUS="PASS"
  NEG_DETAILS="$neg_summary"
}

run_darwin_arm64() {
  build_target macos
  CPU_BIN="$OUT/ollm-macos"
  record_binary "ollm-macos" "$CPU_BIN"
  check_darwin_linkage "ollm-macos" "$CPU_BIN"

  build_target interchange
  IX_BIN="$OUT/ollm-interchange"
  record_binary "ollm-interchange" "$IX_BIN"
  check_darwin_linkage "ollm-interchange" "$IX_BIN"

  DNA_BIN=""
  if [ -f "$here/native/univ/spdna.om" ]; then
    build_target dna
    DNA_BIN="$OUT/ollm-dna"
    record_binary "ollm-dna" "$DNA_BIN"
    check_darwin_linkage "ollm-dna" "$DNA_BIN"
  fi

  if [ "$CPU_ONLY" -eq 1 ]; then
    ACCEL_STATUS="METAL_STATUS=SKIPPED_CPU_ONLY"
    ACCEL_DETAILS="Apple Silicon Metal skipped (--cpu-only)"
  else
    set +e
    has_ng_backend
    ng_rc=$?
    set -e
    if [ "$ng_rc" -eq 2 ]; then
      fail_exit "NG or OMEGA_MV was set explicitly but is invalid" 1
    elif [ "$ng_rc" -eq 0 ]; then
      build_target macos-metal
      METAL_BIN="$OUT/ollm-metal"
      record_binary "ollm-metal" "$METAL_BIN"
      check_darwin_linkage "ollm-metal" "$METAL_BIN"

      log_msg "=== Running live Apple Silicon Metal GPU checks ==="
      run_logged "metal:mdev" "$METAL_BIN" mdev
      if [ "$LAST_RC" -eq 75 ]; then
        if [ "$LIVE_GPU" -eq 1 ]; then
          fail_exit "Metal GPU required (--live-gpu) but mdev returned rc=75 (no Metal device)" 1
        fi
        ACCEL_STATUS="METAL_STATUS=SKIPPED_NO_DEVICE"
        ACCEL_DETAILS="ollm-metal built; mdev rc=75 (headless/no Metal device)"
      elif [ "$LAST_RC" -ne 0 ] || ! grep -q 'PASS' "$LAST_OUT_FILE"; then
        fail_exit "Metal mdev failed (rc=$LAST_RC)" 1
      else
        mdev_line=$(grep '^MDEV ' "$LAST_OUT_FILE" | head -n 1 || true)

        run_logged "metal:mgemm" "$METAL_BIN" mgemm 256 256 256 5
        if [ "$LAST_RC" -ne 0 ] || ! grep -q 'PASS' "$LAST_OUT_FILE"; then
          fail_exit "Metal mgemm 256x256x256x5 failed (rc=$LAST_RC)" 1
        fi
        mgemm_line=$(grep '^MGEMM ' "$LAST_OUT_FILE" | head -n 1 || true)

        python3 - "$work/mtl_corp.txt" <<'PY'
import sys
s = "Verify Omega Helix Apple Silicon Metal parity against IEEE-754 CPU f64.\n" * 64
open(sys.argv[1], "w", encoding="utf-8").write(s)
PY
        run_logged "metal:init" "$METAL_BIN" init "$work/mtl.ck" 256 16 16 2 32 2 2 1 10 3000 1
        if [ "$LAST_RC" -ne 0 ]; then
          fail_exit "Metal init failed (rc=$LAST_RC)" "$LAST_RC"
        fi

        run_logged "metal:gcmp" "$METAL_BIN" gcmp "$work/mtl.ck" "$work/mtl_corp.txt" 0
        if [ "$LAST_RC" -ne 0 ] || ! grep -q 'GCMP PASS' "$LAST_OUT_FILE"; then
          fail_exit "Metal gcmp vs CPU f64 failed (rc=$LAST_RC)" 1
        fi
        gcmp_line=$(grep '^GC ' "$LAST_OUT_FILE" | head -n 1 || true)

        run_logged "metal:cpu_eval" "$CPU_BIN" eval "$work/mtl.ck" "$work/mtl_corp.txt" 4
        if [ "$LAST_RC" -ne 0 ]; then
          fail_exit "CPU eval for Metal parity failed (rc=$LAST_RC)" "$LAST_RC"
        fi
        cpu_loss=$(awk '/^E / { print $4; exit }' "$LAST_OUT_FILE")

        run_logged "metal:geval" "$METAL_BIN" geval "$work/mtl.ck" "$work/mtl_corp.txt" 4
        if [ "$LAST_RC" -ne 0 ]; then
          fail_exit "Metal geval failed (rc=$LAST_RC)" "$LAST_RC"
        fi
        mtl_loss=$(awk '/^E / { print $4; exit }' "$LAST_OUT_FILE")
        loss_diff=$((mtl_loss - cpu_loss))
        if [ "$loss_diff" -lt 0 ]; then
          loss_diff=$(( -loss_diff ))
        fi
        if [ "$loss_diff" -ge 1000 ]; then
          fail_exit "Metal geval vs CPU eval diverged: cpu=$cpu_loss mtl=$mtl_loss diff=$loss_diff" 1
        fi

        run_logged "metal:gtrain" "$METAL_BIN" gtrain "$work/mtl.ck" "$work/mtl_corp.txt" 4 2
        if [ "$LAST_RC" -ne 0 ] || ! grep -q '^T 4 ' "$LAST_OUT_FILE" || ! grep -q '^MT ' "$LAST_OUT_FILE"; then
          fail_exit "Metal gtrain mini smoke failed (rc=$LAST_RC)" 1
        fi
        gtrain_mt=$(grep '^MT ' "$LAST_OUT_FILE" | head -n 1 || true)

        ACCEL_STATUS="METAL_STATUS=PASS"
        ACCEL_DETAILS="$mdev_line | $mgemm_line | $gcmp_line | eval_diff=${loss_diff}u | $gtrain_mt"
      fi
    else
      if [ "$LIVE_GPU" -eq 1 ]; then
        fail_exit "Metal GPU required (--live-gpu) but ng / OMEGA_MV was not found" 1
      fi
      ACCEL_STATUS="METAL_STATUS=SKIPPED_NO_NG"
      ACCEL_DETAILS="ng compiler / omega-mv seed/ng/src not found on host"
    fi
  fi

  run_oracle_parity "$CPU_BIN" "$IX_BIN"
  run_negative_controls "$CPU_BIN" "$IX_BIN" "$DNA_BIN"
}

run_linux_aarch64() {
  build_target gb10
  GB10_BIN="$OUT/ollm-gb10"
  record_binary "ollm-gb10" "$GB10_BIN"
  check_linux_static_linkage "ollm-gb10" "$GB10_BIN" "aarch64"

  build_target interchange
  IX_BIN="$OUT/ollm-interchange"
  record_binary "ollm-interchange" "$IX_BIN"
  check_linux_static_linkage "ollm-interchange" "$IX_BIN" "aarch64"

  DNA_BIN=""
  if [ -f "$here/native/univ/spdna.om" ]; then
    build_target dna
    DNA_BIN="$OUT/ollm-dna"
    record_binary "ollm-dna" "$DNA_BIN"
    check_linux_static_linkage "ollm-dna" "$DNA_BIN" "aarch64"
  fi

  NVDISASM=${NVDISASM:-$(command -v nvdisasm 2>/dev/null || printf '/usr/local/cuda/bin/nvdisasm')}
  if [ -x "$NVDISASM" ]; then
    log_msg "=== Running SM121 SASS disassembly (tests/gpu-disasm.sh --all) ==="
    run_logged "disasm:all" env OMC="$OMC_PATH" OUT="$OUT/gpu-disasm" NVDISASM="$NVDISASM" sh "$here/tests/gpu-disasm.sh" --all
    if [ "$LAST_RC" -ne 0 ] || ! grep -q 'bad=0' "$LAST_OUT_FILE"; then
      fail_exit "tests/gpu-disasm.sh --all failed (rc=$LAST_RC)" 1
    fi
    disasm_line=$(grep 'GPU_DISASM_OK' "$LAST_OUT_FILE" | head -n 1 || true)
    DISASM_STATUS="PASS"
    DISASM_DETAILS="${disasm_line:-bad=0}"
  elif [ "$LIVE_GPU" -eq 1 ]; then
    fail_exit "--live-gpu requires nvdisasm on Linux/aarch64 for SM121 SASS check" 1
  else
    DISASM_STATUS="SKIPPED_NO_NVDISASM"
    DISASM_DETAILS="nvdisasm not installed"
  fi

  run_oracle_parity "$GB10_BIN" "$IX_BIN"
  run_negative_controls "$GB10_BIN" "$IX_BIN" "$DNA_BIN"

  if [ "$CPU_ONLY" -eq 1 ]; then
    ACCEL_STATUS="GB10_STATUS=SKIPPED_CPU_ONLY"
    ACCEL_DETAILS="GB10 live GPU checks skipped (--cpu-only)"
  elif [ -e /dev/nvidia0 ] || [ -e /dev/aien-gpu ]; then
    log_msg "=== Running live DGX Spark GB10 GPU checks ==="
    run_logged "gb10:gnv4gemm" "$GB10_BIN" gnv4gemm 256 512 256
    if [ "$LAST_RC" -ne 0 ] || ! grep -q 'PASS' "$LAST_OUT_FILE"; then
      fail_exit "gnv4gemm 256 512 256 failed (rc=$LAST_RC)" 1
    fi
    nv4gemm_line=$(grep '^GNV4GEMM ' "$LAST_OUT_FILE" | head -n 1 || true)

    python3 - "$work/nv4_corp.txt" <<'PY'
import sys
s = "Verify Omega Helix GB10 NVFP4 parity against IEEE-754 CPU f64.\n" * 128
open(sys.argv[1], "w", encoding="utf-8").write(s)
PY
    run_logged "gb10:init_nv4" "$GB10_BIN" init "$work/nv4.ck" 256 64 256 4 512 2 2 1 10 3000 1
    if [ "$LAST_RC" -ne 0 ]; then
      fail_exit "init for gnv4cmp failed (rc=$LAST_RC)" "$LAST_RC"
    fi
    run_logged "gb10:gnv4cmp" "$GB10_BIN" gnv4cmp "$work/nv4.ck" "$work/nv4_corp.txt" 0
    if [ "$LAST_RC" -ne 0 ] || ! grep -q 'PASS' "$LAST_OUT_FILE"; then
      fail_exit "gnv4cmp failed (rc=$LAST_RC)" 1
    fi
    nv4cmp_line=$(grep '^GNV4CMP ' "$LAST_OUT_FILE" | head -n 1 || true)

    ck5_note="OLLMCK05_parity=SKIPPED_NO_CKFAST"
    if grep -q 'ckfast' "$here/native/cli.om" 2>/dev/null; then
      run_logged "gb10:ckfast" "$GB10_BIN" ckfast "$work/nv4.ck" "$work/nv4.ck5" bf16
      if [ "$LAST_RC" -ne 0 ]; then
        fail_exit "ckfast for OLLMCK05 parity failed (rc=$LAST_RC)" "$LAST_RC"
      fi
      run_logged "gb10:geval_ck2" "$GB10_BIN" geval "$work/nv4.ck" "$work/nv4_corp.txt" 2
      if [ "$LAST_RC" -ne 0 ]; then
        fail_exit "geval on OLLMCK02 failed (rc=$LAST_RC)" "$LAST_RC"
      fi
      e_ck2=$(grep '^E ' "$LAST_OUT_FILE" | head -n 1 || true)
      run_logged "gb10:geval_ck5" "$GB10_BIN" geval "$work/nv4.ck5" "$work/nv4_corp.txt" 2
      if [ "$LAST_RC" -ne 0 ]; then
        fail_exit "geval on OLLMCK05 failed (rc=$LAST_RC)" "$LAST_RC"
      fi
      e_ck5=$(grep '^E ' "$LAST_OUT_FILE" | head -n 1 || true)
      if [ "$e_ck2" != "$e_ck5" ]; then
        fail_exit "OLLMCK05 vs OLLMCK02 geval mismatch: ck2='$e_ck2' ck5='$e_ck5'" 1
      fi
      ck5_note="OLLMCK05_geval_parity=PASS($e_ck5)"
    fi

    ACCEL_STATUS="GB10_STATUS=PASS"
    ACCEL_DETAILS="$nv4gemm_line | $nv4cmp_line | $ck5_note"
  else
    if [ "$LIVE_GPU" -eq 1 ]; then
      fail_exit "--live-gpu requested on Linux/aarch64 but neither /dev/nvidia0 nor /dev/aien-gpu exists" 1
    fi
    ACCEL_STATUS="GB10_STATUS=SKIPPED_NO_GPU_DEV"
    ACCEL_DETAILS="/dev/nvidia0 and /dev/aien-gpu not present"
  fi
}

run_linux_x86_64() {
  build_target x86linux
  X86_BIN="$OUT/ollm-x86linux"
  record_binary "ollm-x86linux" "$X86_BIN"
  check_linux_static_linkage "ollm-x86linux" "$X86_BIN" "x86-64"

  build_target interchange
  IX_BIN="$OUT/ollm-interchange"
  record_binary "ollm-interchange" "$IX_BIN"
  check_linux_static_linkage "ollm-interchange" "$IX_BIN" "x86-64"

  DNA_BIN=""
  if [ -f "$here/native/univ/spdna.om" ]; then
    build_target dna
    DNA_BIN="$OUT/ollm-dna"
    record_binary "ollm-dna" "$DNA_BIN"
    check_linux_static_linkage "ollm-dna" "$DNA_BIN" "x86-64"
  fi

  if [ "$LIVE_GPU" -eq 1 ]; then
    fail_exit "--live-gpu requested on Linux/$ARCH, which has no GPU backend" 1
  fi
  ACCEL_STATUS="GPU_STATUS=SKIPPED_CPU_ONLY_PLATFORM"
  ACCEL_DETAILS="Linux/$ARCH fallback is CPU-only"

  run_oracle_parity "$X86_BIN" "$IX_BIN"
  run_negative_controls "$X86_BIN" "$IX_BIN" "$DNA_BIN"
}

case "$OS/$ARCH" in
  Darwin/arm64)
    run_darwin_arm64
    ;;
  Linux/aarch64)
    run_linux_aarch64
    ;;
  Linux/x86_64|Linux/amd64)
    run_linux_x86_64
    ;;
  *)
    fail_exit "unsupported host platform: $OS/$ARCH (supported: Darwin/arm64, Linux/aarch64, Linux/x86_64)" 2
    ;;
esac

VERDICT="PASS"
write_receipt
if [ "$JSON_MODE" -eq 1 ]; then
  emit_json_summary
else
  cat "$RECEIPT_FILE"
  printf 'Receipt File: %s\n' "$RECEIPT_FILE"
  printf 'Receipt SHA-256: %s\n' "$RECEIPT_SHA256"
fi
