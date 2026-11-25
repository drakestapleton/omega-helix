#!/bin/sh
# Independent verification suite for Omega Helix.
# Runs all host-verifiable builds, CPU engine self-checks, Python f64 referee oracle
# cross-checks, OpenWALDO dataset & provenance negative controls, tool suites, SM121
# SASS disassembly checks (when nvdisasm is available), and Lean 4 proofs (when lean
# is available). Does NOT touch the GB10 GPU hardware (safe outside a GPU window).
#
# Usage:
#   sh scripts/verify.sh
# Env:
#   OMC   Path to seeded omc compiler (auto-detected if unset)
#   OUT   Build directory (default: ./build/verify)
set -eu

here=$(cd "$(dirname "$0")/.." && pwd)
OUT=${OUT:-$here/build/verify}
mkdir -p "$OUT"

OS=$(uname -s)
ARCH=$(uname -m)
echo "=== Omega Helix Independent Verification ($OS $ARCH) ==="

echo "--- 1. Build native CPU, universal zero-cap engine, interchange & OpenWALDO ingestor ---"
if [ "$OS" = Darwin ]; then
  OUT="$OUT" TARGET=macos sh "$here/scripts/build.sh"
  OUT="$OUT" TARGET=tok sh "$here/scripts/build.sh"
  OUT="$OUT" TARGET=openwaldo sh "$here/scripts/build.sh"
  OUT="$OUT" TARGET=univ sh "$here/scripts/build.sh"
  OUT="$OUT" TARGET=interchange sh "$here/scripts/build.sh"
  BIN="$OUT/ollm-macos"
else
  OUT="$OUT" TARGET=aarch64 sh "$here/scripts/build.sh"
  OUT="$OUT" TARGET=gb10 sh "$here/scripts/build.sh"
  OUT="$OUT" TARGET=x86linux sh "$here/scripts/build.sh"
  OUT="$OUT" TARGET=x86efi sh "$here/scripts/build.sh"
  OUT="$OUT" TARGET=a64efi sh "$here/scripts/build.sh"
  OUT="$OUT" TARGET=tok sh "$here/scripts/build.sh"
  OUT="$OUT" TARGET=wtrain sh "$here/scripts/build.sh"
  OUT="$OUT" TARGET=openwaldo sh "$here/scripts/build.sh"
  OUT="$OUT" TARGET=univ sh "$here/scripts/build.sh"
  OUT="$OUT" TARGET=interchange sh "$here/scripts/build.sh"
  BIN="$OUT/ollm"
  if command -v ldd >/dev/null 2>&1; then
    if ldd "$OUT/ollm-gb10" 2>&1 | grep -qiE 'libc|libcuda|cublas'; then
      echo "ERROR: ollm-gb10 must be statically linked with zero CUDA/libc" >&2
      exit 1
    fi
    echo "STATIC_ELF_ZERO_CUDA_OK"
  fi
fi

echo "--- 2. End-to-end CPU smoke suite (adamck, gcheck, train, eval, BPE, kvcmp, DNA, loop) ---"
sh "$here/scripts/quickstart-cpu.sh" "$BIN"

echo "--- 3. Python f64 referee oracle selftests & binary cross-checks ---"
python3 "$here/tests/oracle.py" selftest
python3 "$here/tests/oracle.py" selftest-v1

xdir=$(mktemp -d "${TMPDIR:-/tmp}/omega-helix-oracle.XXXXXX")
trap 'rm -rf "$xdir"' EXIT
python3 - "$xdir/raw.txt" <<'PY'
import sys
s = "Once upon a time there was a verifiable pure-Omega language model.\n<|endoftext|>\n" * 30
open(sys.argv[1], "w", encoding="utf-8").write(s)
PY
python3 "$here/tests/oracle.py" xprep --ollm "$BIN" --data "$xdir/raw.txt" --outdir "$xdir"
python3 "$here/tests/oracle.py" xbase --ollm "$BIN" --train "$xdir/ollm-train.bin" --held "$xdir/ollm-held.bin"
python3 "$here/tests/oracle.py" xinit --ollm "$BIN" --cfg small
python3 "$here/tests/oracle.py" xgrad --ollm "$BIN" --data "$xdir/ollm-train.bin" --cfg small
python3 "$here/tests/oracle.py" xstep --ollm "$BIN" --data "$xdir/ollm-train.bin" --cfg small
python3 "$here/tests/oracle.py" xeval --ollm "$BIN" --held "$xdir/ollm-held.bin" --cfg small

echo "--- 3b. Universal zero-cap engine oracle (llama/qwen2/qwen3/mistral/gemma/gpt/gelu/loop/u32/caps) ---"
python3 "$here/tests/oracle_universal.py" --bin "$OUT/ollm-univ"

echo "--- 3c. Safetensors / GGUF v3 / CK01-03 upgrade interchange suite ---"
IX_BIN="$OUT/ollm-interchange" UNIV_BIN="$OUT/ollm-univ" LEGACY_BIN="$BIN" \
  python3 -W error::ResourceWarning "$here/tests/test_interchange.py" -v

echo "--- 3d. Adaptive looped core (per-token exit & inter-pass messages) suite ---"
OLLM_BIN="$BIN" python3 "$here/tests/test_adaptive_loop.py"

echo "--- 4. OpenWALDO dataset verifier, pure-Omega ingestor, multi-arch backend & BOM ---"
OLLM_WALDO_BIN="$OUT/ollm-waldo" OLLM_UNIV_BIN="$OUT/ollm-univ" \
OLLM_INTERCHANGE_BIN="$OUT/ollm-interchange" OLLM_BIN="$BIN" \
  python3 -m unittest discover -s "$here/tools/openwaldo" -p "test_*.py" -v

echo "--- 4b. Omega Helix Studio, OpenWALDO catalog, installer CLI & SSE forge suite ---"
OLLM_WALDO_BIN="$OUT/ollm-waldo" OLLM_UNIV_BIN="$OUT/ollm-univ" \
OLLM_INTERCHANGE_BIN="$OUT/ollm-interchange" OLLM_BIN="$BIN" \
  python3 -W error::ResourceWarning -m unittest discover -s "$here/tools/studio" -p "test_*.py" -v
OMEGA_HELIX_BIN_DIR="$OUT" "$here/bin/omega-helix" doctor --json >/dev/null
OMEGA_HELIX_BIN_DIR="$OUT" "$here/bin/omega-helix" catalog --search permissive --json >/dev/null

echo "--- 4c. Sovereign Distillation Pipeline Network (OLLMDK01, verifiers, teachers, Studio API, CLI) ---"
OLLM_WALDO_BIN="$OUT/ollm-waldo" OLLM_UNIV_BIN="$OUT/ollm-univ" \
OLLM_INTERCHANGE_BIN="$OUT/ollm-interchange" OLLM_BIN="$BIN" \
  python3 -W error::ResourceWarning -m unittest discover -s "$here/tools/distill" -p "test_*.py" -v
OMEGA_HELIX_BIN_DIR="$OUT" "$here/bin/omega-helix" distill probe --json >/dev/null

echo "--- 4d. Sovereign Runtime Firewalls, Grammar Decoding, Sleep Cycle, Compile-Until-Green & HF Demo ---"
OLLM_WALDO_BIN="$OUT/ollm-waldo" OLLM_UNIV_BIN="$OUT/ollm-univ" \
OLLM_INTERCHANGE_BIN="$OUT/ollm-interchange" OLLM_BIN="$BIN" \
  python3 -W error::ResourceWarning -m unittest discover -s "$here/tools/runtime" -p "test_*.py" -v
WORK="$xdir/demo10s" OUT="$OUT" sh "$here/scripts/demo-10s-wow.sh" >/dev/null

echo "--- 4e. Sovereign Commons, Trust Audit, Memory Migration, Porting Kit, Leases & USB Bundle ---"
OLLM_BIN="$BIN" OMEGA_HELIX_BIN_DIR="$OUT" \
  python3 -W error::ResourceWarning -m unittest discover -s "$here/tools/commons" -p "test_*.py" -v
OMEGA_HELIX_BIN_DIR="$OUT" "$here/bin/omega-helix" verify --json >/dev/null

if [ "$OS" = Linux ] && [ "$ARCH" = aarch64 ]; then
  GSL_WORK="$xdir/gsl" sh "$here/tests/test_gpu_spec_loop.sh" build
  GSL_WORK="$xdir/gsl" sh "$here/tests/test_gpu_spec_loop.sh" host
fi

echo "--- 5. Combined experiment Python controls ---"
python3 -m unittest discover -s "$here/tools/combined/experiment" -p "test_*.py" -v

if [ "$OS" = Linux ] && [ "$ARCH" = aarch64 ]; then
  OMC_BIN=${OMC:-$(command -v omc 2>/dev/null || ([ -x "$HOME/.local/bin/omc" ] && echo "$HOME/.local/bin/omc") || echo "$HOME/canon/omega-mv/seed/out/omc")}
  echo "--- 6. 165-task Omega coding benchmark integrity check ---"
  "$OMC_BIN" < "$here/tools/bench/bench.om" > "$OUT/bench.tmp"
  chmod +x "$OUT/bench.tmp" && mv -f "$OUT/bench.tmp" "$OUT/bench"
  mkdir -p "$xdir/bench"
  "$OUT/bench" check "$here/tools/bench/tasks.om" "$here/tools/bench/expect.txt" "$OMC_BIN" "$xdir/bench"

  if command -v jq >/dev/null 2>&1; then
    echo "--- 7. Contrast tool build & replay ---"
    make -C "$here/tools/contrast" OMC="$OMC_BIN" OUT="$OUT/omega-contrast" replay
    if command -v xxd >/dev/null 2>&1; then
      echo "--- 8. Controller tool build & replay ---"
      OMC="$OMC_BIN" OUT="$OUT/omega-controller" bash "$here/tools/controller/build.sh"
      OMC="$OMC_BIN" OUT="$OUT/omega-controller" bash "$here/tools/controller/replay.sh"
    fi
  fi

  NVDISASM=${NVDISASM:-$(command -v nvdisasm 2>/dev/null || echo /usr/local/cuda/bin/nvdisasm)}
  if [ -x "$NVDISASM" ]; then
    echo "--- 9. SM121 Blackwell SASS disassembly & GPR ceiling check (all 43 kernel slots) ---"
    OMC="$OMC_BIN" OUT="$OUT/gpu-disasm" NVDISASM="$NVDISASM" sh "$here/tests/gpu-disasm.sh" --all
  else
    echo "--- 9. SM121 SASS disassembly: NOT_RUN (nvdisasm not installed) ---"
  fi
fi

if command -v lean >/dev/null 2>&1 || [ -x "$HOME/.elan/bin/lean" ]; then
  echo "--- 10. Lean 4 proof check (tools/combined/experiment/Budget.lean) ---"
  PATH="$HOME/.elan/bin:$PATH" lean "$here/tools/combined/experiment/Budget.lean"
else
  echo "--- 10. Lean 4 proof check: NOT_RUN (lean not installed) ---"
fi

echo "OMEGA_HELIX_VERIFY_ALL_PASS"
