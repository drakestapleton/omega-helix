#!/bin/sh
# End-to-end OpenWALDO -> Omega Helix verified dataset preparation & training pipeline.
#
# Usage:
#   sh scripts/openwaldo-train.sh <openwaldo-export-dir> [out-dir]
#   OPENWALDO_CORPUS=core/example sh scripts/openwaldo-train.sh ./waldo-export ./run-waldo
#
# Env:
#   OPENWALDO_CORPUS  Optional OpenWALDO index path (e.g. core/example); runs
#                     `waldo index export --format jsonl` if export-dir does not exist yet
#   VOCAB             BPE vocabulary size (default: 4096; set 256 for byte-level)
#   HELD_MOD          Held-out split modulus (default: 10 -> 10% held-out; 0 disables)
#   TRAIN             Set to cpu or gb10 to launch training after preparation (default: none)
#   OMC               Path to seeded omc compiler (auto-detected if unset)
set -eu

if [ $# -lt 1 ]; then
  echo "usage: $0 <openwaldo-export-dir> [out-dir]" >&2
  exit 2
fi

here=$(cd "$(dirname "$0")/.." && pwd)
EXPORT_DIR=$1
OUT_DIR=${2:-$here/run-openwaldo}
VOCAB=${VOCAB:-4096}
HELD_MOD=${HELD_MOD:-10}
TRAIN=${TRAIN:-none}

if [ ! -f "$EXPORT_DIR/EXPORT.json" ] && [ -n "${OPENWALDO_CORPUS:-}" ]; then
  command -v waldo >/dev/null 2>&1 || {
    echo "error: 'waldo' CLI not found; install from https://github.com/openwaldo/waldo" >&2
    exit 1
  }
  echo "=== Exporting OpenWALDO corpus '$OPENWALDO_CORPUS' as canonical JSONL ==="
  waldo index export --format jsonl "$OPENWALDO_CORPUS" "$EXPORT_DIR"
fi

WALDO_BIN=$here/build/ollm-waldo
[ -x "$WALDO_BIN" ] || TARGET=openwaldo sh "$here/scripts/build.sh"

if [ "$(uname -s)" = Darwin ]; then
  OLLM_BIN=$here/build/ollm-macos
  [ -x "$OLLM_BIN" ] || TARGET=macos sh "$here/scripts/build.sh"
elif [ "$TRAIN" = gb10 ]; then
  OLLM_BIN=$here/build/ollm-gb10
  [ -x "$OLLM_BIN" ] || TARGET=gb10 sh "$here/scripts/build.sh"
else
  OLLM_BIN=$here/build/ollm
  [ -x "$OLLM_BIN" ] || TARGET=aarch64 sh "$here/scripts/build.sh"
fi

echo "=== Verifying OpenWALDO EXPORT.json and ingesting via pure-Omega ollm-waldo ==="
python3 "$here/tools/openwaldo/openwaldo_bridge.py" prepare \
  "$EXPORT_DIR" "$OUT_DIR" \
  --ollm-waldo "$WALDO_BIN" \
  --ollm "$OLLM_BIN" \
  --vocab "$VOCAB" \
  --held-mod "$HELD_MOD"

if [ "$TRAIN" = gb10 ]; then
  echo "=== Launching GB10 BF16 training on verified OpenWALDO corpus ==="
  RUNDIR="$OUT_DIR/gb10" sh "$here/scripts/train-gb10-bf16.sh" \
    "$OUT_DIR/train.tok" "$OUT_DIR/held.tok" "$OUT_DIR/merges${VOCAB}.bin"
  python3 "$here/tools/openwaldo/openwaldo_bridge.py" bind-checkpoint \
    "$OUT_DIR/openwaldo-helix-bom.json" "$OUT_DIR/gb10/ck.bin"
elif [ "$TRAIN" = cpu ]; then
  echo "=== Running CPU training smoke run on verified OpenWALDO corpus ==="
  CK="$OUT_DIR/cpu.ck"
  DATA="$OUT_DIR/train.bin"
  [ "$VOCAB" -gt 256 ] && DATA="$OUT_DIR/train.tok"
  "$OLLM_BIN" init "$CK" "$VOCAB" 64 64 4 256 2 4 1 50 3000 1
  "$OLLM_BIN" train "$CK" "$DATA" 50 10
  python3 "$here/tools/openwaldo/openwaldo_bridge.py" bind-checkpoint \
    "$OUT_DIR/openwaldo-helix-bom.json" "$CK"
fi

echo "OPENWALDO_PIPELINE_OK bom=$OUT_DIR/openwaldo-helix-bom.json"
