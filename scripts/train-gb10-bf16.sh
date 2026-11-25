#!/bin/sh
# Reproduce the DGX Spark (NVIDIA GB10, sm_121) 85k tok/s BF16 (or 68k tok/s FP32)
# 15.8M v2 training configuration:
#   V=4096, T=256, D=384, H=12, F=1536, L=8, B=64  (16,384 tokens/step, P=15,833,472)
#
# Usage:
#   sh scripts/train-gb10-bf16.sh <train.tok> [held.tok] [merges4096.bin]
#
# Env:
#   MODE    bf16 (default, gtrainb ~85k tok/s) or fp32 (gtrainf ~68k tok/s)
#   STEPS   total training steps (default: 30000; Rung 10 used 60000)
#   EVERY   checkpoint/log interval in steps (default: 250)
#   LOOP    set to 1 to initialize a looped-core OLLMCK03 checkpoint (LP=2, LC=4, NP=3 -> 16 apps)
#   RUNDIR  output directory for checkpoints & logs (default: ./run-gb10)
set -eu

if [ $# -lt 1 ]; then
  echo "usage: $0 <train.tok> [held.tok] [merges4096.bin]" >&2
  exit 2
fi

here=$(cd "$(dirname "$0")/.." && pwd)
TRAIN_TOK=$1
HELD_TOK=${2:-}
MERGES=${3:-}
MODE=${MODE:-bf16}
STEPS=${STEPS:-30000}
EVERY=${EVERY:-250}
LOOP=${LOOP:-0}
RUNDIR=${RUNDIR:-$here/run-gb10}

BIN=$here/build/ollm-gb10
[ -x "$BIN" ] || TARGET=gb10 sh "$here/scripts/build.sh"

mkdir -p "$RUNDIR"
CK=$RUNDIR/ck.bin

if [ ! -f "$CK" ]; then
  if [ "$LOOP" = 1 ]; then
    echo "Initializing 15.8M looped-core OLLMCK03 checkpoint (LP=2, LC=4, NP=3 -> 16 layer apps)..."
    "$BIN" init "$CK" 4096 256 384 12 1536 8 64 1 "$STEPS" 3000 16 2 4 3
  else
    echo "Initializing 15.8M v2 checkpoint (V=4096 T=256 D=384 H=12 F=1536 L=8 B=64)..."
    "$BIN" init "$CK" 4096 256 384 12 1536 8 64 1 "$STEPS" 3000 16
  fi
fi

CMD=gtrainb
[ "$MODE" = fp32 ] && CMD=gtrainf

echo "Starting GB10 $CMD ($MODE) for $STEPS steps (16,384 tok/step)..."
"$BIN" "$CMD" "$CK" "$TRAIN_TOK" "$STEPS" "$EVERY"

if [ -n "$HELD_TOK" ]; then
  echo "Running on-chip held-out evaluation (geval)..."
  if [ -n "$MERGES" ]; then
    "$BIN" geval "$CK" "$HELD_TOK" 2243 "$MERGES"
  else
    "$BIN" geval "$CK" "$HELD_TOK" 2243
  fi
fi
