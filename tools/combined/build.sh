#!/bin/sh
set -eu
umask 077
[ "$#" = 1 ] || { echo 'usage: build.sh NEW_KIT_DIRECTORY' >&2; exit 2; }
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH= cd -- "$here/../.." && pwd)
if [ -z "${OMC:-}" ]; then
  for cand in "$(command -v omc 2>/dev/null || true)" "$HOME/.local/bin/omc" "$HOME/canon/omega-mv/seed/out/omc" "$root/../omega-mv/seed/out/omc"; do
    if [ -n "$cand" ] && [ -x "$cand" ]; then OMC=$cand; break; fi
  done
fi
[ -n "${OMC:-}" ] && [ -x "$OMC" ] || { echo 'error: omc not found (set OMC=/path/to/omc)' >&2; exit 1; }
FACTORY=${HELIX_FACTORY_DIR:-$HOME/canon/omega-mv/seed/helix/ollm_factory}
for name in om_formalize om_sketch om_synth; do
  [ -f "$FACTORY/$name.om" ] || { echo "error: missing $FACTORY/$name.om (set HELIX_FACTORY_DIR=/path/to/ollm_factory)" >&2; exit 1; }
done
mkdir -- "$1"
kit=$(CDPATH= cd -- "$1" && pwd)
cp "$OMC" "$kit/omc"
for name in om_formalize om_sketch om_synth; do
  "$kit/omc" < "$FACTORY/$name.om" > "$kit/$name"
  chmod 700 "$kit/$name"
done
sed '/^fn main to int$/,$d' "$here/../bench/bench.om" > "$kit/combined.src"
cat "$here/combined.om" >> "$kit/combined.src"
"$kit/omc" < "$kit/combined.src" > "$kit/combined"
chmod 700 "$kit/combined"
cp "$here/solve" "$kit/solve"
