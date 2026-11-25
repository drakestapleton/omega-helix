#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=$(cd ../.. && pwd)
out=${OUT:-$root/build/omega-controller}
if [[ -z "${OMC:-}" ]]; then
  for cand in "$(command -v omc 2>/dev/null || true)" "$HOME/.local/bin/omc" "$HOME/canon/omega-mv/seed/out/omc" "$root/../omega-mv/seed/out/omc"; do
    if [[ -n "$cand" && -x "$cand" ]]; then OMC="$cand"; break; fi
  done
fi
omc=${OMC:?error: set OMC=/path/to/omc}
mkdir -p "$out"
driver=$out/driver.om
grader=$out/grader.om
sed 's/^fn main to int$/fn wbmain to int/' ../wdrive/wbuild.om >"$driver"
sed 's/^fn main to int$/fn priorgrade to int/' ../contrast/grade.om >"$grader"
compile() {
  local name=$1
  shift
  cat "$@" | "$omc" >"$out/$name.tmp"
  chmod +x "$out/$name.tmp"
  mv "$out/$name.tmp" "$out/$name"
}
compile normalize "$driver" normalize.om
compile train "$driver" policy.om policy-train.om
compile train-second "$driver" policy.om second.om train-second.om
compile controller "$driver" "$root/native/net/json.om" "$grader" ../contrast/read.om policy.om second.om controller.om
compile resident "$driver" "$root/native/net/http.om" "$root/native/net/notls.om" "$root/native/net/json.om" resident.om
compile grade "$driver" "$root/native/net/json.om" ../contrast/grade.om
for data in training second-training second-augmented-training; do
  xxd -r -p "calibration/$data.hex" >"$out/$data.bin"
done
"$out/train" "$out/training.bin" >"$out/controller.ck"
"$out/train-second" "$out/second-training.bin" >"$out/second.ck"
"$out/train-second" "$out/second-augmented-training.bin" >"$out/second-augmented.ck"
(cd "$out" && sha256sum -c "$root/tools/controller/calibration/checkpoints.sha256")
