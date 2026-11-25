#!/bin/sh
# GB10 frozen-core specialist training and looped / small-shape BF16 checks.
#
#   (1) gtrainf / gtrainb CK TRAIN UNTIL EVERY FREEZE_LAYERS (gg.om ggfz): embeddings and layers
#       0..F-1 skipped in backward, untouched by the chip's ADAM, saved bit for bit; gcmp / gacmp
#       CK DATA S FREEZE_LAYERS against the CPU's trainf freeze; grow / growpro -> gtrainf / gtrainb
#       -> spext -> spapp, whose CPU eval must equal the trained checkpoint's; mutant fz must fail.
#   (2) BF16 at the small shape (gbcmp used to end in ggdie, exit 84: gg.om ggks split R = 128 into
#       K slices of 16, below the BF16 K tile 32) and on a looped checkpoint (NP 3): gbcmp, gcmp2,
#       gbcmp2 with the per-tensor cosine / normwise check (gb.om gbtens) and mutant lg failing it,
#       gtrainb; f32 gtrainf and BF16 gtrainb byte-identical to the binary before the change where
#       that binary ran.
#
# Usage (on the DGX Spark, Linux AArch64):
#   sh tests/test_gpu_spec_loop.sh build   # product, mutants fz and lg, and BASE_SRC's binary if set
#   sh tests/test_gpu_spec_loop.sh host    # checkpoints, data and the refusals (CPU only, no GPU)
#   sh tests/test_gpu_spec_loop.sh chip    # GPU checks: only inside a spark-window that holds the chip
#   sh tests/test_gpu_spec_loop.sh all     # build, host, chip
# Env:
#   OMC       seeded omc (default ~/canon/omega-mv/seed/out/omc)
#   GSL_WORK  work directory (default /tmp/spark-gpu-worker/run)
#   BASE_SRC  optional source tree of the binary before the change (the same layout as this clone)
# Exit 0 when every check holds, 1 otherwise. One GPU process at a time, 3 s apart.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
OMC=${OMC:-$HOME/canon/omega-mv/seed/out/omc}
W=${GSL_WORK:-/tmp/spark-gpu-worker/run}
mode=${1:-all}
mkdir -p "$W/log"
fail=0
SUM=$W/summary.txt

# mkbin NAME SED: copy native/ and scripts/ into $W/src-NAME, apply SED to native/gpu/gg.om, build
mkbin() {
	s=$W/src-$1
	rm -rf "$s"
	mkdir -p "$s"
	cp -R "$here/native" "$here/scripts" "$s/"
	if [ -n "$2" ]; then
		sed "$2" "$here/native/gpu/gg.om" > "$s/native/gpu/gg.om"
		cmp -s "$here/native/gpu/gg.om" "$s/native/gpu/gg.om" && { echo "MUTANT_NOOP $1"; exit 1; }
	fi
	OMC=$OMC TARGET=gb10 OUT=$W/bin-$1 sh "$s/scripts/build.sh" || { echo "BUILD_FAIL $1"; exit 1; }
}

build() {
	echo "omc $(sha256sum "$OMC" | cut -d' ' -f1)"
	mkbin prod ''
	mkbin mutfz 's/^const OMUTFZ ulong 1$/const OMUTFZ ulong 0/'
	mkbin mutlg 's/^const OMUTLG ulong 1$/const OMUTLG ulong 0/'
	if [ -n "${BASE_SRC:-}" ]; then
		OMC=$OMC TARGET=gb10 OUT=$W/bin-base sh "$BASE_SRC/scripts/build.sh" || { echo "BUILD_FAIL base"; exit 1; }
	fi
	for b in "$W"/bin-*/ollm-gb10; do echo "bin $b $(sha256sum "$b" | cut -d' ' -f1)"; done
}

# ck NAME WANT_RC BIN ARGS...: run, keep stdout + stderr in log/NAME.txt, record the verdict
ck() {
	n=$1 want=$2 b=$W/bin-$3/ollm-gb10
	shift 3
	f=$W/log/$n.txt
	echo "\$ $b $*" > "$f"
	"$b" "$@" >> "$f" 2>&1
	rc=$?
	v=PASS
	[ "$rc" = "$want" ] || { v=FAIL; fail=1; }
	echo "$n rc=$rc expect=$want $v" | tee -a "$SUM"
	grep -E '^(GC |GCF|GCMP|GA |GAF|GACMP|GFZ|GC2|GCMP2|GB2|GBT|GBC|GBCMP|GX|SPEXT|SPAPP|E |GP |F )' "$f" | sed 's/^/    /'
	grep -E '^T ' "$f" | tail -n 2 | sed 's/^/    /'
}

# has NAME REGEX: the log line must exist
has() {
	if grep -Eq "$2" "$W/log/$1.txt"; then
		echo "$1 has /$2/ PASS" | tee -a "$SUM"
	else
		echo "$1 has /$2/ FAIL" | tee -a "$SUM"
		fail=1
	fi
}

# same A B: the loss and bpb of two E lines are equal
same() {
	a=$(grep '^E ' "$W/log/$1.txt" | awk '{print $3, $4}')
	b=$(grep '^E ' "$W/log/$2.txt" | awk '{print $3, $4}')
	if [ -n "$a" ] && [ "$a" = "$b" ]; then
		echo "eval $1 = $2 ($a) PASS" | tee -a "$SUM"
	else
		echo "eval $1 ($a) != $2 ($b) FAIL" | tee -a "$SUM"
		fail=1
	fi
}

# bytes A B: two checkpoints byte-identical
bytes() {
	if cmp -s "$1" "$2"; then
		echo "cmp $(basename "$1") $(basename "$2") identical $(sha256sum "$1" | cut -c1-16) PASS" | tee -a "$SUM"
	else
		echo "cmp $(basename "$1") $(basename "$2") differ FAIL" | tee -a "$SUM"
		fail=1
	fi
}

D=$W/data.bin
host() {
	cat "$here"/native/*.om > "$D"
	# base: V 256 T 64 D 64 H 2 F 128 L 2 B 4 (R 256), trained 20 steps on the CPU (f64 weights)
	ck init_base 0 prod init "$W/base.ck" 256 64 64 2 128 2 4 1 200 3000 1
	ck train_base 0 prod train "$W/base.ck" "$D" 20 20
	ck grow 0 prod grow "$W/base.ck" "$W/g4.ck" 4 200 3000
	ck growpro 0 prod growpro "$W/base.ck" "$W/gp4.ck" 4 200 3000
	# looped: docs/LOOP.md's small shape V 256 T 32 D 64 H 2 F 128 L 4 B 4, LP 1 LC 2 NP 3 (R 128)
	ck init_loop 0 prod init "$W/loop.ck" 256 32 64 2 128 4 4 1 200 3000 1 1 2 3
	# R 2048 (B 64, T 32): the shape where the binary before the change already ran BF16
	ck init_r2k 0 prod init "$W/r2k.ck" 256 32 64 2 128 2 64 1 200 3000 1
	# refusals before the chip opens: a freeze on a loop (17), FREEZE_LAYERS above L (2)
	cp "$W/loop.ck" "$W/loopz.ck"
	ck refuse_loop 17 prod gtrainf "$W/loopz.ck" "$D" 2 2 1
	cmp -s "$W/loop.ck" "$W/loopz.ck" || { echo "refuse_loop wrote the checkpoint FAIL"; fail=1; }
	ck refuse_big 2 prod gtrainf "$W/g4.ck" "$D" 2 2 5
	ck refuse_gcmp 2 prod gcmp "$W/g4.ck" "$D" 0 5
}

gap() { sleep 3; }

chip() {
	if [ -z "${AIEN_GB10_LOCK_HELD:-}${SPARK_WINDOW:-}" ]; then
		echo "chip: not inside a spark-window (run via spark-window run)" >&2
		exit 3
	fi
	# (1) frozen core, f32: gradient and Adam parity with the freeze
	ck gcmp_fz 0 prod gcmp "$W/g4.ck" "$D" 0 2; has gcmp_fz '^GCF 2 [0-9]+ 0$'; has gcmp_fz '^GCMP PASS'; gap
	ck gcmp_nofz 0 prod gcmp "$W/g4.ck" "$D" 0; has gcmp_nofz '^GCMP PASS'; gap
	ck gacmp_fz 0 prod gacmp "$W/g4.ck" "$D" 0 2; has gacmp_fz '^GAF 2 [0-9]+ 0$'; has gacmp_fz '^GACMP PASS'; gap
	# grow -> gtrainf F 2 -> spext -> spapp
	cp "$W/g4.ck" "$W/g4f.ck"
	ck gtrainf_fz 0 prod gtrainf "$W/g4f.ck" "$D" 20 10 2; has gtrainf_fz '^GFZ 2 '; gap
	ck spext_g4 0 prod spext "$W/base.ck" "$W/g4f.ck" "$W/g4.sp"
	ck spapp_g4 0 prod spapp "$W/base.ck" "$W/g4.sp" "$W/g4app.ck"
	ck eval_g4f 0 prod eval "$W/g4f.ck" "$D" 8
	ck eval_g4app 0 prod eval "$W/g4app.ck" "$D" 8
	same eval_g4f eval_g4app
	# growpro -> gtrainb F 2 (BF16) -> spext -> spapp
	cp "$W/gp4.ck" "$W/gp4b.ck"
	ck gtrainb_fz 0 prod gtrainb "$W/gp4b.ck" "$D" 20 10 2; has gtrainb_fz '^GFZ 2 '; gap
	ck spext_gp4 0 prod spext "$W/base.ck" "$W/gp4b.ck" "$W/gp4.sp"
	ck spapp_gp4 0 prod spapp "$W/base.ck" "$W/gp4.sp" "$W/gp4app.ck"
	ck eval_gp4b 0 prod eval "$W/gp4b.ck" "$D" 8
	ck eval_gp4app 0 prod eval "$W/gp4app.ck" "$D" 8
	same eval_gp4b eval_gp4app
	# mutant fz: the chip ignores the freeze
	ck mutfz_gcmp 45 mutfz gcmp "$W/g4.ck" "$D" 0 2; gap
	ck mutfz_gacmp 46 mutfz gacmp "$W/g4.ck" "$D" 0 2; gap
	cp "$W/g4.ck" "$W/g4m.ck"
	ck mutfz_gtrainf 0 mutfz gtrainf "$W/g4m.ck" "$D" 10 10 2; gap
	ck mutfz_spext 62 prod spext "$W/base.ck" "$W/g4m.ck" "$W/g4m.sp"
	# (2) BF16 small shape and looped core
	ck gbcmp_base 0 prod gbcmp "$W/base.ck" "$D" 0; has gbcmp_base '^GBCMP PASS'; gap
	ck gbcmp_loop 0 prod gbcmp "$W/loop.ck" "$D" 0; has gbcmp_loop '^GBCMP PASS'; gap
	ck gcmp2_loop 0 prod gcmp2 "$W/loop.ck" "$D" 0; gap
	ck gbcmp2_loop 0 prod gbcmp2 "$W/loop.ck" "$D" 0; has gbcmp2_loop '^GBT 35 '; gap
	ck gbcmp2_base 0 prod gbcmp2 "$W/base.ck" "$D" 0; gap
	ck mutlg_gbcmp2_loop 49 mutlg gbcmp2 "$W/loop.ck" "$D" 0; has mutlg_gbcmp2_loop '^GBT .* (32|48)$'; gap
	ck mutlg_gbcmp2_base 0 mutlg gbcmp2 "$W/base.ck" "$D" 0; gap
	cp "$W/loop.ck" "$W/loopb.ck"
	ck gtrainb_loop 0 prod gtrainb "$W/loopb.ck" "$D" 10 5; gap
	ck geval_loopb 0 prod geval "$W/loopb.ck" "$D" 8; gap
	ck eval_loopb 0 prod eval "$W/loopb.ck" "$D" 8
	# unchanged where the binary before the change ran: f32 gtrainf (small) and BF16 gtrainb (R 2048)
	if [ -x "$W/bin-base/ollm-gb10" ]; then
		ck base_gbcmp_loop 84 base gbcmp "$W/loop.ck" "$D" 0; gap
		cp "$W/base.ck" "$W/f_old.ck"; cp "$W/base.ck" "$W/f_new.ck"
		ck base_gtrainf 0 base gtrainf "$W/f_old.ck" "$D" 30 10; gap
		ck prod_gtrainf 0 prod gtrainf "$W/f_new.ck" "$D" 30 10; gap
		bytes "$W/f_old.ck" "$W/f_new.ck"
		cp "$W/r2k.ck" "$W/b_old.ck"; cp "$W/r2k.ck" "$W/b_new.ck"
		ck base_gtrainb 0 base gtrainb "$W/b_old.ck" "$D" 5 5; gap
		ck prod_gtrainb 0 prod gtrainb "$W/b_new.ck" "$D" 5 5
		bytes "$W/b_old.ck" "$W/b_new.ck"
	fi
}

case $mode in
build) build ;;
host) : > "$SUM"; host ;;
chip) chip ;;
all) build; : > "$SUM"; host; chip ;;
*) echo "usage: $0 build|host|chip|all" >&2; exit 2 ;;
esac
[ "$fail" = 0 ] && echo "TEST_GPU_SPEC_LOOP PASS ($mode)" || echo "TEST_GPU_SPEC_LOOP FAIL ($mode)"
exit "$fail"
