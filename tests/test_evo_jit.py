#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pillar 6 Verification: Live Self-Evolving SM121 SASS JIT (native/gpu/ejit.om).

Verifies:
  1. Pure-OM2 compilation for BOTH `target a64macos` and `gb10` (`om2`), plus full gb10 stack composition.
  2. Self-contained SHA-256 bit-exactness against Python hashlib.sha256.
  3. Positive evolutions:
     - G6 Ascetic: yield=1 (w3 |= 8192) on FFMA/I2F/MUFU, 0 swaps, safe stall compaction, bit-exact parity.
     - G1 Predator, G2 Mitosis Hybrid, G5 Trickster: safe swaps, BB leaders pinned, rawFloor>=6, I2F preserved.
  4. Live Double-Buffer Hot-Swap Protocol (ejit_step_tick):
     - Executes step t on active buffer A, mutates & proves into standby B, checks parity, flips A <-> B.
     - Fail-closed rejection keeps active buffer untouched and does not flip activeSlot.
  5. Planted negative controls for ALL Antibodies #1-#15 + nested BSSY (#16) + GPR ceiling (#17)
     + EGP2 structural negative controls (#18) + parity mismatch (#19).

Usage:
  python3 -W error::ResourceWarning tests/test_evo_jit.py
"""
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EJIT_OM = os.path.join(REPO, "native", "gpu", "ejit.om")

PASSES = []
FAILS = []


def check(cond, name, detail=""):
    if cond:
        PASSES.append(name)
        print("PASS " + name)
    else:
        FAILS.append(name)
        print("FAIL " + name + (": " + detail if detail else ""))


def find_omc():
    cand = os.environ.get("OMC") or shutil.which("omc") or os.path.expanduser("~/.local/bin/omc")
    if not cand or not os.path.isfile(cand):
        raise RuntimeError("omc compiler not found")
    return cand


def compile_om(omc, src_text, out_bin, sign_macos=False):
    with tempfile.NamedTemporaryFile("w", suffix=".om", delete=False) as tf:
        tf.write(src_text)
        om_path = tf.name
    try:
        with open(om_path, "rb") as fin, open(out_bin + ".tmp", "wb") as fout:
            p = subprocess.run([omc], stdin=fin, stdout=fout, stderr=subprocess.PIPE)
        if p.returncode != 0:
            raise RuntimeError("omc failed rc=%d:\n%s" % (p.returncode, p.stderr.decode("utf-8", "replace")))
        os.chmod(out_bin + ".tmp", 0o755)
        os.replace(out_bin + ".tmp", out_bin)
        if sign_macos and platform.system() == "Darwin" and shutil.which("codesign"):
            subprocess.run(["codesign", "-s", "-", "-f", out_bin],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    finally:
        if os.path.exists(om_path):
            os.remove(om_path)


# SM121 instruction encoders matching native/gpu/gk.om and bw_hazard.om
def cw(st, wb=7, rb=7, wt=0, yld=0):
    return (st << 9) | ((yld & 1) << 13) | ((wb & 7) << 14) | ((rb & 7) << 17) | ((wt & 63) << 20)


def enc_movi(rd, imm, st=6, wb=7, rb=7, wt=0, yld=0):
    return (30722 | (rd << 16), imm & 0xffffffff, 3840, cw(st, wb, rb, wt, yld))


def enc_iadd3(rd, ra, rb, rc=255, st=6, wb=7, rb_bar=7, wt=0, yld=0):
    return (29200 | (rd << 16) | (ra << 24), rb, 134209791 if rc == 255 else rc, cw(st, wb, rb_bar, wt, yld))


def enc_imad(rd, ra, rb, rc=255, st=6, wb=7, rb_bar=7, wt=0, yld=0):
    return (29220 | (rd << 16) | (ra << 24), rb, 126747136 | rc, cw(st, wb, rb_bar, wt, yld))


def enc_ffma(rd, ra, rb, rc, st=6, yld=0):
    return (29219 | (rd << 16) | (ra << 24), rb, rc, cw(st, 7, 7, 0, yld))


def enc_i2f(rd, rb, st=6, yld=0):
    return (29253 | (rd << 16), rb, 0, cw(st, 7, 7, 0, yld))


def enc_mufu(rd, ra, op=5120, st=6, wb=0, wt=0, yld=0):
    return (29448 | (rd << 16), ra, op, cw(st, wb, 7, wt, yld))


def enc_ldg(rd, ra, off=0, wb=0, wt=0, st=4):
    return (2433 | (7 << 12) | (rd << 16) | (ra << 24), 4 | (off << 8), 203299072, cw(st, wb, 7, wt, 1))


def enc_isetp(p, ra, rb, cc=6, st=14):
    return (29196 | (ra << 24), rb, 66060400 | (cc << 12) | (p << 17), cw(st, 7, 7, 0, 0))


def enc_bra(here, tgt, p=7, st=5):
    d = (tgt - here - 1) & 0xffffffff
    w1 = (d >> 6) << 2
    c = 58720256
    if tgt <= here:
        w1 |= 0xf0000000
        c = 58982399
    return (2375 | (p << 12) | ((d & 63) << 18), w1, c, cw(st, 7, 7, 0, 0))


def enc_nop(st=4, wt=0):
    return (2328, 0, 0, cw(st, 7, 7, wt, 0))


def enc_bssy(st=2):
    return (2373, 0, 0, cw(st, 7, 7, 0, 0))


def enc_bsync(ra=255, st=2):
    return (2447 | (ra << 24), 0, 0, cw(st, 7, 7, 0, 0))


def enc_exit(st=5):
    return (2381, 0, 58720256, cw(st, 7, 7, 0, 1))


def emit_om_load_insns(fn_name, insns):
    lines = ["fn %s to void" % fn_name]
    for i, (w0, w1, w2, w3) in enumerate(insns):
        b = i * 4
        lines.append("put ejRef %d %d" % (b, w0))
        lines.append("put ejRef %d %d" % (b + 1, w1))
        lines.append("put ejRef %d %d" % (b + 2, w2))
        lines.append("put ejRef %d %d" % (b + 3, w3))
    lines.append("end")
    return "\n".join(lines)


def build_test_driver():
    # Kernel 1: Full realistic SM121 multi-block kernel with:
    #   pc=0:  MOVI R8, 10 (BB leader pc=0)
    #   pc=1:  LDG R10, [R8] (wb=0, var=1)
    #   pc=2:  NOP wait=1, st=4 (wait-NOP, compactible to st=1!)
    #   pc=3:  IADD3 R11, R10, R8 (depends on LDG, st=8 -> compactible to 6)
    #   pc=4:  IADD3 R20, R21, R22 (independent Tile B, st=7)
    #   pc=5:  I2F R12, R11 (nm=29, st=6, yld=0 -> G6 sets yld=1, NEVER shave stall!)
    #   pc=6:  FFMA R14, R12, R13, R14 (nm=24, st=6, yld=0 -> G6 sets yld=1!)
    #   pc=7:  MUFU R15, R20 (nm=31, st=6, yld=0 -> G6 sets yld=1!)
    #   pc=8:  ISETP P0, R11 >= R8 (anchor)
    #   pc=9:  BRA P0 -> pc=12 (creates BB target pc=12 and fall-through leader pc=10!)
    #   pc=10: IADD3 R16, R14, R8 (BB fall-through leader!)
    #   pc=11: IADD3 R17, R20, R21 (independent non-anchor)
    #   pc=12: FFMA R18, R15, R16, R17 (BB target leader!)
    #   pc=13: IADD3 R23, R24, R25 (independent non-anchor)
    #   pc=14: EXIT
    k1 = [
        enc_movi(8, 10, st=6),
        enc_ldg(10, 8, off=0, wb=0, wt=0, st=4),
        enc_nop(st=4, wt=1),
        enc_iadd3(11, 10, 8, st=8),
        enc_iadd3(20, 21, 22, st=7),
        enc_i2f(12, 11, st=6, yld=0),
        enc_ffma(14, 12, 13, 14, st=6, yld=0),
        enc_mufu(15, 20, st=6, wb=7, wt=0, yld=0),
        enc_isetp(0, 11, 8, cc=6, st=14),
        enc_bra(9, 12, p=0, st=5),
        enc_iadd3(16, 14, 8, st=6),
        enc_iadd3(17, 20, 21, st=6),
        enc_ffma(18, 15, 16, 17, st=6, yld=0),
        enc_iadd3(23, 24, 25, st=6),
        enc_exit(st=5),
    ]

    # Kernel 2: Micro-kernel for I2F Antibody #15 (op 581, nm=29, st=4 + EXIT st=5 -> sum=9)
    k_i2f = [
        enc_i2f(12, 11, st=4, yld=0),
        enc_exit(st=5),
    ]

    # Kernel 3: Nested BSSY test kernel (pc=0: BSSY, pc=1: BSSY, pc=2: BSYNC, pc=3: EXIT)
    k_bssy_bad = [
        enc_bssy(),
        enc_bssy(),
        enc_bsync(),
        enc_exit(),
    ]
    k_bssy_ok = [
        enc_bssy(),
        enc_iadd3(10, 11, 12, st=6),
        enc_bsync(),
        enc_exit(),
    ]

    # Kernel 4: GPR ceiling tests (R62 in 64-reg kernel, R126 in 128-reg kernel, R125 valid in 128-reg)
    k_gpr62 = [enc_iadd3(62, 8, 9, st=6), enc_exit()]
    k_gpr126 = [enc_iadd3(126, 8, 9, st=6), enc_exit()]
    k_gpr125 = [enc_iadd3(125, 8, 9, st=6), enc_exit()]

    # Kernel 5: Fixed-latency ALU RAW floor=6 test (pc=0 IADD3 R11 -> pc=1 IADD3 R12, R11)
    k_alu6 = [
        enc_iadd3(11, 8, 9, st=6),
        enc_iadd3(12, 11, 10, st=6),
        enc_exit(st=5),
    ]

    # SHA-256 test message: "abc" (3 bytes)
    msg = b"abc"
    want_sha = hashlib.sha256(msg).hexdigest()

    driver = []
    driver.append(emit_om_load_insns("loadK1", k1))
    driver.append(emit_om_load_insns("loadKI2f", k_i2f))
    driver.append(emit_om_load_insns("loadKBssyBad", k_bssy_bad))
    driver.append(emit_om_load_insns("loadKBssyOk", k_bssy_ok))
    driver.append(emit_om_load_insns("loadKGpr62", k_gpr62))
    driver.append(emit_om_load_insns("loadKGpr126", k_gpr126))
    driver.append(emit_om_load_insns("loadKGpr125", k_gpr125))
    driver.append(emit_om_load_insns("loadKAlu6", k_alu6))

    driver.append(r"""
text hexTbl "0123456789abcdef"
text tSp " "
text tNl "\n"
text tSha "SHA_ABC "
text kLoadK1 "LOAD_K1"
text kInitAct "INIT_ACTIVE"
text kG6Rc "TICK_G6_RC"
text kG6Act "TICK_G6_ACTIVE"
text kG6Swp "TICK_G6_SWAPS"
text kG6Yld "TICK_G6_YIELDS"
text kG6Flp "TICK_G6_FLIPS"
text kG6YI "TICK_G6_YLD_I2F"
text kG6YF "TICK_G6_YLD_FFMA"
text kG6YM "TICK_G6_YLD_MUFU"
text kG6SI "TICK_G6_STALL_I2F"
text kG6SN "TICK_G6_STALL_NOP"
text kG1Rc "TICK_G1_RC"
text kG1Act "TICK_G1_ACTIVE"
text kG1Swp "TICK_G1_SWAPS"
text kG1Flp "TICK_G1_FLIPS"
text kG1SR "TICK_G1_STALL_REF"
text kG1SC "TICK_G1_STALL_CND"
text kG2Rc "TICK_G2_RC"
text kG2Act "TICK_G2_ACTIVE"
text kG2Yld "TICK_G2_YIELDS"
text kG2Flp "TICK_G2_FLIPS"
text kG5Rc "TICK_G5_RC"
text kG5Act "TICK_G5_ACTIVE"
text kG5Flp "TICK_G5_FLIPS"
text kParBad "PARITY_CORRUPT_OK"
text kAb1 "AB1_DIGEST"
text kAb2 "AB2_BIJECT"
text kAb3 "AB3_SMUGGLE"
text kAb4 "AB4_ANCHOR"
text kAb5 "AB5_DEP"
text kAb6 "AB6_HAZARD"
text kAb7 "AB7_LENGTH"
text kAb8 "AB8_SIG"
text kAb9 "AB9_TRAIL"
text kAb10 "AB10_QMD"
text kAb11 "AB11_GEN"
text kAb12 "AB12_OPCODE"
text kAb13a "AB13_PC0"
text kAb13b "AB13_BRATGT"
text kAb13c "AB13_BRAFALL"
text kAb14 "AB14_FLOOR6"
text kAb15 "AB15_I2F"
text kBsOk "BSSY_FLAT"
text kAb16 "AB16_BSSY"
text kAb17a "AB17_GPR62"
text kAb17b "AB17_GPR126"
text kGpr125 "GPR125_VALID"
text kEg0 "EGP2_K0"
text kEg1 "EGP2_K1"
text kEgN1 "EGP2_NEG_RULE99"
text kEgN2 "EGP2_NEG_SPOOF"
text kL3Self "LOCK3_SELFTEST"
text kL3R0 "LOCK3_RCPT_INIT"
text kL3R4 "LOCK3_RCPT_AFTER4"
text kL3Rs "LOCK3_RESET_OK"
text kL3RpOk "LOCK3_REPLAY_OK"
text kL3BufEq "LOCK3_BUF_EXACT"
text kL3ActEq "LOCK3_ACT_EXACT"
text kL3ParEq "LOCK3_PAR_EXACT"
text kL3FlpEq "LOCK3_FLIP_COUNT"
text kL3Stp4Rc "LOCK3_STEP4_RC"
text kL3Stp4Flp "LOCK3_STEP4_FLIPS"
text kL3TmpRc "LOCK3_TAMPER_RC"
text kL3TmpCd "LOCK3_TAMPER_LASTCODE"
text kL3TmpAct "LOCK3_TAMPER_NOFLIP"
text kL3Ord1 "LOCK3_ORDER_SKIP"
text kL3Ord2 "LOCK3_ORDER_BACK"
text kL3Ord3 "LOCK3_ORDER_CURSOR"
text kL3R0S "LOCK3_R0_STEP"
text kL3R0G "LOCK3_R0_GEN"
text kL3R1S "LOCK3_R1_STEP"
text kL3R1G "LOCK3_R1_GEN"
text kL3R2S "LOCK3_R2_STEP"
text kL3R2G "LOCK3_R2_GEN"
text kL3R3S "LOCK3_R3_STEP"
text kL3R3G "LOCK3_R3_GEN"
text kL3Sha3 "LOCK3_RCPT3_SHA "
text kL3Words "LOCK3_FINAL_WORDS "

fn wrt s view uchar * to void
r = write 1 s 0 len s
end

fn wrtNum v ulong to void
b = new uchar 24 0
p = ulong 24
x = v
while true bound 24
p = sub p 1
put b p uchar add 48 rem x 10
x = div x 10
if eq x 0
break
end
end
r = write 1 b p sub 24 p
end

fn prLine tag view uchar * v ulong to void
wrt tag
wrt tSp
wrtNum v
wrt tNl
end

fn printSha to void
hx = new uchar 64 0
for i ulong 32
b = uint get ejShaDig i
put hx mul i 2 get hexTbl shr b 4
put hx add mul i 2 1 get hexTbl and b 15
end
wrt tSha
r = write 1 hx 0 64
wrt tNl
end

fn setStall pc ulong nst ulong to void
idx = add mul pc 4 3
w3 = get ejCand idx
nw3 = or and w3 not shl ulong 15 9 shl and nst 15 9
put ejCand idx nw3
end

fn b2u b bool to ulong
if b
ret 1
end
ret 0
end

fn checkYld pc ulong to ulong
ret and shr get ejCand add mul pc 4 3 13 1
end

fn runPositives to void
loadK1
ok = ejLoadKernelRef 15 64
prLine kLoadK1 b2u ok
prLine kInitAct get ejState 0

note Step t=0: G6 Ascetic on A -> B
rc0 = ejit_step_tick 6
prLine kG6Rc rc0
prLine kG6Act get ejState 0
prLine kG6Swp get ejState 10
prLine kG6Yld get ejState 11
prLine kG6Flp get ejState 4
prLine kG6YI checkYld 5
prLine kG6YF checkYld 6
prLine kG6YM checkYld 7
prLine kG6SI ejCndStall 5
prLine kG6SN ejCndStall 2

note Step t=1: G1 Predator on B -> A
rc1 = ejit_step_tick 1
prLine kG1Rc rc1
prLine kG1Act get ejState 0
prLine kG1Swp get ejState 10
prLine kG1Flp get ejState 4
prLine kG1SR get ejState 12
prLine kG1SC get ejState 13

note Step t=2: G2 Mitosis Hybrid on A -> B
rc2 = ejit_step_tick 2
prLine kG2Rc rc2
prLine kG2Act get ejState 0
prLine kG2Yld get ejState 11
prLine kG2Flp get ejState 4

note Step t=3: G5 Trickster on B -> A
rc3 = ejit_step_tick 5
prLine kG5Rc rc3
prLine kG5Act get ejState 0
prLine kG5Flp get ejState 4

note Fail-closed hot-swap test: corrupt BufB, verify parity check fails
put ejBufB add mul ulong 6 4 0 29216
parBad = ejit_parity_check 2 15
prLine kParBad b2u parBad
end

fn runNegatives1 to void
loadK1
ok2 = ejLoadKernelRef 15 64

note Antibody #1: Digest spoof
ejCopyRefToCand 15
ejHashWords true 15
for i ulong 32
put ejCertShaCnd i get ejCndSha i
end
put ejCertShaCnd 0 xor get ejCertShaCnd 0 1
prLine kAb1 ejVerifySigned 15 240 240 64

note Antibody #2: Non-bijection (duplicate perm[3]=perm[4])
ejCopyRefToCand 15
put ejPerm 3 get ejPerm 4
prLine kAb2 ejTestVerifyCand 15 64

note Antibody #3: Opcode smuggle (flip functional bit in w0 at pc=3)
ejCopyRefToCand 15
put ejCand mul ulong 3 4 xor get ejCand mul ulong 3 4 1
prLine kAb3 ejTestVerifyCand 15 64

note Antibody #4: Non-leader anchor escape (swap ISETP pc=8 with pc=7)
ejCopyRefToCand 15
ejSwapCand 7 8
prLine kAb4 ejTestVerifyCand 15 64

note Antibody #5: RAW/WAR/WAW dependency inversion (swap pc=3 IADD3 R11<-R10 with pc=5 I2F R12<-R11)
ejCopyRefToCand 15
ejSwapCand 3 5
prLine kAb5 ejTestVerifyCand 15 64

note Antibody #6: Scoreboard wait-barrier strip (clear wait mask on pc=2 NOP waiting on sb0)
ejCopyRefToCand 15
w3nop = get ejCand add mul ulong 2 4 3
put ejCand add mul ulong 2 4 3 and w3nop not shl ulong 63 20
prLine kAb6 ejTestVerifyCand 15 64

note Antibody #7: Length mismatch
ejCopyRefToCand 15
ejHashWords true 15
for i ulong 32
put ejCertShaCnd i get ejCndSha i
end
prLine kAb7 ejVerifySigned 15 240 224 64

note Antibodies #8-#11: uor-gpu Launch Gate
prLine kAb8 ejCheckLaunchGate 0 512 512 4096 384 1
prLine kAb9 ejCheckLaunchGate 1 512 513 4096 384 1
prLine kAb10 ejCheckLaunchGate 1 512 512 4100 384 1
prLine kAb11 ejCheckLaunchGate 1 512 512 4096 384 0
end

fn runNegatives2 to void
loadK1
ok3 = ejLoadKernelRef 15 64
ejCopyRefToCand 15
w3bad = get ejCand add mul ulong 3 4 3
w3bad = or and w3bad not shl ulong 7 14 shl ulong 6 14
put ejCand add mul ulong 3 4 3 w3bad
prLine kAb12 ejTestVerifyCand 15 64

note Antibody #13a: BB leader pc=0 escape (swap pc=0 and pc=3)
loadK1
ok4 = ejLoadKernelRef 15 64
ejCopyRefToCand 15
ejSwapCand 0 3
prLine kAb13a ejTestVerifyCand 15 64

note Antibody #13b: BB leader BRA target pc=12 escape (swap pc=12 and pc=13)
ejCopyRefToCand 15
ejSwapCand 12 13
prLine kAb13b ejTestVerifyCand 15 64

note Antibody #13c: BB leader BRA fall-through pc=10 escape (swap pc=10 and pc=11)
ejCopyRefToCand 15
ejSwapCand 10 11
prLine kAb13c ejTestVerifyCand 15 64

note Antibody #14: Fixed-latency ALU RAW floor < 6 (shave pc=0 IADD3 st=6 -> 5; even with flr=4 passed!)
loadKAlu6
oka6 = ejLoadKernelRef 3 64
ejCopyRefToCand 3
setStall 0 5
ejHashWords true 3
for i ulong 32
put ejCertShaCnd i get ejCndSha i
end
prLine kAb14 ejMonkVerify 3 4 64

note Antibody #15: I2F (op 581, nm=29) stall shave 4 -> 1 even when I2F+EXIT sum=6
loadKI2f
oki = ejLoadKernelRef 2 64
ejCopyRefToCand 2
setStall 0 1
prLine kAb15 ejTestVerifyCand 2 64

note Antibody #16: Nested BSSY vs flat BSSY
loadKBssyOk
okbo = ejLoadKernelRef 4 64
prLine kBsOk ejTestVerifyCand 4 64
loadKBssyBad
okbb = ejLoadKernelRef 4 64
prLine kAb16 ejTestVerifyCand 4 64

note Antibody #17: GPR ceiling (R62 in 64-reg, R126 in 128-reg, R125 valid in 128-reg)
loadKGpr62
okg1 = ejLoadKernelRef 2 64
prLine kAb17a ejTestVerifyCand 2 64
loadKGpr126
okg2 = ejLoadKernelRef 2 128
prLine kAb17b ejTestVerifyCand 2 128
loadKGpr125
okg3 = ejLoadKernelRef 2 128
prLine kGpr125 ejTestVerifyCand 2 128

note EGP2 Structural Proof + Negative Controls
prLine kEg0 ejEgVerifyKernel 0
prLine kEg1 ejEgVerifyKernel 1
ejEgReset
n0 = ejEgAdd 2 0 0 0 0
n1 = ejEgAdd 1 0 0 0 1
n2 = ejEgAdd 3 n0 n1 0 0
n3 = ejEgAdd 5 n0 n1 0 0
prLine kEgN1 b2u ejEgVerifyStep 99 n2 n3
prLine kEgN2 b2u ejEgVerifyStep 3 n2 n0
end

fn printRcptSha idx ulong to void
hx = new uchar 64 0
for i ulong 32
b = uint ejit_rcpt_sha_byte idx i
put hx mul i 2 get hexTbl shr b 4
put hx add mul i 2 1 get hexTbl and b 15
end
wrt kL3Sha3
r = write 1 hx 0 64
wrt tNl
end

fn printBufWords act ulong cntWords ulong to void
hx = new uchar 8 0
wrt kL3Words
for i ulong cntWords bound 64
w = ejSimWord add act 1 i
for k ulong 8
sh = mul sub 7 k 4
put hx k get hexTbl and shr w sh 15
end
r = write 1 hx 0 8
end
wrt tNl
end

fn runLock3Tests to void
wantBuf = new ulong 60 0
st = ejit_selftest
prLine kL3Self ulong st

loadK1
okL3 = ejLoadKernelRef 15 64
prLine kL3R0 ejit_rcpt_count
rcT = ejit_step_tick 6
rcT = ejit_step_tick 1
rcT = ejit_step_tick 2
rcT = ejit_step_tick 5
prLine kL3R4 ejit_rcpt_count
prLine kL3R0S ejit_rcpt_step 0
prLine kL3R0G ejit_rcpt_gen 0
prLine kL3R1S ejit_rcpt_step 1
prLine kL3R1G ejit_rcpt_gen 1
prLine kL3R2S ejit_rcpt_step 2
prLine kL3R2G ejit_rcpt_gen 2
prLine kL3R3S ejit_rcpt_step 3
prLine kL3R3G ejit_rcpt_gen 3
printRcptSha 3

wantAct = get ejState 0
wantPar = get ejState 14
for i ulong 60
put wantBuf i ejSimWord add wantAct 1 i
end
printBufWords wantAct 60

ejResetForReplay
rstOk = ulong 1
if ne get ejState 0 0
rstOk = ulong 0
end
if ne get ejState 3 0
rstOk = ulong 0
end
if ne get ejState 4 0
rstOk = ulong 0
end
if ne ejit_rcpt_count 4
rstOk = ulong 0
end
prLine kL3Rs rstOk

rplyOk = ulong 1
for s ulong 4
if ne ejit_replay_step s s 0
rplyOk = ulong 0
end
end
prLine kL3RpOk rplyOk
prLine kL3ActEq b2u eq get ejState 0 wantAct
prLine kL3ParEq b2u eq get ejState 14 wantPar
prLine kL3FlpEq get ejState 4

bufEq = ulong 1
for i ulong 60
if ne ejSimWord add wantAct 1 i get wantBuf i
bufEq = ulong 0
end
end
prLine kL3BufEq bufEq

rcT = ejit_replay_step 4 4
prLine kL3Stp4Rc rcT
prLine kL3Stp4Flp get ejState 4
end

fn runLock3Negatives to void
note Negative control 1: 1-bit SHA tamper on receipt 2 -> EJ_ERR_REPLAY_DIGEST (20)
ejResetForReplay
rcN = ejit_replay_step 0 0
rcN = ejit_replay_step 1 1
actBefore = get ejState 0
ejit_tamper_rcpt_sha 2 15 uchar 4
rcTmp = ejit_replay_step 2 2
ejit_tamper_rcpt_sha 2 15 uchar 4
prLine kL3TmpRc rcTmp
prLine kL3TmpCd get ejState 6
prLine kL3TmpAct b2u eq get ejState 0 actBefore

note Negative control 2: out-of-order replay -> EJ_ERR_REPLAY_ORDER (21)
ejResetForReplay
prLine kL3Ord1 ejit_replay_step 1 0
ejResetForReplay
rcN = ejit_replay_step 0 0
prLine kL3Ord2 ejit_replay_step 0 1
ejResetForReplay
prLine kL3Ord3 ejit_replay_step 0 1
end

fn main to int
note 1. Self-contained SHA-256 check on "abc"
put ejShaDat 0 97
put ejShaDat 1 98
put ejShaDat 2 99
ejSha256 3
printSha

runPositives
runNegatives1
runNegatives2
runLock3Tests
runLock3Negatives
ret 0
end
""")
    return "\n".join(driver), want_sha


def main():
    omc = find_omc()
    with open(EJIT_OM, "r", encoding="utf-8") as f:
        ejit_src = f.read()

    driver_src, want_sha = build_test_driver()

    with tempfile.TemporaryDirectory() as tmp:
        # 1. Verify compilation for gb10 (om2) and host target
        gb10_om = "om2\n" + ejit_src + "\n" + driver_src
        gb10_bin = os.path.join(tmp, "ejit_test_gb10")
        compile_om(omc, gb10_om, gb10_bin, sign_macos=False)
        check(os.path.isfile(gb10_bin) and os.path.getsize(gb10_bin) > 4096,
              "compile_gb10_om2", "size=%d" % os.path.getsize(gb10_bin))

        if platform.system() == "Darwin" and platform.machine() == "arm64":
            mac_om = "target a64macos\nom2\n" + ejit_src + "\n" + driver_src
            mac_bin = os.path.join(tmp, "ejit_test_a64macos")
            compile_om(omc, mac_om, mac_bin, sign_macos=True)
            check(os.path.isfile(mac_bin) and os.path.getsize(mac_bin) > 4096,
                  "compile_a64macos_om2", "size=%d" % os.path.getsize(mac_bin))
            run_bin = mac_bin
        elif platform.system() == "Linux" and platform.machine() == "aarch64":
            run_bin = gb10_bin
        else:
            run_bin = None

        if run_bin is None:
            print("SKIP runtime execution on unsupported host arch:", platform.machine())
            sys.exit(1 if FAILS else 0)

        p = subprocess.run([run_bin], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out = p.stdout.decode("utf-8", "replace")
        err = p.stderr.decode("utf-8", "replace")
        check(p.returncode == 0, "runtime_exit_0", "rc=%d err=%s" % (p.returncode, err))

        kv = {}
        for ln in out.splitlines():
            parts = ln.strip().split()
            if len(parts) == 2:
                kv[parts[0]] = parts[1]

        # 2. SHA-256 check
        check(kv.get("SHA_ABC") == want_sha, "sha256_abc_exact",
              "got=%s want=%s" % (kv.get("SHA_ABC"), want_sha))

        # 3. Positive G6 Ascetic & Live Double-Buffer Hot-Swap ticks
        check(kv.get("LOAD_K1") == "1" and kv.get("INIT_ACTIVE") == "0", "init_buf_A", str(kv))
        check(kv.get("TICK_G6_RC") == "0", "tick_g6_pass", "rc=%s" % kv.get("TICK_G6_RC"))
        check(kv.get("TICK_G6_ACTIVE") == "1" and kv.get("TICK_G6_FLIPS") == "1",
              "tick_g6_atomic_flip_A_to_B", "active=%s flips=%s" % (kv.get("TICK_G6_ACTIVE"), kv.get("TICK_G6_FLIPS")))
        check(kv.get("TICK_G6_SWAPS") == "0", "g6_zero_swaps", "swaps=%s" % kv.get("TICK_G6_SWAPS"))
        check(int(kv.get("TICK_G6_YIELDS", "0")) >= 3, "g6_yields_tuned", "yields=%s" % kv.get("TICK_G6_YIELDS"))
        check(kv.get("TICK_G6_YLD_I2F") == "1" and kv.get("TICK_G6_YLD_FFMA") == "1" and kv.get("TICK_G6_YLD_MUFU") == "1",
              "g6_yield1_on_ffma_i2f_mufu")
        check(kv.get("TICK_G6_STALL_I2F") == "6", "g6_preserves_i2f_stall",
              "i2f_stall=%s" % kv.get("TICK_G6_STALL_I2F"))
        check(kv.get("TICK_G6_STALL_NOP") == "1", "g6_compacts_wait_nop_stall",
              "nop_stall=%s" % kv.get("TICK_G6_STALL_NOP"))

        # G1 Predator, G2 Mitosis Hybrid, G5 Trickster + A <-> B round-trips
        check(kv.get("TICK_G1_RC") == "0" and kv.get("TICK_G1_ACTIVE") == "0" and kv.get("TICK_G1_FLIPS") == "2",
              "tick_g1_flip_B_to_A")
        check(int(kv.get("TICK_G1_SWAPS", "0")) >= 1, "g1_predator_safe_swaps",
              "swaps=%s" % kv.get("TICK_G1_SWAPS"))
        check(int(kv.get("TICK_G1_STALL_CND", "999")) < int(kv.get("TICK_G1_STALL_REF", "0")),
              "g1_compacts_total_stalls", "%s -> %s" % (kv.get("TICK_G1_STALL_REF"), kv.get("TICK_G1_STALL_CND")))
        check(kv.get("TICK_G2_RC") == "0" and kv.get("TICK_G2_ACTIVE") == "1" and kv.get("TICK_G2_FLIPS") == "3",
              "tick_g2_mitosis_flip_A_to_B")
        check(int(kv.get("TICK_G2_YIELDS", "0")) >= 3, "g2_mitosis_yields_tuned")
        check(kv.get("TICK_G5_RC") == "0" and kv.get("TICK_G5_ACTIVE") == "0" and kv.get("TICK_G5_FLIPS") == "4",
              "tick_g5_trickster_flip_B_to_A")
        check(kv.get("PARITY_CORRUPT_OK") == "0", "parity_gate_catches_corrupted_buf",
              "got=%s" % kv.get("PARITY_CORRUPT_OK"))

        # 4. Exact Antibody Rejection Codes (#1-#17 + EGP2)
        expected_rejections = [
            ("AB1_DIGEST", "1", "antibody_1_digest_spoof"),
            ("AB2_BIJECT", "2", "antibody_2_non_bijection"),
            ("AB3_SMUGGLE", "3", "antibody_3_opcode_smuggle"),
            ("AB4_ANCHOR", "4", "antibody_4_anchor_escape"),
            ("AB5_DEP", "5", "antibody_5_dep_inversion"),
            ("AB6_HAZARD", "6", "antibody_6_scoreboard_hazard"),
            ("AB7_LENGTH", "7", "antibody_7_length_mismatch"),
            ("AB8_SIG", "8", "antibody_8_p0_sig_smuggle"),
            ("AB9_TRAIL", "9", "antibody_9_trailing_bytes"),
            ("AB10_QMD", "10", "antibody_10_misaligned_qmd"),
            ("AB11_GEN", "11", "antibody_11_zero_gen_cap"),
            ("AB12_OPCODE", "12", "antibody_12_illegal_opcode"),
            ("AB13_PC0", "13", "antibody_13_bb_leader_pc0"),
            ("AB13_BRATGT", "13", "antibody_13_bb_leader_bra_target"),
            ("AB13_BRAFALL", "13", "antibody_13_bb_leader_bra_fallthrough"),
            ("AB14_FLOOR6", "14", "antibody_14_6_cycle_alu_floor_unconditional"),
            ("AB15_I2F", "15", "antibody_15_i2f_fixed_latency_shave"),
            ("BSSY_FLAT", "0", "flat_bssy_accepted"),
            ("AB16_BSSY", "16", "nested_bssy_rejected"),
            ("AB17_GPR62", "17", "gpr_ceiling_64reg_r62_rejected"),
            ("AB17_GPR126", "17", "gpr_ceiling_128reg_r126_rejected"),
            ("GPR125_VALID", "0", "gpr_ceiling_128reg_r125_accepted"),
            ("EGP2_K0", "0", "egp2_structural_proof_k0"),
            ("EGP2_K1", "0", "egp2_structural_proof_k1"),
            ("EGP2_NEG_RULE99", "0", "egp2_neg_unsound_rule_99_rejected"),
            ("EGP2_NEG_SPOOF", "0", "egp2_neg_structural_spoof_rejected"),
        ]
        for key, want, name in expected_rejections:
            check(kv.get(key) == want, name, "got=%s want=%s" % (kv.get(key), want))

        # 5. Byte-Traceability Lock 3: Receipt-Logged Promotion & Deterministic Replay
        check(kv.get("LOCK3_SELFTEST") == "0", "lock3_pure_om_selftest",
              "rc=%s" % kv.get("LOCK3_SELFTEST"))
        check(kv.get("LOCK3_RCPT_INIT") == "0" and kv.get("LOCK3_RCPT_AFTER4") == "4",
              "lock3_receipt_count_0_to_4",
              "init=%s after4=%s" % (kv.get("LOCK3_RCPT_INIT"), kv.get("LOCK3_RCPT_AFTER4")))
        want_rcpt_pairs = [("0", "6"), ("1", "1"), ("2", "2"), ("3", "5")]
        got_rcpt_pairs = [
            (kv.get("LOCK3_R%d_STEP" % i), kv.get("LOCK3_R%d_GEN" % i)) for i in range(4)
        ]
        check(got_rcpt_pairs == want_rcpt_pairs, "lock3_receipt_step_and_genome_log",
              "got=%s want=%s" % (got_rcpt_pairs, want_rcpt_pairs))

        final_words_hex = kv.get("LOCK3_FINAL_WORDS", "")
        check(len(final_words_hex) == 60 * 8, "lock3_final_words_len",
              "len=%d" % len(final_words_hex))
        if len(final_words_hex) == 60 * 8:
            words_bytes = b"".join(
                int(final_words_hex[i * 8:(i + 1) * 8], 16).to_bytes(4, "little")
                for i in range(60)
            )
            want_rcpt3_sha = hashlib.sha256(words_bytes).hexdigest()
            check(kv.get("LOCK3_RCPT3_SHA") == want_rcpt3_sha, "lock3_receipt_sha256_matches_python",
                  "got=%s want=%s" % (kv.get("LOCK3_RCPT3_SHA"), want_rcpt3_sha))

        check(kv.get("LOCK3_RESET_OK") == "1", "lock3_reset_for_replay_preserves_receipts")
        check(kv.get("LOCK3_REPLAY_OK") == "1", "lock3_replay_steps_0_to_3_ok")
        check(kv.get("LOCK3_BUF_EXACT") == "1" and kv.get("LOCK3_ACT_EXACT") == "1"
              and kv.get("LOCK3_PAR_EXACT") == "1" and kv.get("LOCK3_FLIP_COUNT") == "4",
              "lock3_replay_bit_exact_buffer_and_parity",
              "buf=%s act=%s par=%s flips=%s" % (
                  kv.get("LOCK3_BUF_EXACT"), kv.get("LOCK3_ACT_EXACT"),
                  kv.get("LOCK3_PAR_EXACT"), kv.get("LOCK3_FLIP_COUNT")))
        check(kv.get("LOCK3_STEP4_RC") == "0" and kv.get("LOCK3_STEP4_FLIPS") == "4",
              "lock3_replay_non_promotion_step_4_no_swap",
              "rc=%s flips=%s" % (kv.get("LOCK3_STEP4_RC"), kv.get("LOCK3_STEP4_FLIPS")))
        check(kv.get("LOCK3_TAMPER_RC") == "20" and kv.get("LOCK3_TAMPER_LASTCODE") == "20"
              and kv.get("LOCK3_TAMPER_NOFLIP") == "1",
              "lock3_1bit_sha_tamper_rejected_err_20",
              "rc=%s code=%s noflip=%s" % (
                  kv.get("LOCK3_TAMPER_RC"), kv.get("LOCK3_TAMPER_LASTCODE"),
                  kv.get("LOCK3_TAMPER_NOFLIP")))
        check(kv.get("LOCK3_ORDER_SKIP") == "21" and kv.get("LOCK3_ORDER_BACK") == "21"
              and kv.get("LOCK3_ORDER_CURSOR") == "21",
              "lock3_out_of_order_replay_rejected_err_21",
              "skip=%s back=%s cur=%s" % (
                  kv.get("LOCK3_ORDER_SKIP"), kv.get("LOCK3_ORDER_BACK"),
                  kv.get("LOCK3_ORDER_CURSOR")))

    print("\nSUMMARY: passed=%d failed=%d" % (len(PASSES), len(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
