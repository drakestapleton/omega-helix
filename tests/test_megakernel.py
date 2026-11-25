#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Pillar 1: Persistent sm_121 Megakernel verification suite.

Checks:
  1. Pure-OM2 compilation with ~/.local/bin/omc for both `gb10` (`om2` + full
     native/gpu/{context,vector,launch,batch,gk,gr,gb,gg,gm}.om stack) and
     `macos` (`target a64macos` + `nogpu.om` + CPU-sim wrapper + `gm.om`).
  2. Built-in `gm_selftest` pass (`rc == 0`).
  3. SM121 SASS emitter invariants: single-slot fit (<= 16 KiB), double-buffered
     slot parity (`gm_swap_buf`), `max_gpr <= 61`, `gkgpr >= max_gpr + 3`,
     `nested_bssy == 0`, `+16 B` bank-padded SMEM stride (144 B/blk),
     L2-resident weight budget (`<= 8 MB`), and Spark `nvdisasm -b SM121`
     disassembly check (0 illegal/unknown insns + MEMBAR.SC.SYS /
     LDG.E.MMIO.GPU / STG.E.STRONG.SYS / GPR headroom).
  4. Launch-count = 1 invariant (`gm_launch_persistent` & `gm_sim_steps`).
  5. Bit-exact parity between `gm_sim_steps` (persistent step loop) and
     sequential `obatch -> ograd -> oadam` steps across weights, Adam moments,
     and per-step losses (both synthetic and real `odopen` dataset file), plus
     training loss decrease and boundary-error rejections.

Usage:
  python3 -W error::ResourceWarning tests/test_megakernel.py
"""
import hashlib
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "native")
GPU = os.path.join(SRC, "gpu")

PASSES = []
FAILS = []


def check(cond, name, detail=""):
    if cond:
        PASSES.append(name)
        print(f"PASS {name}" + (f" ({detail})" if detail else ""))
    else:
        FAILS.append(name)
        print(f"FAIL {name}" + (f": {detail}" if detail else ""))


def find_omc():
    for cand in (
        os.environ.get("OMC"),
        shutil.which("omc"),
        os.path.expanduser("~/.local/bin/omc"),
        os.path.expanduser("~/canon/omega-mv/seed/out/omc"),
        os.path.join(REPO, "..", "omega-mv", "seed", "out", "omc"),
        os.path.join(REPO, "..", "omega-mv-macos", "seed", "out", "omc"),
    ):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    raise RuntimeError("seeded omc compiler not found")


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def compile_om_bytes(omc, src_text, out_path=None):
    p = subprocess.run(
        [omc],
        input=src_text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if out_path is not None and p.returncode == 0:
        with open(out_path, "wb") as f:
            f.write(p.stdout)
        os.chmod(out_path, 0o755)
        if platform.system() == "Darwin" and shutil.which("codesign"):
            subprocess.run(
                ["codesign", "-s", "-", "-f", out_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
    return p.returncode, p.stderr.decode("utf-8", errors="replace"), len(p.stdout)


MAC_SIM_SHIM = """
global gkcode uchar 1048576
global gkthr uint 64
global gkgpr uint 64
global gksm uint 64
global gkarg uint 32
global gkna uint 1
global gkt long 4

fn gklaunch id uint gx uint gy uint gz uint to bool
put gkt 0 add get gkt 0 1
ret true
end
"""

DRIVER_MAIN = """
fn gm_print_u tag ulong v ulong to void
oputx tag
oputc 32
oputu v
oputc 10
oflush
end

fn gm_print_f tag ulong v f64 to void
oputx tag
oputc 32
oputx bitcast ulong v
oputc 32
oputl rmic v
oputc 10
oflush
end

fn gm_init_shape to void
put og 0 256
put og 1 8
put og 2 16
put og 3 2
put og 4 32
put og 5 2
put og 6 2
put og 40 12345
put og 41 0
put og 42 100
put og 43 bitcast ulong 0.001f64
put og 44 1
put og 59 1
if not ogeom
exit 11
end
oinit
end

fn gm_cmd_d to int
if not gm_emit_megakernel 0
ret 20
end
if not gm_emit_megakernel 1
ret 21
end
oputu get gmcfg 1
oputc 32
oputu get gmcfg 2
oputc 32
oputu get gmcfg 3
oputc 32
oputu get gmcfg 5
oputc 32
oputu get gmcfg 6
oputc 32
oputu get gmcfg 7
oputc 32
oputu get gmstats 5
oputc 10
oflush
ret 0
end

fn gm_cmd_b to int
if not gm_emit_megakernel 0
ret 20
end
if not gm_emit_megakernel 1
ret 21
end
slot = ulong 0
if ge oargc 3
slot = oargu 2
end
base = mul slot 16384
n = get gmcfg add 1 slot
w = write 1 gmcode base n
if ne w long n
ret 22
end
ret 0
end

fn gm_cmd_p to int
gm_init_shape
use_synth = true
if ge oargc 3
odopen 2
use_synth = false
end
nn = ulong 6
lseq = gm_seq_steps nn use_synth
gm_save_state
put og 41 0
oinit
lmeg = gm_sim_steps nn use_synth
diff = gm_parity_diff nn
oputc 80
oputc 65
oputc 82
oputc 32
oputu diff
oputc 32
oputu get gmstats 0
oputc 32
oputu get gmstats 1
oputc 32
oputu get gmstats 2
oputc 32
oputu get gmstats 3
oputc 32
oputu get gmstats 4
oputc 32
oputu get gmstats 5
oputc 32
oputu get gmstats 6
oputc 32
oputx bitcast ulong lseq
oputc 32
oputx bitcast ulong lmeg
oputc 32
oputl rmic get gmlos_meg 0
oputc 32
oputl rmic get gmlos_meg sub nn 1
oputc 32
oputu get gmcfg 11
oputc 10
oflush
if ne diff 0
ret 30
end
ret 0
end

fn gm_cmd_l to int
gm_init_shape
if not gm_emit_megakernel 0
ret 40
end
if not gm_emit_megakernel 1
ret 41
end
sw1 = gm_swap_buf
sl1 = get gmcfg 13
sw2 = gm_swap_buf
sl2 = get gmcfg 13
ok1 = gm_launch_persistent 8
ln1 = get gmstats 0
oputc 76
oputc 78
oputc 32
oputu ulong sw1
oputc 32
oputu sl1
oputc 32
oputu ulong sw2
oputc 32
oputu sl2
oputc 32
oputu ln1
oputc 32
oputu get gmstats 1
oputc 32
oputu get gmstats 2
oputc 32
oputu get gmcfg 12
oputc 10
oflush
if or not ok1 ne ln1 1
ret 42
end
if or gm_emit_megakernel 2 or gm_launch_persistent 0 gm_launch_persistent 65537
ret 43
end
if ge gm_sim_steps 0 true 0.0f64
ret 44
end
ret 0
end

fn gm_putb b bool to void
if b
oputc 49
else
oputc 48
end
end

fn gm_cmd_r_neg a0 bool a1 bool to ulong
saved = gm_r_at 16384
gm_w_at 16384 0x3a8
ok = not gm_audit_no_fp_atomics 1
gm_w_at 16384 saved
if ok
ok = gm_audit_probe 0x9a8
end
if ok
ok = gm_audit_probe 0x3aa
end
if ok
ok = gm_audit_probe 0x9aa
end
if ok
ok = gm_audit_probe 0x3a9
end
if ok
ok = gm_audit_probe 0x9a9
end
a1p = gm_audit_no_fp_atomics 1
gm_w_at 16384 0x3a8
if ne gm_swap_buf 1
gm_w_at 16384 saved
ret 0
end
lb = not gm_launch_persistent 4
gm_w_at 16384 saved
if ne gm_swap_buf 0
ret 0
end
oputc 76
oputc 79
oputc 67
oputc 75
oputc 49
oputc 32
gm_putb a0
oputc 32
gm_putb a1
oputc 32
gm_putb ok
oputc 32
gm_putb a1p
oputc 32
gm_putb lb
if and ok and a1p lb
ret 1
end
ret 0
end

fn gm_cmd_r to int
if not gm_emit_megakernel 0
ret 50
end
if not gm_emit_megakernel 1
ret 51
end
a0 = gm_audit_no_fp_atomics 0
a1 = gm_audit_no_fp_atomics 1
if or not a0 not a1
ret 52
end
if eq gm_cmd_r_neg a0 a1 0
ret 53
end
fwd = new f64 24 0.0f64
rev = new f64 24 0.0f64
intl = new f64 24 0.0f64
gm_init_cta_terms edit fwd
for i ulong 24
put rev i get fwd sub 23 i
if lt i 12
put intl i get fwd mul i 2
else
put intl i get fwd add mul sub i 12 2 1
end
end
qf = gm_reduce_i64_fixed view fwd 24
qr = gm_reduce_i64_fixed view rev 24
qi = gm_reduce_i64_fixed view intl 24
oputc 32
oputl qf
oputc 32
oputl qr
oputc 32
oputl qi
oputc 32
oputx bitcast ulong gm_fixed_to_f64 qf
oputc 32
oputx bitcast ulong gm_fixed_to_f64 qr
oputc 32
oputx bitcast ulong gm_fixed_to_f64 qi
oputc 32
oputx bitcast ulong gm_reduce_tree_f64 view fwd 24
oputc 32
oputx bitcast ulong gm_reduce_tree_f64 view fwd 19
oputc 10
oflush
if or ne qf qr ne qf qi
ret 54
end
ret 0
end

fn main to int
cmd = oargr 1
if lt cmd 0
exit 2
end
c0 = get oarg 0
if eq c0 115
rc = gm_selftest
oputu ulong rc
oputc 10
oflush
ret rc
end
if eq c0 100
ret gm_cmd_d
end
if eq c0 98
ret gm_cmd_b
end
if eq c0 112
ret gm_cmd_p
end
if eq c0 108
ret gm_cmd_l
end
if eq c0 114
ret gm_cmd_r
end
ret 2
end
"""


def build_macos_source():
    parts = [
        "math.om", "data.om", "lpool.om", "model.om", "adam.om",
        "ckpt.om", "bpe.om", "spec.om", "rev.om", "arboris.om",
        "sp_solve.om", "omg.om", "cli.om", "mio.om", "noserve.om", "nogpu.om",
    ]
    body = []
    for name in parts:
        txt = read_text(os.path.join(SRC, name))
        txt = txt.replace("fn main to int", "fn ocpumain to int")
        body.append(txt)
    body.append(MAC_SIM_SHIM)
    body.append(read_text(os.path.join(GPU, "gm.om")))
    body.append(DRIVER_MAIN)
    return "target a64macos\nom2\n" + "".join(body)


def build_gb10_source():
    parts = [
        os.path.join(SRC, "net", "http.om"),
        os.path.join(SRC, "net", "json.om"),
        os.path.join(SRC, "net", "jsonw.om"),
        os.path.join(SRC, "math.om"),
        os.path.join(SRC, "data.om"),
        os.path.join(SRC, "lpool.om"),
        os.path.join(SRC, "model.om"),
        os.path.join(SRC, "adam.om"),
        os.path.join(SRC, "ckpt.om"),
        os.path.join(SRC, "bpe.om"),
        os.path.join(SRC, "spec.om"),
        os.path.join(SRC, "rev.om"),
        os.path.join(SRC, "arboris.om"),
        os.path.join(SRC, "sp_solve.om"),
        os.path.join(SRC, "omg.om"),
        os.path.join(SRC, "cli.om"),
        os.path.join(SRC, "lio.om"),
        os.path.join(SRC, "serve.om"),
        os.path.join(GPU, "context.om"),
        os.path.join(GPU, "vector.om"),
        os.path.join(GPU, "launch.om"),
        os.path.join(GPU, "batch.om"),
        os.path.join(GPU, "gk.om"),
        os.path.join(GPU, "gr.om"),
        os.path.join(GPU, "gb.om"),
        os.path.join(GPU, "gg.om"),
        os.path.join(GPU, "gt.om"),
        os.path.join(GPU, "gm.om"),
        os.path.join(GPU, "ring.om"),
        os.path.join(GPU, "ejit.om"),
    ]
    return "om2\n" + "".join(read_text(p) for p in parts)


def build_gb10_minimal_source():
    parts = [
        os.path.join(SRC, "math.om"),
        os.path.join(SRC, "data.om"),
        os.path.join(SRC, "lpool.om"),
        os.path.join(SRC, "model.om"),
        os.path.join(SRC, "adam.om"),
        os.path.join(SRC, "ckpt.om"),
        os.path.join(SRC, "bpe.om"),
        os.path.join(SRC, "spec.om"),
        os.path.join(SRC, "rev.om"),
        os.path.join(SRC, "arboris.om"),
        os.path.join(SRC, "sp_solve.om"),
        os.path.join(SRC, "omg.om"),
        os.path.join(SRC, "cli.om"),
        os.path.join(SRC, "lio.om"),
        os.path.join(SRC, "noserve.om"),
        os.path.join(GPU, "context.om"),
        os.path.join(GPU, "vector.om"),
        os.path.join(GPU, "launch.om"),
        os.path.join(GPU, "batch.om"),
        os.path.join(GPU, "gk.om"),
        os.path.join(GPU, "gr.om"),
        os.path.join(GPU, "gb.om"),
        os.path.join(GPU, "gg.om"),
        os.path.join(GPU, "gt.om"),
        os.path.join(GPU, "gm.om"),
        os.path.join(GPU, "ring.om"),
        os.path.join(GPU, "ejit.om"),
    ]
    return "om2\n" + "".join(read_text(p) for p in parts)


def max_gpr_in_sass(sass_text):
    mx = -1
    for line in sass_text.splitlines():
        m = re.search(r"/\*[0-9a-f]+\*/\s*(.*)", line)
        if not m:
            continue
        rhs = re.sub(r";.*", "", m.group(1)).strip()
        if rhs.startswith("@"):
            parts = rhs.split(None, 1)
            rhs = parts[1] if len(parts) > 1 else ""
        parts = rhs.split(None, 1)
        op = parts[0] if parts else ""
        rest = parts[1] if len(parts) > 1 else ""
        w = 1
        if ".128" in op:
            w = 4
        elif ".64" in op or ".WIDE" in op:
            w = 2
        # Bracketed address operands have width 0 extension unless .64
        for br in re.findall(r"\[[^\]]*\]", rest):
            for rm in re.finditer(r"(?<![A-Za-z0-9_.])R(\d+)(\.64)?", br):
                ext = 1 if rm.group(2) else 0
                mx = max(mx, int(rm.group(1)) + ext)
        no_br = re.sub(r"\[[^\]]*\]", " ", rest)
        for rm in re.finditer(r"(?<![A-Za-z0-9_.])R(\d+)(\.64)?", no_br):
            ext = 1 if rm.group(2) else w - 1
            mx = max(mx, int(rm.group(1)) + ext)
    return mx


def run_nvdisasm(code_bytes):
    local_nv = shutil.which("nvdisasm") or "/usr/local/cuda/bin/nvdisasm"
    if os.path.isfile(local_nv) and os.access(local_nv, os.X_OK):
        with tempfile.NamedTemporaryFile(suffix=".bin") as tf:
            tf.write(code_bytes)
            tf.flush()
            p = subprocess.run([local_nv, "-b", "SM121", tf.name],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return "local", p.returncode, p.stdout.decode("utf-8", errors="replace"), p.stderr.decode("utf-8", errors="replace")
    if shutil.which("ssh"):
        p = subprocess.run(
            [
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "spark",
                "tmp=$(mktemp /tmp/gm_disasm.XXXXXX) && cat > \"$tmp\" && /usr/local/cuda/bin/nvdisasm -b SM121 \"$tmp\"; rc=$?; rm -f \"$tmp\"; exit $rc",
            ],
            input=code_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return "spark-ssh", p.returncode, p.stdout.decode("utf-8", errors="replace"), p.stderr.decode("utf-8", errors="replace")
    return None, None, "", "nvdisasm unavailable"


def main():
    omc = find_omc()
    print(f"Using omc: {omc}")

    # 1. Compile gb10 full stack + gm.om and gb10 minimal stack + gm.om
    rc, err, nbytes = compile_om_bytes(omc, build_gb10_source())
    check(rc == 0 and nbytes > 0, "compile_gb10_full", f"rc={rc} bytes={nbytes} {err.strip()}")

    rc_m, err_m, nbytes_m = compile_om_bytes(omc, build_gb10_minimal_source())
    check(rc_m == 0 and nbytes_m > 0, "compile_gb10_noserve", f"rc={rc_m} bytes={nbytes_m} {err_m.strip()}")

    # 2. Compile and run macos target on Darwin (Linux omc only supports Linux targets)
    if platform.system() != "Darwin":
        print("Non-macOS host: skipping target a64macos compile/execution.")
        return 0 if not FAILS else 1

    with tempfile.TemporaryDirectory() as td:
        bin_path = os.path.join(td, "gm_test")
        rc_mac, err_mac, nbytes_mac = compile_om_bytes(omc, build_macos_source(), bin_path)
        check(rc_mac == 0 and nbytes_mac > 0, "compile_macos_a64", f"rc={rc_mac} bytes={nbytes_mac} {err_mac.strip()}")
        if rc_mac != 0:
            return 1

        # 3. Built-in gm_selftest
        p = subprocess.run([bin_path, "s"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        check(p.returncode == 0, "gm_selftest", f"rc={p.returncode} out={p.stdout.decode().strip()}")

        # 4. Emitter metadata & double-buffer checks
        p = subprocess.run([bin_path, "d"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        check(p.returncode == 0, "gm_emit_stats_run", f"rc={p.returncode}")
        b0, b1, ins0, max_gpr, decl_gpr, smem_b, nested_bssy = [int(x) for x in p.stdout.decode().split()]
        check(0 < b0 <= 16384 and b0 == b1, "single_slot_fit_double_buf", f"b0={b0} b1={b1} insns={ins0}")
        check(max_gpr <= 61 and decl_gpr >= max_gpr + 3, "gpr_headroom", f"max_gpr=R{max_gpr} decl={decl_gpr}")
        check(nested_bssy == 0, "zero_nested_bssy", f"nested_bssy={nested_bssy}")
        check(smem_b == 4608, "smem_bank_padded_layout", f"smem_bytes={smem_b}")

        # 5. Extract slot 0 and slot 1 bytes, check sha256 parity
        p_s0 = subprocess.run([bin_path, "b", "0"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p_s1 = subprocess.run([bin_path, "b", "1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        check(
            p_s0.returncode == 0 and p_s1.returncode == 0 and len(p_s0.stdout) == b0
            and hashlib.sha256(p_s0.stdout).hexdigest() == hashlib.sha256(p_s1.stdout).hexdigest(),
            "double_buffer_code_parity",
            f"sha256={hashlib.sha256(p_s0.stdout).hexdigest()[:16]}",
        )

        # 6. Spark nvdisasm -b SM121 verification (CPU-only, no GPU)
        mode, d_rc, sass, d_err = run_nvdisasm(p_s0.stdout)
        if mode is not None and d_rc == 0:
            has_illegal = bool(re.search(r"illegal|unknown", sass, re.IGNORECASE))
            check(not has_illegal, f"nvdisasm_sm121_clean ({mode})", f"insns={ins0}")
            req_insns = [
                "LDCU.64 UR4",
                "LDG.E.MMIO.GPU",
                "MEMBAR.SC.SYS",
                "STG.E.STRONG.SYS",
                "SHFL.BFLY",
                "MUFU.RSQ",
                "MUFU.EX2",
                "MUFU.LG2",
                "BAR.SYNC.DEFER_BLOCKING",
                "LDS R34, [R48+URZ]",
                "LDS R35, [R48+URZ+0x90]",
                "BRA 0x270",
                "EXIT",
            ]
            missing = [tok for tok in req_insns if tok not in sass]
            check(not missing, "nvdisasm_sm121_required_ops", f"missing={missing}")
            has_fp_atom = bool(re.search(r"\bATOM[GS]?\b|\bRED[GS]?\b", sass))
            check(not has_fp_atom, "nvdisasm_sm121_no_atomics", f"insns={ins0}")
            dis_max_gpr = max_gpr_in_sass(sass)
            check(
                0 <= dis_max_gpr <= 61 and decl_gpr >= dis_max_gpr + 3,
                "nvdisasm_gpr_headroom",
                f"dis_max=R{dis_max_gpr} need<={dis_max_gpr+3} decl={decl_gpr}",
            )
        else:
            print(f"SKIP nvdisasm_sm121 ({mode}: rc={d_rc} {d_err.strip()})")

        # 7. Launch-count=1, double-buffer swap, and negative boundary checks
        p_ln = subprocess.run([bin_path, "l"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        check(p_ln.returncode == 0, "launch_persistent_and_bounds", p_ln.stdout.decode().strip())
        ln_toks = p_ln.stdout.decode().split()
        check(
            ln_toks[0] == "LN" and int(ln_toks[1]) == 1 and int(ln_toks[2]) == 61
            and int(ln_toks[3]) == 0 and int(ln_toks[4]) == 60
            and int(ln_toks[5]) == 1 and int(ln_toks[6]) == 8 and int(ln_toks[7]) == 16 and int(ln_toks[8]) == 2,
            "swap_buf_and_single_launch",
            " ".join(ln_toks),
        )

        # 8. Parity test with synthetic in-memory batch
        p_ps = subprocess.run([bin_path, "p"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        check(p_ps.returncode == 0, "sim_parity_synth_run", p_ps.stdout.decode().strip())
        pt = p_ps.stdout.decode().split()
        check(
            pt[0] == "PAR" and int(pt[1]) == 0 and int(pt[2]) == 1 and int(pt[3]) == 6
            and int(pt[7]) == 0 and int(pt[8]) == 6 and pt[9] == pt[10]
            and int(pt[12]) < int(pt[11]) and int(pt[13]) <= 8388608,
            "sim_parity_synth_bit_exact",
            f"diff={pt[1]} launches={pt[2]} loss0={pt[11]} loss5={pt[12]} l2_bytes={pt[13]}",
        )

        # 9. Parity test with real odopen dataset file
        ds_path = os.path.join(td, "train.bin")
        # Deterministic byte corpus (> T + 1 tokens, V=256)
        corpus = bytes(((i * 37 + 13) % 256) for i in range(512))
        with open(ds_path, "wb") as f:
            f.write(corpus)
        p_pf = subprocess.run([bin_path, "p", ds_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        check(p_pf.returncode == 0, "sim_parity_odopen_run", p_pf.stdout.decode().strip())
        pf = p_pf.stdout.decode().split()
        check(
            pf[0] == "PAR" and int(pf[1]) == 0 and int(pf[2]) == 1 and int(pf[3]) == 6
            and int(pf[8]) == 6 and pf[9] == pf[10] and int(pf[12]) < int(pf[11]),
            "sim_parity_odopen_bit_exact",
            f"diff={pf[1]} launches={pf[2]} loss0={pf[11]} loss5={pf[12]}",
        )

        # 10. Byte-Traceability Lock 1: FP atomic audit + negative control + fixed/tree reductions
        p_l1 = subprocess.run([bin_path, "r"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        check(p_l1.returncode == 0, "lock1_audit_and_reduction_run", p_l1.stdout.decode().strip())
        lt = p_l1.stdout.decode().split()
        check(
            len(lt) == 14 and lt[0] == "LOCK1"
            and int(lt[1]) == 1 and int(lt[2]) == 1
            and int(lt[3]) == 1 and int(lt[4]) == 1 and int(lt[5]) == 1,
            "lock1_no_fp_atomics_and_neg_ctrl",
            f"a0={lt[1]} a1={lt[2]} neg_ok={lt[3]} a1_post={lt[4]} launch_blocked={lt[5]}" if len(lt) >= 6 else str(lt),
        )
        check(
            len(lt) == 14 and int(lt[6]) == int(lt[7]) == int(lt[8]),
            "lock1_i64_fixed_perm_invariant",
            f"qf={lt[6]} qr={lt[7]} qi={lt[8]}" if len(lt) >= 9 else str(lt),
        )
        check(
            len(lt) == 14 and lt[9] == lt[10] == lt[11],
            "lock1_f64_fixed_bit_identical",
            f"ff={lt[9]} fr={lt[10]} fi={lt[11]}" if len(lt) >= 12 else str(lt),
        )
        if len(lt) == 14:
            f_fixed = struct.unpack(">d", bytes.fromhex(lt[9].zfill(16)))[0]
            t24 = struct.unpack(">d", bytes.fromhex(lt[12].zfill(16)))[0]
            t19 = struct.unpack(">d", bytes.fromhex(lt[13].zfill(16)))[0]
            check(
                abs(f_fixed - 2.3764) < 1e-6 and abs(t24 - 2.3764) < 1e-9 and abs(t19 - 13.7511) < 1e-9,
                "lock1_tree_f64_accuracy",
                f"f_fixed={f_fixed:.10f} t24={t24:.10f} t19={t19:.10f}",
            )

    print(f"\nSUMMARY: passed={len(PASSES)} failed={len(FAILS)}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
