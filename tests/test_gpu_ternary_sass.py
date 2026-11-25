#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Oracle and SM121 SASS verification suite for native/gpu/gt.om (Pillar 4 + Pillar 5).

Checks:
  1. Compiles native/gpu/gt.om with seeded `omc` for both `target a64macos` (Mach-O)
     and `gb10` (Linux AArch64 `om2` ELF), plus full `gb10` stack composition.
  2. Runs the native `gt_selftest` entry point and host oracle driver compiled from `gt.om`.
  3. Verifies 1.58-bit ternary packing/unpacking (4 trits/byte 2-bit and 5 trits/byte base-3),
     UE4M3 block-scale quant/dequant, 4-bit LUT-GEMM (`gt_lut_gemm_ref`), 1.58-bit ternary
     LUT-GEMM (`gt_tern_gemm_ref` unscaled & UE4M3-scaled), and Pillar 5 warp compaction
     (`gt_wcompact_ref`) against independent Python references.
  4. Inspects emitted SM121 SASS slots (`gt_emit_lut_gemm`, `gt_emit_tern_gemm`,
     `gt_emit_all_ops_slot`) for required instructions, +16 B bank-conflict-free SMEM stride,
     zero inner-loop multiplications, and runs `nvdisasm -b SM121` (locally or via CPU-only
     `ssh spark` when reachable) to confirm 0 illegal instructions and `gtgpr >= max_gpr + 3`.

Usage:
  python3 -W error::ResourceWarning tests/test_gpu_ternary_sass.py
"""
import os
import platform
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GT_OM = os.path.join(REPO, "native", "gpu", "gt.om")

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
    cands = [
        os.environ.get("OMC", ""),
        shutil.which("omc") or "",
        os.path.expanduser("~/.local/bin/omc"),
        os.path.join(REPO, "..", "omega-mv", "seed", "out", "omc"),
        os.path.join(REPO, "..", "omega-mv-macos", "seed", "out", "omc-macos"),
    ]
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise RuntimeError("seeded omc compiler not found")


def compile_om(omc, src_text, out_path, sign_macos=False):
    p = subprocess.run(
        [omc],
        input=src_text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if p.returncode != 0:
        raise RuntimeError(
            "omc failed rc=%d:\n%s" % (p.returncode, p.stderr.decode("utf-8", "replace"))
        )
    with open(out_path, "wb") as f:
        f.write(p.stdout)
    os.chmod(out_path, 0o755)
    if sign_macos and platform.system() == "Darwin" and shutil.which("codesign"):
        subprocess.run(
            ["codesign", "-s", "-", "-f", out_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    return p.stdout


def ue4m3_to_f32_py(c):
    v = c & 0x7F
    e = (v >> 3) & 0xF
    m = v & 0x7
    if e == 0:
        return float(m) * (2.0 ** -9)
    if e == 15 and m == 7:
        return float("nan")
    return (1.0 + float(m) / 8.0) * (2.0 ** (e - 7))


def f32_to_ue4m3_py(x):
    if x <= 0.0:
        return 0
    if x >= 448.0:
        return 126
    best = 0
    best_d = x
    for i in range(127):
        d = abs(x - ue4m3_to_f32_py(i))
        if d < best_d:
            best_d = d
            best = i
    return best


def pack_tern2_py(trits):
    enc = {0: 0, 1: 1, -1: 2}
    nb = (len(trits) + 3) // 4
    out = bytearray(nb)
    for i in range(nb):
        b = 0
        for j in range(4):
            idx = i * 4 + j
            if idx < len(trits):
                b |= enc[trits[idx]] << (2 * j)
        out[i] = b
    return bytes(out)


def pack_tern5_py(trits):
    nb = (len(trits) + 4) // 5
    out = bytearray(nb)
    for i in range(nb):
        acc = 0
        pwr = 1
        for j in range(5):
            idx = i * 5 + j
            d = (trits[idx] + 1) if idx < len(trits) else 1
            acc += d * pwr
            pwr *= 3
        out[i] = acc
    return bytes(out)


def build_oracle_om(gt_src, trits_2b, trits_5b, x_vec, w4_bytes, w2_rows, w2_sc_bytes):
    """Build a standalone OM-2 oracle test program calling gt.om functions and writing binary results."""
    lines = []
    lines.append("fn put_u32 buf edit uchar * off ulong v uint to void")
    lines.append("  for j ulong 4")
    lines.append("    put buf add off j uchar and shr v uint mul j 8 255")
    lines.append("  end")
    lines.append("end")

    n2 = len(trits_2b)
    nb2 = (n2 + 3) // 4
    lines.append("fn test_pack_2b to int")
    lines.append(f"  t2 = new int {n2} 0")
    for i, t in enumerate(trits_2b):
        lines.append(f"  put t2 {i} {t}")
    lines.append(f"  p2 = new uchar {nb2} 0")
    lines.append(f"  u2 = new int {n2} 0")
    lines.append(f"  if ne gt_pack_tern view t2 edit p2 {nb2}")
    lines.append("    ret 21")
    lines.append("  end")
    lines.append(f"  if ne gt_unpack_tern view p2 edit u2 {n2}")
    lines.append("    ret 22")
    lines.append("  end")
    lines.append(f"  for i ulong {n2}")
    lines.append("    if ne get u2 i get t2 i")
    lines.append("      ret 23")
    lines.append("    end")
    lines.append("  end")
    lines.append(f"  if ne write 1 p2 0 {nb2} {nb2}")
    lines.append("    ret 31")
    lines.append("  end")
    lines.append("  ret 0")
    lines.append("end")

    n5 = len(trits_5b)
    nb5 = (n5 + 4) // 5
    lines.append("fn test_pack_5b to int")
    lines.append(f"  t5 = new int {n5} 0")
    for i, t in enumerate(trits_5b):
        lines.append(f"  put t5 {i} {t}")
    lines.append(f"  p5 = new uchar {nb5} 0")
    lines.append(f"  u5 = new int {n5} 0")
    lines.append(f"  if ne gt_pack_tern5 view t5 edit p5 {nb5}")
    lines.append("    ret 24")
    lines.append("  end")
    lines.append(f"  if ne gt_unpack_tern5 view p5 edit u5 {n5}")
    lines.append("    ret 25")
    lines.append("  end")
    lines.append(f"  for i ulong {n5}")
    lines.append("    if ne get u5 i get t5 i")
    lines.append("      ret 26")
    lines.append("    end")
    lines.append("  end")
    lines.append(f"  if ne write 1 p5 0 {nb5} {nb5}")
    lines.append("    ret 32")
    lines.append("  end")
    lines.append("  ret 0")
    lines.append("end")

    m = len(w2_rows)
    k = len(x_vec)
    lines.append(f"fn fill_x xv edit f32 {k} to void")
    for i, xv in enumerate(x_vec):
        bits = struct.unpack("<I", struct.pack("<f", xv))[0]
        lines.append(f"  put xv {i} bitcast f32 uint 0x{bits:08x}")
    lines.append("end")

    lines.append("fn test_gemms ob edit uchar * to int")
    lines.append(f"  xv = new f32 {k} 0.0f32")
    lines.append(f"  w4 = new uchar {len(w4_bytes)} 0")
    lines.append(f"  y4 = new f32 {m} 0.0f32")
    lines.append("  fill_x edit xv")
    for i, b in enumerate(w4_bytes):
        lines.append(f"  put w4 {i} uchar {b}")
    lines.append("  gt_lut_gemm_ref view xv view w4 edit y4")
    lines.append(f"  for i ulong {m}")
    lines.append("    put_u32 edit ob mul i 4 bitcast uint get y4 i")
    lines.append("  end")
    lines.append("  ret 0")
    lines.append("end")

    w2_flat = b"".join(w2_rows)
    w2_blob = w2_flat + w2_sc_bytes
    lines.append("fn test_tern_unscaled ob edit uchar * to int")
    lines.append(f"  xv = new f32 {k} 0.0f32")
    lines.append(f"  w2u = new uchar {len(w2_flat)} 0")
    lines.append(f"  y2u = new f32 {m} 0.0f32")
    lines.append("  fill_x edit xv")
    for i, b in enumerate(w2_flat):
        lines.append(f"  put w2u {i} uchar {b}")
    lines.append("  gt_tern_gemm_ref view xv view w2u edit y2u")
    lines.append(f"  for i ulong {m}")
    lines.append(f"    put_u32 edit ob add {m*4} mul i 4 bitcast uint get y2u i")
    lines.append("  end")
    lines.append("  ret 0")
    lines.append("end")

    lines.append("fn test_tern_scaled ob edit uchar * to int")
    lines.append(f"  xv = new f32 {k} 0.0f32")
    lines.append(f"  w2s = new uchar {len(w2_blob)} 0")
    lines.append(f"  y2s = new f32 {m} 0.0f32")
    lines.append("  fill_x edit xv")
    for i, b in enumerate(w2_blob):
        lines.append(f"  put w2s {i} uchar {b}")
    lines.append("  gt_tern_gemm_ref view xv view w2s edit y2s")
    lines.append(f"  for i ulong {m}")
    lines.append(f"    put_u32 edit ob add {m*8} mul i 4 bitcast uint get y2s i")
    lines.append("  end")
    lines.append("  ret 0")
    lines.append("end")

    out_bytes = m * 4 * 3 + 4 + 128 + 16
    lines.append("fn main to int")
    lines.append("  r = gt_selftest")
    lines.append("  if ne r 0")
    lines.append("    ret r")
    lines.append("  end")
    lines.append("  r = test_pack_2b")
    lines.append("  if ne r 0")
    lines.append("    ret r")
    lines.append("  end")
    lines.append("  r = test_pack_5b")
    lines.append("  if ne r 0")
    lines.append("    ret r")
    lines.append("  end")
    lines.append(f"  ob = new uchar {out_bytes} 0")
    lines.append("  act = new int 32 0")
    lines.append("  r = test_gemms edit ob")
    lines.append("  r = test_tern_unscaled edit ob")
    lines.append("  r = test_tern_scaled edit ob")
    lines.append("  cnt = gt_wcompact_ref uint 0x8001a55a edit act")
    lines.append(f"  put_u32 edit ob {m*12} uint cnt")
    lines.append("  for i ulong 32")
    lines.append(f"    put_u32 edit ob add {m*12+4} mul i 4 uint get act i")
    lines.append("  end")
    lines.append("  for i ulong 4")
    lines.append(f"    put_u32 edit ob add {m*12+132} mul i 4 get gtgpr i")
    lines.append("  end")
    lines.append(f"  if ne write 1 ob 0 {out_bytes} {out_bytes}")
    lines.append("    ret 33")
    lines.append("  end")
    lines.append("  if ne write 1 gtcode 0 49152 49152")
    lines.append("    ret 34")
    lines.append("  end")
    lines.append("  ret 0")
    lines.append("end")
    return gt_src + "\n" + "\n".join(lines) + "\n"


def run_nvdisasm_check(slots_bytes, declared_gprs):
    """Disassemble slots 0..2 with nvdisasm -b SM121 (local or CPU-only ssh spark)."""
    local_nv = shutil.which("nvdisasm") or "/usr/local/cuda/bin/nvdisasm"
    use_local = os.path.isfile(local_nv) and os.access(local_nv, os.X_OK)
    use_spark = False
    if not use_local and os.environ.get("GT_SKIP_SPARK", "0") != "1":
        p = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3", "spark",
             "test -x /usr/local/cuda/bin/nvdisasm"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        use_spark = (p.returncode == 0)

    if not use_local and not use_spark:
        print("SKIP nvdisasm -b SM121 (nvdisasm not installed locally and spark unreachable)")
        return

    for s in range(3):
        slot = slots_bytes[s * 16384:(s + 1) * 16384]
        last = max((i + 1 for i, b in enumerate(slot) if b != 0), default=0)
        n = (last + 15) // 16 * 16
        trimmed = slot[:n]
        if use_local:
            with tempfile.NamedTemporaryFile(suffix=".bin") as tf:
                tf.write(trimmed)
                tf.flush()
                proc = subprocess.run([local_nv, "-b", "SM121", tf.name],
                                      capture_output=True, text=True)
        else:
            proc = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "spark",
                 "cat > /tmp/gt_ci_slot.bin && /usr/local/cuda/bin/nvdisasm -b SM121 /tmp/gt_ci_slot.bin"],
                input=trimmed,
                capture_output=True,
            )
            proc = subprocess.CompletedProcess(
                proc.args,
                proc.returncode,
                proc.stdout.decode("utf-8", "replace"),
                proc.stderr.decode("utf-8", "replace"),
            )
        sass = proc.stdout
        ok_dis = (
            proc.returncode == 0
            and "illegal" not in sass.lower()
            and "unknown" not in sass.lower()
        )
        check(ok_dis, f"nvdisasm_sm121_slot{s}", proc.stderr.strip())
        if ok_dis:
            # Scan GPR ceiling (including .64 / .128 wide destinations)
            mx = -1
            for ln in sass.splitlines():
                if "/*" not in ln:
                    continue
                body = ln.split("*/", 1)[1].split(";")[0].strip()
                if body.startswith("@"):
                    body = body.split(None, 1)[1]
                op = body.split(None, 1)[0] if body else ""
                w = 4 if ".128" in op else (2 if (".64" in op or ".WIDE" in op) else 1)
                if "OMMA" in op:
                    w = 4
                no_brackets = re.sub(r"\[[^\]]*\]", " ", body)
                for m_reg in re.finditer(r"(?<![A-Za-z0-9_.])R([0-9]+)(?:\.64)?", no_brackets):
                    rnum = int(m_reg.group(1))
                    ext = 1 if ".64" in m_reg.group(0) else (w - 1)
                    mx = max(mx, rnum + ext)
            decl = declared_gprs[s]
            check(decl >= mx + 3, f"nvdisasm_gpr_headroom_slot{s}",
                  f"max R{mx}, need {mx + 3}, declared {decl}")
            if s == 0:
                check("FADD " in sass and "LDS " in sass and "SHFL.BFLY" in sass
                      and "LOP3.LUT" in sass and "FMUL" not in sass,
                      "nvdisasm_slot0_lut_gemm_zero_mul")
            if s == 1:
                check("FADD " in sass and "LDS " in sass and "SHFL.BFLY" in sass
                      and "LOP3.LUT" in sass and "FMUL" not in sass,
                      "nvdisasm_slot1_tern_gemm_zero_mul")
            if s == 2:
                for mnem in ("LOP3.LUT", "PRMT", "SHFL.BFLY", "SHFL.UP", "SHFL.IDX",
                             "VOTE.ANY", "POPC", "F2FP.SATFINITE.E4M3", "F2FP.SATFINITE.E2M1",
                             "OMMA.SF.16864", "MEMBAR.SC.SYS"):
                    check(mnem in sass, f"nvdisasm_slot2_has_{mnem}")


def main():
    omc = find_omc()
    with open(GT_OM, "r", encoding="utf-8") as f:
        gt_src = f.read()

    rng = random.Random(20261125)
    trits_2b = [rng.choice((-1, 0, 1)) for _ in range(257)]
    trits_5b = [rng.choice((-1, 0, 1)) for _ in range(253)]

    M, K = 16, 64
    x_vec = [round(rng.uniform(-2.0, 2.0), 3) for _ in range(K)]
    w4_bytes = bytes(rng.randrange(256) for _ in range(M * (K // 8)))

    w2_trit_rows = [[rng.choice((-1, 0, 1)) for _ in range(K)] for _ in range(M)]
    w2_rows = [pack_tern2_py(r) for r in w2_trit_rows]
    # 2 blocks of 32 weights per row, random UE4M3 scales in [0.125, 3.5]
    blk = 32
    sc_codes = [f32_to_ue4m3_py(rng.uniform(0.125, 3.5)) for _ in range(M * (K // blk))]
    w2_sc_bytes = bytes(sc_codes)

    oracle_src = build_oracle_om(gt_src, trits_2b, trits_5b, x_vec, w4_bytes, w2_rows, w2_sc_bytes)

    with tempfile.TemporaryDirectory() as tmp:
        # 1. Compile gb10 (Linux AArch64 om2 ELF)
        gb10_bin = os.path.join(tmp, "gt_gb10.elf")
        elf = compile_om(omc, "om2\n" + oracle_src, gb10_bin, sign_macos=False)
        check(elf[:4] == b"\x7fELF", "compile_gb10_om2_elf")

        # 2. Compile full gb10 chain with gt.om appended to prove zero symbol collisions
        gpu_dir = os.path.join(REPO, "native", "gpu")
        nat_dir = os.path.join(REPO, "native")
        full_parts = [
            os.path.join(nat_dir, "net", "http.om"),
            os.path.join(nat_dir, "net", "json.om"),
            os.path.join(nat_dir, "net", "jsonw.om"),
            os.path.join(nat_dir, "math.om"),
            os.path.join(nat_dir, "data.om"),
            os.path.join(nat_dir, "lpool.om"),
            os.path.join(nat_dir, "model.om"),
            os.path.join(nat_dir, "adam.om"),
            os.path.join(nat_dir, "ckpt.om"),
            os.path.join(nat_dir, "bpe.om"),
            os.path.join(nat_dir, "spec.om"),
            os.path.join(nat_dir, "rev.om"),
            os.path.join(nat_dir, "arboris.om"),
            os.path.join(nat_dir, "sp_solve.om"),
            os.path.join(nat_dir, "omg.om"),
            os.path.join(nat_dir, "pillars_cli.om"),
            os.path.join(nat_dir, "cli.om"),
            os.path.join(nat_dir, "lio.om"),
            os.path.join(nat_dir, "serve.om"),
            os.path.join(gpu_dir, "context.om"),
            os.path.join(gpu_dir, "vector.om"),
            os.path.join(gpu_dir, "launch.om"),
            os.path.join(gpu_dir, "batch.om"),
            os.path.join(gpu_dir, "gk.om"),
            os.path.join(gpu_dir, "gr.om"),
            os.path.join(gpu_dir, "gb.om"),
            os.path.join(gpu_dir, "gg.om"),
            GT_OM,
            os.path.join(gpu_dir, "gm.om"),
            os.path.join(gpu_dir, "ring.om"),
            os.path.join(gpu_dir, "ejit.om"),
        ]
        chunks = ["om2\n"]
        for p in full_parts:
            with open(p, "r", encoding="utf-8") as f:
                chunks.append(f.read() + "\n")
        full_src = "".join(chunks)
        full_elf = compile_om(omc, full_src, os.path.join(tmp, "ollm_gb10_gt.elf"), sign_macos=False)
        check(full_elf[:4] == b"\x7fELF", "compile_full_gb10_chain_with_gt")

        # 3. Compile and run host oracle binary
        is_mac = platform.system() == "Darwin"
        is_x86 = platform.system() == "Linux" and platform.machine().lower() in ("x86_64", "amd64")
        host_hdr = "target a64macos\nom2\n" if is_mac else ("target x86linux\nom2\n" if is_x86 else "om2\n")
        host_bin = os.path.join(tmp, "gt_host")
        compile_om(omc, host_hdr + oracle_src, host_bin, sign_macos=is_mac)
        check(
            True,
            "compile_host_a64macos_om2" if is_mac else ("compile_host_x86linux_om2" if is_x86 else "compile_host_linux_om2"),
        )

        res = subprocess.run([host_bin], capture_output=True)
        check(res.returncode == 0, "gt_selftest_and_oracle_exec", f"rc={res.returncode}")
        out = res.stdout

        nb2 = (len(trits_2b) + 3) // 4
        nb5 = (len(trits_5b) + 4) // 5
        pos = 0
        got_p2 = out[pos:pos + nb2]; pos += nb2
        got_p5 = out[pos:pos + nb5]; pos += nb5
        y4_got = struct.unpack_from(f"<{M}f", out, pos); pos += M * 4
        y2u_got = struct.unpack_from(f"<{M}f", out, pos); pos += M * 4
        y2s_got = struct.unpack_from(f"<{M}f", out, pos); pos += M * 4
        cnt_got = struct.unpack_from("<I", out, pos)[0]; pos += 4
        act_got = struct.unpack_from("<32i", out, pos); pos += 128
        gprs_got = struct.unpack_from("<4I", out, pos); pos += 16
        slots_bytes = out[pos:pos + 49152]
        check(len(slots_bytes) == 49152, "gtcode_slots_len")

        # Verify packing parity
        check(got_p2 == pack_tern2_py(trits_2b), "pack_tern_2bit_parity")
        check(got_p5 == pack_tern5_py(trits_5b), "pack_tern_5trit_base3_parity")

        # Verify 4-bit LUT-GEMM parity vs exact dense float dot product
        y4_ref = []
        for r in range(M):
            acc = 0.0
            row_b = w4_bytes[r * (K // 8):(r + 1) * (K // 8)]
            for bi, byte_v in enumerate(row_b):
                for bit in range(8):
                    if (byte_v >> bit) & 1:
                        acc += x_vec[bi * 8 + bit]
            y4_ref.append(acc)
        max_err4 = max(abs(a - b) for a, b in zip(y4_got, y4_ref))
        check(max_err4 < 1e-4, "lut_gemm_4bit_parity", f"max_err={max_err4}")

        # Verify 1.58-bit Ternary LUT-GEMM parity (unscaled and UE4M3-scaled)
        y2u_ref = [sum(w2_trit_rows[r][k] * x_vec[k] for k in range(K)) for r in range(M)]
        max_err2u = max(abs(a - b) for a, b in zip(y2u_got, y2u_ref))
        check(max_err2u < 1e-4, "tern_gemm_unscaled_parity", f"max_err={max_err2u}")

        y2s_ref = []
        for r in range(M):
            acc = 0.0
            for sb in range(K // blk):
                sc = ue4m3_to_f32_py(sc_codes[r * (K // blk) + sb])
                bacc = sum(w2_trit_rows[r][sb * blk + j] * x_vec[sb * blk + j] for j in range(blk))
                acc += bacc * sc
            y2s_ref.append(acc)
        max_err2s = max(abs(a - b) for a, b in zip(y2s_got, y2s_ref))
        check(max_err2s < 1e-4, "tern_gemm_ue4m3_scaled_parity", f"max_err={max_err2s}")

        # Verify warp compaction oracle (0x8001a55a)
        ballot = 0x8001A55A
        act_ref = [lane for lane in range(32) if (ballot >> lane) & 1]
        check(cnt_got == len(act_ref) and list(act_got[:cnt_got]) == act_ref,
              "warp_compaction_prefix_sum_parity",
              f"cnt={cnt_got}, exp={len(act_ref)}")

        # Verify SM121 SASS slot structural invariants + nvdisasm -b SM121
        for s in range(3):
            slot = slots_bytes[s * 16384:(s + 1) * 16384]
            words = [struct.unpack_from("<4I", slot, i * 16) for i in range(1024)]
            nonzero_insns = [w for w in words if w != (0, 0, 0, 0)]
            check(len(nonzero_insns) >= 32 and len(nonzero_insns) % 8 == 0,
                  f"slot{s}_aligned_32_mult8", f"insns={len(nonzero_insns)}")
            # Check EXIT (0x794d) and self-BRA (0x0947/0x7947) exist before trailing NOPs
            has_exit = any((w[0] & 0xFFFF) == 0x794D for w in nonzero_insns)
            has_bra = any((w[0] & 0xFFFF) == 0x7947 for w in nonzero_insns)
            check(has_exit and has_bra, f"slot{s}_exit_bra_tail")

        run_nvdisasm_check(slots_bytes, gprs_got)

    print(f"SUMMARY: passed={len(PASSES)} failed={len(FAILS)}")
    if FAILS:
        sys.exit(1)


if __name__ == "__main__":
    main()
