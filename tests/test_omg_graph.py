#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""tests/test_omg_graph.py: Referee & A/B verification suite for native/omg.om (Pillar 8).

Verifies:
  1. Cross-target compilation of native/omg.om with native/math.om, native/data.om,
     native/lpool.om for BOTH `target a64macos` and `gb10` (`om2` Linux AArch64 ELF).
  2. Built-in `omg_selftest` and V=256 vs V=4096 logit/CE footprint metrics.
  3. Alpha-equivalent & formatting-invariant De Bruijn canonicalization (`OMG` u16 stream
     and 64-bit/256-bit `SEMANTIC_ID`), plus non-equivalent program separation.
  4. Binary-to-binary .om -> OMG -> .om -> omc -> execute parity roundtrips across
     arithmetic, loops, branches, local arrays, view/edit slices, floats, and calls.
  5. 1,000 adversarial/random `omg_valid_mask` samples (0% invalid graphs, 100% omc
     compile on decoded batch) + invalid-token mutant rejection + O(1) mask invariants.
  6. `OMG_MACRO` registration, semantic deduplication, context-typed mask gating,
     deterministic expansion, and semantic-ID equivalence.
  7. `OMGC` binary corpus packing, semantic deduplication, and >=3x token compression vs
     raw `.om` text and JSON AST.
"""

import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def find_omc():
    for cand in (
        os.environ.get("OMC", ""),
        shutil.which("omc") or "",
        os.path.expanduser("~/.local/bin/omc"),
    ):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    raise FileNotFoundError("seeded omc compiler not found")


OMC = find_omc()

IO_STUBS = r"""
fn oargc to ulong
  ret ulong argcnt
end
fn oargr i ulong to long
  ret argread i oarg 0 4096
end
fn oflush to void
  n = get oon 0
  p = ulong 0
  while lt p n bound 4097
    w = write 1 oout p sub n p
    if le w 0
      exit 70
    end
    p = add p ulong w
  end
  put oon 0 0
end
fn owflush k ulong to void
end
fn owopen k ulong i ulong to void
end
fn owclose k ulong to void
end
fn oprd ai ulong pos ulong off ulong n ulong to ulong
  ret 0
end
fn odread i ulong to ulong
  ret 0
end
fn ooffrd ai ulong to void
end
fn ofsz ai ulong to ulong
  ret 0
end
"""

TEST_DRIVER = r"""
fn read_stdin_src to void
  p = ulong 0
  while lt p 65536 bound 65536
    r = read 0 omg_src p sub 65536 p
    if le r 0
      break
    end
    p = add p ulong r
  end
  put omg_nsrc 0 p
end

fn read_stdin_toks to void
  lenb = ulong 0
  while lt lenb 32768 bound 32768
    r = read 0 omg_corpus lenb sub 32768 lenb
    if le r 0
      break
    end
    lenb = add lenb ulong r
  end
  nt = div lenb 2
  for i ulong nt
    lo = ulong get omg_corpus mul i 2
    hi = ulong get omg_corpus add mul i 2 1
    put omg_toks i ushort or shl hi 8 lo
  end
  put omg_ntok 0 nt
end

fn write_dec_all fd int n ulong to void
  p = ulong 0
  while lt p n bound 65536
    w = write fd omg_dec p sub n p
    if le w 0
      break
    end
    p = add p ulong w
  end
end

fn write_corp_all fd int n ulong to void
  p = ulong 0
  while lt p n bound 65536
    w = write fd omg_corpus p sub n p
    if le w 0
      break
    end
    p = add p ulong w
  end
end

fn setup_test_macros to void
  put omg_mac_cnt 0 0
  put omg_mac_pcur 0 0
  note MAC_0: expr (U64) -> U64: add mul x 2 1
  put omg_mac_tmp_in 0 OMG_TY_U64
  put omg_mac_tmp_tok 0 ushort OMG_OP_ADD
  put omg_mac_tmp_tok 1 ushort OMG_OP_MUL
  put omg_mac_tmp_tok 2 ushort 256
  put omg_mac_tmp_tok 3 ushort add OMG_LIT_BASE 2
  put omg_mac_tmp_tok 4 ushort add OMG_LIT_BASE 1
  put omg_mac_tmp_len 0 5
  m0 = omg_macro_register 0 OMG_TY_U64 1
  note MAC_1: stmt (U64, U64) -> VOID: v = add x y
  put omg_mac_tmp_in 0 OMG_TY_U64
  put omg_mac_tmp_in 1 OMG_TY_U64
  put omg_mac_tmp_tok 0 ushort OMG_LET
  put omg_mac_tmp_tok 1 ushort OMG_TY_U64
  put omg_mac_tmp_tok 2 ushort OMG_OP_ADD
  put omg_mac_tmp_tok 3 ushort 256
  put omg_mac_tmp_tok 4 ushort 257
  put omg_mac_tmp_len 0 5
  m1 = omg_macro_register 1 OMG_TY_VOID 2
end

fn cmd_parse to int
  read_stdin_src
  rc = omg_parse_om
  if ne rc 0
    ret int rc
  end
  sid = omg_semantic_id
  oputx get omg_sem_hash 0
  oputx get omg_sem_hash 1
  oputx get omg_sem_hash 2
  oputx get omg_sem_hash 3
  oputc 32
  oputu get omg_ntok 0
  oputc 10
  oflush
  nb = omg_encode_stream
  write_corp_all 2 nb
  ret 0
end

fn cmd_decode to int
  read_stdin_toks
  rc = omg_decode_om
  if ne rc 0
    ret 10
  end
  write_dec_all 1 get omg_ndec 0
  ret 0
end

fn cmd_validate to int
  read_stdin_toks
  rc = omg_validate
  if ne rc 0
    oputl rc
    oputc 32
    oputu get omg_st 14
    oputc 10
    oflush
    ret 1
  end
  ret 0
end

fn cmd_sample_n to int
  count = oargu 2
  budget = oargu 3
  setup_test_macros
  for sd ulong count
    rc = omg_sample_masked add 5000 sd budget
    if ne rc 0
      oputu sd
      oputc 32
      oputl rc
      oputc 10
      oflush
      ret 1
    end
    vr = omg_validate
    if ne vr 0
      oputu sd
      oputc 32
      oputl vr
      oputc 10
      oflush
      ret 2
    end
    if lt sd 200
      rc2 = omg_decode_om
      if ne rc2 0
        ret 3
      end
      oputc 61
      oputc 61
      oputc 61
      oputu sd
      oputc 10
      oflush
      write_dec_all 1 get omg_ndec 0
    end
  end
  ret 0
end

fn cmd_mask_probe to int
  setup_test_macros
  omg_init
  rc = omg_step OMG_BEGIN
  rc = omg_step OMG_FN_MAIN
  note Inside main (ret I32): check mask at EK_STMT
  omg_valid_mask 40
  for i ulong 256
    oputu ulong get omg_mask i
  end
  oputc 10
  note Start OMG_LET TY_U64 -> check mask at EK_EXPR(TY_U64) (MAC_0 should be allowed!)
  rc = omg_step OMG_LET
  rc = omg_step OMG_TY_U64
  omg_valid_mask 40
  for i2 ulong 256
    oputu ulong get omg_mask i2
  end
  oputc 10
  note Step OMG_MACRO -> check mask at EK_MAC_SLOT(TY_U64) (only MAC_0=224 allowed)
  rc = omg_step OMG_MACRO
  omg_valid_mask 40
  for i3 ulong 256
    oputu ulong get omg_mask i3
  end
  oputc 10
  note Step MAC_0 + LIT_5, then start OMG_LET TY_F32 -> MAC_0 must NOT be allowed!
  rc = omg_step add OMG_MAC_BASE 0
  rc = omg_step add OMG_LIT_BASE 5
  rc = omg_step OMG_LET
  rc = omg_step OMG_TY_F32
  omg_valid_mask 40
  for i4 ulong 256
    oputu ulong get omg_mask i4
  end
  oputc 10
  oflush
  ret 0
end

fn cmd_macro_expand to int
  setup_test_macros
  read_stdin_toks
  rc = omg_validate
  if ne rc 0
    ret 1
  end
  rc2 = omg_macro_expand
  if ne rc2 0
    ret 2
  end
  sid = omg_semantic_id
  oputx sid
  oputc 32
  oputu get omg_ntok 0
  oputc 10
  oflush
  nb = omg_encode_stream
  write_corp_all 2 nb
  ret 0
end

fn cmd_corpus to int
  omg_corp_reset
  note Program 1 (scale vA)
  omg_load_text omg_ex_a
  if ne omg_parse_om 0
    ret 1
  end
  r1 = omg_corpus_add
  note Program 2 (alpha-renamed scale vB -> must dedup to 0!)
  omg_load_text omg_ex_b
  if ne omg_parse_om 0
    ret 2
  end
  r2 = omg_corpus_add
  note Program 3 (distinct program: simple add)
  omg_init_parse
  put omg_ntok 0 0
  omg_emit OMG_BEGIN
  omg_emit OMG_FN_MAIN
  omg_emit OMG_RET
  omg_emit add OMG_LIT_BASE 7
  omg_emit OMG_END
  omg_emit OMG_END_UNIT
  r3 = omg_corpus_add
  if or ne r1 1 or ne r2 0 ne r3 1
    ret 3
  end
  nb = omg_pack_corpus
  if eq nb 0
    ret 4
  end
  write_corp_all 1 nb
  ret 0
end

fn main to int
  na = oargc
  if lt na 2
    ret 2
  end
  m = oargu 1
  if eq m 0
    ret int omg_selftest
  end
  if eq m 1
    ret cmd_parse
  end
  if eq m 2
    ret cmd_decode
  end
  if eq m 3
    ret cmd_validate
  end
  if eq m 4
    ret cmd_sample_n
  end
  if eq m 5
    ret cmd_mask_probe
  end
  if eq m 6
    ret cmd_macro_expand
  end
  if eq m 7
    ret cmd_corpus
  end
  ret 2
end
"""


def compile_om(src_text, out_path):
    p = subprocess.run([OMC], input=src_text.encode("utf-8"), capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"omc failed rc={p.returncode}: {p.stderr.decode('latin-1')}")
    with open(out_path + ".tmp", "wb") as f:
        f.write(p.stdout)
    os.chmod(out_path + ".tmp", 0o755)
    os.replace(out_path + ".tmp", out_path)
    if platform.system() == "Darwin" and shutil.which("codesign"):
        subprocess.run(
            ["codesign", "-s", "-", "-f", out_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def pack_tokens(toks):
    return struct.pack(f"<{len(toks)}H", *toks)


def unpack_tokens(raw):
    assert len(raw) % 2 == 0
    return list(struct.unpack(f"<{len(raw) // 2}H", raw))


class TestOmgGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp(prefix="omg_graph_test_")
        unit_files = [
            os.path.join(REPO, "native", "math.om"),
            os.path.join(REPO, "native", "data.om"),
            os.path.join(REPO, "native", "lpool.om"),
            os.path.join(REPO, "native", "omg.om"),
        ]
        parts = []
        for path in unit_files:
            with open(path, "r", encoding="utf-8") as f:
                parts.append(f.read())
        cls.base_om = "\n".join(parts) + "\n" + IO_STUBS + "\n" + TEST_DRIVER + "\n"
        cls.bin_gb10 = os.path.join(cls.tmpdir, "omg_test_gb10")
        compile_om("om2\n" + cls.base_om, cls.bin_gb10)
        if platform.system() == "Darwin":
            cls.bin_host = os.path.join(cls.tmpdir, "omg_test_macos")
            cls.host_hdr = "target a64macos\n"
            compile_om("target a64macos\nom2\n" + cls.base_om, cls.bin_host)
        elif platform.machine() in ("x86_64", "amd64"):
            cls.bin_host = os.path.join(cls.tmpdir, "omg_test_x86")
            cls.host_hdr = "target x86linux\n"
            compile_om("target x86linux\nom2\n" + cls.base_om, cls.bin_host)
        else:
            cls.bin_host = cls.bin_gb10
            cls.host_hdr = ""

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def run_cli(self, *args, stdin_bytes=b""):
        p = subprocess.run(
            [self.bin_host] + [str(a) for a in args],
            input=stdin_bytes,
            capture_output=True,
            timeout=60,
        )
        return p.returncode, p.stdout, p.stderr

    def parse_om(self, src):
        rc, out, err = self.run_cli("1", stdin_bytes=src.encode("utf-8"))
        self.assertEqual(rc, 0, f"parse failed rc={rc} err={err}")
        fields = out.decode("utf-8").strip().split()
        sem_id_hex = fields[0]
        ntok = int(fields[1])
        toks = unpack_tokens(err)
        self.assertEqual(len(toks), ntok)
        for t in toks:
            self.assertLess(t, 256)
        return sem_id_hex, toks

    def decode_omg(self, toks):
        rc, out, err = self.run_cli("2", stdin_bytes=pack_tokens(toks))
        self.assertEqual(rc, 0, f"decode failed rc={rc} err={err}")
        return out.decode("utf-8")

    def test_01_cross_target_gb10_elf_and_macos(self):
        with open(self.bin_gb10, "rb") as f:
            hdr = f.read(20)
        self.assertEqual(hdr[:4], b"\x7fELF")
        e_machine = struct.unpack("<H", hdr[18:20])[0]
        self.assertEqual(e_machine, 183, "gb10 output must be AArch64 ELF (EM_AARCH64=183)")
        if platform.system() == "Darwin":
            with open(self.bin_host, "rb") as f:
                mhdr = f.read(4)
            self.assertEqual(mhdr, b"\xcf\xfa\xed\xfe", "macos output must be 64-bit Mach-O")

    def test_02_builtin_selftest_and_metrics(self):
        rc, out, err = self.run_cli("0")
        txt = out.decode("utf-8")
        self.assertEqual(rc, 0, f"selftest failed rc={rc}: {txt} {err}")
        self.assertIn("OMG_METRICS V=256 V_BASE=4096 LOGIT_REDUCTION=16x", txt)
        self.assertIn("OMG_SELFTEST PASS samples=200", txt)

    def test_03_alpha_equivalence_and_semantic_id_separation(self):
        pairs = [
            (
                """om2
note Variant A: helper + loop + branches
fn accum_step acc ulong x ulong to ulong
  t = add mul acc 3 x
  ret add t 1
end
fn main to int
  s = ulong 1
  for i ulong 4
    s = accum_step s i
  end
  if gt s 20
    ret int rem s 251
  else
    ret 0
  end
end
""",
                """om2
note Completely different names, parens, blank lines & comments
fn f_renamed p_state ulong p_elem ulong to ulong
  note extra comment inside fn
  intermediate_val = add (mul p_state 3) p_elem
  ret (add intermediate_val 1)
end

fn main to int
  total = ulong 1
  for loop_idx ulong 4
    total = f_renamed total loop_idx
  end
  if gt total 20
    ret int (rem total 251)
  else
    ret 0
  end
end
""",
            ),
            (
                """om2
fn sum_arr a view ulong * n ulong to ulong
  acc = ulong 0
  i = ulong 0
  while lt i n bound 64
    acc = add acc get a i
    i = add i 1
  end
  ret acc
end
fn main to int
  buf = new ulong 4 3
  put buf 1 9
  put buf 2 5
  s = sum_arr view buf len buf
  ret int s
end
""",
                """om2
fn reduce_slice vec view ulong * cnt ulong to ulong
  res = ulong 0
  idx = ulong 0
  while lt idx cnt bound 64
    res = add res (get vec idx)
    idx = add idx 1
  end
  ret res
end
fn main to int
  arr = new ulong 4 3
  put arr 1 9
  put arr 2 5
  total = reduce_slice view arr (len arr)
  ret int total
end
""",
            ),
            (
                """om2
fn fmix x ulong y f64 to ulong
  z = rotr x 3
  w = fma y 2.0f64 1.5f64
  b = bitcast ulong w
  ret xor z b
end
fn main to int
  r = fmix 0xdeadbeef 1.25f64
  ret int and r 255
end
""",
                """om2
fn hash_mix a ulong b f64 to ulong
  r1 = rotr a 3
  r2 = fma b 2.0f64 1.5f64
  bits = bitcast ulong r2
  ret xor r1 bits
end
fn main to int
  out_val = hash_mix 0xdeadbeef 1.25f64
  ret int (and out_val 255)
end
""",
            ),
        ]

        seen_ids = set()
        for idx, (src_a, src_b) in enumerate(pairs):
            sid_a, toks_a = self.parse_om(src_a)
            sid_b, toks_b = self.parse_om(src_b)
            self.assertEqual(toks_a, toks_b, f"pair {idx}: OMG tokens must match under alpha-renaming")
            self.assertEqual(sid_a, sid_b, f"pair {idx}: 256-bit SEMANTIC_ID must match")
            self.assertNotIn(sid_a, seen_ids)
            seen_ids.add(sid_a)

        # Semantically distinct programs MUST produce distinct SEMANTIC_IDs
        d1, _ = self.parse_om("om2\nfn main to int\n  a = int 10\n  b = int 3\n  ret sub a b\nend\n")
        d2, _ = self.parse_om("om2\nfn main to int\n  a = int 10\n  b = int 3\n  ret sub b a\nend\n")
        d3, _ = self.parse_om("om2\nfn main to int\n  a = int 10\n  b = int 3\n  ret add a b\nend\n")
        self.assertEqual(len({d1, d2, d3}), 3)

    def test_04_roundtrip_om_omg_om_compile_and_execute_parity(self):
        programs = [
            (
                "arith_loop_branch",
                """om2
fn step a ulong b ulong to ulong
  ret add mul a 2 xor b 5
end
fn main to int
  x = ulong 3
  for i ulong 5
    x = step x i
  end
  if gt x 50
    ret int rem x 127
  end
  ret 1
end
""",
            ),
            (
                "local_array_edit_view",
                """om2
fn bump a edit ulong * i ulong d ulong to void
  put a i add get a i d
end
fn sum a view ulong * to ulong
  s = ulong 0
  for i ulong len a
    s = add s get a i
  end
  ret s
end
fn main to int
  arr = new ulong 5 2
  bump edit arr 2 10
  bump edit arr 4 7
  ret int sum view arr
end
""",
            ),
            (
                "signed_neg_float_bitcast",
                """om2
fn main to int
  a = long -15
  b = long 28
  c = add a b
  f = fma 2.0f64 3.5f64 1.0f64
  fi = long f
  ret int add c fi
end
""",
            ),
            (
                "while_break_continue",
                """om2
fn main to int
  i = ulong 0
  acc = ulong 0
  while lt i 12 bound 20
    i = add i 1
    if eq rem i 2 0
      continue
    end
    if gt acc 20
      break
    end
    acc = add acc i
  end
  ret int acc
end
""",
            ),
        ]

        for name, src in programs:
            bin_orig = os.path.join(self.tmpdir, f"rt_{name}_orig")
            bin_dec = os.path.join(self.tmpdir, f"rt_{name}_dec")
            compile_om(self.host_hdr + src, bin_orig)
            rc_orig = subprocess.run([bin_orig], capture_output=True).returncode

            sid1, toks1 = self.parse_om(src)
            dec_om = self.decode_omg(toks1)
            compile_om(self.host_hdr + dec_om, bin_dec)
            rc_dec = subprocess.run([bin_dec], capture_output=True).returncode
            self.assertEqual(rc_dec, rc_orig, f"{name}: runtime exit parity mismatch")

            sid2, toks2 = self.parse_om(dec_om)
            self.assertEqual(toks1, toks2, f"{name}: OMG roundtrip idempotence failed")
            self.assertEqual(sid1, sid2, f"{name}: SEMANTIC_ID roundtrip idempotence failed")

    def test_05_1000_masked_samples_and_mutant_rejection(self):
        # Generate & validate 1,000 random/adversarial masked streams, decode & compile first 200
        rc, out, err = self.run_cli("4", "1000", "96")
        self.assertEqual(rc, 0, f"1000 masked samples failed rc={rc}: {out.decode()} {err.decode()}")

        progs = out.decode("utf-8").split("===")
        self.assertEqual(len(progs) - 1, 200)
        for chunk in progs[1:]:
            idx, body = chunk.split("\n", 1)
            p = subprocess.run(
                [OMC],
                input=(self.host_hdr + body).encode("utf-8"),
                capture_output=True,
            )
            self.assertEqual(
                p.returncode,
                0,
                f"decoded sample {idx} failed omc: {p.stderr.decode('latin-1')}\n{body}",
            )

        # Also test tight budgets (12, 24, 48) across 100 samples each
        for b in (12, 24, 48):
            rc_b, out_b, err_b = self.run_cli("4", "100", str(b))
            self.assertEqual(rc_b, 0, f"budget={b} failed: {out_b.decode()} {err_b.decode()}")

        # Mutant rejection tests: every corrupted stream MUST be rejected by omg_validate
        valid_src = "om2\nfn main to int\n  x = ulong 5\n  ret int add x 1\nend\n"
        _, valid_toks = self.parse_om(valid_src)
        rc_v, _, _ = self.run_cli("3", stdin_bytes=pack_tokens(valid_toks))
        self.assertEqual(rc_v, 0)

        mutants = [
            # Out-of-vocab token (>= 256)
            [valid_toks[0], 300] + valid_toks[2:],
            # Wrong header (missing OMG_BEGIN)
            [4] + valid_toks[1:],
            # Type mismatch in OMG_LET: TY_BOOL with LIT_5
            [1, 4, 8, 25, 105, 10, 100, 7, 2],
            # Undeclared De Bruijn REF_3 when n_live == 0
            [1, 4, 10, 163, 7, 2],
            # Statement after OMG_RET (dead code -> omc error 17)
            [1, 4, 10, 100, 10, 100, 7, 2],
            # Missing OMG_RET in non-void fn main
            [1, 4, 7, 2],
            # Truncated stream (missing OMG_END_UNIT)
            valid_toks[:-1],
            # OMG_BREAK outside any loop
            [1, 4, 15, 10, 100, 7, 2],
        ]
        for mi, mut in enumerate(mutants):
            rc_m, out_m, _ = self.run_cli("3", stdin_bytes=pack_tokens(mut))
            self.assertNotEqual(rc_m, 0, f"mutant {mi} unexpectedly accepted: {mut}")

    def test_06_macro_mask_gating_and_expansion(self):
        # Probe O(1) bitmask gating for OMG_MACRO & MAC_0
        rc, out, _ = self.run_cli("5")
        self.assertEqual(rc, 0)
        masks = [ln.strip() for ln in out.decode("utf-8").splitlines()]
        self.assertEqual(len(masks), 4)
        m_stmt, m_u64, m_mac_slot, m_f32 = masks
        # In EK_STMT: MAC_1 is stmt macro -> OMG_MACRO (21) allowed
        self.assertEqual(m_stmt[21], "1")
        # In EK_EXPR(TY_U64): MAC_0 is (U64)->U64 -> OMG_MACRO (21) allowed
        self.assertEqual(m_u64[21], "1")
        # In EK_MAC_SLOT(TY_U64): MAC_0 (224) allowed, MAC_1 (225, VOID) forbidden
        self.assertEqual(m_mac_slot[224], "1")
        self.assertEqual(m_mac_slot[225], "0")
        # In EK_EXPR(TY_F32): no F32 macro registered -> OMG_MACRO (21) forbidden!
        self.assertEqual(m_f32[21], "0")

        # Build stream using MAC_1 (stmt: if true; v1 = add 10 20; end) and MAC_0 (expr: add mul v0 2 1) -> 61
        # OMG_BEGIN=1, OMG_FN_MAIN=4, OMG_LET=8, TY_U64=29, LIT_30=130,
        # OMG_MACRO=21, MAC_1=225, LIT_10=110, LIT_20=120,
        # OMG_RET=10, OP_CAST=67, TY_U64=29, OMG_MACRO=21, MAC_0=224, REF_0=160, OMG_END=7, OMG_END_UNIT=2
        mac_toks = [1, 4, 8, 29, 130, 21, 225, 110, 120, 10, 67, 29, 21, 224, 160, 7, 2]
        rc_e, out_e, err_e = self.run_cli("6", stdin_bytes=pack_tokens(mac_toks))
        self.assertEqual(rc_e, 0, f"macro expand failed rc={rc_e}")
        sid_exp, ntok_exp = out_e.decode("utf-8").strip().split()
        exp_toks = unpack_tokens(err_e)
        self.assertEqual(len(exp_toks), int(ntok_exp))

        # Must match hand-written un-macro'd equivalent program's SEMANTIC_ID!
        equiv_src = """om2
fn main to int
  a = ulong 30
  if true
    x = ulong 10
    y = ulong 20
    v = add x y
  end
  ret int add mul a 2 1
end
"""
        sid_eq, _ = self.parse_om(equiv_src)
        self.assertEqual(sid_exp, sid_eq[:16], "macro-expanded SEMANTIC_ID must match canonical source")

        # Decode macro stream directly (omg_decode_om auto-expands) -> compile -> run == 61!
        dec_om = self.decode_omg(exp_toks)
        bin_mac_prog = os.path.join(self.tmpdir, "mac_prog")
        compile_om(self.host_hdr + dec_om, bin_mac_prog)
        run_rc = subprocess.run([bin_mac_prog], capture_output=True).returncode
        self.assertEqual(run_rc, 61)

    def test_07_omgc_binary_corpus_and_compression_ab(self):
        # 1. Verify OMGC packed corpus binary format & semantic deduplication
        rc, corp_bytes, _ = self.run_cli("7")
        self.assertEqual(rc, 0)
        magic, ver, n_rec, total_toks = struct.unpack("<4sIII", corp_bytes[:16])
        self.assertEqual(magic, b"OMGC")
        self.assertEqual(ver, 1)
        self.assertEqual(n_rec, 2, "3 programs with 1 alpha-duplicate must deduplicate to 2 records")
        expected_len = 16 + n_rec * 16 + total_toks * 2
        self.assertEqual(len(corp_bytes), expected_len)

        dir_entries = []
        for r in range(n_rec):
            off = 16 + r * 16
            sid, toff, tlen = struct.unpack("<QII", corp_bytes[off : off + 16])
            dir_entries.append((sid, toff, tlen))
        self.assertEqual(sum(e[2] for e in dir_entries), total_toks)

        # 2. A/B Token Compression & Footprint Comparison: V=256 OMG vs raw .om & JSON AST
        sample_om = """om2
note Compute polynomial and checksum over loop
fn poly_step state ulong idx ulong to ulong
  scaled = mul state 3
  mixed = xor scaled add idx 7
  ret add mixed 1
end
fn main to int
  acc = ulong 5
  for k ulong 8
    acc = poly_step acc k
  end
  if gt acc 100
    ret int rem acc 199
  else
    ret int acc
  end
end
"""
        _, omg_toks = self.parse_om(sample_om)
        json_ast = json.dumps(
            {
                "kind": "unit",
                "fns": [
                    {
                        "name": "poly_step",
                        "params": [{"name": "state", "type": "u64"}, {"name": "idx", "type": "u64"}],
                        "ret": "u64",
                        "body": [
                            {"let": "scaled", "op": "mul", "args": [{"var": "state"}, {"lit": 3}]},
                            {"let": "mixed", "op": "xor", "args": [{"var": "scaled"}, {"op": "add", "args": [{"var": "idx"}, {"lit": 7}]}]},
                            {"ret": {"op": "add", "args": [{"var": "mixed"}, {"lit": 1}]}},
                        ],
                    },
                    {
                        "name": "main",
                        "params": [],
                        "ret": "i32",
                        "body": [
                            {"let": "acc", "type": "u64", "val": 5},
                            {"for": "k", "type": "u64", "count": 8, "body": [{"set": "acc", "call": "poly_step", "args": [{"var": "acc"}, {"var": "k"}]}]},
                            {"if": {"op": "gt", "args": [{"var": "acc"}, {"lit": 100}]}, "then": [{"ret": {"cast": "i32", "val": {"op": "rem", "args": [{"var": "acc"}, {"lit": 199}]}}}], "else": [{"ret": {"cast": "i32", "val": {"var": "acc"}}}]},
                        ],
                    },
                ],
            },
            separators=(",", ":"),
        )

        raw_om_byte_tokens = len(sample_om.encode("utf-8"))
        json_byte_tokens = len(json_ast.encode("utf-8"))
        omg_token_count = len(omg_toks)

        ratio_vs_om = raw_om_byte_tokens / omg_token_count
        ratio_vs_json = json_byte_tokens / omg_token_count
        self.assertGreaterEqual(ratio_vs_om, 3.0, f"expected >=3x compression vs .om text, got {ratio_vs_om:.2f}x")
        self.assertGreaterEqual(ratio_vs_json, 10.0, f"expected >=10x compression vs JSON AST, got {ratio_vs_json:.2f}x")
        print(
            f"\n[OMG A/B] V=256 vs V=4096 (16x logit reduction): "
            f"omg_toks={omg_token_count}, .om_bytes={raw_om_byte_tokens} ({ratio_vs_om:.2f}x), "
            f"json_bytes={json_byte_tokens} ({ratio_vs_json:.2f}x)"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
