#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Pillar 7 Verification: Closed-Form Specialist Solver (Ridge / Woodbury / 1-Step Gauss-Newton).

Verifies native/sp_solve.om:
  1. Compiles cleanly with math..spec for BOTH `target a64macos` and `gb10` (`om2`).
  2. Exact f64 linear-algebra oracle (< 1e-10 max abs error):
     - Primal Cholesky (D <= N): (H^T H + lambda I) W = H^T Y
     - Dual Woodbury (N < D): W = H^T (H H^T + lambda I)^{-1} Y (auto regime switch)
     - Primal vs Dual Woodbury cross-parity (< 1e-12)
     - 1-Step Gauss-Newton softmax CE refinement (`sp_gauss_newton_step`) for both D <= N and N < D
  3. End-to-end fcache / fcachem -> spforge -> OLLMSP01 .sp -> ollm spapp & ollm spchainapp:
     - Both Primal (F <= N) and Dual Woodbury (N < F) specialist forging regimes
     - Sleep-consolidation entrypoint (`spforge_consolidate_traces`)
     - ollm eval(SPEC.ck) == ollm speval(BASE.ck, OUT.sp) bit-identical, CE loss strictly reduced
     - Multi-link Merkle specialist chain (P1.sp -> P2.sp via ollm spchainapp)
  4. Speed vs iterative backprop (`trainfc`).
  5. Negative controls:
     - 1-bit tampered .sp -> ollm spapp exits rc=61
     - Truncated .sp -> ollm spapp exits rc=60
     - Mismatched base checkpoint -> ollm spapp / spforge exits rc=62
"""
import math
import os
import platform
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC = os.path.join(REPO, "native")

PASSES = []
FAILS = []


def check(cond, name, detail=""):
    if cond:
        PASSES.append(name)
        print(f"PASS {name}" + (f" ({detail})" if detail else ""))
    else:
        FAILS.append(name)
        print(f"FAIL {name}" + (f": {detail}" if detail else ""))


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def find_omc():
    omc = os.environ.get("OMC") or shutil.which("omc") or os.path.expanduser("~/.local/bin/omc")
    if not os.path.isfile(omc):
        raise RuntimeError(f"omc not found at {omc}")
    return omc


def run(args, cwd=None, env=None, ok=(0,)):
    p = subprocess.run(args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = p.stdout.decode("latin-1")
    err = p.stderr.decode("latin-1")
    if ok is not None and p.returncode not in ok:
        raise RuntimeError(f"{' '.join(args)} -> rc={p.returncode}\nstdout:\n{out}\nstderr:\n{err}")
    return p.returncode, out, err


def compile_om(om_text, bin_path):
    omc = find_omc()
    tmp = bin_path + ".tmp"
    p = subprocess.run([omc], input=om_text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError(f"omc failed rc={p.returncode}: {p.stderr.decode('latin-1')}")
    with open(tmp, "wb") as f:
        f.write(p.stdout)
    os.chmod(tmp, 0o755)
    os.replace(tmp, bin_path)
    if platform.system() == "Darwin" and shutil.which("codesign"):
        subprocess.run(["codesign", "-s", "-", "-f", bin_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_stock_ollm(out_dir):
    target = "macos" if platform.system() == "Darwin" else ("x86linux" if platform.machine().lower() in ("x86_64", "amd64") else "aarch64")
    bin_name = {"macos": "ollm-macos", "x86linux": "ollm-x86linux", "aarch64": "ollm"}[target]
    env = dict(os.environ, TARGET=target, OUT=out_dir)
    run(["sh", os.path.join(REPO, "scripts", "build.sh")], env=env)
    return os.path.join(out_dir, bin_name)


# ---------------- Pure-Python IEEE-754 f64 Linear Algebra Oracle ----------------

def py_cholesky_solve(A, B):
    """Exact f64 Cholesky LL^T solve A X = B where A is m x m SPD and B is m x k."""
    m = len(A)
    k = len(B[0]) if m > 0 else 0
    L = [[0.0] * m for _ in range(m)]
    for j in range(m):
        s = A[j][j] - sum(L[j][p] * L[j][p] for p in range(j))
        if s <= 0.0:
            raise ValueError(f"Non-SPD pivot at {j}: {s}")
        L[j][j] = math.sqrt(s)
        invd = 1.0 / L[j][j]
        for i in range(j + 1, m):
            t = A[i][j] - sum(L[i][p] * L[j][p] for p in range(j))
            L[i][j] = t * invd
    X = [[0.0] * k for _ in range(m)]
    for col in range(k):
        z = [0.0] * m
        for i in range(m):
            s = B[i][col] - sum(L[i][j] * z[j] for j in range(i))
            z[i] = s / L[i][i]
        for i in range(m - 1, -1, -1):
            s = z[i] - sum(L[j][i] * X[j][col] for j in range(i + 1, m))
            X[i][col] = s / L[i][i]
    return X


def py_ridge_oracle(H, Y, lam):
    """Exact f64 Ridge regression (H^T H + lam I)^{-1} H^T Y."""
    n = len(H)
    d = len(H[0])
    c = len(Y[0])
    if d <= n:
        G = [[0.0] * d for _ in range(d)]
        for a in range(d):
            for b in range(a + 1):
                v = sum(H[i][a] * H[i][b] for i in range(n))
                if a == b:
                    v += lam
                G[a][b] = G[b][a] = v
        R = [[sum(H[i][a] * Y[i][col] for i in range(n)) for col in range(c)] for a in range(d)]
        return py_cholesky_solve(G, R)
    K = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            v = sum(H[i][a] * H[j][a] for a in range(d))
            if i == j:
                v += lam
            K[i][j] = K[j][i] = v
    A = py_cholesky_solve(K, Y)
    return [[sum(H[i][a] * A[i][col] for i in range(n)) for col in range(c)] for a in range(d)]


def py_softmax_ce(H, W, targets):
    n = len(H)
    d = len(H[0])
    c = len(W[0])
    P = [[0.0] * c for _ in range(n)]
    loss = 0.0
    for i in range(n):
        logits = [sum(H[i][a] * W[a][col] for a in range(d)) for col in range(c)]
        mx = max(logits)
        exps = [math.exp(v - mx) for v in logits]
        sm = sum(exps)
        P[i] = [e / sm for e in exps]
        loss += mx + math.log(sm) - logits[targets[i]]
    return P, loss / n


def py_gauss_newton_oracle(H, W, targets, lam, eta):
    """Exact f64 1-step Gauss-Newton: (H^T diag(p*(1-p)) H + lam I) dW = H^T (y_onehot - p)."""
    n = len(H)
    d = len(H[0])
    c = len(W[0])
    P, l0 = py_softmax_ce(H, W, targets)
    dW = [[0.0] * c for _ in range(d)]
    for col in range(c):
        w = [P[i][col] * (1.0 - P[i][col]) for i in range(n)]
        r = [(1.0 if targets[i] == col else 0.0) - P[i][col] for i in range(n)]
        G = [[0.0] * d for _ in range(d)]
        for a in range(d):
            for b in range(a + 1):
                v = sum(w[i] * H[i][a] * H[i][b] for i in range(n))
                if a == b:
                    v += lam
                G[a][b] = G[b][a] = v
        rhs = [[sum(H[i][a] * r[i] for i in range(n))] for a in range(d)]
        sol = py_cholesky_solve(G, rhs)
        for a in range(d):
            dW[a][col] = sol[a][0]
    W_new = [[W[a][col] + eta * dW[a][col] for col in range(c)] for a in range(d)]
    _, l1 = py_softmax_ce(H, W_new, targets)
    return l0, l1, W_new


# ---------------- OM Test Harness Generator ----------------

HARNESS_OM = r"""
note Minimal test harness for sp_solve.om unit & integration tests (not part of production build)
fn th_rd_f64 off ulong to f64
ret bitcast f64 opw8 mul off 8
end

fn th_rd_u64 off ulong to ulong
ret opw8 mul off 8
end

fn th_load_f64 dst ulong src_off ulong n ulong to void
for i ulong n
ocsp add dst i th_rd_f64 add src_off i
end
end

fn th_load_tgt src_off ulong n ulong to void
for t ulong n
put osp_tgt t th_rd_u64 add src_off t
end
end

fn th_wr_f64 src ulong n ulong out_off ulong to void
for k ulong n
ocput8 mul add out_off k 8 bitcast ulong ocsg add src k
end
end

fn th_ridge_run mode ulong n ulong d ulong c ulong lam f64 to ulong
if eq mode 0
ret sp_ridge_solve n d c lam
end
if eq mode 1
put osp_st 9 0
ret sp_ridge_primal n d c lam
end
put osp_st 9 1
ret sp_ridge_woodbury n d c lam
end

fn th_cmd_ridge to void
c_ocs = ocsinit
sz = ofsz 2
if ne oprd 2 0 0 sz sz
exit 70
end
n = th_rd_u64 0
d = th_rd_u64 1
c = th_rd_u64 2
lam = th_rd_f64 3
put osp_st 0 n
put osp_st 1 d
put osp_st 2 c
put osp_st 3 0
yo = mul n d
put osp_st 4 yo
wo = add yo mul n c
put osp_st 5 wo
ao = add wo mul d c
put osp_st 6 ao
msz = d
if gt n d
msz = n
end
put osp_st 7 add ao mul msz msz
th_load_f64 0 5 yo
th_load_f64 yo add 5 yo mul n c
rc = th_ridge_run th_rd_u64 4 n d c lam
ocwopen 3
ocput8 0 get osp_st 9
ocput8 8 rc
dc = mul d c
th_wr_f64 wo dc 2
ocwput 0 mul add 2 dc 8
ocwend 3
end

fn th_gn_layout n ulong d ulong c ulong to void
wo = mul n d
msz = d
if gt n d
msz = n
end
ao = add wo mul d c
bo = add ao mul msz msz
put osp_st 3 0
put osp_st 5 wo
put osp_st 6 ao
put osp_st 7 bo
put osp_st 8 add bo msz
end

fn th_cmd_gn to void
c_ocs = ocsinit
sz = ofsz 2
if ne oprd 2 0 0 sz sz
exit 70
end
n = th_rd_u64 0
d = th_rd_u64 1
c = th_rd_u64 2
th_gn_layout n d c
nd = mul n d
dc = mul d c
th_load_f64 0 5 nd
th_load_f64 nd add 5 nd dc
th_load_tgt add add 5 nd dc n
rc = sp_gauss_newton_step n d c th_rd_f64 3 th_rd_f64 4
ocwopen 3
ocput8 0 get osp_st 9
ocput8 8 rc
ocput8 16 bitcast ulong get osp_fl 2
ocput8 24 bitcast ulong get osp_fl 3
th_wr_f64 nd dc 4
ocwput 0 mul add 4 dc 8
ocwend 3
end

fn th_cmd_sleep to void
c_ocs = ocsinit
ockloop 2
sz = ofsz 3
if ne oprd 3 0 0 sz sz
exit 70
end
n = th_rd_u64 0
dd = get og 2
nd = mul n dd
th_load_f64 0 3 nd
th_load_tgt add 3 nd n
rc = spforge_consolidate_traces 4 n th_rd_f64 1 th_rd_u64 2
if ne rc 0
exit int rc
end
oputc 83
oputc 80
oputc 67
oputc 79
oputc 78
oputc 83
oputc 79
oputc 76
oputc 32
oputu get osp_st 9
oputc 32
oputl rmic get osp_fl 2
oputc 32
oputl rmic get osp_fl 3
oputc 10
oflush
end

fn th_cmd_ocs to void
c_ocs = ocsinit
mode = oargu 2
if eq mode 1
x = ocsg c_ocs
exit 0
end
if eq mode 2
ocsp c_ocs 1.0f64
exit 0
end
if ne ocsg 100000 0.0f64
exit 71
end
ocsp 0 1.5f64
ocsp 100000 42.25f64
if ne ocsg 0 1.5f64
exit 72
end
if ne ocsg 100000 42.25f64
exit 73
end
end

fn ocmd to int
w = ocmdw
if eq w __SPFORGE__
cmdspforge
ret 0
end
if eq w __TRIDGE__
th_cmd_ridge
ret 0
end
if eq w __TGN__
th_cmd_gn
ret 0
end
if eq w __TSLEEP__
th_cmd_sleep
ret 0
end
if eq w __TOCS__
th_cmd_ocs
ret 0
end
ret ocmd_orig
end
"""


def u64_pack_word(word):
    v = 0
    for i, ch in enumerate(word.encode("ascii")):
        v |= ch << (8 * i)
    return v


def build_solver_harness(out_dir):
    harness_om = (
        HARNESS_OM
        .replace("__SPFORGE__", str(u64_pack_word("spforge")))
        .replace("__TRIDGE__", str(u64_pack_word("tridge")))
        .replace("__TGN__", str(u64_pack_word("tgn")))
        .replace("__TSLEEP__", str(u64_pack_word("tsleep")))
        .replace("__TOCS__", str(u64_pack_word("tocs")))
    )
    is_mac = platform.system() == "Darwin"
    is_x86 = platform.system() == "Linux" and platform.machine().lower() in ("x86_64", "amd64")
    plat = "mio.om" if is_mac else "lio.om"
    rev_unit = "norev.om" if is_x86 else "rev.om"
    units = ["math.om", "data.om", "lpool.om", "model.om", "adam.om", "ckpt.om", "bpe.om", "spec.om", rev_unit, "arboris.om", "sp_solve.om", "spiral.om", "omg.om", "pillars_cli.om"]
    cli_txt = read_text(os.path.join(SRC, "cli.om")).replace("fn ocmd to int", "fn ocmd_orig to int", 1)
    src_text = "target a64macos\nom2\n" if is_mac else ("target x86linux\nom2\n" if is_x86 else "om2\n")
    src_text += "\n".join(read_text(os.path.join(SRC, u)) for u in units)
    src_text += "\n" + cli_txt + "\n" + harness_om + "\n" + read_text(os.path.join(SRC, plat))
    src_text += "\n" + read_text(os.path.join(SRC, "noserve.om")) + "\n" + read_text(os.path.join(SRC, "nogpu.om"))
    harness_bin = os.path.join(out_dir, "sp_harness")
    compile_om(src_text, harness_bin)
    return harness_bin


def test_cross_target_compile():
    """Verify sp_solve.om compiles cleanly with math..spec for host target and gb10 (om2)."""
    with tempfile.TemporaryDirectory() as td:
        units = ["math.om", "data.om", "lpool.om", "model.om", "adam.om", "ckpt.om", "bpe.om", "spec.om", "rev.om", "arboris.om", "sp_solve.om", "spiral.om", "omg.om", "pillars_cli.om", "cli.om"]
        if platform.system() == "Darwin":
            mac_text = "target a64macos\nom2\n" + "\n".join(read_text(os.path.join(SRC, u)) for u in units + ["mio.om", "noserve.om", "nogpu.om"])
            mac_bin = os.path.join(td, "ollm-macos")
            compile_om(mac_text, mac_bin)
            check(os.path.getsize(mac_bin) > 0, "compile_target_a64macos", f"{os.path.getsize(mac_bin)} B")

        gb_units = ["net/http.om", "net/json.om", "net/jsonw.om"] + units + [
            "lio.om", "serve.om",
            "gpu/context.om", "gpu/vector.om", "gpu/launch.om", "gpu/batch.om",
            "gpu/gk.om", "gpu/gr.om", "gpu/gb.om", "gpu/gg.om",
            "gpu/gt.om", "gpu/gm.om", "gpu/ring.om", "gpu/ejit.om",
        ]
        gb_text = "om2\n" + "\n".join(read_text(os.path.join(SRC, u)) for u in gb_units)
        gb_bin = os.path.join(td, "ollm-gb10")
        compile_om(gb_text, gb_bin)
        check(os.path.getsize(gb_bin) > 0, "compile_target_gb10_om2", f"{os.path.getsize(gb_bin)} B")


def run_ridge_case(harness, td, tag, H, Y, lam, mode=0):
    n, d, c = len(H), len(H[0]), len(Y[0])
    inp = os.path.join(td, f"{tag}_in.bin")
    outp = os.path.join(td, f"{tag}_out.bin")
    with open(inp, "wb") as f:
        f.write(struct.pack("<QQQdQ", n, d, c, lam, mode))
        for row in H:
            f.write(struct.pack(f"<{d}d", *row))
        for row in Y:
            f.write(struct.pack(f"<{c}d", *row))
    run([harness, "tridge", inp, outp])
    data = read_bytes(outp)
    regime, rc = struct.unpack("<QQ", data[:16])
    flat = struct.unpack(f"<{d * c}d", data[16:])
    W = [list(flat[a * c:(a + 1) * c]) for a in range(d)]
    return regime, rc, W


def run_gn_case(harness, td, tag, H, W, targets, lam, eta):
    n, d, c = len(H), len(H[0]), len(W[0])
    inp = os.path.join(td, f"{tag}_in.bin")
    outp = os.path.join(td, f"{tag}_out.bin")
    with open(inp, "wb") as f:
        f.write(struct.pack("<QQQdd", n, d, c, lam, eta))
        for row in H:
            f.write(struct.pack(f"<{d}d", *row))
        for row in W:
            f.write(struct.pack(f"<{c}d", *row))
        f.write(struct.pack(f"<{n}Q", *targets))
    run([harness, "tgn", inp, outp])
    data = read_bytes(outp)
    regime, rc, l0, l1 = struct.unpack("<QQdd", data[:32])
    flat = struct.unpack(f"<{d * c}d", data[32:])
    W_new = [list(flat[a * c:(a + 1) * c]) for a in range(d)]
    return regime, rc, l0, l1, W_new


def max_abs_diff(A, B):
    return max(abs(A[i][j] - B[i][j]) for i in range(len(A)) for j in range(len(A[0])))


def test_linear_algebra_oracle(harness):
    rng = random.Random(20261125)
    with tempfile.TemporaryDirectory() as td:
        # 1. Primal Ridge (D <= N): N=24, D=8, C=4
        H1 = [[rng.gauss(0.0, 1.0) for _ in range(8)] for _ in range(24)]
        Y1 = [[rng.gauss(0.0, 1.0) for _ in range(4)] for _ in range(24)]
        reg1, rc1, W1 = run_ridge_case(harness, td, "primal", H1, Y1, 0.25, mode=0)
        W1_ref = py_ridge_oracle(H1, Y1, 0.25)
        err1 = max_abs_diff(W1, W1_ref)
        check(rc1 == 0 and reg1 == 0 and err1 < 1e-10, "ridge_primal_d_le_n", f"regime={reg1}, max_err={err1:.3e}")

        # 2. Dual Woodbury (N < D): N=10, D=28, C=5
        H2 = [[rng.gauss(0.0, 1.0) for _ in range(28)] for _ in range(10)]
        Y2 = [[rng.gauss(0.0, 1.0) for _ in range(5)] for _ in range(10)]
        reg2, rc2, W2 = run_ridge_case(harness, td, "woodbury", H2, Y2, 0.15, mode=0)
        W2_ref = py_ridge_oracle(H2, Y2, 0.15)
        err2 = max_abs_diff(W2, W2_ref)
        check(rc2 == 0 and reg2 == 1 and err2 < 1e-10, "ridge_woodbury_n_lt_d", f"regime={reg2}, max_err={err2:.3e}")

        # 3. Primal vs Dual Woodbury cross-parity on same N < D problem
        _, rc2p, W2p = run_ridge_case(harness, td, "wood_vs_pri", H2, Y2, 0.15, mode=1)
        err_pw = max_abs_diff(W2, W2p)
        check(rc2p == 0 and err_pw < 1e-12, "ridge_primal_woodbury_parity", f"max_err={err_pw:.3e}")

        # 4. 1-Step Gauss-Newton (Primal D <= N): N=20, D=6, C=5
        H3 = [[rng.gauss(0.0, 0.8) for _ in range(6)] for _ in range(20)]
        W3 = [[rng.gauss(0.0, 0.3) for _ in range(5)] for _ in range(6)]
        tgt3 = [rng.randrange(5) for _ in range(20)]
        g_reg3, g_rc3, l0_3, l1_3, W3_new = run_gn_case(harness, td, "gn_primal", H3, W3, tgt3, 0.1, 1.0)
        l0_3_ref, l1_3_ref, W3_ref = py_gauss_newton_oracle(H3, W3, tgt3, 0.1, 1.0)
        err3 = max_abs_diff(W3_new, W3_ref)
        check(
            g_rc3 == 0 and g_reg3 == 0 and err3 < 1e-10 and abs(l0_3 - l0_3_ref) < 1e-12 and abs(l1_3 - l1_3_ref) < 1e-12 and l1_3 < l0_3,
            "gauss_newton_primal",
            f"regime={g_reg3}, max_err={err3:.3e}, ce={l0_3:.6f}->{l1_3:.6f}",
        )

        # 5. 1-Step Gauss-Newton (Dual Weighted Woodbury N < D): N=8, D=20, C=4
        H4 = [[rng.gauss(0.0, 0.7) for _ in range(20)] for _ in range(8)]
        W4 = [[rng.gauss(0.0, 0.2) for _ in range(4)] for _ in range(20)]
        tgt4 = [rng.randrange(4) for _ in range(8)]
        g_reg4, g_rc4, l0_4, l1_4, W4_new = run_gn_case(harness, td, "gn_wood", H4, W4, tgt4, 0.2, 1.0)
        l0_4_ref, l1_4_ref, W4_ref = py_gauss_newton_oracle(H4, W4, tgt4, 0.2, 1.0)
        err4 = max_abs_diff(W4_new, W4_ref)
        check(
            g_rc4 == 0 and g_reg4 == 1 and err4 < 1e-10 and abs(l0_4 - l0_4_ref) < 1e-12 and abs(l1_4 - l1_4_ref) < 1e-12 and l1_4 < l0_4,
            "gauss_newton_woodbury",
            f"regime={g_reg4}, max_err={err4:.3e}, ce={l0_4:.6f}->{l1_4:.6f}",
        )


def parse_eval(out):
    m = re.search(r"E\s+(\d+)\s+(\d+)\s+(-?\d+)\s+(-?\d+)", out)
    if not m:
        raise RuntimeError(f"Cannot parse eval output:\n{out}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))


def parse_spforge(out):
    m = re.search(r"SPFORGE\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+)\s+(-?\d+)\s+(\d+)", out)
    if not m:
        raise RuntimeError(f"Cannot parse SPFORGE output:\n{out}")
    return tuple(int(m.group(i)) for i in range(1, 8))


def write_pattern_tokens_and_mask(tok_path, mask_path, n_bytes, T, mask_every=4):
    """Deterministic pattern tokens so a frozen base has meaningful learnable structure."""
    pattern = bytes([(32 + (i % 13) * 3) & 0xFF for i in range(n_bytes)])
    with open(tok_path, "wb") as f:
        f.write(pattern)
    # Mask: 1 for target prediction, 0 periodically (masked out by fcachem!)
    mask = bytes([0 if (i % mask_every == 0) else 1 for i in range(n_bytes)])
    with open(mask_path, "wb") as f:
        f.write(mask)
    return pattern, mask


def test_end_to_end_spforge_and_negative_controls(ollm, harness):
    with tempfile.TemporaryDirectory() as td:
        # Model: V=256, T=8, D=16, H=2, F=32, L=1, B=1
        base_ck = os.path.join(td, "base.ck")
        run([ollm, "init", base_ck, "256", "8", "16", "2", "32", "1", "1", "77", "20", "5000"])

        toks = os.path.join(td, "train.tok")
        mask = os.path.join(td, "train.msk")
        # 12 windows * 8 = 96 + 1 = 97 bytes; with mask_every=4 -> 72 unmasked tokens (> F=32 -> Primal!)
        write_pattern_tokens_and_mask(toks, mask, 97, 8, mask_every=4)

        # Warm up both binaries once (macOS codesign first-launch cache) before timing
        run([harness], ok=None)
        run([ollm], ok=None)

        # ---- Regime A: fcachem -> spforge Primal (F=32 <= N=72) ----
        hc_primal = os.path.join(td, "primal.hc")
        run([ollm, "fcachem", base_ck, toks, mask, "12", hc_primal])

        sp_primal = os.path.join(td, "primal.sp")
        t0 = time.perf_counter()
        _, out_pf, _ = run([harness, "spforge", base_ck, hc_primal, sp_primal, "100000", "1"])
        forge_s = time.perf_counter() - t0
        bl, sl, n_act, reg_p, l0_m, l1_m, nspec = parse_spforge(out_pf)
        check(
            bl == 1 and sl == 2 and n_act == 72 and reg_p == 0 and l1_m < l0_m,
            "spforge_fcachem_primal",
            f"N={n_act}, regime={reg_p}, loss={l0_m}->{l1_m} unats, {forge_s*1000:.2f}ms",
        )

        # Apply via stock ollm spapp and verify eval == speval bit-for-bit & loss reduction
        spec_ck = os.path.join(td, "spec_primal.ck")
        run([ollm, "spapp", base_ck, sp_primal, spec_ck])
        _, ev_base, _ = run([ollm, "eval", base_ck, toks, "12"])
        _, ev_spec, _ = run([ollm, "eval", spec_ck, toks, "12"])
        _, ev_sp, _ = run([ollm, "speval", base_ck, sp_primal, toks, "12"])
        b_loss = parse_eval(ev_base)[2]
        s_loss = parse_eval(ev_spec)[2]
        sp_loss = parse_eval(ev_sp)[2]
        check(
            s_loss == sp_loss and s_loss < b_loss,
            "ollm_spapp_eval_speval_parity",
            f"base={b_loss} -> spec={s_loss} (speval={sp_loss})",
        )

        # ---- Speed vs iterative backprop (full-batch B=12 trainfc, 30 steps) ----
        grow_ck = os.path.join(td, "grow_fc.ck")
        run([ollm, "growpro", base_ck, grow_ck, "2", "30", "50000"])
        t1 = time.perf_counter()
        run([ollm, "trainfc", grow_ck, hc_primal, "30", "30", "1", "12"])
        trainfc_s = time.perf_counter() - t1
        _, ev_fc, _ = run([ollm, "eval", grow_ck, toks, "12"])
        fc_loss = parse_eval(ev_fc)[2]
        check(
            forge_s < trainfc_s and s_loss < b_loss and fc_loss < b_loss,
            "spforge_vs_backprop_speed_and_loss",
            f"spforge={forge_s*1000:.2f}ms (loss={s_loss}) vs trainfc_B12_30steps={trainfc_s*1000:.2f}ms (loss={fc_loss})",
        )

        # ---- Regime B: fcachem -> spforge Dual Woodbury (N=18 < F=32) ----
        # 3 windows * 8 = 24 tokens -> mask_every=4 leaves 18 active tokens (< F=32 -> Woodbury!)
        hc_wood = os.path.join(td, "wood.hc")
        run([ollm, "fcachem", base_ck, toks, mask, "3", hc_wood])
        sp_wood = os.path.join(td, "wood.sp")
        _, out_pw, _ = run([harness, "spforge", base_ck, hc_wood, sp_wood, "100000", "1"])
        _, _, n_w, reg_w, l0_w, l1_w, _ = parse_spforge(out_pw)
        spec_wood = os.path.join(td, "spec_wood.ck")
        run([ollm, "spapp", base_ck, sp_wood, spec_wood])
        check(
            n_w == 18 and reg_w == 1 and l1_w < l0_w,
            "spforge_fcachem_woodbury",
            f"N={n_w}, regime={reg_w}, loss={l0_w}->{l1_w} unats",
        )

        # ---- Multi-link Merkle specialist chain: P1.sp -> P2.sp via ollm spchainapp ----
        hc_link2 = os.path.join(td, "link2.hc")
        run([ollm, "fcache", spec_ck, toks, "8", hc_link2])
        sp_link2 = os.path.join(td, "link2.sp")
        _, out_l2, _ = run([harness, "spforge", spec_ck, hc_link2, sp_link2, "100000", "1"])
        bl2, sl2, _, _, l2_0, l2_1, _ = parse_spforge(out_l2)
        chain_ck = os.path.join(td, "chain.ck")
        run([ollm, "spchainapp", base_ck, chain_ck, "2", sp_primal, sp_link2])
        _, ev_ch, _ = run([ollm, "eval", chain_ck, toks, "8"])
        ch_loss = parse_eval(ev_ch)[2]
        check(
            bl2 == 2 and sl2 == 3 and l2_1 < l2_0 and ch_loss < s_loss,
            "spforge_spchainapp_multilink",
            f"L1->L2={s_loss} -> L2->L3={ch_loss}",
        )

        # ---- Sleep-consolidation entrypoint (`spforge_consolidate_traces`) ----
        # Extract H_raw (N=24) directly from wood.hc (3 windows * 8) and feed to tsleep
        raw = read_bytes(hc_wood)
        T, D = 8, 16
        rec_bytes = (T + T * D) * 8
        H_sleep = []
        tgt_sleep = []
        for w in range(3):
            rec = raw[48 + w * rec_bytes: 48 + (w + 1) * rec_bytes]
            ys = struct.unpack(f"<{T}Q", rec[:T * 8])
            hs = struct.unpack(f"<{T * D}d", rec[T * 8:])
            for t in range(T):
                if ys[t] < 65536:
                    tgt_sleep.append(ys[t])
                    H_sleep.extend(hs[t * D:(t + 1) * D])
        sleep_in = os.path.join(td, "sleep.bin")
        with open(sleep_in, "wb") as f:
            f.write(struct.pack("<QdQ", len(tgt_sleep), 0.1, 1))
            f.write(struct.pack(f"<{len(H_sleep)}d", *H_sleep))
            f.write(struct.pack(f"<{len(tgt_sleep)}Q", *tgt_sleep))
        sp_sleep = os.path.join(td, "sleep.sp")
        _, out_sl, _ = run([harness, "tsleep", base_ck, sleep_in, sp_sleep])
        check(
            out_sl.startswith("SPCONSOL 1 ") and read_bytes(sp_sleep) == read_bytes(sp_wood),
            "spforge_consolidate_traces_bit_identical",
            out_sl.strip(),
        )

        # ---- Negative controls on forged OLLMSP01 .sp ----
        # 1. Flip 1 bit in payload -> ollm spapp MUST exit 61
        bad_sp = os.path.join(td, "tampered.sp")
        sp_bytes = bytearray(read_bytes(sp_primal))
        sp_bytes[-1] ^= 0x01
        with open(bad_sp, "wb") as f:
            f.write(sp_bytes)
        rc_tamper, _, _ = run([ollm, "spapp", base_ck, bad_sp, os.path.join(td, "nope.ck")], ok=None)
        check(rc_tamper == 61, "neg_control_tampered_sp_rc61", f"rc={rc_tamper}")

        # 2. Truncated .sp -> ollm spapp exits 60
        trunc_sp = os.path.join(td, "trunc.sp")
        with open(trunc_sp, "wb") as f:
            f.write(sp_bytes[:32])
        rc_trunc, _, _ = run([ollm, "spapp", base_ck, trunc_sp, os.path.join(td, "nope2.ck")], ok=None)
        check(rc_trunc == 60, "neg_control_truncated_sp_rc60", f"rc={rc_trunc}")

        # 3. Mismatched base checkpoint -> ollm spapp exits 62 & spforge exits 62
        other_ck = os.path.join(td, "other.ck")
        run([ollm, "init", other_ck, "256", "8", "16", "2", "32", "1", "1", "99", "20", "5000"])
        rc_mis_app, _, _ = run([ollm, "spapp", other_ck, sp_primal, os.path.join(td, "nope3.ck")], ok=None)
        rc_mis_fge, _, _ = run([harness, "spforge", other_ck, hc_primal, os.path.join(td, "nope.sp")], ok=None)
        check(rc_mis_app == 62 and rc_mis_fge == 62, "neg_control_mismatch_base_rc62", f"spapp={rc_mis_app}, spforge={rc_mis_fge}")


def test_cold_store_growth_and_bounds(harness):
    """Verify cold store works when og[8]==0, grows past initial floor preserving data, and traps at OCSN."""
    rc0, _, _ = run([harness, "tocs", "0"], ok=None)
    check(rc0 == 0, "cold_store_zero_model_and_dynamic_grow", f"rc={rc0}")
    rc1, _, _ = run([harness, "tocs", "1"], ok=None)
    check(rc1 == 103, "cold_store_oob_read_traps_103", f"rc={rc1}")
    rc2, _, _ = run([harness, "tocs", "2"], ok=None)
    check(rc2 == 103, "cold_store_oob_write_traps_103", f"rc={rc2}")


def main():
    test_cross_target_compile()
    with tempfile.TemporaryDirectory() as td:
        ollm = build_stock_ollm(td)
        harness = build_solver_harness(td)
        test_cold_store_growth_and_bounds(harness)
        test_linear_algebra_oracle(harness)
        test_end_to_end_spforge_and_negative_controls(ollm, harness)
    print(f"\nSUMMARY: {len(PASSES)} passed, {len(FAILS)} failed")
    if FAILS:
        sys.exit(1)


if __name__ == "__main__":
    main()
