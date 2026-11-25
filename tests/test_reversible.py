#!/usr/bin/env python3
"""Pillar 2 (Reversible Residual Blocks) pure-Python-stdlib f64 oracle & test suite.

Verifies `native/rev.om` across:
1. Compilation with `native/{math,data,lpool,model,adam,ckpt,bpe,mio/lio,nogpu}.om`
   for both `target a64macos` (OM-1 & OM-2) and `gb10` (Linux AArch64 OM-2).
2. Built-in `rev_selftest` and `mode0` (model.om pool integration).
3. Algebraic inverse reconstruction error < 1e-12 across L layers & looped cores.
4. Analytical vs central finite-difference gradient check < 1e-8 across all tensor kinds.
5. Zero-activation-buffer (`rev_act_bytes_used == 0`) training equivalence vs
   cached-activation Python `f64` reference across multi-step Adam / AdamW runs.
6. `OLLMCK04` checkpoint & `arch_flags` (`FL_REV = 128` + RoPE/SwiGLU/GQA/QKNorm/
   Untied/GELU/GeGLU/LoopedCore) compatibility.

Run with:
    python3 -W error::ResourceWarning tests/test_reversible.py
"""

import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile

M64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
SQRT_2_OVER_PI = 0.79788456080286536
MAGIC_CK4 = b"OLLMCK04"
CKHDR = 208

FL_ROPE = 1
FL_SWIGLU = 2
FL_UNTIED = 4
FL_QKNORM = 8
FL_ADAMW = 16
FL_GELU = 32
FL_GEGLU = 64
FL_REV = 128
FL_Q32 = 256
Q32_SCALE = 1073741824.0


def find_repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def find_omc():
    cands = [
        os.environ.get("OMC", ""),
        shutil.which("omc") or "",
        os.path.expanduser("~/.local/bin/omc"),
    ]
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    raise RuntimeError("omc compiler not found")


def splitmix64(x):
    z = (x + GOLDEN) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def fnv1a64(data):
    h = FNV_OFFSET
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & M64
    return h


def bits2f(u):
    return struct.unpack("<d", struct.pack("<Q", u))[0]


def f2bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def gelu(x):
    c = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)
    return 0.5 * x * (1.0 + math.tanh(c))


def gelu_d(x):
    x2 = x * x
    c = SQRT_2_OVER_PI * (x + 0.044715 * x2 * x)
    th = math.tanh(c)
    sech2 = 1.0 - th * th
    dc = SQRT_2_OVER_PI * (1.0 + 0.134145 * x2)
    return 0.5 * (1.0 + th) + 0.5 * x * sech2 * dc


# ---------------------------------------------------------------- geometry & CK04

def phys_layer(cfg, l):
    np_ = cfg.get("NP", 0)
    if np_ <= 1:
        return l
    lp, lc = cfg["LP"], cfg["LC"]
    if l < lp:
        return l
    j = l - lp
    if j < lc * np_:
        return lp + (j % lc)
    return l - lc * (np_ - 1)


def enrich_cfg(cfg):
    out = dict(cfg)
    D, H, H_kv = cfg["D"], cfg["H"], cfg["H_kv"]
    assert D % H == 0 and H % H_kv == 0
    E = D // H
    fl = cfg["arch_flags"]
    np_ = cfg.get("NP", 0)
    lp, lc = cfg.get("LP", 0), cfg.get("LC", 0)
    L = cfg["L"]
    LV = L + lc * (np_ - 1) if np_ > 1 else L
    out["E"] = E
    out["KVD"] = H_kv * E
    out["LV"] = LV
    out["has_rope"] = bool(fl & FL_ROPE)
    out["is_gated"] = bool(fl & (FL_SWIGLU | FL_GEGLU))
    out["has_untied"] = bool(fl & FL_UNTIED)
    out["has_qknorm"] = bool(fl & FL_QKNORM)
    out["is_adamw"] = bool(fl & FL_ADAMW)
    out["has_gelu"] = bool(fl & FL_GELU)
    out["has_swiglu"] = bool(fl & FL_SWIGLU)
    out["has_geglu"] = bool(fl & FL_GEGLU)
    out["has_rev"] = bool(fl & FL_REV)
    return out


def tensor_specs(g):
    V, T, D, F, L = g["V"], g["T"], g["D"], g["F"], g["L"]
    E, KVD = g["E"], g["KVD"]
    specs = [("wte", V, D)]
    if not g["has_rope"]:
        specs.append(("wpe", T, D))
    for l in range(L):
        specs.append((f"g1.{l}", 1, D))
        specs.append((f"wq.{l}", D, D))
        specs.append((f"wk.{l}", D, KVD))
        specs.append((f"wv.{l}", D, KVD))
        specs.append((f"wo.{l}", D, D))
        if g["has_qknorm"]:
            specs.append((f"qn.{l}", 1, E))
            specs.append((f"kn.{l}", 1, E))
        specs.append((f"g2.{l}", 1, D))
        specs.append((f"w1.{l}", D, F))
        if g["is_gated"]:
            specs.append((f"wup.{l}", D, F))
        specs.append((f"w2.{l}", F, D))
    specs.append(("gf", 1, D))
    if g["has_untied"]:
        specs.append(("lmh", V, D))
    out, off = [], 0
    for name, r, c in specs:
        out.append((name, off, r, c))
        off += r * c
    return out, off


def is_gain_name(name):
    return name.split(".")[0] in ("g1", "g2", "qn", "kn", "gf")


def unpack_params(g, flat):
    specs, _ = tensor_specs(g)
    out = {}
    for name, off, r, c in specs:
        if r == 1 and is_gain_name(name):
            out[name] = list(flat[off:off + c])
        else:
            out[name] = [list(flat[off + i * c:off + (i + 1) * c]) for i in range(r)]
    return out


def pack_params(g, tdict):
    specs, _ = tensor_specs(g)
    flat = []
    for name, _, r, _ in specs:
        v = tdict[name]
        if r == 1 and is_gain_name(name):
            flat.extend(v)
        else:
            for row in v:
                flat.extend(row)
    return flat


def zeros_like(g):
    specs, _ = tensor_specs(g)
    out = {}
    for name, _, r, c in specs:
        out[name] = [0.0] * c if (r == 1 and is_gain_name(name)) else [[0.0] * c for _ in range(r)]
    return out


def read_ck4(path):
    with open(path, "rb") as f:
        raw = f.read()
    assert len(raw) >= CKHDR
    assert raw[:8] == MAGIC_CK4, f"bad magic {raw[:8]!r}"
    V, T, D, H, H_kv, F, L, B = struct.unpack("<8Q", raw[8:72])
    seed, step, total, lr_bits, P = struct.unpack("<5Q", raw[72:112])
    fl, rope_milli, eps_bits, wd_bits, warmup = struct.unpack("<5Q", raw[112:152])
    LP, LC, NP = struct.unpack("<3Q", raw[152:176])
    (fnv_got,) = struct.unpack("<Q", raw[200:208])
    assert len(raw) == CKHDR + 24 * P
    body = raw[CKHDR:]
    assert fnv1a64(body) == fnv_got, "CK04 FNV-1a mismatch"
    pays = [list(struct.unpack(f"<{P}d", body[8 * P * k:8 * P * (k + 1)])) for k in range(3)]
    return dict(
        V=V, T=T, D=D, H=H, H_kv=H_kv, F=F, L=L, B=B,
        seed=seed, step=step, total=total, lr=bits2f(lr_bits), P=P,
        arch_flags=fl, rope_theta_milli=rope_milli, rms_eps=bits2f(eps_bits),
        wd=bits2f(wd_bits), warmup=warmup, LP=LP, LC=LC, NP=NP,
        params=pays[0], m=pays[1], v=pays[2],
    )


# ---------------------------------------------------------------- Python f64 Reversible Oracle

def rms_fwd_mat(X, g, eps):
    T, D = len(X), len(g)
    Y, dens = [], []
    for t in range(T):
        row = X[t]
        s = math.sqrt(sum(v * v for v in row) / D + eps)
        dens.append(s)
        Y.append([row[j] / s * g[j] for j in range(D)])
    return Y, dens


def rms_bwd_mat(X, g, dens, dY, dg):
    T, D = len(X), len(g)
    dX = []
    for t in range(T):
        row, dy, s = X[t], dY[t], dens[t]
        for j in range(D):
            dg[j] += dy[j] * row[j] / s
        dot = sum(dy[j] * g[j] * row[j] for j in range(D))
        k = dot / (D * s * s)
        dX.append([(dy[j] * g[j] - row[j] * k) / s for j in range(D)])
    return dX


def mm_fwd(X, W):
    T, K, N = len(X), len(W), len(W[0])
    return [[sum(X[t][p] * W[p][j] for p in range(K)) for j in range(N)] for t in range(T)]


def mm_dx(dY, W):
    T, K, N = len(dY), len(W), len(W[0])
    return [[sum(dY[t][j] * W[p][j] for j in range(N)) for p in range(K)] for t in range(T)]


def mm_dw(X, dY, dW):
    T, K, N = len(X), len(dW), len(dW[0])
    for t in range(T):
        xt, dyt = X[t], dY[t]
        for p in range(K):
            xv = xt[p]
            if xv != 0.0:
                row = dW[p]
                for j in range(N):
                    row[j] += xv * dyt[j]


def build_rope_tables(g):
    T, E = g["T"], g["E"]
    he = E // 2
    th = g["rope_theta_milli"] / 1000.0
    if th <= 0.0:
        th = 10000.0
    lth = math.log(th)
    cos_t = [[0.0] * he for _ in range(T)]
    sin_t = [[0.0] * he for _ in range(T)]
    for i in range(he):
        freq = math.exp(-lth * (2.0 * i) / E)
        for pos in range(T):
            ang = pos * freq
            cos_t[pos][i] = math.cos(ang)
            sin_t[pos][i] = math.sin(ang)
    return cos_t, sin_t


def apply_rope_vec(v, pos, cos_t, sin_t, inv=False):
    he = len(cos_t[0])
    out = list(v)
    c_row, s_row = cos_t[pos], sin_t[pos]
    for i in range(he):
        x0, x1 = v[i], v[i + he]
        c, s = c_row[i], s_row[i]
        if inv:
            out[i] = x0 * c + x1 * s
            out[i + he] = x1 * c - x0 * s
        else:
            out[i] = x0 * c - x1 * s
            out[i + he] = x0 * s + x1 * c
    return out


def F_fwd(g, P, pl, x2, rope_tab):
    T, D, H, H_kv, E, KVD = g["T"], g["D"], g["H"], g["H_kv"], g["E"], g["KVD"]
    eps = g["rms_eps"]
    g1 = P[f"g1.{pl}"]
    a, s1 = rms_fwd_mat(x2, g1, eps)
    q = mm_fwd(a, P[f"wq.{pl}"])
    k = mm_fwd(a, P[f"wk.{pl}"])
    u = mm_fwd(a, P[f"wv.{pl}"])
    qpre, kpre, qns, kns = None, None, None, None
    if g["has_qknorm"]:
        qpre = [list(r) for r in q]
        kpre = [list(r) for r in k]
        qn_w, kn_w = P[f"qn.{pl}"], P[f"kn.{pl}"]
        qns = [[0.0] * H for _ in range(T)]
        kns = [[0.0] * H_kv for _ in range(T)]
        for t in range(T):
            for hi in range(H):
                sl = qpre[t][hi * E:(hi + 1) * E]
                s = math.sqrt(sum(v * v for v in sl) / E + eps)
                qns[t][hi] = s
                q[t][hi * E:(hi + 1) * E] = [sl[j] / s * qn_w[j] for j in range(E)]
            for kh in range(H_kv):
                sl = kpre[t][kh * E:(kh + 1) * E]
                s = math.sqrt(sum(v * v for v in sl) / E + eps)
                kns[t][kh] = s
                k[t][kh * E:(kh + 1) * E] = [sl[j] / s * kn_w[j] for j in range(E)]
    if g["has_rope"]:
        cos_t, sin_t = rope_tab
        for t in range(T):
            for hi in range(H):
                q[t][hi * E:(hi + 1) * E] = apply_rope_vec(q[t][hi * E:(hi + 1) * E], t, cos_t, sin_t, False)
            for kh in range(H_kv):
                k[t][kh * E:(kh + 1) * E] = apply_rope_vec(k[t][kh * E:(kh + 1) * E], t, cos_t, sin_t, False)
    sE = math.sqrt(E)
    o = [[0.0] * D for _ in range(T)]
    pa = []
    grp = H // H_kv
    for hi in range(H):
        kh = hi // grp
        qh_off, kh_off = hi * E, kh * E
        p_h = []
        for t in range(T):
            qt = q[t][qh_off:qh_off + E]
            sc = [sum(qt[c] * k[j][kh_off + c] for c in range(E)) / sE for j in range(t + 1)]
            mx = max(sc)
            ew = [math.exp(v - mx) for v in sc]
            sm = sum(ew)
            pr = [v / sm for v in ew]
            p_h.append(pr)
            for j in range(t + 1):
                pj = pr[j]
                uj = u[j]
                for c in range(E):
                    o[t][qh_off + c] += pj * uj[kh_off + c]
        pa.append(p_h)
    fout = mm_fwd(o, P[f"wo.{pl}"])
    cache = dict(a=a, s1=s1, q=q, k=k, u=u, qpre=qpre, kpre=kpre, qns=qns, kns=kns, pa=pa, o=o)
    return fout, cache


def F_bwd(g, P, G, pl, x2, dy1, cache, rope_tab):
    T, D, H, H_kv, E, KVD = g["T"], g["D"], g["H"], g["H_kv"], g["E"], g["KVD"]
    a, s1 = cache["a"], cache["s1"]
    q, k, u, pa, o = cache["q"], cache["k"], cache["u"], cache["pa"], cache["o"]
    mm_dw(o, dy1, G[f"wo.{pl}"])
    do = mm_dx(dy1, P[f"wo.{pl}"])
    dq = [[0.0] * D for _ in range(T)]
    dk = [[0.0] * KVD for _ in range(T)]
    du = [[0.0] * KVD for _ in range(T)]
    sE = math.sqrt(E)
    grp = H // H_kv
    for hi in range(H):
        kh = hi // grp
        qh_off, kh_off = hi * E, kh * E
        for t in range(T):
            pr = pa[hi][t]
            dot = do[t][qh_off:qh_off + E]
            dp = [sum(dot[c] * u[j][kh_off + c] for c in range(E)) for j in range(t + 1)]
            for j in range(t + 1):
                pj = pr[j]
                for c in range(E):
                    du[j][kh_off + c] += pj * dot[c]
            sdp = sum(pr[j] * dp[j] for j in range(t + 1))
            ds = [pr[j] * (dp[j] - sdp) / sE for j in range(t + 1)]
            qt = q[t][qh_off:qh_off + E]
            for j in range(t + 1):
                dsj = ds[j]
                for c in range(E):
                    dq[t][qh_off + c] += dsj * k[j][kh_off + c]
                    dk[j][kh_off + c] += dsj * qt[c]
    if g["has_rope"]:
        cos_t, sin_t = rope_tab
        for t in range(T):
            for hi in range(H):
                dq[t][hi * E:(hi + 1) * E] = apply_rope_vec(dq[t][hi * E:(hi + 1) * E], t, cos_t, sin_t, True)
            for kh in range(H_kv):
                dk[t][kh * E:(kh + 1) * E] = apply_rope_vec(dk[t][kh * E:(kh + 1) * E], t, cos_t, sin_t, True)
    if g["has_qknorm"]:
        qpre, kpre, qns, kns = cache["qpre"], cache["kpre"], cache["qns"], cache["kns"]
        qn_w, kn_w = P[f"qn.{pl}"], P[f"kn.{pl}"]
        dqn, dkn = G[f"qn.{pl}"], G[f"kn.{pl}"]
        for t in range(T):
            for hi in range(H):
                pre = qpre[t][hi * E:(hi + 1) * E]
                dv = dq[t][hi * E:(hi + 1) * E]
                s = qns[t][hi]
                for j in range(E):
                    dqn[j] += dv[j] * pre[j] / s
                dot = sum(dv[j] * qn_w[j] * pre[j] for j in range(E))
                kk = dot / (E * s * s)
                dq[t][hi * E:(hi + 1) * E] = [(dv[j] * qn_w[j] - pre[j] * kk) / s for j in range(E)]
            for kh in range(H_kv):
                pre = kpre[t][kh * E:(kh + 1) * E]
                dv = dk[t][kh * E:(kh + 1) * E]
                s = kns[t][kh]
                for j in range(E):
                    dkn[j] += dv[j] * pre[j] / s
                dot = sum(dv[j] * kn_w[j] * pre[j] for j in range(E))
                kk = dot / (E * s * s)
                dk[t][kh * E:(kh + 1) * E] = [(dv[j] * kn_w[j] - pre[j] * kk) / s for j in range(E)]
    mm_dw(a, dq, G[f"wq.{pl}"])
    mm_dw(a, dk, G[f"wk.{pl}"])
    mm_dw(a, du, G[f"wv.{pl}"])
    da_q = mm_dx(dq, P[f"wq.{pl}"])
    da_k = mm_dx(dk, P[f"wk.{pl}"])
    da_u = mm_dx(du, P[f"wv.{pl}"])
    da = [[da_q[t][j] + da_k[t][j] + da_u[t][j] for j in range(D)] for t in range(T)]
    return rms_bwd_mat(x2, P[f"g1.{pl}"], s1, da, G[f"g1.{pl}"])


def G_fwd(g, P, pl, y1):
    T, F = g["T"], g["F"]
    eps = g["rms_eps"]
    b, s2 = rms_fwd_mat(y1, P[f"g2.{pl}"], eps)
    fp = mm_fwd(b, P[f"w1.{pl}"])
    up = mm_fwd(b, P[f"wup.{pl}"]) if g["is_gated"] else None
    if g["has_swiglu"]:
        rl = [[fp[t][i] * sigmoid(fp[t][i]) * up[t][i] for i in range(F)] for t in range(T)]
    elif g["has_geglu"]:
        rl = [[gelu(fp[t][i]) * up[t][i] for i in range(F)] for t in range(T)]
    elif g["has_gelu"]:
        rl = [[gelu(fp[t][i]) for i in range(F)] for t in range(T)]
    else:
        rl = [[v if v > 0.0 else 0.0 for v in row] for row in fp]
    gout = mm_fwd(rl, P[f"w2.{pl}"])
    return gout, dict(b=b, s2=s2, fp=fp, up=up, rl=rl)


def G_bwd(g, P, G, pl, y1, dy2, cache):
    T, D, F = g["T"], g["D"], g["F"]
    b, s2, fp, up, rl = cache["b"], cache["s2"], cache["fp"], cache["up"], cache["rl"]
    mm_dw(rl, dy2, G[f"w2.{pl}"])
    df1 = mm_dx(dy2, P[f"w2.{pl}"])
    df2 = None
    if g["has_swiglu"]:
        df2 = [[0.0] * F for _ in range(T)]
        for t in range(T):
            for i in range(F):
                dy, gv, uv = df1[t][i], fp[t][i], up[t][i]
                sig = sigmoid(gv)
                df2[t][i] = dy * gv * sig
                ds = sig * (1.0 + gv * (1.0 - sig))
                df1[t][i] = dy * uv * ds
    elif g["has_geglu"]:
        df2 = [[0.0] * F for _ in range(T)]
        for t in range(T):
            for i in range(F):
                dy, gv, uv = df1[t][i], fp[t][i], up[t][i]
                df2[t][i] = dy * gelu(gv)
                df1[t][i] = dy * uv * gelu_d(gv)
    elif g["has_gelu"]:
        df1 = [[df1[t][i] * gelu_d(fp[t][i]) for i in range(F)] for t in range(T)]
    else:
        df1 = [[df1[t][i] if fp[t][i] > 0.0 else 0.0 for i in range(F)] for t in range(T)]
    mm_dw(b, df1, G[f"w1.{pl}"])
    db = mm_dx(df1, P[f"w1.{pl}"])
    if g["is_gated"]:
        mm_dw(b, df2, G[f"wup.{pl}"])
        db_up = mm_dx(df2, P[f"wup.{pl}"])
        db = [[db[t][j] + db_up[t][j] for j in range(D)] for t in range(T)]
    return rms_bwd_mat(y1, P[f"g2.{pl}"], s2, db, G[f"g2.{pl}"])


def rev_seq_loss_grad(g, P, G, xs, ys, scale, rope_tab, zero_act_cache=True):
    T, D, V, LV = g["T"], g["D"], g["V"], g["LV"]
    wte = P["wte"]
    if g["has_rope"]:
        h0 = [[wte[xs[t]][j] for j in range(D)] for t in range(T)]
    else:
        wpe = P["wpe"]
        h0 = [[wte[xs[t]][j] + wpe[t][j] for j in range(D)] for t in range(T)]
    x1 = [list(r) for r in h0]
    x2 = [list(r) for r in h0]
    layer_caches = [] if not zero_act_cache else None
    for l in range(LV):
        pl = phys_layer(g, l)
        fout, fcache = F_fwd(g, P, pl, x2, rope_tab)
        y1 = [[x1[t][j] + fout[t][j] for j in range(D)] for t in range(T)]
        gout, gcache = G_fwd(g, P, pl, y1)
        y2 = [[x2[t][j] + gout[t][j] for j in range(D)] for t in range(T)]
        if not zero_act_cache:
            layer_caches.append((x1, x2, y1, y2, fcache, gcache))
        x1, x2 = y1, y2
    hf = [[0.5 * (x1[t][j] + x2[t][j]) for j in range(D)] for t in range(T)]
    gf = P["gf"]
    z, sf = rms_fwd_mat(hf, gf, g["rms_eps"])
    lmw = P["lmh"] if g["has_untied"] else P["wte"]
    dlmw = G["lmh"] if g["has_untied"] else G["wte"]
    lsum = 0.0
    dz = [[0.0] * D for _ in range(T)]
    for t in range(T):
        lg = [sum(z[t][j] * lmw[c][j] for j in range(D)) for c in range(V)]
        mx = max(lg)
        ew = [math.exp(v - mx) for v in lg]
        sm = sum(ew)
        pr = [v / sm for v in ew]
        lsum += -(lg[ys[t]] - mx - math.log(sm))
        for c in range(V):
            dl = (pr[c] - (1.0 if c == ys[t] else 0.0)) * scale
            for j in range(D):
                dlmw[c][j] += dl * z[t][j]
                dz[t][j] += dl * lmw[c][j]
    dhf = rms_bwd_mat(hf, gf, sf, dz, G["gf"])
    dy1 = [[0.5 * dhf[t][j] for j in range(D)] for t in range(T)]
    dy2 = [[0.5 * dhf[t][j] for j in range(D)] for t in range(T)]
    y1, y2 = x1, x2
    for l in range(LV - 1, -1, -1):
        pl = phys_layer(g, l)
        if zero_act_cache:
            gout, gcache = G_fwd(g, P, pl, y1)
            x2_rec = [[y2[t][j] - gout[t][j] for j in range(D)] for t in range(T)]
            dy1_g = G_bwd(g, P, G, pl, y1, dy2, gcache)
            dy1 = [[dy1[t][j] + dy1_g[t][j] for j in range(D)] for t in range(T)]
            fout, fcache = F_fwd(g, P, pl, x2_rec, rope_tab)
            x1_rec = [[y1[t][j] - fout[t][j] for j in range(D)] for t in range(T)]
            dx2_f = F_bwd(g, P, G, pl, x2_rec, dy1, fcache, rope_tab)
            dy2 = [[dy2[t][j] + dx2_f[t][j] for j in range(D)] for t in range(T)]
            y1, y2 = x1_rec, x2_rec
        else:
            x1_c, x2_c, y1_c, y2_c, fcache, gcache = layer_caches[l]
            dy1_g = G_bwd(g, P, G, pl, y1_c, dy2, gcache)
            dy1 = [[dy1[t][j] + dy1_g[t][j] for j in range(D)] for t in range(T)]
            dx2_f = F_bwd(g, P, G, pl, x2_c, dy1, fcache, rope_tab)
            dy2 = [[dy2[t][j] + dx2_f[t][j] for j in range(D)] for t in range(T)]
    for t in range(T):
        dh0 = [dy1[t][j] + dy2[t][j] for j in range(D)]
        for j in range(D):
            G["wte"][xs[t]][j] += dh0[j]
            if not g["has_rope"]:
                G["wpe"][t][j] += dh0[j]
    return lsum / T


def rev_loss_and_grad(g, flat, batch_seqs, zero_act_cache=True):
    P = unpack_params(g, flat)
    G = zeros_like(g)
    rope_tab = build_rope_tables(g) if g["has_rope"] else None
    B, T = g["B"], g["T"]
    sc = 1.0 / (B * T)
    tot = 0.0
    for xs, ys in batch_seqs:
        tot += rev_seq_loss_grad(g, P, G, xs, ys, sc, rope_tab, zero_act_cache=zero_act_cache)
    return tot / B, pack_params(g, G)


def rev_inv_error(g, flat, xs):
    T, D, LV = g["T"], g["D"], g["LV"]
    P = unpack_params(g, flat)
    rope_tab = build_rope_tables(g) if g["has_rope"] else None
    if g["has_rope"]:
        h0 = [[P["wte"][xs[t]][j] for j in range(D)] for t in range(T)]
    else:
        h0 = [[P["wte"][xs[t]][j] + P["wpe"][t][j] for j in range(D)] for t in range(T)]
    x1_init = [list(r) for r in h0]
    x2_init = [list(r) for r in h0]
    x1, x2 = [list(r) for r in h0], [list(r) for r in h0]
    for l in range(LV):
        pl = phys_layer(g, l)
        fout, _ = F_fwd(g, P, pl, x2, rope_tab)
        y1 = [[x1[t][j] + fout[t][j] for j in range(D)] for t in range(T)]
        gout, _ = G_fwd(g, P, pl, y1)
        y2 = [[x2[t][j] + gout[t][j] for j in range(D)] for t in range(T)]
        x1, x2 = y1, y2
    for l in range(LV - 1, -1, -1):
        pl = phys_layer(g, l)
        gout, _ = G_fwd(g, P, pl, x1)
        x2_rec = [[x2[t][j] - gout[t][j] for j in range(D)] for t in range(T)]
        fout, _ = F_fwd(g, P, pl, x2_rec, rope_tab)
        x1_rec = [[x1[t][j] - fout[t][j] for j in range(D)] for t in range(T)]
        x1, x2 = x1_rec, x2_rec
    mx = 0.0
    for t in range(T):
        for j in range(D):
            mx = max(mx, abs(x1[t][j] - x1_init[t][j]), abs(x2[t][j] - x2_init[t][j]))
    return mx


def adam_step(g, params, m, v, grad, step):
    lr = g["lr"]
    wu = g["warmup"]
    tot = g["total"]
    if wu > 0 and step < wu:
        lrs = lr * (step + 1) / wu
    else:
        rt = max(1, tot - wu)
        prog = min(1.0, (step - wu) / rt)
        if g["is_adamw"]:
            lrs = lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * prog)))
        else:
            lrs = lr * (1.0 - 0.9 * prog)
    gn = math.sqrt(sum(gv * gv for gv in grad))
    if gn > 1.0:
        sc = 1.0 / gn
        grad = [gv * sc for gv in grad]
    b1 = 0.9
    b2 = 0.95 if g["is_adamw"] else 0.99
    bc1 = 1.0 / (1.0 - b1 ** (step + 1))
    bc2 = 1.0 / (1.0 - b2 ** (step + 1))
    wd = g["wd"] if g["is_adamw"] else 0.0
    specs, _ = tensor_specs(g)
    gain_mask = []
    for name, _, r, c in specs:
        gain_mask.extend([is_gain_name(name)] * (r * c))
    p_new, m_new, v_new = [], [], []
    for i, (w, mi, vi, gi) in enumerate(zip(params, m, v, grad)):
        m2 = b1 * mi + (1.0 - b1) * gi
        v2 = b2 * vi + (1.0 - b2) * gi * gi
        if wd > 0.0 and not gain_mask[i]:
            w = w - lrs * wd * w
        upd = (lrs * m2 * bc1) / (math.sqrt(v2 * bc2) + 1e-8)
        p_new.append(w - upd)
        m_new.append(m2)
        v_new.append(v2)
    return p_new, m_new, v_new


# ---------------------------------------------------------------- compiler & test runner

def compile_target(omc, root, out_path, mode):
    nat = os.path.join(root, "native")
    units = ["math.om", "data.om", "lpool.om", "model.om", "adam.om", "ckpt.om", "bpe.om"]
    if mode in ("mac_om1", "mac_om2"):
        units.append("mio.om")
        hdr = "target a64macos\n" + ("om2\n" if mode == "mac_om2" else "")
    elif mode == "gb10_om2":
        units.append("lio.om")
        hdr = "om2\n"
    else:
        raise ValueError(mode)
    units.extend(["nogpu.om", "rev.om"])
    parts = [hdr]
    for u in units:
        with open(os.path.join(nat, u), "r", encoding="utf-8") as f:
            parts.append(f.read())
    parts.append("\nfn olast to void\nend\nfn ocmd to int\nret rev_cli\nend\n")
    src = "\n".join(parts)
    res = subprocess.run([omc], input=src.encode("utf-8"), capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(f"omc failed for {mode}: {res.stderr.decode('utf-8', 'replace')}")
    with open(out_path, "wb") as f:
        f.write(res.stdout)
    os.chmod(out_path, 0o755)
    if mode in ("mac_om1", "mac_om2") and shutil.which("codesign"):
        cs = subprocess.run(["codesign", "-s", "-", "-f", out_path], capture_output=True)
        if cs.returncode != 0:
            raise RuntimeError(f"codesign failed: {cs.stderr.decode('utf-8', 'replace')}")


def run_oracle_case(bin_path, tmpdir, label, cfg):
    g = enrich_cfg(cfg)
    out_bin = os.path.join(tmpdir, f"{label}.bin")
    ck4_path = os.path.join(tmpdir, f"{label}.ck4")
    steps = 5
    argv = [
        bin_path, "oracle",
        str(g["arch_flags"]), str(g["V"]), str(g["T"]), str(g["D"]),
        str(g["H"]), str(g["H_kv"]), str(g["F"]), str(g["L"]), str(g["B"]),
        str(g.get("LP", 0)), str(g.get("LC", 0)), str(g.get("NP", 0)),
        str(g["seed"]), str(steps), out_bin, ck4_path,
    ]
    res = subprocess.run(argv, capture_output=True, text=True)
    assert res.returncode == 0, f"{label} oracle failed: rc={res.returncode} {res.stderr}"

    with open(out_bin, "rb") as f:
        raw = f.read()
    P, inv_err_om, ls_om, act_bytes, pair_bytes = struct.unpack("<QddQQ", raw[:40])
    _, exp_P = tensor_specs(g)
    assert P == exp_P, f"{label}: P mismatch {P} != {exp_P}"
    assert act_bytes == 0, f"{label}: expected rev_act_bytes_used == 0, got {act_bytes}"
    assert pair_bytes == 2 * g["T"] * g["D"] * 8, f"{label}: pair_bytes={pair_bytes}"
    assert inv_err_om < 1e-12, f"{label}: OM inverse error {inv_err_om:.3e} >= 1e-12"

    pos = 40
    params0 = list(struct.unpack(f"<{P}d", raw[pos:pos + 8 * P])); pos += 8 * P
    grads_om = list(struct.unpack(f"<{P}d", raw[pos:pos + 8 * P])); pos += 8 * P
    step_losses_om = list(struct.unpack(f"<{steps}d", raw[pos:pos + 8 * steps])); pos += 8 * steps
    params_final_om = list(struct.unpack(f"<{P}d", raw[pos:pos + 8 * P])); pos += 8 * P
    assert pos == len(raw)

    batch_seqs = [
        ([splitmix64(101 + b * 31 + t) % g["V"] for t in range(g["T"])],
         [splitmix64(701 + b * 31 + t) % g["V"] for t in range(g["T"])])
        for b in range(g["B"])
    ]

    # 1. Python algebraic inverse reconstruction check
    py_inv_err = rev_inv_error(g, params0, batch_seqs[0][0])
    assert py_inv_err < 1e-12, f"{label}: Python inverse error {py_inv_err:.3e}"

    # 2. Zero-activation vs cached-activation equivalence in Python + vs rev.om
    py_loss_z, py_grad_z = rev_loss_and_grad(g, params0, batch_seqs, zero_act_cache=True)
    py_loss_c, py_grad_c = rev_loss_and_grad(g, params0, batch_seqs, zero_act_cache=False)
    assert abs(py_loss_z - ls_om) < 1e-10, f"{label}: loss diff {abs(py_loss_z - ls_om):.3e}"
    assert abs(py_loss_z - py_loss_c) < 1e-12

    max_g_om_vs_cached = 0.0
    max_g_z_vs_c = 0.0
    g_scale = max(1.0, max(abs(x) for x in py_grad_c))
    for a, b, c in zip(grads_om, py_grad_z, py_grad_c):
        max_g_om_vs_cached = max(max_g_om_vs_cached, abs(a - c) / g_scale)
        max_g_z_vs_c = max(max_g_z_vs_c, abs(b - c) / g_scale)
    assert max_g_om_vs_cached < 1e-10, f"{label}: grad OM vs cached err {max_g_om_vs_cached:.3e}"
    assert max_g_z_vs_c < 1e-12, f"{label}: grad zero vs cached err {max_g_z_vs_c:.3e}"

    # 3. Central finite-difference check across every tensor type (4-point O(h^4) central FD)
    h = 1e-4
    specs, _ = tensor_specs(g)
    p_work = list(params0)
    worst_fd = 0.0
    for name, off, r, cc in specs:
        n = r * cc
        idxs = [off, off + n // 2, off + n - 1]
        for idx in idxs:
            orig = p_work[idx]
            p_work[idx] = orig + 2.0 * h
            lp2, _ = rev_loss_and_grad(g, p_work, batch_seqs, zero_act_cache=True)
            p_work[idx] = orig + h
            lp1, _ = rev_loss_and_grad(g, p_work, batch_seqs, zero_act_cache=True)
            p_work[idx] = orig - h
            lm1, _ = rev_loss_and_grad(g, p_work, batch_seqs, zero_act_cache=True)
            p_work[idx] = orig - 2.0 * h
            lm2, _ = rev_loss_and_grad(g, p_work, batch_seqs, zero_act_cache=True)
            p_work[idx] = orig
            gn = (-lp2 + 8.0 * lp1 - 8.0 * lm1 + lm2) / (12.0 * h)
            ga = grads_om[idx]
            rel = abs(gn - ga) / max(abs(ga), abs(gn), 1e-2)
            worst_fd = max(worst_fd, rel)
            assert rel < 1e-8, f"{label} {name}[{idx - off}]: FD rel={rel:.3e} ga={ga:.10e} gn={gn:.10e}"

    # 4. Multi-step Adam/AdamW training equivalence vs cached-activation Python reference
    p_cur = list(params0)
    m_cur = [0.0] * P
    v_cur = [0.0] * P
    for stp in range(steps):
        l_ref, g_ref = rev_loss_and_grad(g, p_cur, batch_seqs, zero_act_cache=False)
        assert abs(l_ref - step_losses_om[stp]) < 1e-10, (
            f"{label} step {stp}: loss {step_losses_om[stp]} != {l_ref}"
        )
        p_cur, m_cur, v_cur = adam_step(g, p_cur, m_cur, v_cur, g_ref, stp)
    assert step_losses_om[-1] < step_losses_om[0], (
        f"{label}: loss did not decrease ({step_losses_om[0]:.6f} -> {step_losses_om[-1]:.6f})"
    )
    max_p_diff = max(abs(a - b) for a, b in zip(params_final_om, p_cur))
    assert max_p_diff < 1e-10, f"{label}: final params diff {max_p_diff:.3e}"

    # 5. OLLMCK04 checkpoint & flag compatibility check
    ck = read_ck4(ck4_path)
    assert ck["arch_flags"] == g["arch_flags"], f"{label}: CK04 flags {ck['arch_flags']} != {g['arch_flags']}"
    assert ck["arch_flags"] & FL_REV, f"{label}: FL_REV bit missing in CK04"
    assert ck["step"] == steps and ck["P"] == P
    assert ck["LP"] == g.get("LP", 0) and ck["LC"] == g.get("LC", 0) and ck["NP"] == g.get("NP", 0)
    assert max(abs(a - b) for a, b in zip(ck["params"], params_final_om)) == 0.0
    ck_res = subprocess.run([bin_path, "ck4load", ck4_path], capture_output=True, text=True)
    assert ck_res.returncode == 0, f"{label}: ck4load failed: {ck_res.stderr}"
    ck_toks = ck_res.stdout.strip().split()
    assert ck_toks == ["C4", str(P), str(g["arch_flags"]), str(g["LV"])], f"unexpected ck4load: {ck_toks}"

    print(
        f"PASS {label:18s} inv={inv_err_om:.2e} py_inv={py_inv_err:.2e} "
        f"grad_eq={max_g_om_vs_cached:.2e} fd={worst_fd:.2e} "
        f"loss={step_losses_om[0]:.4f}->{step_losses_om[-1]:.4f} act_bytes={act_bytes} ck4_flags={ck['arch_flags']}"
    )


def f32_round(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bf16_sim(x):
    bits = struct.unpack("<I", struct.pack("<f", x))[0] & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def q32_quant(x):
    s = x * Q32_SCALE
    return int(s + 0.5) if s >= 0.0 else int(s - 0.5)


def run_q32_contrast_test(oracle_bin, tmpdir):
    # 1. Demonstrate FP32 & BF16 bit-loss at 1e5 / 1e-4 ratio (1e9 dynamic range)
    n = 64
    nl = 8
    x_init = [
        (100000.0 + ((splitmix64(5000 + i) >> 11) * (2.0 ** -53)) * 1000.0) * (-1.0 if (i & 1) else 1.0)
        for i in range(n)
    ]
    perts = []
    for l in range(nl):
        layer_p = []
        for i in range(n):
            r = (splitmix64(l * 1009 + i * 17 + 31337) >> 11) * (2.0 ** -53)
            v = (2.0 * r - 1.0) * 1.5e-4
            if abs(v) < 5.0e-5:
                v = v + 5.0e-5 if v >= 0.0 else v - 5.0e-5
            layer_p.append(bf16_sim(v))
        perts.append(layer_p)

    # In FP32, adding 1e-4 perturbations onto ~1e5 magnitudes loses bits completely
    x_f32 = [f32_round(v) for v in x_init]
    for l in range(nl):
        for i in range(n):
            x_f32[i] = f32_round(x_f32[i] + perts[l][i])
    assert x_f32 == [f32_round(v) for v in x_init], "expected FP32 to drop 1e-4 updates on 1e5 residuals"

    # Even with slightly larger BF16-scale perturbations, float addition+subtraction fails bit-reversibility
    x_bf = bf16_sim(100000.0)
    f_bf = bf16_sim(512.0 * 0.25)
    assert bf16_sim(bf16_sim(x_bf + f_bf) - f_bf) != x_bf, "expected BF16 (x+F)-F != x"

    # Q30 i64 integer stream preserves every bit of the 1e-4 updates and reverses to 0 bit diff
    x_q = [q32_quant(v) for v in x_init]
    x_q_orig = list(x_q)
    for l in range(nl):
        for i in range(n):
            x_q[i] += q32_quant(perts[l][i])
    assert x_q != x_q_orig, "Q30 stream failed to accumulate 1e-4 updates"
    for l in range(nl - 1, -1, -1):
        for i in range(n):
            x_q[i] -= q32_quant(perts[l][i])
    assert x_q == x_q_orig, "Q30 inverse stream failed bit-exact reconstruction"

    # 2. Full transformer oracle check with FL_REV | FL_Q32 (flag 256) enabled
    out_bin = os.path.join(tmpdir, "q32_rev.bin")
    ck4_path = os.path.join(tmpdir, "q32_rev.ck4")
    fl = FL_REV | FL_Q32 | FL_ROPE | FL_SWIGLU | FL_ADAMW
    argv = [
        oracle_bin, "oracle",
        str(fl), "32", "4", "8", "2", "1", "16", "2", "2",
        "0", "0", "0", "77", "2", out_bin, ck4_path,
    ]
    res = subprocess.run(argv, capture_output=True, text=True)
    assert res.returncode == 0, f"q32_rev oracle failed: rc={res.returncode} {res.stderr}"
    with open(out_bin, "rb") as f:
        raw = f.read()
    P, inv_err_om, ls_om, act_bytes, pair_bytes = struct.unpack("<QddQQ", raw[:40])
    assert inv_err_om == 0.0, f"expected exact 0.0 inverse error with FL_Q32, got {inv_err_om:.3e}"
    assert act_bytes == 0
    ck = read_ck4(ck4_path)
    assert ck["arch_flags"] & FL_Q32
    print(
        f"PASS q32_lock4        fp32_dropped=True q30_inv_bits=0 "
        f"model_q32_inv={inv_err_om:.2e} loss={ls_om:.4f} ck4_flags={ck['arch_flags']}"
    )


def main():
    root = find_repo_root()
    omc = find_omc()
    is_mac = platform.system() == "Darwin"
    is_a64_linux = platform.system() == "Linux" and platform.machine() == "aarch64"
    with tempfile.TemporaryDirectory() as tmpdir:
        gb10_om2 = os.path.join(tmpdir, "rev-gb10-om2")
        compile_target(omc, root, gb10_om2, "gb10_om2")
        assert os.path.getsize(gb10_om2) > 0

        if is_mac:
            mac_om1 = os.path.join(tmpdir, "rev-mac-om1")
            mac_om2 = os.path.join(tmpdir, "rev-mac-om2")
            compile_target(omc, root, mac_om1, "mac_om1")
            compile_target(omc, root, mac_om2, "mac_om2")
            print(
                f"PASS compile          mac_om1={os.path.getsize(mac_om1)}B "
                f"mac_om2={os.path.getsize(mac_om2)}B gb10_om2={os.path.getsize(gb10_om2)}B"
            )
            run_pairs = [("mac_om1", mac_om1), ("mac_om2", mac_om2)]
            oracle_bin = mac_om2
        elif is_a64_linux:
            print(f"PASS compile          gb10_om2={os.path.getsize(gb10_om2)}B")
            run_pairs = [("gb10_om2", gb10_om2)]
            oracle_bin = gb10_om2
        else:
            print(f"PASS compile          gb10_om2={os.path.getsize(gb10_om2)}B (non-arm64 runtime skipped)")
            return

        for label, bin_p in run_pairs:
            st = subprocess.run([bin_p, "selftest"], capture_output=True, text=True)
            assert st.returncode == 0, f"{label} selftest failed:\n{st.stdout}\n{st.stderr}"
            for tok in ("PASS I1 ", "PASS G1 ", "PASS I2 ", "PASS G2 ", "PASS TR ", "PASS Q32 0 ", "PASS DETBWD 0 "):
                assert tok in st.stdout, f"{label} selftest missing {tok!r} in:\n{st.stdout}"
            m0 = subprocess.run([bin_p, "mode0"], capture_output=True, text=True)
            assert m0.returncode == 0, f"{label} mode0 failed:\n{m0.stdout}\n{m0.stderr}"
            q32 = subprocess.run([bin_p, "q32"], capture_output=True, text=True)
            assert q32.returncode == 0 and "PASS Q32 0 8" in q32.stdout, (
                f"{label} q32 failed:\n{q32.stdout}\n{q32.stderr}"
            )
            det = subprocess.run([bin_p, "detbwd"], capture_output=True, text=True)
            assert det.returncode == 0 and "PASS DETBWD 0 " in det.stdout, (
                f"{label} detbwd failed:\n{det.stdout}\n{det.stderr}"
            )
        print(f"PASS selftest+mode0   {[lbl for lbl, _ in run_pairs]} passed built-in self-tests (incl. Q32 & DETBWD)")

        # Verify Lock 4 contrast: FP32/BF16 stream drops/corrupts 1e-4 updates on 1e5 residuals,
        # whereas Q30 i64 fixed-point stream preserves every bit across 8 layers and reverses to 0 bit diff.
        run_q32_contrast_test(oracle_bin, tmpdir)

        cases = [
            ("gpt_relu_rev", dict(
                arch_flags=FL_REV, V=32, T=4, D=8, H=2, H_kv=2, F=16, L=2, B=2,
                LP=0, LC=0, NP=0, seed=42, total=50, lr=0.01,
                rope_theta_milli=10_000_000, rms_eps=1e-5, wd=0.05, warmup=10,
            )),
            ("llama_rope_gqa", dict(
                arch_flags=FL_REV | FL_ROPE | FL_SWIGLU | FL_ADAMW,
                V=32, T=4, D=8, H=2, H_kv=1, F=16, L=2, B=2,
                LP=0, LC=0, NP=0, seed=101, total=50, lr=0.01,
                rope_theta_milli=10_000_000, rms_eps=1e-5, wd=0.05, warmup=10,
            )),
            ("qwen3_looped_np3", dict(
                arch_flags=FL_REV | FL_ROPE | FL_SWIGLU | FL_UNTIED | FL_QKNORM | FL_ADAMW,
                V=32, T=4, D=8, H=2, H_kv=1, F=16, L=3, B=2,
                LP=1, LC=1, NP=3, seed=202, total=50, lr=0.01,
                rope_theta_milli=10_000_000, rms_eps=1e-5, wd=0.05, warmup=10,
            )),
            ("gelu_rev", dict(
                arch_flags=FL_REV | FL_GELU,
                V=24, T=4, D=8, H=2, H_kv=2, F=16, L=2, B=2,
                LP=0, LC=0, NP=0, seed=303, total=50, lr=0.01,
                rope_theta_milli=10_000_000, rms_eps=1e-5, wd=0.05, warmup=10,
            )),
            ("geglu_rope_rev", dict(
                arch_flags=FL_REV | FL_ROPE | FL_GEGLU | FL_ADAMW,
                V=24, T=4, D=8, H=2, H_kv=1, F=16, L=2, B=2,
                LP=0, LC=0, NP=0, seed=404, total=50, lr=0.01,
                rope_theta_milli=10_000_000, rms_eps=1e-5, wd=0.05, warmup=10,
            )),
        ]
        for label, cfg in cases:
            run_oracle_case(oracle_bin, tmpdir, label, cfg)

    print("ALL PILLAR 2 REVERSIBLE CHECKS PASSED")


if __name__ == "__main__":
    main()
