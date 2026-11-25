#!/usr/bin/env python3
"""Universal Omega Helix (OLLMCK04 / ollm-univ) pure-Python-stdlib f64 referee.

Independently implements the OLLMCK04 / OLLMGR04 binary format, universal
geometry & dynamic offsets, deterministic init, RoPE / WPE, RMSNorm,
QK-RMSNorm, GQA causal attention, ReLU / GELU / SwiGLU / GeGLU FFNs,
tied / untied LM head, Looped Core (LP/LC/NP), u8/u16/u32 token streams,
Adam / AdamW with linear or cosine LR schedule, KV-cached incremental decode,
and temperature / greedy sampling — and cross-checks `ollm-univ` against it.

Verification only (Python 3 standard library only); never part of the build.
"""

import argparse
import contextlib
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from operator import mul

M64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
INIT_SCALE = 0.034641016151377546  # 0.02 * sqrt(3)
FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
LN2 = 0.6931471805599453
SQRT_2_OVER_PI = 0.79788456080286536
MAGIC_CK4 = b"OLLMCK04"
MAGIC_GR4 = b"OLLMGR04"
CKHDR = 208

# Arch flag bits
FL_ROPE = 1
FL_SWIGLU = 2
FL_UNTIED = 4
FL_QKNORM = 8
FL_ADAMW_COS = 16
FL_GELU = 32
FL_GEGLU = 64

PRESETS = {
    "llama": dict(arch_flags=23, rope_theta_milli=500_000_000, rms_eps=1e-5, wd=0.1, warmup=100),
    "mistral": dict(arch_flags=23, rope_theta_milli=500_000_000, rms_eps=1e-5, wd=0.1, warmup=100),
    "qwen2": dict(arch_flags=23, rope_theta_milli=1_000_000_000, rms_eps=1e-6, wd=0.1, warmup=100),
    "qwen3": dict(arch_flags=31, rope_theta_milli=1_000_000_000, rms_eps=1e-6, wd=0.1, warmup=100),
    "gemma": dict(arch_flags=81, rope_theta_milli=10_000_000, rms_eps=1e-6, wd=0.1, warmup=100),
    "gpt": dict(arch_flags=0, rope_theta_milli=10_000_000, rms_eps=1e-5, wd=0.0, warmup=100),
    "gelu": dict(arch_flags=32, rope_theta_milli=10_000_000, rms_eps=1e-5, wd=0.0, warmup=100),
}

RTOL = 1e-9
FLOOR_FRAC = 1e-3

DEFAULT_BIN = "/tmp/omega-helix-verify-mac/ollm-univ"


class FormatError(Exception):
    pass


# ---------------------------------------------------------------- primitives

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


def f2bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def bits2f(u):
    return struct.unpack("<d", struct.pack("<Q", u))[0]


def micro(x):
    v = x * 1e6
    q = math.trunc(v)
    if abs(v - q) >= 0.5:
        q += 1 if v > 0 else -1
    return int(q)


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


# ---------------------------------------------------------------- geometry & layout

def validate_geom(cfg):
    for k in ("V", "T", "D", "H", "H_kv", "F", "L", "B"):
        if cfg[k] <= 0:
            raise FormatError(f"{k}={cfg[k]} must be > 0")
    if cfg["V"] < 2:
        raise FormatError("V must be >= 2")
    D, H, H_kv = cfg["D"], cfg["H"], cfg["H_kv"]
    if D % H != 0:
        raise FormatError("D not divisible by H")
    if H_kv > H or H % H_kv != 0:
        raise FormatError("invalid GQA H_kv")
    E = D // H
    fl = cfg["arch_flags"]
    if (fl & FL_SWIGLU) and (fl & FL_GEGLU):
        raise FormatError("SwiGLU and GeGLU mutually exclusive")
    if (fl & (FL_SWIGLU | FL_GEGLU)) and (fl & FL_GELU):
        raise FormatError("gated FFN and GELU mutually exclusive")
    if (fl & FL_ROPE) and (E % 2 != 0):
        raise FormatError("RoPE requires even head dim E")
    NP = cfg.get("NP", 0)
    LP = cfg.get("LP", 0)
    LC = cfg.get("LC", 0)
    L = cfg["L"]
    if NP > 1:
        if LC == 0 or LP + LC > L:
            raise FormatError("invalid loop core (LP, LC, NP)")
        LV = L + LC * (NP - 1)
    else:
        LV = L
    out = dict(cfg)
    out["E"] = E
    out["KVD"] = H_kv * E
    out["LV"] = LV
    out["has_rope"] = bool(fl & FL_ROPE)
    out["is_gated"] = bool(fl & (FL_SWIGLU | FL_GEGLU))
    out["has_untied"] = bool(fl & FL_UNTIED)
    out["has_qknorm"] = bool(fl & FL_QKNORM)
    out["is_adamw_cos"] = bool(fl & FL_ADAMW_COS)
    out["has_gelu"] = bool(fl & FL_GELU)
    out["has_swiglu"] = bool(fl & FL_SWIGLU)
    out["has_geglu"] = bool(fl & FL_GEGLU)
    out["tok_bytes"] = 4 if cfg["V"] > 65535 else (2 if cfg["V"] > 256 else 1)
    return out


def phys_layer(g, l):
    NP = g.get("NP", 0)
    if NP <= 1:
        return l
    LP, LC = g["LP"], g["LC"]
    if l < LP:
        return l
    j = l - LP
    if j < LC * NP:
        return LP + (j % LC)
    return l - LC * (NP - 1)


def tensor_specs(g):
    """Returns (specs_list, total_P) matching uarena.om ugeom_params exactly."""
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
    base = name.split(".")[0]
    return base in ("g1", "g2", "qn", "kn", "gf")


def is_outproj_name(name):
    base = name.split(".")[0]
    return base in ("wo", "w2")


def unpack_params(g, flat):
    specs, _ = tensor_specs(g)
    out = {}
    for name, off, r, c in specs:
        if r == 1 and is_gain_name(name):
            out[name] = flat[off:off + c]
        else:
            out[name] = [flat[off + i * c:off + (i + 1) * c] for i in range(r)]
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
        if r == 1 and is_gain_name(name):
            out[name] = [0.0] * c
        else:
            out[name] = [[0.0] * c for _ in range(r)]
    return out


# ---------------------------------------------------------------- init

def init_params(g, seed):
    specs, P = tensor_specs(g)
    rs = 1.0 / math.sqrt(2.0 * g["L"])
    flat = [0.0] * P
    for name, off, r, c in specs:
        n = r * c
        if is_gain_name(name):
            for i in range(off, off + n):
                flat[i] = 1.0
        else:
            outp = is_outproj_name(name)
            for i in range(off, off + n):
                ri = (splitmix64((seed + i) & M64) >> 11) * (2.0 ** -53)
                val = (2.0 * ri - 1.0) * INIT_SCALE
                if outp:
                    val *= rs
                flat[i] = val
    return flat


# ---------------------------------------------------------------- checkpoint / grad I/O

def encode_ck4(g, seed, step, total, lr, params, m, v):
    _, P = tensor_specs(g)
    assert len(params) == P and len(m) == P and len(v) == P
    body = struct.pack(f"<{P}d", *params) + struct.pack(f"<{P}d", *m) + struct.pack(f"<{P}d", *v)
    fnv = fnv1a64(body)
    hdr = MAGIC_CK4 + struct.pack(
        "<25Q",
        g["V"], g["T"], g["D"], g["H"], g["H_kv"], g["F"], g["L"], g["B"],
        seed & M64, step, total, f2bits(lr), P,
        g["arch_flags"], g["rope_theta_milli"], f2bits(g["rms_eps"]), f2bits(g["wd"]),
        g["warmup"], g.get("LP", 0), g.get("LC", 0), g.get("NP", 0),
        0, 0, 0, fnv,
    )
    assert len(hdr) == CKHDR
    return hdr + body


def read_ck4_or_gr4(path, expect_magic=MAGIC_CK4):
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < CKHDR:
        raise FormatError("short header")
    if raw[:8] != expect_magic:
        raise FormatError(f"bad magic {raw[:8]!r}")
    vals = struct.unpack("<25Q", raw[8:CKHDR])
    V, T, D, H, H_kv, F, L, B = vals[0:8]
    seed, step, total, lrb, P_file = vals[8:13]
    arch_flags, rope_theta_milli, eps_b, wd_b, warmup, LP, LC, NP = vals[13:21]
    want_fnv = vals[24]
    g = validate_geom(dict(
        V=V, T=T, D=D, H=H, H_kv=H_kv, F=F, L=L, B=B,
        arch_flags=arch_flags, rope_theta_milli=rope_theta_milli,
        rms_eps=bits2f(eps_b), wd=bits2f(wd_b), warmup=warmup,
        LP=LP, LC=LC, NP=NP,
    ))
    _, P = tensor_specs(g)
    if P != P_file:
        raise FormatError(f"P mismatch: header {P_file} vs geom {P}")
    npay = 3 if expect_magic == MAGIC_CK4 else 1
    want_sz = CKHDR + 8 * P * npay
    if len(raw) != want_sz:
        raise FormatError(f"size mismatch: {len(raw)} vs {want_sz}")
    body = raw[CKHDR:]
    if fnv1a64(body) != want_fnv:
        raise FormatError("FNV-1a checksum mismatch")
    pools = [list(struct.unpack(f"<{P}d", body[8 * P * i:8 * P * (i + 1)])) for i in range(npay)]
    res = dict(
        g=g, seed=seed, step=step, total=total, lr=bits2f(lrb), lr_bits=lrb,
        P=P, fnv=want_fnv, raw=raw,
    )
    if npay == 3:
        res["params"], res["m"], res["v"] = pools
    else:
        res["grad"] = pools[0]
    return res


def read_ck4(path):
    return read_ck4_or_gr4(path, MAGIC_CK4)


def read_gr4(path):
    return read_ck4_or_gr4(path, MAGIC_GR4)


# ---------------------------------------------------------------- token streams

def encode_tokens(tokens, tok_bytes):
    if tok_bytes == 1:
        return bytes(tokens)
    if tok_bytes == 2:
        return struct.pack(f"<{len(tokens)}H", *tokens)
    return struct.pack(f"<{len(tokens)}I", *tokens)


def batch_offsets(seed, step, B, data_N, T):
    return [
        splitmix64((seed ^ (((step * B + b) * GOLDEN) & M64)) & M64) % (data_N - T)
        for b in range(B)
    ]


# ---------------------------------------------------------------- model math (f64)

def rms_fwd(x, g, eps):
    n = len(x)
    s = math.sqrt(sum(map(mul, x, x)) / n + eps)
    inv = 1.0 / s
    return [x[j] * inv * g[j] for j in range(n)], s


def rms_bwd(x, g, s, dy, dg):
    n = len(x)
    inv = 1.0 / s
    dot = 0.0
    for j in range(n):
        dot += dy[j] * g[j] * x[j]
    c1 = dot / (n * s * s)
    dx = [0.0] * n
    for j in range(n):
        dg[j] += dy[j] * x[j] * inv
        dx[j] = (dy[j] * g[j] - x[j] * c1) * inv
    return dx


def build_rope_table(g):
    if not g["has_rope"]:
        return None, None
    T, E = g["T"], g["E"]
    he = E // 2
    th = g["rope_theta_milli"] / 1000.0
    if th <= 0.0:
        th = 10000.0
    lth = math.log(th)
    cos_tab = [[0.0] * he for _ in range(T)]
    sin_tab = [[0.0] * he for _ in range(T)]
    for i in range(he):
        freq = math.exp(-lth * (2 * i) / E)
        for pos in range(T):
            ang = pos * freq
            cos_tab[pos][i] = math.cos(ang)
            sin_tab[pos][i] = math.sin(ang)
    return cos_tab, sin_tab


def rope_apply(v, c_row, s_row, inv=False):
    he = len(c_row)
    out = list(v)
    if not inv:
        for i in range(he):
            x0, x1 = v[i], v[i + he]
            c, s = c_row[i], s_row[i]
            out[i] = x0 * c - x1 * s
            out[i + he] = x0 * s + x1 * c
    else:
        for i in range(he):
            x0, x1 = v[i], v[i + he]
            c, s = c_row[i], s_row[i]
            out[i] = x0 * c + x1 * s
            out[i + he] = x1 * c - x0 * s
    return out


def transpose(W):
    return [list(col) for col in zip(*W)]


def vecmat(x, WT):
    return [sum(map(mul, x, col)) for col in WT]


def outer_acc(dW, x, dy):
    for r, xr in enumerate(x):
        if xr != 0.0:
            row = dW[r]
            for j, dj in enumerate(dy):
                row[j] += xr * dj


def seq_fwd_bwd(g, P, PT, x_seq, y_seq, scale, G, rope_tabs, want_loss=True, want_bwd=True):
    """Forward (and optional backward) on one sequence of length ct = len(x_seq)."""
    ct = len(x_seq)
    D, H, H_kv, E, F = g["D"], g["H"], g["H_kv"], g["E"], g["F"]
    KVD = g["KVD"]
    V = g["V"]
    LV = g["LV"]
    eps = g["rms_eps"]
    sE = math.sqrt(E)
    inv_sE = 1.0 / sE
    g_ratio = H // H_kv
    cos_tab, sin_tab = rope_tabs

    wte = P["wte"]
    if g["has_rope"]:
        h = [list(wte[x_seq[t]]) for t in range(ct)]
    else:
        wpe = P["wpe"]
        h = [[a + b for a, b in zip(wte[x_seq[t]], wpe[t])] for t in range(ct)]

    caches = []
    for l in range(LV):
        pl = phys_layer(g, l)
        g1 = P[f"g1.{pl}"]
        g2 = P[f"g2.{pl}"]
        hin = h

        an = [rms_fwd(v, g1, eps) for v in hin]
        a = [pair[0] for pair in an]

        q = [vecmat(row, PT[f"wq.{pl}"]) for row in a]
        k = [vecmat(row, PT[f"wk.{pl}"]) for row in a]
        v = [vecmat(row, PT[f"wv.{pl}"]) for row in a]

        qpre = kpre = qns = kns = None
        if g["has_qknorm"]:
            qn_g = P[f"qn.{pl}"]
            kn_g = P[f"kn.{pl}"]
            qpre = [list(row) for row in q]
            kpre = [list(row) for row in k]
            qns = [[0.0] * H for _ in range(ct)]
            kns = [[0.0] * H_kv for _ in range(ct)]
            for t in range(ct):
                for hi in range(H):
                    lo, hi_e = hi * E, (hi + 1) * E
                    q_normed, sq = rms_fwd(qpre[t][lo:hi_e], qn_g, eps)
                    q[t][lo:hi_e] = q_normed
                    qns[t][hi] = sq
                for kh in range(H_kv):
                    lo, hi_e = kh * E, (kh + 1) * E
                    k_normed, sk = rms_fwd(kpre[t][lo:hi_e], kn_g, eps)
                    k[t][lo:hi_e] = k_normed
                    kns[t][kh] = sk

        if g["has_rope"]:
            for t in range(ct):
                cr, sr = cos_tab[t], sin_tab[t]
                for hi in range(H):
                    lo, hi_e = hi * E, (hi + 1) * E
                    q[t][lo:hi_e] = rope_apply(q[t][lo:hi_e], cr, sr, inv=False)
                for kh in range(H_kv):
                    lo, hi_e = kh * E, (kh + 1) * E
                    k[t][lo:hi_e] = rope_apply(k[t][lo:hi_e], cr, sr, inv=False)

        o = [[0.0] * D for _ in range(ct)]
        probs = []
        for hi in range(H):
            kh = hi // g_ratio
            qlo, qhi = hi * E, (hi + 1) * E
            klo, khi = kh * E, (kh + 1) * E
            p_head = []
            for t in range(ct):
                qt = q[t][qlo:qhi]
                sc = [sum(map(mul, qt, k[j][klo:khi])) * inv_sE for j in range(t + 1)]
                mx = max(sc)
                ew = [math.exp(sv - mx) for sv in sc]
                invs = 1.0 / sum(ew)
                pw = [e_val * invs for e_val in ew]
                p_head.append(pw)
                ot = o[t]
                for j in range(t + 1):
                    pj = pw[j]
                    vj = v[j]
                    for c in range(E):
                        ot[qlo + c] += pj * vj[klo + c]
            probs.append(p_head)

        att = [vecmat(row, PT[f"wo.{pl}"]) for row in o]
        hmid = [[hv + av for hv, av in zip(hin[t], att[t])] for t in range(ct)]

        bn = [rms_fwd(row, g2, eps) for row in hmid]
        b = [pair[0] for pair in bn]
        f1 = [vecmat(row, PT[f"w1.{pl}"]) for row in b]
        f2 = [vecmat(row, PT[f"wup.{pl}"]) for row in b] if g["is_gated"] else None

        f3 = [[0.0] * F for _ in range(ct)]
        for t in range(ct):
            if g["has_swiglu"]:
                for i in range(F):
                    gv, uv = f1[t][i], f2[t][i]
                    f3[t][i] = gv * sigmoid(gv) * uv
            elif g["has_geglu"]:
                for i in range(F):
                    f3[t][i] = gelu(f1[t][i]) * f2[t][i]
            elif g["has_gelu"]:
                for i in range(F):
                    f3[t][i] = gelu(f1[t][i])
            else:
                for i in range(F):
                    val = f1[t][i]
                    f3[t][i] = val if val > 0.0 else 0.0

        ffn_out = [vecmat(row, PT[f"w2.{pl}"]) for row in f3]
        h_next = [[hm + fo for hm, fo in zip(hmid[t], ffn_out[t])] for t in range(ct)]
        if want_bwd:
            caches.append((hin, an, a, q, k, v, qpre, kpre, qns, kns, o, probs, hmid, bn, b, f1, f2, f3))
        h = h_next

    gf = P["gf"]
    zn = [rms_fwd(row, gf, eps) for row in h]
    lmw = P["lmh"] if g["has_untied"] else P["wte"]
    dlmw = G["lmh"] if (want_bwd and g["has_untied"]) else (G["wte"] if want_bwd else None)

    if not want_loss and not want_bwd:
        z_last = zn[-1][0]
        logits_last = [sum(map(mul, z_last, lmw[c])) for c in range(V)]
        return 0.0, logits_last

    loss_sum = 0.0
    dz_all = []
    for pos in range(ct):
        z = zn[pos][0]
        logits = [sum(map(mul, z, lmw[c])) for c in range(V)]
        mx = max(logits)
        ew = [math.exp(lv - mx) for lv in logits]
        sm = sum(ew)
        yt = y_seq[pos]
        loss_sum += math.log(sm) - (logits[yt] - mx)
        if want_bwd:
            probs_v = [ev / sm for ev in ew]
            dz = [0.0] * D
            for c in range(V):
                dl = probs_v[c]
                if c == yt:
                    dl -= 1.0
                dl *= scale
                row = lmw[c]
                drow = dlmw[c]
                for j in range(D):
                    drow[j] += dl * z[j]
                    dz[j] += dl * row[j]
            dz_all.append(dz)

    mean_loss = loss_sum / ct
    if not want_bwd:
        return mean_loss, None

    dgf = G["gf"]
    dh = [rms_bwd(h[pos], gf, zn[pos][1], dz_all[pos], dgf) for pos in range(ct)]

    for l in range(LV - 1, -1, -1):
        pl = phys_layer(g, l)
        hin, an, a, q, k, v, qpre, kpre, qns, kns, o, probs, hmid, bn, b, f1, f2, f3 = caches[l]
        W_wo = P[f"wo.{pl}"]
        W_w1 = P[f"w1.{pl}"]
        W_w2 = P[f"w2.{pl}"]
        W_wq = P[f"wq.{pl}"]
        W_wk = P[f"wk.{pl}"]
        W_wv = P[f"wv.{pl}"]
        dW_wo = G[f"wo.{pl}"]
        dW_w1 = G[f"w1.{pl}"]
        dW_w2 = G[f"w2.{pl}"]
        dW_wq = G[f"wq.{pl}"]
        dW_wk = G[f"wk.{pl}"]
        dW_wv = G[f"wv.{pl}"]
        g1, g2 = P[f"g1.{pl}"], P[f"g2.{pl}"]
        dg1, dg2 = G[f"g1.{pl}"], G[f"g2.{pl}"]

        # FFN backward
        df3 = []
        for t in range(ct):
            outer_acc(dW_w2, f3[t], dh[t])
            df3.append([sum(map(mul, dh[t], row)) for row in W_w2])

        df1 = [[0.0] * F for _ in range(ct)]
        df2 = [[0.0] * F for _ in range(ct)] if g["is_gated"] else None
        for t in range(ct):
            if g["has_swiglu"]:
                for i in range(F):
                    dy = df3[t][i]
                    gv, uv = f1[t][i], f2[t][i]
                    sig = sigmoid(gv)
                    df2[t][i] = dy * gv * sig
                    df1[t][i] = dy * uv * sig * (1.0 + gv * (1.0 - sig))
            elif g["has_geglu"]:
                for i in range(F):
                    dy = df3[t][i]
                    gv, uv = f1[t][i], f2[t][i]
                    df2[t][i] = dy * gelu(gv)
                    df1[t][i] = dy * uv * gelu_d(gv)
            elif g["has_gelu"]:
                for i in range(F):
                    df1[t][i] = df3[t][i] * gelu_d(f1[t][i])
            else:
                for i in range(F):
                    df1[t][i] = df3[t][i] if f1[t][i] > 0.0 else 0.0

        da_ffn = [[0.0] * D for _ in range(ct)]
        for t in range(ct):
            outer_acc(dW_w1, b[t], df1[t])
            for j in range(D):
                da_ffn[t][j] += sum(map(mul, df1[t], W_w1[j]))
        if g["is_gated"]:
            W_wup = P[f"wup.{pl}"]
            dW_wup = G[f"wup.{pl}"]
            for t in range(ct):
                outer_acc(dW_wup, b[t], df2[t])
                for j in range(D):
                    da_ffn[t][j] += sum(map(mul, df2[t], W_wup[j]))

        dh_mid = []
        for t in range(ct):
            dr = rms_bwd(hmid[t], g2, bn[t][1], da_ffn[t], dg2)
            dh_mid.append([dh[t][j] + dr[j] for j in range(D)])

        # Attention backward
        do = []
        for t in range(ct):
            outer_acc(dW_wo, o[t], dh_mid[t])
            do.append([sum(map(mul, dh_mid[t], row)) for row in W_wo])

        dq = [[0.0] * D for _ in range(ct)]
        dk = [[0.0] * KVD for _ in range(ct)]
        dv = [[0.0] * KVD for _ in range(ct)]
        for hi in range(H):
            kh = hi // g_ratio
            qlo, qhi = hi * E, (hi + 1) * E
            klo = kh * E
            for t in range(ct):
                pw = probs[hi][t]
                dob = do[t][qlo:qhi]
                dp = []
                dotp = 0.0
                for j in range(t + 1):
                    vj = v[j]
                    dp_j = sum(dob[c] * vj[klo + c] for c in range(E))
                    dp.append(dp_j)
                    dotp += pw[j] * dp_j
                    pj = pw[j]
                    dvj = dv[j]
                    for c in range(E):
                        dvj[klo + c] += pj * dob[c]
                dqt = dq[t]
                qt = q[t]
                for j in range(t + 1):
                    ds = pw[j] * (dp[j] - dotp) * inv_sE
                    kj = k[j]
                    dkj = dk[j]
                    for c in range(E):
                        dqt[qlo + c] += ds * kj[klo + c]
                        dkj[klo + c] += ds * qt[qlo + c]

        if g["has_rope"]:
            for t in range(ct):
                cr, sr = cos_tab[t], sin_tab[t]
                for hi in range(H):
                    lo, hi_e = hi * E, (hi + 1) * E
                    dq[t][lo:hi_e] = rope_apply(dq[t][lo:hi_e], cr, sr, inv=True)
                for kh in range(H_kv):
                    lo, hi_e = kh * E, (kh + 1) * E
                    dk[t][lo:hi_e] = rope_apply(dk[t][lo:hi_e], cr, sr, inv=True)

        if g["has_qknorm"]:
            qn_g = P[f"qn.{pl}"]
            kn_g = P[f"kn.{pl}"]
            dqn_g = G[f"qn.{pl}"]
            dkn_g = G[f"kn.{pl}"]
            for t in range(ct):
                for hi in range(H):
                    lo, hi_e = hi * E, (hi + 1) * E
                    dq[t][lo:hi_e] = rms_bwd(qpre[t][lo:hi_e], qn_g, qns[t][hi], dq[t][lo:hi_e], dqn_g)
                for kh in range(H_kv):
                    lo, hi_e = kh * E, (kh + 1) * E
                    dk[t][lo:hi_e] = rms_bwd(kpre[t][lo:hi_e], kn_g, kns[t][kh], dk[t][lo:hi_e], dkn_g)

        dh_prev = []
        for t in range(ct):
            outer_acc(dW_wq, a[t], dq[t])
            outer_acc(dW_wk, a[t], dk[t])
            outer_acc(dW_wv, a[t], dv[t])
            da = [
                sum(map(mul, dq[t], W_wq[j]))
                + sum(map(mul, dk[t], W_wk[j]))
                + sum(map(mul, dv[t], W_wv[j]))
                for j in range(D)
            ]
            dr = rms_bwd(hin[t], g1, an[t][1], da, dg1)
            dh_prev.append([dh_mid[t][j] + dr[j] for j in range(D)])
        dh = dh_prev

    dwte = G["wte"]
    if g["has_rope"]:
        for pos in range(ct):
            tok = x_seq[pos]
            drow = dwte[tok]
            dhp = dh[pos]
            for j in range(D):
                drow[j] += dhp[j]
    else:
        dwpe = G["wpe"]
        for pos in range(ct):
            tok = x_seq[pos]
            drow = dwte[tok]
            dprow = dwpe[pos]
            dhp = dh[pos]
            for j in range(D):
                drow[j] += dhp[j]
                dprow[j] += dhp[j]
    return mean_loss, None


def loss_and_grad(g, flat, seqs):
    P = unpack_params(g, flat)
    PT = {n: transpose(v) for n, v in P.items() if not is_gain_name(n)}
    G = zeros_like(g)
    rope_tabs = build_rope_table(g)
    scale = 1.0 / (len(seqs) * g["T"])
    tot = 0.0
    for x_seq, y_seq in seqs:
        ls, _ = seq_fwd_bwd(g, P, PT, x_seq, y_seq, scale, G, rope_tabs, want_loss=True, want_bwd=True)
        tot += ls
    return tot / len(seqs), pack_params(g, G)


def eval_loss(g, flat, data, W):
    P = unpack_params(g, flat)
    PT = {n: transpose(v) for n, v in P.items() if not is_gain_name(n)}
    rope_tabs = build_rope_table(g)
    T = g["T"]
    tot = 0.0
    for i in range(W):
        o = i * T
        x_seq = data[o:o + T]
        y_seq = data[o + 1:o + T + 1]
        ls, _ = seq_fwd_bwd(g, P, PT, x_seq, y_seq, 0.0, None, rope_tabs, want_loss=True, want_bwd=False)
        tot += ls
    return tot / W


# ---------------------------------------------------------------- KV-cached decode & sampling

class KVCache:
    def __init__(self, g, flat):
        self.g = g
        self.P = unpack_params(g, flat)
        self.PT = {n: transpose(v) for n, v in self.P.items() if not is_gain_name(n)}
        self.rope_tabs = build_rope_table(g)
        LV = g["LV"]
        self.k_cache = [[] for _ in range(LV)]
        self.v_cache = [[] for _ in range(LV)]

    def step(self, pos, tok):
        g = self.g
        P, PT = self.P, self.PT
        D, H, H_kv, E, F = g["D"], g["H"], g["H_kv"], g["E"], g["F"]
        KVD = g["KVD"]
        eps = g["rms_eps"]
        inv_sE = 1.0 / math.sqrt(E)
        g_ratio = H // H_kv
        cos_tab, sin_tab = self.rope_tabs

        if g["has_rope"]:
            h = list(P["wte"][tok])
        else:
            h = [a + b for a, b in zip(P["wte"][tok], P["wpe"][pos])]

        for l in range(g["LV"]):
            pl = phys_layer(g, l)
            a, _ = rms_fwd(h, P[f"g1.{pl}"], eps)
            q = vecmat(a, PT[f"wq.{pl}"])
            k = vecmat(a, PT[f"wk.{pl}"])
            v = vecmat(a, PT[f"wv.{pl}"])

            if g["has_qknorm"]:
                qn_g = P[f"qn.{pl}"]
                kn_g = P[f"kn.{pl}"]
                for hi in range(H):
                    lo, hi_e = hi * E, (hi + 1) * E
                    q[lo:hi_e], _ = rms_fwd(q[lo:hi_e], qn_g, eps)
                for kh in range(H_kv):
                    lo, hi_e = kh * E, (kh + 1) * E
                    k[lo:hi_e], _ = rms_fwd(k[lo:hi_e], kn_g, eps)

            if g["has_rope"]:
                cr, sr = cos_tab[pos], sin_tab[pos]
                for hi in range(H):
                    lo, hi_e = hi * E, (hi + 1) * E
                    q[lo:hi_e] = rope_apply(q[lo:hi_e], cr, sr, inv=False)
                for kh in range(H_kv):
                    lo, hi_e = kh * E, (kh + 1) * E
                    k[lo:hi_e] = rope_apply(k[lo:hi_e], cr, sr, inv=False)

            self.k_cache[l].append(k)
            self.v_cache[l].append(v)

            o = [0.0] * D
            kc = self.k_cache[l]
            vc = self.v_cache[l]
            for hi in range(H):
                kh = hi // g_ratio
                qlo, qhi = hi * E, (hi + 1) * E
                klo = kh * E
                qt = q[qlo:qhi]
                sc = [sum(map(mul, qt, kc[p2][klo:klo + E])) * inv_sE for p2 in range(pos + 1)]
                mx = max(sc)
                ew = [math.exp(sv - mx) for sv in sc]
                invs = 1.0 / sum(ew)
                for p2 in range(pos + 1):
                    pw = ew[p2] * invs
                    vp2 = vc[p2]
                    for c in range(E):
                        o[qlo + c] += pw * vp2[klo + c]

            att = vecmat(o, PT[f"wo.{pl}"])
            h = [hv + av for hv, av in zip(h, att)]

            b, _ = rms_fwd(h, P[f"g2.{pl}"], eps)
            f1 = vecmat(b, PT[f"w1.{pl}"])
            if g["is_gated"]:
                f2 = vecmat(b, PT[f"wup.{pl}"])
                if g["has_swiglu"]:
                    f3 = [f1[i] * sigmoid(f1[i]) * f2[i] for i in range(F)]
                else:
                    f3 = [gelu(f1[i]) * f2[i] for i in range(F)]
            elif g["has_gelu"]:
                f3 = [gelu(f1[i]) for i in range(F)]
            else:
                f3 = [v if v > 0.0 else 0.0 for v in f1]
            ff = vecmat(f3, PT[f"w2.{pl}"])
            h = [hv + fv for hv, fv in zip(h, ff)]

        z, _ = rms_fwd(h, P["gf"], eps)
        lmw = P["lmh"] if g["has_untied"] else P["wte"]
        return [sum(map(mul, z, lmw[c])) for c in range(g["V"])]


def sample_tokens(g, flat, seed, N, tm):
    nn = min(N, g["T"])
    kv = KVCache(g, flat)
    cur = 0
    st = seed & M64
    out = []
    logits_list = []
    for s in range(nn):
        logits = kv.step(s, cur)
        logits_list.append(logits)
        st = (st + GOLDEN) & M64
        if tm == 0:
            best = 0
            bv = logits[0]
            for c in range(g["V"]):
                if logits[c] > bv:
                    bv = logits[c]
                    best = c
            nxt = best
        else:
            invt = 1000.0 / float(tm)
            mx = max(logits)
            w = [math.exp((lv - mx) * invt) for lv in logits]
            sm = sum(w)
            u = ((splitmix64(st) >> 11) * (2.0 ** -53)) * sm
            acc = 0.0
            nxt = g["V"] - 1
            for c, wc in enumerate(w):
                acc += wc
                if u < acc:
                    nxt = c
                    break
        out.append(nxt)
        cur = nxt
    return out, logits_list


# ---------------------------------------------------------------- Adam / AdamW

def lr_at(g, lr, s, total):
    wu = g["warmup"]
    if wu > 0 and s < wu:
        return lr * (s + 1) / wu
    rt = max(1, total - wu)
    prog = min(1.0, (s - wu) / rt)
    if g["is_adamw_cos"]:
        return lr * (0.1 + 0.45 * (1.0 + math.cos(math.pi * prog)))
    return lr * (1.0 - 0.9 * prog)


def adam_step(ck, grad):
    g = ck["g"]
    s = ck["step"]
    gr = list(grad)
    gn = math.sqrt(sum(v * v for v in gr))
    if gn > 1.0:
        sc = 1.0 / gn
        gr = [v * sc for v in gr]
    is_aw = g["is_adamw_cos"]
    b1 = 0.9
    b2 = 0.95 if is_aw else 0.99
    p1 = p2 = 1.0
    for _ in range(s + 1):
        p1 *= b1
        p2 *= b2
    bc1 = 1.0 / (1.0 - p1)
    bc2 = 1.0 / (1.0 - p2)
    lrs = lr_at(g, ck["lr"], s, ck["total"])
    wd = g["wd"] if is_aw else 0.0

    specs, P = tensor_specs(g)
    gain_mask = [False] * P
    for name, off, r, c in specs:
        if is_gain_name(name):
            for i in range(off, off + r * c):
                gain_mask[i] = True

    params_out = [0.0] * P
    m_out = [0.0] * P
    v_out = [0.0] * P
    params, m, v = ck["params"], ck["m"], ck["v"]
    for i in range(P):
        gi = gr[i]
        mi = b1 * m[i] + (1.0 - b1) * gi
        vi = b2 * v[i] + (1.0 - b2) * gi * gi
        m_out[i] = mi
        v_out[i] = vi
        w = params[i]
        if wd > 0.0 and not gain_mask[i]:
            w = w - lrs * wd * w
        params_out[i] = w - (lrs * mi * bc1) / (math.sqrt(vi * bc2) + 1e-8)

    out = dict(ck)
    out.update(params=params_out, m=m_out, v=v_out, step=s + 1)
    return out


# ---------------------------------------------------------------- comparison helpers

def compare_flat(g, got, ref, rtol=RTOL):
    specs, _ = tensor_specs(g)
    res, ok = {}, True
    for name, off, r, c in specs:
        seg_r = ref[off:off + r * c]
        seg_g = got[off:off + r * c]
        floor = FLOOR_FRAC * max((abs(v) for v in seg_r), default=0.0)
        worst = (0.0, -1, 0.0, 0.0)
        for i, (a, b) in enumerate(zip(seg_g, seg_r)):
            den = max(abs(b), floor)
            err = abs(a - b) / den if den > 0.0 else (0.0 if a == b else math.inf)
            if not (err <= worst[0]) and not (err == 0.0 and worst[1] >= 0):
                worst = (err, i, a, b)
        if worst[1] < 0:
            worst = (0.0, 0, seg_g[0], seg_r[0])
        res[name] = worst
        if not (worst[0] <= rtol):
            ok = False
    return ok, res


def assert_close_scalar(a, b, label, rtol=RTOL):
    den = max(abs(b), 1e-12)
    err = abs(a - b) / den
    if not (err <= rtol):
        raise AssertionError(f"{label}: got {a!r}, ref {b!r}, relerr {err:.3e} > {rtol}")
    return err


def assert_flat_close(g, got, ref, label, rtol=RTOL):
    ok, res = compare_flat(g, got, ref, rtol=rtol)
    worst_name, worst = max(res.items(), key=lambda kv: kv[1][0])
    if not ok:
        bad = {k: v for k, v in res.items() if not (v[0] <= rtol)}
        raise AssertionError(f"{label}: tensors exceeded rtol={rtol}: {bad}")
    return worst[0], worst_name


# ---------------------------------------------------------------- runner helpers

@contextlib.contextmanager
def scratch():
    with tempfile.TemporaryDirectory() as d:
        yield d


def run_bin(bin_path, args, expect_rc=0):
    p = subprocess.run([bin_path] + [str(a) for a in args], capture_output=True, text=True)
    if expect_rc is not None and p.returncode != expect_rc:
        raise AssertionError(
            f"command {[bin_path] + [str(a) for a in args]} exited {p.returncode}, expected {expect_rc}\n"
            f"stdout={p.stdout!r}\nstderr={p.stderr!r}"
        )
    return p


def make_synth_tokens(V, N, seed):
    toks = []
    st = seed & M64
    for i in range(N):
        st = splitmix64(st + i)
        toks.append(st % V)
    return toks


def build_init_args(ck_path, arch_or_flags, cfg, seed, total, lrmicro, extra=False):
    args = [
        "init", ck_path, str(arch_or_flags),
        cfg["V"], cfg["T"], cfg["D"], cfg["H"], cfg["H_kv"], cfg["F"], cfg["L"], cfg["B"],
        seed, total, lrmicro,
    ]
    if extra or cfg.get("NP", 0) > 1:
        args += [cfg["rope_theta_milli"], micro(cfg["wd"]), cfg["warmup"]]
    if cfg.get("NP", 0) > 1:
        args += [cfg["LP"], cfg["LC"], cfg["NP"]]
    return args


# ---------------------------------------------------------------- test suites

def verify_preset(bin_path, name, cfg, seed=42):
    pdef = PRESETS[name]
    g = validate_geom({**cfg, **pdef, "rms_eps": pdef["rms_eps"]})
    _, P = tensor_specs(g)
    total = 20
    lrmicro = 3000  # 3e-3
    lr = lrmicro / 1e6

    with scratch() as d:
        ck_path = os.path.join(d, f"{name}.ck")
        gr_path = os.path.join(d, f"{name}.gr")
        dat_path = os.path.join(d, f"{name}.dat")
        offs_path = os.path.join(d, f"{name}.offs")
        tune_path = os.path.join(d, f"{name}_tune.ck")

        # 1. init + info
        r = run_bin(bin_path, build_init_args(ck_path, name, g, seed, total, lrmicro))
        assert r.stdout.strip() == f"I {P}", f"unexpected init stdout: {r.stdout!r}"
        ck = read_ck4(ck_path)
        ref_params = init_params(g, seed)
        e_init, _ = assert_flat_close(g, ck["params"], ref_params, f"{name} init params")
        assert all(x == 0.0 for x in ck["m"]) and all(x == 0.0 for x in ck["v"])

        r_info = run_bin(bin_path, ["info", ck_path])
        want_info = (
            f"U4 {g['V']} {g['T']} {g['D']} {g['H']} {g['H_kv']} {g['F']} {g['L']} {g['B']} "
            f"{P} {g['arch_flags']} 0 {g['LV']}"
        )
        assert r_info.stdout.strip() == want_info, f"{name} info {r_info.stdout!r} != {want_info!r}"

        # 2. synth data + grad
        data_N = max(g["T"] * 6 + 5, 64)
        toks = make_synth_tokens(g["V"], data_N, seed + 100)
        with open(dat_path, "wb") as f:
            f.write(encode_tokens(toks, g["tok_bytes"]))

        offs = batch_offsets(seed, 0, g["B"], data_N, g["T"])
        with open(offs_path, "wb") as f:
            f.write(struct.pack(f"<{g['B']}Q", *offs))

        r_grad = run_bin(bin_path, ["grad", ck_path, dat_path, offs_path, gr_path])
        parts = r_grad.stdout.strip().split()
        assert len(parts) == 3 and parts[0] == "G", f"{name} bad grad line: {r_grad.stdout!r}"
        got_loss = bits2f(int(parts[1], 16))
        got_micro = int(parts[2])

        seqs = [(toks[o:o + g["T"]], toks[o + 1:o + g["T"] + 1]) for o in offs]
        ref_loss, ref_grad = loss_and_grad(g, ck["params"], seqs)
        e_gloss = assert_close_scalar(got_loss, ref_loss, f"{name} grad loss")
        assert got_micro == micro(ref_loss), f"{name} grad micro {got_micro} != {micro(ref_loss)}"

        gr = read_gr4(gr_path)
        assert gr["step"] == g["B"], f"{name} grad header step={gr['step']} != B={g['B']}"
        e_grad, _ = assert_flat_close(g, gr["grad"], ref_grad, f"{name} grad vector")

        # 3. train 3 steps (check intermediate + final weights, m, v)
        nsteps = 3
        r_tr = run_bin(bin_path, ["train", ck_path, dat_path, nsteps, 1])
        tr_lines = [ln.strip() for ln in r_tr.stdout.strip().splitlines() if ln.strip()]
        assert len(tr_lines) == nsteps

        ref_ck = dict(ck)
        for st_idx in range(nsteps):
            b_offs = batch_offsets(seed, st_idx, g["B"], data_N, g["T"])
            b_seqs = [(toks[o:o + g["T"]], toks[o + 1:o + g["T"] + 1]) for o in b_offs]
            st_loss, st_grad = loss_and_grad(g, ref_ck["params"], b_seqs)
            ref_ck = adam_step(ref_ck, st_grad)
            want_line = f"T {st_idx + 1} {micro(st_loss)}"
            assert tr_lines[st_idx] == want_line, f"{name} train line {tr_lines[st_idx]!r} != {want_line!r}"

        ck_after = read_ck4(ck_path)
        assert ck_after["step"] == nsteps
        e_wt, _ = assert_flat_close(g, ck_after["params"], ref_ck["params"], f"{name} train params")
        e_mt, _ = assert_flat_close(g, ck_after["m"], ref_ck["m"], f"{name} train m")
        e_vt, _ = assert_flat_close(g, ck_after["v"], ref_ck["v"], f"{name} train v")

        # 4. eval
        W = 3
        r_ev = run_bin(bin_path, ["eval", ck_path, dat_path, W])
        ev_parts = r_ev.stdout.strip().split()
        ref_eval = eval_loss(g, ref_ck["params"], toks, W)
        want_ev = ["E", str(nsteps), str(W), str(micro(ref_eval)), str(micro(ref_eval / LN2))]
        assert ev_parts == want_ev, f"{name} eval {ev_parts!r} != {want_ev!r}"

        # 5. kvcmp + gcheck
        r_kv = run_bin(bin_path, ["kvcmp", ck_path, g["T"]])
        assert r_kv.stdout.startswith("KVCMP "), f"{name} kvcmp: {r_kv.stdout!r}"
        kv_parts = r_kv.stdout.strip().split()
        assert kv_parts[2] == "0", f"{name} kvcmp bad={kv_parts[2]}"
        kv_err = bits2f(int(kv_parts[3], 16))
        assert kv_err <= 1e-11, f"{name} kvcmp err={kv_err}"

        run_bin(bin_path, ["gcheck", ck_path, dat_path, 24])

        # 6. sample (greedy tm=0 and stochastic tm=800) + verify logits via full vs KV
        for tm in (0, 800):
            r_sm = run_bin(bin_path, ["sample", ck_path, 777 + tm, g["T"], tm])
            got_toks = [int(x) for x in r_sm.stdout.strip().split()]
            ref_toks, kv_logits = sample_tokens(g, ref_ck["params"], 777 + tm, g["T"], tm)
            assert got_toks == ref_toks, f"{name} sample tm={tm} got={got_toks} != ref={ref_toks}"
            # Also verify final-step KV logits match full sequence forward logits
            P_dict = unpack_params(g, ref_ck["params"])
            PT_dict = {n: transpose(v) for n, v in P_dict.items() if not is_gain_name(n)}
            rope_tabs = build_rope_table(g)
            prefix = [0] + ref_toks[:-1]
            _, full_logits = seq_fwd_bwd(
                g, P_dict, PT_dict, prefix, prefix, 0.0, None, rope_tabs, want_loss=False, want_bwd=False
            )
            max_lerr = max(abs(a - b) / max(abs(b), 1e-12) for a, b in zip(kv_logits[-1], full_logits))
            assert max_lerr <= 1e-12, f"{name} KV vs full logits relerr={max_lerr}"

        # 7. tune resets step, m, v and preserves weights
        run_bin(bin_path, ["tune", ck_path, tune_path, 50, 500])
        ck_t = read_ck4(tune_path)
        assert ck_t["step"] == 0 and ck_t["total"] == 50 and ck_t["lr"] == 0.0005
        assert all(x == 0.0 for x in ck_t["m"]) and all(x == 0.0 for x in ck_t["v"])
        assert_flat_close(g, ck_t["params"], ck_after["params"], f"{name} tune params")

    return dict(
        init=e_init, gloss=e_gloss, grad=e_grad,
        wt=e_wt, m=e_mt, v=e_vt, kv_err=kv_err,
    )


def verify_looped_vs_unrolled(bin_path):
    """Verify looped core (L=3, LP=1, LC=1, NP=3 -> LV=5) against referee and unrolled twin."""
    cfg_loop = dict(
        V=64, T=6, D=16, H=4, H_kv=2, F=32, L=3, B=2,
        arch_flags=PRESETS["qwen3"]["arch_flags"],
        rope_theta_milli=PRESETS["qwen3"]["rope_theta_milli"],
        rms_eps=PRESETS["qwen3"]["rms_eps"],
        wd=PRESETS["qwen3"]["wd"],
        warmup=PRESETS["qwen3"]["warmup"],
        LP=1, LC=1, NP=3,
    )
    g_loop = validate_geom(cfg_loop)
    assert g_loop["LV"] == 5

    cfg_unroll = dict(cfg_loop)
    cfg_unroll.update(L=5, LP=0, LC=0, NP=0)
    g_unroll = validate_geom(cfg_unroll)
    assert g_unroll["LV"] == 5

    seed = 123
    data_N = 48
    toks = make_synth_tokens(g_loop["V"], data_N, seed + 5)

    with scratch() as d:
        ck_loop = os.path.join(d, "loop.ck")
        ck_unroll = os.path.join(d, "unroll.ck")
        gr_loop = os.path.join(d, "loop.gr")
        gr_unroll = os.path.join(d, "unroll.gr")
        dat_path = os.path.join(d, "dat.bin")
        offs_path = os.path.join(d, "offs.bin")

        with open(dat_path, "wb") as f:
            f.write(encode_tokens(toks, g_loop["tok_bytes"]))

        run_bin(bin_path, build_init_args(ck_loop, "qwen3", g_loop, seed, 20, 2000, extra=True))
        ckL = read_ck4(ck_loop)
        assert ckL["g"] == g_loop, f"loop header geom mismatch: {ckL['g']} != {g_loop}"
        ref_init = init_params(g_loop, seed)
        e_init, _ = assert_flat_close(g_loop, ckL["params"], ref_init, "looped init")

        # Train 2 steps on looped core and verify against referee
        run_bin(bin_path, ["train", ck_loop, dat_path, 2, 1])
        ckL_tr = read_ck4(ck_loop)
        ref_ck = dict(ckL)
        for st_idx in range(2):
            offs = batch_offsets(seed, st_idx, g_loop["B"], data_N, g_loop["T"])
            seqs = [(toks[o:o + g_loop["T"]], toks[o + 1:o + g_loop["T"] + 1]) for o in offs]
            _, gr_ref = loss_and_grad(g_loop, ref_ck["params"], seqs)
            ref_ck = adam_step(ref_ck, gr_ref)
        e_wt, _ = assert_flat_close(g_loop, ckL_tr["params"], ref_ck["params"], "looped train params")

        # Construct unrolled twin checkpoint by copying physical layer weights for l=0..4
        PL = unpack_params(g_loop, ckL_tr["params"])
        PU = {
            "wte": [list(row) for row in PL["wte"]],
            "gf": list(PL["gf"]),
            "lmh": [list(row) for row in PL["lmh"]],
        }
        layer_fields = ("g1", "wq", "wk", "wv", "wo", "qn", "kn", "g2", "w1", "wup", "w2")
        for l in range(g_unroll["L"]):
            pl = phys_layer(g_loop, l)
            for fld in layer_fields:
                src = PL[f"{fld}.{pl}"]
                PU[f"{fld}.{l}"] = list(src) if isinstance(src[0], float) else [list(r) for r in src]
        flat_U = pack_params(g_unroll, PU)
        _, P_U = tensor_specs(g_unroll)
        with open(ck_unroll, "wb") as f:
            f.write(encode_ck4(g_unroll, seed, ckL_tr["step"], ckL_tr["total"], ckL_tr["lr"],
                               flat_U, [0.0] * P_U, [0.0] * P_U))

        # Compare eval, kvcmp, and sample between looped core and unrolled twin
        r_ev_l = run_bin(bin_path, ["eval", ck_loop, dat_path, 4])
        r_ev_u = run_bin(bin_path, ["eval", ck_unroll, dat_path, 4])
        assert r_ev_l.stdout == r_ev_u.stdout, f"loop eval {r_ev_l.stdout!r} != unroll {r_ev_u.stdout!r}"

        run_bin(bin_path, ["kvcmp", ck_loop, g_loop["T"]])
        run_bin(bin_path, ["kvcmp", ck_unroll, g_unroll["T"]])

        for tm in (0, 800):
            r_sm_l = run_bin(bin_path, ["sample", ck_loop, 999, g_loop["T"], tm])
            r_sm_u = run_bin(bin_path, ["sample", ck_unroll, 999, g_unroll["T"], tm])
            assert r_sm_l.stdout == r_sm_u.stdout, f"loop sample tm={tm} != unroll"
            ref_toks, _ = sample_tokens(g_loop, ckL_tr["params"], 999, g_loop["T"], tm)
            assert [int(x) for x in r_sm_l.stdout.strip().split()] == ref_toks

        # Compare grad: looped physical layer gradient == sum of unrolled twin layer gradients
        offs = batch_offsets(seed, 2, g_loop["B"], data_N, g_loop["T"])
        with open(offs_path, "wb") as f:
            f.write(struct.pack(f"<{g_loop['B']}Q", *offs))
        r_gl = run_bin(bin_path, ["grad", ck_loop, dat_path, offs_path, gr_loop])
        r_gu = run_bin(bin_path, ["grad", ck_unroll, dat_path, offs_path, gr_unroll])
        loss_l = bits2f(int(r_gl.stdout.strip().split()[1], 16))
        loss_u = bits2f(int(r_gu.stdout.strip().split()[1], 16))
        e_loss = assert_close_scalar(loss_l, loss_u, "loop vs unroll grad loss")

        gL_dict = unpack_params(g_loop, read_gr4(gr_loop)["grad"])
        gU_dict = unpack_params(g_unroll, read_gr4(gr_unroll)["grad"])

        # Sum unrolled layer grads back into looped physical layers
        summed = zeros_like(g_loop)
        for fld in ("wte", "gf", "lmh"):
            summed[fld] = gU_dict[fld]
        for l in range(g_unroll["L"]):
            pl = phys_layer(g_loop, l)
            for fld in layer_fields:
                dst = summed[f"{fld}.{pl}"]
                src = gU_dict[f"{fld}.{l}"]
                if isinstance(dst[0], float):
                    for j in range(len(dst)):
                        dst[j] += src[j]
                else:
                    for r_i in range(len(dst)):
                        for c_i in range(len(dst[r_i])):
                            dst[r_i][c_i] += src[r_i][c_i]

        e_twin_grad, _ = assert_flat_close(
            g_loop, pack_params(g_loop, gL_dict), pack_params(g_loop, summed),
            "looped grad vs summed unrolled twin grad",
        )
        run_bin(bin_path, ["gcheck", ck_loop, dat_path, 24])

    return dict(init=e_init, train_wt=e_wt, twin_loss=e_loss, twin_grad=e_twin_grad)


def verify_u32_vocab(bin_path):
    """Verify V=70000 (u32 token stream, token IDs > 65535)."""
    pdef = PRESETS["qwen3"]
    g = validate_geom(dict(
        V=70000, T=4, D=16, H=4, H_kv=2, F=32, L=1, B=2,
        arch_flags=pdef["arch_flags"],
        rope_theta_milli=pdef["rope_theta_milli"],
        rms_eps=pdef["rms_eps"],
        wd=pdef["wd"],
        warmup=pdef["warmup"],
        LP=0, LC=0, NP=0,
    ))
    assert g["tok_bytes"] == 4
    seed = 555
    data_N = 40
    # Ensure tokens span above 65535 so u32 high bits are exercised
    toks = [((i * 4129 + 65500) % 70000) for i in range(data_N)]
    assert max(toks) >= 65536

    with scratch() as d:
        ck_path = os.path.join(d, "u32.ck")
        gr_path = os.path.join(d, "u32.gr")
        dat_path = os.path.join(d, "u32.dat")
        offs_path = os.path.join(d, "u32.offs")
        with open(dat_path, "wb") as f:
            f.write(encode_tokens(toks, 4))

        run_bin(bin_path, build_init_args(ck_path, "qwen3", g, seed, 10, 2000))
        ck = read_ck4(ck_path)
        ref_init = init_params(g, seed)
        e_init, _ = assert_flat_close(g, ck["params"], ref_init, "u32 init")

        offs = batch_offsets(seed, 0, g["B"], data_N, g["T"])
        with open(offs_path, "wb") as f:
            f.write(struct.pack(f"<{g['B']}Q", *offs))
        r_g = run_bin(bin_path, ["grad", ck_path, dat_path, offs_path, gr_path])
        got_loss = bits2f(int(r_g.stdout.strip().split()[1], 16))
        seqs = [(toks[o:o + g["T"]], toks[o + 1:o + g["T"] + 1]) for o in offs]
        ref_loss, ref_grad = loss_and_grad(g, ck["params"], seqs)
        e_gloss = assert_close_scalar(got_loss, ref_loss, "u32 grad loss")
        e_grad, _ = assert_flat_close(g, read_gr4(gr_path)["grad"], ref_grad, "u32 grad")

        run_bin(bin_path, ["train", ck_path, dat_path, 2, 1])
        ck_tr = read_ck4(ck_path)
        ref_ck = dict(ck)
        for st_idx in range(2):
            bo = batch_offsets(seed, st_idx, g["B"], data_N, g["T"])
            bs = [(toks[o:o + g["T"]], toks[o + 1:o + g["T"] + 1]) for o in bo]
            _, sg = loss_and_grad(g, ref_ck["params"], bs)
            ref_ck = adam_step(ref_ck, sg)
        e_wt, _ = assert_flat_close(g, ck_tr["params"], ref_ck["params"], "u32 train params")

        r_ev = run_bin(bin_path, ["eval", ck_path, dat_path, 3])
        ref_ev = eval_loss(g, ref_ck["params"], toks, 3)
        assert r_ev.stdout.strip() == f"E 2 3 {micro(ref_ev)} {micro(ref_ev / LN2)}"

        run_bin(bin_path, ["kvcmp", ck_path, g["T"]])
        r_sm = run_bin(bin_path, ["sample", ck_path, 333, g["T"], 800])
        ref_toks, _ = sample_tokens(g, ref_ck["params"], 333, g["T"], 800)
        assert [int(x) for x in r_sm.stdout.strip().split()] == ref_toks

    return dict(init=e_init, gloss=e_gloss, grad=e_grad, wt=e_wt)


def verify_past_old_caps(bin_path):
    """Verify V=8192, T=320, D=288, L=20 (past all v0/v1 static caps)."""
    pdef = PRESETS["qwen3"]
    g = validate_geom(dict(
        V=8192, T=320, D=288, H=4, H_kv=1, F=64, L=20, B=1,
        arch_flags=pdef["arch_flags"],
        rope_theta_milli=pdef["rope_theta_milli"],
        rms_eps=pdef["rms_eps"],
        wd=pdef["wd"],
        warmup=pdef["warmup"],
        LP=0, LC=0, NP=0,
    ))
    assert g["tok_bytes"] == 2
    _, P = tensor_specs(g)
    seed = 77
    data_N = 400
    toks = [(i * 131 + 7) % g["V"] for i in range(data_N)]

    with scratch() as d:
        ck_path = os.path.join(d, "big.ck")
        dat_path = os.path.join(d, "big.dat")
        with open(dat_path, "wb") as f:
            f.write(encode_tokens(toks, 2))

        r_init = run_bin(bin_path, build_init_args(ck_path, "qwen3", g, seed, 10, 1000))
        assert r_init.stdout.strip() == f"I {P}"
        ck = read_ck4(ck_path)
        ref_init = init_params(g, seed)
        e_init, _ = assert_flat_close(g, ck["params"], ref_init, "past-old-caps init")

        r_info = run_bin(bin_path, ["info", ck_path])
        assert r_info.stdout.strip() == f"U4 8192 320 288 4 1 64 20 1 {P} 31 0 20"

        # Full T=320 forward pass across all 20 layers vs referee
        r_ev = run_bin(bin_path, ["eval", ck_path, dat_path, 1])
        ref_ev = eval_loss(g, ck["params"], toks, 1)
        want_ev = f"E 0 1 {micro(ref_ev)} {micro(ref_ev / LN2)}"
        assert r_ev.stdout.strip() == want_ev, f"past-old-caps eval {r_ev.stdout!r} != {want_ev!r}"

        # KV cache vs full forward across 8 steps & sampling parity
        run_bin(bin_path, ["kvcmp", ck_path, 8])
        for tm in (0, 800):
            r_sm = run_bin(bin_path, ["sample", ck_path, 888 + tm, 6, tm])
            ref_toks, _ = sample_tokens(g, ck["params"], 888 + tm, 6, tm)
            assert [int(x) for x in r_sm.stdout.strip().split()] == ref_toks

    return dict(P=P, init=e_init, eval_loss=ref_ev)


def verify_negative_controls(bin_path):
    """Verify fail-closed exit codes for corrupt FNV-1a, truncated file, bad flags, etc."""
    with scratch() as d:
        ck_path = os.path.join(d, "valid.ck")
        dat_path = os.path.join(d, "dat.bin")
        with open(dat_path, "wb") as f:
            f.write(bytes([i % 16 for i in range(64)]))
        run_bin(bin_path, ["init", ck_path, "llama", 16, 4, 8, 2, 2, 16, 1, 1, 1, 10, 1000])
        raw = open(ck_path, "rb").read()

        # 1. Corrupt FNV-1a checksum: flip byte in payload -> exit 14
        bad_body = os.path.join(d, "bad_body.ck")
        arr = bytearray(raw)
        arr[CKHDR + 5] ^= 0x01
        open(bad_body, "wb").write(arr)
        run_bin(bin_path, ["info", bad_body], expect_rc=14)

        # Corrupt stored checksum field -> exit 14
        bad_sum = os.path.join(d, "bad_sum.ck")
        arr = bytearray(raw)
        arr[200] ^= 0x80
        open(bad_sum, "wb").write(arr)
        run_bin(bin_path, ["eval", bad_sum, dat_path, 1], expect_rc=14)

        # 2. Truncated file (header < 208, body short, trailing extra bytes)
        trunc_hdr = os.path.join(d, "trunc_hdr.ck")
        open(trunc_hdr, "wb").write(raw[:100])
        run_bin(bin_path, ["info", trunc_hdr], expect_rc=10)

        trunc_body = os.path.join(d, "trunc_body.ck")
        open(trunc_body, "wb").write(raw[:-8])
        run_bin(bin_path, ["info", trunc_body], expect_rc=10)

        extra_bytes = os.path.join(d, "extra.ck")
        open(extra_bytes, "wb").write(raw + b"\x00" * 8)
        run_bin(bin_path, ["info", extra_bytes], expect_rc=15)

        # Bad magic -> exit 11
        bad_magic = os.path.join(d, "bad_magic.ck")
        open(bad_magic, "wb").write(b"OLLMCK03" + raw[8:])
        run_bin(bin_path, ["info", bad_magic], expect_rc=11)

        # Bad P in header -> exit 13
        bad_p = os.path.join(d, "bad_p.ck")
        arr = bytearray(raw)
        struct.pack_into("<Q", arr, 104, 123456)
        open(bad_p, "wb").write(arr)
        run_bin(bin_path, ["info", bad_p], expect_rc=13)

        # 3. Bad flags (exit 12) both on CLI init and on crafted CK file
        bad_flag_cases = [
            ("swiglu+geglu", 2 | 64, 8, 2),
            ("swiglu+gelu", 2 | 32, 8, 2),
            ("geglu+gelu", 64 | 32, 8, 2),
            ("rope_odd_E", 1, 18, 6),  # E=3 with RoPE
        ]
        for tag, fl, D, H in bad_flag_cases:
            bad_init = os.path.join(d, f"bad_init_{tag}.ck")
            run_bin(bin_path, ["init", bad_init, fl, 16, 4, D, H, H, 16, 1, 1, 1, 10, 1000], expect_rc=12)
            # Also craft a CK header with bad flags
            arr = bytearray(raw)
            struct.pack_into("<Q", arr, 24, D)
            struct.pack_into("<Q", arr, 32, H)
            struct.pack_into("<Q", arr, 40, H)
            struct.pack_into("<Q", arr, 112, fl)
            crafted = os.path.join(d, f"crafted_{tag}.ck")
            open(crafted, "wb").write(arr)
            run_bin(bin_path, ["info", crafted], expect_rc=12)

        # Bad loop geometry -> exit 12
        run_bin(
            bin_path,
            ["init", os.path.join(d, "bad_loop1.ck"), 23, 16, 4, 8, 2, 2, 16, 2, 1, 1, 10, 1000, 10000000, 0, 100, 1, 0, 3],
            expect_rc=12,
        )
        run_bin(
            bin_path,
            ["init", os.path.join(d, "bad_loop2.ck"), 23, 16, 4, 8, 2, 2, 16, 2, 1, 1, 10, 1000, 10000000, 0, 100, 2, 1, 3],
            expect_rc=12,
        )

        # Bad token in data (>= V) -> exit 25
        bad_tok_dat = os.path.join(d, "bad_tok.dat")
        open(bad_tok_dat, "wb").write(bytes([20] * 32))
        run_bin(bin_path, ["eval", ck_path, bad_tok_dat, 1], expect_rc=25)

    return 14


def run_all(bin_path):
    t0 = time.time()
    if not os.path.isfile(bin_path) or not os.access(bin_path, os.X_OK):
        raise SystemExit(f"error: ollm-univ binary not executable at {bin_path}")

    preset_cfgs = {
        "llama": dict(V=40, T=6, D=16, H=4, H_kv=2, F=32, L=2, B=2),
        "qwen2": dict(V=40, T=6, D=16, H=4, H_kv=2, F=32, L=2, B=2),
        "qwen3": dict(V=40, T=6, D=16, H=4, H_kv=2, F=32, L=2, B=2),
        "mistral": dict(V=40, T=6, D=16, H=4, H_kv=2, F=32, L=2, B=2),
        "gemma": dict(V=40, T=6, D=16, H=4, H_kv=2, F=32, L=2, B=2),
        "gpt": dict(V=40, T=6, D=16, H=4, H_kv=2, F=32, L=2, B=2),
        "gelu": dict(V=40, T=6, D=16, H=4, H_kv=2, F=32, L=2, B=2),
    }

    for i, (name, cfg) in enumerate(preset_cfgs.items()):
        m = verify_preset(bin_path, name, cfg, seed=42 + i)
        print(
            f"PASS preset={name:8s} init={m['init']:.2e} gloss={m['gloss']:.2e} "
            f"grad={m['grad']:.2e} wt={m['wt']:.2e} m={m['m']:.2e} v={m['v']:.2e} kv={m['kv_err']:.2e}"
        )

    lm = verify_looped_vs_unrolled(bin_path)
    print(
        f"PASS looped_core(LP=1,LC=1,NP=3) init={lm['init']:.2e} train_wt={lm['train_wt']:.2e} "
        f"twin_loss={lm['twin_loss']:.2e} twin_grad={lm['twin_grad']:.2e}"
    )

    u32m = verify_u32_vocab(bin_path)
    print(
        f"PASS u32_vocab(V=70000) init={u32m['init']:.2e} gloss={u32m['gloss']:.2e} "
        f"grad={u32m['grad']:.2e} wt={u32m['wt']:.2e}"
    )

    bigm = verify_past_old_caps(bin_path)
    print(
        f"PASS past_old_caps(V=8192,T=320,D=288,L=20) P={bigm['P']} "
        f"init={bigm['init']:.2e} eval_loss={bigm['eval_loss']:.6f}"
    )

    ncount = verify_negative_controls(bin_path)
    print(f"PASS negative_controls checks={ncount} (corrupt_fnv1a, truncated, bad_flags, bad_magic, bad_loop, bad_tok)")

    print(f"ALL PASS (rtol <= {RTOL:.0e}, elapsed {time.time() - t0:.2f}s)")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(prog="oracle_universal.py")
    ap.add_argument(
        "--bin",
        default=os.environ.get("OLLM_UNIV", DEFAULT_BIN),
        help=f"path to ollm-univ binary (default: {DEFAULT_BIN})",
    )
    args = ap.parse_args(argv)
    return run_all(args.bin)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
