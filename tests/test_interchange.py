#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""tests/test_interchange.py: stdlib-only referee for `ollm-interchange` (native/univ/interchange.om).

Covers the Phase 3 universal weight interchange contract:
  * import-hf  (llama, mistral, qwen2, qwen3, gemma, helix; F32/BF16/F16/F64 safetensors)
  * export-hf  (f32 and bf16 HF directories: config.json + model.safetensors)
  * export-gguf (GGUF v3, 32-byte aligned F32 tensors, llama q/k row permutation)
  * upgrade    (OLLMCK01/02/03 -> OLLMCK04, bit-exact p/m/v)
  * fail-closed exit codes 2, 70, 80-88 and checkpoint refusals 10-15.

Numerical parity is checked against an independent pure-Python f64 HF-semantics forward
(RMSNorm, rotate-half RoPE, GQA causal attention, SwiGLU / tanh-GeGLU, tied/untied heads) and
`ollm-univ eval` / `ollm-univ sample`; legacy upgrade parity against `ollm-macos eval`.

Build:
  TARGET=interchange OUT=/tmp/swarm-interchange sh scripts/build.sh
  TARGET=univ        OUT=/tmp/swarm-interchange sh scripts/build.sh
  TARGET=macos       OUT=/tmp/swarm-interchange sh scripts/build.sh
Run:
  python3 tests/test_interchange.py -v
Env overrides: IX_OUT (default /tmp/swarm-interchange), IX_BIN, UNIV_BIN, LEGACY_BIN.
"""

import json
import math
import os
import random
import shutil
import struct
import subprocess
import tempfile
import unittest
from collections import OrderedDict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("IX_OUT", "/tmp/swarm-interchange")


def _pick(env, name):
    if os.environ.get(env):
        return os.environ[env]
    for d in (OUT, "/tmp/omega-helix-verify-mac"):
        p = os.path.join(d, name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return os.path.join(OUT, name)


IX_BIN = _pick("IX_BIN", "ollm-interchange")
UNIV_BIN = _pick("UNIV_BIN", "ollm-univ")
LEGACY_BIN = _pick("LEGACY_BIN", "ollm-macos")


def have(p):
    return os.path.isfile(p) and os.access(p, os.X_OK)


HAVE_IX = have(IX_BIN)
HAVE_UNIV = have(UNIV_BIN)
HAVE_LEGACY = have(LEGACY_BIN)

MAGIC_CK4 = 3760588441286102095
MAGIC_CK1 = 3544415659172318287
MAGIC_CK2 = 3616473253210246223
MAGIC_CK3 = 3688530847248174159
GGUF_MAGIC = 1179993927
LN2 = 0.6931471805599453


def run(bin_, *args, timeout=300):
    r = subprocess.run([bin_] + [str(a) for a in args], capture_output=True, timeout=timeout)
    return r.returncode, r.stdout.decode("latin-1"), r.stderr.decode("latin-1")


# ----------------------------------------------------------------------------- numerics

def rb(path):
    with open(path, "rb") as f:
        return f.read()


def wb(path, data):
    with open(path, "wb") as f:
        f.write(data)


def jload(path):
    with open(path) as f:
        return json.load(f)


def dbytes(xs):
    return struct.pack("<%dd" % len(xs), *xs)


def f32r(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f64bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def bf16_rne(x):
    """f64 -> bf16 bits, single round-to-nearest-even directly from f64."""
    if x != x:
        return 0x7FC0
    b = f64bits(x)
    sign = b >> 63
    e = (b >> 52) & 0x7FF
    m = b & ((1 << 52) - 1)
    if e == 0x7FF:
        return (sign << 15) | 0x7F80
    if e == 0:
        return sign << 15
    E = e - 1023
    M = (1 << 52) | m
    if E < -126:  # bf16 subnormal, unit 2^-133; value = M * 2^(E-52)
        sh = (52 - E) - 133
        if sh > 60:
            return sign << 15
        q, rem = M >> sh, M & ((1 << sh) - 1)
        half = 1 << (sh - 1)
        if rem > half or (rem == half and (q & 1)):
            q += 1
        return (sign << 15) | q
    q, rem = M >> 45, M & ((1 << 45) - 1)
    half = 1 << 44
    if rem > half or (rem == half and (q & 1)):
        q += 1
    if q == (1 << 8):
        q >>= 1
        E += 1
    if E > 127:
        return (sign << 15) | 0x7F80
    return (sign << 15) | ((E + 127) << 7) | (q & 0x7F)


def bf16_val(bits):
    return struct.unpack("<f", struct.pack("<I", (bits & 0xFFFF) << 16))[0]


DTSIZE = {"F32": 4, "F64": 8, "F16": 2, "BF16": 2}


def encode(vals, dt):
    n = len(vals)
    if dt == "F32":
        return struct.pack("<%df" % n, *vals)
    if dt == "F64":
        return struct.pack("<%dd" % n, *vals)
    if dt == "F16":
        return struct.pack("<%de" % n, *vals)
    if dt == "BF16":
        return struct.pack("<%dH" % n, *[bf16_rne(v) for v in vals])
    raise ValueError(dt)


def decode(raw, dt):
    n = len(raw) // DTSIZE[dt]
    if dt == "F32":
        return list(struct.unpack("<%df" % n, raw))
    if dt == "F64":
        return list(struct.unpack("<%dd" % n, raw))
    if dt == "F16":
        return list(struct.unpack("<%de" % n, raw))
    if dt == "BF16":
        return [bf16_val(h) for h in struct.unpack("<%dH" % n, raw)]
    raise ValueError(dt)


def quantize(vals, dt):
    return decode(encode(vals, dt), dt)


def urmic(v):
    """round half away from zero of the single f64 product v*1e6 (umath.om urmic)."""
    x = v * 1000000.0
    q = int(x)
    fr = x - q
    if fr >= 0.5:
        q += 1
    if fr <= -0.5:
        q -= 1
    return q


def fnv1a(data):
    h = 14695981039346656037
    for c in data:
        h = ((h ^ c) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def transpose(flat, R, C):
    """[R][C] row-major -> [C][R] row-major."""
    out = [0.0] * (R * C)
    for r in range(R):
        base = r * C
        for c in range(C):
            out[c * R + r] = flat[base + c]
    return out


# ----------------------------------------------------------------------------- safetensors

def st_bytes(tensors, metadata=None, align=8, raw_header=None, extra_tail=b""):
    """tensors: list of (name, dtype, shape, raw). Returns file bytes."""
    if raw_header is None:
        hdr = OrderedDict()
        if metadata is not None:
            hdr["__metadata__"] = metadata
        off = 0
        for name, dt, shape, raw in tensors:
            hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(raw)]}
            off += len(raw)
        hj = json.dumps(hdr, separators=(",", ":")).encode()
    else:
        hj = raw_header
    hj += b" " * ((-(8 + len(hj))) % align)
    return struct.pack("<Q", len(hj)) + hj + b"".join(t[3] for t in tensors) + extra_tail


def st_parse(b):
    n = struct.unpack_from("<Q", b, 0)[0]
    hraw = b[8:8 + n]
    seen = []

    def hook(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate key")
        return OrderedDict(pairs)

    hdr = json.loads(hraw.decode("utf-8"), object_pairs_hook=hook)
    meta = hdr.pop("__metadata__", None)
    ents = []
    for name, e in hdr.items():
        ents.append((name, e["dtype"], e["shape"], e["data_offsets"]))
    return n, hraw, meta, ents, b[8 + n:]


# ----------------------------------------------------------------------------- HF model generation

GEOM = dict(D=16, H=4, HKV=2, F=24, L=2, T=16)


class HFModel:
    """A synthetic HF checkpoint (values are the f64 decoding of the stored dtype)."""

    def __init__(self, kind, V, D=16, H=4, HKV=2, F=24, L=2, T=16, tied=None, dtype="F32",
                 seed=1, theta=None, eps=None, dtype_of=None):
        self.kind, self.V, self.D, self.H, self.HKV, self.F, self.L, self.T = kind, V, D, H, HKV, F, L, T
        self.E = D // H
        self.KVD = HKV * self.E
        if tied is None:
            tied = kind == "gemma"
        self.tied = tied
        self.theta = theta if theta is not None else {"qwen2": 1000000.0, "qwen3": 1000000.0}.get(kind, 10000.0)
        self.eps = eps if eps is not None else {"llama": 1e-5, "mistral": 1e-5}.get(kind, 1e-6)
        self.gemma = kind == "gemma"
        self.qkn = kind == "qwen3"
        self.t = OrderedDict()   # name -> (shape, vals)
        self.dt = {}
        rng = random.Random(seed * 7919 + V)

        def add(name, shape, std, base=0.0):
            n = 1
            for s in shape:
                n *= s
            vals = []
            for _ in range(n):
                v = base + rng.gauss(0.0, std)
                if base == 0.0 and abs(v) < 1e-3:
                    v = 1e-3 if v >= 0 else -1e-3
                vals.append(v)
            d = dtype_of(name) if dtype_of else dtype
            self.t[name] = (list(shape), quantize(vals, d))
            self.dt[name] = d

        nb = 0.0 if self.gemma else 1.0
        add("model.embed_tokens.weight", (V, D), 0.15)
        for l in range(L):
            p = "model.layers.%d." % l
            add(p + "input_layernorm.weight", (D,), 0.1, nb)
            add(p + "self_attn.q_proj.weight", (D, D), 0.5)
            add(p + "self_attn.k_proj.weight", (self.KVD, D), 0.5)
            add(p + "self_attn.v_proj.weight", (self.KVD, D), 0.5)
            add(p + "self_attn.o_proj.weight", (D, D), 0.5)
            if self.qkn:
                add(p + "self_attn.q_norm.weight", (self.E,), 0.1, 1.0)
                add(p + "self_attn.k_norm.weight", (self.E,), 0.1, 1.0)
            add(p + "post_attention_layernorm.weight", (D,), 0.1, nb)
            add(p + "mlp.gate_proj.weight", (F, D), 0.5)
            add(p + "mlp.up_proj.weight", (F, D), 0.5)
            add(p + "mlp.down_proj.weight", (D, F), 0.5)
        add("model.norm.weight", (D,), 0.1, nb)
        if not tied:
            add("lm_head.weight", (V, D), 0.3)

    def config(self):
        arch = {"llama": "LlamaForCausalLM", "mistral": "MistralForCausalLM", "qwen2": "Qwen2ForCausalLM",
                "qwen3": "Qwen3ForCausalLM", "gemma": "GemmaForCausalLM"}[self.kind]
        c = OrderedDict()
        c["architectures"] = [arch]
        c["model_type"] = self.kind
        c["vocab_size"] = self.V
        c["hidden_size"] = self.D
        c["intermediate_size"] = self.F
        c["num_hidden_layers"] = self.L
        c["num_attention_heads"] = self.H
        c["num_key_value_heads"] = self.HKV
        c["head_dim"] = self.E
        c["max_position_embeddings"] = self.T
        c["rms_norm_eps"] = self.eps
        c["rope_theta"] = self.theta
        c["tie_word_embeddings"] = self.tied
        c["hidden_act"] = "gelu_pytorch_tanh" if self.gemma else "silu"
        if self.gemma:
            c["hidden_activation"] = "gelu_pytorch_tanh"
        c["attention_bias"] = False
        c["rope_scaling"] = None
        c["torch_dtype"] = "float32"
        if self.kind == "mistral":
            c["sliding_window"] = None
        if self.kind == "qwen2":
            c["use_sliding_window"] = False
            c["sliding_window"] = 32768
        return c

    def st_tensors(self, names=None):
        out = []
        for name, (shape, vals) in self.t.items():
            if names is not None and name not in names:
                continue
            out.append((name, self.dt[name], shape, encode(vals, self.dt[name])))
        return out

    def write(self, d, cfg=None, tensors=None, st=None):
        os.makedirs(d, exist_ok=True)
        cp, sp = os.path.join(d, "config.json"), os.path.join(d, "model.safetensors")
        with open(cp, "w") as f:
            if isinstance(cfg, (bytes, str)):
                f.write(cfg if isinstance(cfg, str) else cfg.decode())
            else:
                json.dump(cfg if cfg is not None else self.config(), f, indent=2)
        with open(sp, "wb") as f:
            f.write(st if st is not None else st_bytes(tensors if tensors is not None else self.st_tensors(),
                                                      metadata={"format": "pt"}))
        return cp, sp

    # ---- flags & expected CK
    def flags(self):
        if self.kind == "gemma":
            return 1 | 64 | 16 | 4
        f = 1 | 2 | 16
        if self.kind == "qwen3":
            f |= 8
        if not self.tied:
            f |= 4
        return f

    def expected_p(self):
        D, F, V, E, KVD = self.D, self.F, self.V, self.E, self.KVD
        g = self.gemma
        t = lambda n: self.t[n][1]
        sq = math.sqrt(D)
        nrm = (lambda v: [1.0 + x for x in v]) if g else (lambda v: list(v))
        emb = t("model.embed_tokens.weight")
        p = [x * sq for x in emb] if g else list(emb)
        for l in range(self.L):
            pre = "model.layers.%d." % l
            p += nrm(t(pre + "input_layernorm.weight"))
            p += transpose(t(pre + "self_attn.q_proj.weight"), D, D)
            p += transpose(t(pre + "self_attn.k_proj.weight"), KVD, D)
            p += transpose(t(pre + "self_attn.v_proj.weight"), KVD, D)
            p += transpose(t(pre + "self_attn.o_proj.weight"), D, D)
            if self.qkn:
                p += list(t(pre + "self_attn.q_norm.weight"))
                p += list(t(pre + "self_attn.k_norm.weight"))
            p += nrm(t(pre + "post_attention_layernorm.weight"))
            p += transpose(t(pre + "mlp.gate_proj.weight"), F, D)
            p += transpose(t(pre + "mlp.up_proj.weight"), F, D)
            p += transpose(t(pre + "mlp.down_proj.weight"), D, F)
        p += nrm(t("model.norm.weight"))
        if g:
            p += list(emb)
        elif not self.tied:
            p += list(t("lm_head.weight"))
        return p

    # ---- independent HF-semantics f64 forward
    def logits(self, toks):
        D, H, HKV, E, F, V = self.D, self.H, self.HKV, self.E, self.F, self.V
        W = {n: v for n, (s, v) in self.t.items()}
        g = self.gemma
        eps = self.eps

        def rows(name, R, C):
            v = W[name]
            return [v[r * C:(r + 1) * C] for r in range(R)]

        def lin(M, x):
            return [sum(a * b for a, b in zip(r, x)) for r in M]

        def rms(x, w, gem):
            s = sum(v * v for v in x) / len(x)
            inv = 1.0 / math.sqrt(s + eps)
            if gem:
                return [x[i] * inv * (1.0 + w[i]) for i in range(len(x))]
            return [x[i] * inv * w[i] for i in range(len(x))]

        half = E // 2
        inv_freq = [1.0 / (self.theta ** ((2 * i) / E)) for i in range(half)]

        def rope(v, pos):
            out = list(v)
            for i in range(half):
                c, s = math.cos(pos * inv_freq[i]), math.sin(pos * inv_freq[i])
                x1, x2 = v[i], v[i + half]
                out[i] = x1 * c - x2 * s
                out[i + half] = x2 * c + x1 * s
            return out

        def act(x):
            if g:
                return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
            return x / (1.0 + math.exp(-x))

        T = len(toks)
        emb = W["model.embed_tokens.weight"]
        sq = math.sqrt(D)
        h = []
        for tk in toks:
            row = emb[tk * D:(tk + 1) * D]
            h.append([x * sq for x in row] if g else list(row))
        rep = H // HKV
        for l in range(self.L):
            pre = "model.layers.%d." % l
            Wq = rows(pre + "self_attn.q_proj.weight", D, D)
            Wk = rows(pre + "self_attn.k_proj.weight", HKV * E, D)
            Wv = rows(pre + "self_attn.v_proj.weight", HKV * E, D)
            Wo = rows(pre + "self_attn.o_proj.weight", D, D)
            Wg = rows(pre + "mlp.gate_proj.weight", F, D)
            Wu = rows(pre + "mlp.up_proj.weight", F, D)
            Wd = rows(pre + "mlp.down_proj.weight", D, F)
            g1 = W[pre + "input_layernorm.weight"]
            g2 = W[pre + "post_attention_layernorm.weight"]
            Q, K, Vv = [], [], []
            for t in range(T):
                a = rms(h[t], g1, g)
                q, k, v = lin(Wq, a), lin(Wk, a), lin(Wv, a)
                qh = [q[i * E:(i + 1) * E] for i in range(H)]
                kh = [k[i * E:(i + 1) * E] for i in range(HKV)]
                vh = [v[i * E:(i + 1) * E] for i in range(HKV)]
                if self.qkn:
                    qn = W[pre + "self_attn.q_norm.weight"]
                    kn = W[pre + "self_attn.k_norm.weight"]
                    qh = [rms(x, qn, False) for x in qh]
                    kh = [rms(x, kn, False) for x in kh]
                Q.append([rope(x, t) for x in qh])
                K.append([rope(x, t) for x in kh])
                Vv.append(vh)
            nh = []
            for t in range(T):
                o = []
                for hi in range(H):
                    kv = hi // rep
                    sc = [sum(a * b for a, b in zip(Q[t][hi], K[j][kv])) / math.sqrt(E) for j in range(t + 1)]
                    mx = max(sc)
                    ex = [math.exp(s - mx) for s in sc]
                    z = sum(ex)
                    acc = [0.0] * E
                    for j in range(t + 1):
                        w = ex[j] / z
                        for c in range(E):
                            acc[c] += w * Vv[j][kv][c]
                    o += acc
                ao = lin(Wo, o)
                hm = [h[t][i] + ao[i] for i in range(D)]
                b = rms(hm, g2, g)
                gt, up = lin(Wg, b), lin(Wu, b)
                f = [act(gt[i]) * up[i] for i in range(F)]
                dn = lin(Wd, f)
                nh.append([hm[i] + dn[i] for i in range(D)])
            h = nh
        gf = W["model.norm.weight"]
        head = W["model.embed_tokens.weight"] if (self.tied or g) else W["lm_head.weight"]
        Hd = [head[c * D:(c + 1) * D] for c in range(V)]
        out = []
        for t in range(T):
            z = rms(h[t], gf, g)
            out.append([sum(a * b for a, b in zip(r, z)) for r in Hd])
        return out

    def seq_loss(self, x, y):
        lg = self.logits(x)
        tot = 0.0
        for t, row in enumerate(lg):
            mx = max(row)
            tot += math.log(sum(math.exp(v - mx) for v in row)) - (row[y[t]] - mx)
        return tot / len(x)

    def eval_line(self, data, T, W, step=0):
        s = 0.0
        for i in range(W):
            o = i * T
            s += self.seq_loss(data[o:o + T], data[o + 1:o + T + 1])
        ml = s / W
        return ml, (step, W, urmic(ml), urmic(ml / LN2))

    def greedy(self, n):
        seq, out, safe = [0], [], True
        for _ in range(n):
            row = self.logits(seq)[-1]
            best = 0
            for c in range(len(row)):
                if row[c] > row[best]:
                    best = c
            srt = sorted(row, reverse=True)
            if srt[0] - srt[1] < 1e-9:
                safe = False
            out.append(best)
            seq.append(best)
            if not safe:
                break
        return out, safe


def tok_width(V):
    return 1 if V <= 256 else (2 if V <= 65535 else 4)


def write_tokens(path, toks, V):
    w = tok_width(V)
    fmt = {1: "B", 2: "H", 4: "I"}[w]
    with open(path, "wb") as f:
        f.write(struct.pack("<%d%s" % (len(toks), fmt), *toks))


# ----------------------------------------------------------------------------- OLLMCK04

CK4_FIELDS = ["V", "T", "D", "H", "HKV", "F", "L", "B", "seed", "step", "total", "lr_bits", "P",
              "flags", "theta_milli", "eps_bits", "wd_bits", "warmup", "LP", "LC", "NP"]


def ck4_P(V, T, D, H, HKV, F, L, flags, **_):
    E = D // H
    KVD = HKV * E
    rope, gated, untied, qkn = flags & 1, (flags & 66) != 0, flags & 4, flags & 8
    p = V * D + (0 if rope else T * D)
    lay = D + D * D + 2 * D * KVD + D * D + (2 * E if qkn else 0) + D + D * F + (D * F if gated else 0) + F * D
    p += L * lay + D
    if untied:
        p += V * D
    return p


def ck4_segments(h):
    """Ordered (name, rows, cols) list of the OLLMCK04 parameter layout (Helix [in][out])."""
    V, T, D, H, HKV, F, L, fl = h["V"], h["T"], h["D"], h["H"], h["HKV"], h["F"], h["L"], h["flags"]
    E = D // H
    KVD = HKV * E
    segs = [("wte", V, D)]
    if not fl & 1:
        segs.append(("wpe", T, D))
    for l in range(L):
        segs += [("g1.%d" % l, 1, D), ("wq.%d" % l, D, D), ("wk.%d" % l, D, KVD), ("wv.%d" % l, D, KVD),
                 ("wo.%d" % l, D, D)]
        if fl & 8:
            segs += [("qn.%d" % l, 1, E), ("kn.%d" % l, 1, E)]
        segs.append(("g2.%d" % l, 1, D))
        segs.append(("w1.%d" % l, D, F))
        if fl & 66:
            segs.append(("wup.%d" % l, D, F))
        segs.append(("w2.%d" % l, F, D))
    segs.append(("gf", 1, D))
    if fl & 4:
        segs.append(("lmh", V, D))
    return segs


def ck4_read(path, check_fnv=True):
    b = rb(path)
    assert len(b) >= 208, "short CK04"
    magic = struct.unpack_from("<Q", b, 0)[0]
    assert magic == MAGIC_CK4, "bad CK04 magic %d" % magic
    vals = struct.unpack_from("<21Q", b, 8)
    h = dict(zip(CK4_FIELDS, vals))
    h["reserved"] = struct.unpack_from("<3Q", b, 176)
    h["fnv"] = struct.unpack_from("<Q", b, 200)[0]
    P = h["P"]
    assert len(b) == 208 + 24 * P, "CK04 size %d != %d" % (len(b), 208 + 24 * P)
    if check_fnv:
        assert fnv1a(b[208:]) == h["fnv"], "CK04 FNV mismatch"
    h["praw"] = b[208:208 + 8 * P]
    h["mraw"] = b[208 + 8 * P:208 + 16 * P]
    h["vraw"] = b[208 + 16 * P:]
    h["p"] = list(struct.unpack_from("<%dd" % P, b, 208))
    h["raw"] = b
    return h


def ck4_write(path, hdr, p):
    P = len(p)
    payload = dbytes(p) + b"\0" * (16 * P)
    vals = [hdr[k] for k in CK4_FIELDS[:12]] + [P] + [hdr[k] for k in CK4_FIELDS[13:]]
    head = struct.pack("<Q", MAGIC_CK4) + struct.pack("<21Q", *vals) + b"\0" * 24 + struct.pack("<Q", fnv1a(payload))
    with open(path, "wb") as f:
        f.write(head + payload)


def ck4_split(h):
    out, o = OrderedDict(), 0
    for name, r, c in ck4_segments(h):
        out[name] = (r, c, h["p"][o:o + r * c])
        o += r * c
    assert o == h["P"], "layout P %d != header P %d" % (o, h["P"])
    return out


def lp_lc_np(h):
    return h["LP"], h["LC"], h["NP"]


def export_type(h, seg):
    fl = h["flags"]
    if h["NP"] <= 1 and fl & 1 and fl & 2 and fl & 8:
        return "qwen3"
    if h["NP"] <= 1 and fl & 1 and fl & 2:
        return "llama"
    if h["NP"] <= 1 and fl & 1 and fl & 64 and fl & 4:
        sq = math.sqrt(h["D"])
        wte, lmh = seg["wte"][2], seg["lmh"][2]
        if dbytes(wte) == dbytes([x * sq for x in lmh]):
            return "gemma"
    return "helix"


def ck_to_hf(h, gguf=False):
    """Expected export: (model_type, [(hf_name, hf_shape, values)]) from a parsed CK04.
    gguf=True keeps gemma norms as the stored 1+w (GGUF rule)."""
    seg = ck4_split(h)
    mt = export_type(h, seg)
    V, T, D, H, HKV, F, L, fl = h["V"], h["T"], h["D"], h["H"], h["HKV"], h["F"], h["L"], h["flags"]
    E = D // H
    KVD = HKV * E
    gem = mt == "gemma"
    nrm = (lambda v: [x - 1.0 for x in v]) if (gem and not gguf) else (lambda v: list(v))
    out = []
    out.append(("model.embed_tokens.weight", [V, D], list(seg["lmh"][2]) if gem else list(seg["wte"][2])))
    if not fl & 1:
        out.append(("model.embed_positions.weight", [T, D], list(seg["wpe"][2])))
    for l in range(L):
        pre = "model.layers.%d." % l
        out.append((pre + "input_layernorm.weight", [D], nrm(seg["g1.%d" % l][2])))
        out.append((pre + "self_attn.q_proj.weight", [D, D], transpose(seg["wq.%d" % l][2], D, D)))
        out.append((pre + "self_attn.k_proj.weight", [KVD, D], transpose(seg["wk.%d" % l][2], D, KVD)))
        out.append((pre + "self_attn.v_proj.weight", [KVD, D], transpose(seg["wv.%d" % l][2], D, KVD)))
        out.append((pre + "self_attn.o_proj.weight", [D, D], transpose(seg["wo.%d" % l][2], D, D)))
        if fl & 8:
            out.append((pre + "self_attn.q_norm.weight", [E], list(seg["qn.%d" % l][2])))
            out.append((pre + "self_attn.k_norm.weight", [E], list(seg["kn.%d" % l][2])))
        out.append((pre + "post_attention_layernorm.weight", [D], nrm(seg["g2.%d" % l][2])))
        if fl & 66:
            out.append((pre + "mlp.gate_proj.weight", [F, D], transpose(seg["w1.%d" % l][2], D, F)))
            out.append((pre + "mlp.up_proj.weight", [F, D], transpose(seg["wup.%d" % l][2], D, F)))
        else:
            out.append((pre + "mlp.up_proj.weight", [F, D], transpose(seg["w1.%d" % l][2], D, F)))
        out.append((pre + "mlp.down_proj.weight", [D, F], transpose(seg["w2.%d" % l][2], F, D)))
    out.append(("model.norm.weight", [D], nrm(seg["gf"][2])))
    if fl & 4 and not gem:
        out.append(("lm_head.weight", [V, D], list(seg["lmh"][2])))
    return mt, out


GGUF_NAME = [
    ("model.embed_tokens.weight", "token_embd.weight"),
    ("model.embed_positions.weight", "position_embd.weight"),
    ("input_layernorm.weight", "attn_norm.weight"),
    ("self_attn.q_proj.weight", "attn_q.weight"),
    ("self_attn.k_proj.weight", "attn_k.weight"),
    ("self_attn.v_proj.weight", "attn_v.weight"),
    ("self_attn.o_proj.weight", "attn_output.weight"),
    ("self_attn.q_norm.weight", "attn_q_norm.weight"),
    ("self_attn.k_norm.weight", "attn_k_norm.weight"),
    ("post_attention_layernorm.weight", "ffn_norm.weight"),
    ("mlp.gate_proj.weight", "ffn_gate.weight"),
    ("mlp.up_proj.weight", "ffn_up.weight"),
    ("mlp.down_proj.weight", "ffn_down.weight"),
    ("model.norm.weight", "output_norm.weight"),
    ("lm_head.weight", "output.weight"),
]


def gguf_name(hf):
    if hf.startswith("model.layers."):
        rest = hf[len("model.layers."):]
        n, tail = rest.split(".", 1)
        for a, b in GGUF_NAME:
            if tail == a:
                return "blk.%s.%s" % (n, b)
        raise KeyError(hf)
    for a, b in GGUF_NAME:
        if hf == a:
            return b
    raise KeyError(hf)


def llama_permute(vals, rows_total, cols, n_head):
    E = rows_total // n_head
    out = [0.0] * len(vals)
    for hh in range(n_head):
        for a in range(2):
            for b in range(E // 2):
                dst = hh * E + 2 * b + a
                src = hh * E + a * (E // 2) + b
                out[dst * cols:(dst + 1) * cols] = vals[src * cols:(src + 1) * cols]
    return out


# ----------------------------------------------------------------------------- GGUF parser

def gguf_parse(b):
    pos = [0]

    def rd(fmt):
        v = struct.unpack_from("<" + fmt, b, pos[0])
        pos[0] += struct.calcsize("<" + fmt)
        return v[0] if len(v) == 1 else v

    def rstr():
        n = rd("Q")
        s = b[pos[0]:pos[0] + n]
        pos[0] += n
        return s.decode("utf-8")

    scal = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def rval(t):
        if t in scal:
            return rd(scal[t])
        if t == 8:
            return rstr()
        if t == 9:
            at, n = rd("I"), rd("Q")
            return [rval(at) for _ in range(n)]
        raise ValueError("gguf kv type %d" % t)

    magic, ver, nt, nkv = rd("I"), rd("I"), rd("Q"), rd("Q")
    kvs = OrderedDict()
    for _ in range(nkv):
        k = rstr()
        t = rd("I")
        kvs[k] = (t, rval(t))
    tens = []
    for _ in range(nt):
        name = rstr()
        nd = rd("I")
        dims = [rd("Q") for _ in range(nd)]
        typ = rd("I")
        off = rd("Q")
        tens.append((name, dims, typ, off))
    return dict(magic=magic, version=ver, nt=nt, nkv=nkv, kvs=kvs, tensors=tens, info_end=pos[0])


# ----------------------------------------------------------------------------- legacy CK01/02/03

def legacy_read(path):
    b = rb(path)
    mg = struct.unpack_from("<Q", b, 0)[0]
    hl = {MAGIC_CK1: 112, MAGIC_CK2: 120, MAGIC_CK3: 144}[mg]
    ver = {MAGIC_CK1: 1, MAGIC_CK2: 2, MAGIC_CK3: 3}[mg]
    V, T, D, H, F, L, B = struct.unpack_from("<7Q", b, 8)
    seed, step, total, lr, P = struct.unpack_from("<5Q", b, 64)
    G = struct.unpack_from("<Q", b, 104)[0] if hl >= 120 else 1
    LP = LC = NP = 0
    if hl == 144:
        LP, LC, NP = struct.unpack_from("<3Q", b, 112)
    return dict(ver=ver, hl=hl, V=V, T=T, D=D, H=H, F=F, L=L, B=B, seed=seed, step=step, total=total,
                lr_bits=lr, P=P, G=G, LP=LP, LC=LC, NP=NP, payload=b[hl:], raw=b)


# ============================================================================= tests

class Base(unittest.TestCase):
    maxDiff = 4000

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ixtest-")

    def tearDown(self):
        if os.environ.get("IX_KEEP") != "1":
            shutil.rmtree(self.tmp, ignore_errors=True)

    def path(self, *p):
        return os.path.join(self.tmp, *p)

    def ix(self, *args, rc=0, mkdir=True):
        # contract (#12): export-hf requires OUT_DIR to exist; a missing OUT_DIR is exit 70
        if mkdir and len(args) >= 3 and args[0] == "export-hf":
            os.makedirs(args[2], exist_ok=True)
        r, o, e = run(IX_BIN, *args)
        if rc is not None:
            self.assertEqual(r, rc, "ollm-interchange %s -> rc %d (want %d)\nstdout=%r\nstderr=%r" %
                             (" ".join(map(str, args)), r, rc, o, e))
        return o

    def univ(self, *args, rc=0):
        r, o, e = run(UNIV_BIN, *args)
        self.assertEqual(r, rc, "ollm-univ %s -> rc %d\nstdout=%r stderr=%r" % (" ".join(map(str, args)), r, o, e))
        return o

    def assertBitsEqual(self, got, want, what):
        if dbytes(got) != dbytes(want):
            self.assertEqual(len(got), len(want), "%s: length %d != %d" % (what, len(got), len(want)))
            for i, (a, b) in enumerate(zip(got, want)):
                if f64bits(a) != f64bits(b):
                    self.fail("%s: first bit mismatch at %d: got %r want %r" % (what, i, a, b))

    def eval_check(self, model, ck, nwin=3, seed=5):
        rng = random.Random(seed + model.V)
        T = model.T
        data = [rng.randrange(model.V) for _ in range(nwin * T + 1 + 7)]
        dp = self.path("data-%d.bin" % seed)
        write_tokens(dp, data, model.V)
        out = self.univ("eval", ck, dp, nwin).split()
        self.assertEqual(out[0], "E")
        ml, (step, W, lm, bm) = model.eval_line(data, T, nwin)
        self.assertEqual(int(out[1]), step)
        self.assertEqual(int(out[2]), W)
        self.assertLessEqual(abs(int(out[3]) - lm), 1, "loss_micro engine %s vs ref %d (%.12f)" % (out[3], lm, ml))
        self.assertLessEqual(abs(int(out[4]) - bm), 1, "bpb_micro engine %s vs ref %d" % (out[4], bm))
        return int(out[3]), lm

    def sample_check(self, model, ck, n=None):
        n = n or model.T
        got = [int(x) for x in self.univ("sample", ck, 1, n, 0).split()]
        want, safe = model.greedy(n)
        self.assertEqual(got[:len(want)], want, "greedy sample engine %s vs ref %s" % (got, want))
        if safe:
            self.assertEqual(len(got), n)
        return got

    def hdr_expect(self, model, h, T=None, B=1, total=1000, lrmicro=100):
        T = T or model.T
        exp = dict(V=model.V, T=T, D=model.D, H=model.H, HKV=model.HKV, F=model.F, L=model.L, B=B,
                   seed=0, step=0, total=total, lr_bits=f64bits(lrmicro / 1000000.0), flags=model.flags(),
                   theta_milli=int(round(model.theta * 1000)), eps_bits=f64bits(model.eps))
        for k, v in exp.items():
            self.assertEqual(h[k], v, "CK04 header %s: got %r want %r" % (k, h[k], v))
        self.assertEqual(h["P"], ck4_P(**{k: h[k] for k in ("V", "T", "D", "H", "HKV", "F", "L", "flags")}))
        self.assertEqual((h["LP"], h["LC"]), (0, 0))
        self.assertIn(h["NP"], (0, 1))
        self.assertEqual(h["mraw"], b"\0" * (8 * h["P"]), "Adam m not zero")
        self.assertEqual(h["vraw"], b"\0" * (8 * h["P"]), "Adam v not zero")


# ----------------------------------------------------------------------------- oracle self-check

@unittest.skipUnless(HAVE_UNIV, "ollm-univ not built (%s)" % UNIV_BIN)
class T00OracleSelfCheck(Base):
    """Validates the Python HF-semantics reference against ollm-univ using a Python-written CK04
    (independent of ollm-interchange)."""

    def _one(self, kind, V, tied=None):
        m = HFModel(kind, V, tied=tied, seed=3)
        hdr = dict(V=m.V, T=m.T, D=m.D, H=m.H, HKV=m.HKV, F=m.F, L=m.L, B=1, seed=0, step=0, total=1000,
                   lr_bits=f64bits(1e-4), flags=m.flags(), theta_milli=int(round(m.theta * 1000)),
                   eps_bits=f64bits(m.eps), wd_bits=0, warmup=100, LP=0, LC=0, NP=0)
        ck = self.path("py.ck")
        ck4_write(ck, hdr, m.expected_p())
        info = self.univ("info", ck).split()
        self.assertEqual(int(info[9]), len(m.expected_p()))
        self.eval_check(m, ck)
        self.sample_check(m, ck)

    def test_llama_tied(self):
        self._one("llama", 96, tied=True)

    def test_qwen3_untied_u16(self):
        self._one("qwen3", 300, tied=False)

    def test_gemma(self):
        self._one("gemma", 80)


# ----------------------------------------------------------------------------- import-hf

@unittest.skipUnless(HAVE_IX, "ollm-interchange not built (%s)" % IX_BIN)
class T10Import(Base):

    def do_import(self, m, extra=(), sub="hf", **kw):
        cp, sp = m.write(self.path(sub), **kw)
        ck = self.path(sub + ".ck")
        out = self.ix("import-hf", cp, sp, ck, *extra)
        return ck, out

    def check_stdout(self, out, m, h):
        want = "I %d %d %d %d %d %d %d %d %d\n" % (m.V, h["T"], m.D, m.H, m.HKV, m.F, m.L, h["P"], m.flags())
        self.assertEqual(out, want)

    def test_dtypes(self):
        for dt in ("F32", "BF16", "F16", "F64"):
            with self.subTest(dtype=dt):
                m = HFModel("llama", 96, dtype=dt, seed=11)
                ck, out = self.do_import(m, sub="d" + dt)
                h = ck4_read(ck)
                self.hdr_expect(m, h)
                self.check_stdout(out, m, h)
                self.assertBitsEqual(h["p"], m.expected_p(), "llama %s params" % dt)

    def test_mixed_dtypes(self):
        cyc = ["F32", "BF16", "F16", "F64"]
        m = HFModel("qwen3", 72, tied=False, seed=12,
                    dtype_of=lambda n: cyc[sum(map(ord, n)) % 4])
        self.assertEqual(len(set(m.dt.values())), 4)
        ck, out = self.do_import(m)
        h = ck4_read(ck)
        self.hdr_expect(m, h)
        self.assertBitsEqual(h["p"], m.expected_p(), "mixed-dtype params")

    def _parity(self, kind, V, tied=None, dtype="F32", sample=True):
        m = HFModel(kind, V, tied=tied, dtype=dtype, seed=21)
        ck, out = self.do_import(m)
        h = ck4_read(ck)
        self.hdr_expect(m, h)
        self.check_stdout(out, m, h)
        self.assertBitsEqual(h["p"], m.expected_p(), "%s params" % kind)
        self.eval_check(m, ck)
        if sample:
            self.sample_check(m, ck)
        return m, ck

    @unittest.skipUnless(HAVE_UNIV, "ollm-univ missing")
    def test_llama_parity(self):
        self._parity("llama", 96, tied=True)

    @unittest.skipUnless(HAVE_UNIV, "ollm-univ missing")
    def test_llama_untied_bf16_parity(self):
        self._parity("llama", 64, tied=False, dtype="BF16")

    @unittest.skipUnless(HAVE_UNIV, "ollm-univ missing")
    def test_qwen3_parity(self):
        self._parity("qwen3", 300, tied=False)

    @unittest.skipUnless(HAVE_UNIV, "ollm-univ missing")
    def test_qwen3_tied_parity(self):
        self._parity("qwen3", 128, tied=True, sample=False)

    @unittest.skipUnless(HAVE_UNIV, "ollm-univ missing")
    def test_gemma_parity(self):
        self._parity("gemma", 80)

    @unittest.skipUnless(HAVE_UNIV, "ollm-univ missing")
    def test_qwen2_parity(self):
        self._parity("qwen2", 90, tied=False, sample=False)

    @unittest.skipUnless(HAVE_UNIV, "ollm-univ missing")
    def test_mistral_parity(self):
        self._parity("mistral", 70, tied=False, sample=False)

    def test_explicit_args(self):
        m = HFModel("llama", 64, tied=True, seed=4)
        ck, out = self.do_import(m, extra=(8, 2, 500, 3000))
        h = ck4_read(ck)
        self.hdr_expect(m, h, T=8, B=2, total=500, lrmicro=3000)
        self.check_stdout(out, m, h)
        self.assertBitsEqual(h["p"], m.expected_p(), "explicit-args params")

    def test_default_T(self):
        m = HFModel("llama", 64, tied=True, seed=4)
        c = m.config()
        c["max_position_embeddings"] = 4096
        ck, _ = self.do_import(m, sub="big", cfg=c)
        self.assertEqual(ck4_read(ck)["T"], 256)
        c2 = m.config()
        del c2["max_position_embeddings"]
        ck2, _ = self.do_import(m, sub="none", cfg=c2)
        self.assertEqual(ck4_read(ck2)["T"], 256)

    def test_optional_fields_defaults(self):
        m = HFModel("llama", 64, tied=False, seed=5, eps=1e-6, theta=10000.0, HKV=4)
        c = m.config()
        for k in ("num_key_value_heads", "rms_norm_eps", "rope_theta", "tie_word_embeddings", "head_dim",
                  "rope_scaling", "attention_bias", "torch_dtype", "architectures"):
            del c[k]
        ck, _ = self.do_import(m, cfg=c)
        h = ck4_read(ck)
        self.hdr_expect(m, h)
        self.assertBitsEqual(h["p"], m.expected_p(), "defaults params")

    def test_ignored_and_zero_bias_tensors(self):
        m = HFModel("qwen2", 64, tied=True, seed=6)
        ref_ck, _ = self.do_import(m, sub="plain")
        extra = list(m.st_tensors())
        D, KVD = m.D, m.KVD
        for l in range(m.L):
            pre = "model.layers.%d." % l
            extra.append((pre + "self_attn.q_proj.bias", "F32", [D], encode([0.0] * D, "F32")))
            extra.append((pre + "self_attn.k_proj.bias", "BF16", [KVD], encode([0.0] * KVD, "BF16")))
            extra.append((pre + "self_attn.v_proj.bias", "F32", [KVD], encode([-0.0] * KVD, "F32")))
            extra.append((pre + "self_attn.rotary_emb.inv_freq", "F32", [m.E // 2], encode([1.0, 0.01], "F32")))
        extra.append(("lm_head.weight", "F32", [m.V, m.D], encode([0.5] * (m.V * m.D), "F32")))  # tied: ignored
        ck, _ = self.do_import(m, sub="extra", tensors=extra)
        self.assertEqual(rb(ck), rb(ref_ck))

    def test_reordered_tensors_and_no_metadata(self):
        m = HFModel("llama", 64, tied=False, seed=7)
        ref_ck, _ = self.do_import(m, sub="plain")
        ts = list(reversed(m.st_tensors()))
        st = st_bytes(ts, metadata=None)
        ck, _ = self.do_import(m, sub="rev", st=st)
        self.assertEqual(rb(ck), rb(ref_ck))


# ----------------------------------------------------------------------------- export-hf / round trip

@unittest.skipUnless(HAVE_IX, "ollm-interchange not built (%s)" % IX_BIN)
class T20ExportHF(Base):

    CFG_KEYS = ["architectures", "model_type", "vocab_size", "hidden_size", "intermediate_size",
                "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim",
                "max_position_embeddings", "rms_norm_eps", "rope_theta", "tie_word_embeddings", "hidden_act",
                "attention_bias", "rope_scaling", "torch_dtype", "helix_arch_flags", "helix_loop_lp",
                "helix_loop_lc", "helix_loop_np"]

    def check_export(self, ck, outdir, dt="f32", args=None):
        h = ck4_read(ck)
        mt, want = ck_to_hf(h)
        o = self.ix("export-hf", ck, outdir, *([dt] if args is None else args))
        st = rb(os.path.join(outdir, "model.safetensors"))
        cfg = jload((os.path.join(outdir, "config.json")))
        self.assertEqual(o, "XH %d %d\n" % (len(want), len(st)))
        # --- safetensors structure
        n, hraw, meta, ents, data = st_parse(st)
        self.assertTrue(2 <= n <= 983040)
        self.assertEqual((8 + n) % 8, 0, "header not padded to 8")
        js = hraw.rstrip(b" ")
        self.assertTrue(js.startswith(b'{"__metadata__"'), "first key must be __metadata__: %r" % hraw[:40])
        self.assertTrue(all(c == 0x20 for c in hraw[len(js):]), "header padding must be spaces")
        self.assertEqual(meta, {"format": "pt"})
        self.assertEqual([e[0] for e in ents], [w[0] for w in want], "tensor order")
        sdt = "F32" if dt == "f32" else "BF16"
        off = 0
        for (name, edt, shape, (b0, b1)), (wn, wshape, wvals) in zip(ents, want):
            self.assertEqual(edt, sdt, name)
            self.assertEqual(shape, wshape, name)
            self.assertEqual(b0, off, "%s not contiguous" % name)
            self.assertEqual(b1 - b0, len(wvals) * DTSIZE[sdt], name)
            raw = data[b0:b1]
            if sdt == "F32":
                exp = struct.pack("<%df" % len(wvals), *wvals)
            else:
                exp = struct.pack("<%dH" % len(wvals), *[bf16_rne(v) for v in wvals])
            if raw != exp:
                got = decode(raw, sdt)
                ref = decode(exp, sdt)
                bad = [i for i in range(len(got)) if dbytes([got[i]]) != dbytes([ref[i]])]
                self.fail("%s %s data mismatch at %d of %d (first idx %d: got %r want %r from f64 %r)" %
                          (name, sdt, len(bad), len(got), bad[0], got[bad[0]], ref[bad[0]], wvals[bad[0]]))
            off = b1
        self.assertEqual(len(data), off, "trailing data after last tensor")
        # --- config.json
        for k in self.CFG_KEYS:
            self.assertIn(k, cfg, "config.json missing %s" % k)
        D, H = h["D"], h["H"]
        self.assertEqual(cfg["model_type"], mt)
        self.assertIsInstance(cfg["architectures"], list)
        exp = dict(vocab_size=h["V"], hidden_size=D, intermediate_size=h["F"], num_hidden_layers=h["L"],
                   num_attention_heads=H, num_key_value_heads=h["HKV"], head_dim=D // H,
                   max_position_embeddings=h["T"], attention_bias=False, rope_scaling=None,
                   torch_dtype="float32" if dt == "f32" else "bfloat16", helix_arch_flags=h["flags"],
                   helix_loop_lp=h["LP"], helix_loop_lc=h["LC"], helix_loop_np=h["NP"])
        for k, v in exp.items():
            self.assertEqual(cfg[k], v, "config %s" % k)
        self.assertEqual(f64bits(float(cfg["rms_norm_eps"])), h["eps_bits"], "rms_norm_eps not bit-exact")
        self.assertEqual(float(cfg["rope_theta"]), h["theta_milli"] / 1000.0)
        tie = {"gemma": True}.get(mt, not (h["flags"] & 4))
        self.assertEqual(cfg["tie_word_embeddings"], tie)
        if mt in ("llama", "qwen3"):
            self.assertEqual(cfg["hidden_act"], "silu")
        self.assertIsInstance(cfg["hidden_act"], str)
        return h, mt, want, st, cfg

    def _roundtrip_hf(self, kind, V, tied=None, sub="rt"):
        m = HFModel(kind, V, tied=tied, seed=31)
        cp, sp = m.write(self.path(sub))
        ck1 = self.path(sub + "1.ck")
        self.ix("import-hf", cp, sp, ck1)
        out = self.path(sub + "-out")
        h, mt, want, st, cfg = self.check_export(ck1, out)
        self.assertEqual(mt, kind if kind != "mistral" and kind != "qwen2" else "llama")
        # byte-identical tensors vs the F32 source
        src = {name: raw for name, _, _, raw in m.st_tensors()}
        n, hraw, meta, ents, data = st_parse(st)
        self.assertEqual(sorted(e[0] for e in ents), sorted(src), "exported tensor set != source set")
        for name, _, _, (b0, b1) in ents:
            self.assertEqual(data[b0:b1], src[name], "%s not byte-identical to F32 source" % name)
        # re-import -> byte-identical CK
        ck2 = self.path(sub + "2.ck")
        self.ix("import-hf", os.path.join(out, "config.json"), os.path.join(out, "model.safetensors"), ck2)
        self.assertEqual(rb(ck2), rb(ck1), "re-imported CK not byte-identical")
        # bf16 export
        self.check_export(ck1, self.path(sub + "-bf16"), "bf16")
        return m, ck1

    def test_llama_roundtrip(self):
        self._roundtrip_hf("llama", 96, tied=True)

    def test_qwen3_roundtrip(self):
        self._roundtrip_hf("qwen3", 300, tied=False)

    def test_gemma_roundtrip(self):
        self._roundtrip_hf("gemma", 80)

    def test_default_dtype_is_f32(self):
        m = HFModel("llama", 64, tied=False, seed=32)
        cp, sp = m.write(self.path("a"))
        ck = self.path("a.ck")
        self.ix("import-hf", cp, sp, ck)
        self.check_export(ck, self.path("o1"), args=[])
        self.check_export(ck, self.path("o2"), "f32")
        for f in ("config.json", "model.safetensors"):
            self.assertEqual(rb(self.path("o1", f)), rb(self.path("o2", f)))

    # ---- helix / univ-init checkpoints (f64 weights -> f32 export -> f32-rounded reimport)
    def _helix(self, arch, V, D, H, HKV, F, L, T, extra=(), want_mt="helix"):
        if not HAVE_UNIV:
            self.skipTest("ollm-univ missing")
        ck = self.path("u.ck")
        self.univ("init", ck, arch, V, T, D, H, HKV, F, L, 1, 0, 1000, 100, *extra)
        h0 = ck4_read(ck)
        out = self.path("hx")
        h, mt, want, st, cfg = self.check_export(ck, out)
        self.assertEqual(mt, want_mt, "export model_type for arch %s flags %d" % (arch, h0["flags"]))
        if mt == "helix":
            self.assertIn("helix", json.dumps(cfg["architectures"]).lower() + cfg["model_type"])
        ck2 = self.path("u2.ck")
        o = self.ix("import-hf", os.path.join(out, "config.json"), os.path.join(out, "model.safetensors"), ck2)
        h2 = ck4_read(ck2)
        for k in ("V", "T", "D", "H", "HKV", "F", "L", "P", "flags", "theta_milli", "eps_bits", "LP", "LC", "NP"):
            self.assertEqual(h2[k], h0[k], "reimport header %s" % k)
        self.assertEqual(o, "I %d %d %d %d %d %d %d %d %d\n" % (V, T, D, H, HKV, F, L, h0["P"], h0["flags"]))
        self.assertBitsEqual(h2["p"], [f32r(x) for x in h0["p"]], "reimport p == f32(orig p)")
        # idempotent second export
        out2 = self.path("hx2")
        self.ix("export-hf", ck2, out2)
        self.assertEqual(rb(os.path.join(out2, "model.safetensors")), st)
        self.assertEqual(jload((os.path.join(out2, "config.json"))), cfg)
        # bf16 straight from f64 weights (single RNE)
        self.check_export(ck, self.path("hxb"), "bf16")
        return h0

    def test_helix_gpt_loop(self):
        self._helix("gpt", 256, 16, 2, 2, 32, 3, 16, extra=(10000000, 0, 100, 1, 1, 3))

    def test_helix_swiglu_wpe_untied(self):
        self._helix("6", 70, 16, 4, 2, 24, 2, 12)

    def test_helix_rope_relu(self):
        self._helix("1", 64, 16, 4, 4, 20, 2, 8)

    def test_helix_gelu_rope_qkn_untied(self):
        self._helix("45", 64, 16, 4, 2, 20, 2, 8)

    def test_llama_preset_with_loop_is_helix(self):
        self._helix("llama", 64, 16, 4, 2, 24, 3, 8, extra=(500000000, 100000, 100, 1, 1, 2))

    def test_tied_gemma_preset_is_helix(self):
        self._helix("gemma", 64, 16, 4, 2, 24, 2, 8)

    def test_univ_llama_preset_is_llama(self):
        self._helix("llama", 64, 16, 4, 2, 24, 2, 8, want_mt="llama")

    def test_univ_qwen3_preset_is_qwen3(self):
        self._helix("qwen3", 300, 16, 4, 2, 24, 2, 8, want_mt="qwen3")


# ----------------------------------------------------------------------------- export-gguf

@unittest.skipUnless(HAVE_IX, "ollm-interchange not built (%s)" % IX_BIN)
class T30GGUF(Base):

    def check_gguf(self, ck):
        h = ck4_read(ck)
        mt, want = ck_to_hf(h, gguf=True)
        gp = self.path("m.gguf")
        o = self.ix("export-gguf", ck, gp)
        b = rb(gp)
        self.assertEqual(o, "XG %d %d\n" % (len(want), len(b)))
        g = gguf_parse(b)
        self.assertEqual(g["magic"], GGUF_MAGIC)
        self.assertEqual(b[:4], b"GGUF")
        self.assertEqual(g["version"], 3)
        self.assertEqual(g["nt"], len(want))
        self.assertEqual(g["nkv"], 20)
        kv = g["kvs"]
        arch = mt
        D, H, E = h["D"], h["H"], h["D"] // h["H"]
        U32, F32, STR = 4, 6, 8
        exp = OrderedDict([
            ("general.architecture", (STR, arch)), ("general.name", (STR, "omega-helix")),
            ("general.alignment", (U32, 32)), ("general.file_type", (U32, 0)),
            (arch + ".vocab_size", (U32, h["V"])), (arch + ".context_length", (U32, h["T"])),
            (arch + ".embedding_length", (U32, D)), (arch + ".block_count", (U32, h["L"])),
            (arch + ".feed_forward_length", (U32, h["F"])), (arch + ".attention.head_count", (U32, H)),
            (arch + ".attention.head_count_kv", (U32, h["HKV"])), (arch + ".attention.key_length", (U32, E)),
            (arch + ".attention.value_length", (U32, E)),
            (arch + ".attention.layer_norm_rms_epsilon", (F32, f32r(struct.unpack("<d", struct.pack("<Q", h["eps_bits"]))[0]))),
            (arch + ".rope.freq_base", (F32, f32r(h["theta_milli"] / 1000.0))),
            (arch + ".rope.dimension_count", (U32, E)),
            ("helix.arch_flags", (U32, h["flags"])), ("helix.loop.lp", (U32, h["LP"])),
            ("helix.loop.lc", (U32, h["LC"])), ("helix.loop.np", (U32, h["NP"])),
        ])
        self.assertEqual(set(kv), set(exp), "GGUF KV key set")
        for k, v in exp.items():
            self.assertEqual(kv[k], v, "GGUF KV %s" % k)
        # tensors
        ds = (g["info_end"] + 31) // 32 * 32
        self.assertEqual([t[0] for t in g["tensors"]], [gguf_name(w[0]) for w in want], "GGUF tensor names/order")
        end = 0
        for (name, dims, typ, off), (hn, shape, vals) in zip(g["tensors"], want):
            self.assertEqual(typ, 0, name)
            self.assertEqual(dims, list(reversed(shape)), "%s ggml dims" % name)
            self.assertEqual(off % 32, 0, "%s offset not 32-aligned" % name)
            self.assertGreaterEqual(off, end, "%s overlaps previous tensor" % name)
            self.assertLess(off - end, 32 if end else 1, "%s gap beyond alignment padding" % name)
            if arch == "llama" and hn.endswith("q_proj.weight"):
                vals = llama_permute(vals, shape[0], shape[1], H)
            if arch == "llama" and hn.endswith("k_proj.weight"):
                vals = llama_permute(vals, shape[0], shape[1], h["HKV"])
            raw = b[ds + off: ds + off + 4 * len(vals)]
            self.assertEqual(raw, struct.pack("<%df" % len(vals), *vals), "%s F32 data" % name)
            end = off + 4 * len(vals)
        self.assertEqual(len(b), ds + end, "GGUF must end at end of last tensor")
        return g, mt

    def _hf(self, kind, V, tied=None, want=None):
        m = HFModel(kind, V, tied=tied, seed=41)
        cp, sp = m.write(self.path("hf"))
        ck = self.path("a.ck")
        self.ix("import-hf", cp, sp, ck)
        g, mt = self.check_gguf(ck)
        self.assertEqual(mt, want or kind)
        return g

    def test_llama_tied(self):
        g = self._hf("llama", 96, tied=True)
        self.assertNotIn("output.weight", [t[0] for t in g["tensors"]])

    def test_llama_untied_permute(self):
        g = self._hf("llama", 64, tied=False)
        self.assertIn("output.weight", [t[0] for t in g["tensors"]])

    def test_qwen3(self):
        self._hf("qwen3", 300, tied=False)

    def test_gemma(self):
        g = self._hf("gemma", 80)
        self.assertNotIn("output.weight", [t[0] for t in g["tensors"]])

    @unittest.skipUnless(HAVE_UNIV, "ollm-univ missing")
    def test_helix_loop_wpe(self):
        ck = self.path("u.ck")
        self.univ("init", ck, "gpt", 256, 16, 16, 2, 2, 32, 3, 1, 0, 1000, 100, 10000000, 0, 100, 1, 1, 3)
        g, mt = self.check_gguf(ck)
        self.assertEqual(mt, "helix")
        self.assertIn("position_embd.weight", [t[0] for t in g["tensors"]])

    @unittest.skipUnless(HAVE_UNIV, "ollm-univ missing")
    def test_helix_untied_swiglu(self):
        ck = self.path("u.ck")
        self.univ("init", ck, "6", 70, 12, 16, 4, 2, 24, 2, 1, 0, 1000, 100)
        g, mt = self.check_gguf(ck)
        self.assertEqual(mt, "helix")
        self.assertIn("output.weight", [t[0] for t in g["tensors"]])


# ----------------------------------------------------------------------------- upgrade legacy

LEG = dict(V=256, T=16, D=16, H=2, F=32, B=2)


@unittest.skipUnless(HAVE_IX and HAVE_LEGACY and HAVE_UNIV, "needs ollm-interchange, ollm-macos, ollm-univ")
class T40Upgrade(Base):

    def corpus(self):
        rng = random.Random(9)
        w = ("the quick brown fox jumps over a lazy dog while omega helix trains seven small "
             "models on bytes and every story ends here").split()
        tr, he = self.path("train.bin"), self.path("held.bin")
        wb(tr, " ".join(rng.choice(w) for _ in range(3000)).encode())
        wb(he, " ".join(rng.choice(w) for _ in range(600)).encode())
        return tr, he

    def _up(self, ver, L, extra, steps=3):
        tr, he = self.corpus()
        ck = self.path("leg.ck")
        r, o, e = run(LEGACY_BIN, "init", ck, LEG["V"], LEG["T"], LEG["D"], LEG["H"], LEG["F"], L, LEG["B"],
                      7, 50, 3000, *extra)
        self.assertEqual(r, 0, o + e)
        r, o, e = run(LEGACY_BIN, "train", ck, tr, steps, steps)
        self.assertEqual(r, 0, o + e)
        lh = legacy_read(ck)
        self.assertEqual(lh["ver"], ver)
        self.assertEqual(lh["step"], steps)
        up = self.path("up.ck")
        o = self.ix("upgrade", ck, up)
        self.assertEqual(o, "U %d %d\n" % (ver, lh["P"]))
        h = ck4_read(up)
        exp = dict(V=256, T=LEG["T"], D=LEG["D"], H=LEG["H"], HKV=LEG["H"], F=LEG["F"], L=L, B=LEG["B"], seed=7,
                   step=steps, total=50, lr_bits=lh["lr_bits"], P=lh["P"], flags=0, eps_bits=f64bits(1e-5))
        for k, v in exp.items():
            self.assertEqual(h[k], v, "upgraded header %s" % k)
        if ver == 3:
            self.assertEqual(lp_lc_np(h), (lh["LP"], lh["LC"], lh["NP"]))
        else:
            self.assertEqual((h["LP"], h["LC"]), (0, 0))
            self.assertIn(h["NP"], (0, 1))
        self.assertEqual(h["raw"][208:], lh["payload"], "p/m/v not copied bit-exact")
        for W in (1, 4, 10):
            r1, o1, e1 = run(LEGACY_BIN, "eval", ck, he, W)
            self.assertEqual(r1, 0, o1 + e1)
            o2 = self.univ("eval", up, he, W)
            self.assertEqual(o2, o1, "eval parity W=%d legacy %r univ %r" % (W, o1, o2))
        # upgraded checkpoints export as helix models
        mt, _ = ck_to_hf(h)
        self.assertEqual(mt, "helix")
        return ck, up

    def test_ck01(self):
        self._up(1, 2, ())

    def test_ck02(self):
        self._up(2, 2, (2,))

    def test_ck03_loop(self):
        self._up(3, 3, (1, 1, 1, 3))

    def test_upgraded_trains_in_univ(self):
        ck, up = self._up(1, 2, (), steps=2)
        tr, _ = self.corpus()
        o = self.univ("train", up, tr, 3, 1)
        self.assertTrue(o.startswith("T 3 "), o)

    def _continued(self, ver, L, extra):
        """Training continued after upgrade matches the legacy engine line for line."""
        ck, up = self._up(ver, L, extra, steps=3)
        tr, he = self.corpus()
        r, o1, e1 = run(LEGACY_BIN, "train", ck, tr, 8, 1)
        self.assertEqual(r, 0, o1 + e1)
        o2 = self.univ("train", up, tr, 8, 1)
        self.assertEqual(len(o1.splitlines()), 5, o1)
        self.assertEqual(o2, o1, "continued train T lines: legacy %r univ %r" % (o1, o2))
        r, e1, _ = run(LEGACY_BIN, "eval", ck, he, 6)
        self.assertEqual(r, 0)
        self.assertEqual(self.univ("eval", up, he, 6), e1, "eval after continued training")
        # the two engines sum gradients in different orders, so after continued training the
        # parameters agree to rounding (not bit for bit); the printed T and E lines are identical
        P = legacy_read(ck)["P"]
        a = struct.unpack_from("<%dd" % P, rb(up), 208)
        b = struct.unpack_from("<%dd" % P, legacy_read(ck)["payload"], 0)
        mx = max(abs(x - y) / max(abs(x), abs(y), 1e-30) for x, y in zip(a, b))
        self.assertLess(mx, 1e-9, "params after continued training: max rel diff %.3e" % mx)

    def test_continued_training_ck01(self):
        self._continued(1, 2, ())

    def test_continued_training_ck03_loop(self):
        self._continued(3, 3, (1, 1, 1, 3))

    def test_upgrade_refusals(self):
        ck, up = self._up(1, 2, (), steps=1)
        b = bytearray(rb(ck))
        bad = self.path("bad.ck")
        c = bytearray(b)
        c[0] ^= 0xFF
        wb(bad, c)
        self.ix("upgrade", bad, self.path("x.ck"), rc=11)
        c = bytearray(b)
        c[-5] ^= 0x01
        wb(bad, c)
        self.ix("upgrade", bad, self.path("x.ck"), rc=14)
        wb(bad, b[:-8])
        self.ix("upgrade", bad, self.path("x.ck"), rc=10)
        wb(bad, b + b"\0")
        self.ix("upgrade", bad, self.path("x.ck"), rc=15)
        wb(bad, b[:50])
        self.ix("upgrade", bad, self.path("x.ck"), rc=10)
        self.ix("upgrade", up, self.path("x.ck"), rc=11)  # CK04 is not a legacy checkpoint
        self.assertFalse(os.path.exists(self.path("x.ck")))


# ----------------------------------------------------------------------------- negative controls

@unittest.skipUnless(HAVE_IX, "ollm-interchange not built (%s)" % IX_BIN)
class T50Refusals(Base):

    def setUp(self):
        super().setUp()
        self.m = HFModel("llama", 64, tied=False, seed=51)

    def imp(self, rc, cfg=None, tensors=None, st=None, model=None, sub="n"):
        m = model or self.m
        cp, sp = m.write(self.path(sub), cfg=cfg, tensors=tensors, st=st)
        ck = self.path(sub + ".ck")
        self.ix("import-hf", cp, sp, ck, rc=rc)
        if rc != 0:
            self.assertFalse(os.path.exists(ck), "refused import left %s behind" % ck)

    def cfg(self, **kw):
        c = self.m.config()
        for k, v in kw.items():
            if v is KeyError:
                c.pop(k, None)
            else:
                c[k] = v
        return c

    def test_positive_control(self):
        self.imp(0)

    def test_usage_2(self):
        self.ix(rc=2)
        self.ix("bogus", rc=2)
        self.ix("import-hf", "a", "b", rc=2)
        self.ix("import-hf", "a", "b", "c", "1", rc=2)
        self.ix("export-hf", "a", rc=2)
        self.ix("export-gguf", "a", rc=2)
        self.ix("upgrade", "a", rc=2)

    def test_io_70(self):
        cp, sp = self.m.write(self.path("ok"))
        self.ix("import-hf", self.path("nope.json"), sp, self.path("x.ck"), rc=70)
        self.ix("import-hf", cp, self.path("nope.st"), self.path("x.ck"), rc=70)
        self.ix("import-hf", cp, sp, self.path("nodir", "sub", "x.ck"), rc=70)
        ck = self.path("ok.ck")
        self.ix("import-hf", cp, sp, ck)
        self.ix("export-gguf", ck, self.path("nodir", "sub", "x.gguf"), rc=70)
        self.ix("export-hf", ck, self.path("nodir", "sub"), rc=70, mkdir=False)
        self.ix("export-hf", self.path("missing.ck"), self.path("o"), rc=70)

    def test_json_80(self):
        self.imp(80, cfg='{"model_type": "llama", "vocab_size": 64,')
        self.imp(80, cfg="not json at all")
        self.imp(80, cfg='{"model_type": "llama"} trailing')

    def test_config_81(self):
        for k in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads"):
            with self.subTest(missing=k):
                self.imp(81, cfg=self.cfg(**{k: KeyError}), sub="m" + k)
        self.imp(81, cfg=self.cfg(model_type="bert"), sub="bert")
        self.imp(81, cfg=self.cfg(model_type=KeyError), sub="nomt")
        self.imp(81, cfg=self.cfg(vocab_size="64"), sub="strv")
        self.imp(81, cfg=self.cfg(hidden_size=-16), sub="neg")
        cp, sp = self.m.write(self.path("arr"), cfg="[1, 2, 3]")  # valid JSON, not an object
        r, o, e = run(IX_BIN, "import-hf", cp, sp, self.path("arr.ck"))
        self.assertIn(r, (80, 81), "non-object config.json -> rc %d" % r)

    def test_unsupported_82(self):
        self.imp(82, cfg=self.cfg(rope_scaling={"type": "linear", "factor": 2.0}), sub="rs")
        self.imp(82, cfg=self.cfg(head_dim=8), sub="hd")
        self.imp(82, cfg=self.cfg(hidden_act="gelu"), sub="act")
        mi = HFModel("mistral", 64, tied=False, seed=52)
        c = mi.config()
        c["sliding_window"] = 4
        self.imp(82, cfg=c, model=mi, sub="sw")
        c["sliding_window"] = 4096
        self.imp(0, cfg=c, model=mi, sub="swok")
        ts = self.m.st_tensors() + [("model.layers.0.self_attn.q_proj.bias", "F32", [16],
                                     encode([0.0] * 15 + [0.25], "F32"))]
        self.imp(82, tensors=ts, sub="bias")

    def test_header_83(self):
        ts = self.m.st_tensors()
        good = st_bytes(ts, metadata={"format": "pt"})
        n = struct.unpack_from("<Q", good, 0)[0]
        body = good[8 + n:]
        hdr = json.loads(good[8:8 + n].decode())
        cases = {}
        cases["N1"] = struct.pack("<Q", 1) + b"{" + body
        cases["Nbig"] = struct.pack("<Q", 983041) + good[8:]
        cases["badjson"] = struct.pack("<Q", n) + (b"{" + b" " * (n - 1)) + body
        h2 = json.loads(json.dumps(hdr))
        h2["model.norm.weight"]["dtype"] = 7
        cases["badentry"] = st_bytes(ts, raw_header=json.dumps(h2).encode())
        h3 = json.loads(json.dumps(hdr))
        e = h3["model.norm.weight"]
        e["shape"] = [8]  # size != numel*dtype
        cases["size"] = st_bytes(ts, raw_header=json.dumps(h3).encode())
        h4 = json.loads(json.dumps(hdr))
        e = h4["model.norm.weight"]
        e["data_offsets"] = [e["data_offsets"][0] - 4, e["data_offsets"][1] - 4]  # overlap
        cases["overlap"] = st_bytes(ts, raw_header=json.dumps(h4).encode())
        cases["trailing"] = good + b"\0" * 8
        hh = json.loads(json.dumps(hdr))
        last = max((v["data_offsets"][1] for k, v in hh.items() if k != "__metadata__"))
        # hole: shift lm_head (last tensor) up by 4 bytes and append 4 bytes
        lk = [k for k, v in hh.items() if k != "__metadata__" and v["data_offsets"][1] == last][0]
        hh[lk]["data_offsets"] = [hh[lk]["data_offsets"][0] + 4, last + 4]
        lraw = [t for t in ts if t[0] == lk][0][3]
        hole_body = body[:len(body) - len(lraw)] + b"\0\0\0\0" + lraw
        hj = json.dumps(hh).encode()
        hj += b" " * ((-(8 + len(hj))) % 8)
        cases["hole"] = struct.pack("<Q", len(hj)) + hj + hole_body
        js = good[8:8 + n].rstrip(b" ")
        dup = js[:-1] + b',"model.norm.weight":' + json.dumps(hdr["model.norm.weight"]).encode() + b"}"
        dup += b" " * ((-(8 + len(dup))) % 8)
        cases["duplicate"] = struct.pack("<Q", len(dup)) + dup + body
        for name, st in cases.items():
            with self.subTest(case=name):
                self.imp(83, st=st, sub="h" + name)

    def test_tensor_84(self):
        ts = self.m.st_tensors()
        self.imp(84, tensors=[t for t in ts if t[0] != "model.norm.weight"], sub="missing")
        self.imp(84, tensors=[t for t in ts if t[0] != "lm_head.weight"], sub="nolmh")
        self.imp(84, tensors=ts + [("model.foo.weight", "F32", [2], encode([1.0, 2.0], "F32"))], sub="unk")
        self.imp(84, tensors=ts + [("model.layers.2.input_layernorm.weight", "F32", [16],
                                    encode([1.0] * 16, "F32"))], sub="layer2")
        bad = [(n, ("I32" if n == "model.norm.weight" else d), s, r) for n, d, s, r in ts]
        self.imp(84, tensors=bad, sub="i32")
        bad = [(n, ("F8_E4M3" if n == "model.norm.weight" else d), s,
                (r[:16] if n == "model.norm.weight" else r)) for n, d, s, r in ts]
        self.imp(84, tensors=bad, sub="f8")

    def test_shape_85(self):
        ts = self.m.st_tensors()
        k = "model.layers.0.self_attn.k_proj.weight"
        self.imp(85, tensors=[(n, d, ([16, 8] if n == k else s), r) for n, d, s, r in ts], sub="kT")
        g = "model.layers.1.mlp.gate_proj.weight"
        self.imp(85, tensors=[(n, d, ([16, 24] if n == g else s), r) for n, d, s, r in ts], sub="gT")
        e = "model.embed_tokens.weight"
        self.imp(85, tensors=[(n, d, ([16, 64] if n == e else s), r) for n, d, s, r in ts], sub="eT")
        nn = "model.norm.weight"
        self.imp(85, tensors=[(n, d, ([4, 4] if n == nn else s), r) for n, d, s, r in ts], sub="n2d")

    def test_nonfinite_86(self):
        for dt in ("F32", "BF16", "F16", "F64"):
            for bad in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(dtype=dt, v=bad):
                    m = HFModel("llama", 64, tied=False, seed=53, dtype=dt)
                    name = "model.layers.1.mlp.down_proj.weight"
                    shape, vals = m.t[name]
                    vals = list(vals)
                    vals[7] = bad
                    m.t[name] = (shape, vals)
                    self.imp(86, model=m, sub="nf%s%r" % (dt, bad))

    def test_truncated_87(self):
        good = st_bytes(self.m.st_tensors(), metadata={"format": "pt"})
        n = struct.unpack_from("<Q", good, 0)[0]
        for name, cut in (("tail", good[:-10]), ("half", good[:8 + n + 100]), ("hdronly", good[:8 + n]),
                          ("midhdr", good[:8 + n // 2]), ("4bytes", good[:4]), ("empty", b"")):
            with self.subTest(case=name):
                self.imp(87, st=cut, sub="t" + name)

    def test_geometry_88(self):
        self.imp(88, cfg=self.cfg(num_key_value_heads=3), sub="hkv3")
        self.imp(88, cfg=self.cfg(num_key_value_heads=8), sub="hkv8")
        self.imp(88, cfg=self.cfg(hidden_size=18, head_dim=KeyError), sub="dmodh")

    def test_helix_tie_mismatch_81(self):
        if not HAVE_UNIV:
            self.skipTest("ollm-univ missing")
        ck = self.path("u.ck")
        self.univ("init", ck, "6", 70, 12, 16, 4, 2, 24, 2, 1, 0, 1000, 100)
        out = self.path("hx")
        self.ix("export-hf", ck, out)
        cfg = jload((os.path.join(out, "config.json")))
        cfg["tie_word_embeddings"] = True
        cp = self.path("hx", "bad.json")
        wb(cp, json.dumps(cfg).encode())
        self.ix("import-hf", cp, os.path.join(out, "model.safetensors"), self.path("x.ck"), rc=81)

    def test_ck_refusals(self):
        cp, sp = self.m.write(self.path("ok"))
        ck = self.path("ok.ck")
        self.ix("import-hf", cp, sp, ck)
        b = rb(ck)
        bad = self.path("bad.ck")
        cases = [("short", b[:100], 10), ("trunc", b[:-8], 10), ("magic", b"X" + b[1:], 11),
                 ("geom", b[:24] + struct.pack("<Q", 0) + b[32:], 12),
                 ("P", b[:104] + struct.pack("<Q", struct.unpack_from("<Q", b, 104)[0] + 1) + b[112:], 13),
                 ("fnv", b[:-3] + bytes([b[-3] ^ 1]) + b[-2:], 14), ("long", b + b"\0" * 8, 15)]
        for name, data, rc in cases:
            with self.subTest(case=name):
                wb(bad, data)
                self.ix("export-hf", bad, self.path("o-" + name), rc=rc)
                self.ix("export-gguf", bad, self.path(name + ".gguf"), rc=rc)
                self.assertFalse(os.path.exists(self.path(name + ".gguf")))


if __name__ == "__main__":
    print("ollm-interchange: %s (%s)" % (IX_BIN, "found" if HAVE_IX else "MISSING"))
    print("ollm-univ:        %s (%s)" % (UNIV_BIN, "found" if HAVE_UNIV else "MISSING"))
    print("ollm-macos:       %s (%s)" % (LEGACY_BIN, "found" if HAVE_LEGACY else "MISSING"))
    unittest.main()
