#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Comprehensive verification & f64 oracle suite for Pillar 3 (native/gpu/ring.om)
and Pillar 5 (native/arboris.om).

Verifies:
  1. Compilation with seeded omc for BOTH `target a64macos` (CPU simulation) and
     `gb10` (`om2` ELF), including `context.om + ring.om` NValloc uncached composition
     and collision-free composition with the full `native/` stack.
  2. Pillar 3 (`native/gpu/ring.om`):
     - Cacheline-padded layout, GPU-uncached `RING_NVALLOC_CACHED == 0`, GPU VA offsets
     - Lock-free SPSC/MPMC ticket ring: full backpressure, wrap-around across 200 items,
       FNV-1a-64 negative control (1-bit payload corruption fails closed with -2 & fault latch)
     - Concurrent Grace CPU worker tasks:
       * BPE/batch packing (`grg_pack_batch` / `grg_unpack_batch`)
       * Sparse top-K distillation (`OLLMDK01`, `grg_worker_dk_step`) vs Python f64 oracle
         and finite-difference gradient check (< 1e-12), plus `qsum > 1.0001` negative control
       * Exact f64 spot-check verification (`grg_worker_spot_check`) + fail-closed mismatch
       * Async AdamW master-weight updates (`grg_worker_weight_update`) vs Python f64 oracle (< 1e-12)
  3. Pillar 5 (`native/arboris.om`):
     - Open-addressed associative pattern table (`arboris_put` / `arboris_probe`) with
       FNV-1a-64 + SplitMix64 collision signatures and automatic ambiguous-ngram demotion
     - ~1 us CPU probe speed via `monons` (assert mean probe <= 2000 ns)
     - Sequence bypass (`arboris_bypass_seq`) exactness & contested index collection
     - `d9b7422` adaptive exit + `spclip` JSD strand disagreement warp-compaction
       (`arboris_compact_mask`) across multiple 32-thread warps vs Python bitmask oracle
     - Addendum #8: `arboris_insert_verified_omg`, `arboris_omg_recall`,
       `arboris_guide_omg_step` (bitmask wall + Arboris bias + LTP0 trigger), and
       `arboris_export_sleep_traces`

Usage:
  python3 -W error::ResourceWarning tests/test_split_bypass.py
"""
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

PASSES = []
FAILS = []


def check(cond, name, detail=""):
    if cond:
        PASSES.append(name)
        print(f"PASS {name}")
    else:
        FAILS.append(name)
        print(f"FAIL {name}" + (f": {detail}" if detail else ""))


def find_omc():
    for cand in (
        os.environ.get("OMC", ""),
        os.path.expanduser("~/.local/bin/omc"),
        shutil.which("omc") or "",
    ):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    raise FileNotFoundError("omc compiler not found")


def compile_src(omc, src_text, out_bin, target_name):
    p = subprocess.run([omc], input=src_text.encode("utf-8"), capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"omc ({target_name}) failed rc={p.returncode}: {p.stderr.decode('utf-8', errors='replace')}"
        )
    if out_bin is not None:
        with open(out_bin, "wb") as f:
            f.write(p.stdout)
        os.chmod(out_bin, 0o755)
        if target_name == "a64macos" and platform.system() == "Darwin" and shutil.which("codesign"):
            subprocess.run(["codesign", "-s", "-", "-f", out_bin], capture_output=True, check=True)
    return p.stdout


def read_text(rel_path):
    with open(os.path.join(REPO, rel_path), "r", encoding="utf-8") as f:
        return f.read()


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def f32_round(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def u64_to_f64(u):
    return struct.unpack("<d", struct.pack("<Q", u))[0]


def f64_to_u64(x):
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


# ---- Python f64 Oracles ----

def oracle_ollmdk01(stud_logits, y, top_ids, top_probs, alpha, tau):
    v = len(stud_logits)
    mx = max(stud_logits)
    w1 = [math.exp(z - mx) for z in stud_logits]
    s1 = sum(w1)
    p1 = [w / s1 for w in w1]
    wt = [math.exp((z - mx) / tau) for z in stud_logits]
    st = sum(wt)
    pt = [w / st for w in wt]

    py = max(p1[y], 1e-300)
    ce = -math.log(py)
    kl = 0.0
    qsum = 0.0
    for j, qj in zip(top_ids, top_probs):
        qsum += qj
        if qj > 1e-15:
            kl += qj * (math.log(qj) - math.log(max(pt[j], 1e-300)))
    kl *= tau * tau
    tot = (1.0 - alpha) * ce + alpha * kl

    om = 1.0 - alpha
    at = alpha * tau
    grad = [0.0] * v
    for c in range(v):
        g = om * (p1[c] - (1.0 if c == y else 0.0)) + at * qsum * pt[c]
        grad[c] = g
    for j, qj in zip(top_ids, top_probs):
        grad[j] -= at * qj
    return tot, ce, kl, qsum, grad


def oracle_adamw_steps(w0, grads_list, lr, wd):
    w = list(w0)
    m = [0.0] * len(w)
    v = [0.0] * len(w)
    for step, grads in enumerate(grads_list, start=1):
        ib1 = 1.0 / (1.0 - (0.9 ** step))
        ib2 = 1.0 / (1.0 - (0.999 ** step))
        for i, g_raw in enumerate(grads):
            g = f32_round(g_raw)
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            mh = m[i] * ib1
            vh = v[i] * ib2
            w[i] = w[i] * (1.0 - lr * wd) - lr * mh / (math.sqrt(vh) + 1e-8)
    return w


def oracle_jsd_milli(pa, pb):
    def H(p):
        return -sum(x * math.log(x) for x in p if x > 1e-15)
    pm = [0.5 * (a + b) for a, b in zip(pa, pb)]
    jsd = max(0.0, H(pm) - 0.5 * (H(pa) + H(pb)))
    return int(jsd * 1000.0 + 0.5)


# ---- Omega Driver Harnesses ----

RING_DRIVER_OM = r"""
text t_layout_ok "LAYOUT_OK"
text t_nvalloc_c "NVALLOC_CACHED"
text t_tot_b "TOTAL_BYTES"
text t_ph_off "PROD_HEAD_OFF"
text t_ct_off "CONS_TAIL_OFF"
text t_dva5 "DESC_VA_5"
text t_pva5 "PAY_VA_5"
text t_fill64 "FILL64_OK"
text t_dfull "DEPTH_FULL"
text t_cpfull "CAN_PUSH_FULL"
text t_ovrc "OVERFLOW_RC"
text t_cont "CONTEND_CNT"
text t_wrap "WRAP200_OK"
text t_ddrain "DEPTH_DRAINED"
text t_epoch "EPOCH"
text t_corrc "CORRUPT_RC"
text t_corflt "CORRUPT_FAULT"
text t_corval "CORRUPT_VALID"
text t_bpeok "BPE_PACK_OK"
text t_bpecnt "BPE_PACK_CNT"
text t_dkok "DK_OK"
text t_dktot "DK_TOT_BITS"
text t_dkce "DK_CE_BITS"
text t_dkkl "DK_KL_BITS"
text t_dkqs "DK_QSUM_BITS"
text t_dkgr "DK_GRAD_BITS"
text t_dkmsk "DK_MASKED_RC"
text t_spok "SPOT_PASS_OK"
text t_spcnt "SPOT_CHK_CNT"
text t_spfrc "SPOT_FAIL_RC"
text t_spflt "SPOT_FAULT"
text t_wcnt "WUPD_CNT"
text t_wm "W_MASTER_BITS"
text t_dqrc "DK_QSUM_NEG_RC"
text t_dqflt "DK_QSUM_NEG_FAULT"
text t_epst "EPOCH_SELFTEST_RC"
text t_ep0 "EPOCH_INIT_EP"
text t_epgb0 "EPOCH_INIT_GB"
text t_epcb0 "EPOCH_INIT_CB"
text t_epact0 "EPOCH_INIT_ACT"
text t_epflt0 "EPOCH_INIT_FLT"
text t_epmact "EPOCH_MID_ACT"
text t_epst1 "EPOCH_STAGE_RC1"
text t_epst2 "EPOCH_STAGE_RC2"
text t_epgdc "EPOCH_GUARD_CPU_RC"
text t_epgdg "EPOCH_GUARD_GPU_RC"
text t_epgdf "EPOCH_GUARD_GPU_FLT"
text t_epcmr "EPOCH_MID_COMMIT_RC"
text t_epcmf "EPOCH_MID_COMMIT_FLT"
text t_epcme "EPOCH_MID_COMMIT_EP"
text t_epoob1 "EPOCH_STAGE_OOB_RC"
text t_epoob2 "EPOCH_GUARD_OOB_RC"
text t_epmgb "EPOCH_MID_GPU_BITS"
text t_epc1r "EPOCH_CM1_RC"
text t_epc1g "EPOCH_CM1_GB"
text t_epc1c "EPOCH_CM1_CB"
text t_epc1b "EPOCH_CM1_GPU_BITS"
text t_epc2r "EPOCH_CM2_RC"
text t_epc2g "EPOCH_CM2_GB"
text t_epc2b "EPOCH_CM2_GPU_BITS"

fn p_str s view uchar * to void
r = write 1 s 0 len s
end

fn p_u v ulong to void
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

fn p_l v long to void
t = new uchar 1 45
if lt v 0
r = write 1 t 0 1
p_u ulong neg v
else
p_u ulong v
end
end

fn b2u b bool to ulong
if b
ret 1
end
ret 0
end

fn p_kv k view uchar * v ulong to void
p_str k
sp = new uchar 1 32
r = write 1 sp 0 1
p_u v
nl = new uchar 1 10
r = write 1 nl 0 1
end

fn p_kvl k view uchar * v long to void
p_str k
sp = new uchar 1 32
r = write 1 sp 0 1
p_l v
nl = new uchar 1 10
r = write 1 nl 0 1
end

fn test_layout_and_fill to void
va_base = ulong 0x1000000000
grg_init va_base true true
p_kv t_layout_ok b2u grg_valid_hdr
p_kv t_nvalloc_c RING_NVALLOC_CACHED
p_kv t_tot_b RING_TOTAL_BYTES
p_kv t_ph_off RING_OFF_PROD_HEAD
p_kv t_ct_off RING_OFF_CONS_TAIL
p_kv t_dva5 grg_gpu_desc_va 5
p_kv t_pva5 grg_gpu_pay_va 5

all_ok = ulong 1
for i ulong 64
grg_tw64 0 add 1000 i
tk = grg_push_tmp uint RING_OP_BPE_PACK 0 uint i uint i 0 8
if ne tk long i
all_ok = ulong 0
end
end
p_kv t_fill64 all_ok
p_kv t_dfull grg_depth
p_kv t_cpfull b2u grg_spsc_can_push

grg_tw64 0 9999
ov = grg_push_tmp uint RING_OP_BPE_PACK 0 64 0 0 8
p_kvl t_ovrc ov
p_kv t_cont grg_r64 RING_OFF_MPMC_CONTEND
end

fn test_wrap_200 to void
meta = new ulong 10 0
wrap_ok = ulong 1
for j ulong 10
pk = grg_pop_tmp meta
if or ne pk long j ne grg_tr64 0 add 1000 j
wrap_ok = ulong 0
end
end
for k ulong 200
grg_tw64 0 add 5000 k
tk2 = grg_push_tmp uint RING_OP_BPE_PACK 1 uint k 0 0 8
if ne tk2 long add 64 k
wrap_ok = ulong 0
end
pk2 = grg_pop_tmp meta
exp = add 5000 sub k 54
if lt k 54
exp = add 1000 add 10 k
end
if or ne pk2 long add 10 k ne grg_tr64 0 exp
wrap_ok = ulong 0
end
end
for r ulong 54
pk3 = grg_pop_tmp meta
exp3 = add 5000 add 146 r
if or lt pk3 0 ne grg_tr64 0 exp3
wrap_ok = ulong 0
end
end
p_kv t_wrap wrap_ok
p_kv t_ddrain grg_depth
p_kv t_epoch grg_r64 RING_OFF_PROD_EPOCH
end

fn test_corruption_negctrl to void
grg_init 0x2000000000 false true
grg_tw64 0 0x1122334455667788
tk = grg_push_tmp uint RING_OP_BPE_PACK 0 1 0 0 8
poff = grg_pay_off ulong tk
put grg_ring poff xor get grg_ring poff 1
meta = new ulong 10 0
pk = grg_pop_tmp meta
p_kvl t_corrc pk
p_kv t_corflt grg_r64 RING_OFF_FAULT
p_kv t_corval b2u grg_valid_hdr
end

fn test_bpe_unpack_chk plen ulong exp_toks view uint * exp_msk view uchar * to ulong
otoks = new uint 6 0
omsk = new uchar 6 0
ohdr = new uint 4 0
if not grg_unpack_batch plen otoks omsk ohdr
ret 0
end
for i ulong 6
if or ne get otoks i get exp_toks i ne get omsk i get exp_msk i
ret 0
end
end
ret 1
end

fn test_bpe_pack to void
grg_init 0x3000000000 true true
toks = new uint 6 0
put toks 0 101
put toks 1 202
put toks 2 303
put toks 3 404
put toks 4 505
put toks 5 606
msk = new uchar 6 1
put msk 0 0
put msk 5 0
tk = grg_pack_batch toks msk 7 3
meta = new ulong 10 0
pk = grg_pop_tmp meta
match = ulong 0
if and ge tk 0 eq pk tk
match = test_bpe_unpack_chk get meta 4 toks msk
end
p_kv t_bpeok match
p_kv t_bpecnt grg_r64 RING_OFF_BPE_PACKED
end

fn test_dk_step to void
grg_init 0x4000000000 true true
ok_cfg = grg_dk_init 8 3 2 700000 1500
grg_dk_set_target 0 2 1
grg_dk_set_target 1 5 0
ok0 = grg_dk_set_entry 0 0 2 0x3f000000
ok1 = grg_dk_set_entry 0 1 4 0x3e99999a
ok2 = grg_dk_set_entry 0 2 1 0x3dcccccd
stud = new f64 8 0.0f64
put stud 0 0.25f64
put stud 1 -0.5f64
put stud 2 1.75f64
put stud 3 0.1f64
put stud 4 0.9f64
put stud 5 -1.2f64
put stud 6 0.4f64
put stud 7 -0.3f64
tk = grg_worker_dk_step 0 stud 11 2
meta = new ulong 10 0
pk = grg_pop_tmp meta
p_kv t_dkok b2u and ok_cfg and ok0 and ok1 and ok2 and ge tk 0 eq pk tk
p_kv t_dktot bitcast ulong get grg_dk_stats 0
p_kv t_dkce bitcast ulong get grg_dk_stats 1
p_kv t_dkkl bitcast ulong get grg_dk_stats 2
p_kv t_dkqs bitcast ulong get grg_dk_stats 3
for c ulong 8
p_kv t_dkgr bitcast ulong get grg_dk_grad c
end
sk = grg_worker_dk_step 1 stud 11 2
p_kvl t_dkmsk sk
end

fn test_spot_check to void
yref = new f64 8 0.0f64
ybad = new f64 8 0.0f64
for i ulong 8
v = mul f64 add i 1 0.125f64
put yref i v
put ybad i v
end
tkf = grg_push_gpu_fwd_f32 yref 2 11
chk_ok = grg_worker_spot_check yref 1.0e-5f64 1
p_kv t_spok b2u and ge tkf 0 ge chk_ok 0
p_kv t_spcnt grg_r64 RING_OFF_CHK_OK
put ybad 4 add get yref 4 0.25f64
tkb = grg_push_gpu_fwd_f32 ybad 2 12
chk_bad = grg_worker_spot_check yref 1.0e-5f64 2
p_kvl t_spfrc chk_bad
p_kv t_spflt grg_r64 RING_OFF_FAULT
end

fn test_adamw_and_dk_neg to void
grg_init 0x5000000000 true true
for w ulong 4
put grg_w_master w sub mul f64 add w 1 0.5f64 1.0f64
put grg_m_state w 0.0f64
put grg_v_state w 0.0f64
end
g1 = new f64 4 0.0f64
for stp ulong 3
for k ulong 4
put g1 k mul f64 add stp 1 mul 0.1f64 sub f64 k 1.5f64
end
tg = grg_push_grad_f32 g1 3 uint stp
wu = grg_worker_weight_update 0 add stp 1 0.01f64 0.05f64
end
p_kv t_wcnt grg_r64 RING_OFF_W_UPD
for w2 ulong 4
p_kv t_wm bitcast ulong get grg_w_master w2
end

grg_init 0x6000000000 true true
ok_q = grg_dk_init 4 2 1 500000 1000
grg_dk_set_target 0 0 1
q0 = grg_dk_set_entry 0 0 0 0x3f4ccccd
q1 = grg_dk_set_entry 0 1 1 0x3f4ccccd
st4 = new f64 4 0.1f64
qrc = grg_worker_dk_step 0 st4 1 0
p_kvl t_dqrc qrc
p_kv t_dqflt grg_r64 RING_OFF_FAULT
end

fn test_epoch_double_buffer to void
p_kvl t_epst long grg_epoch_selftest
grg_init 0x7000000000 true true
put grg_w_master 0 0.25f64
put grg_w_master 1 -1.5f64
put grg_w_master 2 2.75f64
put grg_w_master 3 0.0f64
put grg_w_master 4 -0.125f64
put grg_w_master 5 3.5f64
put grg_w_master 6 4.25f64
put grg_w_master 7 -8.5f64
grg_epoch_init 8
p_kv t_ep0 grg_r64 RING_OFF_EPOCH_ID
p_kv t_epgb0 grg_epoch_gpu_bank
p_kv t_epcb0 grg_epoch_cpu_bank
p_kv t_epact0 grg_r64 RING_OFF_GPU_STEP_ACTIVE
p_kv t_epflt0 grg_r64 RING_OFF_FAULT

grg_epoch_begin_gpu_step
p_kv t_epmact grg_r64 RING_OFF_GPU_STEP_ACTIVE
p_kvl t_epst1 grg_epoch_stage_weight 1 42.125f64
p_kvl t_epst2 grg_epoch_stage_weight 5 -77.5f64
p_kvl t_epgdc grg_epoch_write_bank_guarded grg_epoch_cpu_bank 3 9.25f64

p_kvl t_epgdg grg_epoch_write_bank_guarded grg_epoch_gpu_bank 0 999.0f64
p_kv t_epgdf grg_r64 RING_OFF_FAULT
grg_w64_rel RING_OFF_FAULT RING_ERR_NONE

p_kvl t_epcmr grg_epoch_commit_step_boundary 8
p_kv t_epcmf grg_r64 RING_OFF_FAULT
p_kv t_epcme grg_r64 RING_OFF_EPOCH_ID

p_kvl t_epoob1 grg_epoch_stage_weight RING_W_CAP 1.0f64
p_kvl t_epoob2 grg_epoch_write_bank_guarded 2 0 1.0f64
for i ulong 8
p_kv t_epmgb bitcast ulong grg_epoch_read_gpu i
end

grg_epoch_end_gpu_step
grg_w64_rel RING_OFF_FAULT RING_ERR_NONE
cm1 = grg_epoch_commit_step_boundary 8
p_kvl t_epc1r cm1
p_kv t_epc1g grg_epoch_gpu_bank
p_kv t_epc1c grg_epoch_cpu_bank
for j ulong 8
p_kv t_epc1b bitcast ulong grg_epoch_read_gpu j
end

grg_epoch_begin_gpu_step
st_ep2 = grg_epoch_stage_weight 0 11.5f64
grg_epoch_end_gpu_step
cm2 = grg_epoch_commit_step_boundary 8
p_kvl t_epc2r cm2
p_kv t_epc2g grg_epoch_gpu_bank
for k ulong 8
p_kv t_epc2b bitcast ulong grg_epoch_read_gpu k
end
end

fn main to int
test_layout_and_fill
test_wrap_200
test_corruption_negctrl
test_bpe_pack
test_dk_step
test_spot_check
test_adamw_and_dk_neg
test_epoch_double_buffer
ret 0
end
"""

ARBORIS_DRIVER_OM = r"""
text a_aput "ASSOC_PUT_OK"
text a_ahit "ASSOC_HIT1"
text a_ahitok "ASSOC_HIT1_TOK"
text a_amb "ASSOC_AMBIG_BLOCKED"
text a_dem "ASSOC_DEMOTIONS"
text a_p10h "PROBE_10K_HITS"
text a_p10ns "PROBE_10K_NS"
text a_sbp "SEQ_BYPASS_CNT"
text a_scnt "SEQ_CONTESTED_CNT"
text a_sp2 "SEQ_PRED_POS2"
text a_sp4 "SEQ_PRED_POS4"
text a_se2 "SEQ_EXIT_POS2"
text a_se6 "SEQ_EXIT_POS6"
text a_jeq "JSD_AGREE"
text a_jhi "JSD_DISAGREE"
text a_cact "COMPACT_ACTIVE"
text a_cw0 "COMPACT_WARP0"
text a_cw1 "COMPACT_WARP1"
text a_cp0 "COMPACT_POP0"
text a_cp1 "COMPACT_POP1"
text a_cidx "COMPACT_IDX"
text a_et0 "EXIT_T0_CPU"
text a_et1 "EXIT_T1_L1"
text a_et2 "EXIT_T2_L2"
text a_et5 "EXIT_T5_CONTESTED"
text a_orec "OMG_RECALL_OK"
text a_otok "OMG_RECALL_TOK"
text a_oact "OMG_RECALL_ACT"
text a_gft "GUIDE_FAST_TOK"
text a_gfl "GUIDE_FAST_LTP0"
text a_glt "GUIDE_LOWACT_TOK"
text a_gll "GUIDE_LOWACT_LTP0"
text a_gjl "GUIDE_JSD_LTP0"
text a_sec "SLEEP_EXP_CNT"
text a_sew "SLEEP_EXP_WORDS"
text a_ses "SLEEP_EXP_SEM"
text a_sel "SLEEP_EXP_LEN"
text a_set "SLEEP_EXP_TOK2"

fn p_str s view uchar * to void
r = write 1 s 0 len s
end

fn p_u v ulong to void
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

fn b2u b bool to ulong
if b
ret 1
end
ret 0
end

fn p_kv k view uchar * v ulong to void
p_str k
sp = new uchar 1 32
r = write 1 sp 0 1
p_u v
nl = new uchar 1 10
r = write 1 nl 0 1
end

fn test_seq_bypass to void
seq = new ulong 8 0
put seq 0 10
put seq 1 20
put seq 2 30
put seq 3 50
put seq 4 60
put seq 5 80
put seq 6 90
put seq 7 99
pred = new ulong 8 0
nbp = arboris_bypass_seq seq 3 900000 pred
p_kv a_sbp nbp
p_kv a_scnt get arb_stat 6
p_kv a_sp2 get pred 2
p_kv a_sp4 get pred 4
p_kv a_se2 get arb_exit_layer 2
p_kv a_se6 get arb_exit_layer 6
end

fn test_assoc_and_speed to void
arboris_clear
p1 = new ulong 3 0
put p1 0 10
put p1 1 20
put p1 2 30
ok1 = arboris_put p1 0 3 40 980000
p2 = new ulong 2 0
put p2 0 50
put p2 1 60
ok2 = arboris_put p2 0 2 70 950000
p3 = new ulong 2 0
put p3 0 80
put p3 1 90
ok3a = arboris_put p3 0 2 91 990000
ok3b = arboris_put p3 0 2 92 990000

res = new ulong 2 0
hit1 = arboris_probe p1 0 3 res
hit3 = arboris_probe p3 0 2 res
p_kv a_aput b2u and ok1 and ok2 and ok3a ok3b
p_kv a_ahit b2u hit1
p_kv a_ahitok get res 0
p_kv a_amb b2u not hit3
p_kv a_dem get arb_stat 5

t0 = monons
hits = ulong 0
for i ulong 10000
if arboris_probe p1 0 3 res
hits = add hits 1
end
end
t1 = monons
dt_ns = ulong sub t1 t0
p_kv a_p10h hits
p_kv a_p10ns dt_ns
end

fn test_compact_and_jsd to void
arboris_clear
arboris_set_thresholds 900000 50000 100 25000

note Compute JSD in milli-nats for agreeing vs disagreeing strands
pa = new f64 4 0.0f64
pb_agree = new f64 4 0.0f64
pb_disagree = new f64 4 0.0f64
put pa 0 0.7f64
put pa 1 0.1f64
put pa 2 0.1f64
put pa 3 0.1f64
put pb_agree 0 0.7f64
put pb_agree 1 0.1f64
put pb_agree 2 0.1f64
put pb_agree 3 0.1f64
put pb_disagree 0 0.05f64
put pb_disagree 1 0.05f64
put pb_disagree 2 0.1f64
put pb_disagree 3 0.8f64
j_eq = arboris_jsd_milli pa pb_agree
j_hi = arboris_jsd_milli pa pb_disagree
p_kv a_jeq j_eq
p_kv a_jhi j_hi

note Set up 48 tokens across 2 warps (warp 0: 0..31, warp 1: 32..47)
for t ulong 48
bc = ulong 0
dl = ulong 10000
if eq rem t 4 0
bc = ulong 950000
end
if eq rem t 4 2
dl = ulong 35000
end
arboris_set_token_signals t bc dl j_eq
end
arboris_set_token_signals 3 0 80000 j_eq
arboris_set_token_signals 5 950000 5000 j_hi
arboris_set_token_signals 17 0 60000 j_eq
arboris_set_token_signals 31 0 50000 j_eq
arboris_set_token_signals 33 0 5000 j_hi
arboris_set_token_signals 40 0 90000 j_eq

nact = arboris_compact_mask 48
p_kv a_cact nact
p_kv a_cw0 get arb_warp_mask 0
p_kv a_cw1 get arb_warp_mask 1
p_kv a_cp0 arboris_popcnt32 get arb_warp_mask 0
p_kv a_cp1 arboris_popcnt32 get arb_warp_mask 1
for k ulong nact bound 48
p_kv a_cidx get arb_active_idx k
end
p_kv a_et0 get arb_exit_layer 0
p_kv a_et1 get arb_exit_layer 1
p_kv a_et2 get arb_exit_layer 2
p_kv a_et5 get arb_exit_layer 5
end

fn test_omg_guide_and_sleep to void
arboris_clear
s0 = arboris_insert_verified_omg 0xaaaa1111 0x101 0x0f 42 800
s1 = arboris_insert_verified_omg 0xbbbb2222 0x202 0xf0 77 2500
s1b = arboris_insert_verified_omg 0xbbbb2222 0x202 0xf0 77 500
arboris_set_omg_stream ulong s0 0 11
arboris_set_omg_stream ulong s0 1 12
arboris_set_omg_stream ulong s1 0 21
arboris_set_omg_stream ulong s1 1 22
arboris_set_omg_stream ulong s1 2 23

r = new ulong 4 0
ok_rec = arboris_omg_recall 0xbbbb2222 0x202 0xf0 r
p_kv a_orec b2u and ok_rec eq s1 s1b
p_kv a_otok get r 2
p_kv a_oact get r 3

for c ulong 128
arboris_set_omg_step_in c 0.0f64 1
end
arboris_set_omg_step_in 99 5.0f64 0
arboris_set_omg_step_in 10 3.0f64 1
arboris_set_omg_step_in 77 2.0f64 1
arboris_set_guide_cfg 2000 2500 get r 2
st = new ulong 4 0
picked = arboris_guide_omg_step 128 get r 3 20 st
p_kv a_gft picked
p_kv a_gfl get st 1

st2 = new ulong 4 0
picked2 = arboris_guide_omg_step 128 500 20 st2
p_kv a_glt picked2
p_kv a_gll get st2 1

st3 = new ulong 4 0
picked3 = arboris_guide_omg_step 128 get r 3 250 st3
p_kv a_gjl get st3 1

sbuf = new ulong 128 0
scnt = new ulong 1 0
wlen = arboris_export_sleep_traces 1000 sbuf scnt
p_kv a_sec get scnt 0
p_kv a_sew wlen
p_kv a_ses get sbuf 0
p_kv a_sel get sbuf 5
p_kv a_set get sbuf 8
end

fn main to int
test_assoc_and_speed
test_seq_bypass
test_compact_and_jsd
test_omg_guide_and_sleep
ret 0
end
"""

GB10_NVALLOC_HARNESS_OM = r"""
note Verify ring.om + context.om NValloc(..., false) + mapaddr/cpufence composition on gb10
fn main to int
if not nvfull
ret 1
end
m = nvalloc RING_TOTAL_BYTES false
if le m 0
ret 2
end
va = mapaddr m
grg_init va true false
cpufence
if ne grg_gpu_desc_va 3 add va add RING_DESC_OFF mul 3 RING_DESC_BYTES
ret 3
end
r = mapset m RING_OFF_DOORBELL 8 1
cpufence
if ne mapget m RING_OFF_DOORBELL 8 1
ret 4
end
ok = nvdrop m
ok2 = nvstop
ret 0
end
"""


def parse_kv_output(stdout_bytes):
    kv = {}
    multi = {}
    for line in stdout_bytes.decode("utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) == 2:
            k, v = parts[0], int(parts[1])
            kv[k] = v
            multi.setdefault(k, []).append(v)
    return kv, multi


def main():
    omc = find_omc()
    ring_src = read_text("native/gpu/ring.om")
    arb_src = read_text("native/arboris.om")
    ctx_src = read_text("native/gpu/context.om")

    with tempfile.TemporaryDirectory(prefix="split_bypass_") as tmp:
        # ---- 1. Cross-Target Compilation Checks ----
        is_mac = platform.system() == "Darwin"
        is_a64_linux = platform.system() == "Linux" and platform.machine() == "aarch64"

        ring_gb10_bin = os.path.join(tmp, "ring_gb10")
        ring_gb10 = compile_src(omc, "om2\n" + ring_src + RING_DRIVER_OM, ring_gb10_bin, "gb10")
        check(ring_gb10[:4] == b"\x7fELF", "compile ring.om gb10 om2 (ELF)")

        arb_gb10_bin = os.path.join(tmp, "arb_gb10")
        arb_gb10 = compile_src(omc, "om2\n" + arb_src + ARBORIS_DRIVER_OM, arb_gb10_bin, "gb10")
        check(arb_gb10[:4] == b"\x7fELF", "compile arboris.om gb10 om2 (ELF)")

        nv_gb10 = compile_src(omc, "om2\n" + ctx_src + ring_src + GB10_NVALLOC_HARNESS_OM, None, "gb10")
        check(nv_gb10[:4] == b"\x7fELF", "compile context.om + ring.om nvalloc(..., false) gb10 om2 (ELF)")

        ring_mac_bin = os.path.join(tmp, "ring_mac")
        arb_mac_bin = os.path.join(tmp, "arb_mac")
        if is_mac:
            ring_mac = compile_src(omc, "target a64macos\nom2\n" + ring_src + RING_DRIVER_OM, ring_mac_bin, "a64macos")
            check(ring_mac[:4] == b"\xcf\xfa\xed\xfe", "compile ring.om target a64macos (Mach-O)")
            arb_mac = compile_src(omc, "target a64macos\nom2\n" + arb_src + ARBORIS_DRIVER_OM, arb_mac_bin, "a64macos")
            check(arb_mac[:4] == b"\xcf\xfa\xed\xfe", "compile arboris.om target a64macos (Mach-O)")

        # Full stack collision-free composition check (no edits to existing files)
        full_core = "".join(
            read_text(p)
            for p in (
                "native/math.om",
                "native/data.om",
                "native/lpool.om",
                "native/model.om",
                "native/adam.om",
                "native/ckpt.om",
                "native/bpe.om",
                "native/spec.om",
                "native/rev.om",
                "native/sp_solve.om",
                "native/spiral.om",
                "native/apu_weave.om",
                "native/omg.om",
                "native/pillars_cli.om",
                "native/cli.om",
            )
        )
        if is_mac:
            full_mac = compile_src(
                omc,
                "target a64macos\nom2\n" + arb_src + full_core + ring_src + read_text("native/mio.om") + read_text("native/noserve.om") + read_text("native/nogpu.om") + read_text("native/apu_fabric_macos.om"),
                None,
                "a64macos",
            )
            check(full_mac[:4] == b"\xcf\xfa\xed\xfe", "full stack + arboris.om + ring.om collision-free a64macos")

        gpu_mods = "".join(
            read_text(p)
            for p in (
                "native/gpu/context.om",
                "native/gpu/vector.om",
                "native/gpu/launch.om",
                "native/gpu/batch.om",
                "native/gpu/gk.om",
                "native/gpu/gr.om",
                "native/gpu/gb.om",
                "native/gpu/gg.om",
                "native/gpu/gt.om",
                "native/gpu/gm.om",
                "native/gpu/ejit.om",
                "native/apu_fabric_linux.om",
                "native/apu_fabric_gb10.om",
            )
        )
        full_gb10_om = (
            "om2\n"
            + read_text("native/net/http.om")
            + read_text("native/net/json.om")
            + read_text("native/net/jsonw.om")
            + full_core
            + arb_src
            + ring_src
            + read_text("native/lio.om")
            + read_text("native/serve.om")
            + gpu_mods
        )
        full_gb10 = compile_src(omc, full_gb10_om, None, "gb10")
        check(full_gb10[:4] == b"\x7fELF", "full stack + arboris.om + ring.om collision-free gb10 om2")

        # ---- 2. Execute Pillar 3 (ring.om) on host ARM64 simulation ----
        if not (is_mac or is_a64_linux):
            print("SKIP runtime execution of arm64 binaries on non-arm64 host")
            return 0 if not FAILS else 1

        ring_run_bin = ring_mac_bin if is_mac else ring_gb10_bin
        arb_run_bin = arb_mac_bin if is_mac else arb_gb10_bin
        p_ring = subprocess.run([ring_run_bin], capture_output=True, check=True)
        rkv, rmulti = parse_kv_output(p_ring.stdout)

        check(rkv["LAYOUT_OK"] == 1, "ring header & layout valid")
        check(rkv["NVALLOC_CACHED"] == 0, "ring GB10 nvalloc cached=false enforced")
        check(rkv["TOTAL_BYTES"] == 8192 + 64 * 4096, "ring total bytes page-aligned (270336)")
        check(rkv["PROD_HEAD_OFF"] == 128 and rkv["CONS_TAIL_OFF"] == 256, "ring head/tail 128B cacheline padded")
        check(rkv["DESC_VA_5"] == 0x1000000000 + 1024 + 5 * 64, "ring GPU descriptor VA offset exact")
        check(rkv["PAY_VA_5"] == 0x1000000000 + 8192 + 5 * 4096, "ring GPU payload VA offset exact")

        check(rkv["FILL64_OK"] == 1 and rkv["DEPTH_FULL"] == 64 and rkv["CAN_PUSH_FULL"] == 0, "ring fill 64 slots lock-free")
        check(rkv["OVERFLOW_RC"] == -1 and rkv["CONTEND_CNT"] == 1, "ring full backpressure returns -1")
        check(rkv["WRAP200_OK"] == 1 and rkv["DEPTH_DRAINED"] == 0 and rkv["EPOCH"] == 4, "ring 200-ticket wrap-around bit-exact")

        check(rkv["CORRUPT_RC"] == -2 and rkv["CORRUPT_FAULT"] == 2 and rkv["CORRUPT_VALID"] == 0,
              "ring FNV-1a-64 negative control fails closed on 1-bit flip")

        check(rkv["BPE_PACK_OK"] == 1 and rkv["BPE_PACK_CNT"] == 1, "Grace worker: BPE/batch packing bit-exact")

        # OLLMDK01 oracle cross-check
        stud = [0.25, -0.5, 1.75, 0.1, 0.9, -1.2, 0.4, -0.3]
        top_ids = [2, 4, 1]
        top_probs = [f32_round(0.5), f32_round(0.3), f32_round(0.1)]
        tot, ce, kl, qsum, grad = oracle_ollmdk01(stud, 2, top_ids, top_probs, 0.7, 1.5)
        check(rkv["DK_OK"] == 1 and rkv["DK_MASKED_RC"] == -4, "Grace worker: OLLMDK01 mask & push")
        err_tot = abs(u64_to_f64(rkv["DK_TOT_BITS"]) - tot)
        err_ce = abs(u64_to_f64(rkv["DK_CE_BITS"]) - ce)
        err_kl = abs(u64_to_f64(rkv["DK_KL_BITS"]) - kl)
        err_qs = abs(u64_to_f64(rkv["DK_QSUM_BITS"]) - qsum)
        om_grad = [u64_to_f64(u) for u in rmulti["DK_GRAD_BITS"]]
        max_g_err = max(abs(a - b) for a, b in zip(om_grad, grad))
        # Independent finite-difference check on OLLMDK01 gradient
        eps = 1e-6
        fd_errs = []
        for c in range(len(stud)):
            sp = list(stud)
            sm = list(stud)
            sp[c] += eps
            sm[c] -= eps
            lp = oracle_ollmdk01(sp, 2, top_ids, top_probs, 0.7, 1.5)[0]
            lm = oracle_ollmdk01(sm, 2, top_ids, top_probs, 0.7, 1.5)[0]
            fd_errs.append(abs(om_grad[c] - (lp - lm) / (2.0 * eps)))
        check(max(err_tot, err_ce, err_kl, err_qs, max_g_err) < 1e-12 and max(fd_errs) < 1e-8,
              f"Grace worker: OLLMDK01 f64 & FD oracle parity (max_err={max(err_tot, err_ce, err_kl, err_qs, max_g_err):.2e}, fd_err={max(fd_errs):.2e})")

        check(rkv["DK_QSUM_NEG_RC"] == -5 and rkv["DK_QSUM_NEG_FAULT"] == 3,
              "Grace worker: OLLMDK01 qsum > 1.0001 rejected fail-closed")

        check(rkv["SPOT_PASS_OK"] == 1 and rkv["SPOT_CHK_CNT"] == 1, "Grace worker: f64 spot-check passes valid f32 output")
        check(rkv["SPOT_FAIL_RC"] == -7 and rkv["SPOT_FAULT"] == 4, "Grace worker: f64 spot-check fails closed on perturbed output")

        # Async AdamW master-weight update oracle cross-check
        w0 = [(w + 1) * 0.5 - 1.0 for w in range(4)]
        g_steps = [[(stp + 1) * 0.1 * (k - 1.5) for k in range(4)] for stp in range(3)]
        w_ref = oracle_adamw_steps(w0, g_steps, 0.01, 0.05)
        w_om = [u64_to_f64(u) for u in rmulti["W_MASTER_BITS"]]
        max_w_err = max(abs(a - b) for a, b in zip(w_om, w_ref))
        check(rkv["WUPD_CNT"] == 3 and max_w_err < 1e-12,
              f"Grace worker: async AdamW f64 master-weight update parity (max_err={max_w_err:.2e})")

        # Byte-Traceability Lock 2: Strict Epoch Double-Buffering
        check(rkv["EPOCH_SELFTEST_RC"] == 0, "Lock 2: grg_epoch_selftest returns 0")
        check(rkv["EPOCH_INIT_EP"] == 0 and rkv["EPOCH_INIT_GB"] == 0 and rkv["EPOCH_INIT_CB"] == 1
              and rkv["EPOCH_INIT_ACT"] == 0 and rkv["EPOCH_INIT_FLT"] == 0,
              "Lock 2: grg_epoch_init sets epoch=0, gpu_bank=0, cpu_bank=1, active=0, fault=0")
        w_init = [0.25, -1.5, 2.75, 0.0, -0.125, 3.5, 4.25, -8.5]
        w_init_bits = [f64_to_u64(x) for x in w_init]
        check(rkv["EPOCH_MID_ACT"] == 1 and rkv["EPOCH_STAGE_RC1"] == 0 and rkv["EPOCH_STAGE_RC2"] == 0
              and rkv["EPOCH_GUARD_CPU_RC"] == 0 and rmulti["EPOCH_MID_GPU_BITS"] == w_init_bits,
              "Lock 2: mid-step Grace staging leaves GPU active bank bit-identical")
        check(rkv["EPOCH_GUARD_GPU_RC"] == -8 and rkv["EPOCH_GUARD_GPU_FLT"] == 6,
              "Lock 2 negative control: mid-step active-bank mutation rejected with -8 / RING_ERR_EPOCH_VIOLATION (6)")
        check(rkv["EPOCH_MID_COMMIT_RC"] == -8 and rkv["EPOCH_MID_COMMIT_FLT"] == 6
              and rkv["EPOCH_MID_COMMIT_EP"] == 0,
              "Lock 2 negative control: mid-step commit rejected with -8 / RING_ERR_EPOCH_VIOLATION (6), epoch unchanged")
        check(rkv["EPOCH_STAGE_OOB_RC"] == -3 and rkv["EPOCH_GUARD_OOB_RC"] == -3,
              "Lock 2 negative control: out-of-bounds stage/guarded bank write rejected with -3")
        w_ep1_bits = [f64_to_u64(x) for x in [0.25, 42.125, 2.75, 9.25, -0.125, -77.5, 4.25, -8.5]]
        check(rkv["EPOCH_CM1_RC"] == 1 and rkv["EPOCH_CM1_GB"] == 1 and rkv["EPOCH_CM1_CB"] == 0
              and rmulti["EPOCH_CM1_GPU_BITS"] == w_ep1_bits,
              "Lock 2: step-boundary commit flips epoch 0->1 and GPU reads exact staged + untouched weights")
        w_ep2_bits = [f64_to_u64(x) for x in [11.5, 42.125, 2.75, 9.25, -0.125, -77.5, 4.25, -8.5]]
        check(rkv["EPOCH_CM2_RC"] == 2 and rkv["EPOCH_CM2_GB"] == 0
              and rmulti["EPOCH_CM2_GPU_BITS"] == w_ep2_bits,
              "Lock 2: epoch 1->2 preserves prior epoch weights alongside newly staged deltas")

        # ---- 3. Execute Pillar 5 (arboris.om) on host ARM64 ----
        p_arb = subprocess.run([arb_run_bin], capture_output=True, check=True)
        akv, amulti = parse_kv_output(p_arb.stdout)

        check(akv["ASSOC_PUT_OK"] == 1 and akv["ASSOC_HIT1"] == 1 and akv["ASSOC_HIT1_TOK"] == 40,
              "Arboris associative put/probe deterministic recall")
        check(akv["ASSOC_AMBIG_BLOCKED"] == 1 and akv["ASSOC_DEMOTIONS"] == 1,
              "Arboris ambiguous n-gram auto-demoted (never falsely bypassed)")

        ns_per_probe = akv["PROBE_10K_NS"] / 10000.0
        check(akv["PROBE_10K_HITS"] == 10000 and ns_per_probe <= 1000.0,
              f"Arboris 1 us CPU probe speed ({ns_per_probe:.1f} ns/probe <= 1000 ns)")

        check(akv["SEQ_BYPASS_CNT"] == 2 and akv["SEQ_CONTESTED_CNT"] == 6
              and akv["SEQ_PRED_POS2"] == 40 and akv["SEQ_PRED_POS4"] == 70
              and akv["SEQ_EXIT_POS2"] == 0 and akv["SEQ_EXIT_POS6"] == 99,
              "Arboris sequence bypass exactness & contested routing")

        pa = [0.7, 0.1, 0.1, 0.1]
        pb_dis = [0.05, 0.05, 0.1, 0.8]
        exp_j_dis = oracle_jsd_milli(pa, pb_dis)
        check(akv["JSD_AGREE"] == 0 and abs(akv["JSD_DISAGREE"] - exp_j_dis) <= 1,
              f"spclip JSD disagreement oracle match (om={akv['JSD_DISAGREE']}, ref={exp_j_dis})")

        exp_contested = [3, 5, 17, 31, 33, 40]
        exp_w0 = sum(1 << i for i in exp_contested if i < 32)
        exp_w1 = sum(1 << (i - 32) for i in exp_contested if i >= 32)
        check(akv["COMPACT_ACTIVE"] == len(exp_contested)
              and amulti["COMPACT_IDX"] == exp_contested
              and akv["COMPACT_WARP0"] == exp_w0 and akv["COMPACT_WARP1"] == exp_w1
              and akv["COMPACT_POP0"] == 4 and akv["COMPACT_POP1"] == 2,
              "arboris_compact_mask multi-warp ballot & dense active index list exact")
        check(akv["EXIT_T0_CPU"] == 0 and akv["EXIT_T1_L1"] == 1
              and akv["EXIT_T2_L2"] == 2 and akv["EXIT_T5_CONTESTED"] == 99,
              "arboris_compact_mask Layer 0/1/2 early-exit vs spclip-contested routing")

        # Addendum #8 checks (4000 sem + 2000 io + 4*250 op_mask + min(3000/10, 500) util = 7300)
        check(akv["OMG_RECALL_OK"] == 1 and akv["OMG_RECALL_TOK"] == 77 and akv["OMG_RECALL_ACT"] == 7300,
              f"OMG subgraph spreading-activation recall (tok={akv['OMG_RECALL_TOK']}, act={akv['OMG_RECALL_ACT']})")
        check(akv["GUIDE_FAST_TOK"] == 77 and akv["GUIDE_FAST_LTP0"] == 0,
              "3-layer guided decoding: OMG mask wall blocks illegal tok 99 + Arboris bias picks 77 without LTP0")
        check(akv["GUIDE_LOWACT_TOK"] == 10 and akv["GUIDE_LOWACT_LTP0"] == 1 and akv["GUIDE_JSD_LTP0"] == 1,
              "3-layer guided decoding: low activation or spclip disagreement triggers LTP0=1")
        check(akv["SLEEP_EXP_CNT"] == 1 and akv["SLEEP_EXP_WORDS"] == 9
              and akv["SLEEP_EXP_SEM"] == 0xbbbb2222 and akv["SLEEP_EXP_LEN"] == 3
              and akv["SLEEP_EXP_TOK2"] == 23,
              "Hippocampus->Cortex sleep trace export filters by utility and preserves OMG stream")

    print(f"SUMMARY: {len(PASSES)} passed, {len(FAILS)} failed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
