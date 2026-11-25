# Omega Helix — 8-Pillar Pure-Omega Runtime & 4 Byte-Traceability Locks

Omega Helix extends its static, zero-CUDA, zero-`libc` transformer engine with **8 pure seeded-Omega (`.om`) runtime pillars** and **4 cryptographic/bit-exact byte-traceability locks**, verifiable with **zero Python** via [`scripts/verify-om.sh`](../scripts/verify-om.sh) and cross-checked against independent Python `f64` oracles + SM121 `nvdisasm` in [`scripts/verify.sh`](../scripts/verify.sh).

> **Honest Hardware Scope:** All 8 pillars compile cleanly into the static `ollm-gb10`, `ollm` (`aarch64`), `ollm-macos`, and `ollm-x86linux` binaries (with `norev.om` on `x86linux`), pass bit-exact CPU simulation & `f64` oracle parity, and pass `nvdisasm -b SM121` disassembly & GPR headroom checks (`0` illegal instructions). Live physical GB10 execution of the new Pillar 1/3/4/6 GPU kernels (`gm.om`, `ring.om`, `gt.om`, `ejit.om`) on raw `/dev/nvidia*` hardware is currently **`CPU-SIM + NVDISASM VERIFIED (LIVE GB10 GPU NOT_RUN)`**; the core FP32/BF16 training kernels (`gk.om`, `gr.om`, `gb.om`, `gg.om`) remain **PROVEN on DGX Spark GB10**.

---

## The 8 Pure-Omega Pillars

| # | Pillar | Pure-Omega Source | CLI / Entry Point | Independent Test Suite | What It Does |
|---|---|---|---|---|---|
| 1 | **Persistent SM121 Megakernel** | [`native/gpu/gm.om`](../native/gpu/gm.om) | `ollm-gb10 gmtest` / `gm_selftest` | [`tests/test_megakernel.py`](../tests/test_megakernel.py) | Double-buffered single-slot (`<= 16 KiB`) persistent SM121 SASS megakernel (`+16 B` bank-padded SMEM, `MEMBAR.SC.SYS`, `LDG.E.MMIO.GPU`, `STG.E.STRONG.SYS`), launch-count = 1 invariant, bit-exact parity vs sequential `obatch -> ograd -> oadam`. |
| 2 | **Reversible Zero-Activation Core** | [`native/rev.om`](../native/rev.om) (`norev.om` on `x86linux`) | `ollm rev selftest` | [`tests/test_reversible.py`](../tests/test_reversible.py) | Invertible residual blocks (`act_bytes = 0`) across GPT/ReLU, Llama/RoPE/GQA, Qwen3 looped (`NP=3`), GELU, and GeGLU, plus **Lock 4** exact Q32 integer invertibility (`0`-bit drift) and deterministic backward pass (`DETBWD`). |
| 3 | **Grace ↔ Blackwell Uncached Ring** | [`native/gpu/ring.om`](../native/gpu/ring.om) | `grg_epoch_selftest` | [`tests/test_split_bypass.py`](../tests/test_split_bypass.py) | Cacheline-padded (`128 B`), page-aligned, GPU-uncached (`RING_NVALLOC_CACHED=0`) lock-free SPSC/MPMC ring with FNV-1a-64 payload integrity, Grace CPU workers (BPE packing, `OLLMDK01` top-K distillation, `f64` spot-check, async AdamW), and **Lock 2** double-buffered epoch isolation. |
| 4 | **SM121 1.58-bit Ternary & 4-bit LUT-GEMM** | [`native/gpu/gt.om`](../native/gpu/gt.om) | `ollm-gb10 gttest` / `gt_selftest` | [`tests/test_gpu_ternary_sass.py`](../tests/test_gpu_ternary_sass.py) | 2-bit (4 trits/byte) and base-3 (5 trits/byte) ternary packing, UE4M3 block scaling, zero-multiply LUT-GEMM (`LOP3.LUT`, `PRMT`, `SHFL.*`, `VOTE.ANY`, `POPC`, `OMMA.SF.16864`), and warp compaction verified with `nvdisasm -b SM121`. |
| 5 | **Arboris Associative Bypass & Warp Compaction** | [`native/arboris.om`](../native/arboris.om) | `ollm arboris` | [`tests/test_split_bypass.py`](../tests/test_split_bypass.py) | Sub-microsecond (`~22 ns/probe` on Grace) open-addressed FNV-1a-64 + SplitMix64 associative bypass table with automatic ambiguous n-gram demotion, `spclip` JSD strand-disagreement warp compaction, OMG subgraph spreading activation, and sleep-trace export. |
| 6 | **Proof-Carrying Evolutionary SASS JIT** | [`native/gpu/ejit.om`](../native/gpu/ejit.om) | `ejit_selftest` | [`tests/test_evo_jit.py`](../tests/test_evo_jit.py) | Online SM121 SASS scheduler with atomic A/B buffer flipping, 15 fail-closed safety antibodies (digest spoof, bijection, opcode smuggle, anchor escape, scoreboard hazard, `BSSY` nesting, GPR ceiling, 6-cycle ALU floor, `I2F` latency), structural proofs (`EGP2`), and **Lock 3** receipt-verified deterministic replay. |
| 7 | **Closed-Form Ridge / Gauss-Newton Specialist Forge** | [`native/sp_solve.om`](../native/sp_solve.om) | `ollm spforge` | [`tests/test_sp_closed_form.py`](../tests/test_sp_closed_form.py) | One-shot primal (`D <= N`) and Woodbury (`N < D`) closed-form Ridge + Gauss-Newton solver that forges `OLLMSP01` Merkle specialist packs directly from `fcache`/`fcachem` frozen activations — verifiable via `spapp`, `speval`, and `spchainapp`. |
| 8 | **OMG Canonical Typed Program Graph** | [`native/omg.om`](../native/omg.om) | `ollm omg` | [`tests/test_omg_graph.py`](../tests/test_omg_graph.py) | De Bruijn canonical `u16` graph representation (`V=256` vs `V=4096` -> `16x` logit/CE reduction, `~5.2x` token compression vs `.om` text), 256-bit `SEMANTIC_ID`, `O(1)` grammar-valid token masking (`0%` invalid graphs across 1,000 samples), typed `OMG_MACRO` expansion, and `OMGC` deduplicated binary corpora. |

---

## The 4 Byte-Traceability Locks

1. **Lock 1 — Persistent Megakernel Parity (`gm_selftest`):** Double-buffered SASS slots produce identical SHA-256 code bytes and bit-exact weight/moment/loss parity against sequential stepping.
2. **Lock 2 — Epoch-Isolated Weight Staging (`grg_epoch_selftest`):** Grace CPU stages weight deltas into `cpu_bank` while the GPU reads `gpu_bank`; any mid-step mutation of the active bank or mid-step commit fails closed with `-8 / RING_ERR_EPOCH_VIOLATION (6)`.
3. **Lock 3 — Evolutionary JIT Replay Receipt (`ejit_selftest`):** Every promoted SASS buffer logs a SHA-256 receipt; deterministic replay reproduces identical buffer bytes, rejecting 1-bit SHA tampers (`err 20`) and out-of-order replays (`err 21`).
4. **Lock 4 — Exact Q32 Reversible Invertibility (`rev selftest`):** Integer Q30/Q32 reversible coupling achieves `0`-bit reconstruction error (`q30_inv_bits=0`, `model_q32_inv=0.00e+00`) and bit-exact deterministic backward passes (`DETBWD 0`).

---

## Zero-Python Verification Gate & 15-Second Unified Demo

Run the entire pure-Omega engine + 8-pillar self-tests + 4 byte-traceability locks + unified `omg pack -> train -> omg samp -> omc`, `arboris guide -> omc`, `fcache -> spforge -> spapp -> speval`, and `rev train` (ARM64) pipeline with **zero Python**:

```sh
sh scripts/verify-om.sh        # full pure-Omega zero-Python gate
sh scripts/demo-8-pillars.sh   # 15-second unified 8-pillar laptop demo
```

### Unified Pure-Omega CLI Commands (`ollm`)

```sh
ollm omg pack <in.om> <out.tok>                                    # pack .om -> V=256 OMG token stream
ollm omg decode <in.tok> <out.om>                                  # decode OMG tokens -> .om source
ollm omg samp <ck> <seed> <budget> <temp_milli> <out.om>           # OMG-masked neural decode (100% omc-valid)
ollm arboris guide <ck> <corpus.tok> <seed> <budget> <tm> <out.om> # 3-layer Arboris bypass + OMG wall + LTP0
ollm rev train <steps> [lr_milli]                                  # zero-activation Q32 reversible train (ARM64)
omega-helix pillars --json                                         # Studio CLI 8-pillar & lock verifier
```

### Studio UI & REST API Integration

Studio exposes the unified 8-pillar loop directly inside Tabs 2–4 and over `/api/runtime/{locks,omg,arbsamp,spforge}`:
- **Tab 2 (Design):** `OMG V=256 typed graph` tokenizer option, Closed-Form `spforge`, and Reversible Q32 Core toggles.
- **Tab 3 (Forge & Telemetry):** One-click **8-Pillar Locks** verifier (`OMG ✓ · Arboris ✓ · Rev Q32 (0-bit, 0B act) ✓`).
- **Tab 4 (Playground & Exports):** `OMG-Masked (100% omc-valid)` and `Arboris-Guided (Bypass + OMG + LTP0)` decode engines with bypass hit-rate, `LTP0` count, decoded `.om`, and live `omc` compile check.

---

## Studio SSH Remote & Theme Overhaul

[`tools/studio/`](../tools/studio/) adds Orca-style SSH remote discovery, pairing, and zero-FOUC light/dark/system theming (Python 3 stdlib only, zero telemetry):

```sh
omega-helix host list                    # discover ~/.ssh/config + saved hosts
omega-helix host add spark               # save an SSH target
omega-helix host probe spark             # probe remote GB10 GPU & Studio state
omega-helix remote connect spark         # bootstrap & SSH-tunnel remote Studio
omega-helix remote list                  # list saved remote Studio environments
```
