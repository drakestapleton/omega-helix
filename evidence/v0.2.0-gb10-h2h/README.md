# Reproducible Head-to-Head & Live GB10 Verification (`v0.2.0`, Phase 1 `sm_121` Stack)

Measured on **NVIDIA DGX Spark (Grace + GB10 `sm_121`, Driver `580.173.02`, Linux `7.0.0-1019-nvidia` `aarch64`, CUDA `13.0` comparison arm only)** against `github/main` commit `2ad71e9` (including `9b66406` Phase 1 `128×128` packed-BF16 GEMMs, Fused Flash-Attention-2, `CE_BF16`, CTA-reduced `RMSB_BF16`, 256-CTA `SQN`, and live `OMMA.SF.16864` NVFP4).

- **Window 1 (Exclusive Quiet Timing H2H):** `window_id = 20261126T031444Z-9d174e` (`--gb10 exclusive --quiet exclusive`, `spark_receipt_h2h.json`, `120.605 s`, `wall_W_mean = 175.34 W`, `gpu_W_mean = 9.31 W`, `lock_holders_max = 1`, `nvrm_errors_delta = 8` — 0 Xids / 0 GPU faults, strictly two `kgrctxAllocCtxBuffers` `NV_ERR_NO_MEMORY (0x51)` big-page retry fallbacks during PyTorch context init).
- **Window 2 (Live GB10 Pillars, Full-APU `apuweave`, Spiral-2048 & `verify.sh`):** `window_id = 20261126T031723Z-957b70` (`--gb10 exclusive --quiet none`, `spark_receipt_pillars.json`, `386.718 s`, `wall_W_mean = 176.18 W`, `gpu_W_mean = 5.51 W`, `lock_holders_max = 1`, `nvrm_errors_delta = 118` — 0 Xids / 0 GPU faults, `NV_ERR_NO_MEMORY (0x51)` big-page retry fallbacks only).

---

## 1. Matched BF16 Head-to-Head (`15.83M` v2 Shape, `omega_h2h_v020.json` & `pytorch_h2h_v020.json`)

Exact same `15,833,472`-parameter v2 decoder-only Transformer (`V=4096, T=256, D=384, H=12, F=1536, L=8, B=64` — `16,384` tokens/step, RMSNorm no bias, causal SDPA/FA2, ReLU MLP, tied token/output weights, BF16 mixed precision + AdamW):

| Metric | Omega Helix `v0.1.0` (`64×64` FP32-IO) | Omega Helix `v0.2.0` (`9b66406` Phase 1) | PyTorch `2.14.0+cu130` (Eager) | PyTorch `2.14.0+cu130` (`torch.compile`) |
|---|---:|---:|---:|---:|
| **Runtime / CUDA Dependencies** | **0 CUDA / 0 cuBLAS / 0 Triton / 0 libc** | **0 CUDA / 0 cuBLAS / 0 Triton / 0 libc** | CUDA 13.0 + cuBLAS + cuDNN | CUDA 13.0 + Triton 3.8 + Inductor |
| **Artifact / Disk Footprint** | `791,988 B` (~773 KiB) | **`1,581,312 B`** (~1.51 MiB static ELF) | `5,325,112,578 B` (~4.96 GiB) | `5,325,112,578 B` (~4.96 GiB) |
| **Footprint Ratio vs. `v0.2.0`** | — | **`1×`** | `3,367.5×` larger | `3,367.5×` larger |
| **Cold Start to Step 1** | `7,258.3 ms` *(380 MB ckpt I/O)* | `7,938.7 ms` *(380 MB `f64` ckpt I/O)* | `2,030.2 ms` *(in-RAM init)* | **`1,561.1 ms`** *(in-RAM + cache)* |
| **Steady-State Step Time (30 steps)** | `193.56 ms/step` (`5.8067 s`) | **`62.96 ms/step`** (`1.8887 s`) | `121.08 ms/step` (`3.6324 s`) | **`62.45 ms/step`** (`1.8734 s`) |
| **Steady-State Throughput (tok/s)** | `84,646.7 tok/s` | **`260,243.3 tok/s`** | `135,315.6 tok/s` | **`262,371.0 tok/s`** |
| **vs. Omega `v0.1.0` (`84,646.7`)** | `1.000×` | **`3.074×` faster** | `1.599×` | `3.100×` |
| **Omega `v0.2.0` vs. Arm** | **`3.074×` faster** | `1.000×` | **`1.923×` faster** (`-58.12 ms`) | **`99.19%` parity** (`+0.51 ms`) |

- `build/ollm-gb10` SHA-256: `9db3df052e716e6fd6e7e74b803b464c90c2cb48fe744a0da246f0a74fd692a7` (`file`: statically linked AArch64 ELF, `ldd: not a dynamic executable`).
- Note that Omega Helix `v0.2.0` (`260,243.3 tok/s`) surpasses the `v0.1.0` window's `torch.compile` baseline (`257,463.6 tok/s`) by `1.011×` and lands within **`0.51 ms/step` (`0.81%`)** of `torch.compile` measured inside the same Window 1 (`262,371.0 tok/s`).

---

## 2. Live SM121 `OMMA.SF.16864` NVFP4 Tensor Core GEMM (`nvfp4_h2h_v020.json`)

Measured inside Window 1 on `4096 × 4096 × 8192` (`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`, slot 84, `500 TF` GB10 NVFP4 hardware ceiling):

| Arm | Shape (`M × N × K`) | Iters | Wall Time | Throughput | % Hardware Peak | Parity Check |
|---|---|---:|---:|---:|---:|---|
| **Omega Helix `nvfp4` (20 iters)** | `4096 × 4096 × 8192` | `20` | `11.975 ms` | **`459.09 TFLOPS`** | **`91.81%`** (of 500 TF) | `expected=8192 got=8192 PASS` |
| **Omega Helix `nvfp4` (50 iters)** | `4096 × 4096 × 8192` | `50` | `30.177 ms` | **`455.43 TFLOPS`** | **`91.08%`** (of 500 TF) | `expected=8192 got=8192 PASS` |
| **PyTorch `2.14.0+cu130` cuBLAS BF16 `matmul`** | `4096 × 4096 × 8192` | `20` | `64.048 ms` | `85.84 TFLOPS` | — | Reference (`5.35×` slower than Omega NVFP4) |

---

## 3. Root-Cause `gprofb` Breakdown of the `0.51 ms` Gap vs. `torch.compile` (`gprofb_v020.txt`)

Per-kernel GPU launch profile on step 5 (`T 5 7251558`, `gprofb_v020.txt`, `241` step kernel launches summing to `64.644 ms` under synchronous host `wait_ns` profiling vs. `62.96 ms/step` pipelined, plus `64` bare `ZERO` round-trip probes averaging `6.29 µs/launch`):

| Kernel Slot | Role | Launches | Sum Wait (`ms`) | Share | Mean (`µs`) |
|---|---|---:|---:|---:|---:|
| `73` | `128×128` Packed-BF16 split-K `dW` GEMM | `32` | `11.974` | `18.52%` | `374.18` |
| `70` | `128×128` Packed-BF16 `dF` backward (ReLU ep=2) | `8` | `5.678` | `8.78%` | `709.71` |
| `68` | `128×128` Packed-BF16 `X W + β C` (`Wo`/`W2`) | `16` | `5.220` | `8.07%` | `326.22` |
| `71` | `128×128` Packed-BF16 `dX W^T` backward | `16` | `4.553` | `7.04%` | `284.59` |
| `67` | `128×128` Packed-BF16 `X W` (`QKV`/`W1`/logits) | `9` | `4.225` | `6.54%` | `469.39` |
| `69` | `128×128` Packed-BF16 `ReLU` ep=1 forward | `8` | `4.130` | `6.39%` | `516.29` |
| `76` | CTA-reduced `RMSB_BF16` backward | `17` | `4.052` | `6.27%` | `238.33` |
| `80` / `79` / `78` / `77` | Fused BF16 Flash-Attention-2 (`dKV`, `dQ`, delta, fwd) | `32` | `9.583` | `14.82%` | `299.47` |
| `85` | Fused 3-way QKV backward (`dA = dQ Wq^T + dK Wk^T + dV Wv^T`) | `8` | `2.550` | `3.95%` | `318.79` |
| `87` / `86` / `81` | `CE_BF16` (`2.306 ms`), `dLogits` (`1.920 ms`), `AdamW_BF16` (`2.154 ms`) | `3` | `6.380` | `9.87%` | `2,126.67` |
| `75` | `RMSF_BF16` forward | `17` | `1.754` | `2.71%` | `103.15` |
| `14` / `13` / `20` / `82` / `83` / `74` / `22` / `23` / `9` | Auxiliary `COLSUM` (`50×`, `1.172 ms`), `gain` (`17×`, `0.368 ms`), `ZERO`, casts, scatter/emb | `75` | `4.544` | `7.03%` | `60.59` |

**Measured root cause of the remaining `0.51 ms` (`0.81%`) vs. `torch.compile`:**
Inductor fuses auxiliary reductions/casts and uses CUDA Graphs, whereas `ollm-gb10 gtrainb` currently issues `241` sequential QMD submissions per step — including `50` `COLSUM` (`slot 14`, `1.172 ms`), `17` `gain` (`slot 13`, `0.368 ms`), and `2` F32↔BF16 cast launches (`slots 82/83`, `0.316 ms`) plus `~6.29 µs` per-launch host dispatch overhead across auxiliary ops (`~1.52 ms` across 241 launches vs. single-graph submission). Fusing `COLSUM`/`gain` epilogues or batching QMD chains (`batch.om`) directly closes the remaining sub-millisecond margin.

---

## 4. Window 2 Live GB10 Verification (`pillars_live_summary.txt`)

- `scripts/verify-om.sh`: **`OMEGA_HELIX_PURE_OM_PASS`** (`BTLOCK_OK`, `SPIRAL_WEAVE_OM_PASS`, `APU_WEAVE_OM_PASS`).
- `scripts/verify.sh`: **`OMEGA_HELIX_VERIFY_ALL_PASS`** (`STATIC_ELF_ZERO_CUDA_OK`, `GPU_DISASM_OK slots=78 bad=0`, Lean 4 `Budget.lean` clean, 165-task coding benchmark `CHECK PASS`, Contrast & Controller replays OK, OpenWALDO `25/25`, Studio `12/12`, Distill `7/7`, Runtime `5/5`, Commons `7/7`).
- `OLLM_GB10_LIVE=1` live GB10 silicon tests (`test_gpu_spiral_gb10.py`, `test_apu_weave.py`, `test_spiral_weave.py`): **`Ran 8 tests in 217.869s — OK`** (`0` skipped):
  - **Live GB10 `gbfrag`:** exact `0`-error check across all 10 `64×64` + 10 `128×128` BF16 HMMA kernels (`PASS`).
  - **Live GB10 `gacmp`:** 256-CTA 128-bit `SQN` + 256-thread 128-bit `ADAM` bit-exact parity (`PASS`).
  - **Live GB10 `nvfp4`:** `OMMA.SF.16864` Tensor Core execution (`PASS`).
  - **Live GB10 `gbcmp2`:** Looped Spiral (`L_phys=8 × NP=4`, `LV=32`) and `D=2048` (`n=64` tiled RMS + direct-`oww`) parity (`PASS`).
  - **Live GB10 `gfcache`:** `OLLMFH01` GPU activation cache parity vs. CPU `fcache` (`max_diff < 1e-4`, `PASS`).
  - **Live GB10 Full-APU `apuweave`:** simultaneous GPU + 20-core Grace multi-worker co-execution (`gpu_mode > 0`, `spweval` loss improvement, 1-bit tamper `rc=61`, plus Linux DRA-1023/1024 `getdents64`/`stat` regression tests — `PASS`).
  - **Live GB10 `ollm-gb10` Pillar CLIs:** `gtern` (`GTERN_PASS`), `omg` (`OMG_PASS`), `arboris` (`ARBORIS_PASS`), `rev selftest` (`REV_PASS`, `Q32 0`, `DETBWD 0`).
  - **Honest scope boundary:** Pillar 1 (`gm.om` persistent megakernel), Pillar 3 (`ring.om` raw NValloc GPU execution), Pillar 4 (`gt.om` ternary/4-bit LUT-GEMM GPU launch), and Pillar 6 (`ejit.om` live GPU hot-swap) remain **`CPU-SIM + NVDISASM (LIVE GPU NOT_RUN)`**; calling `ollm-gb10 gmega` without opening `/dev/nvidia0` stops at `gm_launch_persistent` (`rc=18`), while its CPU-sim + SASS audit (`BTLOCK_OK` & `test_megakernel.py`) passes.

---

## 5. Reproduce Yourself

```sh
TARGET=gb10 sh scripts/build.sh
./build/ollm-gb10 init base.ck 4096 256 384 12 1536 8 64 1 60000 3000 16
./build/ollm-gb10 gtrainb ck5.ck train.tok 5 5
./build/ollm-gb10 gtrainb ck35.ck train.tok 35 35
./build/ollm-gb10 gprofb base.ck train.tok 5
./build/ollm-gb10 nvfp4 4096 4096 8192 20
python3 tests/bench_pytorch_h2h.py --mode both --steps 30 --warmup 5 --measure-cold-start --out pytorch_h2h_v020.json

# Inside an exclusive spark-window:
OLLM_GB10_LIVE=1 python3 -W error::ResourceWarning -m unittest -v \
  tests/test_gpu_spiral_gb10.py tests/test_apu_weave.py tests/test_spiral_weave.py
```
