# GB10 Bare-Metal `/dev/aien-gpu` Fused SM121 NVFP4 Training Receipt (2026-11-26)

- **Hardware:** NVIDIA DGX Spark (GB10, SM121, 48 SMs, 24 MiB L2, 273 GB/s nominal / 236.9 GB/s measured LPDDR5X, 20 Grace ARMv9 cores)
- **Driver / Stack:** Pure seeded-Omega (`omc`) over bare-metal `/dev/aien-gpu` (`aien_gpu.ko`, **zero `nvidia.ko`**, **zero CUDA**, **zero Python** in the training path)
- **Measurement Window:** `spark-window` exclusive chip + quiet lock `20261126T232906Z-724174` (raw host state in `window_open.json`)
- **Binary:** `build/ollm-gb10` (`sha256=664d4e7041c105d9e5d3cadc82da8ab3deb9f15857f1c4aca85684dd0c7285f7`)
- **Static Disassembly Gate:** `sh tests/gpu-disasm.sh --all` -> `GPU_DISASM_OK slots=86 bad=0` (`gpu_disasm.txt`)
- **Live Test Gate:** `OLLM_GB10_LIVE=1 python3 -m unittest tests/test_gpu_nvfp4_train.py -v` -> `OK` (`test_gpu_nvfp4_train_live.txt`)
- **Full Pure-Omega Gate:** `sh scripts/verify-om.sh` -> `OMEGA_HELIX_PURE_OM_PASS` (all 13 gates + 4 byte-traceability locks)

## What Changed in Pure `.om` (`native/gpu/{gk,gr,gg,gb}.om`)

1. **Coalesced `[K/64, M, 32 B]` + `[K/64, M, 4 B]` NVFP4 Layout + 0-Conflict `LDS.128` + 2x-Unrolled K-Loop (`gkgemmnv4_128`, `grqnv4`, `grqnv4t`):**
   - Replaced strided row-major `LDG` (8x scale sector overfetch) with 100% coalesced `k_blk`-major `LDG.E.128`/`LDG.E.32` and straight-line `(M/8, K/64, 1)` `slot 88`.
   - Replaced 24 bank-conflicted `LDS.32` with 6 bank-conflict-free `LDS.128` (`lane * 16`), 2x-unrolled ping-pong SMEM stages, and hid `ISETP` predicate latency behind `OMMA.SF.16864`.
   - **Standalone `gnv4gemm` Result:** `4096x4096x4096` jumped from `113.54 TF` -> **`217.79 TF`** (`+91.8%`); `4096x4096x8192` reached **`239.07 TF`** (`265.0 TF` K-loop, **`2.11x`**), all bit-exact `PASS`.
2. **Persistent Pre-Quantized `W_nv4` / `W_nv4^T` Weight Cache (`ggquant_w`, `gga[48..51]`):**
   - Quantizes physical layer weights once after `ggf2b`/`ggadam` instead of re-quantizing static weights on every forward and backward GEMM (reused across all passes in Looped models).
3. **Full Split-K NVFP4 `dW` (`ggsk_nv4`, `ggdw_nv4_ok`) + Convergence Parity (`gggrad_sc`, BF16 LM-head):**
   - Enabled split-K NVFP4 `dW` on compute-dense shapes (`m * n >= 786432`: `D >= 768/1024/2048`) while keeping BF16 `dW` on `D=384` where standalone transpose-quant exceeds BF16 `slot 73`.
   - Kept LM-head backward in BF16 (standard Blackwell FP4 recipe) with dynamic `gggrad_sc = max(4096, 2*R)`, improving `gnv4cmp` full-step CPU `f64` gradient cosine from `0.980908` -> **`0.987541`** (`dloss=0.000597`, `0` NaNs) and cutting `15.83M B=64` step-35 loss drift from `+1.52 nats` to **`+0.0038 nats`** (`5.8737` vs `5.8699`).
4. **Fused Producer-Epilogue NVFP4 Quantization (`slot 75` `RMSF_BF16`, `slot 94` `W1+ReLU`, `slot 95` `dF+ReLU`):**
   - Eliminated standalone `slot 88` DRAM round-trips on `a`, `bn`, `rl` (`50.3 MB`), and `dF` (`50.3 MB`) by quantizing directly out of live registers / dead epilogue SMEM into race-free scratch (`gga1 41/42` and `gga1 43/44`).

## Measured Silicon Results (`window 20261126T232906Z-724174`)

### Standalone SM121 NVFP4 GEMM (`gnv4gemm`, `gnv4gemm_and_cmp.txt`)

| Shape (`M x N x K`) | Baseline `gkgemmnv4_128` | **Upgraded `gkgemmnv4_128`** | Speedup | Status |
|---|---:|---:|---:|---|
| `2048 x 1536 x 384` | `19.00 TF` | **`27.22 TF`** | `1.43x` | `PASS` |
| `2048 x 4096 x 1024` | `69.12 TF` | **`104.18 TF`** | `1.51x` | `PASS` |
| `2048 x 2048 x 8192` (L2-resident) | — | **`200.49 TF`** | — | `PASS` |
| `4096 x 4096 x 4096` | `113.54 TF` | **`217.79 TF`** | `1.92x` | `PASS` |
| `4096 x 4096 x 8192` | — | **`239.07 TF`** (`265.0 TF` K-loop) | `2.11x` | `PASS` |

### Matched Full-Step Training H2H (`gtrainb` BF16 vs `gtrain4` Fused NVFP4, `h2h_bf16_vs_nvfp4.txt`)

| Model & Shape | `gtrainb` (BF16) | **`gtrain4` (Fused NVFP4)** | Full-Step Speedup | Fwd+Bwd Speedup | Final Loss (`BF16` vs `NVFP4`) |
|---|---:|---:|---:|---:|---|
| `15.83M` `v2_b64` (`D=384, L=8, B=64`, 30 steps) | `70.13 ms` (`233,620 tok/s`, `22.05 TF`) | **`64.54 ms` (`253,876 tok/s`, `23.96 TF`)** | **`+8.7%`** | `+4.5%` | `5.870` vs `5.874` (`+0.004`) |
| `15.83M` `loop2x4_b64` (`L_phys=2x4, B=64`, 30 steps) | `67.66 ms` (`242,152 tok/s`, `22.85 TF`) | **`62.32 ms` (`262,909 tok/s`, `24.81 TF`)** | **`+8.6%`** | `+6.3%` | `6.063` vs `6.098` (`+0.035`) |
| `31.5M` `d768_l4_b32` (`D=768, L=4, B=32`, 20 steps) | `53.75 ms` (`152,404 tok/s`, `28.77 TF`) | **`47.08 ms` (`174,006 tok/s`, `32.85 TF`)** | **`+14.2%`** | `+19.5%` | `5.914` vs `5.892` (`-0.022`) |
| `54.5M` `d1024_l4_b32` (`D=1024, L=4, B=32`, 20 steps) | `82.75 ms` (`98,997 tok/s`, `32.39 TF`) | **`67.80 ms` (`120,819 tok/s`, `39.53 TF`)** | **`+22.1%`** | `+30.8%` | `5.642` vs `5.632` (`-0.010`) |
| `58.7M` `d2048_l1_b16` (`D=2048, L=1, B=16`, 20 steps) | `48.16 ms` (`85,045 tok/s`, `29.96 TF`) | **`37.35 ms` (`109,665 tok/s`, `38.64 TF`)** | **`+29.0%`** | **`+49.6%` (`1.50x`, `54.6 TF`)** | `5.600` vs `5.569` (`-0.031`) |

## Roofline Reality Check on Unmeasured `1.05M` / `3.68M tok/s` Projections

Per-slot profiling (`gprofb` on `15.83M`, `D=384, L=8, B=64, T=256`, verified via Wolfram) establishes why the earlier analytical projections of `1.05M tok/s` (`100 TF` BF16 = `15.46 ms/step`) and `3.68M tok/s` (`350 TF` NVFP4 = `4.42 ms/step`) are GEMM-only upper bounds (`6 * P * M / TF`) rather than achievable full-step `15.83M` Transformer speeds on current kernels:

- Non-GEMM operators on `15.83M B=64` currently consume **`24.33 ms/step`** by themselves (`FA2` `11.58 ms` + `RMSNorm` `5.80 ms` + `CE_BF16` `1.44 ms` + `AdamW` `2.31 ms` + `Emb/SQN/SplitK-Reduce` `3.20 ms`).
- Even with hypothetical `0.00 ms` GEMMs, those non-GEMM kernels cap `15.83M B=64` at `16,384 / 0.02433 = 673,407 tok/s`.
- At `B=64`, intermediate FFN tensors (`H[16384, 1536] = 50.3 MB`) and logits (`134.2 MB`) exceed GB10's `24 MiB` L2 cache (`2.1x` and `5.6x`), pinning DRAM-bound ops to `236.9 GB/s` LPDDR5X.
- Pushing beyond `~263k tok/s` toward `~500k–900k+ tok/s` on `15.83M` requires a next-stage architectural pass fusing/accelerating `FA2` (`11.58 ms`), `RMSB` (`4.12 ms`), and `dW` (`13.47 ms`) with chunked-M L2-resident layer recomputation / activation compression.
