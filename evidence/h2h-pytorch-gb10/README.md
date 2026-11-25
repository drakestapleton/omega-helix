# Reproducible Head-to-Head: Omega Helix (Zero-CUDA SM121 SASS) vs. PyTorch 2.14 + CUDA 13.0

Measured in a single exclusive `spark-window` (`20261125T181639Z-eaa30f`) on one **NVIDIA DGX Spark (GB10, `sm_121`, CUDA 13.0, Driver 580 Open Modules)** on the exact same `15,833,472`-parameter v2 decoder-only Transformer shape (`V=4096, T=256, D=384, H=12, F=1536, L=8, B=64` — `16,384` tokens/step, RMSNorm no bias, causal SDPA/attention, ReLU MLP, tied token/output weights, BF16 mixed precision + AdamW).

## Measured Results (`omega_h2h.json` & `pytorch_h2h.json`)

| Metric | Omega Helix (`ollm-gb10 gtrainb`) | PyTorch `2.14.0+cu130` (Eager) | PyTorch `2.14.0+cu130` (`torch.compile`) |
|---|---:|---:|---:|
| **Runtime / CUDA Dependencies** | **0 CUDA / 0 cuBLAS / 0 Triton / 0 libc** (static AArch64 ELF) | Python 3.12 + PyTorch + NVIDIA CUDA 13.0 + cuBLAS + cuDNN | Python 3.12 + PyTorch + CUDA 13.0 + Triton 3.8 + Inductor |
| **Artifact / Disk Footprint** | **`791,988` bytes (~773 KiB)** | `5,325,112,578` bytes (~4.96 GiB packages) | `5,325,112,578` bytes (~4.96 GiB packages) |
| **Footprint Ratio vs. Omega** | `1×` | **`6,724×` larger** | **`6,724×` larger** |
| **Cold Start to Step 1** | `7,258.3 ms` *(includes loading & re-saving `380 MB` `f64` ckpt)* | **`2,646.8 ms`** *(in-RAM tensor init)* | `4,748.5 ms` *(in-RAM init + Inductor compile)* |
| **Steady-State Step Time (30 steps)** | `193.56 ms/step` (`5.8067 s` delta) | **`125.29 ms/step`** (`3.7587 s`) | **`63.64 ms/step`** (`1.9091 s`) |
| **Steady-State Throughput (tok/s)** | **`84,646.7 tok/s`** | **`130,769.2 tok/s`** *(1.54× Omega)* | **`257,463.6 tok/s`** *(3.04× Omega)* |
| **Peak CUDA Allocated / Reserved** | *Uses raw UVM/GPU IOCTL heaps directly (zero CUDA runtime)* | `2.08 GiB` alloc / `2.21 GiB` reserved | `1.65 GiB` alloc / `2.21 GiB` reserved |

## Notes & Transparency

- **What PyTorch wins cleanly:** PyTorch with cuBLAS/SDPA (`1.54×`) and `torch.compile` + Triton (`3.04×`) is significantly faster per step in steady-state GPU throughput and avoids disk checkpoint serialization during 1-step in-RAM benchmarks (`2.65 s` vs `7.26 s`, where `ollm-gb10` currently serializes/deserializes the full `380,003,448`-byte `f64` checkpoint to disk on start and finish).
- **What Omega Helix wins cleanly:** Single **`792 KB` static binary** (**`6,724×` smaller** than the 5.33 GB PyTorch + CUDA + Triton dependency tree), **zero** dynamic libraries (`ldd: not a dynamic executable`), hand-emitted SM121 SASS kernels directly from pure Omega, and **0 µnats** CPU-to-GPU eval parity across Rungs 6–10.

## Reproduce Yourself

```sh
# 1. Build pure-Omega GB10 binary & run 5 vs 35 steps (no intermediate ckpt writes)
TARGET=gb10 sh scripts/build.sh
./build/ollm-gb10 init base.ck 4096 256 384 12 1536 8 64 1 60000 3000 16
./build/ollm-gb10 gtrainb ck35.ck train.tok 35 35

# 2. Run matched PyTorch + CUDA BF16 benchmark
python3 tests/bench_pytorch_h2h.py --mode both --steps 30 --warmup 5 --measure-cold-start --out pytorch_h2h.json
```
