# Community Hardware-Porting Kit (`omega-helix port-check`)

Omega Helix separates the **frozen transformer + DNA contract** ([`CONTRACT.md`](CONTRACT.md), [`CONTRACT-V4.md`](CONTRACT-V4.md)) from silicon-specific execution kernels so community contributors can bring up **Metal**, **Vulkan**, **AMD ROCm / HIP**, **Intel oneAPI / Arc**, **consumer NVIDIA RTX (`sm_89` / `sm_120`)**, or **RISC-V / ARM** backends in pure Omega (`.om`) with bit-level proof instead of guesswork.

---

## 1. Frozen Kernel Contract

Every backend candidate must match the reference `f64` math (`tests/oracle.py`) and binary formats bit-for-bit within IEEE-754 / BF16 tolerances:

1. **Checkpoint formats:** `OLLMCK01`/`02`/`03` (`native/ckpt.om`) and zero-cap `OLLMCK04` (`native/univ/`).
2. **DNA specialist delta format:** `OLLMSP01` (`base_L`, `spec_L`, `base_hash`, `n_spec`, `pack_hash = fnv1a64(payload)`).
3. **AdamW + Cosine LR determinism:** Exact step-by-step parity verified by `ollm adamck` and `tests/oracle.py` `xinit`/`xgrad`/`xstep`/`xeval`.
4. **KV-cache + autoregressive sample parity:** Prefill + decoded step parity verified by `ollm kvcmp`.
5. **BF16 / Mixed-Precision hardware check (optional GPU tier):** On accelerators with hardware BF16 (`sm_121`, `sm_89`, `sm_120`, Metal BF16), `gbcmp` and `gbcmp2` check per-tensor cosine similarity `>= 0.9999` with `0` unats/inf/NaN traps.

---

## 2. One-Command 9-Gate Verification (`port-check`)

Run all 9 automated CPU/oracle gates against any candidate binary:

```sh
./bin/omega-helix port-check --bin ./build/ollm --out-json ./port-parity-receipt.json --json
```

The Python harness (`tools/commons/porting_kit.py`) runs:

| # | Gate | What it proves |
|---|---|---|
| 1 | `adamck` | AdamW momentum + variance update bit accuracy |
| 2 | `init` | Canonical `OLLMCK` layout & deterministic seed initialization |
| 3 | `gcheck` | Finite-difference gradient parity against analytic backward pass |
| 4 | `kvcmp` | Full prefill vs step-by-step KV cache equivalence |
| 5 | `oracle_xprep` | Canonical tokenization & held-out data layout |
| 6 | `oracle_xinit` | Exact parameter equality against independent Python `f64` oracle |
| 7 | `oracle_xgrad` | Analytic backward gradients equal independent `f64` oracle |
| 8 | `oracle_xstep` | Optimizer parameter update equals independent `f64` oracle |
| 9 | `oracle_xeval` | Held-out NLL loss equals independent `f64` oracle |

---

## 3. Submitting a Community Hardware Receipt

1. Run `omega-helix port-check --bin <candidate> --out-json evidence/<arch>/port-parity-receipt.json`.
2. Verify linkage posture (`ldd` static / zero-CUDA on Linux, `otool -L` clean on macOS, or Metal/Vulkan-only ICD when explicitly targeting a vendor GPU backend) with `./bin/omega-helix verify --json`.
3. Generate a deterministic `SHA256SUMS.txt` alongside your hardware receipt and include the exact driver/OS version and disassembled kernel stats (if GPU).
