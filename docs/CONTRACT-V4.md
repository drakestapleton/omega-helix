# Omega Helix v4 Contract (`OLLMCK04` Universal Zero-Cap Engine & Weight Interchange)

v4 removes all static model shape caps and adds multi-architecture support, streaming `u8`/`u16`/`u32` token files, and bidirectional HuggingFace Safetensors & GGUF v3 weight interchange while keeping `OLLMCK01`/`02`/`03` untouched for backward compatibility.

Two independent implementations verify every rule:
1. **Pure seeded Omega (`native/univ/`):** `ollm-univ` (`TARGET=univ`) and `ollm-interchange` (`TARGET=interchange`).
2. **Independent Python-stdlib `f64` referees:** `tests/oracle_universal.py` and `tests/test_interchange.py`.

## 1. Shapes & Dynamic Zero-Cap Arenas

All array limits (`VCAP`, `TCAP`, `DCAP`, `HCAP`, `FCAP`, `LCAP`, `BCAP`, `OCKCAP`) are removed via `enlarge` dynamic arenas (`native/univ/uarena.om`) and `UCH = 500,000` chunked loops. Verified shapes exceed all v1–v3 bounds (including `V = 70,000` `u32` vocabularies and `V = 8,192, T = 512, D = 512, L = 20`, `52.97M` parameters).

Constraints enforced cleanly (`exit 12`):
- Every shape parameter (`V, T, D, H, H_kv, F, L, B`) is `> 0` and `V >= 2`.
- `D mod H == 0`, `H_kv <= H`, and `H mod H_kv == 0`.
- When RoPE is enabled (`arch_flags & 1 != 0`), head dimension `E = D / H` is even.
- Activation gates (`SwiGLU` bit `1` and `GeGLU` bit `6`) are mutually exclusive and mutually exclusive with plain `GELU` (bit `5`).
- Looped core (`NP > 1`): `LC > 0` and `LP + LC <= L`.

Token storage width (`tok_bytes`) is determined solely by `V`:
- `V <= 256`: `1` byte (`u8`)
- `257 <= V <= 65535`: `2` bytes (`u16` LE)
- `V > 65535`: `4` bytes (`u32` LE)

## 2. `arch_flags` Bitmask & Built-In Presets

| Bit | Value | Meaning |
|---:|---:|---|
| 0 | `1` | **RoPE** rotary embeddings (else learned absolute `WPE[T, D]`) |
| 1 | `2` | **SwiGLU** gated FFN (`W1`, `Wup`, `W2`) |
| 2 | `4` | **Untied LM head** (`lmh[V, D]` stored after `gf`; else weight-tied to `wte`) |
| 3 | `8` | **QK-RMSNorm** (`qn[E]`, `kn[E]` per layer) |
| 4 | `16` | **AdamW + Cosine LR** with linear warmup (else Adam + linear decay) |
| 5 | `32` | **GELU** (`tanh`-approx) instead of `ReLU` on non-gated FFN |
| 6 | `64` | **GeGLU** gated FFN (`gelu(W1 x) * (Wup x)`) |

Presets in `ollm-univ init`:
- `llama`, `mistral`: flags `23` (`1|2|4|16`), `rope_theta = 500000.0`, `rms_eps = 1e-5`, `wd = 0.1`, `warmup = 100`
- `qwen2`: flags `23` (`1|2|4|16`), `rope_theta = 1000000.0`, `rms_eps = 1e-6`, `wd = 0.1`, `warmup = 100`
- `qwen3`: flags `31` (`1|2|4|8|16`), `rope_theta = 1000000.0`, `rms_eps = 1e-6`, `wd = 0.1`, `warmup = 100`
- `gemma`: flags `81` (`1|16|64`, untied `wte = embed * sqrt(D)` + `lmh = embed` in `import-hf`), `rope_theta = 10000.0`, `rms_eps = 1e-6`, `wd = 0.1`, `warmup = 100`
- `gpt`: flags `0` (legacy GPT: `WPE`, `ReLU`, tied, Adam + linear LR), `rms_eps = 1e-5`, `wd = 0.0`, `warmup = 100`
- `gelu`: flags `32` (`WPE`, `GELU`, tied, Adam + linear LR), `rms_eps = 1e-5`, `wd = 0.0`, `warmup = 100`

## 3. Binary Format (`OLLMCK04` / `OLLMGR04`, 208-byte Header)

All header fields and `f64` payloads are little-endian:
- Magic constants (Wolfram): `"OLLMCK04"` = `3760588441286102095`, `"OLLMGR04"` = `3760596155047365711`.
- Payload length: `3 * 8 * P` bytes (`params[P]`, `m[P]`, `v[P]`) for `OLLMCK04`; `8 * P` bytes (`grd[P]`) for `OLLMGR04`.

| Offset | Field |
|---|---|
| `0..7` | Magic (`OLLMCK04` or `OLLMGR04`) |
| `8..63` | `V, T, D, H, H_kv, F, L, B` |
| `64..95` | `seed`, `step` (`B` in `OLLMGR04`), `total`, `lr_bits` |
| `96..111` | `P` (`param_count`), `arch_flags` |
| `112..143` | `rope_theta_milli`, `rms_eps_bits`, `wd_bits`, `warmup` |
| `144..167` | `LP`, `LC`, `NP` |
| `168..199` | Reserved (`0`, future extensions) |
| `200..207` | FNV-1a 64 checksum over payload bytes |

Math constants in `native/univ/umath.om` (`ufexp`, `uflog`, `ufsin`, `ufcos`, `ufgelu`, `ufgelu_d`) are derived via `WolframLanguageEvaluator`, including Cody-Waite `pi/2` (`pi_2_hi = 1.5707963267341256`, `pi_2_lo = 6.0771005065061926e-11`, `inv_pi_2 = 0.63661977236758134`) and Taylor coefficients `1/3!..1/15!`, `1/2!..1/14!`.

## 4. Universal Interchange (`ollm-interchange`, `TARGET=interchange`)

- `import-hf CONFIG.json MODEL.safetensors OUT.ck [T B TOTAL LRMICRO]` — imports `llama`, `mistral`, `qwen2`, `qwen3`, `gemma`, `helix` (`F32`, `BF16`, `F16`, `F64`), transposing HF `[out, in]` linear projections to Helix `[in, out]`, failing closed (`exit 80..88`) on malformed JSON, unsupported features, bad shapes, truncated files, or NaN/Inf weights.
- `export-hf IN.ck OUT_DIR [f32|bf16]` — writes `OUT_DIR/config.json` via `jsonw.om` and `OUT_DIR/model.safetensors` (`OUT_DIR` must exist).
- `export-gguf IN.ck OUT.gguf` — writes GGUF v3 (`magic = 1179993927`, `version = 3`, 20 KV metadata entries, 32-byte aligned `F32` tensors, `convert_hf_to_gguf` `Q/K` row permutation for `llama`).
- `upgrade LEGACY.ck OUT.ck` — upgrades legacy `OLLMCK01`/`02`/`03` checkpoints bit-for-bit to `OLLMCK04`.
