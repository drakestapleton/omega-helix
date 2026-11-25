# Omega Helix Sovereign Runtime Capabilities (Phase 7)

This layer turns `OLLMSP01` Merkle chains, `LTP0` looped-core refinement, pure seeded-Omega grammar masking, and HuggingFace weight interchange (`import-hf`) into four composable, cryptographically verifiable workflows:

1. **Anti-DNA Runtime Firewalls & Cryptographic Token Receipts (`ollm spclip` signed `MODE=0` + `ollm spguard`)**
   - `spclip` (`MODE=0`) and `spguard` accept negative milli-weights (`-5000 .. +5000`) normalized by `sum_{w_s > 0} w_s`: a strand weighted `-500` subtracts $0.5 \log P_{\text{anti}}(v)$ from the fused log-probabilities at decode time without retraining or modifying the base trunk.
   - `ollm spguard` mixes `base_hash`, every strand's signed `w_milli` (`bitcast ulong`) and `pack_hash` chain, and each drawn token tuple `(step, tok, jsd_milli, extra_passes, contested, anti_active)` into a running 64-bit FNV-1a receipt chain verifiable offline via `tools/runtime/anti_dna.py:verify_firewall_receipt`.

2. **Strand Disagreement $\rightarrow$ Automatic `LTP0` Deep Thinking (`spguard`)**
   - At every token, `osp_clip_jsd` computes exact Jensen-Shannon Divergence $\text{JSD} = H(\bar{p}) - \frac{1}{|P|}\sum_{s \in P} H(p_s)$ across all positive-weight strands (verified against Wolfram to $2.22 \times 10^{-16}$).
   - When $\text{JSD} \ge \text{DISAGREE\_MILLI} / 1000$, `spguard` flags the token (`contested=1`) and runs up to `MAX_EXTRA` (`0..4`) extra latent refinement passes (`osp_guard_pass` -> `okvlay`) over the shared trunk before re-evaluating all strands.

3. **Zero-Overhead Grammar-Constrained Decoding (`ollm-univ gsamp`)**
   - Implemented purely in [`native/univ/ugrammar.om`](../native/univ/ugrammar.om) directly inside `ollm-univ`'s KV-cached sampler (`ustep_kv`):
     - Mode `1`: RFC-8259 JSON objects (`{"key":"val",...}`, budget-aware closing guarantee).
     - Mode `2`: ASCII identifiers (`[A-Za-z_][A-Za-z0-9_ ]*`).
     - Mode `3`: CSV integers (`[0-9]+(,[0-9]+)*`).

4. **Verifier-Gated Sleep Cycles, Compile-Until-Green & Open-Trunk DNA (`tools/runtime/`)**
   - [`sleep_cycle.py`](../tools/runtime/sleep_cycle.py): Filters interactions via `VerifierPipeline`, consolidates via `growpro` + `fcache` + `trainfc` + `spext` into `day-<date>.sp`, and rolls back / quarantines automatically if held-out regression exceeds `max_regression_nats`.
   - [`compile_green.py`](../tools/runtime/compile_green.py): Iterates `adaptsamp`/`sample` + deterministic verification + ephemeral `tttsamp` micro-adaptation on failure diagnostics before crystallizing passing traces into `crystallized.sp`.
   - [`hf_dna.py`](../tools/runtime/hf_dna.py) + [`scripts/demo-10s-wow.sh`](../scripts/demo-10s-wow.sh): Imports HF Safetensors (`ollm-interchange import-hf`), forges domain specialists + an Anti-DNA strand, composes them via `spguard`, and verifies every token receipt with zero CUDA.
