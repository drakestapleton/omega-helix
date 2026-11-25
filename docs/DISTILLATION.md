# Omega Helix Sovereign Distillation Pipeline Network (`OLLMDK01`)

Omega Helix includes a zero-dependency, offline-first distillation pipeline spanning:

1. **Pure Seeded-Omega Binary Layer ([`native/univ/udistill.om`](../native/univ/udistill.om))** compiled into `ollm-univ` (`TARGET=univ`):
   - `dkemit CK TOKENS.bin OUT.dk K TEMP_MILLI [MAX_SEQS]`: emits sparse top-`K` teacher probabilities at temperature $\tau = \text{TEMP\_MILLI}/1000$ into an FNV-1a-authenticated `OLLMDK01` file.
   - `dcheck CK DK.bin ALPHA_MICRO TEMP_MILLI [NSAMP]`: central finite-difference gradient check comparing the analytic masked hybrid CE+KL gradient against $(L(\theta+h)-L(\theta-h))/(2h)$ (`exit 40` on relative error $> 10^{-3}$).
   - `dtrain CK DK.bin STEPS EVAL_EVERY ALPHA_MICRO TEMP_MILLI`: AdamW/cosine optimizer minimizing masked hybrid CE+KL loss, printing `D <step> <total_micro> <ce_micro> <kl_micro>` and saving `OLLMCK04` atomically.
2. **Sovereign Python 3 Stdlib Network ([`tools/distill/`](../tools/distill/)):**
   - **Harvester ([`harvest.py`](../tools/distill/harvest.py)):** ingests OpenAI/Anthropic chat JSONL (including array-of-parts content), ShareGPT conversations, instruction/response JSONL, OpenWALDO `EXPORT.json` shards, and code/text trees; deduplicates by `sha256(text)`; tracks assistant turn byte spans; exports verified OpenWALDO `EXPORT.json` + `MASKS.jsonl` and binary `OLLMDK01` shards.
   - **Fail-Closed Verifiers ([`verifiers.py`](../tools/distill/verifiers.py)):** `nonempty`, `json_schema`, `python_ast` (`ast.parse` only, never `exec`), `omc_compile` (compile-only via stdin -> `/dev/null`, never executes untrusted code), and `regex`.
   - **Multi-Teacher Router ([`teachers.py`](../tools/distill/teachers.py)):** `OpenAICompatTeacher` (local `omega-router` `:18090`, `q36-serve` `:18091`, `ollm` `:18098`, Ollama `:11434`, vLLM `:8000`, llama.cpp `:8080`, LM Studio `:1234`, requesting `top_logprobs` when supported), `LocalHelixTeacher` (`ollm-univ sample` / `dkemit`), `TraceReplayTeacher`, `TeacherRouter` (`round_robin`, `ensemble_best_of`, `fallback`), and `probe_local_teachers`.
   - **Pipeline Orchestrator ([`pipeline.py`](../tools/distill/pipeline.py)):** drives `dk-univ` (sparse top-K CE+KL), `seq-univ` (verified OpenWALDO sequence distillation via `ollm-univ`), and `dna-strand` (Merkle specialist `OLLMSP01` extraction via `ollm`), always writing `distill-bom.json`.
3. **Studio Tab 5 & CLI:**
   - Web Studio Tab 5 (`Distill Network`) in [`tools/studio/ui/`](../tools/studio/ui/) with `/api/distill/*` SSE telemetry endpoints.
   - CLI `omega-helix distill {probe,harvest,run,status}` in [`tools/studio/cli.py`](../tools/studio/cli.py).

---

## Hybrid CE + KL Objective (`OLLMDK01`)

For each unmasked position ($\text{mask}[b, p] = 1$) across a batch of $M = \sum_{b,p} \text{mask}[b,p]$ supervised tokens:

$$
L_{\text{CE}}(b, p) = -\log p_S(y), \qquad
L_{\text{KL}}(b, p) = \tau^2 \sum_{j=0}^{K-1} q_j \left(\log q_j - \log p_S^{(\tau)}(\text{id}_j)\right),
$$

$$
L = \frac{1}{M} \sum_{\text{mask}[b,p]=1} \Big[(1 - \alpha)\, L_{\text{CE}}(b, p) + \alpha\, L_{\text{KL}}(b, p)\Big].
$$

Verified via `WolframLanguageEvaluator` (`2026-11-25`, residual `[0, 0, 0]`): letting $q_{\text{sum}} = \sum_{j=0}^{K-1} q_j$, the exact gradient with respect to student logit $z_c$ is

$$
\frac{\partial L_{\text{pos}}}{\partial z_c} = (1 - \alpha)\big(p_S(c) - \mathbf{1}_{c = y}\big) + \alpha \tau \left(q_{\text{sum}}\, p_S^{(\tau)}(c) - \sum_{j : \text{id}_j = c} q_j\right).
$$

Because unnormalized sparse top-$K$ mass satisfies $q_{\text{sum}} \le 1$, scaling $p_S^{(\tau)}(c)$ by $q_{\text{sum}}$ keeps the analytic gradient exact (`dcheck` max relative error $< 10^{-3}$) whether or not the top-$K$ teacher weights are renormalized to $1$.

---

## `OLLMDK01` Binary Contract

- **Magic (`u64` LE):** `3544415663467285583` (`"OLLMDK01"`, Wolfram-verified).
- **64-byte header:** `<8Q>`: `[magic, version=1, V, K, T, N, flags=0, fnv_body]`.
  - Requires `V >= 2`, `1 <= K <= min(64, V)`, `T >= 1`, `N >= 1`, `flags == 0`, matching checkpoint `(V, T)`.
- **Per-sample record** ($4(T+1) + T + 8TK$ bytes, repeated $N$ times, authenticated by 64-bit FNV-1a):
  1. `tokens`: `(T + 1)` × `u32` LE (`tokens[0..T-1]` inputs, `tokens[1..T]` hard next-token targets, all `< V`)
  2. `mask`: `T` × `u8` (`0` = prompt/padding ignored, `1` = supervised target)
  3. `top_ids`: planar `T * K` × `u32` LE (token IDs `< V`)
  4. `top_probs`: planar `T * K` × `f32` LE (finite $\ge 0$, per-position $\sum_j q_j \le 1.0001$)

See [`docs/CONTRACT-V4.md`](CONTRACT-V4.md) for full `OLLMCK04` / `OLLMDK01` exit codes.

---

## Quick Usage

```sh
# 1. Discover local open-source teachers (:18090 omega-router, :18091 q36-serve, :11434 Ollama, :8000 vLLM)
./bin/omega-helix distill probe

# 2. Harvest & verifier-gate traces into a canonical OpenWALDO EXPORT.json
./bin/omega-helix distill harvest -i ./traces.jsonl -o ./distilled-export --verify nonempty,json_schema

# 3. Run sparse top-K hybrid CE+KL distillation from a teacher checkpoint or harvested traces
./bin/omega-helix distill run --mode dk-univ --arch qwen3 --trace ./traces.jsonl --K 8 --alpha 0.7 --temperature 1.5 --steps 20
```

## Scope & Hardware Status

- **Verified:** Seeded `OLLMDK01` (`dkemit` / `dcheck` / `dtrain`), Python `f64` oracle cross-checks, negative controls (`10/11/12/14/15/25`), built-in verifiers (`omc_compile`, `python_ast`, `json_schema`, `regex`, `nonempty`), Studio `/api/distill/*` SSE endpoints, and `omega-helix distill` CLI are verified on **macOS Apple Silicon** and **Linux AArch64 CPU** (`tools/distill/test_distill.py` + `scripts/verify.sh`).
- **Pending GPU Kernel Port:** `ollm-univ` (`OLLMDK01`) executes on pure `f64` CPU; custom SM121 Blackwell GB10 kernels for `OLLMDK01` have not yet been added.
