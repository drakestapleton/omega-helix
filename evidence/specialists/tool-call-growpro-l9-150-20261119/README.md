# Frozen-Core Tool-Call Specialist Run (`growpro` + `fcachem` + `trainfc`, `L=8→9`, 150 steps)

Measured host receipt on **Apple Silicon M2 (Darwin 25.6.0 arm64, f64 CPU)** using the `feat/darwin-ollm-specialists` codebase (`e993f179651b8dd5e99a040b6ec7284fff8cb12e`). Strictly local MacBook CPU execution; **no DGX Spark GPU work was performed** (GB10 GPU specialist training remains `NOT_RUN`).

## 1. Environment & Artifact Pins

- **Host:** Apple M2 (`ncpu=8`, `8 GiB` RAM), Darwin 25.6.0 `arm64`, f64 CPU
- **Source commit:** `e993f179651b8dd5e99a040b6ec7284fff8cb12e`
- **Build command:**
  ```sh
  {
    printf 'target a64macos\n'
    cat native/omegallm/math.om native/omegallm/data.om native/omegallm/lpool.om \
        native/omegallm/model.om native/omegallm/adam.om native/omegallm/ckpt.om \
        native/omegallm/bpe.om native/omegallm/spec.om native/omegallm/cli.om \
        native/omegallm/mio.om native/omegallm/noserve.om native/omegallm/nogpu.om
  } > ollm-darwin.om
  omc < ollm-darwin.om > ollm-darwin && chmod +x ollm-darwin
  ```

| Artifact | Size (bytes) | SHA-256 |
|---|---|---|
| `omc` (Darwin arm64 Mach-O) | — | `68a590c37dfb1ec37a183f0032e02db1570db299266f574f65556edfc70feb9c` |
| `ollm-darwin.om` (concatenated source) | — | `5f53ca4a5ecc0b26dee0de4661feafae2cd500b873dee35953e15f77d4bb76cf` |
| `ollm-darwin` (Mach-O arm64 binary) | `214,816` | `dfece10d4e4bd20752c90c5af132f335b0166f65ea3b35e9d77d3eefd2f23846` |
| `omega-v2.ck` (base `L=8`, `15,833,472` params) | `380,003,448` | `a53ced7265818c8f6197449315e9ca4082721665022fec4f81d4538e8d6c2d0f` |
| `merges4096.bin` | `15,384` | `73a29baf97619a746a21232c403bf08af71a76e48b3b2cde6d64dab766083882` |
| `tool-call_aligned.tok` (train tokens) | `146,434` | `1de8d1fad6890c0ac73f679cd2fbfe3e9054d0ef4bdcf052b61cfe64aafcbe65` |
| `tool-call_aligned.msk` (completion-only mask) | `73,217` | `d972401efdbd0bd9b47bd7e471edb76722e0b2375a03a6bfb39bec9b76293b7c` |
| `tool-call_held.tok` (held-out, `W=9` windows) | `4,740` | `634fd8ed6f03f734167ebfea72f48ec53c6878022f9dd07cc8ca4e232df9a2af` |
| `tc_w160.hc` (`fcachem` `W=160` hidden cache) | `126,156,848` | `3f127597f9eed16fc973040c76463e4fd75b9c4837744e8b2beeafdfaf803166` |
| `spec_growpro_l9.ck` (trained `L=9`, `17,603,712` params) | `422,489,208` | `1317cf8784c16ca9ff522d5bf9022b8fa966b67550e19f71a076c98cd53363d4` |
| `tool-call_growpro_l9_150.sp` (`OLLMSP01`, `1,770,624` params) | `14,165,040` | `6e21157c45d463ed5c92d141a85ba632f07cb41353e928efea745a3971e99927` |

The extracted `OLLMSP01` pack is **14.17 MB (`14,165,040` B)** vs **422.49 MB (`422,489,208` B)** for the full `L=9` checkpoint (**29.8× smaller**).

## 2. Exact Commands

```sh
# 1. Precompute masked frozen-base hidden states (W=160 windows, completion-only mask)
$OLLM fcachem $CK $TR $MK 160 $HC

# 2. LLaMA-Pro block-expansion growth (L=8 -> 9, TOTAL=150, lr=0.0035)
$OLLM growpro $CK $SPEC_CK 9 150 3500

# 3. Verify step-0 held-out parity against base checkpoint (W=9 windows)
$OLLM eval $CK $HE 9 $MERGES
$OLLM eval $SPEC_CK $HE 9 $MERGES

# 4. Frozen-core specialist training from hidden cache (FREEZE_LAYERS=8, B=1)
$OLLM trainfc $SPEC_CK $HC 50 25 8 1
$OLLM eval $SPEC_CK $HE 9 $MERGES
$OLLM trainfc $SPEC_CK $HC 100 25 8 1
$OLLM eval $SPEC_CK $HE 9 $MERGES
$OLLM trainfc $SPEC_CK $HC 150 25 8 1
$OLLM eval $SPEC_CK $HE 9 $MERGES

# 5. Extract OLLMSP01 pack and verify in-memory speval + spapp round-trip
$OLLM spext $CK $SPEC_CK $PACK
$OLLM speval $CK $PACK $HE 9 $MERGES
$OLLM spapp $CK $PACK $REBUILT_CK
$OLLM eval $REBUILT_CK $HE 9 $MERGES

# 6. Qualitative sampling (N=32, temp=0.100, seed=42)
$OLLM spsamp $CK $PACK 32 100 "$PROMPT" 42 $MERGES
```

## 3. Training & Held-Out Eval Progression

- **`fcachem` (`W=160`):** `FCACHE 160 256 384 8` in `618.10s` real (`608.94s` user, `3.70s` sys); bit-identical via `cmp` to prior `tool-call_w160.hc`.
- **`growpro` (`L=8→9`):** `GP 8 9 17603712` in `3.30s` real.
- **`trainfc` wall times (`B=1`, `FREEZE_LAYERS=8`):**
  - Leg 1 (`0→50`): `363.16s` real (`7.26 s/step`)
  - Leg 2 (`50→100`): `321.92s` real (`6.44 s/step`)
  - Leg 3 (`100→150`): `313.11s` real (`6.26 s/step`)
  - Total 150-step training time: **`998.19s` (~16.6 min)** on M2 CPU (single-threaded f64).

| Step | Masked Train Loss (`FC`, µnats) | Masked Train Loss (nats/tok) | Held-Out `eval` (`W=9`, µnats) | Held-Out `eval` (nats/tok) | Held-Out `bpb` (µbpb) | Δ vs Base Held-Out |
|---:|---:|---:|---:|---:|---:|---:|
| Base (`omega-v2.ck`, step 60000) | — | — | `7,183,552` | `7.183552` | `6,325,274` | `0.00%` |
| `0` (`growpro` step 0) | — | — | `7,183,552` | `7.183552` | `6,325,274` | `0.00%` (bit-identical) |
| `25` | `5,511,696` | `5.511696` | — | — | — | — |
| `50` | `2,289,094` | `2.289094` | **`4,528,072`** | **`4.528072`** | **`3,987,066`** | **`-36.97%`** |
| `75` | `961,954` | `0.961954` | — | — | — | — |
| `100` | `628,315` | `0.628315` | `4,757,065` | `4.757065` | `4,188,699` | `-33.78%` |
| `125` | `690,457` | `0.690457` | — | — | — | — |
| `150` | `439,281` | `0.439281` | `4,954,553` | `4.954553` | `4,362,591` | `-31.03%` |

**Observation:** Masked completion-only training loss fell monotonically from `5.511696` (step 25) to `0.439281` nats/tok (step 150), while unmasked full-sequence held-out NLL (`tool-call_held.tok`, which scores both prompt and completion tokens) dropped sharply by **`-36.97%`** at step 50 (`7.183552 → 4.528072 nats/tok`) and then drifted mildly upward (`4.757065` at step 100, `4.954553` at step 150) as the single specialist layer sharpened further on completion tokens at the expense of unmasked prompt-token likelihood.

## 4. Base Preservation, Round-Trip & Negative Controls

| Check | Result |
|---|---|
| Step-0 `growpro` parity | `BASE_V2_EVAL` (`E 60000 9 7183552 6325274`) == `STEP0_EVAL` (`E 0 9 7183552 6325274`) |
| `spext` extraction | `SPEXT 8 9 1770624` (`rc=0`, verifies FNV-1a 64 over frozen base) |
| In-memory `speval` round-trip | `E 60000 9 4954553 4362591` == trained `STEP150_EVAL` (`E 150 9 4954553 4362591`) |
| `spapp` + `eval` round-trip | `SPAPP 8 9 1770624` -> `E 60000 9 4954553 4362591` (bit-identical) |
| Independent byte check of frozen base (`OLLMCK02` 120-byte header + `15,833,088` f64 params = `126,664,704` bytes) | `spec_growpro_l9.ck` == `omega-v2.ck` (`True`), `rebuilt_growpro_l9.ck` == `omega-v2.ck` (`True`) |
| Negative control: truncated pack (`1024` B) | `rc=60` |
| Negative control: 1-bit flip in payload (byte 64) | `rc=61` |
| Negative control: wrong base (`omega-v2-edited-20261104.ck`) | `rc=62` |

## 5. Qualitative Sampling (`spsamp` vs Base `omega-v2.ck`, `N=32`, `temp=0.100`, `seed=42`)

Base `omega-v2.ck` (TinyStories pretraining only) produces non-tool story/number continuations on tool prompts, whereas the 1-layer 150-step `OLLMSP01` pack acquires the `<tool_call>...</tool_call>` syntax when given the exact tokenized training boundary (`" User: ...\nCall: "` with trailing space):

| Prompt | Base `omega-v2.ck` (`sample`) | Specialist `spsamp` (`tool-call_growpro_l9_150.sp`) |
|---|---|---|
| `" User: What is 17 + 25?\nCall: "` | *(unspaced `Call:` baseline: ` 15 15 15 15 40 40 5.\nLittle did Light 5 5 `)* | `<tool_call>calc(25 / 15 / 15)</tool_call>` |
| `" User: Read SECURITY.md.\nCall: "` | *(unspaced `Call:` baseline: `\nRead SEST.\nRead SEST.\nRead SEST.\nRead SEST.`)* | `<tool_call>shell(df -n <tool_call>` |
| `" User: Multiply 29 by 14.\nCall: "` | *(unspaced `Call:` baseline: ` 123, 23, 44, 4, 4, 44.\nMile: 4, 4`)* | `<tool_call>search(25 + 15)</tool_call>` |
| `" User: What is 17 + 25?\nCall: <tool_call>"` | — | `calc(25 / 125)</tool_call>` |

**Boundary sensitivity & capability limits:**
- Without the trailing space after `Call:` (i.e. `"User: ...\nCall:"`), all three prompts emitted ` NONE` (a valid no-tool class in the training data, merged by BPE with the leading space).
- With the exact `" Call: "` boundary, the single specialist layer reliably emits `<tool_call>` syntax and selects `calc(...)` on `"What is 17 + 25?"`, but still makes tool-routing and argument-copying errors (`shell`/`search` instead of `read_file`/`calc`, hallucinated operands) at 150 steps / 1 added layer.

Verbatim sanitized outputs are preserved in [outputs.txt](outputs.txt).
