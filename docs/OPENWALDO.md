# OpenWALDO Open-Source Training Dataset Integration

Omega Helix connects directly to [**OpenWALDO**](https://github.com/openwaldo/waldo) (Open Weights, Artifacts, Licenses, Data, Origins) so models can be trained on a **100% open-source, license-audited, cryptographically pinned corpus** with end-to-end provenance from raw shards to trained `OLLMCK0x` checkpoints.

## Architecture & Trust Boundary

```text
OpenWALDO Git Index + Lookaside
        │  waldo index export --format jsonl <corpus> <export-dir>
        ▼
  <export-dir>/EXPORT.json  +  data/**/*.jsonl
        │
        ├─ 1. Fail-closed verification (`tools/openwaldo/openwaldo_bridge.py verify`)
        │     • Validates `kind == "waldo-corpus-export"`, `schema == 1`, `bom.kind == "openwaldo-bom"`
        │     • Rejects path traversal (`..`, absolute paths, non-`data/` paths)
        │     • Verifies every shard's byte size & SHA-256 against `EXPORT.json` and `bom.shards`
        │     • Verifies every JSONL record's `sha256 == sha256(utf8(text))`, `kind`, `source`, `license`
        │
        ├─ 2. Pure seeded-Omega ingestion (`tools/openwaldo/waldo_ingest.om` -> `build/ollm-waldo`)
        │     • Strict RFC 8259 JSON parsing using `native/net/json.om` (zero C, zero Python, zero libc)
        │     • Streams `kind == "pretrain"` text verbatim + NUL (`\0`) EOS terminator
        │     • Renders `kind == "conversation"` turns as `role: content\n` (matching `native/serve.om`) + `\0`
        │     • Deterministic train/held split (`i mod HELD_MOD == HELD_MOD - 1 -> held.bin`)
        │
        ├─ 3. Tokenization & Multi-Architecture Training (`omega_helix_backend.py`)
        │     • BPE (`ollm bpefit`/`bpeenc`), byte-level (`V >= 256`), or `u32` (`V > 65535`)
        │     • Trains ANY `ollm-univ` (`OLLMCK04`) architecture: `llama`, `qwen2`, `qwen3`,
        │       `mistral`, `gemma`, `gpt`, `gelu`, looped cores (`LP/LC/NP`), or `custom` flags
        │     • Imports HuggingFace Safetensors bases & exports Safetensors (`F32`/`BF16`) + GGUF v3
        │
        └─ 4. Provenance BOM (`openwaldo-helix-bom.json` + `openwaldo-helix-observations.json`)
              • Cryptographically links `EXPORT.json` SHA-256, shard `object_sha256`s, licenses,
                tokenizer & token file SHA-256s, trained `OLLMCK04` / HF / GGUF SHA-256s, and eval metrics
```

## Quick Start

### 1. Export a verified OpenWALDO corpus as canonical JSONL
```sh
# Install waldo from https://github.com/openwaldo/waldo
waldo index export --format jsonl core/example ./waldo-export
```

### 2. Universal OpenWALDO Backend Adapter (Any Architecture, End-to-End)
Build `openwaldo`, `univ`, `interchange`, and CPU binaries, then train `llama`, `qwen3`, `gemma`, `gpt`, looped cores, or fine-tune an imported HF Safetensors checkpoint in one fail-closed command:
```sh
TARGET=openwaldo    sh scripts/build.sh
TARGET=univ         sh scripts/build.sh
TARGET=interchange  sh scripts/build.sh
TARGET=macos        sh scripts/build.sh   # or TARGET=aarch64 on Linux

python3 tools/openwaldo/omega_helix_backend.py \
  --export-dir ./waldo-export \
  --workdir ./run-waldo-qwen3 \
  --arch qwen3 --tokenizer bpe --vocab 4096 \
  -T 256 -D 384 -H 12 -H_KV 4 -F 1536 -L 4 -B 2 \
  --steps 100 --lr-micro 3000 --export-hf --export-gguf
```
Outputs verified `openwaldo-helix-bom.json` and `openwaldo-helix-observations.json` with SHA-256 hashes of every dataset shard, token stream, `OLLMCK04` checkpoint, HuggingFace `config.json` + `model.safetensors`, GGUF v3 file, and before/after held-out NLL & BPB. Accepts `--spec job.json`, `--hf-dir ./hf-model`, `--LP/--LC/--NP` looped cores, and custom `--flags`.

### 3. Shell Pipeline & GB10 Fast Path (`scripts/openwaldo-train.sh`)
```sh
# Verify EXPORT.json + ingest via pure-Omega ollm-waldo + tokenize (V=4096)
sh scripts/openwaldo-train.sh ./waldo-export ./run-waldo

# Or export + verify + train on DGX Spark GB10 (85k tok/s BF16) in one command:
OPENWALDO_CORPUS=core/example TRAIN=gb10 sh scripts/openwaldo-train.sh ./waldo-export ./run-waldo
```

### 4. Bind an external checkpoint to the OpenWALDO BOM
```sh
python3 tools/openwaldo/openwaldo_bridge.py bind-checkpoint \
  ./run-waldo/openwaldo-helix-bom.json ./run-waldo/gb10/ck.bin
```

## Standalone Pure-Omega Ingestor (`ollm-waldo`)

Build `ollm-waldo` on Linux AArch64 or Apple Silicon macOS:
```sh
TARGET=openwaldo sh scripts/build.sh
./build/ollm-waldo verified.jsonl train.bin held.bin 10
# stdout: W <total_docs> <train_docs> <held_docs> <train_bytes> <held_bytes>
```

Exit codes (fail-closed):
- `0`: success
- `2`: usage error
- `21`: malformed JSON or line > `983,040` bytes
- `22`: missing/invalid `sha256`, `kind`, `text`, `source`, or `license`
- `23`: invalid `conversation` JSON or missing `messages`/`role`/`content`
- `70`: I/O error

## Tests & Negative Controls

```sh
TARGET=openwaldo   sh scripts/build.sh
TARGET=univ        sh scripts/build.sh
TARGET=interchange sh scripts/build.sh
OLLM_WALDO_BIN=./build/ollm-waldo \
OLLM_UNIV_BIN=./build/ollm-univ \
OLLM_INTERCHANGE_BIN=./build/ollm-interchange \
  python3 -m unittest discover -s tools/openwaldo -p "test_*.py" -v
```
All 25 tests verify valid `waldo-corpus-export` schema-1 fixtures (`pretrain` + `conversation`), pure-Omega ingestion, end-to-end multi-architecture training (`llama`, `mistral`, `qwen2`, `qwen3`, `gemma`, `gpt`, `gelu`, looped cores, custom flags, `u16`/`u32` tokens, BPE), HuggingFace BF16 + GGUF v3 exports & re-import fine-tuning, deterministic reproducibility, BOM binding, and fail-closed refusals.
