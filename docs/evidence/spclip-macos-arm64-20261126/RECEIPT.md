# Omega Helix `feat/spclip-dynamic-edit` Native macOS Apple Silicon & Cross-Platform Qualification Receipt

- **Date:** `2026-11-26`
- **Repository:** `omega-helix` (`/Users/drakestapleton/workspace/omega-helix-spclip`)
- **Branch:** `feat/spclip-dynamic-edit`
- **Tested Source Commit:** `07f72eacdc4ad4871687b1c5f8698ea537fd604f`
- **Base (`spark/main`):** `a909888a63155016a14c180fcebbb098aad3c09e` (`git merge-base --is-ancestor spark/main HEAD` → **fast-forward clean**, 2 feature commits `63c7d279` + `07f72eac` + this receipt commit)
- **Evidence Classes (Honest Scope):**
  - **Apple Silicon macOS (`Darwin 25.6.0 arm64`, Apple M2, 8 GiB unified RAM):** **`HOST_VERIFIED`** (CPU pure-Omega `TARGET=macos` + `univ`/`interchange`/`openwaldo`/`tok` + live Apple M2 Metal GPU `TARGET=macos-metal`).
  - **DGX Spark (`Linux 7.0.0-1019-nvidia aarch64`, Grace 20-core):** **`HOST_VERIFIED`** (CPU `TARGET=aarch64`, `TARGET=gb10` compile check; live GB10 GPU execution `NOT_RUN` here — no GPU window opened).
  - **Linux x86-64 (`TARGET=x86linux`):** **`EXERCISED_EMULATED`** (`qemu-x86_64` on DGX Spark; physical x86-64 `NOT_RUN`).

---

## 1. Compiler Pins & Reproducible Builds (2× Bit-Identical)

| Platform / Target | Compiler (`omc`) & SHA-256 | Artifact | Format / Signing | Size (B) | SHA-256 |
|---|---|---|---|---:|---|
| macOS arm64 (`TARGET=macos`, raw pre-sign) | `~/.local/bin/omc` (`68a590c37dfb1ec37a183f0032e02db1570db299266f574f65556edfc70feb9c`) | `ollm-macos` (2× `cmp` OK) | Mach-O 64-bit arm64 (unsigned) | 743,200 | `9c1de323345e8dcac216889f2584a5d7cd0995ada82ebfaf78b97c82ee3d7b53` |
| macOS arm64 (`TARGET=macos`, signed) | `~/.local/bin/omc` | `ollm-macos` (2× signed match) | Mach-O 64-bit arm64 + `codesign -s - -f` (`--verify --strict` OK) | 757,040 | `9649b9a8912e02dda3506c895dc70387fffbfd8e5ba6397f96e4bd4ea58c9ddc` |
| macOS arm64 (`TARGET=univ`) | `~/.local/bin/omc` | `ollm-univ` (2× signed match) | Mach-O 64-bit arm64 + ad-hoc codesign OK | 182,480 | `25c3efe19a6021f898b3ea7edebf0b88e314f175d891cea0d23016f2a82f9e2c` |
| macOS arm64 (`TARGET=interchange`) | `~/.local/bin/omc` | `ollm-interchange` (2× signed match) | Mach-O 64-bit arm64 + ad-hoc codesign OK | 198,896 | `6451644f1bff7010bbd1a57efc09de9672e1bc3250195f3808c83c3f914dbf68` |
| macOS arm64 (`TARGET=openwaldo`) | `~/.local/bin/omc` | `ollm-waldo` (2× signed match) | Mach-O 64-bit arm64 + ad-hoc codesign OK | 83,984 | `de59212ff75a0a08402b6477bd79bb33e05c6be0a544fb6e3bd5baff7704bbbc` |
| macOS arm64 (`TARGET=tok`) | `~/.local/bin/omc` | `ollm-tok` (2× signed match) | Mach-O 64-bit arm64 + ad-hoc codesign OK | 67,568 | `d8fecc9ff39714475ee648a46b9323b9b938d6361ea4a523bb06514936a01689` |
| macOS arm64 (`TARGET=macos-metal`) | `ng` (`@ng a64 macho`) | `ollm-metal` | Mach-O 64-bit arm64 + ad-hoc codesign OK | 379,504 | Built & verified in `verify-om.sh` §14 |
| Spark aarch64 (`TARGET=aarch64`) | `~/canon/omega-mv/seed/out/omc` (`28fca13d44054e26f78c33d1753fd5c005d08e9144d95985c90bd75b60885072`) | `ollm` (2× `cmp` OK) | Static Linux ELF aarch64 | 1,027,020 | `5a1e5f04f0aacff972757adc35a4e59e418787733d8bdf01c4ffad2f3a7273c2` |
| Spark aarch64 (`TARGET=gb10`) | `~/canon/omega-mv/seed/out/omc` | `ollm-gb10` (compile check) | Static Linux ELF aarch64 | 1,841,988 | Built cleanly in `verify-om.sh` |
| Spark cross (`TARGET=x86linux`) | `~/canon/omega-mv/seed/out/omc` | `ollm-x86linux` | Static Linux ELF x86-64 | 4,497,252 | `8ada0447282aa03c0dd20f46ab2562f234ccbfaee65bbdfb48fb6f237f7e5e40` |

---

## 2. Verification Matrix (Positive + Fail-Closed Negative Controls)

### 2.1 Apple Silicon M2 Native macOS (`Darwin 25.6.0 arm64`)

1. **Live `sh install.sh --dry-run`:** `TOPOLOGY=macos-arm64`, `TARGETS=univ interchange openwaldo tok macos`, `SIGN_MODE=sha256+adhoc-codesign`, `LAUNCH_MODE=local-or-remote`, `SKIP_SSH_TUNNEL=1`, `BROWSER_CMD=open`, `NATIVE_EXE=0`, `BIND=127.0.0.1:18420` (recorded verbatim in `macos_install_dry_run.txt`).
2. **Live prefix install (`sh install.sh --prefix /tmp/om_mac_spclip/inst --no-download`) + `omega-helix doctor --json`:** Built all 5 targets, ad-hoc signed, generated & verified `bin/SHA256SUMS` (byte-identical to standalone builds), linked `omega-helix`; `doctor --json` returned `"ok": true`, `"topology": "macos_arm64"`, all 5 binaries `"ready": true`, `catalog_count: 73`.
3. **Pure-Omega Zero-Python Gate (`sh scripts/verify-om.sh`):**
   - `QUICKSTART_CPU_OK` (`adamck`, `gcheck`, `train`, `eval`, `sample`, `BPE_ROUNDTRIP_OK`, `KVCMP 8 0`, `DNA_SPEC_CHAIN_OK`, `OLLMCK03`)
   - `OMG_SELFTEST PASS`, `ARBORIS_SELFTEST PASS`, `BTLOCK_OK` (all 4 byte-traceability locks + `PASS DETBWD 0 2064`)
   - `SPIRAL_WEAVE_OM_PASS` (W4/f64 `OLLMSP02` + 1-bit tamper `rc=61`)
   - `APU_WEAVE_OM_PASS` (`APU_SUMMARY 2 1 28 ...`, `SPWEVAL` + tamper `rc=61`)
   - `SERVE_SCALING_OM_PASS` (`TARGET=aarch64` cross-build `1,027,020 B` OK)
   - `SPCLIP_EDITABLE_OM_PASS` (`create`/`verify`/`edit`/`eval`/`run`/`tune`/`verify-lineage`/`spapp` + tampered `.spclip` `rc=61` + zero weights `rc=2`)
   - `METAL_GPU_OM_PASS` (`MDEV Apple_M2 unified=1 max_ws=5726633984 pipelines=17 max_tg=1024 simd=32 PASS`, `MOPS GEMM0/1/2 RMSNORM ATTN ADAMW PASS`, `GCMP PASS`)
   - Final verdict: **`OMEGA_HELIX_PURE_OM_PASS`** (exit `0`).
4. **Independent `f64`/SHA-256 Editable `spclip` Suite (`python3 -W error::ResourceWarning tests/test_spclip_editable.py --bin /tmp/om_mac_spclip/ollm-macos`):**
   - `PASS [1]`..`PASS [8]` including **all 41 fail-closed negative controls** (`rc=60/61/62/2`), 0-depth splice recovery, `OLLMCK03` looped-core spclip, SHA-256 boundary lengths (`55/56/63/64/65 B`), and extreme `tau=1` underflow fallback → **`SPCLIP_EDITABLE_SUITE_PASS`** (exit `0`).
5. **Studio Topology & `install.sh` Dry-Run Unit Suite (`python3 -W error::ResourceWarning -m unittest tests/test_studio_topology.py -v` + `py_compile -W error::SyntaxWarning`):** **`Ran 6 tests in 0.291s — OK`** (exit `0`).

### 2.2 DGX Spark (`Linux aarch64`, CPU-Only) & `x86linux` (`qemu-x86_64`)

1. **Live Spark `sh install.sh --dry-run`:** `TOPOLOGY=spark-headless`, `PLATFORM=DGX Spark Linux (aarch64, GB10)`, `GB10=1`, `TARGETS=univ interchange openwaldo tok aarch64 wtrain`, `OPTIONAL_TARGETS=gb10`, `SIGN_MODE=sha256`, `BROWSER_CMD=print-url` (recorded verbatim in `spark_install_dry_run.txt`).
2. **Pure-Omega Zero-Python Gate (`sh scripts/verify-om.sh`):** Full pass including live `serve.om` `/health` check (`max_slots:4`, `max_ctx:8192`, `SERVE_SCALING_OM_PASS`), `SPCLIP_EDITABLE_OM_PASS`, and **`OMEGA_HELIX_PURE_OM_PASS`** (exit `0`).
3. **`tests/test_spclip_editable.py`:** `PASS [1]`..`PASS [8]` + all 41 negative controls → **`SPCLIP_EDITABLE_SUITE_PASS`** (exit `0`).
4. **`tests/test_studio_topology.py`:** **`Ran 6 tests in 0.165s — OK`** (exit `0`).
5. **`TARGET=x86linux` under `qemu-x86_64` (`EXERCISED_EMULATED`):** `adamck` PASS, `spclip create/verify/edit/verify` PASS, 1-strand refused `rc=2`, 1-byte tampered `.spclip` refused `rc=61` (`X86LINUX_QEMU_SPCLIP_PASS one_strand_rc=2 tamper_rc=61`).

### 2.3 Cross-Platform Bit-Exact Parity (Apple M2 macOS == DGX Spark Linux)

| Check | Apple M2 (`ollm-macos`) | DGX Spark (`ollm`) | Parity |
|---|---|---|---|
| `SPCLIP_EVAL` (Section 13b) | `SPCLIP_EVAL 4 5499563 7934192 2 1 500` | `SPCLIP_EVAL 4 5499563 7934192 2 1 500` | **EXACT** |
| `SPCLIP_TUNE` child pack loss | `5519035 -> 5490253` | `5519035 -> 5490253` | **EXACT** |
| `SPCLIP_TUNE` child `OLLMSP01` SHA-256 | `6f81c33c26f37950d847871475a79aacbab9c2ac6c2869368aa86d35022208a6` | `6f81c33c26f37950d847871475a79aacbab9c2ac6c2869368aa86d35022208a6` | **EXACT (256-bit)** |
| `test_spclip_editable` `PASS [4]` tau / splice / zgn | `tau=4964315/4964380/4964394->4964397`, `splice=4934737/4937670/4963545`, `zgn==byp=4958481` | `tau=4964315/4964380/4964394->4964397`, `splice=4934737/4937670/4963545`, `zgn==byp=4958481` | **EXACT** |
| `test_spclip_editable` `PASS [5]` child tune loss | `5611452->5592316 uNats` | `5611452->5592316 uNats` | **EXACT** |
| `BTLOCK` `DETBWD` / `G1` / `G2` / `TR` | `PASS DETBWD 0 2064`, `3e27231de9063d01`, `3e33bc5fdc800000`, `4006ffdaa1b8a2b4` | `PASS DETBWD 0 2064`, `3e27231de9063d01`, `3e33bc5fdc800000`, `4006ffdaa1b8a2b4` | **EXACT** |
