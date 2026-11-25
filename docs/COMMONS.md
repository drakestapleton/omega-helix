# Sovereign Commons, Cryptographic Trust, Memory Migration, Swarm Quorum & Air-Gap Bundles

Phase 8 adds the seven community & edge-sovereignty primitives implemented in [`tools/commons/`](../tools/commons/) (100% Python 3 stdlib, zero telemetry, zero third-party packages) and exposed via [`omega-helix`](../bin/omega-helix) and Studio HTTP endpoints.

---

## 1. DNA Commons — Cognitive Package Manager (`omega-helix pkg`)

Instead of multi-gigabyte unverified LoRA merges, [`tools/commons/pkg.py`](../tools/commons/pkg.py) manages **~14 MB signed `OLLMSP01` `.sp` packs** pinned to a specific trunk via `base_hash` and protected by payload `FNV-1a` (`pack_hash`) plus SHA-256 content addressing:

```sh
# Publish a verified specialist .sp pack to a local/shared Commons registry
./bin/omega-helix pkg publish ./sql-audit.sp sql-audit --registry ./commons-reg --version 1.0.0 --desc "Strict SQL auditor"

# Search compatible packs for a specific base_hash
./bin/omega-helix pkg search --registry ./commons-reg --query sql --base-hash <16-hex-base_hash>

# Install fail-closed (verifies pack_hash, SHA-256 blob digest, and base_hash)
./bin/omega-helix pkg install sql-audit ./my-env --registry ./commons-reg --base-hash <16-hex-base_hash>

# Diff parameter L2/linf deltas between two .sp packs
./bin/omega-helix pkg diff ./v1.sp ./v2.sp

# Right-to-be-forgotten revocation (deletes installed .sp + records revocations.jsonl)
./bin/omega-helix pkg revoke sql-audit ./my-env --reason "dataset-retire"
```

---

## 2. One-Command Cryptographic Trust Audit (`omega-helix verify`)

[`tools/commons/trust.py`](../tools/commons/trust.py) runs a sub-second offline verification across binaries, UI CSP policy, server loopback default, optional OpenWALDO `EXPORT.json` / BOM, and `OLLMSP01` layer-chain continuity:

```sh
./bin/omega-helix verify --base-ck ./trunk.ck --pack ./layer1.sp --pack ./layer2.sp --bom ./EXPORT.json --json
```

Also exposed live in Studio via `GET /api/trust/verify`.

---

## 3. Model-Agnostic Personal Memory & Trunk Migration (`omega-helix memory`)

Keep personal context outside any specific model's weights in an append-only SHA-256 deduplicated ledger, export/import with PBKDF2-HMAC-SHA256 + XOR-stream + HMAC-SHA256 authenticated encryption, and replay verified memories onto any new base trunk (`growpro` + `fcache` + `trainfc` + `spext`) to mint a fresh `.sp` pack in seconds:

```sh
./bin/omega-helix memory add ./memory.jsonl "Prefer explicit error codes in Omega I/O." --tag omega
./bin/omega-helix memory export-enc ./memory.jsonl ./memory.enc --passphrase "your-passphrase"
./bin/omega-helix memory migrate ./memory.jsonl ./new-qwen-trunk.ck ./migrated-out --steps 6
```

Studio API: `POST /api/commons/memory/migrate`.

---

## 4. Community Hardware-Porting Kit (`omega-helix port-check`)

See [`docs/BACKEND-PORTING-KIT.md`](BACKEND-PORTING-KIT.md) for the frozen 9-gate parity harness (`adamck`, `init`, `gcheck`, `kvcmp`, and `tests/oracle.py`).

---

## 5. Trustless Swarm Forging Quorum (`tools/commons/swarm_receipt.py`)

Nodes forge and verify `.sp` packs independently on local hardware, emit self-verifying evidence bundles (`create_evidence_bundle`), and enforce an M-of-N distinct-node quorum (`tally_quorum(..., min_quorum=3)`) without relying on a single trusted operator.

---

## 6. Capability-Bounded Local Agency & Snapshot/Rollback (`omega-helix lease` + `tools/commons/capability.py`)

Enforces explicit path (`allowed_dirs`), binary allowlist (`allowed_cmds`), TTL expiration, revocation, offline proxy-env stripping (`allow_network=False`), chained SHA-256 audit logging, and directory `snapshot_dir` / `rollback_dir`:

```sh
./bin/omega-helix lease issue ./agent.lease.json --dir ./sandbox --cmd /usr/bin/env --ttl 600
./bin/omega-helix lease run ./agent.lease.json --cwd ./sandbox --audit ./audit.jsonl -- /usr/bin/env
./bin/omega-helix lease revoke ./agent.lease.json
```

---

## 7. USB-Stick Sovereignty — Air-Gap Portable Bundle (`omega-helix pack-usb`)

Package static pure-Omega binaries, optional UEFI `.efi` images, CLI & Studio tools, trunk checkpoint, `.sp` packs, personal memory, `SHA256SUMS.txt`, `AIRGAP-MANIFEST.json`, and `START-OFFLINE.sh` into one relocatable directory:

```sh
./bin/omega-helix pack-usb ./usb-bundle --trunk-ck ./trunk.ck --pack ./skill.sp --memory ./memory.jsonl
./usb-bundle/START-OFFLINE.sh verify --json
```

Studio API: `POST /api/commons/pack-usb`.
