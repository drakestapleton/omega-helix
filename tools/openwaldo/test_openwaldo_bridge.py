#!/usr/bin/env python3
"""Unit, integration, and negative-control tests for OpenWALDO <-> Omega Helix bridge."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import omega_helix_backend as backend
import openwaldo_bridge as bridge

REPO = Path(__file__).resolve().parent.parent.parent


def make_record(kind: str, text: str, source: str = "openwaldo/example", lic: str = "CC0-1.0") -> str:
    rec = {
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "kind": kind,
        "text": text,
        "source": source,
        "license": lic,
        "tokens": max(1, len(text.split())),
    }
    return json.dumps(rec, separators=(",", ":"), ensure_ascii=False) + "\n"


def build_export_fixture(root: Path, records: list[str], rel_path: str = "data/example/shard-000.jsonl") -> Path:
    export_dir = root / "export"
    shard_file = export_dir / rel_path
    shard_file.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(records).encode("utf-8")
    shard_file.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    obj_digest = hashlib.sha256(b"parquet-source:" + payload).hexdigest()
    export_doc = {
        "kind": "waldo-corpus-export",
        "schema": 1,
        "generated": "2026-11-25T15:00:00Z",
        "format": "jsonl",
        "bom": {
            "kind": "openwaldo-bom",
            "schema": 1,
            "subject": "corpus",
            "index": {"revision": "0" * 40},
            "paths": ["core/example"],
            "shards": [
                {
                    "manifest": "core/example.yaml",
                    "sha256": obj_digest,
                    "format": "parquet",
                    "license": "CC0-1.0",
                    "licenses": ["CC0-1.0"],
                    "docs": len(records),
                    "tokens": len(records) * 4,
                    "bytes": len(payload) + 15,
                }
            ],
        },
        "files": [
            {
                "path": rel_path,
                "manifest": "core/example.yaml",
                "object_sha256": obj_digest,
                "sha256": digest,
                "format": "jsonl",
                "license": "CC0-1.0",
                "licenses": ["CC0-1.0"],
                "docs": len(records),
                "tokens": len(records) * 4,
                "object_bytes": len(payload) + 15,
                "bytes": len(payload),
            }
        ],
    }
    (export_dir / "EXPORT.json").write_text(json.dumps(export_doc, indent=2) + "\n", encoding="utf-8")
    return export_dir


class OpenWaldoBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tools = _locate_tools()  # build (or locate) ollm-waldo / ollm so the pure-Omega e2e test really runs
        if "waldo" in tools:
            os.environ.setdefault("OLLM_WALDO_BIN", tools["waldo"])
        if "ollm" in tools:
            os.environ.setdefault("OLLM_BIN", tools["ollm"])

    def setUp(self) -> None:
        conv_payload = json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Explain pure Omega."},
                    {"role": "assistant", "content": "Omega compiles directly to static machine code."},
                ]
            },
            separators=(",", ":"),
        )
        self.records = [
            make_record("pretrain", f"OpenWALDO verified document number {i} for Omega Helix training.")
            for i in range(9)
        ] + [make_record("conversation", conv_payload)]

    def test_verify_valid_export(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            exp = build_export_fixture(Path(td), self.records)
            rep = bridge.verify_export(exp)
            self.assertEqual(rep["total_docs"], 10)
            self.assertEqual(rep["licenses"], ["CC0-1.0"])

    def test_refuse_tampered_shard_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            exp = build_export_fixture(Path(td), self.records)
            shard = exp / "data/example/shard-000.jsonl"
            data = bytearray(shard.read_bytes())
            data[0] ^= 0x01
            shard.write_bytes(bytes(data))
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                bridge.verify_export(exp)

    def test_refuse_tampered_record_text_hash(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bad_rec = json.dumps(
                {
                    "sha256": "a" * 64,
                    "kind": "pretrain",
                    "text": "tampered body",
                    "source": "openwaldo/example",
                    "license": "CC0-1.0",
                }
            ) + "\n"
            exp = build_export_fixture(Path(td), [bad_rec])
            with self.assertRaisesRegex(ValueError, "record sha256 mismatch"):
                bridge.verify_export(exp)

    def test_refuse_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            exp = build_export_fixture(Path(td), self.records)
            doc = json.loads((exp / "EXPORT.json").read_text(encoding="utf-8"))
            doc["files"][0]["path"] = "../escape.jsonl"
            (exp / "EXPORT.json").write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsafe or non-canonical"):
                bridge.verify_export(exp)

    def test_refuse_unsupported_kind(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bad = make_record("audio", "not text")
            exp = build_export_fixture(Path(td), [bad])
            with self.assertRaisesRegex(ValueError, "unsupported record kind"):
                bridge.verify_export(exp)

    def test_end_to_end_pure_omega_ingestor_and_controls(self) -> None:
        waldo_bin = os.environ.get("OLLM_WALDO_BIN", "")
        if not waldo_bin:
            self.skipTest("OLLM_WALDO_BIN not set")
        ollm_bin = os.environ.get("OLLM_BIN", "")

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            exp = build_export_fixture(root, self.records)
            out = root / "prepared"
            argv = [
                "prepare",
                str(exp),
                str(out),
                "--ollm-waldo",
                waldo_bin,
                "--held-mod",
                "10",
            ]
            if ollm_bin:
                argv += ["--ollm", ollm_bin, "--vocab", "280"]
            rc = bridge.main(argv)
            self.assertEqual(rc, 0)

            train_bytes = (out / "train.bin").read_bytes()
            held_bytes = (out / "held.bin").read_bytes()
            self.assertEqual(train_bytes.count(b"\x00"), 9)
            self.assertEqual(held_bytes.count(b"\x00"), 1)
            self.assertIn(b"user: Explain pure Omega.\n", held_bytes)
            self.assertIn(b"assistant: Omega compiles directly to static machine code.\n", held_bytes)

            bom = json.loads((out / "openwaldo-helix-bom.json").read_text(encoding="utf-8"))
            self.assertEqual(bom["kind"], "omega-helix-training-bom")
            self.assertEqual(bom["openwaldo_export"]["total_docs"], 10)

            # Negative control 1: malformed JSON refused by pure-Omega ollm-waldo (exit 21)
            bad_json = root / "bad.jsonl"
            bad_json.write_text("{not valid json\n", encoding="utf-8")
            p1 = subprocess.run(
                [waldo_bin, str(bad_json), str(root / "t1"), str(root / "h1")],
                capture_output=True,
                check=False,
            )
            self.assertEqual(p1.returncode, 21)

            # Negative control 2: missing license refused by pure-Omega ollm-waldo (exit 22)
            no_lic = root / "nolic.jsonl"
            no_lic.write_text(
                json.dumps(
                    {
                        "sha256": hashlib.sha256(b"hi").hexdigest(),
                        "kind": "pretrain",
                        "text": "hi",
                        "source": "s",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            p2 = subprocess.run(
                [waldo_bin, str(no_lic), str(root / "t2"), str(root / "h2")],
                capture_output=True,
                check=False,
            )
            self.assertEqual(p2.returncode, 22)

            # End-to-end checkpoint binding when ollm binary is available
            if ollm_bin:
                ck = out / "test.ck"
                subprocess.run(
                    [ollm_bin, "init", str(ck), "280", "16", "16", "2", "32", "2", "2", "1", "4", "3000", "1"],
                    check=True,
                )
                subprocess.run(
                    [ollm_bin, "train", str(ck), str(out / "train.tok"), "2", "2"],
                    check=True,
                )
                self.assertEqual(
                    bridge.main(["bind-checkpoint", str(out / "openwaldo-helix-bom.json"), str(ck)]),
                    0,
                )
                bom2 = json.loads((out / "openwaldo-helix-bom.json").read_text(encoding="utf-8"))
                self.assertEqual(len(bom2["checkpoints"]), 1)
                self.assertEqual(bom2["checkpoints"][0]["step"], 2)
                self.assertEqual(bom2["checkpoints"][0]["shape"]["V"], 280)


# --------------------------------------------------------------------------------------
# Phase 4: OpenWALDO Universal Training Backend Adapter (omega_helix_backend.py)
# --------------------------------------------------------------------------------------

_TOOLS: dict[str, str] = {}
_BUILD_ERR: dict[str, str] = {}
_TOOLS_DIR: tempfile.TemporaryDirectory | None = None


def _locate_tools() -> dict[str, str]:
    """Find (or build with scripts/build.sh) ollm-waldo, ollm-univ, ollm and ollm-interchange."""
    global _TOOLS_DIR
    if _TOOLS:
        return _TOOLS
    spec = {
        "waldo": ("OLLM_WALDO_BIN", "openwaldo", ["ollm-waldo"]),
        "univ": ("OLLM_UNIV_BIN", "univ", ["ollm-univ"]),
        "ollm": ("OLLM_BIN", "macos" if os.uname().sysname == "Darwin" else "aarch64", ["ollm-macos", "ollm"]),
        "interchange": ("OLLM_INTERCHANGE_BIN", "interchange", ["ollm-interchange"]),
    }
    for key, (env, target, names) in spec.items():
        cand = os.environ.get(env, "")
        if cand and os.access(cand, os.X_OK):
            _TOOLS[key] = cand
            continue
        # a stale binary is never trusted for the in-repo sources: rebuild into a private dir
        if _TOOLS_DIR is None:
            _TOOLS_DIR = tempfile.TemporaryDirectory(prefix="openwaldo-be-")
        out = _TOOLS_DIR.name
        if key == "interchange" and not (REPO / "native/univ/interchange.om").is_file():
            continue
        proc = subprocess.run(
            ["sh", str(REPO / "scripts/build.sh")], env={**os.environ, "TARGET": target, "OUT": out},
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0:
            _BUILD_ERR[key] = (proc.stderr or proc.stdout).strip()[-300:]
        for n in names:
            if proc.returncode == 0 and os.access(os.path.join(out, n), os.X_OK):
                _TOOLS[key] = os.path.join(out, n)
                break
    return _TOOLS


def _corpus_records() -> list[str]:
    conv = json.dumps(
        {"messages": [{"role": "user", "content": "Explain pure Omega."},
                      {"role": "assistant", "content": "Omega compiles directly to static machine code."}]},
        separators=(",", ":"),
    )
    return [
        make_record("pretrain", f"OpenWALDO verified document number {i} for Omega Helix training: "
                                f"the universal engine trains any architecture on verified data.")
        for i in range(9)
    ] + [make_record("conversation", conv)]


class OpenWaldoBackendSpecTests(unittest.TestCase):
    """Fail-closed spec validation (no binaries needed)."""

    BASE = {"export_dir": "x", "workdir": "y", "arch": "llama"}

    def bad(self, regex: str, **kw) -> None:
        with self.assertRaisesRegex(backend.BackendError, regex):
            backend.validate_spec({**self.BASE, **kw})

    def test_valid_defaults(self) -> None:
        s = backend.validate_spec(self.BASE)
        self.assertEqual((s["tokenizer"], s["vocab"], s["T"], s["H_KV"], s["_flags"]), ("byte", 256, 32, 2, 23))

    def test_every_preset_validates(self) -> None:
        for arch in backend.PRESETS:
            backend.validate_spec({**self.BASE, "arch": arch})
        backend.validate_spec({**self.BASE, "arch": "custom", "flags": 31})

    def test_refusals(self) -> None:
        self.bad("unknown spec keys", bogus=1)
        self.bad("arch must be one of", arch="resnet")
        self.bad("missing required spec key 'export_dir'", export_dir="")
        self.bad("divisible", D=30, H=4)
        self.bad("multiple of H_KV", H=4, H_KV=3)
        self.bad("even head dimension", D=12, H=4)
        self.bad("custom requires explicit flags", arch="custom")
        self.bad("only valid with arch=custom", flags=3)
        self.bad("exclusive", arch="custom", flags=66)
        self.bad("excludes", arch="custom", flags=96)
        self.bad("tokenizer=u32 requires vocab > 65535", tokenizer="u32", vocab=1000)
        self.bad("tokenizer=bpe requires", tokenizer="bpe", vocab=200)
        self.bad("tokenizer=byte requires vocab >= 256", vocab=100)
        self.bad("looped core needs", LP=1, LC=2, NP=2, L=2)
        self.bad("non-negative integer", steps=-1)
        self.bad("non-negative integer", D="32")
        self.bad("steps must be >= 1", steps=0)
        self.bad("cannot be combined with hf_dir", hf_dir="/tmp/h", NP=2)

    def test_cli_invalid_spec_rc1(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            sp = Path(td) / "spec.json"
            sp.write_text(json.dumps({**self.BASE, "export_dir": td, "workdir": td + "/w", "D": 30, "H": 4}))
            self.assertEqual(backend.main(["--spec", str(sp)]), 1)
            obs = json.loads((Path(td) / "w" / backend.OBS_NAME).read_text())
            self.assertEqual((obs["status"], obs["stage"]), ("failed", "spec"))
            self.assertFalse((Path(td) / "w" / backend.BOM_NAME).exists())
            sp.write_text("{not json")
            self.assertEqual(backend.main(["--spec", str(sp)]), 1)


class OpenWaldoBackendEndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tools = _locate_tools()
        for need in ("waldo", "univ"):
            if need not in cls.tools:
                raise unittest.SkipTest(f"{need} binary unavailable (set OLLM_*_BIN or install omc)")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.export = build_export_fixture(cls.root, _corpus_records())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def spec(self, name: str, **kw):
        s = {
            "export_dir": str(self.export), "workdir": str(self.root / name),
            "ollm_waldo": self.tools["waldo"], "ollm_univ": self.tools["univ"],
            "D": 32, "H": 4, "H_KV": 2, "F": 64, "L": 2, "T": 32, "B": 2, "steps": 20,
        }
        if "ollm" in self.tools:
            s["ollm"] = self.tools["ollm"]
        if "interchange" in self.tools:
            s["ollm_interchange"] = self.tools["interchange"]
        s.update(kw)
        return s

    def ok(self, name: str, **kw):
        rc, obs = backend.run_backend(self.spec(name, **kw))
        self.assertEqual(rc, 0, obs.get("error"))
        self.assertEqual(obs["status"], "ok")
        return obs

    def check_obs(self, obs, arch: str, vocab: int, width: int) -> None:
        wd = Path(obs["spec"]["workdir"])
        # on-disk observations and BOM agree with the returned record
        disk = json.loads((wd / backend.OBS_NAME).read_text())
        self.assertEqual(disk["checkpoint"]["sha256"], obs["checkpoint"]["sha256"])
        ck = wd / "model.ck"
        self.assertEqual(hashlib.sha256(ck.read_bytes()).hexdigest(), obs["checkpoint"]["sha256"])
        self.assertEqual(obs["checkpoint"]["magic"], "OLLMCK04")
        self.assertEqual(obs["checkpoint"]["step"], obs["spec"]["steps"])
        self.assertEqual(obs["checkpoint"]["shape"]["V"], vocab)
        self.assertEqual(obs["tokenization"]["token_width"], width)
        # dataset BOM carried through
        self.assertEqual(obs["dataset"]["total_docs"], 10)
        self.assertEqual(obs["dataset"]["licenses"], ["CC0-1.0"])
        exp_sha = hashlib.sha256((self.export / "EXPORT.json").read_bytes()).hexdigest()
        self.assertEqual(obs["dataset"]["export_json_sha256"], exp_sha)
        bom = json.loads((wd / backend.BOM_NAME).read_text())
        self.assertEqual(bom["kind"], "omega-helix-training-bom")
        self.assertEqual(bom["openwaldo_export"]["export_json_sha256"], exp_sha)
        self.assertEqual(len(bom["checkpoints"]), 1)
        self.assertEqual(bom["checkpoints"][0]["sha256"], obs["checkpoint"]["sha256"])
        self.assertEqual(bom["training"]["spec"]["arch"], arch)
        # every recorded artifact hash matches the file
        for a in (bom["artifacts"]["train_bin"], bom["artifacts"]["held_bin"], obs["tokenization"]["train"]):
            self.assertEqual(bridge.file_sha256_and_bytes(Path(a["path"]))[0], a["sha256"])
        # eval before/after present, finite, and a real train log exists
        for phase in ("eval_before", "eval_after"):
            for split in ("held", "train"):
                if split in obs[phase]:
                    self.assertGreater(obs[phase][split]["loss"], 0.0)
                    self.assertEqual(obs[phase][split]["step"], 0 if phase == "eval_before" else obs["spec"]["steps"])
        self.assertTrue(obs["train_log"])
        self.assertIn("ollm-univ", obs["tools"])

    def test_end_to_end_all_architectures(self) -> None:
        results = {}
        for arch in backend.PRESETS:
            with self.subTest(arch=arch):
                obs = self.ok(f"arch-{arch}", arch=arch)
                self.check_obs(obs, arch, 256, 1)
                self.assertTrue(obs["improved"], f"{arch}: loss did not improve")
                p = obs["eval_before"]["primary_split"]
                results[arch] = (obs["eval_before"][p]["loss"], obs["eval_after"][p]["loss"])
        for arch, (b, a) in results.items():
            print(f"  [PASS] {arch:8s} held/primary loss {b:.6f} -> {a:.6f}")

    def test_deterministic_retrain_same_checkpoint(self) -> None:
        a = self.ok("det-a", arch="llama")
        b = self.ok("det-b", arch="llama")
        self.assertEqual(a["checkpoint"]["sha256"], b["checkpoint"]["sha256"])
        c = self.ok("det-c", arch="llama", seed=7)
        self.assertNotEqual(a["checkpoint"]["sha256"], c["checkpoint"]["sha256"])

    def test_cli_flags_and_spec_json(self) -> None:
        s = self.spec("cli-flags", arch="qwen3")
        argv = ["--export-dir", s["export_dir"], "--workdir", s["workdir"], "--arch", "qwen3",
                "--ollm-waldo", s["ollm_waldo"], "--ollm-univ", s["ollm_univ"], "--steps", "6", "--T", "16",
                "--D", "16", "--F", "32", "--B", "1", "--print-observations"]
        with contextlib.redirect_stdout(io.StringIO()) as cap:
            self.assertEqual(backend.main(argv), 0)
        self.assertEqual(json.loads(cap.getvalue().split("\nOPENWALDO_BACKEND_OK")[0])["status"], "ok")
        sp = self.root / "spec.json"
        s2 = self.spec("cli-spec", arch="gemma", steps=6, T=16, D=16, F=32, B=1)
        sp.write_text(json.dumps(s2))
        self.assertEqual(backend.main(["--spec", str(sp), "--steps", "8"]), 0)
        obs = json.loads((self.root / "cli-spec" / backend.OBS_NAME).read_text())
        self.assertEqual(obs["checkpoint"]["step"], 8)  # CLI flag overrides spec JSON

    def test_looped_core_and_custom_flags(self) -> None:
        obs = self.ok("looped", arch="llama", L=3, LP=1, LC=1, NP=3, steps=6)
        self.assertEqual(obs["checkpoint"]["loop"], {"LP": 1, "LC": 1, "NP": 3})
        self.check_obs(obs, "llama", 256, 1)
        obs = self.ok("custom", arch="custom", flags=31, steps=6, rope_theta_milli=1000000000)
        self.assertEqual(obs["checkpoint"]["flags"], 31)

    def test_u16_widened_byte_tokens_and_u32_vocab(self) -> None:
        obs = self.ok("w2", arch="gpt", vocab=1000, steps=4)
        self.check_obs(obs, "gpt", 1000, 2)
        obs = self.ok("u32", arch="llama", tokenizer="u32", vocab=70000, D=16, F=32, steps=3, B=1, T=16)
        self.check_obs(obs, "llama", 70000, 4)
        self.assertEqual(obs["tokenization"]["mode"], "u32")

    def test_bpe_tokenizer_end_to_end(self) -> None:
        if "ollm" not in self.tools:
            self.skipTest(f"ollm (BPE) binary unavailable: {_BUILD_ERR.get('ollm', 'no build')}")
        for arch in ("llama", "qwen3"):
            with self.subTest(arch=arch):
                obs = self.ok(f"bpe-{arch}", arch=arch, tokenizer="bpe", vocab=280, steps=12)
                self.check_obs(obs, arch, 280, 2)
                self.assertIn("merges", obs["tokenization"])
                self.assertEqual(
                    bridge.file_sha256_and_bytes(Path(obs["tokenization"]["merges"]["path"]))[0],
                    obs["tokenization"]["merges"]["sha256"],
                )

    # ---- negative controls: every failure exits 1, records a failed observation, binds no BOM

    def fails(self, name: str, stage: str, regex: str, **kw):
        rc, obs = backend.run_backend(self.spec(name, **kw))
        self.assertEqual(rc, 1)
        self.assertEqual(obs["status"], "failed")
        self.assertEqual(obs["stage"], stage)
        self.assertRegex(obs["error"], regex)
        wd = Path(self.root / name)
        self.assertFalse((wd / backend.BOM_NAME).exists())
        self.assertFalse((wd / "model.ck").exists() and obs["stage"] in ("openwaldo-verify", "ingest"))
        self.assertEqual(json.loads((wd / backend.OBS_NAME).read_text())["status"], "failed")

    def test_refuses_tampered_dataset(self) -> None:
        bad = build_export_fixture(self.root / "tamper", _corpus_records())
        shard = bad / "data/example/shard-000.jsonl"
        data = bytearray(shard.read_bytes())
        data[5] ^= 0x01
        shard.write_bytes(bytes(data))
        self.fails("neg-tamper", "openwaldo-verify", "sha256 mismatch", export_dir=str(bad), arch="llama")

    def test_refuses_bad_record_hash_and_missing_export(self) -> None:
        rec = json.loads(_corpus_records()[0])
        rec["sha256"] = "b" * 64
        bad = build_export_fixture(self.root / "badrec", [json.dumps(rec) + "\n"])
        self.fails("neg-badrec", "openwaldo-verify", "record sha256 mismatch", export_dir=str(bad), arch="llama")
        self.fails("neg-noexp", "openwaldo-verify", "missing OpenWALDO EXPORT.json",
                   export_dir=str(self.root / "nonexistent"), arch="llama")

    def test_refuses_native_format_export(self) -> None:
        nat = build_export_fixture(self.root / "native", _corpus_records())
        doc = json.loads((nat / "EXPORT.json").read_text())
        doc["format"] = "native"
        (nat / "EXPORT.json").write_text(json.dumps(doc))
        self.fails("neg-native", "openwaldo-verify", "format must be canonical jsonl", export_dir=str(nat), arch="llama")

    def test_refuses_corpus_smaller_than_context(self) -> None:
        self.fails("neg-small", "tokenize", "corpus too small", arch="llama", T=4096)

    def test_refuses_missing_or_bad_binaries(self) -> None:
        self.fails("neg-nobin", "tools", "not executable", arch="llama", ollm_univ=str(self.root / "no-such-univ"))
        notexe = self.root / "notexe"
        notexe.write_text("not a program")
        self.fails("neg-notexe", "tools", "not executable", arch="llama", ollm_waldo=str(notexe))

    def test_refuses_engine_geometry_refusal(self) -> None:
        # passes spec validation but ollm-univ must refuse (V=2^64-1 style overflow is out of scope);
        # an HF dir without model.safetensors is refused by the adapter before any engine call
        hf = self.root / "emptyhf"
        hf.mkdir()
        (hf / "config.json").write_text("{}")
        self.fails("neg-hf", "spec", "config.json and model.safetensors", arch="llama", hf_dir=str(hf))

    def test_refuses_failed_prior_state_is_cleared(self) -> None:
        self.ok("clear", arch="llama", steps=4)
        self.fails("clear", "tokenize", "corpus too small", arch="llama", T=4096)  # old BOM must be gone

    # ---- contract test with a stub ollm-interchange (argument/IO contract of import-hf/export-hf/export-gguf)

    STUB = """#!{py}
import json, os, struct, subprocess, sys
a = sys.argv[1:]
univ = os.environ["STUB_UNIV"]
if a[0] == "import-hf":
    cfg = json.load(open(a[1])); assert os.path.getsize(a[2]) > 0
    T, B, total, lr = a[4:8]
    subprocess.run([univ, "init", a[3], "llama", str(cfg["vocab_size"]), T, str(cfg["hidden_size"]),
        str(cfg["num_attention_heads"]), str(cfg["num_key_value_heads"]), str(cfg["intermediate_size"]),
        str(cfg["num_hidden_layers"]), B, "3", total, lr], check=True)
elif a[0] == "export-hf":
    assert os.path.getsize(a[1]) > 0 and a[3] in ("f32", "bf16")
    assert os.path.isdir(a[2])  # real tool requires OUT_DIR to exist
    json.dump({{"model_type": "llama"}}, open(os.path.join(a[2], "config.json"), "w"))
    open(os.path.join(a[2], "model.safetensors"), "wb").write(struct.pack("<Q", 2) + b"{{}}")
elif a[0] == "export-gguf":
    open(a[2], "wb").write(b"GGUF" + struct.pack("<I", 3))
else:
    sys.exit(2)
"""

    def test_hf_import_and_export_plumbing_with_stub_interchange(self) -> None:
        stub = self.root / "stub-interchange"
        stub.write_text(self.STUB.format(py=sys.executable))
        stub.chmod(0o755)
        os.environ["STUB_UNIV"] = self.tools["univ"]
        hf = self.root / "stubhf"
        hf.mkdir()
        (hf / "config.json").write_text(json.dumps(
            {"vocab_size": 256, "hidden_size": 32, "num_attention_heads": 4, "num_key_value_heads": 2,
             "intermediate_size": 64, "num_hidden_layers": 2, "max_position_embeddings": 2048}))
        (hf / "model.safetensors").write_bytes(b"\x00" * 16)
        obs = self.ok("stub-hf", arch="llama", hf_dir=str(hf), steps=6, ollm_interchange=str(stub),
                      export_hf=True, hf_dtype="bf16", export_gguf=True, T=32)
        self.assertEqual(obs["checkpoint"]["shape"]["V"], 256)
        self.assertEqual(obs["checkpoint"]["shape"]["D"], 32)
        self.assertEqual({e["format"] for e in obs["exports"]}, {"hf-safetensors", "gguf"})
        self.assertEqual([e for e in obs["exports"] if e["format"] == "gguf"][0]["version"], 3)
        bom = json.loads((self.root / "stub-hf" / backend.BOM_NAME).read_text())
        self.assertEqual(bom["training"]["source"], "hf-import")
        self.assertIn("ollm-interchange", obs["tools"])
        # export step failure is fail-closed (no BOM)
        broken = self.root / "broken-interchange"
        broken.write_text("#!/bin/sh\nexit 3\n")
        broken.chmod(0o755)
        self.fails("stub-broken", "export-gguf", "exited 3", arch="llama", ollm_interchange=str(broken), export_gguf=True)

    # ---- optional ollm-interchange: HF Safetensors / GGUF export + HF re-import

    def test_interchange_exports_and_hf_import(self) -> None:
        if "interchange" not in self.tools:
            self.skipTest("ollm-interchange not built (TARGET=interchange / OLLM_INTERCHANGE_BIN)")
        obs = self.ok("export", arch="llama", steps=6, export_hf=True, hf_dtype="bf16", export_gguf=True)
        by_fmt = {e["format"]: e for e in obs["exports"]}
        self.assertEqual(by_fmt["gguf"]["version"], 3)
        names = {f["name"] for f in by_fmt["hf-safetensors"]["files"]}
        self.assertTrue({"config.json", "model.safetensors"} <= names)
        bom = json.loads((self.root / "export" / backend.BOM_NAME).read_text())
        self.assertEqual(len(bom["exports"]), 2)
        for e in bom["exports"]:
            for f in e.get("files", [e]):
                self.assertEqual(bridge.file_sha256_and_bytes(Path(f["path"]))[0], f["sha256"])
        # fine-tune from the exported HF weights on the same verified corpus
        obs2 = self.ok("from-hf", arch="llama", hf_dir=str(self.root / "export" / "export-hf"), steps=6, T=32)
        self.assertEqual(obs2["checkpoint"]["shape"]["V"], 256)
        self.assertEqual(bom["training"]["source"], "init")
        self.assertEqual(json.loads((self.root / "from-hf" / backend.BOM_NAME).read_text())["training"]["source"], "hf-import")


if __name__ == "__main__":
    unittest.main()
