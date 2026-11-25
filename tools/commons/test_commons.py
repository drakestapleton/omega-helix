"""Comprehensive end-to-end tests for Sovereign Commons, Trust Audit, Memory Migration,
Hardware-Porting Kit, Swarm Quorum, Capability Leases, USB Air-Gap Bundles, CLI & Studio API.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

COMMONS_DIR = Path(__file__).resolve().parent
TOOLS_DIR = COMMONS_DIR.parent
ROOT_DIR = TOOLS_DIR.parent
STUDIO_DIR = TOOLS_DIR / "studio"

for _p in (str(TOOLS_DIR), str(STUDIO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import commons  # noqa: E402
import chain_runner  # noqa: E402
import server as studio_server  # noqa: E402


def _resolve_ollm() -> Path:
    env_bin = os.environ.get("OLLM_BIN")
    if env_bin and Path(env_bin).is_file():
        return Path(env_bin).resolve()
    bins = chain_runner.ensure_binaries(["ollm"], root=ROOT_DIR)
    return bins["ollm"]


class SovereignCommonsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ollm = _resolve_ollm()
        cls.td_obj = tempfile.TemporaryDirectory()
        cls.td = Path(cls.td_obj.name)

        cls.data_bin = cls.td / "corpus.bin"
        cls.data_bin.write_bytes((b"Sovereign memory and composable skill deltas.\n") * 24)

        cls.base2_ck = cls.td / "base2.ck"
        cls.base2b_ck = cls.td / "base2b.ck"
        subprocess.run(
            [str(cls.ollm), "init", str(cls.base2_ck), "256", "16", "16", "2", "32", "2", "2", "42", "4", "3000"],
            check=True, capture_output=True,
        )
        subprocess.run(
            [str(cls.ollm), "init", str(cls.base2b_ck), "256", "16", "16", "2", "32", "2", "2", "99", "4", "3000"],
            check=True, capture_output=True,
        )

        cls.hc2 = cls.td / "base2.hc"
        cls.cand3_ck = cls.td / "cand3.ck"
        cls.sp1 = cls.td / "skill_v1.sp"
        subprocess.run([str(cls.ollm), "growpro", str(cls.base2_ck), str(cls.cand3_ck), "3", "4", "2000"], check=True, capture_output=True)
        subprocess.run([str(cls.ollm), "fcache", str(cls.base2_ck), str(cls.data_bin), "4", str(cls.hc2)], check=True, capture_output=True)
        subprocess.run([str(cls.ollm), "trainfc", str(cls.cand3_ck), str(cls.hc2), "4", "2", "2"], check=True, capture_output=True)
        subprocess.run([str(cls.ollm), "spext", str(cls.base2_ck), str(cls.cand3_ck), str(cls.sp1)], check=True, capture_output=True)

        # Second chained pack: L3 -> L4 on top of cand3_ck
        cls.cand4_ck = cls.td / "cand4.ck"
        cls.hc3 = cls.td / "cand3.hc"
        cls.sp2 = cls.td / "skill_v2.sp"
        subprocess.run([str(cls.ollm), "growpro", str(cls.cand3_ck), str(cls.cand4_ck), "4", "4", "2000"], check=True, capture_output=True)
        subprocess.run([str(cls.ollm), "fcache", str(cls.cand3_ck), str(cls.data_bin), "4", str(cls.hc3)], check=True, capture_output=True)
        subprocess.run([str(cls.ollm), "trainfc", str(cls.cand4_ck), str(cls.hc3), "4", "2", "3"], check=True, capture_output=True)
        subprocess.run([str(cls.ollm), "spext", str(cls.cand3_ck), str(cls.cand4_ck), str(cls.sp2)], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.td_obj.cleanup()

    def test_01_pkg_publish_search_install_diff_and_revoke(self) -> None:
        reg_dir = self.td / "commons_reg"
        env_dir = self.td / "user_env"
        entry = commons.publish_pack(
            self.sp1,
            "sql-audit",
            registry_dir=reg_dir,
            version="1.0.0",
            license_id="Apache-2.0",
            description="Deterministic SQL audit strand",
        )
        found = commons.search_packs(reg_dir, query="sql", base_hash=entry["base_hash"])
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["name"], "sql-audit")

        installed = commons.install_pack("sql-audit", env_dir, registry_dir=reg_dir, expected_base_hash=entry["base_hash"])
        self.assertTrue(Path(installed["installed_path"]).is_file())

        with self.assertRaises(ValueError):
            commons.install_pack("sql-audit", env_dir, registry_dir=reg_dir, expected_base_hash="0000000000000000")

        # Tamper blob payload -> fail closed on FNV-1a / SHA-256 mismatch
        bad_reg = self.td / "bad_reg"
        shutil.copytree(reg_dir, bad_reg)
        blob_p = bad_reg / entry["blob_rel"]
        bdata = bytearray(blob_p.read_bytes())
        bdata[-1] ^= 0xFF
        blob_p.write_bytes(bytes(bdata))
        with self.assertRaises(ValueError):
            commons.install_pack("sql-audit", self.td / "bad_env", registry_dir=bad_reg)

        diff = commons.diff_packs(self.sp1, self.sp2)
        self.assertFalse(diff["same_pack"])
        self.assertFalse(diff["same_base_L"])

        rev = commons.revoke_pack("sql-audit", env_dir, reason="right-to-be-forgotten")
        self.assertTrue(rev["file_deleted"])
        self.assertFalse(Path(installed["installed_path"]).exists())
        self.assertIn("right-to-be-forgotten", (env_dir / "revocations.jsonl").read_text(encoding="utf-8"))

    def test_02_cryptographic_trust_audit(self) -> None:
        res = commons.verify_sovereign_trust(
            bin_dir=self.ollm.parent,
            root=ROOT_DIR,
            base_ck=self.base2_ck,
            packs=[self.sp1, self.sp2],
        )
        self.assertTrue(res["ok"], f"Trust check failed: {res}")
        names = {c["name"] for c in res["checks"]}
        self.assertIn("studio_ui_csp", names)
        self.assertIn("server_loopback_default", names)
        self.assertIn("merkle_sp_chain", names)

        # Corrupt .sp payload -> trust audit fails closed
        corrupt_sp = self.td / "corrupt.sp"
        raw = bytearray(self.sp1.read_bytes())
        raw[48] ^= 0x01
        corrupt_sp.write_bytes(bytes(raw))
        bad_res = commons.verify_sovereign_trust(
            bin_dir=self.ollm.parent,
            root=ROOT_DIR,
            base_ck=self.base2_ck,
            packs=[corrupt_sp],
        )
        self.assertFalse(bad_res["ok"])

        # Broken layer chain ([sp2, sp1]: sp2 starts at base_L=3 != base2_ck L=2)
        broken = commons.verify_sovereign_trust(
            bin_dir=self.ollm.parent,
            root=ROOT_DIR,
            base_ck=self.base2_ck,
            packs=[self.sp2, self.sp1],
        )
        self.assertFalse(broken["ok"])

    def test_03_personal_memory_encryption_and_trunk_migration(self) -> None:
        store = self.td / "mem" / "memory.jsonl"
        ms = commons.MemoryStore(store)
        ms.add("Always run verify.sh before pushing main.", tags=["workflow"])
        ms.add("Always run verify.sh before pushing main.", tags=["dup"])  # SHA-256 dedup
        ms.add("Never ignore ResourceWarning in Python stdlib tests.", tags=["python"])
        self.assertEqual(len(ms.list_memories()), 2)

        enc_p = self.td / "mem" / "memory.enc"
        commons.export_encrypted(store, enc_p, "correct-horse-battery")
        with self.assertRaises(ValueError):
            commons.import_encrypted(enc_p, self.td / "mem" / "bad.jsonl", "wrong-passphrase")

        restored_p = self.td / "mem" / "restored.jsonl"
        imp = commons.import_encrypted(enc_p, restored_p, "correct-horse-battery")
        self.assertEqual(imp["records"], 2)

        # Migrate same memories onto two distinct base trunks -> distinct base_hash pins
        mig_a = commons.migrate_brain_to_trunk(self.ollm, restored_p, self.base2_ck, self.td / "mig_a", steps=4)
        mig_b = commons.migrate_brain_to_trunk(self.ollm, restored_p, self.base2b_ck, self.td / "mig_b", steps=4)
        self.assertNotEqual(mig_a["sp_pack"]["base_hash"], mig_b["sp_pack"]["base_hash"])
        self.assertTrue(Path(mig_a["receipt_path"]).is_file())

    def test_04_hardware_porting_kit_parity(self) -> None:
        receipt_p = self.td / "port-receipt.json"
        rep = commons.run_port_parity_check(self.ollm, root=ROOT_DIR, out_json=receipt_p)
        self.assertTrue(rep["ok"], f"Port parity failed: {rep}")
        self.assertEqual(rep["passed_count"], 9)
        self.assertEqual(rep["total_gates"], 9)
        self.assertTrue(receipt_p.is_file())

    def test_05_trustless_swarm_quorum(self) -> None:
        b1 = self.td / "node1.json"
        b2 = self.td / "node2.json"
        b3 = self.td / "node3.json"
        commons.create_evidence_bundle(self.sp1, node_id="node-alpha", eval_before=2.40, eval_after=2.15, out_path=b1)
        commons.create_evidence_bundle(self.sp1, node_id="node-beta", eval_before=2.40, eval_after=2.14, out_path=b2)
        commons.create_evidence_bundle(self.sp1, node_id="node-gamma", eval_before=2.40, eval_after=2.16, out_path=b3)

        under = commons.tally_quorum([b1, b2], self.sp1, min_quorum=3)
        self.assertFalse(under["quorum_met"])

        met = commons.tally_quorum([b1, b2, b3], self.sp1, min_quorum=3)
        self.assertTrue(met["quorum_met"])
        self.assertEqual(met["valid_node_count"], 3)

        # Tamper node3 digest
        tampered = json.loads(b3.read_text(encoding="utf-8"))
        tampered["eval_after_nats"] = 0.01
        b3.write_text(json.dumps(tampered), encoding="utf-8")
        after_tamper = commons.tally_quorum([b1, b2, b3], self.sp1, min_quorum=3)
        self.assertFalse(after_tamper["quorum_met"])
        self.assertEqual(len(after_tamper["invalid_bundles"]), 1)

    def test_06_capability_lease_and_snapshot_rollback(self) -> None:
        work = self.td / "sandbox"
        work.mkdir()
        target_f = work / "keep.txt"
        target_f.write_text("original-state\n", encoding="utf-8")
        snap = self.td / "snap"
        commons.snapshot_dir(work, snap)

        lease_p = self.td / "agent.lease.json"
        audit_p = self.td / "audit.jsonl"
        commons.issue_lease(lease_p, allowed_dirs=[work], allowed_cmds=[sys.executable], allow_network=False, ttl_seconds=600)

        out = commons.execute_under_lease(
            lease_p,
            [sys.executable, "-c", "import os; print(os.environ.get('OMEGA_OFFLINE', ''))"],
            work,
            audit_log_path=audit_p,
        )
        self.assertEqual(out["returncode"], 0)
        self.assertEqual(out["stdout"].strip(), "1")

        # Disallowed directory & disallowed command must fail closed
        outside = self.td / "outside"
        outside.mkdir()
        with self.assertRaises(PermissionError):
            commons.execute_under_lease(lease_p, [sys.executable, "-c", "pass"], outside)
        with self.assertRaises(PermissionError):
            commons.execute_under_lease(lease_p, ["sh", "-c", "echo hi"], work)

        # Snapshot + rollback restores mutated/deleted files
        target_f.unlink()
        (work / "rogue.txt").write_text("bad", encoding="utf-8")
        commons.rollback_dir(work, snap)
        self.assertEqual(target_f.read_text(encoding="utf-8"), "original-state\n")
        self.assertFalse((work / "rogue.txt").exists())

        commons.revoke_lease(lease_p)
        with self.assertRaises(PermissionError):
            commons.execute_under_lease(lease_p, [sys.executable, "-c", "pass"], work)

    def test_07_usb_bundle_cli_and_studio_endpoints(self) -> None:
        mem_store = self.td / "usb_mem.jsonl"
        commons.MemoryStore(mem_store).add("Portable air-gap memory", tags=["airgap"])

        usb_dir = self.td / "usb_bundle"
        usb_info = commons.pack_usb_bundle(
            usb_dir,
            bin_dir=self.ollm.parent,
            root=ROOT_DIR,
            trunk_ck=self.base2_ck,
            packs=[self.sp1],
            memory_store=mem_store,
        )
        self.assertGreater(usb_info["file_count"], 5)
        self.assertTrue(commons.verify_usb_bundle(usb_dir)["verified"])

        # Run offline wrapper inside unpacked bundle
        res_off = subprocess.run(
            ["sh", str(usb_dir / "START-OFFLINE.sh"), "verify", "--json"],
            capture_output=True, text=True, check=True,
        )
        parsed_off = json.loads(res_off.stdout)
        self.assertTrue(parsed_off["ok"])

        # Tamper 1 byte -> fail closed
        sp_in_usb = usb_dir / "packs" / self.sp1.name
        b = bytearray(sp_in_usb.read_bytes())
        b[0] ^= 0xFF
        sp_in_usb.write_bytes(bytes(b))
        v_bad = commons.verify_usb_bundle(usb_dir)
        self.assertFalse(v_bad["verified"])

        # CLI commands smoke check
        cli_env = dict(os.environ)
        cli_env["OMEGA_HELIX_BIN_DIR"] = str(self.ollm.parent)
        p_ver = subprocess.run(
            [sys.executable, str(STUDIO_DIR / "cli.py"), "verify", "--json", "--base-ck", str(self.base2_ck), "--pack", str(self.sp1)],
            env=cli_env, capture_output=True, text=True, check=True,
        )
        self.assertTrue(json.loads(p_ver.stdout)["ok"])

        p_prt = subprocess.run(
            [sys.executable, str(STUDIO_DIR / "cli.py"), "port-check", "--bin", str(self.ollm), "--json"],
            env=cli_env, capture_output=True, text=True, check=True,
        )
        self.assertTrue(json.loads(p_prt.stdout)["ok"])

        # Studio HTTP API endpoints under ResourceWarning hygiene
        mgr = chain_runner.ChainManager(runs_dir=self.td / "studio_runs", bin_dir=self.ollm.parent, root=ROOT_DIR)
        srv = studio_server.StudioServer(("127.0.0.1", 0), mgr)
        port = srv.server_address[1]
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/trust/verify", timeout=10) as resp:
                vdoc = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(vdoc["ok"])

            req_usb = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/commons/pack-usb",
                data=json.dumps({"out_dir": str(self.td / "usb_via_api"), "trunk_ck": str(self.base2_ck)}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req_usb, timeout=15) as resp:
                udoc = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(udoc["ok"])
                self.assertTrue(commons.verify_usb_bundle(self.td / "usb_via_api")["verified"])

            req_mig = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/commons/memory/migrate",
                data=json.dumps({
                    "store_path": str(mem_store),
                    "new_trunk_ck": str(self.base2_ck),
                    "out_dir": str(self.td / "mig_via_api"),
                    "steps": 4,
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req_mig, timeout=30) as resp:
                mdoc = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(mdoc["ok"])
                self.assertIn("sp_pack", mdoc["receipt"])
        finally:
            srv.shutdown()
            srv.server_close()
            th.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
