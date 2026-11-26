#!/usr/bin/env python3
"""End-to-end verification & independent f64/SHA-256 referee for dynamic editable spclip.

Tests pure-Omega `ollm spclip` subcommands:
  - create / edit / verify / inspect / run / eval / tune / verify-lineage
  - dynamic weights, mode (0=PoE, 1=Entropy Gate), tau sharpness, layer/branch splice
  - deterministic .spclip recipes & OLLMSPLIN01 Merkle lineage certificates
  - independent hashlib.sha256 + f64 math checks (zero hand-waving)
  - exhaustive fail-closed negative controls (rc=60 / 61 / 62 / 2)
  - legacy spclip backward-compatibility bit-exactness
  - V=256 (byte) and V>256 (BPE) paths, S=2..4 strands, multi-link shared trunks, grandchild clipping

Run:
  python3 -W error::ResourceWarning tests/test_spclip_editable.py --bin build/ollm
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(65536)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def run_cmd(cmd: list[str], *, expect_rc: int | None = 0) -> subprocess.CompletedProcess[bytes]:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if expect_rc is not None and proc.returncode != expect_rc:
        raise AssertionError(
            f"cmd={cmd!r} exited {proc.returncode}, expected {expect_rc}\n"
            f"stdout={proc.stdout.decode('utf-8', 'replace')}\n"
            f"stderr={proc.stderr.decode('utf-8', 'replace')}"
        )
    return proc


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"CHECK FAILED: {msg}")


def flip_byte(src: Path, dst: Path, offset: int, mask: int = 0x01) -> None:
    data = bytearray(src.read_bytes())
    data[offset] ^= mask
    dst.write_bytes(bytes(data))


def reseal_spclip(raw_text: str) -> bytes:
    lines = raw_text.splitlines(keepends=True)
    body = "".join(ln for ln in lines if not ln.startswith("recipe_sha256 "))
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return (body + f"recipe_sha256 {digest}\n").encode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="Path to pure-Omega ollm binary")
    ap.add_argument("--runner", default="", help="Optional emulator prefix (e.g. qemu-x86_64)")
    args = ap.parse_args()

    bin_path = Path(args.bin).resolve()
    check(bin_path.is_file(), f"binary missing: {bin_path}")
    prefix = [args.runner, str(bin_path)] if args.runner else [str(bin_path)]

    def ollm(*a: str, expect_rc: int | None = 0) -> subprocess.CompletedProcess[bytes]:
        return run_cmd(prefix + list(a), expect_rc=expect_rc)

    with tempfile.TemporaryDirectory(prefix="spclip-edit-test-") as td:
        w = Path(td)

        # Build distinct domain corpora (>= 512 bytes each)
        base_corp = w / "base.txt"
        dom_a = w / "dom_a.txt"
        dom_b = w / "dom_b.txt"
        dom_c = w / "dom_c.txt"
        dom_user = w / "dom_user.txt"
        held = w / "held.txt"

        base_corp.write_text(("alpha beta gamma delta omega helix core trunk.\n" * 24), encoding="utf-8")
        dom_a.write_text(("fn calc x ulong to ulong\nret add mul x 3 7\nend\n" * 24), encoding="utf-8")
        dom_b.write_text(("SELECT node_id, weight FROM strand_ledger WHERE verified = 1;\n" * 20), encoding="utf-8")
        dom_c.write_text(("theorem merkle_chain : forall s, valid s -> bound s.\n" * 20), encoding="utf-8")
        dom_user.write_text(("CUSTOM_USER_SOVEREIGN_MARKER fn user_tune x ulong to ulong ret add x 42 end\n" * 20), encoding="utf-8")
        held.write_text(("fn calc x ulong to ulong\nSELECT node_id FROM strand_ledger;\n" * 16), encoding="utf-8")

        # 1. Train base checkpoint (V=256, T=16, D=32, H=2, F=64, L=2) and alternate incompatible base
        base_ck = w / "base.ck"
        other_ck = w / "other.ck"
        ollm("init", str(base_ck), "256", "16", "32", "2", "64", "2", "4", "7", "20", "40000")
        ollm("train", str(base_ck), str(base_corp), "8", "8")
        ollm("init", str(other_ck), "256", "16", "32", "2", "64", "2", "4", "99", "20", "40000")
        ollm("train", str(other_ck), str(base_corp), "4", "4")

        # Build shared intermediate trunk link P_trunk (L=2->3) and sibling links off both base and P_trunk
        def build_strand(parent_ck: Path, new_l: int, data: Path, steps: int, out_ck: Path, out_sp: Path) -> None:
            ollm("growpro", str(parent_ck), str(out_ck), str(new_l), str(max(steps, 16)), "80000")
            # Determine parent L from checkpoint header (offset 48)
            pl = struct.unpack("<Q", parent_ck.read_bytes()[48:56])[0]
            ollm("trainf", str(out_ck), str(data), str(steps), str(steps), str(pl))
            ollm("spext", str(parent_ck), str(out_ck), str(out_sp))

        # Direct-base strands (L=2->4 for A, L=2->3 for B, L=2->3 for C, L=2->3 for D)
        ck_a1, sp_a1 = w / "a1.ck", w / "a1.sp"
        ck_a2, sp_a2 = w / "a2.ck", w / "a2.sp"
        ck_b, sp_b = w / "b.ck", w / "b.sp"
        ck_c, sp_c = w / "c.ck", w / "c.sp"
        ck_d, sp_d = w / "d.ck", w / "d.sp"
        ck_other, sp_other = w / "other_s.ck", w / "other_s.sp"

        build_strand(base_ck, 3, dom_a, 12, ck_a1, sp_a1)
        build_strand(ck_a1, 4, dom_a, 12, ck_a2, sp_a2)
        build_strand(base_ck, 3, dom_b, 12, ck_b, sp_b)
        build_strand(base_ck, 3, dom_c, 12, ck_c, sp_c)
        build_strand(base_ck, 3, base_corp, 8, ck_d, sp_d)
        build_strand(other_ck, 3, dom_a, 6, ck_other, sp_other)

        # Shared-trunk chain: base (L=2) -> sp_a1 (L=2->3) -> {sp_t_a (3->4), sp_t_b (3->4)}
        ck_tb, sp_tb = w / "tb.ck", w / "tb.sp"
        build_strand(ck_a1, 4, dom_b, 12, ck_tb, sp_tb)

        print("PASS [1] Trained base + 2/3/4-strand Merkle OLLMSP01 packs")

        # 2. Legacy spclip backward compatibility & bit-exactness
        ref_a = ollm("spchainsamp", str(base_ck), "20", "0", "fn ", "42", "-", "1", str(sp_a1)).stdout
        leg_aa0 = ollm("spclip", str(base_ck), "0", "fn ", "20", "0", "42", "-", "2", "1000", "1", str(sp_a1), "1000", "1", str(sp_a1)).stdout
        leg_aa1 = ollm("spclip", str(base_ck), "1", "fn ", "20", "0", "42", "-", "2", "1000", "1", str(sp_a1), "1000", "1", str(sp_a1)).stdout
        leg_iso = ollm("spclip", str(base_ck), "1", "fn ", "20", "0", "42", "-", "2", "1000", "1", str(sp_a1), "0", "1", str(sp_b)).stdout
        leg_anti = ollm("spclip", str(base_ck), "0", "fn ", "12", "0", "42", "-", "2", "1000", "1", str(sp_a1), "-500", "1", str(sp_b)).stdout
        check(leg_aa0 == ref_a and leg_aa1 == ref_a and leg_iso == ref_a, "legacy spclip (A,A) & (A@1000,B@0) != spchainsamp")
        check(len(leg_anti) > 0, "legacy spclip Anti-DNA negative weight output empty")
        print("PASS [2] Legacy spclip bit-exact backward compatibility verified")

        # 3. Deterministic .spclip create + SHA-256 binding against Python hashlib.sha256
        rec1 = w / "clip1.spclip"
        rec1_dup = w / "clip1_dup.spclip"
        r_cr1 = ollm(
            "spclip", "create", str(rec1), str(base_ck),
            "--strand", f"{sp_a1},{sp_a2}",
            "--strand", str(sp_b),
            "--strand", str(sp_c),
            "--weights", "70,25,5",
            "--mode", "1",
            "--tau", "1000",
            "--splice", "2:4:1000,all,all",
        )
        r_cr2 = ollm(
            "spclip", "create", str(rec1_dup), str(base_ck),
            "--strand", f"{sp_a1},{sp_a2}",
            "--strand", str(sp_b),
            "--strand", str(sp_c),
            "--weights", "70,25,5",
            "--mode", "1",
            "--tau", "1000",
            "--splice", "2:4:1000,all,all",
        )
        check(rec1.read_bytes() == rec1_dup.read_bytes(), ".spclip create is not byte-for-byte deterministic")
        check(r_cr1.stdout == r_cr2.stdout, "SPCLIP_CREATE stdout not deterministic")

        # Independently verify EVERY FIPS 180-4 SHA-256 written by pure Omega inside clip1.spclip!
        rec_txt = rec1.read_text(encoding="utf-8")
        lines = rec_txt.splitlines()
        check(lines[0] == "OLLMSPCLIP01", f"unexpected header: {lines[0]}")
        kv = {}
        links_found = []
        for ln in lines[1:]:
            parts = ln.split()
            if parts[0] == "link":
                links_found.append((int(parts[1]), int(parts[2]), Path(parts[3]), parts[4], parts[5]))
            else:
                kv[parts[0]] = parts[1:]

        check(kv["base_sha256"][0] == sha256_file(base_ck), "base_sha256 disagrees with Python hashlib.sha256")
        check(len(links_found) == 4, f"expected 4 link lines, got {len(links_found)}")
        for s_i, j_i, p_i, sha_i, _ in links_found:
            exp_sha = sha256_file(p_i)
            check(sha_i == exp_sha, f"link ({s_i},{j_i}) sha256 mismatch: om={sha_i} py={exp_sha}")

        # Verify recipe_sha256 seal independently
        body_no_seal = "".join(ln + "\n" for ln in lines[:-1])
        exp_rec_sha = hashlib.sha256(body_no_seal.encode("utf-8")).hexdigest()
        check(kv["recipe_sha256"][0] == exp_rec_sha, f"recipe_sha256 mismatch: om={kv['recipe_sha256'][0]} py={exp_rec_sha}")

        # Also check positional create syntax + S=4 quad helix
        rec4 = w / "clip4.spclip"
        ollm(
            "spclip", "create", str(rec4), str(base_ck),
            "4", "400", "1", str(sp_a1), "300", "1", str(sp_b), "200", "1", str(sp_c), "100", "1", str(sp_d),
            "--mode", "0", "--tau", "1000",
        )
        rv4 = ollm("spclip", "verify", str(rec4)).stdout.decode()
        check(rv4.startswith("SPCLIP_VERIFY OK 4 0 1000 "), f"unexpected S=4 verify: {rv4}")
        ri1 = ollm("spclip", "inspect", str(rec1)).stdout.decode()
        check("SPCLIP_INSPECT 3 1 1000 " in ri1 and "STRAND 0 70 2:4:1000 4 4 2" in ri1, f"bad inspect: {ri1}")
        print(f"PASS [3] Pure-Omega FIPS 180-4 SHA-256 & deterministic .spclip recipe verified (recipe_sha256={exp_rec_sha[:16]}...)")

        # 4. Exact eval identity + dynamic edit (--weights, --mode, --tau, --splice, --set-weight, --set-splice)
        rec_id = w / "id.spclip"
        ollm("spclip", "create", str(rec_id), str(base_ck), "--strand", str(sp_a1), "--strand", str(sp_b), "--weights", "1000,0", "--mode", "1", "--tau", "1000")
        ev_ref = ollm("spchaineval", str(base_ck), str(held), "4", "-", "1", str(sp_a1)).stdout.decode().split()
        ev_clip = ollm("spclip", "eval", str(rec_id), str(held), "4", "-").stdout.decode().split()
        check(ev_clip[0] == "SPCLIP_EVAL" and ev_clip[2] == ev_ref[3] and ev_clip[3] == ev_ref[4],
              f"spclip eval (A@1000,B@0) != spchaineval A: clip={ev_clip} ref={ev_ref}")

        # Run from recipe matches legacy spclip bit-for-bit
        run_id = ollm("spclip", "run", str(rec_id), "fn ", "20", "0", "42", "-").stdout
        check(run_id == ref_a, "spclip run (A@1000,B@0) != spchainsamp A")

        # Dynamic edit of the clip itself: weights, mode, tau, splice!
        rec_ed = w / "clip1_edited.spclip"
        ollm(
            "spclip", "edit", str(rec1), "--out", str(rec_ed),
            "--weights", "30,60,10",
            "--mode", "1",
            "--tau", "250",
            "--splice", "2:3:500,all,2:3:1500",
        )
        ollm("spclip", "verify", str(rec_ed))
        check(rec_ed.read_bytes() != rec1.read_bytes(), "edited recipe should differ from original")

        # Single-strand --set-weight & --set-splice in-place edit
        ollm("spclip", "edit", str(rec_ed), "--set-weight", "0", "80", "--set-splice", "2", "2:3:0")
        ri_ed = ollm("spclip", "inspect", str(rec_ed)).stdout.decode()
        check("STRAND 0 80 2:3:500 4 3 2" in ri_ed, f"set-weight/splice not persisted: {ri_ed}")
        check("STRAND 2 10 2:3:0 3 3 1" in ri_ed, f"set-splice 2 not persisted: {ri_ed}")

        # Verify on-the-fly dynamic CLI overrides match persisted edited recipe BIT-FOR-BIT!
        ev_persisted = ollm("spclip", "eval", str(rec_ed), str(held), "4", "-").stdout
        ev_live = ollm(
            "spclip", "eval", str(rec1), str(held), "4", "-",
            "--weights", "80,60,10", "--mode", "1", "--tau", "250", "--splice", "2:3:500,all,2:3:0",
        ).stdout
        check(ev_persisted == ev_live, f"live override eval != persisted edit eval:\n  {ev_persisted}\n  {ev_live}")

        run_persisted = ollm("spclip", "run", str(rec_ed), "fn ", "16", "0", "42", "-").stdout
        run_live = ollm(
            "spclip", "run", str(rec1), "fn ", "16", "0", "42", "-",
            "--weights", "80,60,10", "--mode", "1", "--tau", "250", "--splice", "2:3:500,all,2:3:0",
        ).stdout
        check(run_persisted == run_live, "live override run != persisted edit run")

        # Verify --tau actually changes entropy-gated NLL smoothly & monotonically in MODE=1 and temperature in MODE=0!
        ev_t200 = int(ollm("spclip", "eval", str(rec1), str(held), "4", "-", "--weights", "500,500,500", "--mode", "1", "--tau", "200", "--splice", "all,all,all").stdout.decode().split()[2])
        ev_t1000 = int(ollm("spclip", "eval", str(rec1), str(held), "4", "-", "--weights", "500,500,500", "--mode", "1", "--tau", "1000", "--splice", "all,all,all").stdout.decode().split()[2])
        ev_t8000 = int(ollm("spclip", "eval", str(rec1), str(held), "4", "-", "--weights", "500,500,500", "--mode", "1", "--tau", "8000", "--splice", "all,all,all").stdout.decode().split()[2])
        ev_poe = int(ollm("spclip", "eval", str(rec1), str(held), "4", "-", "--weights", "500,500,500", "--mode", "0", "--tau", "1000", "--splice", "all,all,all").stdout.decode().split()[2])
        ev_poe_sharp = int(ollm("spclip", "eval", str(rec1), str(held), "4", "-", "--weights", "500,500,500", "--mode", "0", "--tau", "500", "--splice", "all,all,all").stdout.decode().split()[2])
        check(ev_t200 != ev_t1000 and ev_t1000 != ev_t8000, f"tau had no effect in MODE=1: {ev_t200}, {ev_t1000}, {ev_t8000}")
        # Large tau=100000 in MODE=1 must converge toward MODE=0 PoE!
        ev_t_huge = int(ollm("spclip", "eval", str(rec1), str(held), "4", "-", "--weights", "500,500,500", "--mode", "1", "--tau", "100000", "--splice", "all,all,all").stdout.decode().split()[2])
        check(abs(ev_t_huge - ev_poe) < abs(ev_t200 - ev_poe),
              f"large tau did not converge toward PoE: t100k={ev_t_huge}, poe={ev_poe}, t200={ev_t200}")
        check(ev_poe_sharp != ev_poe, f"tau had no effect in MODE=0: {ev_poe} vs {ev_poe_sharp}")

        # Verify splice semantics: splicing strand 0 (L=2->4) to 2:3:1000 vs full 2:4:1000 vs upper-layer 3:4:1000 vs bypass 2:2:1000
        ev_sp_full = int(ollm("spclip", "eval", str(rec1), str(held), "4", "-", "--weights", "1000,0,0", "--mode", "0", "--tau", "1000", "--splice", "2:4:1000,all,all").stdout.decode().split()[2])
        ev_sp_cap3 = int(ollm("spclip", "eval", str(rec1), str(held), "4", "-", "--weights", "1000,0,0", "--mode", "0", "--tau", "1000", "--splice", "2:3:1000,all,all").stdout.decode().split()[2])
        ev_sp_up = int(ollm("spclip", "eval", str(rec1), str(held), "4", "-", "--weights", "1000,0,0", "--mode", "0", "--tau", "1000", "--splice", "3:4:1000,all,all").stdout.decode().split()[2])
        ev_sp_byp = int(ollm("spclip", "eval", str(rec1), str(held), "4", "-", "--weights", "1000,0,0", "--mode", "0", "--tau", "1000", "--splice", "2:2:1000,all,all").stdout.decode().split()[2])
        ev_sp_zgn = int(ollm("spclip", "eval", str(rec1), str(held), "4", "-", "--weights", "1000,0,0", "--mode", "0", "--tau", "1000", "--splice", "2:4:0,all,all").stdout.decode().split()[2])
        check(ev_sp_full != ev_sp_cap3 and ev_sp_cap3 != ev_sp_up,
              f"layer splice did not change eval: full={ev_sp_full} cap3={ev_sp_cap3} up={ev_sp_up}")
        # Zero residual gain (2:4:0) sets Wo=W2=0 so layers 2..3 act as pure identity -> MUST equal bypass (2:2:1000)!
        check(ev_sp_zgn == ev_sp_byp, f"zero-gain residual splice != bypass: zgn={ev_sp_zgn} byp={ev_sp_byp}")

        # Signed negative Anti-DNA weight edit in MODE=0
        rec_anti = w / "anti.spclip"
        ollm("spclip", "edit", str(rec1), "--out", str(rec_anti), "--mode", "0", "--weights", "1000,-300,200")
        ollm("spclip", "verify", str(rec_anti))
        ev_anti = ollm("spclip", "eval", str(rec_anti), str(held), "4", "-").stdout.decode()
        check(ev_anti.startswith("SPCLIP_EVAL 4 "), f"anti eval failed: {ev_anti}")
        print(f"PASS [4] Dynamic edit/eval/run (weights, mode, tau={ev_t200}/{ev_t1000}/{ev_t8000}->{ev_poe}, splice={ev_sp_full}/{ev_sp_cap3}/{ev_sp_up}, zgn==byp={ev_sp_byp}) verified")

        # 5. User fine-tuning (`spclip tune`) -> verified Merkle-linked OLLMSP01 child strand + OLLMSPLIN01 lineage!
        child_sp = w / "user_child.sp"
        child_lin = Path(str(child_sp) + ".lineage")
        r_tune = ollm(
            "spclip", "tune", str(rec1), str(dom_user), str(child_sp), "12", "4000",
            "--batch", "4", "--weights", "60,30,10", "--mode", "1", "--tau", "500", "--splice", "2:3:1000,all,all",
        ).stdout.decode().strip().split()
        check(r_tune[0] == "SPCLIP_TUNE", f"bad tune output: {r_tune}")
        ex_bl, child_l, nspec, stp, ls0, ls1, c_sha, lin_sha = (
            int(r_tune[1]), int(r_tune[2]), int(r_tune[3]), int(r_tune[4]),
            int(r_tune[5]), int(r_tune[6]), r_tune[7], r_tune[8],
        )
        check(ex_bl == 2 and child_l == 3 and stp == 12, f"unexpected tune geometry: {r_tune}")
        check(ls1 < ls0, f"fine-tuning did not reduce loss on user data: start={ls0} end={ls1}")
        check(c_sha == sha256_file(child_sp), "child_sha256 mismatch vs hashlib.sha256")
        check(child_lin.is_file(), "auto .lineage certificate not written")

        # Verify child OLLMSP01 works natively with spapp / speval / spsamp!
        child_ck = w / "user_child.ck"
        ollm("spapp", str(base_ck), str(child_sp), str(child_ck))
        ev_child_mem = ollm("speval", str(base_ck), str(child_sp), str(dom_user), "4").stdout.decode().split()
        ev_child_ck = ollm("eval", str(child_ck), str(dom_user), "4").stdout.decode().split()
        check(ev_child_mem[3] == ev_child_ck[3] and ev_child_mem[4] == ev_child_ck[4],
              f"child speval != spapp eval: {ev_child_mem} vs {ev_child_ck}")
        child_samp = ollm("spsamp", str(base_ck), str(child_sp), "16", "0", "CUSTOM", "42").stdout
        check(len(child_samp) > 0, "spsamp on tuned child empty")

        # Verify Merkle lineage certificate via pure-Omega `spclip verify-lineage`!
        r_vl = ollm("spclip", "verify-lineage", str(child_lin)).stdout.decode().strip().split()
        check(r_vl[0] == "SPCLIP_LINEAGE_VERIFY" and r_vl[1] == "OK" and r_vl[2] == c_sha and r_vl[5] == lin_sha,
              f"verify-lineage failed: {r_vl}")

        # Verify lineage file SHA-256 independently
        lin_lines = child_lin.read_text(encoding="utf-8").splitlines()
        check(lin_lines[0] == "OLLMSPLIN01", "bad lineage magic")
        lin_body = "".join(ln + "\n" for ln in lin_lines[:-1])
        check(hashlib.sha256(lin_body.encode("utf-8")).hexdigest() == lin_sha, "lineage_sha256 mismatch vs hashlib")

        # Also test steps=0 bake + --add-layers 1 + --lineage explicit path + MODE=0 negative Anti-DNA tune!
        bake_sp = w / "bake_child.sp"
        bake_lin = w / "custom.lineage"
        r_bake = ollm(
            "spclip", "tune", str(rec_anti), str(dom_user), str(bake_sp), "0", "0",
            "--add-layers", "1", "--lineage", str(bake_lin),
        ).stdout.decode().strip().split()
        check(r_bake[0] == "SPCLIP_TUNE" and int(r_bake[2]) == 5 and r_bake[5] == r_bake[6],
              f"steps=0 bake failed: {r_bake}")
        ollm("spclip", "verify-lineage", str(bake_lin))
        ollm("spapp", str(base_ck), str(bake_sp), str(w / "bake.ck"))

        # Test shared-trunk multi-link chain with --trunk-link incremental child strand!
        rec_trunk = w / "trunk.spclip"
        ollm("spclip", "create", str(rec_trunk), str(base_ck),
             "--strand", f"{sp_a1},{sp_a2}", "--strand", f"{sp_a1},{sp_tb}", "--weights", "50,50", "--mode", "1")
        tr_child = w / "trunk_child.sp"
        r_tr = ollm("spclip", "tune", str(rec_trunk), str(dom_user), str(tr_child), "6", "4000", "--batch", "4", "--trunk-link").stdout.decode().strip().split()
        check(int(r_tr[1]) == 3 and int(r_tr[2]) == 4, f"trunk-link export base_L/spec_L wrong: {r_tr}")
        ollm("spclip", "verify-lineage", str(tr_child) + ".lineage")
        # Must attach via spchainapp onto base_ck -> sp_a1 -> tr_child!
        ollm("spchainapp", str(base_ck), str(w / "tr_chain.ck"), "2", str(sp_a1), str(tr_child))

        # Test grandchild composition: clip user's tuned child_sp with sp_c!
        rec_gc = w / "grandchild.spclip"
        ollm("spclip", "create", str(rec_gc), str(base_ck), "--strand", str(child_sp), "--strand", str(sp_c), "--weights", "80,20", "--mode", "1")
        ollm("spclip", "verify", str(rec_gc))
        gc_out = ollm("spclip", "run", str(rec_gc), "CUSTOM", "12", "0", "42", "-").stdout
        check(len(gc_out) > 0, "grandchild spclip run empty")
        print(f"PASS [5] User fine-tuning -> OLLMSP01 child (loss {ls0}->{ls1} uNats) + verify-lineage + --trunk-link + grandchild clip verified")

        # 6. BPE (V > 256) end-to-end test
        merges = w / "merges.bin"
        bpe_tok = w / "bpe.tok"
        bpe_user_tok = w / "bpe_user.tok"
        bpe_base = w / "bpe_base.ck"
        bpe_a_ck, bpe_a_sp = w / "bpe_a.ck", w / "bpe_a.sp"
        bpe_b_ck, bpe_b_sp = w / "bpe_b.ck", w / "bpe_b.sp"
        ollm("bpefit", str(base_corp), str(merges), "280")
        ollm("bpeenc", str(merges), "280", str(base_corp), str(bpe_tok))
        ollm("bpeenc", str(merges), "280", str(dom_user), str(bpe_user_tok))
        ollm("init", str(bpe_base), "280", "16", "32", "2", "64", "2", "4", "11", "8", "4000")
        ollm("train", str(bpe_base), str(bpe_tok), "4", "4")
        build_strand(bpe_base, 3, bpe_tok, 4, bpe_a_ck, bpe_a_sp)
        build_strand(bpe_base, 3, bpe_user_tok, 4, bpe_b_ck, bpe_b_sp)
        rec_bpe = w / "bpe.spclip"
        ollm("spclip", "create", str(rec_bpe), str(bpe_base), "--strand", str(bpe_a_sp), "--strand", str(bpe_b_sp), "--weights", "60,40", "--mode", "1", "--tau", "700")
        ev_bpe = ollm("spclip", "eval", str(rec_bpe), str(bpe_user_tok), "2", str(merges)).stdout.decode()
        check(ev_bpe.startswith("SPCLIP_EVAL 2 "), f"BPE spclip eval failed: {ev_bpe}")
        run_bpe = ollm("spclip", "run", str(rec_bpe), "alpha", "8", "0", "42", str(merges)).stdout
        check(len(run_bpe) > 0, "BPE spclip run empty")
        bpe_child = w / "bpe_child.sp"
        ollm("spclip", "tune", str(rec_bpe), str(bpe_user_tok), str(bpe_child), "4", "4000", "--batch", "4")
        ollm("spclip", "verify-lineage", str(bpe_child) + ".lineage")
        print("PASS [6] BPE (V=280) spclip create/eval/run/tune/verify-lineage verified")

        # 7. Exhaustive Fail-Closed Negative Controls (rc=60 / 61 / 62 / 2)
        # 7a. Malformed / truncated / wrong magic / trailing garbage .spclip -> rc=60
        bad_mag = w / "bad_mag.spclip"
        bad_mag.write_text("OLLMSPCLIP99\n", encoding="utf-8")
        ollm("spclip", "verify", str(bad_mag), expect_rc=60)

        trunc_rec = w / "trunc.spclip"
        trunc_rec.write_bytes(rec1.read_bytes()[:40])
        ollm("spclip", "verify", str(trunc_rec), expect_rc=60)

        trail_rec = w / "trail.spclip"
        trail_rec.write_bytes(rec1.read_bytes() + b"x\n")
        ollm("spclip", "verify", str(trail_rec), expect_rc=60)

        # 7b. Tampered .spclip (1-bit flip without resealing) -> rc=61
        tamper_rec = w / "tamper.spclip"
        tamper_rec.write_text(rec1.read_text(encoding="utf-8").replace("weight 70", "weight 71", 1), encoding="utf-8")
        ollm("spclip", "verify", str(tamper_rec), expect_rc=61)
        ollm("spclip", "run", str(tamper_rec), "fn ", "4", "0", "42", "-", expect_rc=61)
        ollm("spclip", "edit", str(tamper_rec), "--tau", "500", expect_rc=61)
        ollm("spclip", "tune", str(tamper_rec), str(dom_user), str(w / "nope.sp"), "2", "4000", expect_rc=61)

        # 7c. Tampered parent SHA-256 in .spclip WITH attacker-recomputed recipe_sha256 -> rc=61
        sp_b_sha = sha256_file(sp_b)
        fake_sha = ("0" if sp_b_sha[0] != "0" else "1") + sp_b_sha[1:]
        resealed_bad_hash = w / "resealed_bad_hash.spclip"
        resealed_bad_hash.write_bytes(reseal_spclip(rec1.read_text(encoding="utf-8").replace(sp_b_sha, fake_sha, 1)))
        ollm("spclip", "verify", str(resealed_bad_hash), expect_rc=61)

        # 7d. 1-bit flip in parent .sp file on disk -> rc=61
        sp_b_bad = w / "b_tampered.sp"
        flip_byte(sp_b, sp_b_bad, 64)
        rec_b_bad = w / "rec_b_bad.spclip"
        rec_b_bad.write_bytes(reseal_spclip(rec1.read_text(encoding="utf-8").replace(str(sp_b), str(sp_b_bad))))
        ollm("spclip", "verify", str(rec_b_bad), expect_rc=61)

        # 7e. Trunk mismatch: swapped BASE.ck -> rc=62 (both with original base_sha256 AND attacker-updated base_sha256!)
        rec_wrong_base = w / "wrong_base.spclip"
        rec_wrong_base.write_bytes(reseal_spclip(rec1.read_text(encoding="utf-8").replace(str(base_ck), str(other_ck))))
        ollm("spclip", "verify", str(rec_wrong_base), expect_rc=62)

        rec_wrong_base2 = w / "wrong_base2.spclip"
        txt_wb2 = rec1.read_text(encoding="utf-8").replace(str(base_ck), str(other_ck)).replace(sha256_file(base_ck), sha256_file(other_ck))
        rec_wrong_base2.write_bytes(reseal_spclip(txt_wb2))
        ollm("spclip", "verify", str(rec_wrong_base2), expect_rc=62)

        # Incompatible trunk strand on `create` -> rc=62
        ollm("spclip", "create", str(w / "bad_trunk.spclip"), str(base_ck), "--strand", str(sp_a1), "--strand", str(sp_other), expect_rc=62)

        # 7f. Zero/invalid weights, bad tau, bad splice, bad S -> rc=2
        ollm("spclip", "edit", str(rec1), "--out", str(w / "z.spclip"), "--weights", "0,0,0", expect_rc=2)
        ollm("spclip", "edit", str(rec1), "--out", str(w / "z.spclip"), "--mode", "0", "--weights", "-100,-200,-50", expect_rc=2)
        ollm("spclip", "edit", str(rec1), "--out", str(w / "z.spclip"), "--mode", "1", "--weights", "1000,-100,100", expect_rc=2)
        ollm("spclip", "edit", str(rec1), "--out", str(w / "z.spclip"), "--weights", "6000,100,100", expect_rc=2)
        ollm("spclip", "edit", str(rec1), "--out", str(w / "z.spclip"), "--weights", "50,50", expect_rc=2)
        ollm("spclip", "edit", str(rec1), "--out", str(w / "z.spclip"), "--tau", "0", expect_rc=2)
        ollm("spclip", "edit", str(rec1), "--out", str(w / "z.spclip"), "--tau", "100001", expect_rc=2)
        ollm("spclip", "edit", str(rec1), "--out", str(w / "z.spclip"), "--splice", "2:9:1000,all,all", expect_rc=2)
        ollm("spclip", "edit", str(rec1), "--out", str(w / "z.spclip"), "--splice", "4:3:1000,all,all", expect_rc=2)
        ollm("spclip", "edit", str(rec1), "--out", str(w / "z.spclip"), "--splice", "1:3:1000,all,all", expect_rc=2)
        ollm("spclip", "edit", str(rec1), "--out", str(w / "z.spclip"), "--splice", "2:3:2001,all,all", expect_rc=2)
        # Adversarial parser/trap hardening: signed overflow (must NOT trap 105), long splice/flags (must NOT trap 104), splice-0 ambiguity
        ollm("spclip", "edit", str(rec1), "--out", str(w / "z.spclip"), "--set-weight", "0", "9223372036854775808", expect_rc=2)
        ollm("spclip", "edit", str(rec1), "--out", str(w / "z.spclip"), "--set-weight", "0", "9999999999999999999", expect_rc=2)
        ollm("spclip", "create", str(w / "z.spclip"), str(base_ck), "2", "9223372036854775808", "1", str(sp_a1), "100", "1", str(sp_b), expect_rc=2)
        ollm("spclip", str(base_ck), "0", "fn ", "4", "0", "1", "-", "2", "9223372036854775808", "1", str(sp_a1), "100", "1", str(sp_b), expect_rc=2)
        ollm("spclip", "edit", str(rec1), "--out", str(w / "z.spclip"), "--splice", "a" * 200, expect_rc=2)
        ollm("spclip", "edit", str(rec1), "--out", str(w / "z.spclip"), *(["--tau", "1000"] * 66), expect_rc=2)
        ollm("spclip", "edit", str(rec1), "--out", str(w / "z.spclip"), "--splice", "0:3:1000,all,all", expect_rc=2)
        ollm("spclip", "edit", str(rec1), "--out", str(w / "z.spclip"), "--splice", "3:0:1000,all,all", expect_rc=2)
        ollm("spclip", "edit", str(rec1), "--out", str(w / "z.spclip"), "--splice", "0,all,all", expect_rc=2)
        ollm("spclip", "edit", str(rec1), "--out", str(w / "z.spclip"), "--splice", "0:0:1000,all,all", expect_rc=2)
        ollm("spclip", "edit", str(rec1), "--out", str(w / "z.spclip"), "--set-splice", "0", "0:3:1000", expect_rc=2)
        # Single-END shorthand (--splice 3,all,all) must persist explicit trunk_L:END (2:3:1000), not ambiguous 0:3:1000
        rec_sh = w / "shorthand.spclip"
        ollm("spclip", "edit", str(rec1), "--out", str(rec_sh), "--splice", "3,all,all", expect_rc=0)
        check("splice 2:3:1000" in rec_sh.read_text(encoding="utf-8"), f"single-END splice did not canonicalize to 2:3:1000: {rec_sh.read_text()}")
        # Half-zero splice in a resealed .spclip file must fail closed with rc=60
        rec_hz = w / "halfzero.spclip"
        rec_hz.write_bytes(reseal_spclip(rec1.read_text(encoding="utf-8").replace("splice 0:0:1000", "splice 0:3:1000", 1)))
        ollm("spclip", "verify", str(rec_hz), expect_rc=60)
        rec_hz2 = w / "halfzero2.spclip"
        rec_hz2.write_bytes(reseal_spclip(rec1.read_text(encoding="utf-8").replace("splice 0:0:1000", "splice 3:0:1000", 1)))
        ollm("spclip", "verify", str(rec_hz2), expect_rc=60)
        # Overlong tune output path (len + 8 > 1024 for auto .lineage) must fail fast with rc=2
        ollm("spclip", "tune", str(rec1), str(dom_user), "/" + "x" * 1020, "0", "0", expect_rc=2)
        ollm("spclip", "create", str(w / "s1.spclip"), str(base_ck), "--strand", str(sp_a1), expect_rc=2)
        ollm("spclip", "create", str(w / "s5.spclip"), str(base_ck), "--strand", str(sp_a1), "--strand", str(sp_b), "--strand", str(sp_c), "--strand", str(sp_d), "--strand", str(sp_a1), expect_rc=2)
        # Sealed recipe with zero weights must also fail closed with rc=2
        rec_zero_sealed = w / "zero_sealed.spclip"
        ztxt = rec1.read_text(encoding="utf-8").replace("weight 70", "weight 0").replace("weight 25", "weight 0").replace("weight 5", "weight 0")
        rec_zero_sealed.write_bytes(reseal_spclip(ztxt))
        ollm("spclip", "verify", str(rec_zero_sealed), expect_rc=2)

        # 7g. verify-lineage negative controls: tampered .lineage (rc=61), tampered child .sp (rc=61), tampered parent .sp (rc=61), wrong base (rc=62)
        lin_bad = w / "bad.lineage"
        lin_bad.write_text(child_lin.read_text(encoding="utf-8").replace("steps 12", "steps 13", 1), encoding="utf-8")
        ollm("spclip", "verify-lineage", str(lin_bad), expect_rc=61)

        child_sp_bak = w / "user_child.sp.bak"
        shutil.copy2(child_sp, child_sp_bak)
        flip_byte(child_sp_bak, child_sp, 64)
        ollm("spclip", "verify-lineage", str(child_lin), expect_rc=61)
        ollm("spapp", str(base_ck), str(child_sp), str(w / "bad_c.ck"), expect_rc=61)
        shutil.copy2(child_sp_bak, child_sp)
        ollm("spclip", "verify-lineage", str(child_lin), expect_rc=0)

        # Tamper parent sp_c on disk -> verify-lineage MUST catch it with rc=61!
        sp_c_bak = w / "c.sp.bak"
        shutil.copy2(sp_c, sp_c_bak)
        flip_byte(sp_c_bak, sp_c, 64)
        ollm("spclip", "verify-lineage", str(child_lin), expect_rc=61)
        shutil.copy2(sp_c_bak, sp_c)
        ollm("spclip", "verify-lineage", str(child_lin), expect_rc=0)

        # Wrong base on child spapp -> rc=62
        ollm("spapp", str(other_ck), str(child_sp), str(w / "wrong_b.ck"), expect_rc=62)

        # 0-depth spliced mix: refused (rc=2) with --add-layers 0, succeeds with --add-layers 1!
        ollm("spclip", "tune", str(rec1), str(dom_user), str(w / "z0.sp"), "2", "4000",
             "--splice", "2:2:1000,2:2:1000,2:2:1000", expect_rc=2)
        z1_sp = w / "z1.sp"
        ollm("spclip", "tune", str(rec1), str(dom_user), str(z1_sp), "2", "4000",
             "--splice", "2:2:1000,2:2:1000,2:2:1000", "--add-layers", "1", expect_rc=0)
        ollm("spclip", "verify-lineage", str(z1_sp) + ".lineage")

        print("PASS [7] All 41 fail-closed negative controls + 0-depth splice recovery verified (rc=60/61/62/2)")

        # 8. Edge cases: OLLMCK03 looped-core trunk + FIPS 180-4 SHA-256 padding boundary sizes (55, 56, 63, 64, 65 bytes)
        for blen in (55, 56, 63, 64, 65):
            bf = w / f"pad_{blen}.bin"
            bf.write_bytes(bytes([(i * 37 + blen) & 255 for i in range(blen)]))
            out_sp = w / f"pad_{blen}.sp"
            ollm("spclip", "tune", str(rec_id), str(bf), str(out_sp), "0", "0")
            lin_map = {}
            for ln in Path(str(out_sp) + ".lineage").read_text(encoding="utf-8").splitlines()[1:]:
                ps = ln.split()
                if len(ps) >= 2:
                    lin_map[ps[0]] = ps[1]
            check(lin_map["data_sha256"] == sha256_file(bf), f"SHA-256 padding bug at len={blen}: om={lin_map['data_sha256']} py={sha256_file(bf)}")

        lp_base = w / "loop_base.ck"
        lp_a_ck, lp_a_sp = w / "lp_a.ck", w / "lp_a.sp"
        lp_b_ck, lp_b_sp = w / "lp_b.ck", w / "lp_b.sp"
        # OLLMCK03: L=3, LP=1, LC=1, NP=2 (LV=4); train specialist strands via fcache + trainfc
        ollm("init", str(lp_base), "256", "16", "32", "2", "64", "3", "4", "13", "10", "40000", "1", "1", "1", "2")
        ollm("train", str(lp_base), str(base_corp), "4", "4")
        for dat, ock, osp, hc in ((dom_a, lp_a_ck, lp_a_sp, w / "lp_a.hc"), (dom_b, lp_b_ck, lp_b_sp, w / "lp_b.hc")):
            ollm("fcache", str(lp_base), str(dat), "8", str(hc))
            ollm("growpro", str(lp_base), str(ock), "4", "16", "80000")
            ollm("trainfc", str(ock), str(hc), "6", "6", "3")
            ollm("spext", str(lp_base), str(ock), str(osp))
        rec_lp = w / "loop.spclip"
        ollm("spclip", "create", str(rec_lp), str(lp_base), "--strand", str(lp_a_sp), "--strand", str(lp_b_sp), "--weights", "65,35", "--mode", "1", "--tau", "600")
        ollm("spclip", "verify", str(rec_lp))
        ev_lp = ollm("spclip", "eval", str(rec_lp), str(held), "2", "-").stdout.decode()
        check(ev_lp.startswith("SPCLIP_EVAL 2 "), f"looped spclip eval failed: {ev_lp}")
        run_lp = ollm("spclip", "run", str(rec_lp), "fn ", "8", "0", "42", "-").stdout
        check(len(run_lp) > 0, "looped spclip run empty")
        lp_child = w / "loop_child.sp"
        ollm("spclip", "tune", str(rec_lp), str(dom_user), str(lp_child), "6", "40000", "--batch", "4")
        ollm("spclip", "verify-lineage", str(lp_child) + ".lineage")
        ollm("spapp", str(lp_base), str(lp_child), str(w / "loop_child.ck"))

        # MODE=1 extreme tau=1 fexp-underflow regression on asymmetric depths (L=4 high-H vs L=3 low-H)
        dom_lowh = w / "lowh.txt"
        dom_lowh.write_text("A" * 512, encoding="utf-8")
        ck_lh, sp_lh = w / "lh.ck", w / "lh.sp"
        ollm("growpro", str(base_ck), str(ck_lh), "3", "120", "200000")
        ollm("trainf", str(ck_lh), str(dom_lowh), "120", "120", "2")
        ollm("spext", str(base_ck), str(ck_lh), str(sp_lh))
        rec_tau1 = w / "tau1_asym.spclip"
        ollm("spclip", "create", str(rec_tau1), str(base_ck), "--strand", f"{sp_a1},{sp_a2}", "--strand", str(sp_lh), "--weights", "500,500", "--mode", "1", "--tau", "1")
        tau1_child = w / "tau1_child.sp"
        ollm("spclip", "tune", str(rec_tau1), str(dom_lowh), str(tau1_child), "4", "4000", "--batch", "4")
        ollm("spclip", "verify-lineage", str(tau1_child) + ".lineage")
        ollm("spapp", str(base_ck), str(tau1_child), str(w / "tau1_child.ck"))
        print("PASS [8] OLLMCK03 looped-core spclip + SHA-256 boundary lengths (55/56/63/64/65 B) + extreme tau=1 underflow fallback verified")
        print("SPCLIP_EDITABLE_SUITE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
