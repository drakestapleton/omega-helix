#!/usr/bin/env python3
"""Omega Helix Studio ANSI CLI: studio, doctor, catalog, setup, forge, sample.

Python 3 standard library only — zero external dependencies, zero telemetry.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any

STUDIO_DIR = Path(__file__).resolve().parent
if str(STUDIO_DIR) not in sys.path:
    sys.path.insert(0, str(STUDIO_DIR))

import catalog as studio_catalog  # noqa: E402
import chain_runner as runner  # noqa: E402
import server as studio_server  # noqa: E402
import ssh_remote  # noqa: E402


# --------------------------------------------------------------------------- ANSI styling

def _use_color() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, text: str) -> str:
    if not _use_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def cyan(t: str) -> str: return _c("38;5;51", t)
def violet(t: str) -> str: return _c("38;5;141", t)
def emerald(t: str) -> str: return _c("38;5;49", t)
def amber(t: str) -> str: return _c("38;5;220", t)
def dim(t: str) -> str: return _c("2", t)
def bold(t: str) -> str: return _c("1", t)
def red(t: str) -> str: return _c("38;5;203", t)


def print_banner(subtitle: str = "Sovereign Pure-Omega Model Forge") -> None:
    s1 = cyan("═══╗")
    s2 = violet("╔═══")
    s3 = emerald("═══╗")
    s4 = amber("╔═══")
    print(f"  {s1}{s2}{s3}{s4}  {bold('OMEGA HELIX STUDIO')}")
    print(f"  {cyan('║')}  {violet('╚══')} {emerald('══╝')}  {amber('║')}  {dim(subtitle)}")
    print()


# --------------------------------------------------------------------------- commands


def cmd_doctor(args: argparse.Namespace) -> int:
    info = runner.detect_system(bin_dir=args.bin_dir, runs_dir=Path(args.workspace_dir) if args.workspace_dir else None)
    cat = studio_catalog.load_catalog()
    info["catalog_count"] = cat["count"]
    if args.json:
        print(json.dumps(info, indent=2))
        return 0

    print_banner("System Diagnostics & Pure-Omega Toolchain Check")
    print(f"  {cyan('Platform')}   : {info['os']} {info['arch']} · Python {info['python']}")
    print(f"  {violet('Compiler')}   : {emerald(info['omc']) if info['omc'] else amber('omc not found (install via brew or interplane.io/omega)')}")
    print(f"  {emerald('GB10 GPU')}   : {emerald('detected (/dev/nvidiactl)') if info['has_gb10'] else dim('not detected (CPU mode ready)')}")
    print(f"  {amber('OpenWALDO')}  : {cat['count']} indexed corpora · waldo CLI: {'installed' if info['has_waldo_cli'] else 'not installed (local/preset mode ready)'}")
    print(f"  {dim('Runs Dir')}   : {info['runs_dir']}")
    print(f"  {dim('Bin Dir')}    : {info['bin_dir']}")
    print()
    print(f"  {bold('Pure-Omega Binaries:')}")
    for name, b in info.get("binaries_detail", {}).items():
        if b["ready"]:
            kb = b["bytes"] / 1024.0
            print(f"    {emerald('✓')} {name:<18} {dim(f'{kb:7.1f} KB')}  {b['path']}")
        else:
            print(f"    {amber('○')} {name:<18} {dim('will build on demand when needed')}")
    print()
    return 0


def cmd_catalog(args: argparse.Namespace) -> int:
    if args.sync:
        sync_res = studio_catalog.sync_catalog()
        if not args.json:
            print(f"{emerald('✓')} Synced catalog: {sync_res['total']} corpora (sources: {', '.join(sync_res['synced_from'])})")

    res = studio_catalog.search_catalog(
        q=args.search or "",
        category=args.category or "",
        license=args.license or "",
        prog=args.prog or "",
    )
    if args.json:
        print(json.dumps(res, indent=2))
        return 0

    print_banner(f"OpenWALDO Verified Dataset Catalog ({res['count']}/{res['total']} corpora)")
    rows = res["corpora"][: args.limit]
    if not rows:
        print(amber("  No corpora matched your filters."))
        return 0

    for c in rows:
        cat_tag = violet(f"[{c['category']}]")
        lic_tag = emerald(c["license_summary"])
        progs = f" · {cyan(', '.join(c['programming_languages'][:4]))}" if c.get("programming_languages") else ""
        print(f"  {bold(c['path'])} {cat_tag}{progs}")
        if c.get("description"):
            desc = c["description"] if len(c["description"]) <= 96 else c["description"][:93] + "..."
            print(f"    {dim(desc)}")
        print(f"    {dim('SPDX:')} {lic_tag}")
    if res["count"] > len(rows):
        print(dim(f"\n  ... showing {len(rows)} of {res['count']} results (use --limit to view more)."))
    return 0


def _run_studio_once(args: argparse.Namespace) -> int:
    bind_host = getattr(args, "bind", None) or args.host
    port = int(args.port)
    if getattr(args, "reclaim_stale_lock", False):
        rec = ssh_remote.reclaim_stale_lock(port, yes=bool(getattr(args, "yes", False)))
        if getattr(args, "json", False):
            print(json.dumps(rec, indent=2))
        else:
            print(f"Studio lock ({rec['path']}): {rec['status']}")
        if rec["status"] == "refused":
            return 1
        if rec["status"] == "eligible" and not getattr(args, "yes", False):
            return 2
        return 0

    lock = ssh_remote.InstanceLock(port)
    try:
        lock.acquire()
    except RuntimeError as exc:
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print(red(f"ERROR: {exc}"), file=sys.stderr)
        return 3

    pol = ssh_remote.StudioAccessPolicy(
        allow_public=bool(getattr(args, "allow_public", False)),
        refuse_public=bool(getattr(args, "refuse_public", False)),
        require_pairing=bool(getattr(args, "require_pairing", False)),
        no_pairing=bool(getattr(args, "no_pairing", False)),
        pairing_address=getattr(args, "pairing_address", None),
    )
    try:
        srv = studio_server.create_server(
            host=bind_host,
            port=port,
            workspace_dir=args.workspace_dir,
            bin_dir=args.bin_dir,
            access_policy=pol,
            instance_lock=lock,
        )
    except Exception:
        lock.release()
        raise

    host, bound_port = srv.server_address[:2]
    adv = getattr(args, "pairing_address", None) or f"{host}:{bound_port}"
    tok = pol.pairing_code
    url = f"http://{host}:{bound_port}/"
    pair_url = f"http://{adv}/?pair={tok}" if tok else f"http://{adv}/"
    headless = bool(getattr(args, "headless", False))

    if getattr(args, "json", False):
        print(json.dumps({
            "ok": True,
            "bind": f"{host}:{bound_port}",
            "port": bound_port,
            "pairing_address": adv,
            "pairing_code": tok,
            "url": url,
            "pairing_url": pair_url,
            "ssh_forward_cmd": f"ssh -N -L 18421:127.0.0.1:{bound_port} {os.environ.get('USER', 'user')}@{os.uname().nodename}",
        }), flush=True)
    else:
        print_banner("Sovereign Studio Server — Zero Telemetry, 100% Pure-Omega")
        print(f"  {emerald('●')} Studio UI   : {cyan(url)}")
        if adv != f"{host}:{bound_port}" or tok:
            print(f"  {cyan('●')} Pairing URL : {cyan(pair_url)}")
        print(f"  {violet('●')} Runs Dir    : {dim(str(srv.manager.runs_dir))}")
        print(f"  {amber('●')} SSH Forward : {dim(f'ssh -N -L 18421:127.0.0.1:{bound_port} <host>')}")
        print(f"  {amber('●')} Stop Server : {dim('Ctrl-C')}")
        print()

    if not headless and not args.no_open and sys.stdout.isatty():
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        srv.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        if not getattr(args, "json", False):
            print(f"\n  {dim('Omega Helix Studio stopped.')}")
    finally:
        srv.server_close()
    return 0


def cmd_studio(args: argparse.Namespace) -> int:
    if not getattr(args, "supervise", False):
        return _run_studio_once(args)

    # Orca-style foreground supervisor: bounded restart backoff (max 6 failures in 15m)
    fail_times: list[float] = []
    backoff = 1.0
    while True:
        try:
            rc = _run_studio_once(args)
            if rc in (0, 3):
                return rc
        except KeyboardInterrupt:
            return 0
        except Exception as exc:
            print(red(f"Studio error: {exc}"), file=sys.stderr)
        now = time.monotonic()
        fail_times = [t for t in fail_times if now - t < 900.0]
        fail_times.append(now)
        if len(fail_times) >= 6:
            print(red("Studio supervisor: 6 failures in 15m, exiting."), file=sys.stderr)
            return 0
        time.sleep(backoff)
        backoff = min(backoff * 2.0, 30.0)


def cmd_host(args: argparse.Namespace) -> int:
    act = args.host_action
    if act == "list":
        hosts = ssh_remote.discover_ssh_hosts()
        if getattr(args, "json", False):
            print(json.dumps({"ok": True, "hosts": hosts}, indent=2))
            return 0
        print_banner("Known SSH Hosts (~/.ssh/config + saved)")
        if not hosts:
            print(dim("  No SSH hosts found. Add one with: omega-helix host add <name> <target>"))
            return 0
        for h in hosts:
            print(f"  {emerald('●')} {bold(h['name']):<18} {cyan(h.get('hostname', h['target']))} ({h['source']})")
        return 0
    if act == "add":
        entry = ssh_remote.add_saved_host(args.name, args.target, port=args.ssh_port, label=args.label or "")
        if getattr(args, "json", False):
            print(json.dumps({"ok": True, "host": entry}, indent=2))
        else:
            print(f"{emerald('✓')} Saved host {bold(entry['name'])} -> {entry['target']}")
        return 0
    if act == "rm":
        ok = ssh_remote.remove_saved_host(args.name)
        if getattr(args, "json", False):
            print(json.dumps({"ok": ok, "name": args.name}))
        else:
            print(f"{emerald('✓') if ok else amber('○')} host {args.name}: {'removed' if ok else 'not found'}")
        return 0 if ok else 1
    if act == "probe":
        res = ssh_remote.probe_ssh_host(args.target, ssh_port=args.ssh_port, remote_port=args.remote_port)
        if getattr(args, "json", False):
            print(json.dumps(res, indent=2))
        elif res.get("ok"):
            gpu = emerald("GB10") if res.get("has_gb10") else dim("CPU")
            st = emerald("running") if res.get("studio_running") else dim("stopped")
            print(f"{emerald('✓')} {bold(res['target'])} ({res.get('os')} {res.get('arch')}) GPU={gpu} Studio={st}")
        else:
            print(red(f"✗ {args.target}: {res.get('error')}"), file=sys.stderr)
        return 0 if res.get("ok") else 1
    return 2


def cmd_remote(args: argparse.Namespace) -> int:
    act = args.remote_action
    if act == "list":
        rows = ssh_remote.load_remotes()
        if getattr(args, "json", False):
            print(json.dumps({"ok": True, "remotes": rows}, indent=2))
            return 0
        print_banner("Saved Remote Studio Environments")
        if not rows:
            print(dim("  None saved. Connect with: omega-helix remote connect <ssh-host>"))
            return 0
        for r in rows:
            gpu = emerald("GB10") if r.get("has_gb10") else dim("CPU")
            print(f"  {cyan('●')} {bold(r['id']):<24} {r.get('open_url', '')}  {gpu}")
        return 0
    if act == "add":
        p = ssh_remote.parse_pairing_input(args.pairing)
        entry = ssh_remote.save_remote({
            "id": args.name or p["host"],
            "host": p["host"],
            "open_url": p["open_url"],
            "url": p["url"],
            "source": "paired",
        })
        if getattr(args, "json", False):
            print(json.dumps({"ok": True, "remote": entry}, indent=2))
        else:
            print(f"{emerald('✓')} Added remote {bold(entry['id'])} -> {entry['open_url']}")
        return 0
    if act == "rm":
        ok = ssh_remote.remove_remote(args.id)
        if getattr(args, "json", False):
            print(json.dumps({"ok": ok, "id": args.id}))
        else:
            print(f"{emerald('✓') if ok else amber('○')} remote {args.id}: {'removed' if ok else 'not found'}")
        return 0 if ok else 1
    if act == "connect":
        mgr = ssh_remote.TunnelManager()
        tun = mgr.connect(
            args.target,
            remote_port=args.remote_port,
            local_port=args.local_port,
            ssh_port=args.ssh_port,
            bootstrap=not args.no_bootstrap,
        )
        if getattr(args, "json", False):
            print(json.dumps({"ok": True, "tunnel": tun}, indent=2), flush=True)
        else:
            print_banner(f"SSH Tunnel Active -> {tun['host']}")
            print(f"  {emerald('●')} Local URL  : {cyan(tun['open_url'])}")
            print(f"  {violet('●')} Remote     : {tun['host']}:{tun['remote_port']}")
            print(f"  {amber('●')} Stop       : {dim('Ctrl-C')}")
            print()
        if not args.no_open and sys.stdout.isatty():
            threading.Timer(0.3, lambda: webbrowser.open(tun["open_url"])).start()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            mgr.close_all()
        return 0
    return 2


def _prompt_choice(label: str, default: str) -> str:
    try:
        ans = input(f"  {cyan('▸')} {label} [{amber(default)}]: ").strip()
    except EOFError:
        return default
    return ans or default


def cmd_setup(args: argparse.Namespace) -> int:
    print_banner("Interactive 4-Strand Helix Chain Wizard")
    name = _prompt_choice("Chain name", "my-sovereign-helix")
    mode = _prompt_choice("Mode (dna | universal | hybrid)", "hybrid").lower()
    if mode not in ("dna", "universal", "hybrid"):
        mode = "hybrid"

    if mode == "universal":
        arch = _prompt_choice("Architecture (qwen3 | llama | mistral | gemma | qwen2 | gpt | gelu)", "qwen3").lower()
        tokenizer = _prompt_choice("Tokenizer (byte | bpe | u32)", "byte").lower()
    else:
        arch = "gpt"
        tokenizer = _prompt_choice("Tokenizer (byte | bpe)", "byte").lower()

    V = 256 if tokenizer == "byte" else (70000 if tokenizer == "u32" else int(_prompt_choice("BPE Vocab size (280..4096)", "300")))
    D = int(_prompt_choice("Hidden dim D", "32"))
    H = int(_prompt_choice("Attention heads H", "4"))
    L = int(_prompt_choice("Base layers L", "2"))
    T = int(_prompt_choice("Context window T", "32"))
    base_steps = int(_prompt_choice("Base training steps", "20"))
    ds_val = _prompt_choice("Base dataset (omega-starter | omega-docs | omega-native | /path/to/data)", "omega-starter")
    ds_kind = "preset" if ds_val in ("omega-starter", "omega-docs", "omega-native") else "local"

    strands: list[dict[str, Any]] = []
    if mode in ("dna", "hybrid"):
        n_strands = int(_prompt_choice("Number of Merkle specialist strands (0..4)", "2"))
        defaults = [("Code", "omega-native"), ("Docs", "omega-docs"), ("Reasoning", "omega-starter"), ("Custom", "omega-starter")]
        for i in range(max(0, min(4, n_strands))):
            dn, dds = defaults[i]
            print(f"\n  {violet(f'Strand {i + 1}')}")
            sname = _prompt_choice("  Strand name", dn)
            sds = _prompt_choice("  Strand dataset (preset or path)", dds)
            sadd = int(_prompt_choice("  Added residual layers", "1"))
            ssteps = int(_prompt_choice("  Frozen-core steps", "15"))
            strands.append({
                "id": f"strand-{chr(97 + i)}",
                "name": sname,
                "color": runner.STRAND_COLORS[i % 4],
                "add_layers": sadd,
                "growth": "growpro",
                "use_fcachem": True,
                "steps": ssteps,
                "lr_micro": 2000,
                "weight": 500,
                "dataset": {"kind": "preset" if sds in ("omega-starter", "omega-docs", "omega-native") else "local", "value": sds},
            })

    export_hf = _prompt_choice("Export HF Safetensors? (y/n)", "y").lower().startswith("y")
    export_gguf = _prompt_choice("Export GGUF v3? (y/n)", "y").lower().startswith("y")

    spec: dict[str, Any] = {
        "name": name,
        "mode": mode,
        "arch": arch,
        "tokenizer": tokenizer,
        "V": V,
        "T": T,
        "D": D,
        "H": H,
        "H_KV": max(1, H // 2) if (mode == "universal" and H % 2 == 0) else H,
        "F": max(32, D * 2),
        "L": L,
        "B": 2,
        "base_steps": base_steps,
        "lr_micro": 3000,
        "base_dataset": {"kind": ds_kind, "value": ds_val},
        "strands": strands,
        "spclip_mode": 1,
        "export_hf": export_hf,
        "hf_dtype": "bf16",
        "export_gguf": export_gguf,
    }

    est = runner.estimate_chain(spec)
    print()
    if not est["valid"]:
        print(red(f"  ✗ Invalid geometry: {est['error']}"))
        return 1

    p_str = f"{est['params']:,}"
    print(f"  {emerald('✓')} Estimated Parameters : {bold(p_str)} ({est['tok_width']} tokens, ~{est['est_mb']} MB)")
    print(f"  {violet('◆')} Merkle DAG Nodes     : {' → '.join(n['label'] for n in est['dag']['nodes'])}")

    out_path = Path(args.out or "helix-spec.json").resolve()
    out_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"  {cyan('✓')} Saved spec to        : {out_path}")

    launch = _prompt_choice("Forge this chain now? (y/n)", "n").lower().startswith("y")
    if launch:
        ns = argparse.Namespace(
            spec=str(out_path), name=None, mode=None, arch=None, tokenizer=None,
            V=None, T=None, D=None, H=None, H_KV=None, F=None, L=None, B=None,
            steps=None, lr_micro=None, dataset=None, strand=[], export_hf=False,
            hf_dtype=None, export_gguf=False, workspace_dir=args.workspace_dir,
            bin_dir=args.bin_dir, json=False,
        )
        return cmd_forge(ns)
    print(f"\n  Run anytime with: {bold(f'omega-helix forge --spec {out_path}')}\n")
    return 0


def cmd_forge(args: argparse.Namespace) -> int:
    spec: dict[str, Any] = {}
    if args.spec:
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    for src_k, dst_k in (
        ("name", "name"), ("mode", "mode"), ("arch", "arch"), ("tokenizer", "tokenizer"),
        ("V", "V"), ("T", "T"), ("D", "D"), ("H", "H"), ("H_KV", "H_KV"),
        ("F", "F"), ("L", "L"), ("B", "B"), ("steps", "base_steps"),
        ("lr_micro", "lr_micro"), ("hf_dtype", "hf_dtype"),
    ):
        val = getattr(args, src_k, None)
        if val is not None:
            spec[dst_k] = val
    if args.dataset:
        kind = "preset" if args.dataset in ("omega-starter", "omega-docs", "omega-native") else "local"
        spec["base_dataset"] = {"kind": kind, "value": args.dataset}
    if args.export_hf:
        spec["export_hf"] = True
    if args.export_gguf:
        spec["export_gguf"] = True
    if args.strand:
        strands = list(spec.get("strands", []))
        for idx, item in enumerate(args.strand):
            parts = item.split(":")
            sname = parts[0]
            sds = parts[1] if len(parts) > 1 and parts[1] else "omega-starter"
            sadd = int(parts[2]) if len(parts) > 2 and parts[2] else 1
            ssteps = int(parts[3]) if len(parts) > 3 and parts[3] else 15
            strands.append({
                "id": f"strand-{chr(97 + (len(strands) % 26))}",
                "name": sname,
                "color": runner.STRAND_COLORS[len(strands) % 4],
                "add_layers": sadd,
                "growth": "growpro",
                "use_fcachem": True,
                "steps": ssteps,
                "lr_micro": 2000,
                "weight": 500,
                "dataset": {"kind": "preset" if sds in ("omega-starter", "omega-docs", "omega-native") else "local", "value": sds},
            })
        spec["strands"] = strands
        spec.setdefault("mode", "hybrid" if (spec.get("export_hf") or spec.get("export_gguf")) else "dna")

    est = runner.estimate_chain(spec)
    if not est["valid"]:
        print(red(f"ERROR: {est['error']}"), file=sys.stderr)
        return 1

    mgr = runner.ChainManager(runs_dir=args.workspace_dir, bin_dir=args.bin_dir)
    if not args.json:
        print_banner(f"Forging {est['spec']['name']} ({est['mode'].upper()} · {est['params']:,} params)")

    run = mgr.start_chain(est["spec"], wait=False)
    for ev in run.iter_events():
        if ev is None:
            continue
        if args.json:
            continue
        k = ev["kind"]
        if k == "stage":
            print(f"  {cyan('▶')} {bold(ev.get('label', ev.get('stage')))}")
        elif k == "metric":
            st = ev.get("step", 0)
            loss = ev.get("loss", 0.0)
            bpb = ev.get("bpb", 0.0)
            tag = violet(f"[{ev.get('strand_id')}] ") if ev.get("strand_id") else ""
            print(f"    {tag}step {st:<4}  loss={emerald(f'{loss:.4f}')} nats  bpb={amber(f'{bpb:.4f}')}")
        elif k == "artifact":
            print(f"    {emerald('✓')} {ev.get('name'):<28} sha256={dim(str(ev.get('sha256', ''))[:16])}… ({ev.get('bytes', 0):,} B)")
        elif k == "error":
            print(f"  {red('✗')} {ev.get('error')}", file=sys.stderr)

    data = run.to_dict()
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print()
        if run.status == "done":
            print(f"  {emerald('★ FORGE COMPLETE')}  run_id={bold(run.run_id)}")
            print(f"  Sample with: {cyan(f'omega-helix sample {run.run_id} --prompt \"Omega \"')}\n")
    return 0 if run.status == "done" else 1


def cmd_sample(args: argparse.Namespace) -> int:
    mgr = runner.ChainManager(runs_dir=args.workspace_dir, bin_dir=args.bin_dir)
    weights: dict[str, int] | list[int] | None = None
    if args.weights:
        if "=" in args.weights:
            weights = {}
            for pair in args.weights.split(","):
                k, _, v = pair.partition("=")
                if k.strip() and v.strip():
                    weights[k.strip()] = int(v.strip())
        else:
            weights = [int(x.strip()) for x in args.weights.split(",") if x.strip()]

    res = mgr.sample_chain(
        run_id=args.run,
        prompt=args.prompt,
        n_tokens=args.n_tokens,
        temp_milli=args.temp_milli,
        seed=args.seed,
        spclip_mode=args.spclip_mode,
        strand_weights=weights,
        adapt_thresh_micro=args.adapt_thresh_micro,
        msg_gain_micro=args.msg_gain_micro,
    )
    if args.json:
        print(json.dumps(res, indent=2))
        return 0
    return 0 if res["text"] is not None else 1


def _load_distill():
    tools_dir = STUDIO_DIR.parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from distill import harvest as d_harvest
    from distill import pipeline as d_pipeline
    from distill import teachers as d_teachers
    from distill import verifiers as d_verifiers
    return d_harvest, d_pipeline, d_teachers, d_verifiers


def cmd_distill_probe(args: argparse.Namespace) -> int:
    _, _, d_teachers, _ = _load_distill()
    eps = d_teachers.probe_local_teachers(timeout=float(args.timeout))
    if args.json:
        print(json.dumps({"ok": True, "endpoints": eps}, indent=2))
        return 0
    print_banner("Local Open-Source Teacher Endpoint Probe")
    for ep in eps:
        st = emerald("ONLINE ") if ep["reachable"] else dim("offline")
        mods = ", ".join(ep["models"][:4]) if ep["models"] else (ep.get("error") or "no models")
        lat_s = f"{float(ep.get('latency_ms', 0.0)):.1f}ms"
        print(f"  {st} {bold(ep['name']):<14} {cyan(ep['base_url']):<30} {dim(lat_s)}  {mods}")
    print()
    return 0


def cmd_distill_harvest(args: argparse.Namespace) -> int:
    d_harvest, _, _, d_verifiers = _load_distill()
    verifiers = [v.strip() for v in args.verify.split(",") if v.strip()] if args.verify else ["nonempty"]
    harv = d_harvest.harvest_paths(args.input, source_tag=args.corpus_name, license_id=args.license)
    vpipe = d_verifiers.VerifierPipeline(verifiers)
    vres = vpipe.filter_records(harv["records"])
    export_meta = None
    if args.out and vres["accepted"]:
        export_meta = d_harvest.write_openwaldo_export(
            vres["accepted"],
            args.out,
            corpus_name=args.corpus_name,
            license_id=args.license,
        )
    if args.json:
        print(json.dumps({"ok": True, "harvest_stats": harv["stats"], "verify_stats": vres["stats"], "export": export_meta}, indent=2))
        return 0
    print_banner("Trace Harvest & Verifier Gate")
    print(f"  Scanned files  : {harv['stats']['files_scanned']}")
    print(f"  Unique records : {harv['stats']['unique_records']} (dups skipped: {harv['stats']['duplicates_skipped']})")
    print(f"  Accepted       : {emerald(str(vres['stats']['passed']))} / {vres['stats']['total']} ({vres['stats']['pass_rate']*100:.1f}%)")
    if export_meta:
        print(f"  OpenWALDO BOM  : {cyan(export_meta['export_json'])} (sha256={dim(export_meta['export_json_sha256'][:16])}…)")
    print()
    return 0 if vres["stats"]["passed"] > 0 else 1


def cmd_distill_run(args: argparse.Namespace) -> int:
    _, d_pipeline, _, _ = _load_distill()
    spec: dict[str, Any] = {}
    if args.spec:
        spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    for k in ("mode", "arch", "V", "T", "D", "H", "H_KV", "F", "L", "B", "K", "steps", "alpha", "temperature", "teacher_ckpt", "dk_path", "tokens_bin", "export_hf", "export_gguf"):
        v = getattr(args, k, None)
        if v is not None and v is not False:
            spec[k] = v
    if args.verify:
        spec["verifiers"] = [x.strip() for x in args.verify.split(",") if x.strip()]
    if args.trace:
        spec["trace_paths"] = list(spec.get("trace_paths") or []) + list(args.trace)
    if args.prompt:
        spec["prompts"] = list(spec.get("prompts") or []) + list(args.prompt)
    if args.teacher_url:
        tlist = list(spec.get("teachers") or [])
        tlist.append({"kind": "openai_compat", "base_url": args.teacher_url, "model": args.teacher_model or "default"})
        spec["teachers"] = tlist

    est = d_pipeline.estimate_distill(spec)
    if not est["valid"]:
        print(red(f"ERROR: {est['error']}"), file=sys.stderr)
        return 1

    mgr = d_pipeline.DistillManager(runs_dir=args.workspace_dir, bin_dir=args.bin_dir)
    if not args.json:
        print_banner(f"Distilling {est['spec']['name']} ({est['spec']['mode']} · {est['spec']['arch']})")
    run = mgr.start_run(est["spec"], wait=False)
    for ev in run.iter_events():
        if ev is None or args.json:
            continue
        k = ev["kind"]
        if k == "stage":
            print(f"  {cyan('▶')} {bold(ev.get('label', ev.get('stage')))}")
        elif k == "dcheck":
            print(f"    {emerald('✓')} dcheck {ev.get('samples')} params mxrel={ev.get('mxrel_milli'):.6f} retries={ev.get('retries')}")
        elif k == "metric":
            st = ev.get("step", 0)
            lv = f"{float(ev.get('loss', 0.0)):.4f}"
            cv = f"{float(ev.get('ce_loss', 0.0)):.4f}"
            kv = f"{float(ev.get('kl_loss', 0.0)):.4f}"
            print(f"    step {st:<4} loss={emerald(lv)} ce={amber(cv)} kl={violet(kv)}")
        elif k == "error":
            print(f"  {red('✗')} {ev.get('error')}", file=sys.stderr)
    if args.json:
        print(json.dumps(run.to_dict(), indent=2))
    return 0 if run.status == "done" else 1


def cmd_distill_status(args: argparse.Namespace) -> int:
    _, d_pipeline, _, _ = _load_distill()
    mgr = d_pipeline.DistillManager(runs_dir=args.workspace_dir, bin_dir=args.bin_dir)
    if args.run_id:
        r = mgr.get_run(args.run_id)
        if not r:
            print(red(f"Distill run not found: {args.run_id}"), file=sys.stderr)
            return 1
        print(json.dumps(r.to_dict(), indent=2))
        return 0
    runs = mgr.list_runs()
    print(json.dumps({"ok": True, "runs": runs}, indent=2))
    return 0


def _load_runtime():
    tools_dir = STUDIO_DIR.parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    import runtime as rt
    return rt


def cmd_runtime_gsamp(args: argparse.Namespace) -> int:
    bins = runner.ensure_binaries(["ollm_univ"], bin_dir=args.bin_dir)
    res = subprocess.run(
        [
            str(bins["ollm_univ"]), "gsamp",
            str(Path(args.ck_path).expanduser().resolve()),
            str(args.seed), str(args.n_tokens), str(args.temp_milli), str(args.mode),
        ],
        capture_output=True, text=True, check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"ollm-univ gsamp failed (rc={res.returncode}): {res.stderr.strip()}")
    g_raw = ""
    toks: list[int] = []
    for ln in res.stdout.splitlines():
        if ln.startswith("G "):
            g_raw = ln[2:]
        elif ln.startswith("S "):
            toks = [int(x) for x in ln[2:].split()]
    out = {"ok": True, "mode": args.mode, "text": g_raw, "tokens": toks}
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(g_raw)
    return 0


def cmd_runtime_guard(args: argparse.Namespace) -> int:
    rt = _load_runtime()
    bins = runner.ensure_binaries(["ollm"], bin_dir=args.bin_dir)
    strands: list[dict[str, Any]] = []
    for item in args.strand:
        w_s, _, p_s = item.partition(":")
        packs = [x.strip() for x in p_s.split(",") if x.strip()]
        strands.append({"weight_milli": int(w_s), "packs": packs})
    res = rt.run_firewall_decode(
        bins["ollm"], args.base_ck, strands,
        prompt=args.prompt, n_tokens=args.n_tokens, temp_milli=args.temp_milli,
        seed=args.seed, disagree_milli=args.disagree_milli, max_extra=args.max_extra, merges=args.merges,
    )
    chk = rt.verify_firewall_receipt(res)
    doc = {"ok": chk["verified"], "spguard": res, "verification": chk}
    print(json.dumps(doc, indent=2))
    return 0 if chk["verified"] else 1


def cmd_runtime_sleep(args: argparse.Namespace) -> int:
    rt = _load_runtime()
    bins = runner.ensure_binaries(["ollm"], bin_dir=args.bin_dir)
    vers = [x.strip() for x in args.verify.split(",") if x.strip()]
    res = rt.run_sleep_cycle(
        bins["ollm"], args.base_ck, args.interaction, args.held, args.out_dir,
        verifiers=vers, max_regression_nats=args.max_regression, steps=args.steps, date_tag=args.date_tag,
    )
    print(json.dumps(res, indent=2))
    return 0 if res.get("accepted") else 1


def cmd_runtime_green(args: argparse.Namespace) -> int:
    rt = _load_runtime()
    bins = runner.ensure_binaries(["ollm"], bin_dir=args.bin_dir)
    res = rt.run_compile_until_green(
        bins["ollm"], args.ck_path, args.prompt,
        verifier=args.verifier, max_attempts=args.max_attempts, out_dir=args.out_dir,
    )
    print(json.dumps(res, indent=2))
    return 0 if res.get("passed") else 1


def _load_commons() -> Any:
    tools_dir = str(STUDIO_DIR.parent)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import commons  # type: ignore
    return commons


def cmd_commons_verify(args: argparse.Namespace) -> int:
    cm = _load_commons()
    res = cm.verify_sovereign_trust(
        bin_dir=args.bin_dir,
        base_ck=args.base_ck,
        packs=args.pack,
        bom_path=args.bom,
    )
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        for chk in res["checks"]:
            badge = emerald("PASS") if chk["passed"] else red("FAIL")
            print(f"[{badge}] {chk['name']}: {chk['detail']}")
        print(f"Trust audit {'OK' if res['ok'] else 'FAILED'} in {res['elapsed_ms']} ms")
    return 0 if res.get("ok") else 1


def cmd_commons_pkg(args: argparse.Namespace) -> int:
    cm = _load_commons()
    action = args.pkg_action
    if action == "publish":
        res = cm.publish_pack(
            args.sp_path,
            args.name,
            registry_dir=args.registry,
            version=args.version,
            license_id=args.license,
            description=args.desc,
        )
    elif action == "search":
        res = cm.search_packs(args.registry, query=args.query, base_hash=args.base_hash)
    elif action == "install":
        res = cm.install_pack(args.name, args.env_dir, registry_dir=args.registry, expected_base_hash=args.base_hash)
    elif action == "revoke":
        res = cm.revoke_pack(args.name, args.env_dir, reason=args.reason)
    elif action == "diff":
        res = cm.diff_packs(args.sp_a, args.sp_b)
    else:
        raise ValueError(f"unknown pkg action: {action}")
    print(json.dumps(res, indent=2))
    return 0


def cmd_commons_memory(args: argparse.Namespace) -> int:
    cm = _load_commons()
    action = args.mem_action
    if action == "add":
        res = cm.MemoryStore(args.store_path).add(args.text, tags=args.tag)
    elif action == "list":
        res = cm.MemoryStore(args.store_path).list_memories()
    elif action == "export-enc":
        res = cm.export_encrypted(args.store_path, args.out_path, args.passphrase)
    elif action == "import-enc":
        res = cm.import_encrypted(args.enc_path, args.out_store_path, args.passphrase)
    elif action == "migrate":
        bins = runner.ensure_binaries(["ollm"], bin_dir=args.bin_dir)
        res = cm.migrate_brain_to_trunk(
            bins["ollm"],
            args.store_path,
            args.new_trunk_ck,
            args.out_dir,
            steps=args.steps,
            add_layers=args.add_layers,
            lr_micro=args.lr_micro,
        )
    else:
        raise ValueError(f"unknown memory action: {action}")
    print(json.dumps(res, indent=2))
    return 0


def cmd_commons_lease(args: argparse.Namespace) -> int:
    cm = _load_commons()
    action = args.lease_action
    if action == "issue":
        res = cm.issue_lease(
            args.lease_path,
            allowed_dirs=args.allowed_dir,
            allowed_cmds=args.allowed_cmd,
            allow_network=args.allow_network,
            ttl_seconds=args.ttl,
        )
    elif action == "run":
        cmd = list(args.cmd)
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        res = cm.execute_under_lease(
            args.lease_path,
            cmd,
            args.cwd,
            audit_log_path=args.audit,
            timeout=args.timeout,
        )
    elif action == "revoke":
        res = cm.revoke_lease(args.lease_path)
    else:
        raise ValueError(f"unknown lease action: {action}")
    print(json.dumps(res, indent=2))
    return 0


def cmd_commons_pack_usb(args: argparse.Namespace) -> int:
    cm = _load_commons()
    res = cm.pack_usb_bundle(
        args.out_dir,
        bin_dir=args.bin_dir,
        trunk_ck=args.trunk_ck,
        packs=args.pack,
        memory_store=args.memory,
    )
    print(json.dumps(res, indent=2))
    return 0


def cmd_commons_port_check(args: argparse.Namespace) -> int:
    cm = _load_commons()
    cand = args.bin or str(runner.ensure_binaries(["ollm"], bin_dir=args.bin_dir)["ollm"])
    res = cm.run_port_parity_check(cand, out_json=args.out_json)
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        for g in res["gates"]:
            badge = emerald("PASS") if g["passed"] else red("FAIL")
            print(f"[{badge}] {g['gate']} (rc={g['returncode']}) {g['stdout_tail']}")
        print(f"Port parity: {res['passed_count']}/{res['total_gates']} gates passed")
    return 0 if res.get("ok") else 1


def cmd_pillars(args: argparse.Namespace) -> int:
    import server as studio_server
    res = studio_server._run_eight_pillar_locks(
        Path(args.bin_dir) if args.bin_dir else None,
        None,
        steps=int(args.steps),
        lr_milli=int(args.lr_milli),
    )
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"OMG={res['omg_pass']} Arboris={res['arboris_pass']} Rev={res['rev_pass']} (supported={res['rev_supported']}, act_bytes={res['act_bytes']}, q32_bits={res['q32_inv_bits']})")
    return 0 if res.get("ok") else 1


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workspace-dir", default=argparse.SUPPRESS, help="Directory for forged runs")
    common.add_argument("--bin-dir", default=argparse.SUPPRESS, help="Directory containing pure-Omega binaries")

    p = argparse.ArgumentParser(
        prog="omega-helix",
        description="Omega Helix Studio — Zero-Dependency Pure-Omega Model Forge",
    )
    p.add_argument("--workspace-dir", default=None, help="Directory for forged runs")
    p.add_argument("--bin-dir", default=None, help="Directory containing pure-Omega binaries")
    sub = p.add_subparsers(dest="cmd")

    p_studio = sub.add_parser("studio", parents=[common], help="Launch local or headless Omega Helix Studio server")
    p_studio.add_argument("--host", default="127.0.0.1")
    p_studio.add_argument("--bind", default=None, help="Bind IP (e.g. 127.0.0.1, 0.0.0.0, Tailscale IP)")
    p_studio.add_argument("--port", type=int, default=18420)
    p_studio.add_argument("--pairing-address", dest="pairing_address", default=None, help="Advertised host:port for pairing links")
    p_studio.add_argument("--allow-public", dest="allow_public", action="store_true")
    p_studio.add_argument("--refuse-public", dest="refuse_public", action="store_true")
    p_studio.add_argument("--no-pairing", dest="no_pairing", action="store_true")
    p_studio.add_argument("--require-pairing", dest="require_pairing", action="store_true")
    p_studio.add_argument("--headless", action="store_true", help="Headless server mode (do not open browser)")
    p_studio.add_argument("--supervise", action="store_true", help="Restart on failure with bounded backoff")
    p_studio.add_argument("--reclaim-stale-lock", dest="reclaim_stale_lock", action="store_true")
    p_studio.add_argument("--yes", action="store_true")
    p_studio.add_argument("--json", action="store_true")
    p_studio.add_argument("--no-open", action="store_true", help="Do not open browser automatically")
    p_studio.set_defaults(func=cmd_studio)

    p_host = sub.add_parser("host", parents=[common], help="Manage & probe SSH target hosts (Orca-style)")
    host_sub = p_host.add_subparsers(dest="host_action", required=True)
    p_hl = host_sub.add_parser("list", parents=[common], help="List discovered & saved SSH hosts")
    p_hl.add_argument("--json", action="store_true")
    p_hl.set_defaults(func=cmd_host)
    p_ha = host_sub.add_parser("add", parents=[common], help="Save an SSH host target")
    p_ha.add_argument("name")
    p_ha.add_argument("target", nargs="?", default=None)
    p_ha.add_argument("--ssh-port", dest="ssh_port", type=int, default=22)
    p_ha.add_argument("--label", default="")
    p_ha.add_argument("--json", action="store_true")
    p_ha.set_defaults(func=cmd_host)
    p_hr = host_sub.add_parser("rm", parents=[common], help="Remove a saved SSH host target")
    p_hr.add_argument("name")
    p_hr.add_argument("--json", action="store_true")
    p_hr.set_defaults(func=cmd_host)
    p_hp = host_sub.add_parser("probe", parents=[common], help="Probe an SSH host for GB10 GPU & Studio")
    p_hp.add_argument("target")
    p_hp.add_argument("--ssh-port", dest="ssh_port", type=int, default=22)
    p_hp.add_argument("--remote-port", dest="remote_port", type=int, default=18420)
    p_hp.add_argument("--json", action="store_true")
    p_hp.set_defaults(func=cmd_host)

    p_rem = sub.add_parser("remote", parents=[common], help="Connect & manage remote Studio servers over SSH")
    rem_sub = p_rem.add_subparsers(dest="remote_action", required=True)
    p_rl = rem_sub.add_parser("list", parents=[common], help="List saved remote Studio environments")
    p_rl.add_argument("--json", action="store_true")
    p_rl.set_defaults(func=cmd_remote)
    p_ra = rem_sub.add_parser("add", parents=[common], help="Save a remote Studio from a pairing URL or host:port#token")
    p_ra.add_argument("pairing")
    p_ra.add_argument("--name", default=None)
    p_ra.add_argument("--json", action="store_true")
    p_ra.set_defaults(func=cmd_remote)
    p_rr = rem_sub.add_parser("rm", parents=[common], help="Remove a saved remote Studio")
    p_rr.add_argument("id")
    p_rr.add_argument("--json", action="store_true")
    p_rr.set_defaults(func=cmd_remote)
    p_rc = rem_sub.add_parser("connect", parents=[common], help="Bootstrap & tunnel to a remote Studio over SSH")
    p_rc.add_argument("target")
    p_rc.add_argument("--remote-port", dest="remote_port", type=int, default=18420)
    p_rc.add_argument("--local-port", dest="local_port", type=int, default=0)
    p_rc.add_argument("--ssh-port", dest="ssh_port", type=int, default=22)
    p_rc.add_argument("--no-bootstrap", dest="no_bootstrap", action="store_true")
    p_rc.add_argument("--no-open", dest="no_open", action="store_true")
    p_rc.add_argument("--json", action="store_true")
    p_rc.set_defaults(func=cmd_remote)

    p_doc = sub.add_parser("doctor", parents=[common], help="Inspect toolchain, binaries, GPU, and OpenWALDO catalog")
    p_doc.add_argument("--json", action="store_true")
    p_doc.set_defaults(func=cmd_doctor)

    p_cat = sub.add_parser("catalog", parents=[common], help="Search & sync the OpenWALDO verified dataset catalog")
    p_cat.add_argument("--search", "-q", default="")
    p_cat.add_argument("--category", "-c", default="")
    p_cat.add_argument("--license", "-l", default="")
    p_cat.add_argument("--prog", "-p", default="")
    p_cat.add_argument("--sync", action="store_true", help="Refresh catalog from openwaldo/waldo-index")
    p_cat.add_argument("--limit", type=int, default=20)
    p_cat.add_argument("--json", action="store_true")
    p_cat.set_defaults(func=cmd_catalog)

    p_setup = sub.add_parser("setup", parents=[common], help="Interactive 4-strand ANSI wizard to design a Helix chain")
    p_setup.add_argument("--out", default="helix-spec.json", help="Output spec JSON file")
    p_setup.set_defaults(func=cmd_setup)

    p_forge = sub.add_parser("forge", parents=[common], help="Forge a DNA, Universal, or Hybrid Helix chain from a spec or flags")
    p_forge.add_argument("--spec", default=None, help="Path to chain spec JSON")
    p_forge.add_argument("--name", default=None)
    p_forge.add_argument("--mode", choices=("dna", "universal", "hybrid"), default=None)
    p_forge.add_argument("--arch", default=None)
    p_forge.add_argument("--tokenizer", choices=("byte", "bpe", "u32"), default=None)
    p_forge.add_argument("--V", type=int, default=None)
    p_forge.add_argument("--T", type=int, default=None)
    p_forge.add_argument("--D", type=int, default=None)
    p_forge.add_argument("--H", type=int, default=None)
    p_forge.add_argument("--H-KV", dest="H_KV", type=int, default=None)
    p_forge.add_argument("--F", type=int, default=None)
    p_forge.add_argument("--L", type=int, default=None)
    p_forge.add_argument("--B", type=int, default=None)
    p_forge.add_argument("--steps", type=int, default=None)
    p_forge.add_argument("--lr-micro", dest="lr_micro", type=int, default=None)
    p_forge.add_argument("--dataset", default=None)
    p_forge.add_argument("--strand", action="append", default=[], help="Add strand NAME[:DATASET[:LAYERS[:STEPS]]]")
    p_forge.add_argument("--export-hf", dest="export_hf", action="store_true")
    p_forge.add_argument("--hf-dtype", dest="hf_dtype", choices=("f32", "bf16"), default=None)
    p_forge.add_argument("--export-gguf", dest="export_gguf", action="store_true")
    p_forge.add_argument("--json", action="store_true")
    p_forge.set_defaults(func=cmd_forge)

    p_samp = sub.add_parser("sample", parents=[common], help="Sample interactively from a forged Helix run")
    p_samp.add_argument("run", help="Run ID or path to run directory")
    p_samp.add_argument("--prompt", default="Omega ")
    p_samp.add_argument("--n-tokens", dest="n_tokens", type=int, default=48)
    p_samp.add_argument("--temp-milli", dest="temp_milli", type=int, default=800)
    p_samp.add_argument("--seed", type=int, default=42)
    p_samp.add_argument("--spclip-mode", dest="spclip_mode", type=int, default=1)
    p_samp.add_argument("--weights", default=None, help="Comma-separated weights (e.g. 700,300 or strand-a=700,strand-b=300)")
    p_samp.add_argument("--adapt-thresh-micro", dest="adapt_thresh_micro", type=int, default=0)
    p_samp.add_argument("--msg-gain-micro", dest="msg_gain_micro", type=int, default=0)
    p_samp.add_argument("--json", action="store_true")
    p_samp.set_defaults(func=cmd_sample)

    p_dist = sub.add_parser("distill", parents=[common], help="Sovereign Distillation Pipeline Network (probe | harvest | run | status)")
    dsub = p_dist.add_subparsers(dest="distill_cmd")

    p_dprobe = dsub.add_parser("probe", parents=[common], help="Probe local OpenAI-compatible open-source teachers")
    p_dprobe.add_argument("--timeout", type=float, default=0.6)
    p_dprobe.add_argument("--json", action="store_true")
    p_dprobe.set_defaults(func=cmd_distill_probe)

    p_dharv = dsub.add_parser("harvest", parents=[common], help="Harvest & verify traces into canonical OpenWALDO export")
    p_dharv.add_argument("--input", "-i", action="append", required=True, help="JSONL trace, EXPORT.json, or directory")
    p_dharv.add_argument("--out", "-o", default=None, help="Destination directory for canonical OpenWALDO export")
    p_dharv.add_argument("--verify", default="nonempty", help="Comma-separated verifiers: nonempty,json_schema,python_ast,omc_compile,regex")
    p_dharv.add_argument("--corpus-name", dest="corpus_name", default="distilled-traces")
    p_dharv.add_argument("--license", default="Apache-2.0")
    p_dharv.add_argument("--json", action="store_true")
    p_dharv.set_defaults(func=cmd_distill_harvest)

    p_drun = dsub.add_parser("run", parents=[common], help="Execute dk-univ, seq-univ, or dna-strand distillation")
    p_drun.add_argument("--spec", default=None)
    p_drun.add_argument("--mode", choices=("dk-univ", "seq-univ", "dna-strand"), default=None)
    p_drun.add_argument("--arch", default=None)
    p_drun.add_argument("--V", type=int, default=None)
    p_drun.add_argument("--T", type=int, default=None)
    p_drun.add_argument("--D", type=int, default=None)
    p_drun.add_argument("--H", type=int, default=None)
    p_drun.add_argument("--H-KV", dest="H_KV", type=int, default=None)
    p_drun.add_argument("--F", type=int, default=None)
    p_drun.add_argument("--L", type=int, default=None)
    p_drun.add_argument("--B", type=int, default=None)
    p_drun.add_argument("--K", type=int, default=None)
    p_drun.add_argument("--alpha", type=float, default=None)
    p_drun.add_argument("--temperature", type=float, default=None)
    p_drun.add_argument("--steps", type=int, default=None)
    p_drun.add_argument("--verify", default=None)
    p_drun.add_argument("--trace", action="append", default=[])
    p_drun.add_argument("--prompt", action="append", default=[])
    p_drun.add_argument("--teacher-url", dest="teacher_url", default=None)
    p_drun.add_argument("--teacher-model", dest="teacher_model", default=None)
    p_drun.add_argument("--teacher-ckpt", dest="teacher_ckpt", default=None)
    p_drun.add_argument("--dk-path", dest="dk_path", default=None)
    p_drun.add_argument("--tokens-bin", dest="tokens_bin", default=None)
    p_drun.add_argument("--export-hf", dest="export_hf", action="store_true")
    p_drun.add_argument("--export-gguf", dest="export_gguf", action="store_true")
    p_drun.add_argument("--json", action="store_true")
    p_drun.set_defaults(func=cmd_distill_run)

    p_dstat = dsub.add_parser("status", parents=[common], help="Inspect distillation runs and BOMs")
    p_dstat.add_argument("run_id", nargs="?", default=None)
    p_dstat.set_defaults(func=cmd_distill_status)
    p_dist.set_defaults(func=lambda a: (p_dist.print_help(), 0)[1])

    p_gs = sub.add_parser("gsamp", parents=[common], help="Grammar-constrained finite-state decoding via ollm-univ gsamp")
    p_gs.add_argument("ck_path")
    p_gs.add_argument("--seed", type=int, default=42)
    p_gs.add_argument("--n-tokens", dest="n_tokens", type=int, default=32)
    p_gs.add_argument("--temp-milli", dest="temp_milli", type=int, default=900)
    p_gs.add_argument("--mode", type=int, choices=(1, 2, 3), default=1)
    p_gs.add_argument("--json", action="store_true")
    p_gs.set_defaults(func=cmd_runtime_gsamp)

    p_gd = sub.add_parser("guard", parents=[common], help="Anti-DNA contrastive decode + JSD disagreement LTP0 receipt")
    p_gd.add_argument("base_ck")
    p_gd.add_argument("--strand", action="append", required=True, help="SIGNED_WEIGHT_MILLI:PACK1[,PACK2]")
    p_gd.add_argument("--prompt", default="Omega ")
    p_gd.add_argument("--n-tokens", dest="n_tokens", type=int, default=12)
    p_gd.add_argument("--temp-milli", dest="temp_milli", type=int, default=0)
    p_gd.add_argument("--seed", type=int, default=42)
    p_gd.add_argument("--disagree-milli", dest="disagree_milli", type=int, default=50)
    p_gd.add_argument("--max-extra", dest="max_extra", type=int, default=2)
    p_gd.add_argument("--merges", default="-")
    p_gd.add_argument("--json", action="store_true")
    p_gd.set_defaults(func=cmd_runtime_guard)

    p_slp = sub.add_parser("sleep", parents=[common], help="Verifier-gated offline sleep cycle consolidation -> .sp")
    p_slp.add_argument("base_ck")
    p_slp.add_argument("--held", required=True)
    p_slp.add_argument("--interaction", action="append", required=True)
    p_slp.add_argument("--out-dir", dest="out_dir", required=True)
    p_slp.add_argument("--verify", default="nonempty")
    p_slp.add_argument("--max-regression", dest="max_regression", type=float, default=0.25)
    p_slp.add_argument("--steps", type=int, default=8)
    p_slp.add_argument("--date-tag", dest="date_tag", default=None)
    p_slp.add_argument("--json", action="store_true")
    p_slp.set_defaults(func=cmd_runtime_sleep)

    p_grn = sub.add_parser("green", parents=[common], help="Compile-until-green LTP0 + micro-TTT crystallization -> .sp")
    p_grn.add_argument("ck_path")
    p_grn.add_argument("--prompt", required=True)
    p_grn.add_argument("--verifier", default="omc_compile")
    p_grn.add_argument("--max-attempts", dest="max_attempts", type=int, default=3)
    p_grn.add_argument("--out-dir", dest="out_dir", required=True)
    p_grn.add_argument("--json", action="store_true")
    p_grn.set_defaults(func=cmd_runtime_green)

    p_ver = sub.add_parser("verify", parents=[common], help="One-command offline Cryptographic Trust Audit")
    p_ver.add_argument("--base-ck", dest="base_ck", default=None)
    p_ver.add_argument("--pack", action="append", default=None)
    p_ver.add_argument("--bom", default=None)
    p_ver.add_argument("--json", action="store_true")
    p_ver.set_defaults(func=cmd_commons_verify)

    p_pkg = sub.add_parser("pkg", parents=[common], help="DNA Commons package manager for signed .sp chains")
    pkg_sub = p_pkg.add_subparsers(dest="pkg_action", required=True)

    p_pub = pkg_sub.add_parser("publish", help="Publish an OLLMSP01 .sp pack into a registry")
    p_pub.add_argument("sp_path")
    p_pub.add_argument("name")
    p_pub.add_argument("--registry", required=True)
    p_pub.add_argument("--version", default="0.1.0")
    p_pub.add_argument("--license", default="Apache-2.0")
    p_pub.add_argument("--desc", default="")
    p_pub.set_defaults(func=cmd_commons_pkg)

    p_src = pkg_sub.add_parser("search", help="Search DNA Commons packs by query or base_hash")
    p_src.add_argument("--registry", required=True)
    p_src.add_argument("--query", default="")
    p_src.add_argument("--base-hash", dest="base_hash", default="")
    p_src.set_defaults(func=cmd_commons_pkg)

    p_ins = pkg_sub.add_parser("install", help="Install a verified .sp pack from a registry")
    p_ins.add_argument("name")
    p_ins.add_argument("env_dir")
    p_ins.add_argument("--registry", required=True)
    p_ins.add_argument("--base-hash", dest="base_hash", default=None)
    p_ins.set_defaults(func=cmd_commons_pkg)

    p_rev = pkg_sub.add_parser("revoke", help="Revoke and delete a .sp pack (Right to be Forgotten)")
    p_rev.add_argument("name")
    p_rev.add_argument("env_dir")
    p_rev.add_argument("--reason", default="right-to-be-forgotten")
    p_rev.set_defaults(func=cmd_commons_pkg)

    p_dif = pkg_sub.add_parser("diff", help="Diff two OLLMSP01 .sp packs")
    p_dif.add_argument("sp_a")
    p_dif.add_argument("sp_b")
    p_dif.set_defaults(func=cmd_commons_pkg)

    p_mem = sub.add_parser("memory", parents=[common], help="Model-agnostic personal memory & trunk migration")
    mem_sub = p_mem.add_subparsers(dest="mem_action", required=True)

    p_madd = mem_sub.add_parser("add", help="Append a SHA-256 deduplicated memory record")
    p_madd.add_argument("store_path")
    p_madd.add_argument("text")
    p_madd.add_argument("--tag", action="append", default=None)
    p_madd.set_defaults(func=cmd_commons_memory)

    p_mlist = mem_sub.add_parser("list", help="List stored personal memories")
    p_mlist.add_argument("store_path")
    p_mlist.set_defaults(func=cmd_commons_memory)

    p_mexp = mem_sub.add_parser("export-enc", help="Export PBKDF2+HMAC encrypted memory bundle")
    p_mexp.add_argument("store_path")
    p_mexp.add_argument("out_path")
    p_mexp.add_argument("--passphrase", required=True)
    p_mexp.set_defaults(func=cmd_commons_memory)

    p_mimp = mem_sub.add_parser("import-enc", help="Import PBKDF2+HMAC encrypted memory bundle")
    p_mimp.add_argument("enc_path")
    p_mimp.add_argument("out_store_path")
    p_mimp.add_argument("--passphrase", required=True)
    p_mimp.set_defaults(func=cmd_commons_memory)

    p_mmig = mem_sub.add_parser("migrate", help="Replay personal memory onto a new base trunk -> .sp")
    p_mmig.add_argument("store_path")
    p_mmig.add_argument("new_trunk_ck")
    p_mmig.add_argument("out_dir")
    p_mmig.add_argument("--steps", type=int, default=6)
    p_mmig.add_argument("--add-layers", dest="add_layers", type=int, default=1)
    p_mmig.add_argument("--lr-micro", dest="lr_micro", type=int, default=2000)
    p_mmig.set_defaults(func=cmd_commons_memory)

    p_lse = sub.add_parser("lease", parents=[common], help="Capability-bounded local execution leases")
    lse_sub = p_lse.add_subparsers(dest="lease_action", required=True)

    p_liss = lse_sub.add_parser("issue", help="Issue a bounded, revocable capability lease")
    p_liss.add_argument("lease_path")
    p_liss.add_argument("--dir", dest="allowed_dir", action="append", required=True)
    p_liss.add_argument("--cmd", dest="allowed_cmd", action="append", required=True)
    p_liss.add_argument("--allow-network", dest="allow_network", action="store_true")
    p_liss.add_argument("--ttl", type=int, default=3600)
    p_liss.set_defaults(func=cmd_commons_lease)

    p_lrun = lse_sub.add_parser("run", help="Execute a command strictly inside an issued lease")
    p_lrun.add_argument("lease_path")
    p_lrun.add_argument("--cwd", required=True)
    p_lrun.add_argument("--audit", default=None)
    p_lrun.add_argument("--timeout", type=float, default=15.0)
    p_lrun.add_argument("cmd", nargs=argparse.REMAINDER)
    p_lrun.set_defaults(func=cmd_commons_lease)

    p_lrev = lse_sub.add_parser("revoke", help="Revoke a capability lease")
    p_lrev.add_argument("lease_path")
    p_lrev.set_defaults(func=cmd_commons_lease)

    p_usb = sub.add_parser("pack-usb", parents=[common], help="Assemble an air-gapped USB-Stick Sovereignty bundle")
    p_usb.add_argument("out_dir")
    p_usb.add_argument("--trunk-ck", dest="trunk_ck", default=None)
    p_usb.add_argument("--pack", action="append", default=None)
    p_usb.add_argument("--memory", default=None)
    p_usb.add_argument("--json", action="store_true")
    p_usb.set_defaults(func=cmd_commons_pack_usb)

    p_prt = sub.add_parser("port-check", parents=[common], help="Run Hardware-Porting Kit 9-gate oracle parity check")
    p_prt.add_argument("--bin", default=None)
    p_prt.add_argument("--out-json", dest="out_json", default=None)
    p_prt.add_argument("--json", action="store_true")
    p_prt.set_defaults(func=cmd_commons_port_check)

    p_pil = sub.add_parser("pillars", parents=[common], help="Verify pure-Omega 8-pillar runtime & Q32 reversible locks")
    p_pil.add_argument("--steps", type=int, default=8)
    p_pil.add_argument("--lr-milli", dest="lr_milli", type=int, default=10)
    p_pil.add_argument("--json", action="store_true")
    p_pil.set_defaults(func=cmd_pillars)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args_in = list(sys.argv[1:] if argv is None else argv)
    if not args_in:
        args_in = ["studio"]
    ns = parser.parse_args(args_in)
    if not getattr(ns, "func", None):
        parser.print_help()
        return 0
    try:
        return int(ns.func(ns))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(red(f"ERROR: {exc}"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
