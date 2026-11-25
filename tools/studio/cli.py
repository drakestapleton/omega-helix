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


def cmd_studio(args: argparse.Namespace) -> int:
    srv = studio_server.create_server(
        host=args.host,
        port=args.port,
        workspace_dir=args.workspace_dir,
        bin_dir=args.bin_dir,
    )
    host, port = srv.server_address[:2]
    url = f"http://{host}:{port}/"
    print_banner("Local Browser Studio — Zero Telemetry, 100% Pure-Omega")
    print(f"  {emerald('●')} Studio UI   : {cyan(url)}")
    print(f"  {violet('●')} Runs Dir    : {dim(str(srv.manager.runs_dir))}")
    print(f"  {amber('●')} Stop Server : {dim('Ctrl-C')}")
    print()

    if not args.no_open and sys.stdout.isatty():
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        srv.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print(f"\n  {dim('Omega Helix Studio stopped.')}")
    finally:
        srv.server_close()
    return 0


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

    p_studio = sub.add_parser("studio", parents=[common], help="Launch local Omega Helix Studio web UI (default)")
    p_studio.add_argument("--host", default="127.0.0.1")
    p_studio.add_argument("--port", type=int, default=18420)
    p_studio.add_argument("--no-open", action="store_true", help="Do not open browser automatically")
    p_studio.set_defaults(func=cmd_studio)

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
