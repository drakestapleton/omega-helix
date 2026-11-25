#!/usr/bin/env python3
"""Reproducible PyTorch + CUDA Head-to-Head Benchmark on the Exact Omega Helix v2 Shape.

Matches `native/model.om` v2 shape (`V=4096, T=256, D=384, H=12, F=1536, L=8, B=64`, `P=15,833,472`):
  - Token embedding (tied w_tok / lm_head): V * D
  - Positional embedding (w_pos): T * D
  - L decoder blocks, each with pre-attn RMSNorm (D), Q/K/V/O linear projections (4 * D^2, no bias),
    causal SDPA attention, pre-MLP RMSNorm (D), 2-layer ReLU MLP D -> F -> D (2 * D * F, no bias)
  - Final RMSNorm (D)
Total parameters: 4096*384 + 256*384 + 8*(4*384^2 + 2*384*1536 + 2*384) + 384 = 15,833,472.

Usage:
  python3 tests/bench_pytorch_h2h.py --mode eager --steps 30 --warmup 5 --out receipt.json
  python3 tests/bench_pytorch_h2h.py --mode both --steps 30 --warmup 5 --out receipt.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _pkg_tree_bytes(pkg_names: list[str]) -> dict[str, int]:
    """Compute installed byte footprint of Python/CUDA wheel directories cleanly."""
    seen: set[Path] = set()
    roots: list[Path] = []
    for sp in sys.path:
        p = Path(sp)
        if p.is_dir() and p.name == "site-packages" and p not in seen:
            seen.add(p)
            roots.append(p)
    totals: dict[str, int] = {}
    for root in roots:
        for entry in root.iterdir():
            low = entry.name.lower()
            if any(low == n or low.startswith(n + "-") or low.startswith(n + "_") for n in pkg_names):
                sz = 0
                if entry.is_file():
                    sz = entry.stat().st_size
                elif entry.is_dir():
                    for dp, _, fns in os.walk(entry):
                        for fn in fns:
                            fp = Path(dp) / fn
                            try:
                                sz += fp.stat().st_size
                            except OSError:
                                pass
                totals[entry.name] = sz
    return {"packages": totals, "total_bytes": sum(totals.values())}


def run_benchmark(mode: str, steps: int, warmup: int) -> dict[str, object]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available in this Python environment")

    V, T, D, H, F_dim, L, B = 4096, 256, 384, 12, 1536, 8, 64
    head_dim = D // H

    class RMSNormNoBias(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            xf = x.float()
            inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return (xf * inv).to(x.dtype) * self.weight

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = RMSNormNoBias(D)
            self.wq = nn.Linear(D, D, bias=False)
            self.wk = nn.Linear(D, D, bias=False)
            self.wv = nn.Linear(D, D, bias=False)
            self.wo = nn.Linear(D, D, bias=False)
            self.norm2 = RMSNormNoBias(D)
            self.w1 = nn.Linear(D, F_dim, bias=False)
            self.w2 = nn.Linear(F_dim, D, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.norm1(x)
            bsz, seq, _ = h.shape
            q = self.wq(h).view(bsz, seq, H, head_dim).transpose(1, 2)
            k = self.wk(h).view(bsz, seq, H, head_dim).transpose(1, 2)
            v = self.wv(h).view(bsz, seq, H, head_dim).transpose(1, 2)
            att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            att = att.transpose(1, 2).contiguous().view(bsz, seq, D)
            x = x + self.wo(att)
            m = self.norm2(x)
            x = x + self.w2(F.relu(self.w1(m)))
            return x

    class OmegaV2Ref(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok_emb = nn.Embedding(V, D)
            self.pos_emb = nn.Embedding(T, D)
            self.blocks = nn.ModuleList([Block() for _ in range(L)])
            self.norm_f = RMSNormNoBias(D)

        def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            pos = torch.arange(T, device=idx.device)
            x = self.tok_emb(idx) + self.pos_emb(pos)
            for blk in self.blocks:
                x = blk(x)
            x = self.norm_f(x)
            logits = F.linear(x, self.tok_emb.weight)
            return F.cross_entropy(logits.view(-1, V), targets.view(-1))

    torch.manual_seed(42)
    device = torch.device("cuda")
    model = OmegaV2Ref().to(device=device, dtype=torch.bfloat16)
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == 15_833_472, f"unexpected param count {param_count}"

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    idx = torch.randint(0, V, (B, T), device=device, dtype=torch.long)
    targets = torch.randint(0, V, (B, T), device=device, dtype=torch.long)

    fn = model if mode == "eager" else torch.compile(model)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    for _ in range(max(1, warmup)):
        opt.zero_grad(set_to_none=True)
        loss = fn(idx, targets)
        loss.backward()
        opt.step()
    torch.cuda.synchronize(device)

    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = fn(idx, targets)
        loss.backward()
        opt.step()
    torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0

    tokens_per_step = B * T
    total_tokens = steps * tokens_per_step
    step_ms = (dt / steps) * 1000.0
    tok_per_s = total_tokens / dt
    peak_alloc = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))

    return {
        "mode": mode,
        "param_count": param_count,
        "shape": {"V": V, "T": T, "D": D, "H": H, "F": F_dim, "L": L, "B": B},
        "steps": steps,
        "warmup": warmup,
        "elapsed_s": round(dt, 4),
        "step_ms": round(step_ms, 2),
        "tok_per_s": round(tok_per_s, 1),
        "last_loss": round(float(loss.detach().cpu()), 6),
        "peak_cuda_alloc_bytes": peak_alloc,
        "peak_cuda_reserved_bytes": peak_reserved,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
    }


def measure_cold_start(script_path: str, mode: str) -> float:
    """Spawn a clean subprocess and time from process launch through step 1 GPU sync."""
    t0 = time.perf_counter()
    subprocess.run(
        [sys.executable, script_path, "--mode", mode, "--steps", "1", "--warmup", "0", "--quiet"],
        check=True,
    )
    return round((time.perf_counter() - t0) * 1000.0, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproducible PyTorch BF16 H2H benchmark for Omega Helix v2 (15.83M).")
    ap.add_argument("--mode", choices=["eager", "compile", "both"], default="eager")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--measure-cold-start", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if importlib.util.find_spec("torch") is None:
        print(json.dumps({"ok": False, "error": "torch is not installed in this Python interpreter"}))
        return 2

    modes = ["eager", "compile"] if args.mode == "both" else [args.mode]
    results: dict[str, object] = {}
    for m in modes:
        try:
            res = run_benchmark(m, args.steps, args.warmup)
            if args.measure_cold_start:
                res["cold_start_to_step1_ms"] = measure_cold_start(__file__, m)
            results[m] = res
        except Exception as exc:
            results[m] = {"mode": m, "error": str(exc)}

    out_doc = {
        "kind": "omega-helix-pytorch-h2h",
        "schema": 1,
        "footprint": _pkg_tree_bytes(["torch", "nvidia", "triton"]),
        "runs": results,
    }
    if not args.quiet:
        print(json.dumps(out_doc, indent=2))
    if args.out:
        op = Path(args.out).expanduser().resolve()
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(json.dumps(out_doc, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
