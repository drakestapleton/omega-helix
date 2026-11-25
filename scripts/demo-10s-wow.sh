#!/bin/sh
# 10-second Open-Trunk -> 2x DNA Specialists + Anti-DNA Firewall Composition Demo (zero CUDA).
set -eu

here=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-/tmp/omega-helix-demo-10s}
OUT=${OUT:-$here/build/verify}

echo "=== Building / resolving pure-Omega binaries ==="
if [ ! -x "$OUT/ollm-interchange" ] || [ ! -x "$OUT/ollm-univ" ]; then
  OUT="$OUT" TARGET=univ sh "$here/scripts/build.sh"
  OUT="$OUT" TARGET=interchange sh "$here/scripts/build.sh"
fi
if [ "$(uname -s)" = Darwin ]; then
  [ -x "$OUT/ollm-macos" ] || OUT="$OUT" TARGET=macos sh "$here/scripts/build.sh"
  OLLMBIN="$OUT/ollm-macos"
elif [ "$(uname -m)" = x86_64 ] || [ "$(uname -m)" = amd64 ]; then
  [ -x "$OUT/ollm-x86linux" ] || OUT="$OUT" TARGET=x86linux sh "$here/scripts/build.sh"
  OLLMBIN="$OUT/ollm-x86linux"
else
  [ -x "$OUT/ollm" ] || OUT="$OUT" TARGET=aarch64 sh "$here/scripts/build.sh"
  OLLMBIN="$OUT/ollm"
fi

rm -rf "$WORK"
mkdir -p "$WORK/hf"

python3 - "$here" "$WORK" "$OLLMBIN" "$OUT/ollm-univ" "$OUT/ollm-interchange" <<'PY'
import json, struct, sys
from pathlib import Path

root, work, ollm_p, univ_p, ic_p = sys.argv[1:]
sys.path.insert(0, str(Path(root) / "tools"))
from runtime.hf_dna import run_open_trunk_dna_demo

hf_dir = Path(work) / "hf"
# Minimal valid Llama-style Safetensors open trunk (zero external network download)
V, D, H, F, L = 256, 16, 2, 32, 2
cfg = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "vocab_size": V,
    "max_position_embeddings": 16,
    "hidden_size": D,
    "num_attention_heads": H,
    "num_key_value_heads": H,
    "intermediate_size": F,
    "num_hidden_layers": L,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "tie_word_embeddings": True,
}
(hf_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

tensors = {}
def add_t(name, shape):
    n = 1
    for s in shape:
        n *= s
    data = struct.pack(f"<{n}f", *(((i % 13) - 6) * 0.01 for i in range(n)))
    tensors[name] = (shape, data)

add_t("model.embed_tokens.weight", [V, D])
for li in range(L):
    add_t(f"model.layers.{li}.input_layernorm.weight", [D])
    add_t(f"model.layers.{li}.self_attn.q_proj.weight", [D, D])
    add_t(f"model.layers.{li}.self_attn.k_proj.weight", [D, D])
    add_t(f"model.layers.{li}.self_attn.v_proj.weight", [D, D])
    add_t(f"model.layers.{li}.self_attn.o_proj.weight", [D, D])
    add_t(f"model.layers.{li}.post_attention_layernorm.weight", [D])
    add_t(f"model.layers.{li}.mlp.gate_proj.weight", [F, D])
    add_t(f"model.layers.{li}.mlp.up_proj.weight", [F, D])
    add_t(f"model.layers.{li}.mlp.down_proj.weight", [D, F])
add_t("model.norm.weight", [D])

header = {}
off = 0
blobs = []
for name, (shape, data) in tensors.items():
    header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + len(data)]}
    off += len(data)
    blobs.append(data)
hbytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
while len(hbytes) % 8 != 0:
    hbytes += b" "
(hf_dir / "model.safetensors").write_bytes(struct.pack("<Q", len(hbytes)) + hbytes + b"".join(blobs))

res = run_open_trunk_dna_demo(
    {"ollm": Path(ollm_p), "ollm_univ": Path(univ_p), "ollm_interchange": Path(ic_p)},
    hf_dir,
    domain_a_text="fn verified_kernel x ulong to ulong ret add x 1 end\n" * 6,
    domain_b_text="check boundary assert audit strict hash merkle zero\n" * 6,
    avoid_text="trust unverified unknown leak skip error ignore\n" * 6,
    out_dir=Path(work) / "run",
)
print("DEMO_10S_WOW_PASS", "final_receipt=" + res["spguard"]["summary"]["final_receipt_hex"], "anti_active=" + str(res["spguard"]["summary"]["anti_active"]))
PY
