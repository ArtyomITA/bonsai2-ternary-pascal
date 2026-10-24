"""Encodes every screenshot once with Bonsai's vision tower (weights from the mmproj GGUF, the same tower llama.cpp runs)
and saves the embeddings (image tokens x 4096, fp16), so the tower is not on the GPU during training.
Usage: python vision_cache.py <train.jsonl> [max_image_tokens=1024]
Paths: MMPROJ (Ternary-Bonsai-2-9B-mmproj-BF16.gguf), QWEN_SRC (config.json + preprocessor_config.json from
Qwen/Qwen3.5-9B), CACHE_DIR (output folder), GGUF_PY (gguf-py of the PrismML fork)."""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

sys.path.insert(0, os.environ.get("GGUF_PY", "llama.cpp/gguf-py"))
import gguf  # noqa: E402

MMPROJ = os.environ.get("MMPROJ", "Ternary-Bonsai-2-9B-mmproj-BF16.gguf")
SRC = os.environ.get("QWEN_SRC", "qwen35-9b")
CACHE = os.environ.get("CACHE_DIR", "vision-cache")


def torre(device="cuda"):
    cfg = mq.Qwen3_5Config(**json.load(open(SRC + "/config.json"))).vision_config
    cfg._attn_implementation = "sdpa"
    v = mq.Qwen3_5VisionModel(cfg)
    r = gguf.GGUFReader(MMPROJ)

    def t(nome):
        x = r.get_tensor([i for i, tt in enumerate(r.tensors) if tt.name == nome][0])
        a = np.array(x.data)
        if x.tensor_type == gguf.GGMLQuantizationType.BF16:
            a = (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        return torch.from_numpy(a.astype(np.float32).reshape([int(s) for s in reversed(x.shape)]))

    sd = {"patch_embed.proj.weight": torch.stack([t("v.patch_embd.weight"), t("v.patch_embd.weight.1")], 2),
          "patch_embed.proj.bias": t("v.patch_embd.bias"), "pos_embed.weight": t("v.position_embd.weight"),
          "merger.norm.weight": t("v.post_ln.weight"), "merger.norm.bias": t("v.post_ln.bias"),
          "merger.linear_fc1.weight": t("mm.0.weight"), "merger.linear_fc1.bias": t("mm.0.bias"),
          "merger.linear_fc2.weight": t("mm.2.weight"), "merger.linear_fc2.bias": t("mm.2.bias")}
    for i in range(cfg.depth):
        for hf, gg in (("attn.qkv", "attn_qkv"), ("attn.proj", "attn_out"), ("mlp.linear_fc1", "ffn_up"), ("mlp.linear_fc2", "ffn_down"),
                       ("norm1", "ln1"), ("norm2", "ln2")):
            for k in ("weight", "bias"):
                sd[f"blocks.{i}.{hf}.{k}"] = t(f"v.blk.{i}.{gg}.{k}")
    mancanti = v.load_state_dict(sd, strict=False)
    assert not mancanti.unexpected_keys, mancanti
    assert all("rotary" in k for k in mancanti.missing_keys), mancanti.missing_keys
    return v.to(device).float().eval()


def proc(max_tok):
    p = AutoImageProcessor.from_pretrained(SRC)
    p.max_pixels = max_tok * 32 * 32
    p.min_pixels = 64 * 32 * 32
    p.size = {"shortest_edge": p.min_pixels, "longest_edge": p.max_pixels}
    return p


if __name__ == "__main__":
    righe = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
    mt = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    os.makedirs(CACHE, exist_ok=True)
    v, p = torre(), proc(mt)
    fatti = 0
    for r in {r["cache"]: r for r in righe}.values():
        f = f"{CACHE}/{r['cache']}.pt"
        if os.path.exists(f):
            continue
        x = p(images=[Image.open(r["img"]).convert("RGB")], return_tensors="pt")
        with torch.no_grad():
            e = v(x["pixel_values"].cuda().float(), grid_thw=x["image_grid_thw"].cuda())
        e = e.pooler_output if hasattr(e, "pooler_output") else e
        torch.save({"emb": e.half().cpu(), "thw": x["image_grid_thw"][0]}, f + ".tmp")
        os.replace(f + ".tmp", f)
        fatti += 1
        if fatti % 50 == 0:
            print("encoded", fatti, flush=True)
    print("done", fatti)
