"""Ternary-Bonsai-2-9B (PQ2_0 + Hadamard) rebuilt exactly in PyTorch on top of transformers' Qwen3_5TextModel, to train
LoRA adapters on a GTX 1080. Ternary weights stay packed on the GPU (2 bit codes + one fp16 scale per 128 weights) and are
unpacked to fp32 at each use. The input is rotated as in llama.cpp (build_lora_mm): x' = H(s * x), y = W' x'. The LoRA reads
the input BEFORE the rotation, so the adapter is the same GGUF LoRA the PrismML fork applies with --lora.
Everything is computed in fp32 (fp16 runs at 1/64 speed on Pascal).
HF ordering: V rows of qkv/z/a/b, conv1d channels, dt and A are grouped in HF and interleaved in the GGUF (the converter
interleaves them); ssm_out already has grouped columns in the GGUF, so it needs no permutation.

Paths (environment variables):
  BONSAI_GGUF   Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf (or PQ2_0) from prism-ml/Ternary-Bonsai-2-9B-gguf
  QWEN_SRC      folder with config.json from Qwen/Qwen3.5-9B (same architecture, no weights needed)
  GGUF_PY       gguf-py folder of the PrismML llama.cpp fork (it knows the PQ2_0 type)
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.environ.get("GGUF_PY", "llama.cpp/gguf-py"))
import gguf  # noqa: E402
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq  # noqa: E402
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config  # noqa: E402

GGUF = os.environ.get("BONSAI_GGUF", "Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf")
CONFIG = os.path.join(os.environ.get("QWEN_SRC", "qwen35-9b"), "config.json")
MAPPA = {"linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate", "linear_attn.out_proj": "ssm_out",
         "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
         "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down"}


def hadamard(n, device):
    i = torch.arange(n, device=device)
    par = torch.zeros(n, n, dtype=torch.int64, device=device)
    x = i[:, None] & i[None, :]
    while bool(x.any()):
        par ^= x & 1
        x = x >> 1
    return (1 - 2 * par).float() / math.sqrt(n)


try:  # own CUDA kernel (tern_unpack.py, compiled once): writes W*scale in one pass; without it, plain torch fallback
    import tern_unpack
    _KERNEL = tern_unpack._carica()
except Exception:
    _KERNEL = None


def spacchetta(codici, scale):  # codes (out, in/4) uint8, scales (out, in/128) fp16 -> (out, in) fp32
    if _KERNEL is not None and codici.is_cuda and codici.is_contiguous() and scale.is_contiguous():
        return _KERNEL.unpack(codici, scale, torch.empty(codici.shape[0], codici.shape[1] * 4, device=codici.device))
    # broadcast shift and (q-1)*scale in place on an (out, in/128, 128) view: same values, half the memory traffic
    s = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=codici.device)
    w = ((codici.unsqueeze(-1) >> s) & 3).float().view(codici.shape[0], -1, 128)
    return w.sub_(1.0).mul_(scale.float().unsqueeze(-1)).view(codici.shape[0], -1)


def kron2(z, Hs):  # z (..., a*a) @ kron(Hs, Hs): Sylvester H4096 = H64 x H64, two 64-wide matmuls instead of one dense 4096
    a = Hs.shape[0]
    t = z.reshape(-1, a) @ Hs  # low index
    return torch.matmul(Hs, t.view(-1, a, a)).reshape(z.shape)  # high index


class _MatTernaria(torch.autograd.Function):  # y = x W^T without saving W: unpacked again in backward
    @staticmethod
    def forward(ctx, x, codici, scale):
        ctx.save_for_backward(codici, scale)
        return x @ spacchetta(codici, scale).T

    @staticmethod
    def backward(ctx, gy):
        codici, scale = ctx.saved_tensors
        return gy @ spacchetta(codici, scale), None, None


class Ruota(nn.Module):
    def __init__(self, segni, H):
        super().__init__()
        self.register_buffer("segni", segni, persistent=False)
        self.H = H  # shared, not registered

    def forward(self, x):
        # q/k/v, gate/up and qkv/z get the same tensor: rotate once (like the fork's memo)
        if getattr(self, "_x", None) is x and self._g == torch.is_grad_enabled():
            return self._y
        self._x, self._g, self._y = x, torch.is_grad_enabled(), kron2(x * self.segni, self.H)
        return self._y


class LinearTernario(nn.Module):
    def __init__(self, codici, scale, ruota, rank=0, alpha=None):
        super().__init__()
        self.register_buffer("codici", codici, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        self.ruota = ruota
        self.out_f, self.in_f = codici.shape[0], codici.shape[1] * 4
        self.rank = rank
        if rank:
            self.lora_a = nn.Parameter(torch.randn(rank, self.in_f, device=codici.device) / math.sqrt(self.in_f))
            self.lora_b = nn.Parameter(torch.zeros(self.out_f, rank, device=codici.device))
            self.s = (alpha or rank) / rank

    def forward(self, x):
        if self.out_f > 65536:  # 248k x 4096 output head: unpacked in slices to fit in VRAM
            xr = self.ruota(x.float())
            return torch.cat([_MatTernaria.apply(xr, self.codici[i:i + 16384], self.scale[i:i + 16384])
                              for i in range(0, self.out_f, 16384)], -1)
        y = _MatTernaria.apply(self.ruota(x.float()), self.codici, self.scale)
        if self.rank:
            y = y + (x.float() @ self.lora_a.T) @ self.lora_b.T * self.s
        return y


class EmbedTernario(nn.Module):  # rotated row: h = s * (H z), like build_embd_rows
    def __init__(self, codici, scale, segni, H):
        super().__init__()
        self.register_buffer("codici", codici, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        self.register_buffer("segni", segni, persistent=False)
        self.H = H

    def forward(self, ids):
        z = spacchetta(self.codici[ids.reshape(-1)], self.scale[ids.reshape(-1)])
        h = kron2(z, self.H) * self.segni
        return h.reshape(*ids.shape, -1)


def _gruppi(t, dim, nk, nvpk, hd):  # inverse of _reorder_v_heads: interleaved -> grouped
    s = list(t.shape)
    t = t.reshape(*s[:dim], nvpk, nk, hd, *s[dim + 1:])
    p = list(range(t.ndim))
    p[dim], p[dim + 1] = p[dim + 1], p[dim]
    return t.permute(*p).contiguous().reshape(*s)


def carica(device="cuda", rank=0, alpha=None, moduli=tuple(MAPPA), gguf_path=GGUF):
    """Returns (model, output_head, config). Only the LoRA tensors require grad."""
    cfg = Qwen3_5Config(**{k: v for k, v in json.load(open(CONFIG)).items()}).text_config
    cfg._attn_implementation = "sdpa"
    r = gguf.GGUFReader(gguf_path)
    T = {t.name: t for t in r.tensors}
    sv = r.fields["prism.hadamard.sign_values"].contents()
    sw = r.fields["prism.hadamard.sign_widths"].contents()
    segni, o = {}, 0
    for w in sw:
        segni[w] = torch.tensor(sv[o:o + w], dtype=torch.float32, device=device)
        o += w
    H = hadamard(math.isqrt(int(r.fields["prism.hadamard.block_size"].contents())), device)  # a x a factor, block_size = a*a
    ruote = {w: Ruota(s, H) for w, s in segni.items()}
    nk, nv, hd = cfg.linear_num_key_heads, cfg.linear_num_value_heads, cfg.linear_value_head_dim

    def f32(nome):
        t = T[nome]
        return torch.from_numpy(np.array(t.data, dtype=np.float32).reshape([int(x) for x in reversed(t.shape)]))

    def tern(nome, gruppi_righe=False):
        t = T[nome]
        assert t.tensor_type.name == "PQ2_0", nome
        out, inn = int(t.shape[1]), int(t.shape[0])
        b = np.frombuffer(t.data.tobytes(), dtype=np.uint8).reshape(out, inn // 128, 34)
        sc = torch.from_numpy(b[:, :, :2].copy().view(np.float16).reshape(out, inn // 128))
        cod = torch.from_numpy(b[:, :, 2:].copy().reshape(out, inn // 4))
        if gruppi_righe == "qkv":
            q = 2 * cfg.linear_key_head_dim * nk
            cod = torch.cat([cod[:q], _gruppi(cod[q:], 0, nk, nv // nk, hd)])
            sc = torch.cat([sc[:q], _gruppi(sc[q:], 0, nk, nv // nk, hd)])
        elif gruppi_righe == "z":
            cod, sc = _gruppi(cod, 0, nk, nv // nk, hd), _gruppi(sc, 0, nk, nv // nk, hd)
        return cod.to(device), sc.to(device), inn

    with torch.device("meta"):
        m = mq.Qwen3_5TextModel(cfg)
    for i, L in enumerate(m.layers):
        for hf, gg in MAPPA.items():
            parent, attr = hf.split(".")
            mod = getattr(L, parent, None)
            if mod is None or not hasattr(mod, attr):
                continue
            cod, sc, inn = tern(f"blk.{i}.{gg}.weight", "qkv" if gg == "attn_qkv" else "z" if gg == "attn_gate" else False)
            setattr(mod, attr, LinearTernario(cod, sc, ruote[inn], rank if hf in moduli else 0, alpha))
        p = f"blk.{i}."
        L.input_layernorm.weight = nn.Parameter(f32(p + "attn_norm.weight").to(device) - 1, requires_grad=False)
        L.post_attention_layernorm.weight = nn.Parameter(f32(p + "post_attention_norm.weight").to(device) - 1, requires_grad=False)
        if hasattr(L, "linear_attn"):
            g = L.linear_attn
            g.A_log = nn.Parameter(torch.log(-_gruppi(f32(p + "ssm_a").reshape(-1, 1), 0, nk, nv // nk, 1).reshape(-1)).to(device), requires_grad=False)
            g.dt_bias = nn.Parameter(_gruppi(f32(p + "ssm_dt.bias").reshape(-1, 1), 0, nk, nv // nk, 1).reshape(-1).to(device), requires_grad=False)
            c = f32(p + "ssm_conv1d.weight")  # (channels, 4)
            qk = 2 * cfg.linear_key_head_dim * nk
            c = torch.cat([c[:qk], _gruppi(c[qk:], 0, nk, nv // nk, hd)])
            g.conv1d.weight = nn.Parameter(c.unsqueeze(1).to(device), requires_grad=False)
            for hf, gg in (("in_proj_a", "ssm_alpha"), ("in_proj_b", "ssm_beta")):
                w = _gruppi(f32(p + gg + ".weight"), 0, nk, nv // nk, 1)
                lin = nn.Linear(w.shape[1], w.shape[0], bias=False, device=device)
                lin.weight = nn.Parameter(w.to(device), requires_grad=False)
                setattr(g, hf, lin)
            g.norm.weight = nn.Parameter(f32(p + "ssm_norm.weight").to(device), requires_grad=False)
        else:
            a = L.self_attn
            a.q_norm.weight = nn.Parameter(f32(p + "attn_q_norm.weight").to(device) - 1, requires_grad=False)
            a.k_norm.weight = nn.Parameter(f32(p + "attn_k_norm.weight").to(device) - 1, requires_grad=False)
    m.norm.weight = nn.Parameter(f32("output_norm.weight").to(device) - 1, requires_grad=False)
    cod, sc, inn = tern("token_embd.weight")
    m.embed_tokens = EmbedTernario(cod, sc, segni[inn], H)
    cod, sc, inn = tern("output.weight")
    testa = LinearTernario(cod, sc, ruote[inn])
    m.rotary_emb = mq.Qwen3_5TextRotaryEmbedding(config=cfg, device=device)
    for n, b in list(m.named_buffers()):
        assert b.device.type != "meta", n
    for n, p in m.named_parameters():
        assert p.device.type != "meta", n
        p.requires_grad = "lora_" in n
    return m, testa, cfg


def salva_lora_gguf(m, path, rank, alpha=None):  # same format the fork reads with --lora
    w = gguf.GGUFWriter(path, "qwen35")
    w.add_type(gguf.GGUFType.ADAPTER)
    w.add_string(gguf.Keys.Adapter.TYPE, "lora")
    w.add_float32(gguf.Keys.Adapter.LORA_ALPHA, float(alpha or rank))
    for i, L in enumerate(m.layers):
        for hf, gg in MAPPA.items():
            parent, attr = hf.split(".")
            mod = getattr(getattr(L, parent, None), attr, None)
            if not isinstance(mod, LinearTernario) or not mod.rank:
                continue
            a, b = mod.lora_a.detach().float().cpu(), mod.lora_b.detach().float().cpu()
            if gg == "attn_qkv":  # V rows interleaved again as in the base GGUF
                q = 2 * 128 * 16
                b = torch.cat([b[:q], _affianca(b[q:])])
            elif gg == "attn_gate":
                b = _affianca(b)
            w.add_tensor(f"blk.{i}.{gg}.weight.lora_a", a.to(torch.float16).numpy())
            w.add_tensor(f"blk.{i}.{gg}.weight.lora_b", b.to(torch.float16).numpy())
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _affianca(t, nk=16, nvpk=2, hd=128):
    s = list(t.shape)
    t = t.reshape(nk, nvpk, hd, *s[1:])
    return t.permute(1, 0, 2, *range(3, t.ndim)).contiguous().reshape(s)


# ---- Shared prefix. K targets on the same screenshot = one sequence X = [prefix | suffix 1 | ... | suffix K] (lens = lengths).
# Projections, norms and MLP run on all tokens together (one weight unpack); causal conv, GDN recurrence and attention run per
# segment: the prefix alone, every branch with the prefix's state / keys. Same math as Qwen3_5DecoderLayer on each full
# sequence (different summation order: within tolerance, not bit-exact).
def _gdn(g, x, lens):
    n, k = x.shape[1], g.conv_kernel_size
    mixed = g.in_proj_qkv(x).transpose(1, 2)  # (1, C, T)
    z = g.in_proj_z(x).reshape(1, n, -1, g.head_v_dim)
    beta, a = g.in_proj_b(x).sigmoid(), g.in_proj_a(x)
    seg = mixed.split(lens, -1)
    w, ctx = g.conv1d.weight, seg[0][..., -(k - 1):]  # branches see the last k-1 REAL inputs of the prefix, not zeros
    conv = lambda s: F.conv1d(s, w, None, groups=w.shape[0])  # noqa: E731
    mixed = torch.cat([conv(F.pad(seg[0], (k - 1, 0)))] + [conv(torch.cat([ctx, s], -1)) for s in seg[1:]], -1)
    mixed = mq.ACT2FN[g.activation](mixed).transpose(1, 2)
    q, kk, v = torch.split(mixed, [g.key_dim, g.key_dim, g.value_dim], -1)
    q, kk, v = (t.reshape(1, n, -1, d) for t, d in ((q, g.head_k_dim), (kk, g.head_k_dim), (v, g.head_v_dim)))
    rip = g.num_v_heads // g.num_k_heads
    if rip > 1:
        q, kk = q.repeat_interleave(rip, 2), kk.repeat_interleave(rip, 2)
    dec = -g.A_log.float().exp() * F.softplus(a.float() + g.dt_bias)
    out, stato = [], None
    for i, (qi, ki, vi, di, bi) in enumerate(zip(*(t.split(lens, 1) for t in (q, kk, v, dec, beta)))):
        o, s = mq.torch_chunk_gated_delta_rule(qi, ki, vi, g=di, beta=bi, initial_state=stato, output_final_state=i == 0,
                                               use_qk_l2norm_in_kernel=True)
        out.append(o)
        stato = s if i == 0 else stato  # state after the prefix (padding to 64 uses k=v=beta=0, g=0, so it stays unchanged)
    o = g.norm(torch.cat(out, 1).reshape(-1, g.head_v_dim), z.reshape(-1, g.head_v_dim)).reshape(1, n, -1)
    return g.out_proj(o)


def _attn(at, x, cos, sin, lens):
    n = x.shape[1]
    sh = (1, n, -1, at.head_dim)
    q, gate = torch.chunk(at.q_proj(x).view(1, n, -1, at.head_dim * 2), 2, -1)
    q, k = mq.apply_rotary_pos_emb(at.q_norm(q.reshape(sh)).transpose(1, 2), at.k_norm(at.k_proj(x).view(sh)).transpose(1, 2), cos, sin)
    v = at.v_proj(x).view(sh).transpose(1, 2)
    qs, ks, vs = q.split(lens, 2), k.split(lens, 2), v.split(lens, 2)
    f = mq.ALL_ATTENTION_FUNCTIONS.get_interface(at.config._attn_implementation, mq.eager_attention_forward)
    out = [f(at, qs[0], ks[0], vs[0], None, dropout=0.0, scaling=at.scaling)[0]]  # prefix: causal
    for qi, ki, vi in zip(qs[1:], ks[1:], vs[1:]):  # branch: keys = prefix + own suffix, each query sees the prefix and its past
        kk, vv = torch.cat([ks[0], ki], 2), torch.cat([vs[0], vi], 2)
        mask = torch.ones(qi.shape[2], kk.shape[2], dtype=torch.bool, device=x.device).tril(lens[0])[None, None]
        out.append(f(at, qi, kk, vv, mask, dropout=0.0, scaling=at.scaling)[0])
    return at.o_proj(torch.cat(out, 1).reshape(1, n, -1) * torch.sigmoid(gate.reshape(1, n, -1)))


def strato_ramificato(L, X, cos, sin, lens):
    h = L.input_layernorm(X)
    X = X + (_gdn(L.linear_attn, h, lens) if L.block_type == "linear_attention" else _attn(L.self_attn, h, cos, sin, lens))
    return X + L.mlp(L.post_attention_layernorm(X))
