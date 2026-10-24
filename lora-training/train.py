"""train.py: LoRA training for Ternary-Bonsai-2-9B on a small GPU (written and tested on a GTX 1080, 8 GB, fp32).
This is the trainer that produced TernaryPilot v1. It runs in pieces: start it, stop it, it resumes where it was.

Data: one JSON line per example {"cache", "img", "user", "risposta", "tipo"}:
  cache     key of the screenshot embedding in CACHE_DIR (make it with vision_cache.py)
  user      prompt text (the image always comes first in the user turn)
  risposta  target answer (the loss is only on these tokens)
  tipo      a label for the per-type loss in the log (e.g. "click", "action_pc")
Examples of the same screenshot placed next to each other inside one accumulation window (ACCUM=4) share the image
prefix: it is computed once and every target only pays for its own suffix (3-4x faster with 4 targets per screenshot).
The file is read in order and restarts at the end, so shuffle and group it when you build it.

Usage:
  python train.py                train until HOURS of training time (pauses do not count)
  python train.py --export NAME  write RUN_DIR/NAME.gguf from the last checkpoint, no training
Stop: create RUN_DIR/STOP or press Ctrl+C. It saves the checkpoint and exports RUN_DIR/last.gguf.
Learning rate: warmup, constant, then the last 15% of HOURS decays to 10% (WSD). Before the decay it saves
RUN_DIR/ckpt-stable.pt: to continue a run with a larger HOURS, start from there with FROM_STABLE=1.

Environment: DATA (jsonl), RUN_DIR, HOURS (default 2.5), INIT_ADAPTER (optional .pt from a previous run, same rank),
plus BONSAI_GGUF, QWEN_SRC, GGUF_PY, CACHE_DIR used by bonsai_hf.py and vision_cache.py.
"""
import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq

import bonsai_hf as B
import vision_cache as VC

QUI = os.path.dirname(os.path.abspath(__file__))
RUN = os.environ.get("RUN_DIR", os.path.join(QUI, "run"))
DATI = os.environ.get("DATA", os.path.join(QUI, "data", "train.jsonl"))
DA = os.environ.get("INIT_ADAPTER")
ORE = float(os.environ.get("HOURS", "2.5"))
RANK, ALPHA, LR, ACCUM, CALDA = 16, 32.0, 1e-4, 4, 20
SALVA_OGNI = 10  # steps (3-5 minutes on a GTX 1080)
os.makedirs(RUN, exist_ok=True)
CKPT, STABILE, STOP = (os.path.join(RUN, f) for f in ("ckpt.pt", "ckpt-stable.pt", "STOP"))

tok = AutoTokenizer.from_pretrained(VC.SRC)
IMG_PAD = tok.convert_tokens_to_ids("<|image_pad|>")
m, testa, cfg = B.carica(rank=RANK, alpha=ALPHA)
params = [p for n, p in m.named_parameters() if p.requires_grad]
nomi = [n for n, p in m.named_parameters() if p.requires_grad]


def carica_lora(st):
    for n, p in zip(nomi, params):
        p.data.copy_(st[n])


def esporta(nome):
    path = os.path.join(RUN, nome + ".gguf")
    B.salva_lora_gguf(m, path, RANK, ALPHA)
    torch.save({n: p.detach().cpu() for n, p in zip(nomi, params)}, os.path.join(RUN, nome + ".pt"))
    print("exported", path, flush=True)


if "--export" in sys.argv:
    carica_lora(torch.load(CKPT)["lora"])
    esporta(sys.argv[sys.argv.index("--export") + 1])
    sys.exit(0)

campioni = [json.loads(x) for x in open(DATI, encoding="utf-8")]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
stato = {"passo": 0, "i": 0, "secondi": 0.0, "stabile": False}
sorgente = STABILE if os.environ.get("FROM_STABLE") == "1" else CKPT
if os.path.exists(sorgente):
    ck = torch.load(sorgente)
    carica_lora(ck["lora"])
    opt.load_state_dict(ck["opt"])
    stato = ck["stato"]
    stato["stabile"] = stato["secondi"] >= 0.85 * ORE * 3600  # a larger HOURS opens the constant phase again
    print(f"resumed from {sorgente}: step {stato['passo']}, {stato['secondi'] / 3600:.2f} h", flush=True)
elif DA:
    carica_lora(torch.load(DA))
    print("starting from", DA, flush=True)


def fattore():
    caldo = min(1.0, (stato["passo"] + 1) / CALDA)
    f = stato["secondi"] / (ORE * 3600)
    return caldo * (1.0 if f < 0.85 else max(0.1, 1.0 - 0.9 * (f - 0.85) / 0.15))


def salva(path):
    torch.save({"lora": {n: p.detach().cpu() for n, p in zip(nomi, params)}, "opt": opt.state_dict(), "stato": dict(stato)},
               path + ".tmp")
    os.replace(path + ".tmp", path)


m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
m.disable_input_require_grads()  # transformers forces requires_grad on the embeddings; not needed with non-reentrant checkpointing
m.train()
with torch.device("meta"):
    falso = mq.Qwen3_5Model(mq.Qwen3_5Config(**json.load(open(VC.SRC + "/config.json"))))


def prepara(c):  # ids (1, L), embeddings (1, L, D), M-RoPE positions, prompt and answer lengths
    d = torch.load(f"{VC.CACHE}/{c['cache']}.pt")
    emb, thw = d["emb"].cuda().float(), d["thw"]
    msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": c["user"]}]}]
    pre = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pre = pre.replace("<|image_pad|>", "<|image_pad|>" * emb.shape[0], 1)
    ip = tok(pre, add_special_tokens=False)["input_ids"]
    ir = tok(c["risposta"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = torch.tensor([ip + ir], device="cuda")
    ic = ids.cpu()
    pos, _ = falso.get_rope_index(ic, (ic == IMG_PAD).int(), image_grid_thw=thw.reshape(1, 3), attention_mask=torch.ones_like(ic))
    e = m.embed_tokens(ids).masked_scatter((ids == IMG_PAD).unsqueeze(-1), emb)
    return ids, e, pos.cuda(), len(ip), len(ir)


def perdita(c):
    ids, e, pos, nip, nir = prepara(c)
    h = m(inputs_embeds=e, position_ids=pos, use_cache=False).last_hidden_state[0]
    return torch.nn.functional.cross_entropy(testa(h[nip - 1:nip + nir - 1]).float(), ids[0, nip:])


def perdita_gruppo(gr):  # same screenshot: common prefix computed once
    P = [prepara(c) for c in gr]
    lp = min(p[3] for p in P) - 1
    for p in P[1:]:
        d = (p[0][0, :lp] != P[0][0][0, :lp]).nonzero()
        lp = int(d[0]) if len(d) else lp
    assert lp > 0 and all(torch.equal(p[2][..., :lp], P[0][2][..., :lp]) for p in P)
    cs = [m.rotary_emb(p[1], p[2]) for p in P]
    lens = [lp] + [p[0].shape[1] - lp for p in P]
    X = torch.cat([P[0][1][:, :lp]] + [p[1][:, lp:] for p in P], 1)
    cos, sin = (torch.cat([cs[0][j][:, :lp]] + [c[j][:, lp:] for c in cs], 1) for j in (0, 1))
    for L in m.layers:
        X = torch.utils.checkpoint.checkpoint(B.strato_ramificato, L, X, cos, sin, lens, use_reentrant=False)
    X, ini, sel = m.norm(X), lp, []
    for p in P:
        sel.append(X[0, ini + p[3] - 1 - lp:ini + p[3] - 1 - lp + p[4]])
        ini += p[0].shape[1] - lp
    logit = testa(torch.cat(sel)).float()
    return [torch.nn.functional.cross_entropy(lg, p[0][0, p[3]:]) for lg, p in zip(logit.split([p[4] for p in P]), P)]


print(f"examples {len(campioni)}, HOURS {ORE}, step {stato['passo']}, rank {RANK}, lr {LR}, accum {ACCUM}", flush=True)
acc, per_tipo, t_log = [], {}, time.time()
try:
    while stato["secondi"] < ORE * 3600 and not os.path.exists(STOP):
        t0 = time.time()
        for g in opt.param_groups:
            g["lr"] = LR * fattore()
        opt.zero_grad(set_to_none=True)
        gruppi = []
        for _ in range(ACCUM):
            c = campioni[stato["i"] % len(campioni)]
            stato["i"] += 1
            if gruppi and gruppi[-1][0]["cache"] == c["cache"]:
                gruppi[-1].append(c)
            else:
                gruppi.append([c])
        for gr in gruppi:
            perd = [perdita(gr[0])] if len(gr) == 1 else perdita_gruppo(gr)
            (sum(perd) / ACCUM).backward()
            for c, p in zip(gr, perd):
                acc.append(p.item())
                per_tipo.setdefault(c["tipo"], []).append(p.item())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()  # very different sequence lengths fragment memory: at 8 GB Windows spills to RAM and a step takes 2x
        stato["passo"] += 1
        stato["secondi"] += time.time() - t0
        if not stato["stabile"] and stato["secondi"] >= 0.85 * ORE * 3600:
            salva(STABILE)
            stato["stabile"] = True
            print("saved ckpt-stable (lr decay starts)", flush=True)
        if stato["passo"] % SALVA_OGNI == 0:
            salva(CKPT)
            tipi = " ".join(f"{k}={sum(v) / len(v):.3f}" for k, v in sorted(per_tipo.items()))
            print(f"step {stato['passo']} {stato['secondi'] / 3600:.2f}/{ORE} h loss {sum(acc) / len(acc):.4f} [{tipi}] "
                  f"lr {LR * fattore():.2e} {(time.time() - t_log) / len(acc):.1f} s/example epoch "
                  f"{stato['i'] / len(campioni):.2f}", flush=True)
            acc, per_tipo, t_log = [], {}, time.time()
except KeyboardInterrupt:
    print("interrupted: saving", flush=True)
salva(CKPT)
if os.path.exists(STOP):
    os.remove(STOP)
esporta("last")
print(f"stopped at step {stato['passo']}, {stato['secondi'] / 3600:.2f} h of training", flush=True)
