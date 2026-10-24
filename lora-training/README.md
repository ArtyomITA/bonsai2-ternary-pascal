# Training LoRA adapters on Ternary Bonsai 2 9B with a GTX 1080

By **Adrian Ciprian Iancu** · [GitHub](https://github.com/ArtyomITA) · [Hugging Face](https://huggingface.co/adriandj3) · [LinkedIn](https://www.linkedin.com/in/adrian-ciprian-iancu-a2a477391)

This folder is the training kit behind [TernaryPilot v1](https://huggingface.co/adriandj3/TernaryPilot-v1-Computer-Browser-Use-LoRA), a computer use and browser use LoRA for [Ternary-Bonsai-2-9B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf). Everything here was written and measured on a single GTX 1080 (Pascal, sm_61, 8 GB): no bf16, no fp16 compute worth using (it runs at 1/64 speed), no flash attention, no Triton.

Ternary Bonsai is only published as GGUF (PQ2_0 weights with a Hadamard rotation) and MLX, there are no fp16 weights to fine-tune. So I rebuilt the model exactly in PyTorch from the GGUF and trained the LoRA on top of the frozen ternary weights. The adapter is exported as a GGUF LoRA that the [PrismML llama.cpp fork](https://github.com/PrismML-Eng/llama.cpp) loads with `--lora`.

## Files

| File | What it does |
|---|---|
| `bonsai_hf.py` | PQ2_0 + Hadamard model rebuilt on transformers' `Qwen3_5TextModel`, LoRA layers, GGUF LoRA export, shared image prefix layer |
| `tern_unpack.py` | CUDA kernel that unpacks the 2 bit codes to fp32 in one pass (bit-identical to the torch path), JIT compiled |
| `vision_cache.py` | runs Bonsai's vision tower (from the mmproj GGUF) once per screenshot and caches the embeddings |
| `train.py` | the trainer: checkpoints, stop and resume, time budget, WSD learning rate, per-type loss |
| `make_data.py` | builds the training file from AgentNet (Windows) and Multimodal-Mind2Web (web) |

## Setup

- Python 3.12, `torch==2.14` with CUDA 12.6 (the last PyTorch line with Pascal wheels), `transformers==5.17`, `numpy`, `pillow`, `pyarrow`.
- The PrismML llama.cpp fork for `gguf-py` (it knows the PQ2_0 type) and for inference.
- From [prism-ml/Ternary-Bonsai-2-9B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf): `Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf` and `Ternary-Bonsai-2-9B-mmproj-BF16.gguf`.
- From [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) only the small files (`config.json`, `tokenizer.json`, `tokenizer_config.json`, `preprocessor_config.json`, `chat_template.jinja`): Bonsai 2 9B has the same architecture, the weights come from the GGUF.
- For the CUDA kernel: nvcc and, on Windows, an MSVC shell. Without it the code falls back to plain torch (same numbers, about 15% slower).

```bash
export BONSAI_GGUF=Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf MMPROJ=Ternary-Bonsai-2-9B-mmproj-BF16.gguf
export QWEN_SRC=qwen35-9b GGUF_PY=llama.cpp/gguf-py CACHE_DIR=vision-cache
python make_data.py --agentnet agentnet_win_mac_18k.jsonl --agentnet-images agentnet_png --mind2web "mind2web/train-*.parquet" --out data/train.jsonl --images data/img
python vision_cache.py data/train.jsonl
HOURS=5.5 DATA=data/train.jsonl python train.py
```

Stop it any time with Ctrl+C or by creating `run/STOP`: it saves a checkpoint and writes `run/last.gguf`. Run it again and it resumes from the last checkpoint. `HOURS` counts training time only, so pauses are free.

## How I made it fast

Reference: my first trainer, 20.2 seconds per example (median of 3 runs, all three bit-identical), 5.84 GB of VRAM. Every change below was checked against the reference on the same 8 examples, comparing the per-example loss, the 400 LoRA gradients of the first step (before clipping) and the adapter after two steps. "Bit-identical" means equal to the last bit; "within tolerance" means relative error under 1e-5 on the loss and 1e-4 on gradients and adapter.

| Change | s / example | Check |
|---|---|---|
| reference | 20.2 | 3 runs bit-identical to each other |
| unpack the 2 bit codes with a broadcast shift and an in-place scale, no temporary full-size matrices | | bit-identical, checked byte by byte on the real weights |
| Hadamard 4096 as two 64x64 factors (H4096 = H64 x H64) instead of a dense 4096x4096 matmul | 15.3 | within tolerance (loss 2e-6, gradients 5e-5) |
| own CUDA kernel: one thread per byte, writes four fp32 weights at once | 12.4 | bit-identical to the previous row |
| `disable_input_require_grads()` after enabling gradient checkpointing (see below) | 12.2 | bit-identical |
| **shared image prefix**, 2 targets of the same screenshot | 6.4 | within tolerance (loss 1.3e-6, gradients 1.9e-5), 1 target is bit-identical |
| shared image prefix, 4 targets | 3.3 | same |
| plus training only layers 8 to 31 | 2.7 | the gradients of the trained layers are bit-identical to the full run |

A 10 step end to end run of the real recipe, old trainer against new one, ended with the same printed losses and an adapter within 7.3e-5 (cosine 0.999999993).

After these changes the profile of one example is about 8.2 s of fp32 GEMM (that is the 1080's real limit, about 6 TFLOPS), 3 s of small elementwise ops and the Gated DeltaNet recurrence, 0.4 s of unpacking and 0.3 s of attention. So the only big lever left was to do less GEMM, which is what the shared prefix does: a screenshot is about 1,100 tokens and a target with its answer is 30 to 40 tokens, so 4 targets cost about 1.1 prefixes instead of 4.

Things that did not work:
- a lookup table with `F.embedding` for the unpack: bit-identical but 21% slower than the original;
- measuring which layers or tensors "carry the signal" from a few gradient windows to decide what to train: the ranking did not replicate out of sample (negative rank correlation between halves), because windows that share screenshots are not independent. I decide only with real training runs now.

Two things I had to work around, worth knowing if you train on old GPUs:
- `transformers` 5.17 `gradient_checkpointing_enable()` always calls `enable_input_require_grads()`. With non-reentrant checkpointing it is not needed, and it forces backward from layer 0 even when the lower layers are frozen: freezing them saved nothing until I disabled it (then 8 frozen layers -18%, 16 -35%, 24 -53%).
- `torch.optim.Muon` always runs Newton-Schulz in bfloat16. Pascal has no native bf16, but PyTorch emulates it and on my card it is not slower (two 4096x4096 matmuls: 166 ms in bf16, 214 ms in fp32). I still keep an fp32 version for full precision (cosine 0.9999 against the original on the same input).

## What made it better

Measured on a fixed 200 case stratified subset of ScreenSpot-v2 (clicks), base model 80/200:

| Step | ScreenSpot-v2 subset |
|---|---|
| first grounding LoRA (ShowUI, Wave-UI and GroundCUA clicks) | 140 |
| extrapolating between two adapters (ExPO), no training | 136, no gain |
| new data closer to how people describe targets (functional instructions, target sizes like the benchmark's, labels rounded to 10 units because the model cannot resolve the last digit) | 153 (paired test: 19 better, 6 worse, p = 0.015) |
| TernaryPilot v1: multi-task data (Windows actions from AgentNet, web actions from Mind2Web, screen reading), 5.5 h | 156 |

The jump from v1 is mostly not on clicks: on 145 held out agent steps the right action went from 28 (click-only adapter) to 67, and the format was always right (145/145). Details on the [model page](https://huggingface.co/adriandj3/TernaryPilot-v1-Computer-Browser-Use-LoRA).

## Running it in production with llama.cpp

What I learned serving the adapter next to a normal chat model in the same `llama-server`:

- **Load it at scale 0 and turn it on per request.** `--lora-scaled file.gguf:0` and `"lora": [{"id": 0, "scale": 1}]` only on GUI requests. At scale 0, 12 out of 12 greedy chat answers are identical to the base model and the KL divergence on neutral Italian and English text is at the measurement floor (under 6e-5). With the adapter always on, the chat changes (KL 0.021, 6.4% of tokens with a different top-1) and decoding is about 30% slower (66.4 to 46.8 tok/s with MTP; the MTP acceptance does not drop, the cost is the LoRA math in the verifier).
- **Give GUI requests their own slot.** Changing the adapter on a slot clears that slot's prompt cache. A click on the chat slot cost me 17.7 s of prefill on an 8.6k token context; on a dedicated slot (`"id_slot": 1`) the chat cache stays and the next turn takes 110 ms.
- **Send `"presence_penalty": 0` with clicks.** With the server default of 1.0 I lost 4 clicks out of 200 (137 instead of 141).
- **Same image size as in training:** `--image-max-tokens 1024`.
- **Windows paths:** `--lora-scaled` splits the value on `:`, so `D:\x.gguf:0` breaks. Start the server from the adapter's folder and pass the file name only (I have a fix with a regression test ready for upstream).
- **Training on Windows with 8 GB:** sequences of very different length fragment the CUDA memory and the driver starts spilling to system RAM (I saw 800 MB of "shared" GPU memory and steps twice as slow). One `torch.cuda.empty_cache()` per optimizer step fixed it.

## License and data

Code: MIT (see `LICENSE` in this folder). Training data used for TernaryPilot: AgentNet (MIT), Multimodal-Mind2Web (OpenRAIL), ShowUI-desktop, Wave-UI (per-image licenses of its sources) and GroundCUA (MIT). The kit does not include any data.

## Citation

```bibtex
@misc{iancu2026bonsailora,
  author       = {Iancu, Adrian Ciprian},
  title        = {Training LoRA adapters on Ternary Bonsai 2 9B with a GTX 1080},
  year         = {2026},
  howpublished = {\url{https://github.com/ArtyomITA/bonsai2-ternary-pascal/tree/master/lora-training}}
}
```

Keywords: LoRA, fine-tuning, ternary, 1.58-bit, Ternary Bonsai, PrismML, PQ2_0, Hadamard, Qwen3.5, Gated DeltaNet, GTX 1080, Pascal, sm_61, fp32, low VRAM, GGUF, llama.cpp, computer use, browser use, GUI agent, GUI grounding
