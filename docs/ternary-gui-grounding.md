# What ternary weights do to GUI grounding (and how to get it back)

By **Adrian Ciprian Iancu** · [GitHub](https://github.com/ArtyomITA) · [Hugging Face](https://huggingface.co/adriandj3) · [LinkedIn](https://www.linkedin.com/in/adrian-ciprian-iancu-a2a477391)

I wanted [Ternary-Bonsai-2-9B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf) to be the only model of my local assistant, including the part that clicks on the screen. Out of the box it is bad at pixel clicks. These are the tests I ran on a GTX 1080 (17-20 Oct 2026) to understand why, and what fixed it. All runs: llama.cpp (PrismML fork), greedy decoding, images at up to 1024 tokens, first answer only unless noted.

## 1. The ternarization is what costs the clicks

Same 38 cases (8 desktop screenshots of mine and 30 from ScreenSpot-v2), same prompt:

| Model | Right clicks |
|---|---|
| Qwen3.5-9B Q4_K_M (same architecture, not ternary) | 36 / 38 |
| Holo 0.8B (a small dedicated GUI model) | 35 / 38 |
| Ternary-Bonsai-2-9B | 19 / 38 |
| Ternary-Bonsai-2-9B with Qwen's native `point_2d` format | 19 / 38 |

The prompt and the output format are not the cause: the full-precision model with the same architecture clicks almost perfectly, the ternary one does not. 38 cases is a small sample, so read it as a large effect, not as a precise number.

## 2. Borrowing a GUI expert's weights does not work

UI-Venus-2-9B is a GUI-tuned model on the same base. I took its difference from Qwen3.5-9B and compressed it into a LoRA for Bonsai. At rank 64 the LoRA keeps only 36% of the difference's energy (10% in the worst layer), and it scored 19/38 against 18/38 for the base. The skill lives in a high-rank change, so it does not transfer through a low-rank adapter. The vision tower of UI-Venus is identical to Bonsai's (difference 0.0), so the gap is all in the language model.

## 3. A LoRA trained on the ternary weights brings it back

I rebuilt the PQ2_0 model in PyTorch and trained LoRA adapters directly on the frozen ternary weights, on the same GTX 1080 ([training kit](../lora-training/)). ScreenSpot-v2, 100 cases (50 desktop, 50 web):

| Adapter | Right clicks | Seconds per click |
|---|---|---|
| none | 38 | 4.3 |
| short LoRA (600 examples) | 66 | 4.5 |
| longer LoRA (+600 examples) | 72 | 4.4 |
| short LoRA + two-step zoom | 75 | 9.2 |
| Holo 0.8B, for reference | 92 | 4.6 |

Base to short LoRA: 35 cases gained, 7 lost (McNemar p < 0.0001). On a larger 200-case subset the later adapters reached 153 and then 156 with [TernaryPilot v1](https://huggingface.co/adriandj3/TernaryPilot-v1-Computer-Browser-Use-LoRA), against 80 for the base model.

## 4. When you can, do not click by pixels at all

If the app or the page can give you a list of its elements, picking from that list is almost free and almost perfect, with no training:

| Test | Text list of elements | Numbered boxes on the screenshot (set-of-mark) | Pixel click |
|---|---|---|---|
| Browser, 25 tasks on 7 real pages | 22 / 25 (1.1 s) | 14 / 25 | 2 / 25 |
| Windows desktop, 37 targets in 4 real apps (UI Automation list) | 37 / 37 (3.7 s) | 10 / 37 | 4 / 37 |

That is how I use it now: text list first, pixel click with the LoRA only when there is no usable list.

## Limits

- One model family and one card. I do not claim this for every ternary model.
- The 38-case comparison is small; the 100 and 200 case results are the stronger ones.
- My own desktop and browser cases are not published (they are screenshots of my screen and of third-party sites); ScreenSpot-v2 cases are referenced by index only.
