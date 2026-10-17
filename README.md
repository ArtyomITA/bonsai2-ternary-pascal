# Ternary Bonsai 2 9B on a GTX 1080 (sm_61): +51% decode with bit-identical output

A set of CUDA changes to the [PrismML fork of llama.cpp](https://github.com/PrismML-Eng/llama.cpp) that makes
[Ternary-Bonsai-2-9B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf) (PQ2_0 weights + MTP draft head)
decode 51% faster on a Pascal GPU (GTX 1080, compute capability 6.1), with the same logits as the stock build.

| | stock fork | patched | change |
|---|---|---|---|
| decode tok/s, production setup (PQ2_0 + MTP, see below), GPU starting at <= 55 C | 55.9 | 84.4 | **+51%** |
| same, stock build re-measured in the afternoon session | 57.0 | 84.4 | +48% |
| KL divergence vs stock logits (ubatch 512 / 3 / 1) | | 0 / 0 / 1e-6 | at the floor |
| same top token (ubatch 512 / 3 / 1) | | 100 / 100 / 99.902% | at the floor (see Quality) |
| greedy text, 11k tokens, and 4 concurrent slots | | identical | |
| VRAM | | -44 MiB | |

One file changes: `ggml/src/ggml-cuda/ggml-cuda.cu` (+987 / -22 lines). The patch is in
[`patches/0001-sm61-pq2_0-mtp-decode.patch`](patches/0001-sm61-pq2_0-mtp-decode.patch).

## Setup

- GPU: GTX 1080 8 GB (Pascal, sm_61), Windows 11, CUDA 12.4, MSVC 14.38, `-DCMAKE_CUDA_ARCHITECTURES=61-real`.
- Base: PrismML fork, branch `prism`, commit `0082ac24a`.
- Model: `Ternary-Bonsai-2-9B-PQ2_0.gguf` for the quality gate, `Ternary-Bonsai-2-9B-PQ2_0-MTP-Q8_0.gguf` for speed.
- Production flags (llama-server): `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 0 --spec-draft-p-min 0
  -c 16384 --parallel 1 -ngl 999 -fa on --jinja --reasoning off --cache-ram 0`.
- Workload: a ~5k-token shared context, then 3 greedy answers of 384 tokens (decode at depth ~5k, prefix reused).
  Metric: generated tokens / generation seconds, median of 5 runs.

## What changed

Five changes were kept, each one measured against the previous best and kept only if it gained at least 3%.
The last column is what each piece is worth inside the final program (final / final-without-the-piece - 1),
measured by ablation in the same session. The pieces overlap, so the columns do not add up.

| # | change | gain when kept | worth inside the final program |
|---|---|---|---|
| 1 | CUDA graphs on Pascal, only for small batches (no graph when a quantized mat-mul has more than 8 columns, so prefill keeps the stock path) | +3.9% | +17.1% (A/B, same thermal state) |
| 2 | PQ2_0 mat-vec for 1-4 columns on Pascal that reproduces the summation order of the generic mmvq path bit for bit (warp per 4 rows, no shared memory or barriers, activations in the PQ2 layout, no per-weight decode) | +18.6% | +36.9% |
| 3 | small kernels of the decode cycle without sort: top-k for k <= 32 in two launches (scratch from the pool, no device globals), GDN concat with one thread per element | +6.1% | +3.3% |
| 4 | Hadamard rotation (fwht, block 4096) and q8_1 quantization in the PQ2 layout fused in one kernel, read directly by the mat-vec of #2 | +4.2% | +4.5% (A/B, same thermal state) |
| 5 | fusions with identical output: add + rms_norm (exact in-place alias), rms_norm + mul + swiglu (+ permuted cont), strided F32 copies with fastdiv and the 3 rollback copies in one launch, ssm alpha + beta F32 mat-vec in one launch, GDN conv-state gather + concat + rollback in one kernel | +10.7% | +8.0% (+7.3% A/B) |

Kernel launches per decode cycle: 976 before change 5, 755 after. The PQ2_0 mat-vec stays the largest share of the cycle (~65%),
at ~145 GB/s average, far from the card's bandwidth: with the summation order fixed by the quality gate it is bound by
instructions and occupancy, not memory.

Every fusion has guards and falls back to the stock path when a check fails (multi-sequence batches, non-exact aliases,
overlapping copies). Environment switches, all on by default: `GGML_CUDA_FUSE_ADDNORM`, `GGML_CUDA_FUSE_GATED_NORM`,
`GGML_CUDA_FUSE_F32_PAIR`, `GGML_CUDA_FUSE_CONV_STATE`, `GGML_CUDA_FAST_CPY`; `GGML_CUDA_DISABLE_FUSION` turns the
fusions off.

## Quality

The gate compares the patched build against logits saved from the stock build with `llama-perplexity --kl-divergence`
at three micro-batch sizes, so every kernel path is covered: 512 (prefill), 3 (MTP verification), 1 (single-token
decode). Each must have mean KLD <= 1e-4 and same top token >= 99.5%.

In practice the gate only accepts kernels with the same floating-point summation order as the stock path: a different
order (an earlier attempt, also correct in exact arithmetic) flips some q8_1 rounding and lands at KLD 7e-4. All five
changes were therefore written to be bit-identical to the stock path. The 99.902% at ubatch 1 is not a change: the
stock build measured against its own saved logits gives the same 99.902% (run-to-run nondeterminism at ubatch 1).

Extra checks by an independent auditor role: KLD at ubatch 2 and 4 (floor), greedy A/B on 11k tokens (identical), and
llama-server with `--parallel 4`, 4 concurrent requests and 4 sequential ones: the patched build produces exactly the
same texts as the stock build in both modes (the stock build itself differs between concurrent and sequential at the
same points, from batch composition, and the patched build reproduces that too).

## Caveats

- **Thermals.** Under sustained load the GTX 1080 reaches ~82-86 C and drops from ~1873 to 1632 MHz (temperature
  target, not power: ~130-150 W of 180-200 W). All the numbers above start at <= 55 C. At full heat the patched build
  runs at ~72 tok/s; the stock build was not measured in the same thermal state, so the gain at full heat is not
  measured. Compare builds only A/B in the same thermal state.
- **One workload, one model.** One prompt family, ~5k-token context, 1 slot for speed. Throughput with 4 slots and
  contexts above ~14k tokens were not measured (correctness with 4 slots was).
- **Scope.** Changes 2 and 4 only apply to PQ2_0 (PrismML's group-128 ternary format) and to the Hadamard-rotated
  weights; changes 3 and 5 are partly specific to the Qwen3.5 GDN architecture; change 1 and the generic fusions in 5
  apply to any model on Pascal. Nothing was measured on other models.
- **Measurement hygiene.** Some runs in the log were measured on a stale binary (the build step did not recompile after
  a file copy kept an old modification time); they are marked invalid in `results/`. The kept result was re-checked:
  the binary measured for change 5 is identical in speed to a clean rebuild of the committed source (72.42 vs 72.47
  tok/s, A/B).

## What did not work

- PQ2_0 mat-vec with a different summation order: KLD 7.3e-4 at ubatch 3, rejected by the gate.
- Variants of the ordered mat-vec (4 chunks loaded together, 16-byte loads staged in shared memory, 2 rows per warp,
  a staged kernel for the output head): neutral or slower inside the model (-6%), even when faster in isolation.
- An aligned copy of the PQ2_0 weights (+901 MiB VRAM): +2.4% decode, but the copy at first use slowed prefill.
- CUDA graphs for every batch size: -4.9% prefill.
- Fusing fwht + quantize with the stock layout and mat-vec: neutral (the gain came only with the PQ2 layout of #4).
- Q8_0 mat-vec of the MTP layer in the exact order: no gain (stock path already at 200-215 GB/s).
- Further GDN fusions (conv + silu + l2_norm in one kernel): +1.3%, below the 3% threshold, kept out.

## Reproduce

```bash
git clone https://github.com/PrismML-Eng/llama.cpp && cd llama.cpp
git checkout 0082ac24a
git apply /path/to/patches/0001-sm61-pq2_0-mtp-decode.patch
cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61-real -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-perplexity -j
```

`eval/` holds the evaluator used for every run (`eval_bonsai2.py`, Windows build script `build.py`). It expects the
models in `$BONSAI_MODELS_DIR`, writes the stock references to `$BONSAI_REF_DIR` on a baseline run
(`CHATROOM_KIND=baseline`), and needs a ~65 KB text file at `research/kld.txt` for the KL divergence (any text works:
the gate compares against the stock build on the same text).

## How it was done

An overnight autoresearch run: two worker agents with one git worktree each, a proposer, an auditor, a registrar and a
literature searcher, coordinated in a shared chat with a sealed evaluator; 18 attempts, then ablations, baselines and a
4-slot check. `results/results.tsv` lists every run.

## License

MIT, like llama.cpp. See [LICENSE](LICENSE).
