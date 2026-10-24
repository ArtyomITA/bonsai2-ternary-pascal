"""CUDA kernel that unpacks PQ2_0 ternary weights to fp32 in ONE pass (writes W*scale), used by bonsai_hf.spacchetta.
Same elementwise math as the torch path: (float(q) - 1.0f) * scale_fp32, so the result is bit-identical (building W
involves no rounding). Layout: uint8 codes [out, in/4] (4 weights per byte, weight k = (byte >> 2k) & 3), fp16 scales
[out, in/128] (one group = 128 weights = 32 bytes).
Compiled on first use with torch.utils.cpp_extension (needs nvcc and, on Windows, an MSVC shell). TORCH_CUDA_ARCH_LIST
defaults to 6.1 (Pascal); set it for other GPUs. Self-check without a GPU: python tern_unpack.py"""
import os

import torch
from torch.utils.cpp_extension import load_inline

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cstdint>

__global__ void unpack_pq2(const uint8_t* __restrict__ cod, const __half* __restrict__ sc, float* __restrict__ W,
                           long long total_bytes, int in4, int in128) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;      // one thread = one byte = 4 weights
    if (i >= total_bytes) return;
    long long row = i / in4;
    int col4 = (int)(i - row * in4);
    float s = __half2float(sc[row * in128 + (col4 >> 5)]);                // 128 weights = 32 bytes per scale
    unsigned b = cod[i];
    float4 o;
    o.x = ((float)( b        & 3u) - 1.0f) * s;
    o.y = ((float)((b >> 2)  & 3u) - 1.0f) * s;
    o.z = ((float)((b >> 4)  & 3u) - 1.0f) * s;
    o.w = ((float)((b >> 6)  & 3u) - 1.0f) * s;
    reinterpret_cast<float4*>(W)[i] = o;
}

torch::Tensor unpack(torch::Tensor cod, torch::Tensor sc, torch::Tensor out) {
    TORCH_CHECK(cod.is_cuda() && sc.is_cuda() && out.is_cuda(), "tensors must be on the GPU");
    TORCH_CHECK(cod.scalar_type() == torch::kUInt8 && sc.scalar_type() == torch::kHalf && out.scalar_type() == torch::kFloat32, "dtype");
    TORCH_CHECK(cod.is_contiguous() && sc.is_contiguous() && out.is_contiguous(), "contiguous");
    int64_t rows = cod.size(0), in4 = cod.size(1);
    TORCH_CHECK(out.size(0) == rows && out.size(1) == in4 * 4 && sc.size(0) == rows && sc.size(1) * 32 == in4, "shapes");
    long long total = (long long)rows * in4;
    int threads = 256;
    long long blocks = (total + threads - 1) / threads;
    unpack_pq2<<<(unsigned)blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        cod.data_ptr<uint8_t>(), reinterpret_cast<const __half*>(sc.data_ptr<at::Half>()), out.data_ptr<float>(), total, (int)in4, (int)sc.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
"""
_CPP = "torch::Tensor unpack(torch::Tensor cod, torch::Tensor sc, torch::Tensor out);"
_mod = None


def _carica():
    global _mod
    if _mod is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "6.1")
        _mod = load_inline(name="tern_unpack_pq2", cpp_sources=_CPP, cuda_sources=_CUDA, functions=["unpack"],
                           extra_cuda_cflags=["-O3"], verbose=False)
    return _mod


def spacchetta_ref(codici, scale):  # reference unpack (the original torch path)
    q = torch.stack([(codici >> s) & 3 for s in (0, 2, 4, 6)], -1).reshape(codici.shape[0], -1)
    return (q.float() - 1.0) * scale.float().repeat_interleave(128, dim=1)


if __name__ == "__main__":  # compile, then check the per-thread indexing on CPU against the reference
    _carica()
    print("compiled")
    torch.manual_seed(0)
    out_f, in_f = 6, 512
    cod = (torch.randint(0, 3, (out_f, in_f // 4, 4), dtype=torch.uint8) << torch.tensor([0, 2, 4, 6], dtype=torch.uint8)).sum(-1).to(torch.uint8)
    sc = (torch.randn(out_f, in_f // 128) * 0.05).half()
    ref = spacchetta_ref(cod, sc)
    in4 = in_f // 4
    W = torch.zeros(out_f * in_f)
    for i in range(out_f * in4):
        row, col4 = divmod(i, in4)
        b, s = int(cod[row, col4]), float(sc[row, col4 >> 5].float())
        for k in range(4):
            W[i * 4 + k] = (float((b >> (2 * k)) & 3) - 1.0) * s
    print("thread emulation == reference (bit):", bool(torch.equal(W.reshape(out_f, in_f), ref)))
    if torch.cuda.is_available():
        c, s = cod.cuda(), sc.cuda()
        print("GPU kernel == reference (bit):", bool(torch.equal(_carica().unpack(c, s, torch.empty(out_f, in_f, device="cuda")).cpu(), ref)))
