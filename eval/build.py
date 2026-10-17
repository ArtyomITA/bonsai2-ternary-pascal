"""Sealed build for the research evaluator: MSVC 14.38 + CUDA 12.4, sm_61, build dir build-win (kept between runs).
Run from the repo root: python research/build.py [jobs]. Line-ending safe (no .bat file)."""
import subprocess
import sys
from pathlib import Path

jobs = sys.argv[1] if len(sys.argv) > 1 else '4'
vcvars = r'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
cuda = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4'
steps = [f'call "{vcvars}" -vcvars_ver=14.38 >nul', f'set "CUDA_PATH={cuda}"']
if not Path('build-win/CMakeCache.txt').exists():
    steps.append(f'cmake -S . -B build-win -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON '
                 f'-DCMAKE_CUDA_ARCHITECTURES=61-real -DCMAKE_CUDA_COMPILER="{cuda}\\bin\\nvcc.exe" -DLLAMA_CURL=OFF')
steps.append(f'cmake --build build-win --target llama-perplexity llama-server llama-bench -j {jobs}')
sys.exit(subprocess.run(' && '.join(steps), shell=True).returncode)
