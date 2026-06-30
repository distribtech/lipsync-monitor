"""
Make onnxruntime's CUDA provider loadable on Windows.

Python 3.8+ no longer searches PATH for a DLL's *dependent* DLLs, so even with
CUDA/cuDNN on PATH, onnxruntime's CUDA provider silently falls back to CPU.
We register the CUDA and cuDNN bin directories as DLL search paths and let
onnxruntime preload its dependencies. Safe to import on machines without a GPU
(it just finds nothing and onnxruntime uses CPU).
"""
from __future__ import annotations

import glob
import os


def register_cuda_dlls() -> None:
    dirs: list[str] = []

    cuda_path = os.environ.get("CUDA_PATH", "")
    if cuda_path:
        cuda_bin = os.path.join(cuda_path, "bin")
        if os.path.isdir(cuda_bin):
            dirs.append(cuda_bin)

    # cuDNN ships its DLLs under a CUDA-version subfolder, e.g.
    # C:\Program Files\NVIDIA\CUDNN\v9.6\bin\12.6\cudnn*64_9.dll
    for pat in (
        r"C:\Program Files\NVIDIA\CUDNN\*\bin\**\cudnn*64_*.dll",
        r"C:\Program Files\NVIDIA\CUDNN\*\bin\cudnn*64_*.dll",
    ):
        for dll in glob.glob(pat, recursive=True):
            dirs.append(os.path.dirname(dll))

    for d in dict.fromkeys(dirs):          # de-dup, keep order
        try:
            os.add_dll_directory(d)
        except (OSError, FileNotFoundError):
            pass

    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception:
        pass
