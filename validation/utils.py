"""实测工具: 显存快照 / 结果落盘. 所有实验脚本共用.

用法约定: 每个实验输出一个 JSON 到 validation/results/, 由 analyze.py 统一比对.
"""

import json
import pathlib
import sys
import time

import torch

RESULTS = pathlib.Path(__file__).parent / "results"
GiB = 1024 ** 3


def require_cuda() -> None:
    if not torch.cuda.is_available():
        sys.exit("需要 CUDA GPU。本机无卡请在 Google Colab 运行（见 validation/README.md）")


def snapshot() -> dict:
    """当前显存三件套（字节）."""
    return {
        "allocated": torch.cuda.memory_allocated(),
        "reserved": torch.cuda.memory_reserved(),
        "max_allocated": torch.cuda.max_memory_allocated(),
    }


def reset_peak() -> None:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


def gpu_info() -> dict:
    p = torch.cuda.get_device_properties(0)
    return {"name": p.name, "total_vram": p.total_memory,
            "torch": torch.__version__, "cuda": torch.version.cuda}


def save_result(name: str, data: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    data = {"experiment": name, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu": gpu_info(), **data}
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"[saved] {out}")


def fmt(b: float) -> str:
    return f"{b / GiB:.3f} GiB"
