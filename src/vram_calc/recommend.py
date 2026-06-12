"""显卡推荐: 给定显存需求, 返回能跑的显卡."""

from __future__ import annotations

from .memory import GiB

# 物理显存 (GiB). 显存规格为公开稳定信息.
GPU_VRAM: dict[str, float] = {
    "RTX 4060": 8, "RTX 3060 12GB": 12, "RTX 4070": 12,
    "RTX 4080": 16, "T4": 16, "RTX 5060 Ti 16GB": 16,
    "RTX 3090": 24, "RTX 4090": 24, "L4": 24,
    "RTX 5090": 32, "V100 32GB": 32, "A100 40GB": 40,
    "A100 80GB": 80, "H100 80GB": 80,
}


def recommend(required_bytes: float, *, headroom: float = 0.0) -> list[str]:
    """返回显存 ≥ required×(1+headroom) 的显卡, 按显存升序.

    框架开销已在 estimate_* 中计入, headroom 缺省 0.
    """
    if required_bytes < 0:
        raise ValueError("required_bytes must be >= 0")
    need = required_bytes * (1.0 + headroom) / GiB
    return [n for n, v in sorted(GPU_VRAM.items(), key=lambda kv: (kv[1], kv[0]))
            if v >= need]
