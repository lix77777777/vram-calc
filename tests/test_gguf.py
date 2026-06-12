"""GGUF 估算单测. 金标准: 官方发布的 GGUF 文件大小（公开可查）."""

import pytest

from vram_calc import GiB, GGUF_BPW, estimate_gguf, gguf_weights_memory, get_model

L3 = get_model("llama-3-8b")


def test_q4km_matches_official_file_size():
    # 官方 Meta-Llama-3-8B Q4_K_M GGUF ≈ 4.92 GB(十进制) = 4.58 GiB, 容差 3%
    assert gguf_weights_memory(L3, "Q4_K_M") == pytest.approx(4.58 * GiB, rel=0.03)


def test_bpw_monotonic():
    order = ["Q2_K", "Q3_K_M", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"]
    vals = [GGUF_BPW[k] for k in order]
    assert vals == sorted(vals)


def test_estimate_gguf_components():
    bd = estimate_gguf(L3, quant="Q4_K_M", ctx=8192)
    assert bd.gradients == 0 and bd.optimizer == 0 and bd.activations == 0
    assert bd.kv_cache > 1.0 * GiB            # 8k ctx GQA ≈ 1 GiB + 缓冲
    assert bd.total == pytest.approx(bd.weights + bd.kv_cache + bd.overhead)


def test_unknown_quant_raises():
    with pytest.raises(ValueError):
        gguf_weights_memory(L3, "Q4_X")
