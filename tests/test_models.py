"""预置配置正确性: 计算参数量 vs 官方公布值, 偏差 < 1%."""

import pytest

from vram_calc import MODELS, ModelConfig, get_model

# 官方公布参数量（来源: 各模型 HF 卡片/论文, 2026-06-10 查证）
PUBLISHED = {
    "llama-2-7b": 6.74e9,
    "llama-3-8b": 8.03e9,
    "mistral-7b-v0.1": 7.24e9,
    "qwen2.5-7b": 7.61e9,
    "qwen2.5-0.5b": 0.49e9,
    "tinyllama-1.1b": 1.10e9,
}


def test_at_least_10_models():
    assert len(MODELS) >= 10


@pytest.mark.parametrize("name,published", PUBLISHED.items())
def test_num_params_matches_published(name, published):
    calc = get_model(name).num_params
    assert abs(calc - published) / published < 0.01, f"{name}: {calc:,}"


def test_all_models_have_source():
    assert all(m.source.startswith("https://huggingface.co/") for m in MODELS.values())


def test_gqa_fields():
    assert get_model("llama-2-7b").num_kv_heads == 32   # MHA
    assert get_model("llama-3-8b").num_kv_heads == 8    # GQA
    assert get_model("qwen3-8b").head_dim_ == 128


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        ModelConfig("bad", 0, 1, 1, 1, 1, 1)
    with pytest.raises(ValueError):
        ModelConfig("bad", 8, 1, 4, 8, 1, 1)  # kv_heads > heads


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        get_model("gpt-5")
