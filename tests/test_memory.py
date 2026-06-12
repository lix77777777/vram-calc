"""公式函数单测.

关键数字三类来源交叉验证:
1. docs/formulas.md §7 手算示例
2. Phase 3 实测金标准 (qwen2.5-0.5b, RTX 5060 Ti, 2026-06-10)
3. 边界情况 (0 batch / 超长序列 / INT4 / r=0)
"""

import pytest

from vram_calc import (
    GiB, Precision,
    weights_memory, lora_num_params, qlora_weights_memory,
    gradients_memory, optimizer_memory,
    activations_memory, kv_cache_memory, framework_overhead,
    estimate_training, estimate_inference, get_model,
)

L2 = get_model("llama-2-7b")
L3 = get_model("llama-3-8b")
Q05 = get_model("qwen2.5-0.5b")


# ---------------- §1 权重
def test_weights_basic():
    assert weights_memory(1000, Precision.FP32) == 4000
    assert weights_memory(1000, Precision.FP16) == 2000
    assert weights_memory(1000, Precision.INT8) == 1000
    assert weights_memory(1000, Precision.INT4) == 500


def test_weights_quant_extra():
    assert weights_memory(8000, Precision.INT4, quant_extra_bits=0.127) == pytest.approx(
        8000 * 0.5 + 8000 * 0.127 / 8)


def test_weights_negative_raises():
    with pytest.raises(ValueError):
        weights_memory(-1, Precision.FP16)


# ---------------- LoRA / QLoRA（金标准: peft 与 bitsandbytes 实载）
def test_lora_params_match_peft_exactly():
    assert lora_num_params(Q05, 16, "all-linear") == 8_798_208  # peft 实载值

def test_lora_params_llama2_r16():
    assert lora_num_params(L2, 16, "all-linear") == 39_976_960  # formulas.md §7

def test_qlora_weights_match_measured():
    measured = 0.4592 * GiB                                     # Phase 3 实测
    assert qlora_weights_memory(Q05, 16) == pytest.approx(measured, rel=0.01)

def test_lora_rank_zero():
    assert lora_num_params(L2, 0) == 0


# ---------------- §2/§3 梯度与优化器
def test_gradients():
    assert gradients_memory(100, Precision.BF16) == 200
    assert gradients_memory(100, Precision.FP32) == 400

def test_optimizer_regimes():
    assert optimizer_memory(100, "adamw") == 800                          # FP32 m+v
    assert optimizer_memory(100, "adamw", master_weights=True) == 1200    # K=12
    assert optimizer_memory(100, "adamw", state_precision=Precision.BF16) == 400  # 纯bf16 ✓实测
    assert optimizer_memory(100, "adamw_8bit") == 200
    assert optimizer_memory(100, "sgd") == 0
    assert optimizer_memory(100, "sgd_momentum") == 400

def test_optimizer_bf16_pure_matches_measured():
    # 实测: 494M 参数纯 bf16 AdamW 状态 = 1.845 GiB
    assert optimizer_memory(Q05.num_params, "adamw",
                            state_precision=Precision.BF16) == pytest.approx(
        1.845 * GiB, rel=0.01)

def test_optimizer_unknown():
    with pytest.raises(ValueError):
        optimizer_memory(100, "adafactor")


# ---------------- §4 激活值（金标准: Phase 3 实测, 容差 5%）
MEAS_ACT = {  # (attn_impl, ckpt, lora) -> 实测 GiB, b2 s1024
    ("eager", False, False): 7.892,
    ("sdpa", False, False): 4.023,
    ("eager", True, False): 1.258,
    ("eager", False, True): 8.933,
}

@pytest.mark.parametrize("key,meas", MEAS_ACT.items())
def test_activations_match_measured(key, meas):
    attn, ckpt, lora = key
    pred = activations_memory(Q05, 2, 1024, attn_impl=attn,
                              gradient_checkpointing=ckpt, lora=lora)
    assert pred == pytest.approx(meas * GiB, rel=0.05)

def test_activations_zero():
    assert activations_memory(L2, 0, 4096) == 0.0
    assert activations_memory(L2, 8, 0) == 0.0

def test_activations_logits_term_excludable():
    with_l = activations_memory(Q05, 2, 1024)
    without = activations_memory(Q05, 2, 1024, include_logits=False)
    assert with_l - without == pytest.approx(3.0 * 2 * 1024 * Q05.vocab_size)

def test_activations_sdpa_drops_quadratic():
    assert activations_memory(L2, 1, 8192, attn_impl="sdpa") < \
        activations_memory(L2, 1, 8192, attn_impl="eager")

def test_activations_very_long_seq_finite():
    assert activations_memory(L2, 1, 1_000_000) > 0

def test_activations_bad_attn_impl():
    with pytest.raises(ValueError):
        activations_memory(L2, 1, 128, attn_impl="paged")


# ---------------- §5 KV-Cache（公式实测验证 ✓）
def test_kv_llama2_exact_2gib():
    assert kv_cache_memory(L2, 1, 4096, Precision.FP16) == pytest.approx(2.0 * GiB)

def test_kv_gqa_llama3():
    assert kv_cache_memory(L3, 1, 4096, Precision.FP16) == pytest.approx(0.5 * GiB)

def test_kv_matches_measured_minus_buffer():
    # 实测 0.0324 GiB 含 ~9.2MiB 常驻缓冲; 公式应等于纯 KV 部分
    pred = kv_cache_memory(Q05, 1, 2048, Precision.FP16)
    assert pred == pytest.approx(0.0324 * GiB - 9.2 * 1024**2, rel=0.05)

def test_kv_zero():
    assert kv_cache_memory(L2, 0, 4096) == 0.0


# ---------------- §6 开销
def test_overhead():
    assert framework_overhead(0) == 0.75 * GiB
    assert framework_overhead(100 * GiB) == pytest.approx(0.75 * GiB + 8 * GiB)


# ---------------- §7 端到端
def test_full_ft_megatron_100gb():
    bd = estimate_training(L2, mode="full", mixed_precision="megatron",
                           batch=0, seq=0, include_overhead=False)
    assert (bd.weights + bd.gradients + bd.optimizer) / GiB == pytest.approx(100.4, abs=0.5)

def test_qlora_static_about_4_2gb():
    bd = estimate_training(L2, mode="qlora", lora_rank=16,
                           batch=0, seq=0, include_overhead=False)
    assert (bd.weights + bd.gradients + bd.optimizer) / GiB == pytest.approx(4.2, abs=0.2)

def test_inference_7b_fp16():
    bd = estimate_inference(L2, precision=Precision.FP16, batch=1, seq=4096,
                            include_overhead=False)
    assert bd.weights / GiB == pytest.approx(12.55, abs=0.05)
    assert bd.kv_cache / GiB == pytest.approx(2.0, abs=0.02)  # 含 10MiB 缓冲

def test_breakdown_total_is_sum():
    bd = estimate_training(L2, mode="lora", batch=2, seq=2048)
    assert bd.total == pytest.approx(bd.weights + bd.gradients + bd.optimizer
                                     + bd.activations + bd.kv_cache + bd.overhead)

def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        estimate_training(L2, mode="dora")

def test_unknown_mixed_precision_raises():
    with pytest.raises(ValueError):
        estimate_training(L2, mixed_precision="fp8")
