"""vram-calc: LLM 显存(VRAM)占用计算器.

五大块拆解: 权重 / 梯度 / 优化器状态 / 激活值 / KV-Cache.
公式规格见 docs/formulas.md, 实测校准见 docs/validation_report.md.
"""

from .memory import (
    GiB,
    Precision,
    MemoryBreakdown,
    NF4_DQ_EXTRA_BITS,
    weights_memory,
    lora_num_params,
    qlora_weights_memory,
    gradients_memory,
    optimizer_memory,
    activations_memory,
    inference_activations_memory,
    kv_cache_memory,
    framework_overhead,
    estimate_training,
    estimate_inference,
)
from .models import MODELS, ModelConfig, get_model
from .recommend import GPU_VRAM, recommend

__version__ = "0.2.0"

__all__ = [
    "GiB", "Precision", "MemoryBreakdown", "NF4_DQ_EXTRA_BITS",
    "weights_memory", "lora_num_params", "qlora_weights_memory",
    "gradients_memory", "optimizer_memory", "activations_memory",
    "inference_activations_memory", "kv_cache_memory", "framework_overhead",
    "estimate_training", "estimate_inference",
    "MODELS", "ModelConfig", "get_model", "GPU_VRAM", "recommend",
    "__version__",
]
