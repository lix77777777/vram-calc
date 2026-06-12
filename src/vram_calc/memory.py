"""五大块显存计算核心. 规格以 docs/formulas.md 为准.

激活值/推理工作集的系数为 Phase 3 实测标定值（Qwen2.5-0.5B, RTX 5060 Ti,
transformers 4.55 + torch 2.7, 2026-06-10），跨模型外推待进一步验证.
所有函数返回字节数 (float).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import ModelConfig

GiB = 1024 ** 3
MiB = 1024 ** 2


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


BYTES_PER_PARAM: dict[Precision, float] = {
    Precision.FP32: 4.0,
    Precision.FP16: 2.0,
    Precision.BF16: 2.0,
    Precision.INT8: 1.0,
    Precision.INT4: 0.5,
}

# 量化常数开销 (bit/参数), formulas.md §1
NF4_DQ_EXTRA_BITS = 0.127       # bitsandbytes NF4 + double quantization
GPTQ_G128_EXTRA_BITS = 0.25     # 〔待验证-1〕

# ---- Phase 3 实测标定常数（formulas.md §4, 标定来源见模块 docstring）----
ACT_C_BASE = 31.0        # 每层激活非 MLP 部分 (字节/token/h)
ACT_C_MLP = 8.0          # MLP 部分系数, 乘 f/h
ACT_K_EAGER = 6.0        # eager 注意力 s² 项 (transformers FP32 softmax)
ACT_K_LOGITS = 3.0       # logits + cross-entropy 链 (字节/token/vocab)
LORA_ACT_OVERHEAD = 0.13  # peft LoRA 额外激活 (+13%)
KV_BUFFER = 10 * MiB      # 推理常驻缓冲 (rotary/mask 等, 实测 ~9.2 MiB)


def _check_nonneg(**kw: float) -> None:
    for k, v in kw.items():
        if v < 0:
            raise ValueError(f"{k} must be >= 0, got {v}")


# ---------------------------------------------------------------- §1 权重
def weights_memory(num_params: int, precision: Precision,
                   quant_extra_bits: float = 0.0) -> float:
    """M_weights = P × b + P × quant_extra_bits/8."""
    _check_nonneg(num_params=num_params, quant_extra_bits=quant_extra_bits)
    return num_params * (BYTES_PER_PARAM[precision] + quant_extra_bits / 8.0)


def lora_num_params(cfg: ModelConfig, rank: int, targets: str = "all-linear") -> int:
    """LoRA adapter 参数量. 与 peft 实载完全一致 (qwen2.5-0.5b r16: 8,798,208 ✓实测).

    targets: "all-linear"(q/k/v/o + gate/up/down) 或 "attention"(仅 q/k/v/o).
    """
    _check_nonneg(rank=rank)
    h, f = cfg.hidden_size, cfg.intermediate_size
    q_dim, kv_dim = cfg.num_heads * cfg.head_dim_, cfg.kv_dim
    attn = rank * ((h + q_dim) + 2 * (h + kv_dim) + (q_dim + h))
    if targets == "attention":
        per_layer = attn
    elif targets == "all-linear":
        per_layer = attn + rank * 3 * (h + f)
    else:
        raise ValueError(f"unknown targets {targets!r}")
    return cfg.num_layers * per_layer


def qlora_weights_memory(cfg: ModelConfig, lora_rank: int = 16,
                         lora_targets: str = "all-linear",
                         quant_extra_bits: float = NF4_DQ_EXTRA_BITS) -> float:
    """QLoRA 权重: bitsandbytes 只量化 Linear 层（Phase 3 实测确认）.

    = embedding(BF16) + Linear(INT4+量化常数) + Norm(BF16) + adapter(FP32, peft 默认)
    实测对照 qwen2.5-0.5b: 误差 -0.2% ✓
    """
    emb = cfg.embed_params * 2.0
    lin = cfg.linear_params * (0.5 + quant_extra_bits / 8.0)
    other = (cfg.num_params - cfg.embed_params - cfg.linear_params) * 2.0
    adapters = lora_num_params(cfg, lora_rank, lora_targets) * 4.0
    return emb + lin + other + adapters


# ---------------------------------------------------------------- §2 梯度
def gradients_memory(num_trainable: int, grad_precision: Precision) -> float:
    """M_grad = P_t × b_grad. 梯度 dtype 跟随参数本体 dtype（Phase 3 实测确认）:
    纯 BF16 训练 → 2 字节; peft LoRA adapter 为 FP32 → 4 字节."""
    _check_nonneg(num_trainable=num_trainable)
    return num_trainable * BYTES_PER_PARAM[grad_precision]


# ---------------------------------------------------------------- §3 优化器
_OPTIMIZER_STATE_SLOTS = {
    "adamw": 2.0,        # m + v
    "adamw_8bit": 0.5,   # m + v 各 1 字节 (按 fp32 槽位折算)
    "sgd": 0.0,
    "sgd_momentum": 1.0,
}


def optimizer_memory(num_trainable: int, optimizer: str = "adamw",
                     master_weights: bool = False,
                     state_precision: Precision = Precision.FP32) -> float:
    """M_opt = P_t × (槽位数 × 状态字节 + master×4).

    torch AdamW 的 m/v dtype 跟随参数（Phase 3 实测: BF16 参数 → 4 B/参数 ✓）;
    Megatron/DeepSpeed 式混精: state_precision=FP32, master_weights=True → K=12.
    """
    _check_nonneg(num_trainable=num_trainable)
    if optimizer not in _OPTIMIZER_STATE_SLOTS:
        raise ValueError(f"unknown optimizer {optimizer!r}; "
                         f"available: {sorted(_OPTIMIZER_STATE_SLOTS)}")
    if optimizer == "adamw_8bit":
        state = 2.0      # m+v 各 1 字节, 不随 state_precision 缩放
    else:
        state = _OPTIMIZER_STATE_SLOTS[optimizer] * BYTES_PER_PARAM[state_precision]
    return num_trainable * (state + (4.0 if master_weights else 0.0))


# ---------------------------------------------------------------- §4 激活值
def activations_memory(cfg: ModelConfig, batch: int, seq: int, *,
                       attn_impl: str = "eager",        # eager | sdpa | flash
                       gradient_checkpointing: bool = False,
                       include_logits: bool = True,
                       lora: bool = False) -> float:
    """训练激活值 (BF16), Phase 3 实测标定公式:

        M = L·s·B·h·(C_base + C_mlp·f/h + K_eager·a·s/h) + K_logits·B·s·V
    ckpt 时层内只存输入(2sBh/层) + 单层峰值. logits 项对大词表小模型占大头.
    标定后对照实测误差 ≤ 2.5%（eager/sdpa/ckpt/lora 四种配置）.
    """
    _check_nonneg(batch=batch, seq=seq)
    if batch == 0 or seq == 0:
        return 0.0
    if attn_impl not in ("eager", "sdpa", "flash"):
        raise ValueError(f"unknown attn_impl {attn_impl!r}")
    s, B, h, a, L = seq, batch, cfg.hidden_size, cfg.num_heads, cfg.num_layers
    coef = ACT_C_BASE + ACT_C_MLP * cfg.intermediate_size / h
    if attn_impl == "eager":
        coef += ACT_K_EAGER * a * s / h
    per_layer = s * B * h * coef
    logits = ACT_K_LOGITS * B * s * cfg.vocab_size if include_logits else 0.0
    if gradient_checkpointing:
        act = 2.0 * L * s * B * h + per_layer + logits
    else:
        act = L * per_layer + logits
    return act * (1.0 + LORA_ACT_OVERHEAD) if lora else act


def inference_activations_memory(cfg: ModelConfig, batch: int, seq: int) -> float:
    """推理 prefill 峰值工作集 ≈ logits(2·B·s·V) + 隐层(4·s·B·h·2).

    实测对照: 0.5b -3.7%, 1.5b -7.2% ✓. 逐 token 解码阶段远小于此.
    """
    _check_nonneg(batch=batch, seq=seq)
    if batch == 0 or seq == 0:
        return 0.0
    return 2.0 * batch * seq * cfg.vocab_size + 4.0 * seq * batch * cfg.hidden_size * 2.0


# ---------------------------------------------------------------- §5 KV-Cache
def kv_cache_memory(cfg: ModelConfig, batch: int, seq: int,
                    precision: Precision = Precision.FP16) -> float:
    """M_kv = 2 × L × n_kv × d_head × s × B × b. 公式实测验证 ✓
    （实测另有 ~10 MiB 常驻缓冲, 计在 KV_BUFFER, 由 estimate_inference 加入）."""
    _check_nonneg(batch=batch, seq=seq)
    return (2.0 * cfg.num_layers * cfg.num_kv_heads * cfg.head_dim_
            * seq * batch * BYTES_PER_PARAM[precision])


# ---------------------------------------------------------------- §6 框架开销
def framework_overhead(subtotal: float, base: float = 0.75 * GiB,
                       fraction: float = 0.08) -> float:
    """经验项: CUDA context(base, 待 exp3 实测) + 分配器碎片(fraction×小计).

    碎片系数实测 reserved/peak = 1.05~1.11, 取 8% 居中 〔待验证-8: base〕.
    """
    _check_nonneg(subtotal=subtotal)
    return base + fraction * subtotal


# ------------------------------------------------- GGUF (llama.cpp / Ollama)
# 平均 bit/权重(整文件口径, 已含块内 scale 与高位宽层的混合), 取自 llama.cpp
# 文档与社区整理的近似值〔待验证-10: 未上卡实测, 与官方 GGUF 文件大小对照偏差 ~1-3%〕
GGUF_BPW = {
    "Q2_K": 2.63, "Q3_K_M": 3.91, "Q4_0": 4.55, "Q4_K_S": 4.58,
    "Q4_K_M": 4.85, "Q5_K_M": 5.69, "Q6_K": 6.59, "Q8_0": 8.50, "F16": 16.0,
}
GGUF_GRAPH_BASE = 0.40 * GiB   # llama.cpp 计算图/上下文缓冲经验值〔待验证-10〕


def gguf_weights_memory(cfg: ModelConfig, quant: str = "Q4_K_M") -> float:
    """GGUF 权重显存 = P × bpw / 8（全量 offload 到 GPU 时 ≈ 文件大小）.

    sanity: Llama-3-8B Q4_K_M 估 4.53 GiB vs 官方文件 ≈4.58 GiB (-1%).
    """
    if quant not in GGUF_BPW:
        raise ValueError(f"unknown gguf quant {quant!r}; available: {sorted(GGUF_BPW)}")
    return cfg.num_params * GGUF_BPW[quant] / 8.0


def estimate_gguf(cfg: ModelConfig, *, quant: str = "Q4_K_M",
                  ctx: int = 4096, batch: int = 1,
                  kv_precision: Precision = Precision.FP16) -> MemoryBreakdown:
    """llama.cpp / Ollama 全量 GPU offload 推理估算〔整条路径待实测, 见 formulas.md §8-10〕.

    = 权重(bpw) + KV-Cache(缺省 f16) + 计算图缓冲(经验)
    注意: 与 HF transformers 路径不同, llama.cpp 无 CUDA context 之外的大头开销,
    这里不再乘碎片系数.
    """
    w = gguf_weights_memory(cfg, quant)
    kv = kv_cache_memory(cfg, batch, ctx, kv_precision) + (KV_BUFFER if batch and ctx else 0.0)
    graph = GGUF_GRAPH_BASE + 0.05 * kv   # 随上下文小幅增长, 经验项
    return MemoryBreakdown(weights=w, kv_cache=kv, overhead=graph)


# ---------------------------------------------------------------- §7 汇总
@dataclass(frozen=True)
class MemoryBreakdown:
    weights: float
    gradients: float = 0.0
    optimizer: float = 0.0
    activations: float = 0.0
    kv_cache: float = 0.0
    overhead: float = 0.0

    @property
    def total(self) -> float:
        return (self.weights + self.gradients + self.optimizer
                + self.activations + self.kv_cache + self.overhead)

    def as_dict(self) -> dict[str, float]:
        return {"weights": self.weights, "gradients": self.gradients,
                "optimizer": self.optimizer, "activations": self.activations,
                "kv_cache": self.kv_cache, "overhead": self.overhead,
                "total": self.total}

    def table(self) -> str:
        return "\n".join(f"{k:<12}{v / GiB:>10.2f} GiB" for k, v in self.as_dict().items())


def estimate_training(cfg: ModelConfig, *,
                      mode: str = "full",                # full | lora | qlora
                      mixed_precision: str = "bf16_pure",  # amp | megatron | bf16_pure
                      optimizer: str = "adamw",
                      batch: int = 1, seq: int = 2048,
                      lora_rank: int = 16, lora_targets: str = "all-linear",
                      attn_impl: str = "eager",
                      gradient_checkpointing: bool = False,
                      include_overhead: bool = True) -> MemoryBreakdown:
    """训练显存汇总. 三种全量记账（Phase 3 实测确认 bf16_pure）:

    - amp:       参数/梯度 FP32(4+4), 优化器 FP32 无 master (8)
    - megatron:  参数/梯度 BF16(2+2), 优化器 FP32 + master (12)  —— DeepSpeed/Megatron
    - bf16_pure: 参数/梯度 BF16(2+2), 优化器状态也 BF16 (4)      —— 裸 torch AdamW ✓实测
    LoRA/QLoRA: adapter 为 FP32（peft 默认）: 梯度 4B, 优化器 8B, 无 master.
    """
    P = cfg.num_params
    if mode == "full":
        trainable = P
        if mixed_precision == "amp":
            w = weights_memory(P, Precision.FP32)
            g = gradients_memory(trainable, Precision.FP32)
            o = optimizer_memory(trainable, optimizer)
        elif mixed_precision == "megatron":
            w = weights_memory(P, Precision.BF16)
            g = gradients_memory(trainable, Precision.BF16)
            o = optimizer_memory(trainable, optimizer, master_weights=True)
        elif mixed_precision == "bf16_pure":
            w = weights_memory(P, Precision.BF16)
            g = gradients_memory(trainable, Precision.BF16)
            o = optimizer_memory(trainable, optimizer, state_precision=Precision.BF16)
        else:
            raise ValueError(f"unknown mixed_precision {mixed_precision!r}")
    elif mode in ("lora", "qlora"):
        trainable = lora_num_params(cfg, lora_rank, lora_targets)
        if mode == "qlora":
            w = qlora_weights_memory(cfg, lora_rank, lora_targets)
        else:
            w = weights_memory(P, Precision.BF16) + trainable * 4.0  # adapter FP32
        g = gradients_memory(trainable, Precision.FP32)
        o = optimizer_memory(trainable, optimizer)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    act = activations_memory(cfg, batch, seq, attn_impl=attn_impl,
                             gradient_checkpointing=gradient_checkpointing,
                             lora=mode in ("lora", "qlora"))
    subtotal = w + g + o + act
    oh = framework_overhead(subtotal) if include_overhead else 0.0
    return MemoryBreakdown(weights=w, gradients=g, optimizer=o,
                           activations=act, overhead=oh)


def estimate_inference(cfg: ModelConfig, *,
                       precision: Precision = Precision.FP16,
                       batch: int = 1, seq: int = 2048,
                       quant_extra_bits: float = 0.0,
                       include_overhead: bool = True) -> MemoryBreakdown:
    """推理显存汇总: 权重 + KV-Cache(+常驻缓冲) + prefill 工作集 + 开销."""
    w = weights_memory(cfg.num_params, precision, quant_extra_bits)
    kv_prec = precision if precision in (Precision.FP16, Precision.BF16,
                                         Precision.FP32) else Precision.FP16
    kv = kv_cache_memory(cfg, batch, seq, kv_prec) + (KV_BUFFER if batch and seq else 0.0)
    act = inference_activations_memory(cfg, batch, seq)
    subtotal = w + kv + act
    oh = framework_overhead(subtotal) if include_overhead else 0.0
    return MemoryBreakdown(weights=w, activations=act, kv_cache=kv, overhead=oh)
