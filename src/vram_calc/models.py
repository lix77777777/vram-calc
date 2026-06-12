"""预置模型配置.

字段抓自 HuggingFace config.json（2026-06-10 查证，来源 URL 见各条目）。
参数量由结构推算: qwen2.5-0.5b 推算值与 HF 实载完全一致 (494,032,768, Phase 3 实测).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModelConfig:
    """Decoder-only Transformer（Llama 系结构）配置."""

    name: str
    hidden_size: int          # h
    num_layers: int           # L
    num_heads: int            # a
    num_kv_heads: int         # n_kv (MHA: =a, GQA: <a, MQA: 1)
    intermediate_size: int    # f (SwiGLU: gate/up/down 三个矩阵)
    vocab_size: int
    tie_word_embeddings: bool = False
    qkv_bias: bool = False    # Qwen2 系为 True
    head_dim: Optional[int] = None  # 缺省 h / a
    source: str = ""          # config.json 来源

    def __post_init__(self) -> None:
        for f_ in ("hidden_size", "num_layers", "num_heads",
                   "num_kv_heads", "intermediate_size", "vocab_size"):
            if getattr(self, f_) <= 0:
                raise ValueError(f"{f_} must be positive")
        if self.num_kv_heads > self.num_heads:
            raise ValueError("num_kv_heads cannot exceed num_heads")

    @property
    def head_dim_(self) -> int:
        return self.head_dim if self.head_dim is not None else self.hidden_size // self.num_heads

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim_

    @property
    def embed_params(self) -> int:
        """embedding (+ 未绑定时的 lm_head). 量化时保持半精度（Phase 3 实测确认）."""
        return self.vocab_size * self.hidden_size * (1 if self.tie_word_embeddings else 2)

    @property
    def linear_params(self) -> int:
        """全部 Linear 层参数（q/k/v/o + gate/up/down + bias），即可被量化的部分."""
        h, f = self.hidden_size, self.intermediate_size
        q_dim = self.num_heads * self.head_dim_
        attn = h * q_dim + 2 * h * self.kv_dim + q_dim * h
        if self.qkv_bias:
            attn += q_dim + 2 * self.kv_dim
        return self.num_layers * (attn + 3 * h * f)

    @property
    def num_params(self) -> int:
        """总参数量 = embedding + Linear + RMSNorm(每层 2 个 + final)."""
        norms = self.num_layers * 2 * self.hidden_size + self.hidden_size
        return self.embed_params + self.linear_params + norms


def _hf(repo: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/config.json"


MODELS: dict[str, ModelConfig] = {m.name: m for m in [
    ModelConfig("llama-2-7b", 4096, 32, 32, 32, 11008, 32000,
                source=_hf("NousResearch/Llama-2-7b-hf")),
    ModelConfig("llama-3-8b", 4096, 32, 32, 8, 14336, 128256,
                source=_hf("NousResearch/Meta-Llama-3-8B")),
    ModelConfig("mistral-7b-v0.1", 4096, 32, 32, 8, 14336, 32000,
                source=_hf("mistralai/Mistral-7B-v0.1")),
    ModelConfig("qwen2.5-0.5b", 896, 24, 14, 2, 4864, 151936,
                tie_word_embeddings=True, qkv_bias=True,
                source=_hf("Qwen/Qwen2.5-0.5B")),
    ModelConfig("qwen2.5-1.5b", 1536, 28, 12, 2, 8960, 151936,
                tie_word_embeddings=True, qkv_bias=True,
                source=_hf("Qwen/Qwen2.5-1.5B")),
    ModelConfig("qwen2.5-7b", 3584, 28, 28, 4, 18944, 152064,
                qkv_bias=True, source=_hf("Qwen/Qwen2.5-7B-Instruct")),
    ModelConfig("qwen2.5-14b", 5120, 48, 40, 8, 13824, 152064,
                qkv_bias=True, source=_hf("Qwen/Qwen2.5-14B-Instruct")),
    ModelConfig("qwen3-8b", 4096, 36, 32, 8, 12288, 151936,
                head_dim=128, source=_hf("Qwen/Qwen3-8B")),
    ModelConfig("deepseek-llm-7b", 4096, 30, 32, 32, 11008, 102400,
                source=_hf("deepseek-ai/deepseek-llm-7b-base")),
    ModelConfig("tinyllama-1.1b", 2048, 22, 32, 4, 5632, 32000,
                source=_hf("TinyLlama/TinyLlama-1.1B-Chat-v1.0")),
    ModelConfig("yi-6b", 4096, 32, 32, 4, 11008, 64000,
                source=_hf("01-ai/Yi-6B")),
]}


def get_model(name: str) -> ModelConfig:
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODELS)}") from None
