# vram-calc

> **Can my GPU run this model?** — An LLM VRAM calculator whose formulas were
> derived from first principles, then **validated against real GPU measurements**
> (30 checks, max error 4%, including out-of-sample).

[![CI](https://github.com/lix77777777/vram-calc/actions/workflows/ci.yml/badge.svg)](https://github.com/lix77777777/vram-calc/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**🔗 Live calculator: https://lix77777777.github.io/vram-calc/** (no install, works on mobile, 中英双语)

[中文 README](README_zh.md)

## Why another VRAM calculator?

Most calculators guess. We measured. Every formula in this project was tested
against `torch.cuda` on real hardware — and the first drafts were **wrong by
41–78%** in ways most blog posts never mention:

- **Quantized models are bigger than `params × 0.5 byte`** — bitsandbytes only
  quantizes Linear layers; embeddings stay BF16 (28% of a Qwen2.5-0.5B!).
- **The classic activation formula misses the logits/loss chain** — with a 152k
  vocab that's ~3·B·s·V bytes, bigger than several transformer layers.
- **peft creates LoRA adapters in FP32**, not BF16 — gradients too.
- **Plain-torch BF16 training is a third accounting regime** — AdamW m/v follow
  the param dtype (4 B/param), neither AMP (8) nor Megatron-style (12).

Full derivations in [docs/formulas.md](docs/formulas.md), measurement protocol
and results in [docs/validation_report.md](docs/validation_report.md).

## Quick start (Python)

```bash
pip install git+https://github.com/lix77777777/vram-calc.git
```

```python
from vram_calc import estimate_training, estimate_inference, get_model, recommend, GiB

m = get_model("llama-3-8b")

bd = estimate_inference(m, batch=1, seq=8192)
print(bd.table())              # weights / kv_cache / activations / overhead / total
print(recommend(bd.total))     # ['RTX 4080', 'T4', ..., 'H100 80GB']

bd = estimate_training(m, mode="qlora", batch=2, seq=2048,
                       attn_impl="sdpa", gradient_checkpointing=True)
print(f"{bd.total / GiB:.1f} GiB")
```

Zero runtime dependencies (torch is only used by validation scripts).

## Cheat sheet (per-parameter bytes)

| Scenario | Weights | Grads | Optimizer (AdamW) | Total static |
|---|---|---|---|---|
| Inference FP16 | 2 | — | — | **2** |
| Inference INT4 (NF4+DQ) | ~0.52 × linear + 2 × embed | — | — | |
| Train, AMP (FP32 params) | 4 | 4 | 8 | **16** |
| Train, Megatron/DeepSpeed BF16 | 2 | 2 | 12 (incl. FP32 master) | **16** |
| Train, plain-torch BF16 ✓measured | 2 | 2 | 4 | **8** |
| LoRA/QLoRA (peft, FP32 adapters) | base + 4×P_lora | 4×P_lora | 8×P_lora | |

Plus activations `L·s·B·h·(31 + 8f/h [+ 6as/h if eager]) + 3·B·s·V` and
KV-cache `2·L·n_kv·d_head·s·B·b` — see [docs/formulas.md](docs/formulas.md) §4–5.

## What's inside

- `src/vram_calc/` — pure-Python library (11 preset models, params computed
  from architecture and cross-checked against HF: exact match on Qwen2.5-0.5B)
- `web/` — static web app, same math in JS, **105 test cases × 7 fields agree
  with Python to <1e-9** (enforced in CI)
- `validation/` — the PyTorch measurement scripts; run them yourself on any GPU
- `docs/` — formula derivations, validation report, and a from-zero study guide (中文)

## Scope & roadmap

v1: single-GPU, Llama-family decoder-only (MHA/GQA/MQA), FP32/16/BF16/INT8/INT4,
full/LoRA/QLoRA + inference. Calibration done on transformers 4.55 / torch 2.7.
**v1.5**: MLA (DeepSeek). **v2**: multi-GPU (DeepSpeed/FSDP), throughput estimates.

Numbers are estimates — target accuracy ±10%. Found a config where we're off by
more? Please open an issue with your `validation/` JSON, that's exactly how this
project improves.

## License

MIT © 2026 Lee
