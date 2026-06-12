"""用 Python 库生成 JS 一致性测试用例 → web/test_cases.json.

运行: 仓库根目录 PYTHONPATH=src python tests/gen_web_cases.py
JS 实现必须对每个用例输出逐字段一致的结果(相对误差 < 1e-9).
"""

import itertools
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vram_calc import MODELS, Precision, estimate_inference, estimate_training  # noqa: E402

cases = []
models = ["llama-2-7b", "llama-3-8b", "qwen2.5-0.5b", "qwen2.5-7b", "tinyllama-1.1b"]

# 推理
for name, prec, (b, s) in itertools.product(
        models, ["fp16", "int8", "int4"], [(1, 2048), (4, 8192), (0, 2048)]):
    bd = estimate_inference(MODELS[name], precision=Precision(prec), batch=b, seq=s)
    cases.append({"kind": "inference", "model": name, "precision": prec,
                  "batch": b, "seq": s, "expect": bd.as_dict()})

# 训练
for name, mode, mp, attn, ckpt in itertools.product(
        models[:3], ["full", "lora", "qlora"], ["bf16_pure", "megatron", "amp"],
        ["eager", "sdpa"], [False, True]):
    if mode != "full" and mp != "bf16_pure":
        continue  # lora/qlora 与 mixed_precision 无关, 只生成一份
    bd = estimate_training(MODELS[name], mode=mode, mixed_precision=mp,
                           batch=2, seq=1024, attn_impl=attn,
                           gradient_checkpointing=ckpt)
    cases.append({"kind": "training", "model": name, "mode": mode,
                  "mixed_precision": mp, "attn_impl": attn, "ckpt": ckpt,
                  "batch": 2, "seq": 1024, "lora_rank": 16, "expect": bd.as_dict()})

out = ROOT / "web" / "test_cases.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(cases, indent=1))
print(f"[saved] {out} ({len(cases)} cases)")
