"""比对实测 JSON 与 vram_calc 预测, 生成 docs/validation_report.md.

用法（仓库根目录, 无需 GPU）: python validation/analyze.py
误差目标 ≤ 10%（BRIEF §4）. v2: 使用 Phase 3 标定后的公式.
"""

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from vram_calc import (GiB, Precision, activations_memory, get_model,  # noqa: E402
                       kv_cache_memory, lora_num_params,
                       qlora_weights_memory, weights_memory)
from vram_calc.memory import KV_BUFFER  # noqa: E402

RESULTS = pathlib.Path(__file__).parent / "results"
PREC = {"fp32": Precision.FP32, "fp16": Precision.FP16, "bf16": Precision.BF16}


def err(pred: float, meas: float) -> str:
    if meas <= 0:
        return "n/a"
    e = (pred - meas) / meas * 100
    return f"{e:+.1f}%" + (" ⚠️" if abs(e) > 10 else " ✅")


def row(*cells) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


def main() -> None:
    files = sorted(RESULTS.glob("*.json"))
    if not files:
        sys.exit("results/ 下没有 JSON, 先跑 exp1/exp2/exp3")

    lines = ["# 实测对照报告（Phase 3, 标定后公式）", "",
             "误差 = (预测 − 实测) / 实测，目标 |误差| ≤ 10%。",
             "激活值系数标定于 qwen2.5-0.5b eager/sdpa/ckpt 三组（同批数据，非独立验证），",
             "跨模型外推见「待跨模型验证」标注。", "",
             row("实验", "项目", "预测 GiB", "实测 GiB", "误差"),
             row(*["---"] * 5)]

    for fp in files:
        d = json.loads(fp.read_text())
        name = d["experiment"]
        cfg = get_model(d["preset"]) if "preset" in d else None
        if name.startswith("exp1"):
            p = PREC[d["dtype"]]
            wp = weights_memory(cfg.num_params, p)
            kp = kv_cache_memory(cfg, d["batch"], d["seq"], p) + KV_BUFFER
            lines.append(row(name, "权重", f"{wp/GiB:.3f}",
                             f"{d['weights_measured']/GiB:.3f}", err(wp, d["weights_measured"])))
            lines.append(row(name, "KV-Cache(+缓冲)", f"{kp/GiB:.4f}",
                             f"{d['kv_measured']/GiB:.4f}", err(kp, d["kv_measured"])))
        elif name.startswith("exp2"):
            lora = d["mode"] in ("lora", "qlora")
            if d["mode"] == "full":
                wp = weights_memory(cfg.num_params, Precision.BF16)
            elif d["mode"] == "lora":
                wp = weights_memory(cfg.num_params, Precision.BF16) \
                    + lora_num_params(cfg, d["lora_rank"]) * 4.0
            else:
                wp = qlora_weights_memory(cfg, d["lora_rank"])
            lines.append(row(name, "权重", f"{wp/GiB:.3f}",
                             f"{d['weights_measured']/GiB:.3f}", err(wp, d["weights_measured"])))
            attn = "eager" if d["attn"] == "eager" else "sdpa"
            ap = activations_memory(cfg, d["batch"], d["seq"], attn_impl=attn,
                                    gradient_checkpointing=d["ckpt"], lora=lora)
            am = d["steps"][-1]["activations"]
            lines.append(row(name, "激活值", f"{ap/GiB:.3f}", f"{am/GiB:.3f}", err(ap, am)))
            s = d["steps"][-1]
            lines.append(row(name, "reserved/peak（碎片系数）", "1.05~1.15(经验)",
                             f"{s['reserved'] / max(s['peak'], 1):.3f}", "-"))
            lines.append(row(name, "梯度 dtype", "-", d["grad_dtype"], "-"))
        elif name.startswith("exp3"):
            ctx = d.get("cuda_context", -1)
            lines.append(row(name, "CUDA context", "0.75 GiB(经验)",
                             f"{ctx/GiB:.3f}" if ctx >= 0 else "N/A", "-"))

    out = ROOT / "docs" / "validation_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[saved] {out}\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
