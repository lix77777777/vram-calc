"""实验 1: 权重显存 + KV-Cache 实测（formulas.md §1 §5；待验证-6 推理工作集）.

用法:
    python exp1_weights_kv.py --preset qwen2.5-0.5b --repo Qwen/Qwen2.5-0.5B \
        --dtype fp16 --batch 1 --seq 2048
"""

import argparse

import torch
from transformers import AutoModelForCausalLM

from utils import fmt, require_cuda, reset_peak, save_result, snapshot

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True, help="vram_calc 预置名, 如 qwen2.5-0.5b")
    ap.add_argument("--repo", required=True, help="HF repo, 如 Qwen/Qwen2.5-0.5B")
    ap.add_argument("--dtype", default="fp16", choices=DTYPES)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=2048)
    args = ap.parse_args()
    require_cuda()

    reset_peak()
    base = snapshot()["allocated"]

    model = AutoModelForCausalLM.from_pretrained(
        args.repo, torch_dtype=DTYPES[args.dtype]).cuda().eval()
    after_load = snapshot()["allocated"]
    weights_measured = after_load - base
    print("权重实测:", fmt(weights_measured))

    # prefill 一段 seq, KV-Cache = prefill 后稳定增量
    ids = torch.randint(100, 1000, (args.batch, args.seq), device="cuda")
    reset_peak()
    with torch.no_grad():
        out = model(ids, use_cache=True)
    after_prefill = snapshot()
    # 保留 past_key_values 引用, 释放 logits 等
    pkv = out.past_key_values
    del out
    torch.cuda.empty_cache()
    kv_measured = snapshot()["allocated"] - after_load
    peak_work = after_prefill["max_allocated"] - after_load  # 含 KV+激活工作集+logits
    print("KV-Cache 实测:", fmt(kv_measured), "| prefill 峰值工作集:", fmt(peak_work))

    save_result(f"exp1_{args.preset}_{args.dtype}_b{args.batch}_s{args.seq}", {
        "preset": args.preset, "dtype": args.dtype,
        "batch": args.batch, "seq": args.seq,
        "weights_measured": weights_measured,
        "kv_measured": kv_measured,
        "prefill_peak_workset": peak_work,
        "num_params_hf": sum(p.numel() for p in model.parameters()),
    })
    assert pkv is not None


if __name__ == "__main__":
    main()
