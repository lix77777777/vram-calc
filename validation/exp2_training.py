"""实验 2: 训练步显存拆解（formulas.md §2 §3 §4；待验证-2/3/4/5/9）.

测量顺序（每步打快照, 差分得各块）:
  载入模型 → 权重 | 前向 → 激活值 | 反向 → 梯度 | optimizer.step → 优化器状态

用法示例:
    python exp2_training.py --preset qwen2.5-0.5b --repo Qwen/Qwen2.5-0.5B \
        --mode full --batch 2 --seq 1024
    python exp2_training.py ... --mode lora --lora-rank 16
    python exp2_training.py ... --mode qlora            # 需 bitsandbytes
    python exp2_training.py ... --ckpt                  # gradient checkpointing
    python exp2_training.py ... --attn eager|sdpa|flash_attention_2   # 待验证-5
    python exp2_training.py ... --accum 8               # 待验证-3
"""

import argparse

import torch
from transformers import AutoModelForCausalLM

from utils import fmt, require_cuda, reset_peak, save_result, snapshot


def build_model(args):
    kw = {"torch_dtype": torch.bfloat16, "attn_implementation": args.attn}
    if args.mode == "qlora":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(args.repo, **kw)
    if args.mode != "qlora":
        model = model.cuda()
    if args.mode in ("lora", "qlora"):
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank,
                          target_modules="all-linear", task_type="CAUSAL_LM")
        model = get_peft_model(model, lcfg)
        model.print_trainable_parameters()
    if args.ckpt:
        model.gradient_checkpointing_enable()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--mode", default="full", choices=["full", "lora", "qlora"])
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--attn", default="eager",
                    choices=["eager", "sdpa", "flash_attention_2"])
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--accum", type=int, default=1, help=">1 时观察梯度累积是否有额外副本")
    args = ap.parse_args()
    require_cuda()

    reset_peak()
    model = build_model(args)
    model.train()
    m_weights = snapshot()["allocated"]
    print("权重(+adapter):", fmt(m_weights))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    ids = torch.randint(100, 1000, (args.batch, args.seq), device="cuda")

    records = []
    for step in range(max(2, args.accum)):       # 第 2 步起优化器状态已建, 数字稳定
        reset_peak()
        before_fwd = snapshot()["allocated"]
        out = model(ids, labels=ids)
        after_fwd = snapshot()["allocated"]
        (out.loss / args.accum).backward()
        after_bwd = snapshot()["allocated"]
        if (step + 1) % args.accum == 0:
            opt.step()
            opt.zero_grad(set_to_none=False)     # 保留梯度缓冲便于测量
        after_step = snapshot()
        records.append({
            "step": step,
            "activations": after_fwd - before_fwd,
            "grad_plus_state_delta": after_step["allocated"] - after_bwd,
            "allocated": after_step["allocated"],
            "reserved": after_step["reserved"],          # 待验证-9: 碎片系数
            "peak": after_step["max_allocated"],
        })
        print(f"step{step}: 激活 {fmt(records[-1]['activations'])} "
              f"| 峰值 {fmt(records[-1]['peak'])} "
              f"| reserved/allocated "
              f"{records[-1]['reserved'] / max(records[-1]['allocated'], 1):.3f}")

    grad_dtype = next(p.grad.dtype for p in model.parameters()
                      if p.requires_grad and p.grad is not None)
    print("梯度 dtype（待验证-2）:", grad_dtype)

    tag = f"exp2_{args.preset}_{args.mode}_attn-{args.attn}" \
          f"{'_ckpt' if args.ckpt else ''}_b{args.batch}_s{args.seq}_acc{args.accum}"
    save_result(tag, {
        "preset": args.preset, "mode": args.mode, "attn": args.attn,
        "ckpt": args.ckpt, "batch": args.batch, "seq": args.seq,
        "accum": args.accum, "lora_rank": args.lora_rank,
        "weights_measured": m_weights,
        "grad_dtype": str(grad_dtype),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "steps": records,
    })


if __name__ == "__main__":
    main()
