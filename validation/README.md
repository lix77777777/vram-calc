# validation — 实测验证（Phase 3）

公式 vs `torch.cuda` 实测，误差目标 ≤ 10%。欢迎在你自己的 GPU 上复跑并提交结果。

## 环境

本机有卡：`pip install torch transformers peft bitsandbytes accelerate`
无卡用 [Colab](https://colab.research.google.com)（T4 免费）：上传整个仓库或 `git clone`，Runtime → T4 GPU。

## 推荐运行清单（小模型即可，T4 约 20 分钟）

```bash
cd validation
# 1) 权重 + KV-Cache（§1 §5）
python exp1_weights_kv.py --preset qwen2.5-0.5b --repo Qwen/Qwen2.5-0.5B --dtype fp16 --batch 1 --seq 2048
python exp1_weights_kv.py --preset qwen2.5-1.5b --repo Qwen/Qwen2.5-1.5B --dtype fp16 --batch 2 --seq 1024

# 2) 训练步：全量 / LoRA / QLoRA（§2 §3 §4）
python exp2_training.py --preset qwen2.5-0.5b --repo Qwen/Qwen2.5-0.5B --mode full --batch 2 --seq 1024
python exp2_training.py --preset qwen2.5-0.5b --repo Qwen/Qwen2.5-0.5B --mode lora --batch 2 --seq 1024
python exp2_training.py --preset qwen2.5-0.5b --repo Qwen/Qwen2.5-0.5B --mode qlora --batch 2 --seq 1024
# 开关对照（待验证-3/4/5）
python exp2_training.py --preset qwen2.5-0.5b --repo Qwen/Qwen2.5-0.5B --mode full --batch 2 --seq 1024 --ckpt
python exp2_training.py --preset qwen2.5-0.5b --repo Qwen/Qwen2.5-0.5B --mode full --batch 2 --seq 1024 --attn sdpa
python exp2_training.py --preset qwen2.5-0.5b --repo Qwen/Qwen2.5-0.5B --mode full --batch 2 --seq 1024 --accum 8

# 3) CUDA context（§6）
python exp3_overhead.py
```

## 出报告（无需 GPU）

```bash
python validation/analyze.py   # 仓库根目录运行，生成 docs/validation_report.md
```

结果 JSON 在 `validation/results/`，**请连同报告一起提交进 git**。

## 注意

- Colab T4 不支持 BF16 训练加速但能跑（慢）；flash_attention_2 需 Ampere+，T4 用 `--attn sdpa` 对照即可
- exp2 全量微调 0.5B 约需 12 GB 显存，T4(16G) 可跑；1.5B 全量在 T4 会 OOM，属预期（可写进报告）
- 误差 >10% 的项请连同 results/ 的 JSON 一起提 issue
