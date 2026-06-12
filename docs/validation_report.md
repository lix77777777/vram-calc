# 实测对照报告（Phase 3, 标定后公式）

误差 = (预测 − 实测) / 实测，目标 |误差| ≤ 10%。
激活值系数标定于 qwen2.5-0.5b eager/sdpa/ckpt 三组（同批数据，非独立验证），
跨模型外推见「待跨模型验证」标注。

| 实验 | 项目 | 预测 GiB | 实测 GiB | 误差 |
| --- | --- | --- | --- | --- |
| exp1_qwen2.5-0.5b_fp16_b1_s2048 | 权重 | 0.920 | 0.928 | -0.8% ✅ |
| exp1_qwen2.5-0.5b_fp16_b1_s2048 | KV-Cache(+缓冲) | 0.0332 | 0.0324 | +2.6% ✅ |
| exp1_qwen2.5-1.5b_fp16_b2_s1024 | 权重 | 2.875 | 2.876 | -0.0% ✅ |
| exp1_qwen2.5-1.5b_fp16_b2_s1024 | KV-Cache(+缓冲) | 0.0645 | 0.0636 | +1.3% ✅ |
| exp2_qwen2.5-0.5b_full_attn-eager_b2_s1024_acc1 | 权重 | 0.920 | 0.928 | -0.8% ✅ |
| exp2_qwen2.5-0.5b_full_attn-eager_b2_s1024_acc1 | 激活值 | 7.860 | 7.892 | -0.4% ✅ |
| exp2_qwen2.5-0.5b_full_attn-eager_b2_s1024_acc1 | reserved/peak（碎片系数） | 1.05~1.15(经验) | 1.054 | - |
| exp2_qwen2.5-0.5b_full_attn-eager_b2_s1024_acc1 | 梯度 dtype | - | torch.bfloat16 | - |
| exp2_qwen2.5-0.5b_full_attn-eager_b2_s1024_acc8 | 权重 | 0.920 | 0.928 | -0.8% ✅ |
| exp2_qwen2.5-0.5b_full_attn-eager_b2_s1024_acc8 | 激活值 | 7.860 | 7.896 | -0.5% ✅ |
| exp2_qwen2.5-0.5b_full_attn-eager_b2_s1024_acc8 | reserved/peak（碎片系数） | 1.05~1.15(经验) | 1.059 | - |
| exp2_qwen2.5-0.5b_full_attn-eager_b2_s1024_acc8 | 梯度 dtype | - | torch.bfloat16 | - |
| exp2_qwen2.5-0.5b_full_attn-eager_ckpt_b2_s1024_acc1 | 权重 | 0.920 | 0.928 | -0.8% ✅ |
| exp2_qwen2.5-0.5b_full_attn-eager_ckpt_b2_s1024_acc1 | 激活值 | 1.243 | 1.258 | -1.2% ✅ |
| exp2_qwen2.5-0.5b_full_attn-eager_ckpt_b2_s1024_acc1 | reserved/peak（碎片系数） | 1.05~1.15(经验) | 1.112 | - |
| exp2_qwen2.5-0.5b_full_attn-eager_ckpt_b2_s1024_acc1 | 梯度 dtype | - | torch.bfloat16 | - |
| exp2_qwen2.5-0.5b_full_attn-sdpa_b2_s1024_acc1 | 权重 | 0.920 | 0.928 | -0.8% ✅ |
| exp2_qwen2.5-0.5b_full_attn-sdpa_b2_s1024_acc1 | 激活值 | 3.922 | 4.023 | -2.5% ✅ |
| exp2_qwen2.5-0.5b_full_attn-sdpa_b2_s1024_acc1 | reserved/peak（碎片系数） | 1.05~1.15(经验) | 1.081 | - |
| exp2_qwen2.5-0.5b_full_attn-sdpa_b2_s1024_acc1 | 梯度 dtype | - | torch.bfloat16 | - |
| exp2_qwen2.5-0.5b_lora_attn-eager_b2_s1024_acc1 | 权重 | 0.953 | 0.961 | -0.8% ✅ |
| exp2_qwen2.5-0.5b_lora_attn-eager_b2_s1024_acc1 | 激活值 | 8.881 | 8.933 | -0.6% ✅ |
| exp2_qwen2.5-0.5b_lora_attn-eager_b2_s1024_acc1 | reserved/peak（碎片系数） | 1.05~1.15(经验) | 1.060 | - |
| exp2_qwen2.5-0.5b_lora_attn-eager_b2_s1024_acc1 | 梯度 dtype | - | torch.float32 | - |
| exp2_qwen2.5-0.5b_qlora_attn-eager_b2_s1024_acc1 | 权重 | 0.458 | 0.459 | -0.2% ✅ |
| exp2_qwen2.5-0.5b_qlora_attn-eager_b2_s1024_acc1 | 激活值 | 8.881 | 8.921 | -0.4% ✅ |
| exp2_qwen2.5-0.5b_qlora_attn-eager_b2_s1024_acc1 | reserved/peak（碎片系数） | 1.05~1.15(经验) | 1.069 | - |
| exp2_qwen2.5-0.5b_qlora_attn-eager_b2_s1024_acc1 | 梯度 dtype | - | torch.float32 | - |
| exp2_qwen2.5-1.5b_lora_attn-eager_b1_s1024_acc1 | 权重 | 2.944 | 2.945 | -0.0% ✅ |
| exp2_qwen2.5-1.5b_lora_attn-eager_b1_s1024_acc1 | 激活值 | 6.316 | 6.580 | -4.0% ✅ |
| exp2_qwen2.5-1.5b_lora_attn-eager_b1_s1024_acc1 | reserved/peak（碎片系数） | 1.05~1.15(经验) | 1.054 | - |
| exp2_qwen2.5-1.5b_lora_attn-eager_b1_s1024_acc1 | 梯度 dtype | - | torch.float32 | - |
