# 显存公式推导（v2，Phase 3 实测修订版）

> v1 草稿经 2026-06-10 实测（Qwen2.5-0.5B/1.5B，RTX 5060 Ti 16GB，torch 2.7 + transformers 4.55）
> 修订，**修订处用 🔧 标记**——先推导、上卡实测、被打脸、再修对，是这份文档的全部方法论。
> 标定系数来自单一环境，跨模型外推处已标注〔待跨模型验证〕。单位 GB 一律指 GiB。

## 0. 记号与全局假设

| 符号 | 含义 |
|---|---|
| P | 总参数量；P_t 可训练参数量；V 词表大小 |
| b | 每参数字节数：FP32=4，FP16/BF16=2，INT8=1，INT4=0.5 |
| B, s | batch size、序列长度 |
| h, a, L, f | 隐藏维度、注意力头数、层数、FFN 中间维度 |
| n_kv, d_head | KV 头数、每头维度 |

基准结构：decoder-only Transformer（Llama/Qwen 系，SwiGLU + RMSNorm）。单卡单进程。

---

## 1. 权重显存

```
全精度/半精度:  M_weights = P × b                                    ✅实测 -0.8%
量化(bnb NF4): M_weights = P_embed×2 + P_linear×(0.5 + 0.127/8) + P_norm×2
```

🔧 **实测修正**：bitsandbytes **只量化 Linear 层**；embedding（及未绑定的 lm_head）和
RMSNorm 保持 BF16。v1 草稿把全部 P 按 INT4 算，对 Qwen2.5-0.5B 低估 45%（词表 15 万、
embedding 占总参数 28%）。修正后误差 **-0.2%** ✅。
对大模型影响小（7B 的 embedding 仅占 4%），对小词表大模型几乎无影响。

量化常数：NF4 块 64 + double quantization = +0.127 bit/参数（QLoRA 论文 §3）✅；
GPTQ/AWQ g=128 ≈ +0.25 bit/参数〔待验证-1，未实测〕。

**来源**：QLoRA, arXiv:2305.14314；bitsandbytes 行为为实测确认。

## 2. 梯度显存

```
M_grad = P_t × b_grad，b_grad = 参数本体的 dtype 字节数               ✅实测确认
```

实测确认两条记账规则：
- 纯 BF16 训练（裸 torch AdamW，exp2 full）：梯度 BF16 = 2 字节 ✅
- 🔧 **peft LoRA/QLoRA：adapter 默认创建为 FP32**，梯度也是 FP32 = 4 字节
  （v1 假设 adapter BF16，实测 `grad_dtype: torch.float32` 打脸）
- LoRA 时基座无梯度 ✅（lora_num_params 与 peft 实载值**完全一致**：8,798,208）
- 梯度累积无额外副本 ✅（acc=1 与 acc=8 激活值差 < 0.1%，待验证-3 关闭）

## 3. 优化器状态（AdamW）

```
M_opt = P_t × (2 × b_state + 4×[master])
```

| 方案 | B/参数 | 实测 |
|---|---|---|
| amp 式（参数 FP32） | 8 | 未实测 |
| Megatron/DeepSpeed 式（+FP32 master） | 12 | 未实测（ZeRO 论文 K=12） |
| 🔧 纯 BF16（torch AdamW 直接用于 BF16 参数）| **4**（m/v 跟随参数 dtype）| ✅ 1.845 GiB 分毫不差 |
| peft LoRA（adapter FP32） | 8 | ✅（含在权重对账内） |
| AdamW 8-bit | 2 | 未实测 |

🔧 v1 只列了 amp/Megatron 两种，实测发现裸 torch 训练 BF16 模型是第三种记账
（m/v 也是 BF16），且这正是个人单卡微调最常见的场景。

**来源**：ZeRO arXiv:1910.02054；torch AdamW 状态 dtype 跟随参数为实测确认。

## 4. 激活值显存（🔧 v1 公式重写）

v1 用 Korthikanti 式 (34+5as/h)sBh·L，实测低估 41%~78%。两个原因：
**(a) 漏了 logits/loss 链**——lm_head 输出 + cross-entropy 中间量 ≈ 3·B·s·V 字节，
Qwen 词表 15 万时这一项高达 0.87 GiB，比小模型一层激活大 4 倍；
**(b) 系数假设 GPT 结构**（f=4h、GELU、有 dropout），SwiGLU 的 MLP 要存 gate/up/silu/积
四组中间量，f/h 还更大（Qwen2.5-0.5B 为 5.43）。

**修订公式（标定于实测，四种配置误差 ≤ 2.5%）**：

```
M_act = L·s·B·h·(C_base + C_mlp·f/h + K_attn·a·s/h) + K_logits·B·s·V

C_base   = 31   （LN/注意力 IO/残差等）       〔标定；1.5b 样本外 -4.0% ✅〕
C_mlp    = 8    （SwiGLU 存 ~4 组 f 维中间量×2B）〔结构推导+标定〕
K_attn   = 6    （eager：s² 矩阵，transformers 用 FP32 softmax 所以 >5）
                （sdpa/flash：0，s² 项消失 ✅实测确认）
K_logits = 3    （logits BF16 2B + 分块 CE ~1B；老版 transformers 全量 FP32
                 upcast 时可达 10，版本相关）〔标定〕
```

**Gradient checkpointing** ✅实测 -1.2%：

```
M_act_ckpt = 2·L·s·B·h（每层只存输入）+ 单层完整激活 + K_logits·B·s·V
```

🔧 logits 项不随 ckpt 消失（lm_head 不在被 checkpoint 的层里）——v1 漏了这点，
也是 ckpt 误差曾达 -78% 的主因。

**LoRA/QLoRA**：实测激活比全量多 **+13%**（adapter 的 FP32 输入/输出副本），
公式乘 1.13〔标定，待跨模型验证〕。

**推理 prefill 工作集**（v1 的"约数个 sBh"修正）：

```
M_infer_act ≈ 2·B·s·V（prefill 全位置 logits，占绝对大头）+ 8·s·B·h
```
实测 -3.7%（0.5b）/-7.2%（1.5b）✅。注：vLLM 等推理引擎只算最后位置 logits，会小得多。

**来源**：Korthikanti arXiv:2205.05198（方法论）；系数为本项目实测标定。

## 5. KV-Cache ✅公式实测验证

```
M_kv = 2 × L × n_kv × d_head × s × B × b
```

实测与公式差值为**固定 ~9.2 MiB**（两个模型完全一致），是 rotary/mask 等常驻缓冲，
非比例误差——公式本身正确。计算器以 KV_BUFFER=10 MiB 常数项计入。
GQA 由 n_kv 体现 ✅；MLA（DeepSeek）v1.5 扩展〔待验证-7〕：
`M_mla = L × (d_c + d_rope) × s × B × b`（DeepSeek-V2: d_c=512, d_rope=64）。

**来源**：GQA arXiv:2305.13245；DeepSeek-V2 arXiv:2405.04434。

## 6. 框架开销

```
M_overhead = C_cuda + 碎片系数 × 小计
```

- 碎片系数：✅实测 reserved/peak = **1.05~1.11**（六组配置），计算器取 8%。
  🔧 注意分母必须是 peak 而不是 step 后的 allocated（v1 报表用错分母曾显示 3~11 倍）。
- C_cuda：exp3 未成功（待验证-8），暂用经验值 0.75 GiB。

## 7. 端到端示例（Llama-2-7B，修订后）

| 场景 | 权重 | 梯度 | 优化器 | 合计(静态) |
|---|---|---|---|---|
| 全量 Megatron 式 BF16 | 12.6 | 12.6 | 75.3 | ≈100.4 GB |
| 全量纯 BF16（裸 torch）🔧 | 12.6 | 12.6 | 25.1 | ≈50.2 GB |
| QLoRA(NF4, r=16) 🔧 | 3.75 | 0.15 | 0.30 | ≈4.2 GB |
| 推理 FP16, B1 s4096 | 12.6 | — | — | +KV 2.0 → ≈14.6 GB |

🔧 QLoRA 从 v1 的 3.8 改为 4.2：embedding 不量化 +0.24，adapter FP32 +0.07。
（QLoRA 的 40M LoRA 参数推导见 §2，已被 peft 实载值验证。）

## 8. 验证状态总表

| # | 项目 | 状态 |
|---|---|---|
| 1 | GPTQ/AWQ 量化常数 | 未实测（NF4 路径已 ✅） |
| 2 | 梯度 dtype 规则 | ✅ 关闭（跟随参数 dtype） |
| 3 | 梯度累积额外副本 | ✅ 关闭（无额外副本） |
| 4 | SwiGLU 激活系数 | ✅ 关闭（1.5b 样本外验证 -4.0%，2026-06-10） |
| 5 | flash/sdpa 激活 | ✅ 关闭（s² 项消失） |
| 6 | 推理工作集 | ✅ 关闭（= prefill logits 主导） |
| 7 | MLA 公式 | v1.5 再验 |
| 8 | CUDA context | 未实测（exp3 待重跑） |
| 9 | 碎片系数 | ✅ 关闭（1.05~1.11 对 peak） |
| 10 | GGUF/llama.cpp 路径（bpw 表、计算图缓冲） | 未实测（权重项与官方文件大小对照 -1%；KV/缓冲待 Ollama 实测） |
