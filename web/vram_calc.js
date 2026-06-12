/* vram_calc.js — src/vram_calc/memory.py 的逐行 JS 镜像.
 * 任何公式改动必须两边同步, CI 用 web/test_cases.json 强制一致 (<1e-9).
 * 在浏览器中挂到 window.VC, 在 Node 中走 module.exports. */
(function (root) {
  "use strict";

  const GiB = 1024 ** 3;
  const MiB = 1024 ** 2;

  const BYTES_PER_PARAM = { fp32: 4.0, fp16: 2.0, bf16: 2.0, int8: 1.0, int4: 0.5 };

  // Phase 3 实测标定常数 (docs/formulas.md §4)
  const NF4_DQ_EXTRA_BITS = 0.127;
  const ACT_C_BASE = 31.0, ACT_C_MLP = 8.0, ACT_K_EAGER = 6.0, ACT_K_LOGITS = 3.0;
  const LORA_ACT_OVERHEAD = 0.13;
  const KV_BUFFER = 10 * MiB;

  // GGUF (llama.cpp/Ollama) 平均 bit/权重〔待验证-10〕
  const GGUF_BPW = { Q2_K: 2.63, Q3_K_M: 3.91, Q4_0: 4.55, Q4_K_S: 4.58,
                     Q4_K_M: 4.85, Q5_K_M: 5.69, Q6_K: 6.59, Q8_0: 8.50, F16: 16.0 };
  const GGUF_GRAPH_BASE = 0.40 * GiB;

  // 预置模型 (字段同 models.py, 来源 HF config.json 2026-06-10)
  const MODELS = {
    "llama-2-7b":      { h: 4096, L: 32, a: 32, kv: 32, f: 11008, V: 32000,  tie: false, qkvBias: false, hd: null },
    "llama-3-8b":      { h: 4096, L: 32, a: 32, kv: 8,  f: 14336, V: 128256, tie: false, qkvBias: false, hd: null },
    "mistral-7b-v0.1": { h: 4096, L: 32, a: 32, kv: 8,  f: 14336, V: 32000,  tie: false, qkvBias: false, hd: null },
    "qwen2.5-0.5b":    { h: 896,  L: 24, a: 14, kv: 2,  f: 4864,  V: 151936, tie: true,  qkvBias: true,  hd: null },
    "qwen2.5-1.5b":    { h: 1536, L: 28, a: 12, kv: 2,  f: 8960,  V: 151936, tie: true,  qkvBias: true,  hd: null },
    "qwen2.5-7b":      { h: 3584, L: 28, a: 28, kv: 4,  f: 18944, V: 152064, tie: false, qkvBias: true,  hd: null },
    "qwen2.5-14b":     { h: 5120, L: 48, a: 40, kv: 8,  f: 13824, V: 152064, tie: false, qkvBias: true,  hd: null },
    "qwen3-8b":        { h: 4096, L: 36, a: 32, kv: 8,  f: 12288, V: 151936, tie: false, qkvBias: false, hd: 128 },
    "deepseek-llm-7b": { h: 4096, L: 30, a: 32, kv: 32, f: 11008, V: 102400, tie: false, qkvBias: false, hd: null },
    "tinyllama-1.1b":  { h: 2048, L: 22, a: 32, kv: 4,  f: 5632,  V: 32000,  tie: false, qkvBias: false, hd: null },
    "yi-6b":           { h: 4096, L: 32, a: 32, kv: 4,  f: 11008, V: 64000,  tie: false, qkvBias: false, hd: null },
  };

  const GPU_VRAM = {
    "RTX 4060": 8, "RTX 3060 12GB": 12, "RTX 4070": 12,
    "RTX 4080": 16, "T4": 16, "RTX 5060 Ti 16GB": 16,
    "RTX 3090": 24, "RTX 4090": 24, "L4": 24,
    "RTX 5090": 32, "V100 32GB": 32, "A100 40GB": 40,
    "A100 80GB": 80, "H100 80GB": 80,
  };

  const headDim = (m) => m.hd !== null ? m.hd : Math.floor(m.h / m.a);
  const kvDim = (m) => m.kv * headDim(m);

  function embedParams(m) { return m.V * m.h * (m.tie ? 1 : 2); }
  function linearParams(m) {
    const qDim = m.a * headDim(m);
    let attn = m.h * qDim + 2 * m.h * kvDim(m) + qDim * m.h;
    if (m.qkvBias) attn += qDim + 2 * kvDim(m);
    return m.L * (attn + 3 * m.h * m.f);
  }
  function numParams(m) {
    return embedParams(m) + linearParams(m) + m.L * 2 * m.h + m.h;
  }

  function weightsMemory(numP, precision, quantExtraBits = 0.0) {
    return numP * (BYTES_PER_PARAM[precision] + quantExtraBits / 8.0);
  }

  function loraNumParams(m, rank, targets = "all-linear") {
    const qDim = m.a * headDim(m), kvd = kvDim(m);
    const attn = rank * ((m.h + qDim) + 2 * (m.h + kvd) + (qDim + m.h));
    const perLayer = targets === "attention" ? attn : attn + rank * 3 * (m.h + m.f);
    return m.L * perLayer;
  }

  function qloraWeightsMemory(m, loraRank = 16, quantExtraBits = NF4_DQ_EXTRA_BITS) {
    const emb = embedParams(m) * 2.0;
    const lin = linearParams(m) * (0.5 + quantExtraBits / 8.0);
    const other = (numParams(m) - embedParams(m) - linearParams(m)) * 2.0;
    return emb + lin + other + loraNumParams(m, loraRank) * 4.0;
  }

  function gradientsMemory(numTrainable, gradPrecision) {
    return numTrainable * BYTES_PER_PARAM[gradPrecision];
  }

  const OPT_SLOTS = { adamw: 2.0, adamw_8bit: 0.5, sgd: 0.0, sgd_momentum: 1.0 };
  function optimizerMemory(numTrainable, optimizer = "adamw",
                           masterWeights = false, statePrecision = "fp32") {
    const state = optimizer === "adamw_8bit"
      ? 2.0 : OPT_SLOTS[optimizer] * BYTES_PER_PARAM[statePrecision];
    return numTrainable * (state + (masterWeights ? 4.0 : 0.0));
  }

  function activationsMemory(m, batch, seq, { attnImpl = "eager",
      gradientCheckpointing = false, includeLogits = true, lora = false } = {}) {
    if (batch === 0 || seq === 0) return 0.0;
    let coef = ACT_C_BASE + ACT_C_MLP * m.f / m.h;
    if (attnImpl === "eager") coef += ACT_K_EAGER * m.a * seq / m.h;
    const perLayer = seq * batch * m.h * coef;
    const logits = includeLogits ? ACT_K_LOGITS * batch * seq * m.V : 0.0;
    const act = gradientCheckpointing
      ? 2.0 * m.L * seq * batch * m.h + perLayer + logits
      : m.L * perLayer + logits;
    return lora ? act * (1.0 + LORA_ACT_OVERHEAD) : act;
  }

  function inferenceActivationsMemory(m, batch, seq) {
    if (batch === 0 || seq === 0) return 0.0;
    return 2.0 * batch * seq * m.V + 4.0 * seq * batch * m.h * 2.0;
  }

  function kvCacheMemory(m, batch, seq, precision = "fp16") {
    return 2.0 * m.L * m.kv * headDim(m) * seq * batch * BYTES_PER_PARAM[precision];
  }

  function frameworkOverhead(subtotal, base = 0.75 * GiB, fraction = 0.08) {
    return base + fraction * subtotal;
  }

  function breakdown(w, g, o, act, kvc, oh) {
    const total = w + g + o + act + kvc + oh;
    return { weights: w, gradients: g, optimizer: o,
             activations: act, kv_cache: kvc, overhead: oh, total };
  }

  function estimateTraining(m, { mode = "full", mixedPrecision = "bf16_pure",
      optimizer = "adamw", batch = 1, seq = 2048, loraRank = 16,
      attnImpl = "eager", gradientCheckpointing = false,
      includeOverhead = true } = {}) {
    const P = numParams(m);
    let w, g, o;
    if (mode === "full") {
      if (mixedPrecision === "amp") {
        w = weightsMemory(P, "fp32"); g = gradientsMemory(P, "fp32");
        o = optimizerMemory(P, optimizer);
      } else if (mixedPrecision === "megatron") {
        w = weightsMemory(P, "bf16"); g = gradientsMemory(P, "bf16");
        o = optimizerMemory(P, optimizer, true);
      } else if (mixedPrecision === "bf16_pure") {
        w = weightsMemory(P, "bf16"); g = gradientsMemory(P, "bf16");
        o = optimizerMemory(P, optimizer, false, "bf16");
      } else { throw new Error("unknown mixedPrecision " + mixedPrecision); }
    } else if (mode === "lora" || mode === "qlora") {
      const t = loraNumParams(m, loraRank);
      w = mode === "qlora" ? qloraWeightsMemory(m, loraRank)
                           : weightsMemory(P, "bf16") + t * 4.0;
      g = gradientsMemory(t, "fp32");
      o = optimizerMemory(t, optimizer);
    } else { throw new Error("unknown mode " + mode); }

    const act = activationsMemory(m, batch, seq, { attnImpl, gradientCheckpointing,
      lora: mode === "lora" || mode === "qlora" });
    const oh = includeOverhead ? frameworkOverhead(w + g + o + act) : 0.0;
    return breakdown(w, g, o, act, 0.0, oh);
  }

  function estimateInference(m, { precision = "fp16", batch = 1, seq = 2048,
      quantExtraBits = 0.0, includeOverhead = true } = {}) {
    const w = weightsMemory(numParams(m), precision, quantExtraBits);
    const kvPrec = ["fp16", "bf16", "fp32"].includes(precision) ? precision : "fp16";
    const kvc = kvCacheMemory(m, batch, seq, kvPrec) + (batch && seq ? KV_BUFFER : 0.0);
    const act = inferenceActivationsMemory(m, batch, seq);
    const oh = includeOverhead ? frameworkOverhead(w + kvc + act) : 0.0;
    return breakdown(w, 0.0, 0.0, act, kvc, oh);
  }

  function ggufWeightsMemory(m, quant = "Q4_K_M") {
    if (!(quant in GGUF_BPW)) throw new Error("unknown gguf quant " + quant);
    return numParams(m) * GGUF_BPW[quant] / 8.0;
  }

  function estimateGguf(m, { quant = "Q4_K_M", ctx = 4096, batch = 1 } = {}) {
    const w = ggufWeightsMemory(m, quant);
    const kv = kvCacheMemory(m, batch, ctx, "fp16") + (batch && ctx ? KV_BUFFER : 0.0);
    const graph = GGUF_GRAPH_BASE + 0.05 * kv;
    return breakdown(w, 0.0, 0.0, 0.0, kv, graph);
  }

  function recommend(requiredBytes, headroom = 0.0) {
    const need = requiredBytes * (1.0 + headroom) / GiB;
    return Object.entries(GPU_VRAM)
      .sort((x, y) => x[1] - y[1] || x[0].localeCompare(y[0]))
      .filter(([, v]) => v >= need).map(([n]) => n);
  }

  const VC = { GiB, MODELS, GPU_VRAM, numParams, loraNumParams, weightsMemory,
               qloraWeightsMemory, gradientsMemory, optimizerMemory,
               activationsMemory, inferenceActivationsMemory, kvCacheMemory,
               frameworkOverhead, estimateTraining, estimateInference, recommend,
               GGUF_BPW, ggufWeightsMemory, estimateGguf };

  if (typeof module !== "undefined" && module.exports) module.exports = VC;
  else root.VC = VC;
})(typeof self !== "undefined" ? self : this);
