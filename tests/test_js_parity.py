"""JS/Python 一致性: node 跑 web/vram_calc.js 对照 web/test_cases.json.

test_cases.json 由 tests/gen_web_cases.py 从 Python 库生成;
公式改动后须重新生成再跑本测试.
"""

import json
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

JS_RUNNER = """
const VC = require(process.argv[1] + "/web/vram_calc.js");
const cases = require(process.argv[1] + "/web/test_cases.json");
const out = cases.map(c => {
  const m = VC.MODELS[c.model];
  if (c.kind === "inference")
    return VC.estimateInference(m, { precision: c.precision, batch: c.batch, seq: c.seq });
  if (c.kind === "gguf")
    return VC.estimateGguf(m, { quant: c.quant, ctx: c.ctx });
  return VC.estimateTraining(m, { mode: c.mode, mixedPrecision: c.mixed_precision,
      batch: c.batch, seq: c.seq, loraRank: c.lora_rank,
      attnImpl: c.attn_impl, gradientCheckpointing: c.ckpt });
});
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node 不可用")
def test_js_matches_python():
    cases = json.loads((ROOT / "web" / "test_cases.json").read_text())
    raw = subprocess.check_output(["node", "-e", JS_RUNNER, str(ROOT)], text=True)
    results = json.loads(raw)
    assert len(results) == len(cases)
    for c, got in zip(cases, results):
        for k, expect in c["expect"].items():
            assert got[k] == pytest.approx(expect, rel=1e-9, abs=1e-6), (c, k)
