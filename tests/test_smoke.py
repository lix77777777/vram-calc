"""骨架冒烟测试: 确认包可导入. Phase 2 替换为真实测试."""

import vram_calc


def test_import():
    assert vram_calc.__version__
