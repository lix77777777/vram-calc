import pytest

from vram_calc import GiB, recommend
from vram_calc.recommend import GPU_VRAM


def test_inference_14_6gb_fits_24gb_not_12gb():
    fits = recommend(14.6 * GiB)
    assert "RTX 3090" in fits and "RTX 4090" in fits
    assert "RTX 3060 12GB" not in fits and "RTX 4060" not in fits


def test_sorted_ascending():
    fits = recommend(0)
    sizes = [GPU_VRAM[n] for n in fits]
    assert sizes == sorted(sizes) and len(fits) == len(GPU_VRAM)


def test_nothing_fits_100gb_plus():
    assert recommend(100.5 * GiB) == []


def test_headroom():
    # 22 GiB × 1.2 > 24 → 24GB 卡被排除
    assert "RTX 4090" not in recommend(22 * GiB, headroom=0.2)


def test_negative_raises():
    with pytest.raises(ValueError):
        recommend(-1)
