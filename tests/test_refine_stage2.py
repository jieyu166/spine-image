import cv2
import numpy as np
import pytest

from refine_stage2 import (
    CROP, HEATMAP, POINTS, apply_affine, crop_matrix, decode_heatmap, extract_crop,
    fill_sizes, make_targets, vertebra_center, vertebra_size,
)


def _vertebra(x=100, y=200, w=300, h=250):
    return {
        'anteriorSuperior': {'x': x, 'y': y}, 'middleSuperior': {'x': x + w / 2, 'y': y + 5},
        'posteriorSuperior': {'x': x + w, 'y': y}, 'posteriorInferior': {'x': x + w, 'y': y + h},
        'middleInferior': {'x': x + w / 2, 'y': y + h - 5}, 'anteriorInferior': {'x': x, 'y': y + h},
    }


def test_center_and_size_of_full_and_boundary_vertebrae():
    v = _vertebra()
    assert vertebra_center(v) == pytest.approx((250.0, 325.0))
    assert vertebra_size(v) == pytest.approx(300.0)
    upper_only = {k: v[k] for k in ('anteriorSuperior', 'middleSuperior', 'posteriorSuperior')}
    assert vertebra_size(upper_only) is None


def test_fill_sizes_borrows_from_nearest_full_vertebra():
    full = _vertebra()
    s1 = {'points': {k: full[k] for k in ('anteriorSuperior', 'middleSuperior', 'posteriorSuperior')}}
    sizes = fill_sizes([{'points': _vertebra(w=200, h=180)}, {'points': full}, s1])
    assert sizes == pytest.approx([200.0, 300.0, 300.0])
    assert fill_sizes([s1]) == [None]


@pytest.mark.parametrize('angle', [0.0, 7.5, -10.0])
def test_crop_matrix_maps_center_to_crop_center_and_inverts(angle):
    m = crop_matrix((500.0, 800.0), 660.0, angle)
    assert apply_affine(m, 500.0, 800.0) == pytest.approx((CROP / 2, CROP / 2))
    inv = cv2.invertAffineTransform(m)
    x, y = apply_affine(m, 612.0, 733.0)
    assert apply_affine(inv, x, y) == pytest.approx((612.0, 733.0))
    if angle == 0.0:
        # 邊長 660 → 256：縮放 256/660
        assert apply_affine(m, 500.0 + 330.0, 800.0)[0] == pytest.approx(CROP)


def test_extract_crop_samples_the_right_pixel():
    image = np.zeros((1000, 800), np.uint8)
    image[395:406, 295:306] = 255  # 亮點在 (300, 400)
    patch, m = extract_crop(image, (300.0, 400.0), 200.0)
    ys, xs = np.nonzero(patch > 128)
    assert abs(xs.mean() - CROP / 2) <= 1.5 and abs(ys.mean() - CROP / 2) <= 1.5


def test_targets_mask_missing_and_out_of_crop_points():
    pts = {'anteriorSuperior': (128.0, 64.0), 'posteriorSuperior': (300.0, 10.0)}
    heat, mask = make_targets(pts)
    assert heat.shape == (len(POINTS), HEATMAP, HEATMAP)
    assert mask.tolist() == [1, 0, 0, 0, 0, 0]
    iy, ix = np.unravel_index(int(np.argmax(heat[0])), heat[0].shape)
    assert (ix, iy) == (32, 16)


@pytest.mark.parametrize('x,y', [(100.0, 140.0), (37.3, 201.6), (128.0, 128.0)])
def test_decode_recovers_target_location_with_subpixel_accuracy(x, y):
    heat, _ = make_targets({'anteriorSuperior': (x, y)})
    dx, dy, peak = decode_heatmap(heat[0])
    # heatmap 一格 = 4 裁切 px；子像素解碼誤差應 < 1 px
    assert abs(dx - x) < 1.0 and abs(dy - y) < 1.0
    assert peak > 0.9


def test_case_median_sizes_applies_one_robust_size_to_every_vertebra():
    from refine_stage2 import case_median_sizes
    full = _vertebra()
    s1 = {'points': {k: full[k] for k in ('anteriorSuperior', 'middleSuperior', 'posteriorSuperior')}}
    verts = [{'points': _vertebra(w=200, h=180)}, {'points': full}, {'points': _vertebra(w=900, h=200)}, s1]
    assert case_median_sizes(verts) == pytest.approx([300.0] * 4)
    assert case_median_sizes([s1]) == [None]


class _FakeModel:
    """回傳固定 heatmap 的假模型：所有點都在裁切中心往右 dx 個裁切 px。"""

    def __init__(self, dx):
        import torch
        self.dx = dx
        self._p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._p])

    def eval(self):
        return self

    def __call__(self, x):
        import torch
        heat, _ = make_targets({name: (CROP / 2 + self.dx, CROP / 2) for name in POINTS})
        logits = np.log(np.clip(heat, 1e-6, 1 - 1e-6) / (1 - np.clip(heat, 1e-6, 1 - 1e-6)))
        return torch.from_numpy(np.repeat(logits[None], x.shape[0], 0))


def test_shift_gate_keeps_stage1_when_refinement_jumps_to_a_neighbour():
    from refine_stage2 import refine_vertebrae
    image = np.zeros((2000, 2000), np.uint8)
    verts = [{'name': 'L3', 'points': _vertebra(x=800, y=800, w=300, h=300)}]
    # 裁切邊長 2.2×300 = 660 → 1 裁切 px ≈ 2.58 原圖 px；dx=60 → 約 155 px ≈ 0.52 × 椎體大小
    jumped = refine_vertebrae(_FakeModel(60), image, verts, max_shift=0.25)
    assert jumped[0]['points'] == verts[0]['points']
    small = refine_vertebrae(_FakeModel(10), image, verts, max_shift=0.25)
    assert small[0]['points'] != verts[0]['points']
    ungated = refine_vertebrae(_FakeModel(60), image, verts)
    assert ungated[0]['points']['anteriorSuperior']['x'] == pytest.approx(950 + 60 * 660 / CROP, abs=3)
