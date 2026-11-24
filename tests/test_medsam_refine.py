import hashlib
import io
import json

import cv2
import numpy as np
import pytest

import medsam_refine as mr
import medsam_server as ms


def _vertebra(cx=500.0, cy=500.0, w=300.0, h=220.0, angle_deg=0.0, dip=0.0):
    """以 (cx, cy) 為中心的椎體 6 點；前方在左（x 小）。dip > 0 時終板中點往椎體內凹。"""
    local = {
        'anteriorSuperior': (-w / 2, -h / 2), 'middleSuperior': (0, -h / 2 + dip), 'posteriorSuperior': (w / 2, -h / 2),
        'posteriorInferior': (w / 2, h / 2), 'middleInferior': (0, h / 2 - dip), 'anteriorInferior': (-w / 2, h / 2),
    }
    t = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    out = {}
    for k, (x, y) in local.items():
        px, py = rot @ np.array([x, y])
        out[k] = {'x': float(cx + px), 'y': float(cy + py)}
    return out


def _err(a, b):
    return float(np.hypot(a[0] - b['x'], a[1] - b['y']))


@pytest.mark.parametrize('angle', [0, 12, -20, 28])
@pytest.mark.parametrize('dip', [0, 15])
def test_mask_to_points_recovers_outline_points(angle, dip):
    pts = _vertebra(angle_deg=angle, dip=dip)
    mask = mr.rasterize(mr.outline_polygon(pts), 1024)
    out = mr.mask_to_points(mask, mr.local_axes(pts))
    assert out is not None
    for k in mr.CORNERS:
        assert _err(out[k], pts[k]) < 2.5, (k, out[k], pts[k])
    # 中點：弦長一半位置，對稱終板應落在標註點附近
    for k in ('middleSuperior', 'middleInferior'):
        assert _err(out[k], pts[k]) < 3.0, (k, out[k], pts[k])


@pytest.mark.parametrize('angle', [0, 15, -25])
def test_axes_from_mask_matches_point_axes_and_anterior_left(angle):
    pts = _vertebra(angle_deg=angle)
    mask = mr.rasterize(mr.outline_polygon(pts), 1024)
    u, v = mr.axes_from_mask(mask)
    u0, v0 = mr.local_axes(pts)
    assert np.dot(u, u0) > 0.97 and np.dot(v, v0) > 0.97
    assert u[0] < 0 and v[1] > 0
    out = mr.mask_to_points(mask, (u, v))
    assert all(_err(out[k], pts[k]) < 2.5 for k in mr.CORNERS)


def test_mask_to_points_rejects_empty_or_tiny_masks():
    assert mr.mask_to_points(np.zeros((64, 64), np.uint8), (np.array([-1.0, 0]), np.array([0, 1.0]))) is None
    tiny = np.zeros((64, 64), np.uint8)
    tiny[10:13, 10:13] = 1
    assert mr.mask_to_points(tiny, (np.array([-1.0, 0]), np.array([0, 1.0]))) is None


def test_largest_component_keeps_only_biggest_blob():
    m = np.zeros((100, 100), np.uint8)
    m[10:20, 10:20] = 1
    m[40:90, 40:90] = 1
    comp = mr.largest_component(m)
    assert comp.sum() == 2500 and comp[15, 15] == 0


def test_outline_polygon_requires_four_corners_and_orders_points():
    pts = _vertebra()
    poly = mr.outline_polygon(pts)
    assert poly.shape == (6, 2)
    assert mr.outline_polygon({k: pts[k] for k in mr.SUPERIOR}) is None  # S1 只有上終板
    four = {k: pts[k] for k in mr.CORNERS}
    assert mr.outline_polygon(four).shape == (4, 2)


def test_rasterize_scales_polygon():
    poly = np.array([[0, 0], [400, 0], [400, 400], [0, 400]], np.float32)
    m = mr.rasterize(poly, 256, 256 / 1024)
    assert 0.14 < m.mean() < 0.17  # 100×100 / 256²


def test_jitter_box_is_valid_and_bounded():
    rng = np.random.default_rng(0)
    box = np.array([100.0, 100.0, 400.0, 320.0])
    for _ in range(500):
        b = mr.jitter_box(box, 300.0, rng)
        assert b[2] > b[0] and b[3] > b[1]
        assert np.all(np.abs(b - box) <= 0.8 * 300.0 + 1e-6)


def test_points_box_and_margin():
    pts = _vertebra(w=300, h=200)
    assert np.allclose(mr.points_box(pts), [350, 400, 650, 600])
    assert np.allclose(mr.points_box(pts, 10), [340, 390, 660, 610])


def test_mask_iou():
    a = np.zeros((10, 10), np.uint8)
    b = np.zeros((10, 10), np.uint8)
    a[:5] = 1
    b[:, :5] = 1
    assert mr.mask_iou(a, b) == pytest.approx(25 / 75)
    assert mr.mask_iou(np.zeros((3, 3)), np.zeros((3, 3))) == 0.0


def test_preprocess_modes():
    patch = np.full((1024, 1024), 128, np.uint8)
    patch[0, 0] = 0
    x = mr.preprocess(patch, 'imagenet')
    assert x.shape == (3, 1024, 1024) and x.dtype == np.float32
    assert x[0, 5, 5] == pytest.approx((128 / 255 - 0.485) / 0.229, rel=1e-5)
    y = mr.preprocess(patch, 'minmax')
    assert y.min() == 0.0 and y.max() == 1.0


def test_drawn_boxes_uses_gt_for_full_vertebrae_only():
    gt = [{'name': 'L1', 'points': _vertebra()}, {'name': 'S1', 'points': {k: _vertebra()[k] for k in mr.SUPERIOR}}]
    pred = [{'name': 'L1', 'points': _vertebra(cx=560)}, {'name': 'S1', 'points': {k: _vertebra()[k] for k in mr.SUPERIOR}},
            {'name': 'L2', 'points': _vertebra(cy=200)}]
    boxes, centers, sizes = mr.drawn_boxes(pred, gt, np.random.default_rng(1))
    assert boxes[0] is not None and centers[0] == pytest.approx((500.0, 500.0))
    assert boxes[1] is None and sizes[1] is None  # 邊界椎體
    assert boxes[2] is None  # GT 沒有這節


def test_make_training_item_shapes_and_target_inside_box(tmp_path):
    img = np.zeros((1200, 1000), np.uint8)
    pts = _vertebra()
    cv2.fillPoly(img, [mr.outline_polygon(pts).astype(np.int32)], 200)
    path = tmp_path / 'x.png'
    cv2.imwrite(str(path), img)
    sample = {'image': path, 'points': pts, 'size': mr.vertebra_size(pts)}
    x, box, target = mr.make_training_item(sample, mr.ImageCache(), np.random.default_rng(3), 'imagenet')
    assert x.shape == (3, 1024, 1024) and box.shape == (4,) and target.shape == (256, 256)
    assert 0.05 < target.mean() < 0.4
    ys, xs = np.nonzero(target)
    cx, cy = xs.mean() * 4, ys.mean() * 4
    assert box[0] - 200 < cx < box[2] + 200 and box[1] - 200 < cy < box[3] + 200


# ── 本機服務 ──────────────────────────────────────────────────────────
class FakeSegmenter:
    def __init__(self):
        self.calls = []

    def info(self):
        return {'decoder': 'fake', 'fine_tuned': True, 'device': 'cpu', 'normalize': 'imagenet'}

    def __call__(self, image, box, hint_points=None, neighbor_points=None):
        self.calls.append((image.shape, list(box), hint_points, neighbor_points))
        if box[0] < 0:
            return None, 0.1, None
        pts = _vertebra(cx=(box[0] + box[2]) / 2, cy=(box[1] + box[3]) / 2)
        return pts, 0.93, [(1.0, 2.0), (3.0, 4.0)]


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    png = cv2.imencode('.png', np.full((300, 200), 90, np.uint8))[1].tobytes()
    blocked = cv2.imencode('.png', np.full((300, 200), 10, np.uint8))[1].tobytes()
    fake = FakeSegmenter()
    app = ms.create_app(fake, {hashlib.sha256(blocked).hexdigest(): 'holdout_t', 'ab' * 32: 'holdout_t'})
    return TestClient(app), fake, png, blocked


def test_server_health_and_full_vertebra(client):
    c, fake, png, _ = client
    assert c.get('/health').json()['holdout_images_blocked'] == 2
    r = c.post('/image', files={'file': ('x.png', png, 'image/png')})
    assert r.status_code == 200 and r.json()['width'] == 200 and r.json()['height'] == 300
    image_id = r.json()['image_id']
    r = c.post('/segment', json={'image_id': image_id, 'box': [10, 20, 110, 90]})
    body = r.json()
    assert r.status_code == 200 and set(body['points']) == set(mr.POINTS) and body['iou'] == pytest.approx(0.93)
    assert fake.calls[-1][0] == (300, 200)
    assert r.headers['access-control-allow-private-network'] == 'true'


def test_server_boundary_returns_only_one_endplate(client):
    c, _, png, _ = client
    image_id = c.post('/image', files={'file': ('x.png', png, 'image/png')}).json()['image_id']
    up = c.post('/segment', json={'image_id': image_id, 'box': [0, 0, 50, 50], 'boundary': 'upper'}).json()
    assert set(up['points']) == set(mr.SUPERIOR)
    low = c.post('/segment', json={'image_id': image_id, 'box': [0, 0, 50, 50], 'boundary': 'lower'}).json()
    assert set(low['points']) == set(mr.INFERIOR)
    assert c.post('/segment', json={'image_id': image_id, 'box': [0, 0, 50, 50], 'boundary': 'side'}).status_code == 400


def test_server_refuses_holdout_images_by_content_or_claimed_hash(client):
    c, _, png, blocked = client
    r = c.post('/image', files={'file': ('x.png', blocked, 'image/png')})
    assert r.status_code == 403 and '外部測試集' in r.json()['detail']
    r = c.post('/image', files={'file': ('x.png', png, 'image/png')}, data={'sha256': 'AB' * 32})
    assert r.status_code == 403


def test_server_errors(client):
    c, _, png, _ = client
    assert c.post('/segment', json={'image_id': 'nope', 'box': [0, 0, 1, 1]}).status_code == 404
    image_id = c.post('/image', files={'file': ('x.png', png, 'image/png')}).json()['image_id']
    assert c.post('/segment', json={'image_id': image_id, 'box': [0, 0, 1]}).status_code == 400
    assert c.post('/segment', json={'image_id': image_id, 'box': [-5, 0, 10, 10]}).status_code == 422
    assert c.post('/image', files={'file': ('x.png', b'not an image', 'image/png')}).status_code == 400


def test_box_geometry_inverts_training_margin():
    pts = _vertebra(w=300, h=220)
    size = mr.vertebra_size(pts)
    box = mr.points_box(pts, mr.BOX_MARGIN * size)
    center, est = ms.box_geometry(box)
    assert center == pytest.approx((500.0, 500.0))
    assert est == pytest.approx(size, rel=0.01)
    with pytest.raises(ValueError):
        ms.box_geometry([10, 10, 5, 20])


def test_load_holdout_hashes(tmp_path):
    (tmp_path / 'holdout_a.json').write_text(json.dumps({'kind': 'holdout', 'name': 'holdout_a',
                                                         'entries': [{'image': {'sha256': 'ABC'}}, {'image': {}}]}), encoding='utf-8')
    (tmp_path / 'holdout_b.json').write_text(json.dumps({'kind': 'other', 'entries': [{'image': {'sha256': 'def'}}]}), encoding='utf-8')
    assert ms.load_holdout_hashes(tmp_path) == {'abc': 'holdout_a'}


# ── 審查後補強 ────────────────────────────────────────────────────────
def test_local_axes_single_endplate_follows_endplate_tilt():
    tilted = _vertebra(angle_deg=40)
    s1 = {k: tilted[k] for k in mr.SUPERIOR}          # 只有上終板（S1）
    u, v = mr.local_axes(s1)
    u0, v0 = mr.local_axes(tilted)
    assert np.dot(u, u0) > 0.999 and np.dot(v, v0) > 0.999
    t12 = {k: tilted[k] for k in mr.INFERIOR}          # 只有下終板
    u, v = mr.local_axes(t12)
    assert np.dot(u, u0) > 0.999 and v[1] > 0


@pytest.mark.parametrize('angle', [50, 60])
def test_axes_from_mask_beyond_45_degrees_needs_neighbor_hint(angle):
    pts = _vertebra(w=300, h=220, angle_deg=angle)
    mask = mr.rasterize(mr.outline_polygon(pts), 1024)
    u0, v0 = mr.local_axes(pts)
    u, v = mr.axes_from_mask(mask)                    # 沒參考：會挑錯邊
    assert abs(np.dot(v, v0)) < 0.5
    u, v = mr.axes_from_mask(mask, prefer_v=mr.local_axes(_vertebra(angle_deg=angle - 10))[1])
    assert np.dot(v, v0) > 0.97 and np.dot(u, u0) > 0.97
    out = mr.mask_to_points(mask, (u, v))
    assert all(_err(out[k], pts[k]) < 2.5 for k in mr.CORNERS)


class _FakeSam:
    """模擬 SamModel：遮罩 = 提示框（256 低解析度），用來驗證 原圖 ↔ 1024 ↔ 256 座標來回。"""

    def __init__(self):
        import torch
        self._p = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._p])

    def get_image_embeddings(self, x):
        import torch
        return torch.zeros(x.shape[0], 256, 64, 64)

    def __call__(self, image_embeddings, input_boxes, multimask_output=False):
        import torch
        from types import SimpleNamespace
        b = input_boxes[:, 0]
        n = b.shape[0]
        ys = (torch.arange(256).float() + 0.5) * 4 - 0.5
        xs = ys
        inside = ((xs[None, None, :] >= b[:, 0, None, None]) & (xs[None, None, :] <= b[:, 2, None, None]) &
                  (ys[None, :, None] >= b[:, 1, None, None]) & (ys[None, :, None] <= b[:, 3, None, None]))
        logits = torch.where(inside, 10.0, -10.0)
        return SimpleNamespace(pred_masks=logits[:, None, None], iou_scores=torch.full((n, 1, 1), 0.9))


def test_segment_vertebra_round_trip_coordinates():
    image = np.zeros((3000, 2500), np.uint8)
    box = np.array([1000.0, 1500.0, 1360.0, 1760.0])   # 遮罩 = 這個框 → 角點應回到框角
    center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    pts, iou, outline = mr.segment_vertebra(_FakeSam(), image, center, 2.2 * 360, box)
    assert iou == pytest.approx(0.9)
    expect = {'anteriorSuperior': (1000, 1500), 'posteriorSuperior': (1360, 1500),
              'posteriorInferior': (1360, 1760), 'anteriorInferior': (1000, 1760)}
    for k, (x, y) in expect.items():
        assert _err((x, y), pts[k]) < 4.0, (k, pts[k])
    assert len(outline) >= 4


def test_server_passes_neighbor_points_and_uses_lru(client):
    c, fake, png, _ = client
    image_id = c.post('/image', files={'file': ('x.png', png, 'image/png')}).json()['image_id']
    hint = {'anteriorSuperior': {'x': 1, 'y': 2}}
    c.post('/segment', json={'image_id': image_id, 'box': [0, 0, 50, 50], 'neighbor_points': hint})
    assert fake.calls[-1][3] == hint and fake.calls[-1][2] is None


def test_server_refuses_holdout_by_decoded_pixels_after_reencoding():
    from fastapi.testclient import TestClient

    gray = (np.arange(300 * 200) % 251).astype(np.uint8).reshape(300, 200)
    rgba = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA)
    reencoded = cv2.imencode('.png', rgba, [cv2.IMWRITE_PNG_COMPRESSION, 9])[1].tobytes()  # 位元組不同、像素相同
    app = ms.create_app(FakeSegmenter(), {}, {ms.pixel_hash(gray): 'holdout_t'})
    r = TestClient(app).post('/image', files={'file': ('x.png', reencoded, 'image/png')})
    assert r.status_code == 403 and 'holdout_t' in r.json()['detail']


def test_load_holdout_pixel_hashes_reads_manifest_images(tmp_path):
    gray = np.full((40, 30), 77, np.uint8)
    img = tmp_path / 'h.png'
    cv2.imwrite(str(img), gray)
    (tmp_path / 'holdout_x.json').write_text(json.dumps({'kind': 'holdout', 'name': 'holdout_x', 'entries': [
        {'image': {'path': str(img)}}, {'image': {'path': str(tmp_path / 'missing.png')}}]}), encoding='utf-8')
    logs = []
    hashes = ms.load_holdout_pixel_hashes(tmp_path, log=logs.append)
    assert hashes == {ms.pixel_hash(gray): 'holdout_x'}
    assert any('讀不到' in m for m in logs)


def test_server_cors_only_local_origins(client):
    c, _, _, _ = client
    ok = c.options('/health', headers={'Origin': 'null', 'Access-Control-Request-Method': 'GET'})
    assert ok.headers.get('access-control-allow-origin') == 'null'
    bad = c.options('/health', headers={'Origin': 'https://evil.example', 'Access-Control-Request-Method': 'GET'})
    assert bad.headers.get('access-control-allow-origin') is None


def test_server_unexpected_error_becomes_json_500():
    from fastapi.testclient import TestClient

    class Boom(FakeSegmenter):
        def __call__(self, *a, **k):
            raise RuntimeError('CUDA out of memory')

    c = TestClient(ms.create_app(Boom(), {}))
    png = cv2.imencode('.png', np.full((30, 20), 5, np.uint8))[1].tobytes()
    image_id = c.post('/image', files={'file': ('x.png', png, 'image/png')}).json()['image_id']
    r = c.post('/segment', json={'image_id': image_id, 'box': [0, 0, 10, 10]})
    assert r.status_code == 500 and 'CUDA out of memory' in r.json()['detail']


def test_segmenter_uses_normalize_from_decoder_meta(monkeypatch):
    class M:
        medsam_meta = {'normalize': 'minmax'}

    monkeypatch.setattr(mr, 'load_medsam', lambda path=None: M())
    seg = ms.Segmenter(None).load()
    assert seg.normalize == 'minmax'
