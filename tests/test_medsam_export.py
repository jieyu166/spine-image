"""export_medsam_onnx.py：外部測試集阻擋清單必須完整，否則停止匯出。"""
import hashlib
import json

import cv2
import numpy as np
import pytest

import export_medsam_onnx as ex


def _manifest(tmp_path, entries, name='holdout_x'):
    d = tmp_path / 'manifests'
    d.mkdir(exist_ok=True)
    (d / f'{name}.json').write_text(json.dumps({'kind': 'holdout', 'name': name, 'entries': entries}), encoding='utf-8')
    return d


def _png(tmp_path, name, arr):
    path = tmp_path / name
    path.write_bytes(cv2.imencode('.png', arr)[1].tobytes())
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_rchannel_fingerprint_matches_browser_definition():
    gray = (np.arange(12) * 20).astype(np.uint8).reshape(3, 4)
    bgra = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA)
    want = hashlib.sha256(b'3x4:' + gray.tobytes()).hexdigest()   # = JS：`${h}x${w}:` + R 通道
    assert ex.rchannel_fingerprint(gray) == want
    assert ex.rchannel_fingerprint(bgra) == want
    assert ex.rchannel_fingerprint(np.zeros((3, 4), np.uint16)) is None


def test_holdout_hashes_complete(tmp_path):
    p, sha = _png(tmp_path, 'a.png', np.full((5, 6), 7, np.uint8))
    d = _manifest(tmp_path, [{'image': {'path': str(p), 'sha256': sha.upper()}}])
    h = ex.holdout_hashes(d)
    assert h['files'] == {sha: 'holdout_x'} and len(h['pixels']) == 1 and h['manifests'] == ['holdout_x']


@pytest.mark.parametrize('case', ['no_manifest', 'missing_image', 'jpeg', 'sixteen_bit', 'translucent', 'no_sha'])
def test_holdout_hashes_refuses_incomplete_lists(tmp_path, case):
    if case == 'no_manifest':
        d = tmp_path / 'empty'
        d.mkdir()
    elif case == 'missing_image':
        d = _manifest(tmp_path, [{'image': {'path': str(tmp_path / 'gone.png'), 'sha256': 'ab'}}])
    elif case == 'jpeg':
        path = tmp_path / 'a.jpg'
        path.write_bytes(cv2.imencode('.jpg', np.full((5, 6), 7, np.uint8))[1].tobytes())
        d = _manifest(tmp_path, [{'image': {'path': str(path), 'sha256': 'ab'}}])
    elif case == 'sixteen_bit':
        p, sha = _png(tmp_path, 'a.png', np.full((5, 6), 700, np.uint16))
        d = _manifest(tmp_path, [{'image': {'path': str(p), 'sha256': sha}}])
    elif case == 'translucent':
        rgba = np.full((5, 6, 4), 100, np.uint8)
        rgba[0, 0, 3] = 10
        p, sha = _png(tmp_path, 'a.png', rgba)
        d = _manifest(tmp_path, [{'image': {'path': str(p), 'sha256': sha}}])
    else:
        p, _ = _png(tmp_path, 'a.png', np.full((5, 6), 7, np.uint8))
        d = _manifest(tmp_path, [{'image': {'path': str(p)}}])
    with pytest.raises(ex.HoldoutError):
        ex.holdout_hashes(d)


def test_export_stops_without_decoder(tmp_path):
    with pytest.raises(SystemExit, match='allow-zeroshot'):
        ex.main(['--decoder', str(tmp_path / 'missing.pth'), '--out', str(tmp_path / 'out')])
    assert not (tmp_path / 'out').exists()


def test_pick_check_pair_skips_holdout_and_incomplete(tmp_path):
    full = {'vertebrae': [{'name': 'L1', 'points': {k: {'x': 1, 'y': 2} for k in ex.mr.CORNERS}}]}
    partial = {'vertebrae': [{'name': 'S1', 'points': {'anteriorSuperior': {'x': 1, 'y': 2}}}]}
    a, sha_a = _png(tmp_path, '90000001.png', np.full((5, 6), 1, np.uint8))   # 外部測試集
    b, _ = _png(tmp_path, '90000002.png', np.full((5, 6), 2, np.uint8))       # 不完整
    c, _ = _png(tmp_path, '90000003.png', np.full((5, 6), 3, np.uint8))       # 應選這張
    for path, data in ((a, full), (b, partial), (c, full)):
        path.with_suffix('.json').write_text(json.dumps(data), encoding='utf-8')
    assert ex.pick_check_pair(tmp_path, {sha_a: 'holdout_x'}) == (c, c.with_suffix('.json'))
    with pytest.raises(SystemExit):
        ex.pick_check_pair(tmp_path, {sha_a: 'h', ex.sha256(c): 'h'})
