"""export_medsam_onnx.py 的產出檢查。沒有匯出過（沒有 meta.json）時略過。"""
import json

import numpy as np
import pytest

import export_medsam_onnx as ex

META = ex.DEFAULT_OUT / 'meta.json'
pytestmark = pytest.mark.skipif(not META.exists(), reason='尚未執行 export_medsam_onnx.py')


@pytest.fixture(scope='module')
def meta():
    return json.loads(META.read_text(encoding='utf-8'))


def test_files_match_recorded_hashes(meta):
    for key, info in meta['files'].items():
        path = ex.DEFAULT_OUT / info['name']
        assert path.stat().st_size == info['bytes'], key
        assert ex.sha256(path) == info['sha256'], key


def test_parity_with_pytorch(meta):
    assert meta['checks']['encoder']['mask_iou_vs_torch'] > 0.98
    assert meta['checks']['encoder']['embedding_max_abs_diff'] < 1e-2
    for key in ('encoder_int8', 'encoder_fp16'):
        if key not in meta['checks']:
            continue
        assert meta['checks'][key]['mask_iou_vs_torch'] > 0.9, key


def test_preprocess_settings_match_training(meta):
    import medsam_refine as mr

    p = meta['preprocess']
    assert (p['side'], p['context'], p['box_margin'], p['lowres']) == (mr.SIDE, mr.CONTEXT, mr.BOX_MARGIN, mr.LOWRES)
    assert meta['normalize'] in ('imagenet', 'minmax')


def test_decoder_io_shapes(meta):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ex.DEFAULT_OUT / meta['files']['decoder']['name']), providers=['CPUExecutionProvider'])
    assert [i.name for i in sess.get_inputs()] == ['image_embeddings', 'input_boxes']
    emb = np.random.default_rng(0).standard_normal((1, 256, 64, 64)).astype(np.float32)
    masks, iou = sess.run(None, {'image_embeddings': emb, 'input_boxes': np.array([[[300, 300, 700, 650]]], np.float32)})
    assert masks.shape == (1, 256, 256) and iou.shape == (1,)
    assert np.isfinite(masks).all() and np.isfinite(iou).all()


def test_sha256_helper(tmp_path):
    f = tmp_path / 'a.bin'
    f.write_bytes(b'abc')
    assert ex.sha256(f) == 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
