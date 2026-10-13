import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qa_naming_shift import analyze_case, shifted_distance, ORDER  # noqa: E402


def _chain(names, y0=100, step=100, dx=0):
    return [{'name': n, 'points': {'anteriorSuperior': {'x': 100 + dx, 'y': y0 + i * step},
                                   'posteriorSuperior': {'x': 300 + dx, 'y': y0 + i * step}}}
            for i, n in enumerate(names)]


def _case(gt, pred, spine='L'):
    return {'case_id': 'x', 'status': 'success', 'spine_type': spine, 'fold': 1, 'image_path': 'x.png',
            'ground_truth': {'vertebrae': gt}, 'prediction': {'vertebrae': pred}}


def test_same_names_at_same_place_is_ok():
    gt = _chain(['T12', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1'])
    pred = _chain(['T12', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1'], dx=5)
    r = analyze_case(_case(gt, pred))
    assert r['verdict'] == 'ok'
    assert r['d_same'] == 5.0


def test_prediction_named_one_level_higher_is_flagged():
    # 模型從 y=200 起叫 T12：模型每一節都比 GT 高一級（模型多數了一節，或 GT 整條高一節）
    gt = _chain(['T12', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1'], y0=100)
    pred = _chain(['T11', 'T12', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1'], y0=100)[1:]  # 模型從 y=200 起叫 T12
    pred = _chain(['T12', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1'], y0=200)
    r = analyze_case(_case(gt, pred))
    assert r['verdict'] == 'pred_one_level_higher'
    assert r['d_gt_up'] == 0.0 and r['d_same'] == 100.0


def test_too_few_matches_returns_none_and_failed_cases_skipped():
    gt = _chain(['L5', 'S1'])
    assert analyze_case(_case(gt, gt)) is None
    failed = _case(gt, gt)
    failed['status'] = 'failed'
    assert analyze_case(failed) is None


def test_cervical_uses_cervical_order():
    gt = _chain(['C2', 'C3', 'C4', 'C5', 'C6'])
    pred = _chain(['C3', 'C4', 'C5', 'C6', 'C7'])
    d, n = shifted_distance(gt, pred, +1, ORDER['C'])
    assert d == 0.0 and n == 5
    assert analyze_case(_case(gt, pred, 'C'))['verdict'] == 'pred_one_level_lower'
