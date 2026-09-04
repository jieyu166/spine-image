import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from propose_legacy_upgrade import build_proposal, find_legacy_stems, is_legacy_lspine  # noqa: E402


def _corners(y_top, y_bottom, x0=100, x1=300):
    return {
        'anteriorSuperior': {'x': x0, 'y': y_top},
        'posteriorSuperior': {'x': x1, 'y': y_top},
        'posteriorInferior': {'x': x1, 'y': y_bottom},
        'anteriorInferior': {'x': x0, 'y': y_bottom},
    }


def _legacy_v20():
    """v2.0 舊版：由下而上、S1 標上下緣、沒有 T12、沒有中點。"""
    names = ['S1', 'L5', 'L4', 'L3', 'L2', 'L1']
    vertebrae = []
    for i, name in enumerate(names):
        top = 1000 - i * 150
        vertebrae.append({'name': name, 'points': _corners(top, top + 100), 'anteriorWedgingFracture': False})
    return {'version': '2.0', 'spineType': 'L', 'imageInfo': {'width': 1440, 'height': 3408},
            'vertebrae': vertebrae, 'discs': [], 'abnormalities': {}}


def _oof_case():
    def six(top):
        pts = _corners(top, top + 100)
        pts['middleSuperior'] = {'x': 200, 'y': top + 5}
        pts['middleInferior'] = {'x': 200, 'y': top + 95}
        return pts
    # L1 上緣 y=250、高 100 → T12 下緣放在 y≈200（椎間盤高 50）才合理
    verts = [{'name': 'T12', 'boundaryType': 'lower',
              'points': {'posteriorInferior': {'x': 300, 'y': 200}, 'middleInferior': {'x': 200, 'y': 205},
                         'anteriorInferior': {'x': 100, 'y': 200}}}]
    for i, name in enumerate(['L1', 'L2', 'L3', 'L4', 'L5']):
        verts.append({'name': name, 'boundaryType': None, 'points': six(250 + i * 150)})
    verts.append({'name': 'S1', 'boundaryType': 'upper',
                  'points': {'anteriorSuperior': {'x': 100, 'y': 1000}, 'middleSuperior': {'x': 200, 'y': 1004},
                             'posteriorSuperior': {'x': 300, 'y': 1000}}})
    return {'case_id': 'X', 'fold': 3, 'status': 'success', 'prediction': {'vertebrae': verts}}


def test_is_legacy_detects_missing_t12_or_s1_inferior():
    assert is_legacy_lspine(_legacy_v20())
    modern = {'spineType': 'L', 'vertebrae': [{'name': 'T12', 'points': {}}, {'name': 'S1', 'points': {'anteriorSuperior': {}}}]}
    assert not is_legacy_lspine(modern)
    assert not is_legacy_lspine({'spineType': 'C', 'vertebrae': [{'name': 'S1', 'points': {}}]})


def test_build_proposal_upgrades_convention_without_moving_physician_points():
    src = _legacy_v20()
    frozen = copy.deepcopy(src)
    proposal, changes = build_proposal(src, _oof_case(), '21584353', 'Images/21584353.json', 'run')

    assert src == frozen, 'input must not be mutated'
    names = [v['name'] for v in proposal['vertebrae']]
    assert names == ['T12', 'L1', 'L2', 'L3', 'L4', 'L5', 'S1'], 'top-down order'

    s1 = proposal['vertebrae'][-1]
    assert s1['boundaryType'] == 'upper'
    assert not any(k.endswith('Inferior') for k in s1['points'])
    assert s1['pointSources']['anteriorSuperior'] == 'physician'
    assert s1['pointSources']['middleSuperior'] == 'chord_midpoint'
    assert s1['points']['middleSuperior'] == {'x': 200.0, 'y': 1000.0}

    t12 = proposal['vertebrae'][0]
    assert t12['boundaryType'] == 'lower'
    assert set(t12['points']) == {'posteriorInferior', 'middleInferior', 'anteriorInferior'}
    assert set(t12['pointSources'].values()) == {'model_oof_fold3'}

    l3_src = next(v for v in frozen['vertebrae'] if v['name'] == 'L3')
    l3_new = next(v for v in proposal['vertebrae'] if v['name'] == 'L3')
    for k, p in l3_src['points'].items():
        assert l3_new['points'][k] == p, 'physician corners unchanged'
        assert l3_new['pointSources'][k] == 'physician'
    assert l3_new['pointSources']['middleSuperior'] == 'chord_midpoint'

    assert [d['level'] for d in proposal['discs']] == ['T12/L1', 'L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
    assert proposal['version'] == '2.3'
    assert proposal['reviewProposal']['status'] == 'proposed'
    assert proposal['reviewProposal']['changes'] == changes
    assert proposal['imageInfo']['filename'] == '21584353.png'


def test_build_proposal_rejects_model_t12_that_lands_on_l1_or_below():
    case = _oof_case()
    t12 = next(v for v in case['prediction']['vertebrae'] if v['name'] == 'T12')
    for p in t12['points'].values():
        p['y'] = 400  # 落在 L1 體內
    proposal, changes = build_proposal(_legacy_v20(), case, 's', 'p', 'run')
    assert 'T12' not in [v['name'] for v in proposal['vertebrae']]
    assert any('不在 L1 之上' in c for c in changes)


def test_build_proposal_without_oof_reports_missing_t12():
    proposal, changes = build_proposal(_legacy_v20(), None, 's', 'p', 'run')
    assert 'T12' not in [v['name'] for v in proposal['vertebrae']]
    assert any('未補' in c for c in changes)


def test_find_legacy_stems_skips_nhanes_and_overlays(tmp_path):
    (tmp_path / '20260218NHANES').mkdir()
    (tmp_path / 'a.json').write_text(json.dumps(_legacy_v20()), encoding='utf-8')
    (tmp_path / 'asamp.json').write_text(json.dumps(_legacy_v20()), encoding='utf-8')
    (tmp_path / '20260218NHANES' / 'L1_x.json').write_text(json.dumps(_legacy_v20()), encoding='utf-8')
    (tmp_path / 'bad.json').write_text('{not json', encoding='utf-8')
    assert find_legacy_stems(tmp_path) == ['a']
    assert find_legacy_stems(tmp_path, include_nhanes=True) == ['20260218NHANES/L1_x', 'a']
