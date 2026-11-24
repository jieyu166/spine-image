import json
from pathlib import Path

import pytest

import build_holdout_manifest as hm
import mark_physician_reviewed as mr
import regenerate_splits as splits


def _ann(spine='L', review=None, implant=False):
    data = {'version': '2.3', 'spineType': spine, 'vertebrae': [{'name': 'L1', 'points': {}}],
            'caseFlags': {'hasImplant': implant}}
    if review is not None:
        data['annotationReview'] = review
    return data


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / 'images'
    d.mkdir()
    for stem in ('90000004', '90000005C', '90000005L', '90000001', '90000003', '90000006'):
        (d / f'{stem}.png').write_bytes(stem.encode())
    (d / '90000004.json').write_text(json.dumps(_ann(review={'method': 'AI visual annotation', 'physicianReviewed': False})), encoding='utf-8')
    (d / '90000005C.json').write_text(json.dumps(_ann('C')), encoding='utf-8')
    (d / '90000005L.json').write_text(json.dumps(_ann(implant=True)), encoding='utf-8')
    return d


def test_manifest_classifies_from_json_and_explicit_lists_only(folder):
    m = hm.build(folder, lumbar=['90000001'], cervical=['90000003'], exclude={'90000006': 'AP'}, name='holdout_t')
    by = {e['stem']: e for e in m['entries']}
    assert by['90000004']['status'] == 'annotated_unreviewed'
    assert by['90000005L']['status'] == 'annotated' and by['90000005L']['has_implant'] is True
    assert by['90000001'] == {**by['90000001'], 'spine': 'L', 'status': 'to_annotate'}
    assert by['90000006']['status'] == 'excluded'
    assert m['summary'] == {'L': {'annotated': 1, 'annotated_unreviewed': 1, 'to_annotate': 1},
                            'C': {'annotated': 1, 'annotated_unreviewed': 0, 'to_annotate': 1},
                            'excluded': 1, 'unclassified': 0}
    assert m['groups'] == ['90000001', '90000003', '90000004', '90000005', '90000006']
    assert len(by['90000004']['image']['sha256']) == 64


def test_manifest_flags_unclassified_and_rejects_bad_lists(folder):
    m = hm.build(folder, name='holdout_t')
    assert m['summary']['unclassified'] == 3
    with pytest.raises(ValueError, match='同時'):
        hm.build(folder, lumbar=['90000001'], cervical=['90000001'])
    with pytest.raises(ValueError, match='找不到'):
        hm.build(folder, lumbar=['99999999'])
    with pytest.raises(ValueError, match='JSON 是 C'):
        hm.build(folder, lumbar=['90000005C'])


def test_worklist_lists_only_pending_items(folder):
    text = hm.worklist(hm.build(folder, lumbar=['90000001'], cervical=['90000003'], exclude={'90000006': 'AP'}, name='holdout_t'))
    assert '| 90000001 | L | 待標註 |' in text
    assert '| 90000004 | L | 已有 AI 標註，待醫師審 |' in text
    assert '90000005L' not in text and '90000006' not in text


def test_regenerate_splits_drops_holdout_patients(tmp_path, folder):
    mdir = tmp_path / 'manifests'
    mdir.mkdir()
    m = hm.build(folder, lumbar=['90000001'], cervical=['90000003'], exclude={'90000006': 'AP'}, name='holdout_t')
    (mdir / 'holdout_t.json').write_text(json.dumps(m), encoding='utf-8')
    (mdir / 'training_allowlist_x.json').write_text('{"entries": []}', encoding='utf-8')
    groups = hm.load_holdout_groups(mdir)
    assert groups['90000005'] == 'holdout_t'
    entries = [{'image_path': 'Images\\202610\\90000005-2.png'}, {'image_path': 'Images/12345678.png'}]
    kept, dropped = splits.drop_holdout_patients(entries, groups)
    assert [e['image_path'] for e in kept] == ['Images/12345678.png']
    assert dropped == [('Images\\202610\\90000005-2.png', 'holdout_t')]


def test_mark_physician_reviewed_only_touches_the_review_and_backs_up(folder, capsys):
    original = (folder / '90000004.json').read_text(encoding='utf-8')
    assert mr.main(['--dir', str(folder), '--via', 'viewer 審閱', '--dry-run']) == 0
    assert (folder / '90000004.json').read_text(encoding='utf-8') == original
    assert mr.main(['--dir', str(folder), '--via', 'viewer 審閱', '--stem', '90000004', '--stem', '90000005L']) == 0
    assert '90000005L' in capsys.readouterr().out  # 已是醫師已審 → 回報未處理
    data = json.loads((folder / '90000004.json').read_text(encoding='utf-8'))
    review = data.pop('annotationReview')
    assert review['physicianReviewed'] is True and review['previouslyUnreviewed'] is True
    assert review['method'] == 'AI visual annotation' and review['reviewedWith'] == 'viewer 審閱'
    before = json.loads(original)
    before.pop('annotationReview')
    assert data == before
    backups = list((folder / '_json_backups').glob('90000004.json.*.bak'))
    assert len(backups) == 1 and backups[0].read_text(encoding='utf-8') == original
