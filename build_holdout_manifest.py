#!/usr/bin/env python
"""建立外部測試集（holdout）清單：鎖定影像與標註的 SHA-256，並列出待標註的工作清單。

外部測試集的病人永遠不能進訓練。regenerate_splits.py 會讀 endplate_training_data/manifests/holdout_*.json，
把這些病人群從訓練／驗證 split 剔除並警告。

分類規則：已有 JSON 的用 JSON 內的 spineType；沒有 JSON 的由 --lumbar / --cervical / --exclude 明確指定
（未指定的會列為 unclassified，提醒補上），不從檔名或尺寸自行猜。

用法：
  python build_holdout_manifest.py --source-dir <影像資料夾> --name holdout_20261029 \
      --lumbar 90000001 90000002 ... --cervical 90000003 ... --exclude 90000006:AP_pelvis
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST_DIR = ROOT / 'endplate_training_data' / 'manifests'
GROUP_REGEX = re.compile(r'(?:^|/)(\d{8}|[CL]\d{5})[^/]*$')
IMAGE_EXTS = ('.png', '.jpg', '.jpeg')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_id(path_like: str) -> str | None:
    match = GROUP_REGEX.search(str(path_like).replace('\\', '/'))
    return match.group(1) if match else None


def build(source_dir: Path, lumbar=(), cervical=(), exclude=None, name='holdout'):
    exclude = exclude or {}
    explicit = {**{s: 'L' for s in lumbar}, **{s: 'C' for s in cervical}}
    overlap = set(lumbar) & set(cervical)
    if overlap:
        raise ValueError(f'同時被指定為腰椎與頸椎：{sorted(overlap)}')
    entries = []
    for image in sorted(p for p in source_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        stem = image.stem
        entry = {'stem': stem, 'group': group_id(image.name),
                 'image': {'path': str(image), 'sha256': sha256(image)}}
        js = image.with_suffix('.json')
        if stem in exclude:
            entry.update(spine=None, status='excluded', reason=exclude[stem])
        elif js.exists():
            data = json.loads(js.read_text(encoding='utf-8'))
            review = data.get('annotationReview') or {}
            flags = data.get('caseFlags') or {}
            spine = data.get('spineType')
            if stem in explicit and explicit[stem] != spine:
                raise ValueError(f'{stem}: 指定 {explicit[stem]} 但 JSON 是 {spine}')
            entry.update(
                spine=spine,
                status='annotated' if review.get('physicianReviewed', True) is not False else 'annotated_unreviewed',
                annotation={'path': str(js), 'sha256': sha256(js)},
                n_vertebrae=len(data.get('vertebrae', [])),
                review_method=review.get('method'),
                has_implant=bool(flags.get('hasImplant')),
            )
        elif stem in explicit:
            entry.update(spine=explicit[stem], status='to_annotate')
        else:
            entry.update(spine=None, status='unclassified')
        entries.append(entry)
    unknown = (set(explicit) | set(exclude)) - {e['stem'] for e in entries}
    if unknown:
        raise ValueError(f'指定的主檔名在資料夾裡找不到：{sorted(unknown)}')

    def count(spine, status):
        return sum(1 for e in entries if e['spine'] == spine and e['status'] == status)

    summary = {
        spine: {status: count(spine, status) for status in ('annotated', 'annotated_unreviewed', 'to_annotate')}
        for spine in ('L', 'C')
    }
    summary['excluded'] = sum(e['status'] == 'excluded' for e in entries)
    summary['unclassified'] = sum(e['status'] == 'unclassified' for e in entries)
    return {
        'version': '1.0', 'kind': 'holdout', 'name': name,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'source_dir': str(source_dir),
        'purpose': '外部測試集：只做最終評估，任何病人群不得進入訓練、調參或模型選擇',
        'groups': sorted({e['group'] for e in entries if e['group']}),
        'summary': summary,
        'entries': entries,
    }


def worklist(manifest) -> str:
    todo = [e for e in manifest['entries'] if e['status'] in ('to_annotate', 'annotated_unreviewed', 'unclassified')]
    lines = [f"# {manifest['name']} 待辦", '',
             f"來源：`{manifest['source_dir']}`", '',
             '| 主檔名 | 部位 | 狀態 |', '|---|---|---|']
    label = {'to_annotate': '待標註', 'annotated_unreviewed': '已有 AI 標註，待醫師審', 'unclassified': '未分類'}
    for e in sorted(todo, key=lambda e: (e['spine'] or 'Z', e['stem'])):
        lines.append(f"| {e['stem']} | {e['spine'] or '?'} | {label[e['status']]} |")
    s = manifest['summary']
    lines += ['', f"腰椎：已標註 {s['L']['annotated']}、待審 {s['L']['annotated_unreviewed']}、待標 {s['L']['to_annotate']}；"
                  f"頸椎：已標註 {s['C']['annotated']}、待審 {s['C']['annotated_unreviewed']}、待標 {s['C']['to_annotate']}；"
                  f"排除 {s['excluded']}、未分類 {s['unclassified']}。"]
    return '\n'.join(lines) + '\n'


def load_holdout_groups(manifest_dir: Path = MANIFEST_DIR) -> dict[str, str]:
    """{病人群: 清單名稱}，供 regenerate_splits.py 剔除外部測試集病人。"""
    groups = {}
    for path in sorted(Path(manifest_dir).glob('holdout_*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('kind') != 'holdout':
            continue
        for g in data.get('groups', []):
            groups[g] = data.get('name', path.stem)
    return groups


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--name', required=True, help='清單名稱，輸出 manifests/<name>.json；需以 holdout_ 開頭')
    parser.add_argument('--lumbar', nargs='*', default=[])
    parser.add_argument('--cervical', nargs='*', default=[])
    parser.add_argument('--exclude', nargs='*', default=[], help='主檔名:原因，例如 90000006:AP_pelvis')
    parser.add_argument('--output-dir', type=Path, default=MANIFEST_DIR)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if not args.name.startswith('holdout_'):
        parser.error('--name 需以 holdout_ 開頭（regenerate_splits.py 依此辨識）')
    exclude = dict(item.split(':', 1) if ':' in item else (item, 'excluded') for item in args.exclude)
    manifest = build(args.source_dir, args.lumbar, args.cervical, exclude, args.name)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / f'{args.name}.json'
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output_dir / f'{args.name}_worklist.md').write_text(worklist(manifest), encoding='utf-8')
    print(json.dumps(manifest['summary'], ensure_ascii=False))
    print(f'{len(manifest["groups"])} 個病人群；寫入 {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
