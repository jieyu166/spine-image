#!/usr/bin/env python
"""把醫師已在 annotation-viewer.html 審過、但內容沒改（所以從未重新存檔）的 JSON 標成「醫師已審」。

只改 annotationReview 欄位：physicianReviewed → true，記錄審閱時間與方式，保留原 method / note；
其他欄位逐字不動。改寫前把舊檔備份到 <資料夾>/_json_backups/<檔名>.<時間>.bak（與標註工具相同格式）。

用法：
  python mark_physician_reviewed.py --dir <images 資料夾> --via "annotation-viewer 疊圖審閱" [--stem 90000004 ...] [--dry-run]
未指定 --stem 時，處理資料夾內所有 annotationReview.physicianReviewed == false 的 JSON。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def mark(data: dict, via: str, when: str) -> dict:
    review = dict(data.get('annotationReview') or {})
    previously = review.get('physicianReviewed')
    if 'note' in review:
        review['priorNote'] = review.pop('note')
    review.update({
        'method': review.get('method') or 'unknown',
        'physicianReviewed': True,
        'reviewedAt': when,
        'reviewedWith': via,
    })
    if previously is False:
        review['previouslyUnreviewed'] = True
    out = dict(data)
    out['annotationReview'] = review
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dir', type=Path, required=True)
    parser.add_argument('--via', required=True, help='審閱方式，會寫進 annotationReview.reviewedWith')
    parser.add_argument('--stem', action='append', help='只處理指定主檔名（可重複）')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    now = datetime.now()
    when = now.astimezone(timezone.utc).isoformat(timespec='seconds')
    stamp = now.strftime('%Y%m%d-%H%M%S')
    targets = []
    for path in sorted(args.dir.glob('*.json')):
        if args.stem and path.stem not in args.stem:
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        review = data.get('annotationReview')
        if isinstance(review, dict) and review.get('physicianReviewed') is False:
            targets.append((path, data))
    missing = sorted(set(args.stem or []) - {p.stem for p, _ in targets})
    if missing:
        print(f'未處理（找不到或已是醫師已審）：{missing}')
    backup_dir = args.dir / '_json_backups'
    for path, data in targets:
        print(f"{'[dry-run] ' if args.dry_run else ''}{path.name}: physicianReviewed false → true")
        if args.dry_run:
            continue
        backup_dir.mkdir(exist_ok=True)
        (backup_dir / f'{path.name}.{stamp}.bak').write_bytes(path.read_bytes())
        path.write_text(json.dumps(mark(data, args.via, when), ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'共 {len(targets)} 份' + ('（未寫入）' if args.dry_run else f'；備份在 {backup_dir}'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
