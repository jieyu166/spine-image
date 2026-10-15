#!/usr/bin/env python
"""學習曲線實驗：訓練張數 vs OOF 誤差，判斷「再加資料」是否還有用。

設計
- 固定一組病人隔離的 stratified K 折（與 train_vertebra_model_cv.py 相同的切分）。
- 每折的「訓練集」依病人群抽 fraction（例如 25/50/75/100%）：
  * 巢狀：25% ⊂ 50% ⊂ 75% ⊂ 100%，多出的張數都是「加進去」而非換一批；
  * 分層：每個分層（脊椎類型＋植入物）各取同比例；
  * 同病人的多張影像一起進出。
- 驗證集永遠是完整的那一折 → 各 fraction 評估的是同一批 OOF 案例，可逐案配對比較。
- 每個 fraction 一個子資料夾，裡面是標準 CV run 的結構（fold 權重、cv_results.json），
  再用 eval_oof.py 產生 OOF 指標與 95% 信賴區間，最後彙整成 learning_curve.md / .png。

中斷可續跑：已完成的 fold（cv_results.json 有紀錄且權重存在）會跳過。

用法：
  python learning_curve.py --annotations-dir <run>/annotations --out-dir endplate_training_data/runs/learning_curve_YYYYMMDD
      [--fractions 0.25 0.5 0.75 1.0] [--epochs 30] [--n-folds 5] [--skip-eval]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from train_vertebra_model_cv import (
    DEFAULT_GROUP_REGEX,
    build_group_folds,
    extract_group_id,
    fold_composition,
    load_all_annotations,
    stratum_label,
)

ROOT = Path(__file__).resolve().parent


# ── 純函式：巢狀分層子集 ──────────────────────────────────────────────
def nested_group_subsets(annotations, train_indices, fractions, seed=42, group_regex=DEFAULT_GROUP_REGEX):
    """回傳 {fraction: sorted(indices)}，依病人群巢狀、依分層等比例抽樣。

    每個分層內的病人群先以 seed 打亂成固定順序，fraction f 取前 ceil(f × n) 個群，
    因此小 fraction 一定是大 fraction 的子集；至少保留 1 個群。
    群的分層取該群第一張影像的 stratum_label。
    """
    for f in fractions:
        if not 0 < f <= 1:
            raise ValueError(f"fraction must be in (0, 1]: {f}")
    groups: dict[str, list[int]] = {}
    for index in train_indices:
        groups.setdefault(extract_group_id(annotations[index], group_regex), []).append(index)
    strata: dict[str, list[str]] = {}
    for group_id in sorted(groups):
        label = stratum_label(annotations[groups[group_id][0]])
        strata.setdefault(label, []).append(group_id)
    rng = random.Random(seed)
    ordered = {}
    for label in sorted(strata):
        ids = list(strata[label])
        rng.shuffle(ids)
        ordered[label] = ids
    subsets = {}
    for f in sorted(fractions):
        chosen = []
        for ids in ordered.values():
            take = max(1, math.ceil(f * len(ids))) if f < 1 else len(ids)
            for group_id in ids[:take]:
                chosen.extend(groups[group_id])
        subsets[f] = sorted(chosen)
    return subsets


def fraction_dirname(fraction):
    return f"frac_{int(round(fraction * 100)):03d}"


# ── 訓練 ────────────────────────────────────────────────────────────
def _seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _train_config(args):
    from train_vertebra_model import RADIMAGENET_PATH_DEFAULT, VertebraDataset

    pretrained = args.pretrained_path or str(ROOT / RADIMAGENET_PATH_DEFAULT)
    if not Path(pretrained).is_file():
        raise FileNotFoundError(f"RadImageNet weights not found: {pretrained}")
    return {
        'data_dir': str(ROOT),
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': 1e-3,
        'weight_decay': 1e-3,
        'num_workers': 0,
        'max_vertebrae': 8,
        'points_per_vertebra': VertebraDataset.POINTS_PER_VERTEBRA,
        'repeat_dataset': 8,
        'pretrained_path': str(Path(pretrained).resolve()),
        'unfreeze_epoch': 5,
        'stage_b_lr': 1e-4,
    }


def train_fraction(fraction, annotations, folds, subsets_by_fold, out_dir, args, log):
    """在 out_dir/frac_xxx 以標準 CV 結構訓練 K 折；可續跑。"""
    import torch
    import train_vertebra_model_cv as cv

    run_dir = out_dir / fraction_dirname(fraction)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / 'cv_results.json'
    done = {}
    if results_path.exists():
        previous = json.loads(results_path.read_text(encoding='utf-8'))
        for item in previous.get('folds', []):
            if (run_dir / f"best_vertebra_model_fold{item['fold']}.pth").exists():
                done[item['fold']] = item
    config = _train_config(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    members = [[str(annotations[i].get('image_path', '')).replace('\\', '/') for i in val] for _, val in folds]

    def dump():
        ordered = [done[k] for k in sorted(done)]
        losses = [item['best_val_loss'] for item in ordered]
        payload = {
            'experiment': 'learning_curve',
            'fraction': fraction,
            'n_folds_total': len(folds),
            'n_folds_completed': len(ordered),
            'epochs_per_fold': args.epochs,
            'seed': args.seed,
            'fold_grouping': args.group_regex,
            'fold_strategy': 'stratified',
            'fold_val_members': members,
            'fold_val_composition': fold_composition(annotations, folds),
            'fold_best_val_losses': losses,
            'mean_best_val_loss': float(np.mean(losses)) if losses else None,
            'folds': ordered,
        }
        results_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    cwd = os.getcwd()
    os.chdir(run_dir)  # run_one_fold 把權重寫在 cwd
    try:
        for fold_idx, (_, val_idx) in enumerate(folds):
            fold = fold_idx + 1
            if fold in done:
                log(f"  [{fraction_dirname(fraction)}] fold {fold} 已完成，跳過")
                continue
            train_idx = subsets_by_fold[fold_idx][fraction]
            train_anns = [annotations[i] for i in train_idx]
            val_anns = [annotations[i] for i in val_idx]
            n_groups = len({extract_group_id(a, args.group_regex) for a in train_anns})
            log(f"  [{fraction_dirname(fraction)}] fold {fold}/{len(folds)} 開始：訓練 {len(train_anns)} 張 / "
                f"{n_groups} 位病人，驗證 {len(val_anns)} 張")
            _seed_everything(args.seed + fold)
            started = time.time()
            result = cv.run_one_fold(fold_idx, len(folds), train_anns, val_anns, config, device)
            done[fold] = {
                'fold': fold,
                'best_val_loss': float(result['best_val_loss']),
                'n_train_images': len(train_anns),
                'n_train_groups': n_groups,
                'n_val_images': len(val_anns),
                'train_members': [str(a.get('image_path', '')).replace('\\', '/') for a in train_anns],
                'elapsed_seconds': round(time.time() - started, 1),
            }
            dump()
            log(f"  [{fraction_dirname(fraction)}] fold {fold} 完成：best val loss {result['best_val_loss']:.4f}，"
                f"{(time.time() - started) / 60:.1f} 分鐘")
        dump()
    finally:
        os.chdir(cwd)
    return run_dir


def evaluate_fraction(run_dir, annotations_dir, args, log):
    """呼叫 eval_oof.py（子程序，避免 GPU 記憶體殘留）。"""
    if (run_dir / 'oof_eval' / 'oof_predictions.json').exists():
        log(f"  [{run_dir.name}] OOF 已存在，跳過評估")
        return
    command = [
        sys.executable, '-u', str(ROOT / 'eval_oof.py'),
        '--run-dir', str(run_dir),
        '--project-root', str(ROOT),
        '--train-annotations', str(annotations_dir / 'train_annotations.json'),
        '--val-annotations', str(annotations_dir / 'val_annotations.json'),
        '--n-folds', str(args.n_folds), '--seed', str(args.seed),
        '--fold-strategy', 'stratified',
        '--skip-production-reference', '--worst-count', '0',
    ]
    log(f"  [{run_dir.name}] OOF 評估開始")
    with open(run_dir / 'eval.log', 'w', encoding='utf-8') as out, open(run_dir / 'eval.err.log', 'w', encoding='utf-8') as err:
        code = subprocess.run(command, stdout=out, stderr=err, env=dict(os.environ, PYTHONUTF8='1')).returncode
    if code:
        raise RuntimeError(f"eval_oof failed for {run_dir} (exit {code}); see eval.err.log")
    log(f"  [{run_dir.name}] OOF 評估完成")


# ── 彙整 ────────────────────────────────────────────────────────────
def summarize(out_dir, fractions, args, log):
    from compare_oof_runs import compare
    from oof_metrics import bootstrap_ci

    rows = []
    cases_by_fraction = {}
    for f in fractions:
        run_dir = out_dir / fraction_dirname(f)
        oof_path = run_dir / 'oof_eval' / 'oof_predictions.json'
        if not oof_path.exists():
            continue
        cases = json.loads(oof_path.read_text(encoding='utf-8'))['cases']
        cases_by_fraction[f] = cases
        cv_results = json.loads((run_dir / 'cv_results.json').read_text(encoding='utf-8'))
        n_train = float(np.mean([item['n_train_images'] for item in cv_results['folds']]))
        row = {'fraction': f, 'n_train_images_mean': n_train, 'mean_best_val_loss': cv_results['mean_best_val_loss']}
        for name, keep in (('all', lambda c: True), ('L', lambda c: c.get('spine_type') == 'L')):
            row[name] = bootstrap_ci([c for c in cases if keep(c)], args.group_regex, n_boot=args.bootstrap, seed=args.seed)
        rows.append(row)
    if not rows:
        log('沒有可彙整的 OOF 結果')
        return None

    full = max(cases_by_fraction)
    deltas = {f: compare(cases_by_fraction[f], cases_by_fraction[full], args.group_regex, args.bootstrap, args.seed)
              for f in cases_by_fraction if f != full}

    def cell(ci, key, rate=False):
        m = ci['metrics'][key]
        if m['estimate'] is None:
            return '—'
        fmt = (lambda v: f"{v * 100:.1f}%") if rate else (lambda v: f"{v:.1f}")
        return f"{fmt(m['estimate'])} ({fmt(m['ci_low'])}–{fmt(m['ci_high'])})"

    lines = [
        '# 學習曲線：訓練張數 vs OOF 誤差', '',
        f"資料：`{args.annotations_dir}`，{len(next(iter(cases_by_fraction.values())))} 例；"
        f"{args.n_folds} 折 stratified 病人隔離，每折訓練集依病人群巢狀分層抽樣，驗證集固定為完整一折。"
        f"每折 {args.epochs} epochs。括號為 95% 病人群集 bootstrap 信賴區間。", '',
        '## 腰椎', '',
        '| 訓練比例 | 平均訓練張數 | Count exact | 角點平均 px | 角點中位數 px | 角點 P90 px | val loss |',
        '|---:|---:|---|---|---|---|---:|',
    ]
    for r in rows:
        ci = r['L']
        lines.append(f"| {r['fraction'] * 100:.0f}% | {r['n_train_images_mean']:.1f} | {cell(ci, 'count_exact_match_rate', True)} | "
                     f"{cell(ci, 'corner_mean_px')} | {cell(ci, 'corner_median_px')} | {cell(ci, 'corner_p90_px')} | "
                     f"{r['mean_best_val_loss']:.3f} |")
    lines += ['', '## 全部（含頸椎）', '',
              '| 訓練比例 | Count exact | 角點平均 px | 角點中位數 px |', '|---:|---|---|---|']
    for r in rows:
        ci = r['all']
        lines.append(f"| {r['fraction'] * 100:.0f}% | {cell(ci, 'count_exact_match_rate', True)} | "
                     f"{cell(ci, 'corner_mean_px')} | {cell(ci, 'corner_median_px')} |")
    if deltas:
        lines += ['', f"## 與 {full * 100:.0f}% 的配對差值（{full * 100:.0f}% − 該比例；負值代表加資料後誤差下降）", '',
                  '| 比較 | 子群 | 角點中位數差 (95% CI) | 角點平均差 (95% CI) | Count exact 差 (95% CI) |',
                  '|---|---|---|---|---|']
        for f, res in sorted(deltas.items()):
            for sub in ('腰椎', '全部'):
                if sub not in res:
                    continue
                m = res[sub]['metrics']

                def d(key, rate=False):
                    x = m[key]
                    fmt = (lambda v: f"{v * 100:+.1f}%") if rate else (lambda v: f"{v:+.1f}")
                    star = ' ✔' if x['excludes_zero'] else ''
                    return f"{fmt(x['delta'])} ({fmt(x['ci_low'])}–{fmt(x['ci_high'])}){star}"
                lines.append(f"| {f * 100:.0f}% → {full * 100:.0f}% | {sub} | {d('corner_median_px')} | "
                             f"{d('corner_mean_px')} | {d('count_exact_match_rate', True)} |")
        lines += ['', '✔ = 95% CI 不含 0。']
    report = '\n'.join(lines) + '\n'
    (out_dir / 'learning_curve.md').write_text(report, encoding='utf-8')
    (out_dir / 'learning_curve.json').write_text(json.dumps({'rows': rows}, ensure_ascii=False, indent=1), encoding='utf-8')
    _plot(rows, out_dir / 'learning_curve.png')
    log(f"彙整完成：{out_dir / 'learning_curve.md'}")
    return report


def _plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x = [r['n_train_images_mean'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in ((axes[0], 'corner_median_px', 'Lumbar corner median error (px)'),
                           (axes[1], 'count_exact_match_rate', 'Lumbar count exact rate')):
        est = [r['L']['metrics'][key]['estimate'] for r in rows]
        lo = [r['L']['metrics'][key]['ci_low'] for r in rows]
        hi = [r['L']['metrics'][key]['ci_high'] for r in rows]
        ax.plot(x, est, marker='o', color='#2563eb')
        ax.fill_between(x, lo, hi, color='#2563eb', alpha=0.15, linewidth=0)
        ax.set_xlabel('Training images per fold (mean)')
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ── 主程式 ──────────────────────────────────────────────────────────
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--annotations-dir', type=Path, required=True, help='含 train_annotations.json / val_annotations.json')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--fractions', type=float, nargs='+', default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--group-regex', default=DEFAULT_GROUP_REGEX)
    parser.add_argument('--pretrained-path', default=None)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--plan-only', action='store_true', help='只印出每折每比例的張數，不訓練')
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    # 訓練時會 chdir 到各 fraction 資料夾，所有路徑先轉絕對
    args.out_dir = args.out_dir.resolve()
    args.annotations_dir = args.annotations_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / 'learning_curve.log'

    def log(message):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(log_path, 'a', encoding='utf-8') as handle:
            handle.write(line + '\n')

    fractions = sorted(set(args.fractions))
    if args.summarize_only:
        summarize(args.out_dir, fractions, args, log)
        return 0

    annotations = load_all_annotations(str(args.annotations_dir / 'train_annotations.json'),
                                       str(args.annotations_dir / 'val_annotations.json'))
    folds = build_group_folds(annotations, args.n_folds, args.seed, args.group_regex, 'stratified')
    subsets_by_fold = [nested_group_subsets(annotations, train, fractions, args.seed, args.group_regex)
                       for train, _ in folds]
    plan = {fraction_dirname(f): [len(s[f]) for s in subsets_by_fold] for f in fractions}
    log(f"{len(annotations)} 張，{args.n_folds} 折；各比例每折訓練張數：{plan}")
    (args.out_dir / 'plan.json').write_text(json.dumps({
        'annotations_dir': str(args.annotations_dir), 'n_annotations': len(annotations),
        'fractions': fractions, 'train_images_per_fold': plan,
        'fold_val_composition': fold_composition(annotations, folds),
        'epochs': args.epochs, 'seed': args.seed,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    if args.plan_only:
        return 0

    for f in fractions:  # 小的先跑，前幾個點早點出來
        log(f"=== 訓練比例 {f * 100:.0f}% ===")
        run_dir = train_fraction(f, annotations, folds, subsets_by_fold, args.out_dir, args, log)
        if not args.skip_eval:
            evaluate_fraction(run_dir, args.annotations_dir.resolve(), args, log)
            summarize(args.out_dir, fractions, args, log)
    log('學習曲線實驗完成')
    return 0


if __name__ == '__main__':
    sys.exit(main())
