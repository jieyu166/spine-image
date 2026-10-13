#!/usr/bin/env python
"""比較兩輪 OOF 評估：同一批案例的指標差值與 95% 信賴區間。

兩輪都是 out-of-fold，所以即使 fold 組成不同，每個案例在兩輪中都「沒被訓練到」，
可以逐案配對。重抽單位是病人群（同一病人多張影像不視為獨立）。
差值的信賴區間不含 0 才算有統計意義的變化；含 0 就是雜訊範圍。

用法：
  python compare_oof_runs.py \
      --baseline  endplate_training_data/runs/<A>/oof_eval/oof_predictions.json \
      --candidate endplate_training_data/runs/<B>/oof_eval/oof_predictions.json \
      [--output <B>/oof_eval/compare_vs_A.md] [--bootstrap 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from oof_metrics import paired_bootstrap_delta
from train_vertebra_model_cv import DEFAULT_GROUP_REGEX

METRIC_LABELS = (
    ("count_exact_match_rate", "Count exact", True),
    ("corner_mean_px", "Corner mean px", False),
    ("corner_median_px", "Corner median px", False),
    ("corner_p90_px", "Corner P90 px", False),
    ("middle_mean_px", "Middle mean px", False),
)

SUBGROUPS = (
    ("全部", lambda c: True),
    ("腰椎", lambda c: c.get("spine_type") == "L"),
    ("頸椎", lambda c: c.get("spine_type") == "C"),
    ("醫院影像（非 NHANES）", lambda c: "nhanes" not in str(c.get("image_path", "")).lower()),
)


def load_cases(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["cases"] if isinstance(data, dict) else data


def _fmt(value, is_rate, signed=False):
    if value is None:
        return "—"
    sign = "+" if signed and value > 0 else ""
    return f"{sign}{value * 100:.1f}%" if is_rate else f"{sign}{value:.1f}"


def compare(baseline, candidate, group_regex=DEFAULT_GROUP_REGEX, n_boot=2000, seed=0):
    """回傳 {subgroup: paired_bootstrap_delta 結果}。"""
    results = {}
    for name, keep in SUBGROUPS:
        base = [c for c in baseline if keep(c)]
        cand = [c for c in candidate if keep(c)]
        if not base or not cand:
            continue
        results[name] = paired_bootstrap_delta(base, cand, group_regex, n_boot=n_boot, seed=seed)
    return results


def to_markdown(results, baseline_label, candidate_label):
    lines = [f"# OOF 配對比較：{candidate_label} vs {baseline_label}", ""]
    first = next(iter(results.values()), None)
    if first:
        lines += [
            f"方法：{first['method']}，{first['n_boot']} 次重抽，95% 信賴區間。"
            "差值 = 新 − 舊；角點／中點誤差的差值為負代表改善。**區間不含 0** 才標記為顯著。",
            "",
        ]
        if first["only_in_baseline"] or first["only_in_candidate"]:
            lines += [
                f"只在舊輪：{len(first['only_in_baseline'])} 例；只在新輪：{len(first['only_in_candidate'])} 例（不納入配對）。",
                "",
            ]
    for name, res in results.items():
        lines += [
            f"## {name}（{res['n_common_cases']} 例、{res['n_groups']} 個病人群）",
            "",
            "| 指標 | 舊 | 新 | 差值 | 95% CI | 顯著 |",
            "|---|---:|---:|---:|---|:---:|",
        ]
        for key, label, is_rate in METRIC_LABELS:
            m = res["metrics"][key]
            if m["delta"] is None:
                continue
            ci = "—" if m["ci_low"] is None else f"{_fmt(m['ci_low'], is_rate, True)} – {_fmt(m['ci_high'], is_rate, True)}"
            lines.append(
                f"| {label} | {_fmt(m['baseline'], is_rate)} | {_fmt(m['candidate'], is_rate)} | "
                f"{_fmt(m['delta'], is_rate, True)} | {ci} | {'✔' if m['excludes_zero'] else ''} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, type=Path, help="舊輪 oof_predictions.json")
    parser.add_argument("--candidate", required=True, type=Path, help="新輪 oof_predictions.json")
    parser.add_argument("--output", type=Path, help="輸出 Markdown（另寫同名 .json）")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-regex", default=DEFAULT_GROUP_REGEX)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(f"讀取並重抽 {args.bootstrap} 次 ...", flush=True)
    results = compare(load_cases(args.baseline), load_cases(args.candidate),
                      args.group_regex, args.bootstrap, args.seed)
    label = lambda p: Path(p).resolve().parent.parent.name  # noqa: E731  <run>/oof_eval/oof_predictions.json
    report = to_markdown(results, label(args.baseline), label(args.candidate))
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        args.output.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"寫入 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
