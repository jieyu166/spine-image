#!/usr/bin/env python3
"""Leakage-safe patient-grouped OOF evaluation for six-point checkpoints."""

import argparse
import csv
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import torch

from compare_annotations import compare_vertebrae, draw_overlay
from inference_vertebra import VertebraInference
from oof_metrics import (
    compare_case_landmarks,
    rank_worst_cases,
    summarize_cases,
    validate_fold_assignments,
)
from train_vertebra_model_cv import (
    DEFAULT_GROUP_REGEX,
    load_all_annotations,
    manual_group_kfold,
)


@dataclass(frozen=True)
class EvalConfig:
    project_root: Path
    run_dir: Path
    train_annotations: Path
    val_annotations: Path
    output_dir: Path
    production_model: Path | None = None
    n_folds: int = 5
    seed: int = 42
    group_regex: str = DEFAULT_GROUP_REGEX
    device: str = "auto"
    threshold: float = 0.2
    tta: bool = True
    worst_count: int = 10
    skip_production_reference: bool = False

    def __post_init__(self):
        for name in (
            "project_root",
            "run_dir",
            "train_annotations",
            "val_annotations",
            "output_dir",
            "production_model",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))


def resolve_image_path(annotation, project_root):
    """Resolve an annotation image without silently substituting another case."""
    root = Path(project_root)
    image_path = str(annotation.get("image_path", ""))
    normalized = image_path.replace("\\", os.sep).replace("/", os.sep)
    direct = root / normalized
    if normalized and direct.is_file():
        return direct.resolve()

    candidate_stems = []
    if normalized:
        candidate_stems.append(Path(normalized).stem)
    source_file = str(annotation.get("source_file", ""))
    if source_file:
        source_normalized = source_file.replace("\\", os.sep).replace("/", os.sep)
        candidate_stems.append(Path(source_normalized).stem)

    for stem in dict.fromkeys(candidate_stems):
        for extension in (".dcm", ".png", ".jpg", ".jpeg"):
            candidate = root / "Images" / f"{stem}{extension}"
            if candidate.is_file():
                return candidate.resolve()

    raise FileNotFoundError(
        f"cannot resolve image for annotation image_path={image_path!r}"
    )


def validate_config(config):
    """Fail before inference when inputs or output isolation are unsafe."""
    run_dir = config.run_dir.resolve()
    output_dir = config.output_dir.resolve()
    fold_paths = [
        run_dir / f"best_vertebra_model_fold{fold_number}.pth"
        for fold_number in range(1, config.n_folds + 1)
    ]
    protected = set(path.resolve() for path in fold_paths)
    if config.production_model is not None:
        production = config.production_model.resolve()
        protected.add(production)
        protected.add(Path(f"{production}.ensemble").resolve())

    if output_dir in protected:
        raise ValueError(f"output path would overwrite protected model: {output_dir}")
    if not output_dir.is_relative_to(run_dir) or output_dir == run_dir:
        raise ValueError(
            f"output directory must be inside run directory: {run_dir}"
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output path exists and is not a directory: {output_dir}")

    for annotation_path in (
        config.train_annotations,
        config.val_annotations,
    ):
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"annotation file not found: {annotation_path}"
            )
    for fold_path in fold_paths:
        if not fold_path.is_file():
            raise FileNotFoundError(
                f"missing checkpoint {fold_path.stem}: {fold_path}"
            )
    if (
        config.production_model is not None
        and not config.skip_production_reference
        and not config.production_model.is_file()
    ):
        raise FileNotFoundError(
            f"production model not found: {config.production_model}"
        )


def run_oof(config, inference_factory=VertebraInference):
    """Run each annotation through only its patient-group held-out fold."""
    validate_config(config)
    annotations = load_all_annotations(
        str(config.train_annotations),
        str(config.val_annotations),
    )
    folds = list(
        manual_group_kfold(
            annotations,
            config.n_folds,
            seed=config.seed,
            group_regex=config.group_regex,
        )
    )
    assignments = validate_fold_assignments(
        annotations,
        folds,
        config.group_regex,
    )
    image_paths = [
        resolve_image_path(annotation, config.project_root)
        for annotation in annotations
    ]

    cases = []
    for fold_number, (_, val_indices) in enumerate(folds, start=1):
        checkpoint = (
            config.run_dir
            / f"best_vertebra_model_fold{fold_number}.pth"
        )
        analyzer = inference_factory(
            str(checkpoint),
            device=config.device,
            ensemble_paths=[],
        )
        if analyzer.points_per_vertebra != 6:
            raise ValueError(
                f"fold {fold_number} is not a six-point checkpoint"
            )
        analyzer.tta = config.tta

        for annotation_index in val_indices:
            annotation = annotations[annotation_index]
            image_path = image_paths[annotation_index]
            case = {
                "annotation_index": annotation_index,
                "fold": assignments[annotation_index],
                "case_id": image_path.stem,
                "image_path": str(image_path),
                "spine_type": annotation.get(
                    "spine_type",
                    annotation.get("spineType", "L"),
                ),
                "ground_truth": {
                    "vertebrae": annotation.get("vertebrae", []),
                },
            }
            try:
                result = analyzer.predict(
                    str(image_path),
                    spine_type=case["spine_type"],
                    confidence_threshold=config.threshold,
                )
                prediction = {
                    "predicted_count": result.get(
                        "predicted_count",
                        len(result.get("vertebrae", [])),
                    ),
                    "count_confidence": result.get("count_confidence"),
                    "vertebrae": result.get("vertebrae", []),
                }
                case.update(
                    {
                        "status": "success",
                        "prediction": prediction,
                        "metrics": compare_case_landmarks(
                            annotation.get("vertebrae", []),
                            prediction["vertebrae"],
                        ),
                    }
                )
            except Exception as exc:
                case.update(
                    {
                        "status": "failed",
                        "error": repr(exc),
                    }
                )
            cases.append(case)

        del analyzer
        if str(config.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "n_annotations": len(annotations),
        "n_folds": len(folds),
        "seed": config.seed,
        "group_regex": config.group_regex,
        "assignments": assignments,
        "cases": cases,
    }


def run_production_reference(
    config,
    oof_result,
    inference_factory=VertebraInference,
):
    """Run the current production pipeline as a labelled non-OOF reference."""
    if config.skip_production_reference or config.production_model is None:
        return []

    analyzer = inference_factory(
        str(config.production_model),
        device=config.device,
    )
    analyzer.tta = config.tta
    cases = []
    for oof_case in oof_result["cases"]:
        case = {
            "annotation_index": oof_case["annotation_index"],
            "case_id": oof_case["case_id"],
            "image_path": oof_case["image_path"],
            "spine_type": oof_case["spine_type"],
            "ground_truth": oof_case["ground_truth"],
        }
        try:
            result = analyzer.predict(
                oof_case["image_path"],
                spine_type=oof_case["spine_type"],
                confidence_threshold=config.threshold,
            )
            prediction = {
                "predicted_count": result.get(
                    "predicted_count",
                    len(result.get("vertebrae", [])),
                ),
                "count_confidence": result.get("count_confidence"),
                "vertebrae": result.get("vertebrae", []),
            }
            case.update(
                {
                    "status": "success",
                    "prediction": prediction,
                    "metrics": compare_case_landmarks(
                        oof_case["ground_truth"]["vertebrae"],
                        prediction["vertebrae"],
                    ),
                }
            )
        except Exception as exc:
            case.update({"status": "failed", "error": repr(exc)})
        cases.append(case)

    del analyzer
    if str(config.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cases


def _atomic_write_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_write_text(path, text, encoding="utf-8"):
    _atomic_write_bytes(path, text.encode(encoding))


def _atomic_write_json(path, payload):
    _atomic_write_text(
        path,
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
    )


def _case_csv_row(case):
    metrics = case.get("metrics", {})
    groups = metrics.get("groups", {})
    all_points = groups.get("all", {})
    corners = groups.get("corners", {})
    middle = groups.get("middle", {})
    return {
        "annotation_index": case.get("annotation_index"),
        "fold": case.get("fold"),
        "case_id": case.get("case_id"),
        "image_path": case.get("image_path"),
        "status": case.get("status"),
        "error": case.get("error"),
        "n_gt_vertebrae": metrics.get("n_gt_vertebrae"),
        "n_pred_vertebrae": metrics.get("n_pred_vertebrae"),
        "count_exact": metrics.get("count_exact"),
        "absolute_count_error": metrics.get("absolute_count_error"),
        "n_matched_landmarks": all_points.get("n"),
        "mean_distance_px": all_points.get("mean_distance_px"),
        "median_distance_px": all_points.get("median_distance_px"),
        "p90_distance_px": all_points.get("p90_distance_px"),
        "p95_distance_px": all_points.get("p95_distance_px"),
        "max_distance_px": all_points.get("max_distance_px"),
        "corner_mean_distance_px": corners.get("mean_distance_px"),
        "middle_mean_distance_px": middle.get("mean_distance_px"),
        "missing_gt_landmarks": metrics.get("missing_gt_landmarks"),
        "missing_predicted_landmarks": metrics.get(
            "missing_predicted_landmarks"
        ),
    }


def _write_case_csv(path, cases):
    fieldnames = list(_case_csv_row({}).keys())
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(_case_csv_row(case) for case in cases)
    _atomic_write_bytes(path, stream.getvalue().encode("utf-8-sig"))


def _format_metric(value):
    return "n/a" if value is None else f"{value:.2f}"


def _comparison_markdown(oof_summary, production_summary):
    oof_corners = oof_summary["groups"]["corners"]
    production_corners = (
        production_summary["groups"]["corners"]
        if production_summary is not None
        else {}
    )
    lines = [
        "# Six-point OOF evaluation",
        "",
        "The production result below is a **non-OOF reference**. It may be "
        "**training-exposed** on part of this dataset and is not an unbiased "
        "head-to-head estimate.",
        "",
        "| Evaluation | Cases | Failed | Corner mean px | Corner median px | Corner P90 px | Count exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
        (
            f"| New six-point OOF | {oof_summary['n_cases']} | "
            f"{oof_summary['n_failed_predictions']} | "
            f"{_format_metric(oof_corners['mean_distance_px'])} | "
            f"{_format_metric(oof_corners['median_distance_px'])} | "
            f"{_format_metric(oof_corners['p90_distance_px'])} | "
            f"{_format_metric(oof_summary['count_exact_match_rate'])} |"
        ),
    ]
    if production_summary is not None:
        lines.append(
            f"| Production non-OOF reference | {production_summary['n_cases']} | "
            f"{production_summary['n_failed_predictions']} | "
            f"{_format_metric(production_corners['mean_distance_px'])} | "
            f"{_format_metric(production_corners['median_distance_px'])} | "
            f"{_format_metric(production_corners['p90_distance_px'])} | "
            f"{_format_metric(production_summary['count_exact_match_rate'])} |"
        )
    oof_all = oof_summary["groups"]["all"]
    oof_middle = oof_summary["groups"]["middle"]
    lines.extend(
        [
            "",
            "## New six-point OOF landmark detail",
            "",
            "| Landmark group | N | Mean px | Median px | P90 px |",
            "|---|---:|---:|---:|---:|",
            (
                f"| All six landmarks | {oof_all['n']} | "
                f"{_format_metric(oof_all['mean_distance_px'])} | "
                f"{_format_metric(oof_all['median_distance_px'])} | "
                f"{_format_metric(oof_all['p90_distance_px'])} |"
            ),
            (
                f"| Common four corners | {oof_corners['n']} | "
                f"{_format_metric(oof_corners['mean_distance_px'])} | "
                f"{_format_metric(oof_corners['median_distance_px'])} | "
                f"{_format_metric(oof_corners['p90_distance_px'])} |"
            ),
            (
                f"| Middle-only OOF | {oof_middle['n']} | "
                f"{_format_metric(oof_middle['mean_distance_px'])} | "
                f"{_format_metric(oof_middle['median_distance_px'])} | "
                f"{_format_metric(oof_middle['p90_distance_px'])} |"
            ),
            "",
            "Middle-point values are reported only when the model actually emits "
            "`middleSuperior` and `middleInferior`; missing points are not synthesized.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(config, oof_cases, production_cases=None):
    """Write atomic JSON, CSV, Markdown, and log artifacts."""
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_summary = summarize_cases(oof_cases)
    production_summary = (
        summarize_cases(production_cases)
        if production_cases is not None
        else None
    )
    paths = {
        "oof_predictions": output_dir / "oof_predictions.json",
        "oof_case_metrics": output_dir / "oof_case_metrics.csv",
        "oof_metrics": output_dir / "oof_metrics.json",
        "comparison": output_dir / "comparison.md",
        "eval_log": output_dir / "eval.log",
    }
    _atomic_write_json(
        paths["oof_predictions"],
        {
            "evaluation": "patient-grouped out-of-fold",
            "seed": config.seed,
            "group_regex": config.group_regex,
            "cases": oof_cases,
        },
    )
    _write_case_csv(paths["oof_case_metrics"], oof_cases)
    _atomic_write_json(paths["oof_metrics"], oof_summary)
    _atomic_write_text(
        paths["comparison"],
        _comparison_markdown(oof_summary, production_summary),
    )
    _atomic_write_text(
        paths["eval_log"],
        (
            f"run_dir={config.run_dir}\n"
            f"n_folds={config.n_folds}\n"
            f"seed={config.seed}\n"
            f"tta={config.tta}\n"
            f"n_cases={oof_summary['n_cases']}\n"
            f"n_failed={oof_summary['n_failed_predictions']}\n"
        ),
    )
    if production_summary is not None:
        path = output_dir / "production_reference_metrics.json"
        _atomic_write_json(path, production_summary)
        paths["production_reference_metrics"] = path
    return paths


def render_worst_overlays(config, cases):
    """Render auditable overlays for the worst ranked OOF cases."""
    output_dir = config.output_dir / "worst_cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = rank_worst_cases(cases, config.worst_count)
    manifest_cases = []

    for rank, case in enumerate(ranked, start=1):
        entry = {
            "rank": rank,
            "annotation_index": case.get("annotation_index"),
            "fold": case.get("fold"),
            "case_id": case.get("case_id"),
            "status": case.get("status"),
            "error": case.get("error"),
        }
        try:
            image_rgb = VertebraInference.load_image(
                None,
                case["image_path"],
            )
            gt_vertebrae = case.get("ground_truth", {}).get(
                "vertebrae",
                [],
            )
            pred_vertebrae = case.get("prediction", {}).get(
                "vertebrae",
                [],
            )
            per_vertebra, _ = compare_vertebrae(
                gt_vertebrae,
                pred_vertebrae,
            )
            overlay = draw_overlay(
                image_rgb,
                gt_vertebrae,
                pred_vertebrae,
                per_vertebra,
            )
            filename = (
                f"{rank:02d}_fold{case.get('fold', 0)}_"
                f"idx{case.get('annotation_index', 0)}_"
                f"{case.get('case_id', 'case')}.png"
            )
            ok, encoded = cv2.imencode(".png", overlay)
            if not ok:
                raise RuntimeError("OpenCV could not encode overlay")
            _atomic_write_bytes(output_dir / filename, encoded.tobytes())
            entry["overlay"] = filename
        except Exception as exc:
            entry["overlay_error"] = repr(exc)
        manifest_cases.append(entry)

    manifest_path = output_dir / "manifest.json"
    _atomic_write_json(manifest_path, {"cases": manifest_cases})
    return manifest_path


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Patient-grouped OOF evaluation for six-point vertebra checkpoints"
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument(
        "--train-annotations",
        default="endplate_training_data/annotations/train_annotations.json",
    )
    parser.add_argument(
        "--val-annotations",
        default="endplate_training_data/annotations/val_annotations.json",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-regex", default=DEFAULT_GROUP_REGEX)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
    )
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--worst-count", type=int, default=10)
    parser.add_argument("--output-dir", default="oof_eval")
    parser.add_argument(
        "--production-model",
        default="best_vertebra_model.pth",
    )
    parser.add_argument(
        "--skip-production-reference",
        action="store_true",
    )
    return parser


def _resolve_cli_path(value, base):
    path = Path(value)
    return (path if path.is_absolute() else Path(base) / path).resolve()


def config_from_args(args):
    project_root = Path(args.project_root).resolve()
    run_dir = _resolve_cli_path(args.run_dir, project_root)
    output_dir = _resolve_cli_path(args.output_dir, run_dir)
    production_model = (
        _resolve_cli_path(args.production_model, project_root)
        if args.production_model
        else None
    )
    return EvalConfig(
        project_root=project_root,
        run_dir=run_dir,
        train_annotations=_resolve_cli_path(
            args.train_annotations,
            project_root,
        ),
        val_annotations=_resolve_cli_path(
            args.val_annotations,
            project_root,
        ),
        output_dir=output_dir,
        production_model=production_model,
        n_folds=args.n_folds,
        seed=args.seed,
        group_regex=args.group_regex,
        device=args.device,
        threshold=args.threshold,
        tta=not args.no_tta,
        worst_count=args.worst_count,
        skip_production_reference=args.skip_production_reference,
    )


def main(argv=None):
    config = config_from_args(build_parser().parse_args(argv))
    validate_config(config)
    print(
        f"OOF evaluation: {config.n_folds} folds, seed={config.seed}, "
        f"device={config.device}, TTA={config.tta}"
    )
    print(f"Run directory: {config.run_dir}")
    print(f"Output directory: {config.output_dir}")

    oof_result = run_oof(config)
    production_cases = None
    if not config.skip_production_reference and config.production_model is not None:
        production_cases = run_production_reference(config, oof_result)
    paths = write_reports(config, oof_result["cases"], production_cases)
    manifest = render_worst_overlays(config, oof_result["cases"])
    summary = json.loads(paths["oof_metrics"].read_text(encoding="utf-8"))

    print(
        f"OOF complete: {summary['n_successful_predictions']}/"
        f"{summary['n_cases']} successful, "
        f"{summary['n_failed_predictions']} failed"
    )
    print(
        "Mean/median/P90 px: "
        f"{_format_metric(summary['groups']['all']['mean_distance_px'])} / "
        f"{_format_metric(summary['groups']['all']['median_distance_px'])} / "
        f"{_format_metric(summary['groups']['all']['p90_distance_px'])}"
    )
    print(f"Comparison: {paths['comparison']}")
    print(f"Worst cases: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
