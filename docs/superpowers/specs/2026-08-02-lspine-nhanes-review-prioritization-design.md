# L-spine-first and NHANES-deferred physician review design

Date: 2026-08-02

## Goal

Reorder the completed six-point OOF physician review so the physician can review clinically higher-yield L-spine cases first, while deferring low-resolution NHANES images. Preserve the complete 61-case OOF result and every physician-authored comment.

## Approved policy

1. The formal OOF result remains the full 61-case, 54-patient-group evaluation.
2. The primary physician queue contains L-spine, non-NHANES cases only.
3. NHANES cases remain in the evaluation and training provenance but move to a secondary stress-test section with no mandatory case-by-case review.
4. C-spine remains reported but is lower review priority because physician interpretation is slower.
5. This change does not alter the training allowlist, annotations, checkpoints, OOF predictions, or production model.

## Source of truth and preservation rules

- Input report: `endplate_training_data/runs/sixpoint_reviewed_cv5_20260802_040614/oof_eval/physician_review_summary.md`.
- Prediction inventory: the adjacent `oof_predictions.json`.
- Physician comments already appended to the ten worst-case table are source-of-truth clinical review notes. Preserve their wording and case association.
- Preserve the original all-case comparison table so the formal OOF result stays auditable.
- Identify NHANES by the dataset directory in `image_path`, case-insensitively, rather than by a filename prefix alone.

## Report structure

The revised report will contain, in this order:

1. Formal 61-case evaluation scope and full-cohort comparison.
2. The approved review policy.
3. Stratified OOF metrics for:
   - all 61 cases;
   - 48 non-NHANES cases;
   - 13 NHANES stress-test cases;
   - 40 L-spine non-NHANES cases.
4. A primary L-spine, non-NHANES review queue.
5. The original ten worst-case table with physician comments preserved.
6. Deferred sections for C-spine and NHANES cases.
7. The unchanged deployment decision that production must not yet be replaced.

## Primary review queue

Build the primary queue as the union of:

- L-spine, non-NHANES cases among the ten worst OOF overlays; and
- L-spine, non-NHANES count mismatches.

Remove duplicates while retaining these ten cases:

1. `81312013`
2. `80339761`
3. `21383869`
4. `18971571-2`
5. `81314815`
6. `80552846`
7. `18971571-1`
8. `81161252`
9. `80813115-1`
10. `21584353`

Cases with an existing overlay link retain it. Count-mismatch-only cases may link to their row in the report but do not require new inference or new overlays for this report-only change.

## Metrics and interpretation

The subgroup values already calculated from the completed OOF predictions are:

| Subgroup | Cases | Count exact | Corner mean | Middle mean |
|---|---:|---:|---:|---:|
| Full cohort | 61 | 75.41% | 126.38 px | 118.89 px |
| Non-NHANES | 48 | 81.25% | 128.71 px | 121.68 px |
| NHANES stress test | 13 | 53.85% | 115.15 px | 106.01 px |
| L-spine non-NHANES | 40 | 87.50% | 127.30 px | 121.39 px |

NHANES has lower count accuracy, but its landmark-distance mean is not uniformly worse. Therefore the report must attribute deferral to image quality and physician review efficiency, not claim that every NHANES landmark metric is inferior.

## Deferred review

- NHANES: retain all 13 cases and metrics as a secondary stress test; no mandatory individual review in this round.
- C-spine: retain the known weakness and mismatch list, but place it after the primary L-spine queue.
- Do not delete or relabel any annotation based solely on subgroup membership.

## Validation

After editing the report:

1. Confirm all existing physician comments remain present and attached to the same case.
2. Confirm the formal full-cohort table is unchanged.
3. Confirm all ten primary queue case IDs appear exactly once.
4. Confirm every linked overlay exists and is Unicode-safe readable.
5. Confirm `oof_predictions.json` still contains 61 cases across folds 1 through 5.
6. Confirm no model, sidecar, annotation, or checkpoint file changed.

## Non-goals

- No retraining or new fold generation.
- No removal of NHANES from the training allowlist.
- No modification of OOF metrics JSON or production-reference metrics.
- No production deployment decision change.
