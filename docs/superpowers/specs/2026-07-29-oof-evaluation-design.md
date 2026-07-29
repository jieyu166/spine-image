# Six-point 5-fold OOF evaluation design

Date: 2026-07-29

## Goal

Evaluate the new six-point lumbar-spine detector without training-data leakage,
compare it with the current production pipeline as a clearly labelled reference,
and export the worst cases for physician review. The evaluation must not replace
or modify any production checkpoint or ensemble sidecar.

## Approaches considered

### A. Patient-grouped out-of-fold evaluation (selected)

Reconstruct the exact five patient-grouped folds used for training. Each sample
is inferred only by the fold model for which that sample was held out.

Advantages:

- Every labelled sample receives a prediction from a model that did not train on it.
- Uses all available labelled cases while preserving patient-level isolation.
- Produces an honest estimate of the new pipeline's generalization.

Limitations:

- It evaluates five related checkpoints rather than the later all-data production model.
- The current production model may have trained on some of these cases, so its score on
  this dataset is a reference, not an unbiased head-to-head result.

### B. Evaluate the five-fold ensemble on all labelled cases

Rejected for model selection because four of five members trained on each evaluated
case. This creates leakage and makes the score optimistic. It may be evaluated only
on a future external holdout set.

### C. Immediately train on all data and visually inspect examples

Rejected as the first step because it removes the only available held-out evidence
before deciding whether the six-point change improved the system.

## Inputs

- Run directory:
  `endplate_training_data/runs/sixpoint_cv5_20260729_132517`
- Fold checkpoints:
  `best_vertebra_model_fold1.pth` through
  `best_vertebra_model_fold5.pth`
- Annotation inputs: the same train and validation JSON files used by the CV run,
  concatenated in the same order as `train_vertebra_model_cv.load_all_annotations`.
- Fold reconstruction: `manual_group_kfold`, five folds, seed 42, and the recorded
  group regex from the CV run.
- Production reference: project-root `best_vertebra_model.pth`, including its
  existing sidecar behavior when present.

The evaluator must accept command-line overrides for all paths, fold count, seed,
group regex, device, TTA state, and worst-case count. It must not depend on the
malformed or platform-sensitive `config` section of `cv_results.json`.

## Leakage safeguards

Before inference, the evaluator must verify:

1. Every annotation index appears in exactly one validation fold.
2. No patient/subject group occurs in both the training and validation side of a fold.
3. Every expected fold checkpoint exists.
4. Each loaded fold checkpoint uses the six-point layout.
5. Output paths are under a dedicated evaluation directory and do not target the
   project-root production checkpoint or sidecar.

Any failed safeguard stops the evaluation with a non-zero exit code.

## Evaluation flow

1. Load and concatenate annotations.
2. Reconstruct the exact grouped folds.
3. Load one fold model at a time to limit GPU memory use.
4. Infer only that fold's held-out samples using the current inference/post-processing
   chain. TTA is enabled by default because that is the operational pipeline.
5. Save a normalized prediction record for every held-out sample.
6. Compute per-case and aggregate metrics.
7. Optionally run the current production pipeline on the same cases and report it as
   a non-OOF reference.
8. Rank the new model's failures and render the worst ten overlays.

Inference errors are recorded per case with an error message and counted as failures;
they are not silently omitted. A completely missing or unreadable input aborts before
GPU inference.

## Metrics

Aggregate and per-case output will include:

- Number of samples, successful predictions, and failed predictions.
- Vertebral count exact-match rate and absolute count error.
- Landmark distance in original-image pixels: mean, median, P90, P95, and maximum.
- Common four-corner metrics:
  `anteriorSuperior`, `posteriorSuperior`, `posteriorInferior`,
  `anteriorInferior`.
- Middle-point metrics:
  `middleSuperior`, `middleInferior`.
- Per-landmark and per-vertebral-level summaries.
- Threshold counts below 50 px and 100 px.
- Missing GT/predicted landmark counts.

The production reference is compared on the common landmarks it actually outputs.
If it lacks middle points, the report must say so rather than synthesizing them.
No millimetre metric is claimed unless trustworthy pixel spacing is present.

## Outputs

All artifacts go under:

`endplate_training_data/runs/sixpoint_cv5_20260729_132517/oof_eval`

- `oof_predictions.json`: fold assignment, prediction, and error per sample.
- `oof_case_metrics.csv`: sortable per-case metrics.
- `oof_metrics.json`: aggregate six-point and four-corner metrics.
- `production_reference_metrics.json`: optional reference results and leakage caveat.
- `comparison.md`: concise new-OOF versus production-reference table.
- `worst_cases/`: overlays and a manifest for the ten worst OOF cases.
- `eval.log`: command, configuration, progress, and failures.

Writing is atomic where practical so an interrupted run does not look complete.

## Testing strategy

Unit tests are written before implementation and must demonstrate:

- Exact fold reconstruction assigns every sample once.
- Multiple views from the same patient never cross folds.
- Duplicate or missing validation assignments fail validation.
- Six-point metrics separate middle points from the four corners.
- Missing landmarks are counted and never converted into zero error.
- Percentiles and threshold counts are correct on a deterministic fixture.
- Production reference with four-point output is reported without fabricated middle
  landmarks.
- Worst-case ranking places inference failures first, then largest errors.
- Output validation refuses paths that could overwrite production weights.

After unit tests pass, an integration check runs against a tiny local fixture, followed
by the full GPU OOF evaluation. Completion requires fresh test output, 67 expected OOF
assignments (or the currently discovered exact total), zero fold leakage, and readable
JSON/CSV/overlay artifacts.

## Deployment decision after evaluation

The production model remains unchanged during this task. If OOF results and physician
review are acceptable, the next separate task is to train a final checkpoint on all
labelled data. The CV checkpoints indicate best epochs around 2--5 (median 3), so the
full-data schedule should start near three frozen-backbone epochs and be verified on a
new external holdout before deployment.
