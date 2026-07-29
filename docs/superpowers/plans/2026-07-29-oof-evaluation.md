# Six-point OOF Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a patient-grouped, leakage-safe OOF evaluator for the five six-point checkpoints, with a clearly caveated production reference and worst-case overlays.

**Architecture:** Put deterministic fold validation and landmark aggregation in a small pure-Python module, then keep checkpoint loading, GPU inference, filesystem output, and CLI orchestration in `eval_oof.py`. Reuse the training script's exact grouped-fold implementation and the existing inference/overlay functions so evaluation matches the operational pipeline without duplicating model logic.

**Tech Stack:** Python 3.14, pytest, NumPy, PyTorch/CUDA, OpenCV, existing `VertebraInference`, existing `manual_group_kfold`.

## Global Constraints

- Never modify or replace project-root `best_vertebra_model.pth` or `best_vertebra_model.pth.ensemble`.
- Reconstruct five folds with seed `42` and the recorded patient-group regex unless overridden by CLI.
- A sample must be inferred only by the fold model that held it out.
- Load fold models with `ensemble_paths=[]` so no sidecar can silently join OOF inference.
- Report production output as a potentially training-exposed reference, not unbiased OOF evidence.
- Do not fabricate `middleSuperior` or `middleInferior` for a four-point reference model.
- Record inference failures; never silently drop them from denominators.
- Write all results below the selected run directory's dedicated `oof_eval` directory.

---

### Task 1: Pure fold and landmark metric core

**Files:**
- Create: `oof_metrics.py`
- Create: `tests/test_oof_metrics.py`

**Interfaces:**
- Consumes: annotations compatible with `train_vertebra_model_cv.extract_group_id`; fold tuples of `(train_indices, val_indices)`.
- Produces:
  - `validate_fold_assignments(annotations, folds, group_regex) -> dict[int, int]`
  - `compare_case_landmarks(gt_vertebrae, pred_vertebrae) -> dict`
  - `summarize_cases(cases) -> dict`
  - `rank_worst_cases(cases, limit) -> list[dict]`
  - constants `POINT_NAMES_6`, `CORNER_NAMES_4`, `MIDDLE_NAMES_2`

- [ ] **Step 1: Write failing fold-safety tests**

Add fixtures containing two views for one eight-digit patient and several independent
patients. Assert the returned mapping assigns each annotation exactly once, and assert
that duplicate validation assignment, missing assignment, and a patient split across
train/validation each raise `ValueError`.

```python
def test_validate_fold_assignments_maps_every_sample_once():
    annotations = [
        {"image_path": r"Images\202607\80145593.png"},
        {"image_path": r"Images\202607\80145593-2.png"},
        {"image_path": r"Images\202607\80294005.png"},
        {"image_path": r"Images\202607\80655287.png"},
    ]
    folds = [
        ([2, 3], [0, 1]),
        ([0, 1], [2]),
        ([0, 1, 2], [3]),
    ]
    assert validate_fold_assignments(
        annotations, folds, DEFAULT_GROUP_REGEX
    ) == {0: 1, 1: 1, 2: 2, 3: 3}
```

- [ ] **Step 2: Run fold-safety tests and verify RED**

Run:

```powershell
python -m pytest tests/test_oof_metrics.py -v
```

Expected: collection fails with `ModuleNotFoundError: No module named 'oof_metrics'`.

- [ ] **Step 3: Implement minimal fold validation**

In `oof_metrics.py`, import `extract_group_id` and implement exact set-based checks:

```python
def validate_fold_assignments(annotations, folds, group_regex):
    assignments = {}
    expected = set(range(len(annotations)))
    for fold_number, (train_indices, val_indices) in enumerate(folds, start=1):
        train_set, val_set = set(train_indices), set(val_indices)
        if train_set & val_set:
            raise ValueError(f"fold {fold_number} has train/validation index overlap")
        train_groups = {
            extract_group_id(annotations[i], group_regex) for i in train_set
        }
        val_groups = {
            extract_group_id(annotations[i], group_regex) for i in val_set
        }
        if train_groups & val_groups:
            raise ValueError(f"fold {fold_number} has patient-group leakage")
        for index in val_set:
            if index in assignments:
                raise ValueError(f"annotation {index} appears in multiple validation folds")
            assignments[index] = fold_number
    missing = expected - set(assignments)
    if missing:
        raise ValueError(f"annotations missing validation assignment: {sorted(missing)}")
    return assignments
```

- [ ] **Step 4: Run fold-safety tests and verify GREEN**

Run `python -m pytest tests/test_oof_metrics.py -v`.
Expected: fold-safety tests pass.

- [ ] **Step 5: Write failing six-point metric tests**

Use literal coordinates with hand-calculated distances. Cover a four-corner perfect
match plus `middleSuperior=3 px` and `middleInferior=4 px`; assert the all-point mean is
`7 / 6`, corner mean is `0`, and middle mean is `3.5`. Add a missing-prediction fixture
and assert it increments `missing_predicted_landmarks` without appending a zero distance.

```python
assert metrics["groups"]["all"]["mean_distance_px"] == pytest.approx(7 / 6)
assert metrics["groups"]["corners"]["mean_distance_px"] == 0.0
assert metrics["groups"]["middle"]["mean_distance_px"] == 3.5
assert missing["missing_predicted_landmarks"] == 1
```

- [ ] **Step 6: Run the new metric tests and verify RED**

Run `python -m pytest tests/test_oof_metrics.py -v`.
Expected: import or attribute failure for `compare_case_landmarks`.

- [ ] **Step 7: Implement metric extraction and aggregation**

Match vertebrae by `name`, iterate the six fixed point names, calculate Euclidean
distance only when both GT and prediction exist, and preserve landmark name plus
vertebral level on each distance record. Implement a reusable literal percentile
summary with NumPy:

```python
def summarize_distances(values):
    if not values:
        return {
            "n": 0, "mean_distance_px": None, "median_distance_px": None,
            "p90_distance_px": None, "p95_distance_px": None,
            "max_distance_px": None, "n_below_50_px": 0, "n_below_100_px": 0,
        }
    data = np.asarray(values, dtype=float)
    return {
        "n": int(data.size),
        "mean_distance_px": float(np.mean(data)),
        "median_distance_px": float(np.median(data)),
        "p90_distance_px": float(np.percentile(data, 90)),
        "p95_distance_px": float(np.percentile(data, 95)),
        "max_distance_px": float(np.max(data)),
        "n_below_50_px": int(np.sum(data < 50)),
        "n_below_100_px": int(np.sum(data < 100)),
    }
```

`compare_case_landmarks` must return count exactness, absolute count error, six-point,
corner, middle, per-landmark, and per-level summaries plus raw distance records.
`summarize_cases` aggregates successful cases while retaining failed-case denominators.

- [ ] **Step 8: Write and pass percentile, production-layout, and ranking tests**

Add deterministic assertions for P90/P95, a four-point prediction whose middle group
has `n == 0`, and ranking where inference failures precede successful cases ordered by
descending maximum then mean error. Run:

```powershell
python -m pytest tests/test_oof_metrics.py -v
```

Expected: all Task 1 tests pass.

- [ ] **Step 9: Commit Task 1**

```powershell
git add oof_metrics.py tests/test_oof_metrics.py
git commit -m "feat: add leakage-safe OOF metric core"
```

---

### Task 2: OOF CLI, inference orchestration, and artifact writing

**Files:**
- Create: `eval_oof.py`
- Create: `tests/test_eval_oof.py`

**Interfaces:**
- Consumes Task 1 functions and existing:
  - `load_all_annotations`
  - `manual_group_kfold`
  - `VertebraInference`
  - `draw_overlay`
- Produces:
  - `EvalConfig` dataclass
  - `resolve_image_path(annotation, project_root) -> Path`
  - `validate_config(config) -> None`
  - `run_oof(config, inference_factory=VertebraInference) -> dict`
  - `write_reports(config, oof_cases, production_cases) -> dict[str, Path]`
  - CLI `main(argv=None) -> int`

- [ ] **Step 1: Write failing configuration and resolution tests**

Use `tmp_path` to create nested images and five small stub checkpoint files. Assert:

- nested `image_path` resolves before basename fallback;
- a missing image raises `FileNotFoundError`;
- a missing fold checkpoint raises `FileNotFoundError`;
- an output directory equal to a checkpoint path or production sidecar raises
  `ValueError`;
- a normal `<run_dir>/oof_eval` path passes.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest tests/test_eval_oof.py -v
```

Expected: collection fails because `eval_oof` does not exist.

- [ ] **Step 3: Implement configuration, path resolution, and atomic writers**

Define `EvalConfig` with defaults matching the approved design. Resolve direct paths
relative to project root, then `Images/<basename>` with `.dcm`, `.png`, `.jpg`,
`.jpeg`. Implement JSON and text writes via a sibling `.tmp` followed by
`Path.replace`; write CSV with `csv.DictWriter` and the same atomic pattern.

- [ ] **Step 4: Run Task 2 configuration tests and verify GREEN**

Run `python -m pytest tests/test_eval_oof.py -v`.
Expected: configuration and resolution tests pass.

- [ ] **Step 5: Write failing orchestration test with an injected lightweight factory**

The fake factory is permitted only at the GPU boundary. It must return the complete
real prediction shape: `vertebrae`, `predicted_count`, `count_confidence`, and
`original_image`. Assert:

- each annotation is predicted exactly once by its held-out fold;
- fold construction receives the configured seed and regex;
- fold inference is created with `ensemble_paths=[]`;
- a raised prediction error becomes a failed case instead of disappearing;
- every resulting case includes `annotation_index`, `fold`, `image_path`, and status.

- [ ] **Step 6: Run orchestration test and verify RED**

Run `python -m pytest tests/test_eval_oof.py -v`.
Expected: failure because `run_oof` is absent.

- [ ] **Step 7: Implement OOF orchestration**

Load annotations, materialize the fold iterator, call `validate_fold_assignments`,
then process folds sequentially. The model call must be:

```python
analyzer = inference_factory(
    str(config.run_dir / f"best_vertebra_model_fold{fold_number}.pth"),
    device=config.device,
    ensemble_paths=[],
)
if analyzer.points_per_vertebra != 6:
    raise ValueError(f"fold {fold_number} is not a six-point checkpoint")
analyzer.tta = config.tta
```

Call `predict` with per-annotation spine type and configured confidence threshold.
On exception, append a case with `status="failed"` and `error=repr(exc)`. Release each
fold analyzer and call `torch.cuda.empty_cache()` only when CUDA is active.

- [ ] **Step 8: Write failing report and overlay tests**

With real temporary JSON/CSV files and a small RGB image, assert:

- reports parse successfully after writing;
- CSV contains failed and successful cases;
- `comparison.md` labels the production result as non-OOF reference;
- the production factory is called without explicit `ensemble_paths`, preserving its
  sidecar behavior;
- worst-case manifest and exactly the requested number of overlay PNGs are written;
- a four-point production case produces no synthetic middle-point count.

- [ ] **Step 9: Implement production reference, reports, and overlays**

Run the production reference only when `--production-model` exists and the user has
not supplied `--skip-production-reference`. Use stored normalized OOF predictions to
rank cases, reload the original image with `VertebraInference.load_image`, and call
`draw_overlay` for the selected worst cases. Encode PNG with `cv2.imencode` to support
the Chinese Windows path.

Write:

- `oof_predictions.json`
- `oof_case_metrics.csv`
- `oof_metrics.json`
- `production_reference_metrics.json`
- `comparison.md`
- `worst_cases/manifest.json`
- `eval.log`

- [ ] **Step 10: Add CLI and run all Task 2 tests**

Support:

```text
--run-dir --project-root --train-annotations --val-annotations
--n-folds --seed --group-regex --device --threshold
--no-tta --worst-count --output-dir
--production-model --skip-production-reference
```

Run:

```powershell
python -m pytest tests/test_eval_oof.py tests/test_oof_metrics.py -v
```

Expected: all OOF tests pass.

- [ ] **Step 11: Commit Task 2**

```powershell
git add eval_oof.py tests/test_eval_oof.py
git commit -m "feat: add six-point OOF evaluation runner"
```

---

### Task 3: Integration verification and full GPU evaluation

**Files:**
- Modify only if a verified integration defect requires it:
  `eval_oof.py`, `oof_metrics.py`, or their corresponding tests
- Generate, do not commit:
  `endplate_training_data/runs/sixpoint_cv5_20260729_132517/oof_eval/**`

**Interfaces:**
- Consumes: Task 2 CLI and the five completed checkpoints.
- Produces: complete OOF artifacts and an evidence-backed deployment recommendation.

- [ ] **Step 1: Run the complete existing and new test suite**

Run:

```powershell
python -m pytest tests -v
```

Expected: zero failed tests.

- [ ] **Step 2: Verify input inventory and production hashes**

Record sample count, unique patient-group count, fold checkpoint existence, and
SHA-256 for project-root production checkpoint and sidecar before evaluation.

- [ ] **Step 3: Run full OOF plus production-reference GPU evaluation**

Run from the project root:

```powershell
python eval_oof.py `
  --run-dir "endplate_training_data/runs/sixpoint_cv5_20260729_132517" `
  --device cuda `
  --worst-count 10
```

Expected: exit code `0`, five fold checkpoints loaded sequentially, every annotation
assigned exactly once, and no leakage safeguard failure.

- [ ] **Step 4: Validate generated artifacts**

Parse every JSON, load every PNG, and verify:

- OOF case count equals the discovered annotation total.
- `n_failed_predictions == 0`, or every failure is explicitly listed.
- Five distinct fold numbers occur.
- Middle-point metrics have observations when GT includes middle points.
- Exactly ten worst-case overlays exist unless fewer than ten cases were successful.
- `comparison.md` contains the production-exposure caveat.

- [ ] **Step 5: Re-check production hashes**

Recompute SHA-256 for the project-root checkpoint and sidecar and assert equality with
Step 2. If either differs, stop and report the integrity failure.

- [ ] **Step 6: Run fresh completion verification**

Run:

```powershell
python -m pytest tests -q
git diff --check
git status --short
```

Expected: zero test failures, no whitespace errors, and only intentional source changes
plus ignored evaluation artifacts.

- [ ] **Step 7: Commit any integration-only fixes**

If Step 3 exposed a real defect, first add a failing regression test, then fix it and
commit only the test and relevant source:

```powershell
git add eval_oof.py oof_metrics.py tests/test_eval_oof.py tests/test_oof_metrics.py
git commit -m "fix: harden OOF evaluation integration"
```

If no integration fix was necessary, do not create an empty commit.
