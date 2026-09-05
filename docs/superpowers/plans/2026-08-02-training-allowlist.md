# Reviewed-only Training Allowlist Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a physician-reviewed, hash-locked training allowlist and verify the resulting patient-grouped split in dry-run mode.

**Architecture:** A focused allowlist builder converts the viewer manifest plus explicit physician overrides into an auditable JSON artifact. The existing split generator consumes and validates that artifact before building annotations, so the CV pipeline continues to use its established inputs.

**Tech Stack:** Python 3.14 standard library, pytest, existing JSON annotation pipeline.

## Global Constraints

- Do not start GPU training.
- Do not write or back up train/validation annotations during dry-run.
- Keep absent middle points absent.
- Require exact paths and current SHA-256 hashes before an allowlisted case is used.

---

### Task 1: Allowlist builder

**Files:**
- Create: `build_training_allowlist.py`
- Create: `tests/test_training_allowlist.py`

**Interfaces:**
- Consumes: viewer `pair_review.json`, scanned `(image_rel_path, json_path)` pairs, repeated approved stems.
- Produces: `build_allowlist(review_manifest, pairs, approved_stems, root) -> dict` and a versioned JSON artifact.

- [x] Write failing tests for reviewed matches, mtime-only drift, size changes, overrides, malformed input, and unknown overrides.
- [x] Run the focused tests and confirm failures are caused by the missing feature.
- [x] Implement minimal selection, audit evidence, SHA-256 locking, and trainable-annotation validation.
- [x] Run focused tests green and refactor without changing behavior.

### Task 2: Split-generator enforcement

**Files:**
- Modify: `regenerate_splits.py`
- Modify: `tests/test_regenerate_splits.py`

**Interfaces:**
- Consumes: `--allowlist PATH` containing exact image/JSON paths and SHA-256 hashes.
- Produces: a filtered pair list or a hard validation error before split generation.

- [x] Write failing tests for review-manifest exclusion, allowlist filtering, stale hashes, missing pairs, and dry-run non-mutation.
- [x] Run the focused tests and confirm the expected failures.
- [x] Implement `--allowlist`, validation, and filtering.
- [x] Run focused tests green.

### Task 3: Real cohort artifact and dry-run

**Files:**
- Create locally under ignored data: `endplate_training_data/manifests/training_allowlist_20260802.json`

**Interfaces:**
- Consumes: the 2026-08-02 review manifest and four physician-approved override stems.
- Produces: a selected cohort summary and patient-grouped dry-run split counts.

- [x] Build the allowlist and inspect summary/evidence counts.
- [x] Hash the current train/validation files.
- [x] Run `regenerate_splits.py --allowlist ... --dry-run` with the production patient grouping regex.
- [x] Re-hash train/validation files and confirm no mutation.
- [x] Run the full Python and Node regression suites and inspect `git diff`.

## Dry-run outcome

- Review manifest: 74 exact image/JSON pairs.
- Selected for training: 61 samples across 54 patient groups.
- Evidence: 57 physician review matches plus 4 explicit physician overrides.
- Excluded: 6 unresolved review mismatches and 7 measurement-only JSON files without `vertebrae`.
- Patient-grouped preview: 49 train / 12 validation at seed 42 and validation ratio 0.2.
