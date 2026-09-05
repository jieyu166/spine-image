# Reviewed-only training allowlist design

Date: 2026-08-02

## Goal

Create a reproducible training cohort from the physician pair-review manifest without modifying the existing train/validation split during the first run.

## Selection policy

- Start from exact image/JSON pairs found by the production scanner.
- Include a `match` review when the reviewed image and JSON paths and byte sizes still match.
- Treat an mtime-only difference as OneDrive metadata drift, record it in the audit evidence, and lock the current content with SHA-256.
- Include a previously mismatched case only through an explicit physician-approved stem override.
- Require the annotation JSON to be an object with a `vertebrae` list; measurement-only JSON is not trainable by this model.
- Reject unknown, ambiguous, missing, size-changed, or malformed pairs.
- Never infer or fabricate missing middle points.

For the 2026-08-02 cohort, the approved overrides are `20285873L1`, `81340744`, `52780208`, and `80324056`.

## Artifacts and data flow

`build_training_allowlist.py` reads the exported review manifest and current `Images` tree, applies the policy, computes SHA-256 for both files, and writes a versioned JSON allowlist. Each entry records paths, current size/mtime/hash, review evidence, and the original review status.

`regenerate_splits.py --allowlist <path> --dry-run` validates every allowlisted path and current hash, filters the production scan to exactly those entries, builds annotations, and reports the patient-grouped train/validation counts without writing or backing up split files.

## Safety and errors

- A stale allowlist hash is a hard error.
- An allowlist path absent from the current scan is a hard error.
- Duplicate allowlist paths or duplicate override stems are rejected.
- The exported `pair_review.json` itself is ignored by the image/annotation scanner.
- The initial command is dry-run only; GPU training is not started.

## Verification

- Unit tests cover matching reviews, mtime-only drift, size changes, explicit overrides, malformed/unknown overrides, hash validation, filtering, and dry-run non-mutation.
- The full Python and browser/Node regression suites run after implementation.
- The real dry-run must report the selected cohort and leave existing train/validation JSON files byte-identical.
