# L-spine-first and NHANES-deferred Physician Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Revise the completed OOF physician report so L-spine non-NHANES cases form the primary review queue, NHANES is retained as a secondary stress-test subgroup, and all physician-authored comments remain unchanged.

**Architecture:** Treat the existing 61-case OOF artifacts as immutable evidence and modify only the physician-facing Markdown report. Add a stratified interpretation layer and a deterministic ten-case primary queue derived from existing predictions; verify preservation with a Unicode-safe, read-only Python audit.

**Tech Stack:** Markdown, Python 3.14, JSON, NumPy/OpenCV for validation, existing `oof_metrics.summarize_cases`.

## Global Constraints

- Keep the formal 61-case, 54-patient-group OOF result unchanged.
- Preserve every physician comment already attached to the ten worst-case rows.
- Identify NHANES from case-insensitive `image_path` dataset-directory membership, not filename alone.
- Keep NHANES in evaluation and training provenance; defer only mandatory physician review.
- Do not modify annotations, allowlists, checkpoints, OOF JSON/CSV, production model, or production sidecar.
- Keep C-spine results visible but lower priority than L-spine for this review round.
- Do not change the decision that production must not yet be replaced.

---

### Task 1: Add the approved review-priority layer

**Files:**
- Modify: `endplate_training_data/runs/sixpoint_reviewed_cv5_20260802_040614/oof_eval/physician_review_summary.md`
- Reference: `endplate_training_data/runs/sixpoint_reviewed_cv5_20260802_040614/oof_eval/oof_predictions.json`
- Reference: `docs/superpowers/specs/2026-08-02-lspine-nhanes-review-prioritization-design.md`

**Interfaces:**
- Consumes: the immutable OOF case list and existing physician comments.
- Produces: one physician-facing report with full-cohort metrics, subgroup metrics, a ten-case primary queue, and deferred C-spine/NHANES sections.

- [ ] **Step 1: Run a preservation-and-feature check to establish RED**

Run from the project root:

```powershell
@'
from pathlib import Path

report = Path(
    "endplate_training_data/runs/sixpoint_reviewed_cv5_20260802_040614/"
    "oof_eval/physician_review_summary.md"
)
text = report.read_text(encoding="utf-8")

comments = [
    "T12有2點不知為何偏移很多，GT正確，模型錯誤",
    "L5/S1很好，其他的有很大問題，可能有implant要另外訓練",
    "影像品質差幾乎不可判讀",
    "早期標註為S1上下緣(新版為S1上緣)，不含T12(新版有)，可能須把舊資料升級成V2.3",
]
assert all(comment in text for comment in comments)
assert "## Approved review policy" in text
assert "## Primary review queue: L-spine non-NHANES" in text
assert "## Deferred NHANES stress test" in text
'@ | python -
```

Expected: FAIL at `Approved review policy` because the physician comments exist but the approved prioritization sections do not.

- [ ] **Step 2: Preserve the current physician comments verbatim**

Before editing, copy the exact ten worst-case table from `## Ten worst OOF overlays` through the line before `## All count mismatches`. During the report edit, leave that block byte-for-byte unchanged.

- [ ] **Step 3: Insert the approved policy and subgroup table**

Immediately after the existing spine-type table, replace the sentence that prioritizes C-spine with:

```markdown
## Approved review policy

- Formal OOF reporting retains all 61 cases.
- This physician-review round prioritizes L-spine, non-NHANES cases because C-spine interpretation is slower.
- NHANES remains a secondary stress test because many images have limited resolution; no case-by-case NHANES review is required in this round.
- No case is removed from training or OOF provenance by this prioritization.

| OOF subgroup | Cases | Count exact | Corner mean | Middle mean |
|---|---:|---:|---:|---:|
| Full cohort | 61 | 75.41% | 126.38 px | 118.89 px |
| Non-NHANES | 48 | 81.25% | 128.71 px | 121.68 px |
| NHANES stress test | 13 | 53.85% | 115.15 px | 106.01 px |
| L-spine non-NHANES | 40 | 87.50% | 127.30 px | 121.39 px |

NHANES is deferred for image quality and review-efficiency reasons. Its landmark-distance means are not uniformly worse, so this table does not label every NHANES prediction as inferior.
```

- [ ] **Step 4: Insert the deterministic primary queue before the original worst-case table**

Insert this exact queue:

```markdown
## Primary review queue: L-spine non-NHANES

| Priority | Case | Fold | Reason | Existing overlay |
|---:|---|---:|---|---|
| 1 | 81312013 | 5 | Extreme landmark outlier | [overlay](worst_cases/01_fold5_idx2_81312013.png) |
| 2 | 80339761 | 3 | Large multi-level error; implant-related failure suspected | [overlay](worst_cases/03_fold3_idx38_80339761.png) |
| 3 | 21383869 | 4 | Large landmark error | [overlay](worst_cases/04_fold4_idx6_21383869.png) |
| 4 | 18971571-2 | 3 | Count mismatch and legacy annotation-version issue | [overlay](worst_cases/07_fold3_idx10_18971571-2.png) |
| 5 | 81314815 | 3 | Large landmark outlier | [overlay](worst_cases/09_fold3_idx11_81314815.png) |
| 6 | 80552846 | 3 | Large landmark outlier | [overlay](worst_cases/10_fold3_idx34_80552846.png) |
| 7 | 18971571-1 | 3 | Count mismatch 6→7 | Not generated |
| 8 | 81161252 | 1 | Count mismatch 6→7 | Not generated |
| 9 | 80813115-1 | 4 | Count mismatch 6→7 | Not generated |
| 10 | 21584353 | 3 | Count mismatch 6→7 | Not generated |
```

- [ ] **Step 5: Reframe the complete mismatch table as an audit appendix**

Keep all 15 rows unchanged. Replace only the instruction above the table with:

```markdown
This complete mismatch inventory is retained for audit. The five L-spine non-NHANES mismatches are represented in the primary queue; C-spine and NHANES cases are deferred in this review round.
```

- [ ] **Step 6: Add deferred-review sections and update the decision**

Before `## Current decision`, add:

```markdown
## Deferred NHANES stress test

The 13 NHANES cases remain in formal metrics and training provenance. They are optional review cases in this round because image resolution is often limited. Do not delete or relabel them solely because they belong to NHANES.

## Deferred C-spine review

C-spine remains a model weakness, but its review is deferred because interpretation is slower. Preserve the existing C-spine mismatch rows for a later dedicated review batch.
```

Replace the current decision paragraph with:

```markdown
Do not replace production yet. Complete the ten-case L-spine non-NHANES primary queue first. Use the physician comments to separate model failures from annotation-version, implant, crop, and image-quality problems. NHANES and C-spine remain secondary review cohorts, and an independent external C-spine/L-spine holdout is still required before final all-data training or deployment.
```

- [ ] **Step 7: Run the report feature check to verify GREEN**

Run the Step 1 script again.

Expected: PASS with exit code `0`; all four physician-comment sentinels and all three new section headings are present.

---

### Task 2: Verify queue membership and immutable evidence

**Files:**
- Verify: `endplate_training_data/runs/sixpoint_reviewed_cv5_20260802_040614/oof_eval/physician_review_summary.md`
- Verify: `endplate_training_data/runs/sixpoint_reviewed_cv5_20260802_040614/oof_eval/oof_predictions.json`
- Verify: `endplate_training_data/runs/sixpoint_reviewed_cv5_20260802_040614/oof_eval/integrity_manifest.json`

**Interfaces:**
- Consumes: the revised report plus immutable OOF and hash manifests.
- Produces: fresh evidence that queue classification, overlay links, OOF coverage, and production integrity are correct.

- [ ] **Step 1: Run the deterministic queue and subgroup validator**

```powershell
@'
import json
import re
from pathlib import Path

from oof_metrics import summarize_cases

root = Path.cwd()
out = root / (
    "endplate_training_data/runs/sixpoint_reviewed_cv5_20260802_040614/"
    "oof_eval"
)
cases = json.loads(
    (out / "oof_predictions.json").read_text(encoding="utf-8")
)["cases"]
report = (out / "physician_review_summary.md").read_text(encoding="utf-8")

assert len(cases) == 61
assert {case["fold"] for case in cases} == {1, 2, 3, 4, 5}
is_nhanes = lambda case: "nhanes" in case["image_path"].lower()
nhanes = [case for case in cases if is_nhanes(case)]
non_nhanes = [case for case in cases if not is_nhanes(case)]
l_non_nhanes = [
    case for case in non_nhanes if case["spine_type"] == "L"
]
assert (len(nhanes), len(non_nhanes), len(l_non_nhanes)) == (13, 48, 40)

summary = summarize_cases(l_non_nhanes)
assert round(summary["count_exact_match_rate"] * 100, 2) == 87.50
assert round(summary["groups"]["corners"]["mean_distance_px"], 2) == 127.30
assert round(summary["groups"]["middle"]["mean_distance_px"], 2) == 121.39

section = report.split(
    "## Primary review queue: L-spine non-NHANES", 1
)[1].split("## Ten worst OOF overlays", 1)[0]
expected = {
    "81312013", "80339761", "21383869", "18971571-2",
    "81314815", "80552846", "18971571-1", "81161252",
    "80813115-1", "21584353",
}
found = set(re.findall(r"\|\s*\d+\s*\|\s*([^|\s]+)\s*\|", section))
assert found == expected
print("queue_validation=PASS primary=10 nhanes=13 non_nhanes=48")
'@ | python -
```

Expected: `queue_validation=PASS primary=10 nhanes=13 non_nhanes=48`.

- [ ] **Step 2: Verify comments, overlay links, and protected hashes**

Use Unicode-safe PNG decoding and compare current SHA-256 values with `integrity_manifest.json`:

```powershell
@'
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np

root = Path.cwd()
out = root / (
    "endplate_training_data/runs/sixpoint_reviewed_cv5_20260802_040614/"
    "oof_eval"
)
text = (out / "physician_review_summary.md").read_text(encoding="utf-8")
integrity = json.loads(
    (out / "integrity_manifest.json").read_text(encoding="utf-8")
)
run_manifest = json.loads(
    (out.parent / "run_manifest.json").read_text(encoding="utf-8")
)

comments = [
    "T12有2點不知為何偏移很多，GT正確，模型錯誤",
    "L5/S1很好，其他的有很大問題，可能有implant要另外訓練",
    "影像品質差幾乎不可判讀",
    "早期標註為S1上下緣(新版為S1上緣)，不含T12(新版有)，可能須把舊資料升級成V2.3",
]
assert all(comment in text for comment in comments)
comment_block = text.split("## Ten worst OOF overlays", 1)[1].split(
    "## All count mismatches", 1
)[0]
assert hashlib.sha256(comment_block.encode("utf-8")).hexdigest().upper() == (
    "1C5B1E9B8A374461F122865A92AD000531B119E0CC51309888FDD9310643DBC4"
)

links = set(re.findall(r"\]\((worst_cases/[^)]+\.png)\)", text))
assert len(links) == 10
for link in links:
    path = out / link
    image = cv2.imdecode(
        np.frombuffer(path.read_bytes(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert image is not None

sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest().upper()
assert sha256(root / "best_vertebra_model.pth") == (
    integrity["production_model"]["sha256_after"]
)
assert sha256(root / "best_vertebra_model.pth.ensemble") == (
    integrity["production_sidecar"]["sha256_after"]
)
for fold_number in range(1, 6):
    assert sha256(
        out.parent / f"best_vertebra_model_fold{fold_number}.pth"
    ) == integrity["fold_checkpoints"][f"fold{fold_number}"]
assert sha256(out.parent / "annotations/train_annotations.json") == (
    run_manifest["cohort"]["train_annotations_sha256"]
)
assert sha256(out.parent / "annotations/val_annotations.json") == (
    run_manifest["cohort"]["val_annotations_sha256"]
)
assert sha256(
    root / "endplate_training_data/manifests/training_allowlist_20260802.json"
) == run_manifest["cohort"]["allowlist_sha256"]
print(
    "evidence_validation=PASS comments=all overlays=10 "
    "models_annotations_allowlist_unchanged=True"
)
'@ | python -
```

Expected: `evidence_validation=PASS comments=all overlays=10 models_annotations_allowlist_unchanged=True`.

- [ ] **Step 3: Run repository-level checks**

```powershell
python -m pytest tests -q
git diff --check
git status --short
```

Expected: 77 tests pass, `git diff --check` exits `0`, and the existing unrelated dirty files remain untouched. The run-directory report is a generated evaluation artifact and is not committed.

- [ ] **Step 4: Report completion without a generated-artifact commit**

Provide the physician with links to the revised report and primary overlay queue. State explicitly that NHANES remains in formal metrics/training provenance and that production/checkpoint hashes were unchanged.
