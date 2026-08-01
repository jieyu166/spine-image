# Annotation Case Safety and Pair Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent cross-patient annotation state from surviving an image change and extend `annotation-viewer.html` into a resumable physician review tool for image/JSON pairing.

**Architecture:** Keep both deliverables as standalone HTML files and test their real inline JavaScript through Node's built-in `node:test` and `vm` modules. `spinal-annotation-web.html` receives one atomic case-state boundary plus shared image-identity validation. `annotation-viewer.html` keeps its existing decoder and overlay renderer and adds isolated `PairScanner`, `ReviewStore`, and `BatchReviewController` units.

**Tech Stack:** Plain HTML/CSS/JavaScript, Web File APIs, Web Crypto SHA-256, localStorage, Node.js 22 built-in test runner, Python 3.14/pytest for existing regressions.

## Global Constraints

- Original images and physician JSON files remain read-only.
- The pages must remain directly usable from Chrome/Edge as local standalone HTML files; add no CDN, server, framework, or package dependency.
- Keep the existing single-image viewer workflow available.
- Match exact full stems within the same relative directory; `-2`, `_2`, `C`, `C0`, `L`, and `L0` remain part of the ID.
- Exclude stems ending in `samp`, `ai`, or `model`, case-insensitively.
- Never auto-guess a fuzzy pairing; unmatched files require explicit physician pairing.
- Store review metadata only, never image bytes or full annotation JSON, in localStorage.
- Preserve the unrelated untracked `HANDOFF_2026-07-11.md` file.

## File Map

- Create `tests/html_script_harness.js`: load the real inline script from a standalone HTML file into a controlled Node VM context.
- Create `tests/spinal_annotation_web.test.js`: regression coverage for case reset, dirty state, image identity, and export identity.
- Modify `spinal-annotation-web.html`: atomic new-case transition, complete reset, identity validation, and identity-aware export.
- Create `tests/annotation_viewer_batch.test.js`: behavior tests for scan rules, review persistence, stale handling, decisions, and exports.
- Modify `annotation-viewer.html`: batch-review markup, styles, scanner, store, controller, and integration with the current renderer.
- Modify this plan only to check completed boxes while executing; do not edit the approved design specification.

---

### Task 1: Real Inline-Script Test Harness

**Files:**
- Create: `tests/html_script_harness.js`
- Test: `tests/spinal_annotation_web.test.js`

**Interfaces:**
- Produces: `loadHtmlScript(relativePath, overrides = {}) -> { context, run, snapshot, elements, alerts, downloads }`
- Produces: DOM element fakes with `classList`, `style`, `addEventListener`, `click`, `getContext`, and `getBoundingClientRect`.
- Consumes: Node built-ins only: `fs`, `path`, and `vm`.

- [ ] **Step 1: Write the failing harness smoke test**

Create `tests/spinal_annotation_web.test.js` with a smoke test that loads the real script and evaluates an existing function:

```js
const test = require('node:test');
const assert = require('node:assert/strict');
const { loadHtmlScript } = require('./html_script_harness');

test('loads the real spinal annotation inline script', () => {
  const app = loadHtmlScript('spinal-annotation-web.html');
  assert.equal(app.run('midpoint({x: 2, y: 4}, {x: 6, y: 8}).x'), 4);
  assert.equal(app.run('midpoint({x: 2, y: 4}, {x: 6, y: 8}).y'), 6);
});
```

- [ ] **Step 2: Run the smoke test and verify the missing harness failure**

Run:

```powershell
node --test tests/spinal_annotation_web.test.js
```

Expected: FAIL with `Cannot find module './html_script_harness'`.

- [ ] **Step 3: Implement the VM harness**

The harness must:

1. Read the requested HTML as UTF-8.
2. Extract the only inline `<script>...</script>` block.
3. Create stable fake elements on demand from `document.getElementById`.
4. Provide `window`, `document`, `navigator`, `localStorage`, `URL`, `Blob`, `FileReader`, `Image`, `crypto.webcrypto`, `alert`, `confirm`, `setTimeout`, and `clearTimeout`.
5. Evaluate the real script once and return a `run(source)` helper backed by `vm.runInContext`.
6. Return `snapshot(source)`, which JSON-clones serializable VM results before host-side deep equality checks so cross-realm prototypes do not create false failures.

Use this storage behavior:

```js
function createMemoryStorage() {
  const values = new Map();
  return {
    getItem: key => values.has(key) ? values.get(key) : null,
    setItem: (key, value) => values.set(key, String(value)),
    removeItem: key => values.delete(key),
    clear: () => values.clear(),
  };
}
```

The fake anchor element must append `{ href, download }` to `downloads` when `click()` runs. Do not assert on fake internals except to inspect the filename and Blob created by the production export path.

- [ ] **Step 4: Run the smoke test and verify it passes**

Run:

```powershell
node --test tests/spinal_annotation_web.test.js
```

Expected: 1 test passes with no warning or unhandled rejection.

- [ ] **Step 5: Commit the test harness**

```powershell
git add tests/html_script_harness.js tests/spinal_annotation_web.test.js
git commit -m "test: add standalone html script harness"
```

---

### Task 2: Atomic Annotation Case Reset

**Files:**
- Modify: `tests/spinal_annotation_web.test.js`
- Modify: `spinal-annotation-web.html:526-539,619-671,791-965,1055-1117,1232-1295`

**Interfaces:**
- Produces: `hasAnnotationWork() -> boolean`
- Produces: `resetCaseState({ dirty = false, keepImage = true } = {}) -> void`
- Produces: `commitLoadedImage(file, img, { fitToWidth = false } = {}) -> void`
- Produces state: `annotationDirty: boolean`, `currentImageMeta: object | null`
- Consumes: existing `resetZoom`, `redraw`, and `updateUI` functions.

- [ ] **Step 1: Write failing state-transition tests**

Add tests using the real functions and lexical state through `app.run(...)`:

```js
test('committing a different-sized image clears every previous case state', () => {
  const app = loadHtmlScript('spinal-annotation-web.html');
  app.run(`
    canvas = document.getElementById('mainCanvas');
    ctx = canvas.getContext('2d');
    redraw = () => {};
    updateUI = () => {};
    vertebrae = [{name:'L1', points:[{x:900,y:1200}]}];
    currentPoints = [{x:50,y:60}];
    annotationFinished = true;
    selectedVertebrae.add(0);
    dragInfo = {vertebraIdx:0, pointIdx:0};
    groupDrag = {lastImgX:1,lastImgY:2};
    selectBox = {startX:1,startY:2,endX:3,endY:4};
    isDragging = true;
    wasDragging = true;
    _justDragged = true;
    commitLoadedImage({name:'next.png',size:12,lastModified:34}, {width:1200,height:1440,naturalWidth:1200,naturalHeight:1440});
  `);
  assert.deepEqual(app.snapshot(`({
    vertebrae: vertebrae.length,
    currentPoints: currentPoints.length,
    finished: annotationFinished,
    selected: selectedVertebrae.size,
    dragInfo, groupDrag, selectBox, isDragging, wasDragging, justDragged:_justDragged,
    width:canvas.width, height:canvas.height, zoom, panX, panY,
    fileName:currentImageMeta.fileName, dirty:annotationDirty
  })`), {
    vertebrae:0, currentPoints:0, finished:false, selected:0,
    dragInfo:null, groupDrag:null, selectBox:null,
    isDragging:false, wasDragging:false, justDragged:false,
    width:1200, height:1440, zoom:1, panX:0, panY:0,
    fileName:'next.png', dirty:false,
  });
});

test('clear all clears transient selection and marks the case dirty', () => {
  const app = loadHtmlScript('spinal-annotation-web.html', { confirm: () => true });
  app.run(`
    canvas = document.getElementById('mainCanvas'); ctx = canvas.getContext('2d');
    redraw = () => {}; updateUI = () => {};
    vertebrae=[{name:'L1',points:[]}]; selectedVertebrae.add(0);
    dragInfo={vertebraIdx:0,pointIdx:0}; groupDrag={}; selectBox={};
    clearAll();
  `);
  assert.deepEqual(app.snapshot(`({v:vertebrae.length,s:selectedVertebrae.size,dragInfo,groupDrag,selectBox,dirty:annotationDirty})`),
    {v:0,s:0,dragInfo:null,groupDrag:null,selectBox:null,dirty:true});
});
```

Add a third test in which a fake image emits `onerror`; assert the existing `originalImage`, `vertebrae`, and `currentImageMeta` remain unchanged.

- [ ] **Step 2: Run the tests and verify the expected missing-function failures**

Run:

```powershell
node --test tests/spinal_annotation_web.test.js
```

Expected: the smoke test passes; new tests fail because `commitLoadedImage`, `resetCaseState`, and `annotationDirty` do not exist.

- [ ] **Step 3: Implement the minimal atomic case boundary**

Add `annotationDirty` and `currentImageMeta` beside existing state. Implement `resetCaseState` so all annotation and interaction fields in the design are reset from one function. `keepImage: false` also clears `originalImage`, processed image state, and image metadata.

Implement `commitLoadedImage` so it:

1. Calls `resetCaseState({dirty:false, keepImage:false})` only after an image has decoded.
2. Sets `originalImage` and `currentImageMeta.fileName/size/lastModified`.
3. Uses natural dimensions.
4. Uses full natural-size canvas for normal annotation loads and the existing 1200-pixel fit only for `_loadImageFile` compatibility.
5. Resets zoom/pan and updates the UI.

Make `loadImageFromFile` and `_loadImageFile` call this helper. Add `img.onerror` and `reader.onerror` alerts that do not reset the current case. Clear each file input value in its `change` handler.

Replace `clearAll` internals with `resetCaseState({dirty:true, keepImage:true})`. Mark `annotationDirty = true` after point creation, point/group dragging, undo, deletion, early finish, spine-type changes that clear points, and JSON import. Mark it false only after a successful export or a successful new-image commit.

- [ ] **Step 4: Run state tests and all existing Node tests**

Run:

```powershell
node --test tests/spinal_annotation_web.test.js
```

Expected: all state tests pass.

- [ ] **Step 5: Commit the atomic reset**

```powershell
git add spinal-annotation-web.html tests/spinal_annotation_web.test.js
git commit -m "fix: isolate annotation state between images"
```

---

### Task 3: Image Identity Validation and Export

**Files:**
- Modify: `tests/spinal_annotation_web.test.js`
- Modify: `spinal-annotation-web.html:1297-1441,1443-1590,2283-2385`

**Interfaces:**
- Produces: `fileStem(fileName) -> string`
- Produces: `computeFileSha256(file) -> Promise<string | null>`
- Produces: `validateAnnotationImage(data, jsonFileName) -> { ok:boolean, errors:string[], warnings:string[] }`
- Produces: `buildExportImageInfo() -> Promise<{width,height,fileName,sha256}>`
- Changes: `importAnnotationJson(data, options = {}) -> Promise<boolean>`
- Changes: `exportData() -> Promise<boolean>`

- [ ] **Step 1: Write failing identity tests**

Use literal fixtures and preserve a sentinel annotation to prove rejected imports do not mutate state:

```js
test('rejects a legacy JSON whose dimensions differ without clearing current points', async () => {
  const app = loadHtmlScript('spinal-annotation-web.html');
  app.run(`
    canvas=document.getElementById('mainCanvas'); ctx=canvas.getContext('2d');
    redraw=()=>{}; updateUI=()=>{};
    originalImage={width:1200,height:1440,naturalWidth:1200,naturalHeight:1440};
    currentImageMeta={fileName:'C00172_cervical_masks.png',sha256:null};
    vertebrae=[{name:'C2',points:[{x:10,y:20}]}];
  `);
  const result = await app.run(`importAnnotationJson({
    version:'2.3', imageInfo:{width:1462,height:1755},
    vertebrae:[{name:'C2',boundaryType:'lower',points:{
      anteriorInferior:{x:779,y:873},middleInferior:{x:813,y:916},posteriorInferior:{x:824,y:966}
    }}]
  }, {jsonFileName:'C00172_cervical_masks.json'})`);
  assert.equal(result, false);
  assert.equal(app.run('vertebrae[0].points[0].x'), 10);
});
```

Also add tests for:

- same dimensions but new-format filename mismatch;
- same filename/dimensions but SHA-256 mismatch;
- legacy exact-stem and exact-dimension import succeeds with a warning;
- exported `imageInfo` contains natural dimensions, filename, and a 64-character SHA-256;
- exported download filename is the image stem plus `.json`.

- [ ] **Step 2: Run tests and verify current unsafe behavior**

Run:

```powershell
node --test tests/spinal_annotation_web.test.js
```

Expected: mismatch tests fail because current import accepts coordinates unchanged; export metadata tests fail because current output has only width/height and a timestamp filename.

- [ ] **Step 3: Implement identity helpers and strict imports**

Implement `fileStem` by removing only the final extension. Compute SHA-256 from `file.arrayBuffer()` through `crypto.subtle.digest`; return null with an explicit metadata warning when unavailable.

When committing an image, start and retain `currentImageMeta.sha256Promise`; populate `currentImageMeta.sha256` when resolved. `validateAnnotationImage` must compare:

1. natural dimensions;
2. `data.imageInfo.fileName` when present;
3. `data.imageInfo.sha256` when present;
4. otherwise the supplied JSON filename stem against the current image stem.

Return all errors without mutating state. `importAnnotationJson` validates before `_finishImport`; `_applySpineFMData` uses the same path and removes the old confirm-and-proceed mismatch branch. A JSON containing `originalImageBase64` validates against the decoded embedded image, not the previously visible image.

Update `handleJsonImport` to pass the actual JSON filename. A JSON-only SpineFM selection must not silently bind to the previously displayed image unless the strict validator accepts it.

- [ ] **Step 4: Implement identity-aware export**

Make `exportData` async. Await `buildExportImageInfo`, place its result in `data.imageInfo`, and download to `${fileStem(currentImageMeta.fileName)}.json`. Set `annotationDirty = false` only after `a.click()` succeeds.

- [ ] **Step 5: Run identity tests and the full annotation test file**

Run:

```powershell
node --test tests/spinal_annotation_web.test.js
```

Expected: all tests pass with rejected imports leaving sentinel annotations intact.

- [ ] **Step 6: Commit image identity safeguards**

```powershell
git add spinal-annotation-web.html tests/spinal_annotation_web.test.js
git commit -m "fix: validate annotation json image identity"
```

---

### Task 4: PairScanner and Batch Inventory

**Files:**
- Create: `tests/annotation_viewer_batch.test.js`
- Modify: `annotation-viewer.html:221-242` and add batch scanner functions before existing init code.

**Interfaces:**
- Produces: `PairScanner.scan(files) -> {rootName, exactPairs, unpairedImages, unpairedJson, conflicts, excluded, errors}`
- Produces pair: `{id,imageFile,jsonFile,imagePath,jsonPath,manual:false}`
- Produces: `PairScanner.createManualPair(scanResult, imagePath, jsonPath) -> pair`
- Consumes: File-like values with `name`, `webkitRelativePath`, `size`, and `lastModified`.

- [ ] **Step 1: Write failing table-driven scanner tests**

Create literal File-like fixtures:

```js
function file(path, size = 100, lastModified = 1) {
  const name = path.split('/').at(-1);
  return {name, webkitRelativePath:path, size, lastModified};
}

test('scanner preserves formal suffixes and excludes only review artifacts', () => {
  const app = loadHtmlScript('annotation-viewer.html');
  const result = app.run(`PairScanner.scan(${JSON.stringify([
    file('Images/202607/81312903-2.png'),
    file('Images/202607/81312903-2.json'),
    file('Images/202607/81312903_2.png'),
    file('Images/202607/81312903_2.json'),
    file('Images/202607/123C0.png'),
    file('Images/202607/123C0.json'),
    file('Images/202607/123L0.png'),
    file('Images/202607/123L0.json'),
    file('Images/202607/123samp.png'),
    file('Images/202607/123ai.json'),
    file('Images/202607/123model.json'),
  ])})`);
  assert.deepEqual(result.exactPairs.map(p => p.imagePath), [
    '202607/123C0.png', '202607/123L0.png',
    '202607/81312903-2.png', '202607/81312903_2.png',
  ]);
  assert.equal(result.excluded.length, 3);
});
```

Add separate tests proving:

- `-2.json` does not pair `_2.png`;
- missing JSON and missing image are separate errors;
- two image formats for one stem produce a conflict;
- `createManualPair` records `manual:true` and complete relative paths;
- reusing an already paired image without explicit override throws.

- [ ] **Step 2: Run scanner tests and verify `PairScanner` is missing**

Run:

```powershell
node --test tests/annotation_viewer_batch.test.js
```

Expected: FAIL with `PairScanner is not defined`.

- [ ] **Step 3: Implement PairScanner as pure JavaScript**

Use case-insensitive extension classification but preserve original relative paths. Strip the selected root folder prefix from `webkitRelativePath`. Group only by `${relativeDirectory}/${fullStem}`. Sort every output array by relative path for deterministic review order and exports.

Artifact exclusion applies when the stem's lowercase text ends in exactly `samp`, `ai`, or `model`. Do not normalize dashes, underscores, or formal suffixes. Supported image extensions must match the formats the current viewer can decode.

- [ ] **Step 4: Run scanner tests**

Run:

```powershell
node --test tests/annotation_viewer_batch.test.js
```

Expected: all PairScanner tests pass.

- [ ] **Step 5: Commit PairScanner**

```powershell
git add annotation-viewer.html tests/annotation_viewer_batch.test.js
git commit -m "feat: scan annotation image json pairs"
```

---

### Task 5: ReviewStore, Resume, Stale, and Exports

**Files:**
- Modify: `tests/annotation_viewer_batch.test.js`
- Modify: `annotation-viewer.html` near the new PairScanner unit.

**Interfaces:**
- Produces: `new ReviewStore(storage, now = () => new Date().toISOString())`
- Produces: `ReviewStore.datasetKey(rootName, formalPaths) -> Promise<string>`
- Produces: `ReviewStore.reconcile(previousManifest, scanResult) -> manifest`
- Produces: `ReviewStore.record(manifest, pairId, decision, reason, note) -> review`
- Produces: `ReviewStore.toJson(manifest) -> string`
- Produces: `ReviewStore.toCsv(manifest) -> string`
- Produces: `fileSignature(file, relativePath, sha256) -> {path,size,lastModified,sha256}`

- [ ] **Step 1: Write failing ReviewStore behavior tests**

Use an in-memory Storage double only at the browser boundary; assert ReviewStore results rather than calls to the double.

Required tests:

```js
test('mismatch requires a reason and other also requires a note', () => {
  const app = loadHtmlScript('annotation-viewer.html');
  assert.throws(() => app.run(`new ReviewStore(localStorage).validateDecision('mismatch', '', '')`));
  assert.throws(() => app.run(`new ReviewStore(localStorage).validateDecision('mismatch', 'other', '')`));
  assert.equal(app.run(`new ReviewStore(localStorage).validateDecision('mismatch', 'image_unreadable', '')`), true);
});
```

Add tests proving:

- `match` saves a timestamp and no mismatch reason;
- save/load resumes the same pair decisions;
- identical paths and signatures retain decisions;
- a changed size, lastModified, or hash marks the review stale and pending;
- an imported manifest merges by pair ID/signature rather than array order;
- adding or removing a path under the same root name finds the most recent root index entry and carries forward only unchanged pair signatures;
- multiple incompatible saved datasets with the same root name require explicit manifest selection instead of silent merging;
- removed pairs stay in history but not the active total;
- JSON summary equals counts derived from reviews;
- CSV begins with UTF-8 BOM and contains escaped Traditional Chinese notes.

- [ ] **Step 2: Run tests and verify `ReviewStore` is missing**

Run:

```powershell
node --test tests/annotation_viewer_batch.test.js
```

Expected: scanner tests pass; ReviewStore tests fail with `ReviewStore is not defined`.

- [ ] **Step 3: Implement ReviewStore and deterministic schemas**

Implement manifest version `1.0`, the six fixed mismatch reason codes, pair-ID/signature reconciliation, derived summaries, and localStorage keys namespaced as `spine-pair-review:`. Dataset identity uses root name plus sorted formal relative paths; per-pair signatures use path, size, lastModified, and lazily computed SHA-256. Maintain a small `spine-pair-review:index:<rootName>` list of saved dataset keys so an inventory change can locate the most recent compatible state. If more than one incompatible candidate remains, return a `selectionRequired` result and do not merge automatically.

Do not store File objects in localStorage. Keep live File references only in controller memory and serialize metadata only.

`toCsv` must use a literal ordered header, RFC-4180 escaping, CRLF rows, and prefix `\uFEFF`.

- [ ] **Step 4: Run ReviewStore tests**

Run:

```powershell
node --test tests/annotation_viewer_batch.test.js
```

Expected: scanner and store tests pass.

- [ ] **Step 5: Commit review persistence**

```powershell
git add annotation-viewer.html tests/annotation_viewer_batch.test.js
git commit -m "feat: persist pair review decisions"
```

---

### Task 6: BatchReviewController and Physician UI

**Files:**
- Modify: `tests/annotation_viewer_batch.test.js`
- Modify: `annotation-viewer.html:1-223,244-397,1064-1145`

**Interfaces:**
- Produces: `new BatchReviewController({scanner, store})`
- Produces: `controller.loadFolder(files) -> Promise<void>`
- Produces: `controller.openPair(pairId) -> Promise<void>`
- Produces: `controller.decide(decision, reason = null, note = '') -> boolean`
- Produces: `controller.nextPending()`, `previous()`, `next()`, `filter(mode)`
- Consumes: existing `setImage`, `normalizeAnnotation`, `updateScaleFactors`, `resetView`, `renderSidebar`, and `redraw`.

- [ ] **Step 1: Write failing controller tests**

Add behavior tests proving:

- folder load selects the first pending exact pair;
- `match` saves and advances to the next pending pair;
- invalid mismatch does not advance;
- `skip` leaves the decision pending and advances;
- filter `mismatch` contains only mismatch decisions;
- an older asynchronous `openPair` completion cannot overwrite a newer selection (request-token test);
- batch mode leaves `scaleX` and `scaleY` at `1` when JSON dimensions differ, so automatic fit-to-image scaling cannot hide the mismatch;
- keyboard handling ignores events from `INPUT`, `TEXTAREA`, and `SELECT`.

Use controlled FileReader/Image fakes to exercise the real controller and existing `setImage`/normalizer path. Assert controller state and rendered consumer-visible counters, not the existence of fake elements.

- [ ] **Step 2: Run tests and verify controller behavior is absent**

Run:

```powershell
node --test tests/annotation_viewer_batch.test.js
```

Expected: scanner/store tests pass; controller tests fail because `BatchReviewController` is undefined.

- [ ] **Step 3: Add batch-review markup and styles**

Add:

- a hidden `folderInput` with `webkitdirectory multiple`;
- top controls for folder selection, progress counters, filters, manifest import, JSON export, and CSV export;
- review actions for match, mismatch, skip, previous, and next;
- a mismatch panel with the six reasons and note input;
- a pending-pair panel with unused same-directory candidates;
- expandable excluded/error lists;
- high-risk red and legacy yellow warning cards.

Reuse the existing responsive container and sidebar. Do not remove single-image buttons.

- [ ] **Step 4: Implement BatchReviewController integration**

`loadFolder` scans, reconciles prior localStorage state, renders the summary, and opens the first pending pair. `openPair` reads the selected JSON and image, computes checks/hashes, calls existing rendering functions, and discards stale async completions using an incrementing token.

In batch-review mode, mismatched `imageInfo` dimensions must render with raw annotation coordinates (`scaleX = 1`, `scaleY = 1`) so the viewer shows what the training pipeline would consume. Preserve the existing single-file viewer's fit-scaling behavior outside batch mode for backward compatibility.

`decide` delegates validation and persistence to ReviewStore, re-renders counters, and advances to the next pending pair. Manual pairing updates the in-memory scan result and manifest before opening the pair.

Keyboard shortcuts:

- `M`: match
- `X`: open mismatch panel
- `J` or right arrow: next
- `K` or left arrow: previous
- `S`: skip

- [ ] **Step 5: Implement manifest import and downloads**

Manifest import validates `version === '1.0'` and reconciles by pair ID/signatures. JSON and CSV export derive summaries immediately before Blob creation. Download names are `pair_review.json` and `pair_review.csv`.

- [ ] **Step 6: Run all Node tests**

Run:

```powershell
node --test tests/spinal_annotation_web.test.js tests/annotation_viewer_batch.test.js
```

Expected: every test passes with no unhandled asynchronous work.

- [ ] **Step 7: Commit the physician review UI**

```powershell
git add annotation-viewer.html tests/annotation_viewer_batch.test.js
git commit -m "feat: add resumable physician pair review"
```

---

### Task 7: Full Regression and Chrome Workflow Verification

**Files:**
- Modify only if a failing verification exposes an in-scope defect: `spinal-annotation-web.html`, `annotation-viewer.html`, or their two Node test files.

**Interfaces:**
- Consumes all interfaces from Tasks 1-6.
- Produces no new production API.

- [ ] **Step 1: Run complete automated tests**

Run:

```powershell
node --test tests/spinal_annotation_web.test.js tests/annotation_viewer_batch.test.js
python -m pytest -q
```

Expected: all Node and Python tests pass.

- [ ] **Step 2: Validate HTML scripts parse cleanly**

Use the VM harness to load both HTML files without invoking UI initialization:

```powershell
node -e "const {loadHtmlScript}=require('./tests/html_script_harness'); loadHtmlScript('spinal-annotation-web.html'); loadHtmlScript('annotation-viewer.html'); console.log('HTML_SCRIPT_PARSE_OK')"
```

Expected: `HTML_SCRIPT_PARSE_OK`.

- [ ] **Step 3: Run current-folder inventory smoke check**

In Chrome, select the current `Images` folder and verify:

1. `samp`, `ai`, and `model` appear only under excluded files.
2. `-2`, `_2`, `C0`, and `L0` exact pairs remain distinct.
3. unmatched primary files appear under pending/errors.
4. `80145593.json` can be manually paired to `8014559.png` without reusing the image.

- [ ] **Step 4: Verify different-size annotation workflow**

In `spinal-annotation-web.html`:

1. Load and annotate an image larger than 1200×1440.
2. Export its JSON and confirm image filename, natural dimensions, and SHA-256 are present.
3. Load a 1200×1440 image and confirm the canvas contains zero old points before any click.
4. Zoom, pan, place/move a point, export, re-import with the same image, and confirm visual round-trip.
5. Attempt to import the first image's JSON onto the second image and confirm the import is blocked without changing current points.

- [ ] **Step 5: Verify review persistence and exports**

In `annotation-viewer.html`:

1. Mark one pair match and one mismatch with a required reason and Traditional Chinese note.
2. Close/reopen the page, reselect the folder, and confirm both decisions resume.
3. Modify a copied fixture file, reselect the fixture folder, and confirm the old decision becomes stale/pending.
4. Export/import `pair_review.json` and export `pair_review.csv`; verify counts, paths, reason, note, manual flag, and timestamps agree.

- [ ] **Step 6: Inspect final diff and working tree scope**

Run:

```powershell
git diff --check
git status --short
git log --oneline -8
```

Expected: no whitespace errors; only intended files are modified/committed; `HANDOFF_2026-07-11.md` remains untouched and untracked.

- [ ] **Step 7: Commit any verification-only corrections**

Only when Step 1-5 revealed and fixed an in-scope issue:

```powershell
git add spinal-annotation-web.html annotation-viewer.html tests/html_script_harness.js tests/spinal_annotation_web.test.js tests/annotation_viewer_batch.test.js
git commit -m "fix: harden annotation review workflow"
```

If no corrections were needed, do not create an empty commit.
