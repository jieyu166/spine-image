'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const { loadHtmlScript } = require('./html_script_harness');

const pagePath = path.join(__dirname, '..', 'annotation-viewer.html');

function fakeFile(relativePath, options = {}) {
  const name = relativePath.replaceAll('\\', '/').split('/').pop();
  return {
    name,
    webkitRelativePath: relativePath,
    size: options.size ?? 100,
    lastModified: options.lastModified ?? 10,
    type: options.type ?? '',
    textContent: options.textContent ?? '',
  };
}

function scan(page, files, manualPairs = []) {
  page.context.__files = files;
  page.context.__manualPairs = manualPairs;
  return page.snapshot('PairScanner.scan(__files, __manualPairs)');
}

test('viewer script loads and exposes batch review components', () => {
  const page = loadHtmlScript(pagePath);
  assert.equal(page.evaluate('typeof PairScanner'), 'function');
  assert.equal(page.evaluate('typeof ReviewStore'), 'function');
  assert.equal(page.evaluate('typeof BatchReviewController'), 'function');
});

test('viewer initializes with batch controls in a browser-like environment', () => {
  const page = loadHtmlScript(pagePath);
  page.initialize();
  assert.equal(page.evaluate('canvas !== null && ctx !== null'), true);
});

test('six-point vertebra outline follows both superior and inferior middle points', () => {
  const page = loadHtmlScript(pagePath);
  page.evaluate(`
    scaleX = 1;
    scaleY = 1;
    __outlineCalls = [];
    ctx = {
      beginPath() {},
      moveTo(x, y) { __outlineCalls.push(['M', x, y]); },
      lineTo(x, y) { __outlineCalls.push(['L', x, y]); },
      closePath() { __outlineCalls.push(['Z']); },
      stroke() {},
    };
    drawVertebra({
      points: {
        anteriorSuperior: { x: 10, y: 10 },
        middleSuperior: { x: 20, y: 8 },
        posteriorSuperior: { x: 30, y: 10 },
        posteriorInferior: { x: 30, y: 30 },
        middleInferior: { x: 20, y: 32 },
        anteriorInferior: { x: 10, y: 30 },
      },
    }, 0, false, true, false, false);
  `);

  assert.deepEqual(page.snapshot('__outlineCalls'), [
    ['M', 10, 10],
    ['L', 20, 8],
    ['L', 30, 10],
    ['L', 30, 30],
    ['L', 20, 32],
    ['L', 10, 30],
    ['Z'],
  ]);
});

test('legacy four-point vertebra keeps the quadrilateral outline order', () => {
  const page = loadHtmlScript(pagePath);
  page.evaluate(`
    scaleX = 1;
    scaleY = 1;
    __outlineCalls = [];
    ctx = {
      beginPath() {},
      moveTo(x, y) { __outlineCalls.push(['M', x, y]); },
      lineTo(x, y) { __outlineCalls.push(['L', x, y]); },
      closePath() { __outlineCalls.push(['Z']); },
      stroke() {},
    };
    drawVertebra({
      points: {
        anteriorSuperior: { x: 10, y: 10 },
        posteriorSuperior: { x: 30, y: 10 },
        posteriorInferior: { x: 30, y: 30 },
        anteriorInferior: { x: 10, y: 30 },
      },
    }, 0, false, true, false, false);
  `);

  assert.deepEqual(page.snapshot('__outlineCalls'), [
    ['M', 10, 10],
    ['L', 30, 10],
    ['L', 30, 30],
    ['L', 10, 30],
    ['Z'],
  ]);
});

test('scanner preserves formal suffixes and excludes samp, ai, and model artifacts', () => {
  const page = loadHtmlScript(pagePath);
  const result = scan(page, [
    fakeFile('Images/123-2.png'), fakeFile('Images/123-2.json'),
    fakeFile('Images/123_2.png'), fakeFile('Images/123_2.json'),
    fakeFile('Images/123C.png'), fakeFile('Images/123C.json'),
    fakeFile('Images/123C0.png'), fakeFile('Images/123C0.json'),
    fakeFile('Images/123L0.png'), fakeFile('Images/123L0.json'),
    fakeFile('Images/ignoreSAMP.png'), fakeFile('Images/ignoreSAMP.json'),
    fakeFile('Images/ignoreai.json'), fakeFile('Images/ignoreMODEL.json'),
  ]);

  assert.deepEqual(result.pairs.map(pair => pair.stem), ['123-2', '123C', '123C0', '123L0', '123_2']);
  assert.deepEqual(result.excluded.map(item => item.path).sort(), [
    'Images/ignoreMODEL.json',
    'Images/ignoreSAMP.json',
    'Images/ignoreSAMP.png',
    'Images/ignoreai.json',
  ]);
  assert.equal(result.unmatchedImages.length, 0);
  assert.equal(result.unmatchedJson.length, 0);
});

test('scanner excludes exported pair-review manifests from annotation pairing', () => {
  const page = loadHtmlScript(pagePath);
  const result = scan(page, [
    fakeFile('Images/case.png'),
    fakeFile('Images/case.json'),
    fakeFile('Images/pair_review.json'),
    fakeFile('Images/20260802pair_review.json'),
  ]);

  assert.equal(result.pairs.length, 1);
  assert.equal(result.unmatchedJson.length, 0);
  assert.deepEqual(result.excluded.map(item => item.path).sort(), [
    'Images/20260802pair_review.json',
    'Images/pair_review.json',
  ]);
});

test('scanner pairs only exact full stem in the same directory', () => {
  const page = loadHtmlScript(pagePath);
  const result = scan(page, [
    fakeFile('Images/A/case.png'),
    fakeFile('Images/B/case.json'),
    fakeFile('Images/A/case2.png'),
    fakeFile('Images/A/case2.json'),
  ]);

  assert.equal(result.pairs.length, 1);
  assert.equal(result.pairs[0].imagePath, 'Images/A/case2.png');
  assert.deepEqual(result.unmatchedImages.map(item => item.path), ['Images/A/case.png']);
  assert.deepEqual(result.unmatchedJson.map(item => item.path), ['Images/B/case.json']);
});

test('known filename mismatch can be paired manually without reusing either file', () => {
  const page = loadHtmlScript(pagePath);
  const files = [
    fakeFile('Images/8014559.png'),
    fakeFile('Images/80145593.json'),
    fakeFile('Images/other.png'),
    fakeFile('Images/other.json'),
  ];
  const result = scan(page, files, [{
    imagePath: 'Images/8014559.png',
    jsonPath: 'Images/80145593.json',
  }]);

  const manual = result.pairs.find(pair => pair.manual);
  assert.equal(manual.imagePath, 'Images/8014559.png');
  assert.equal(manual.jsonPath, 'Images/80145593.json');
  assert.equal(result.unmatchedImages.length, 0);
  assert.equal(result.unmatchedJson.length, 0);

  const reused = scan(page, files, [
    { imagePath: 'Images/8014559.png', jsonPath: 'Images/80145593.json' },
    { imagePath: 'Images/8014559.png', jsonPath: 'Images/other.json' },
  ]);
  assert.ok(reused.errors.some(error => error.code === 'manual_pair_reuse'));
});

test('scanner surfaces missing pairs and duplicate conflicts instead of guessing', () => {
  const page = loadHtmlScript(pagePath);
  const result = scan(page, [
    fakeFile('Images/missing.png'),
    fakeFile('Images/duplicate.png'), fakeFile('Images/duplicate.jpg'),
    fakeFile('Images/duplicate.json'),
  ]);

  assert.deepEqual(result.unmatchedImages.map(item => item.path), ['Images/missing.png']);
  assert.deepEqual(result.unmatchedJson, []);
  assert.ok(result.conflicts.some(conflict => conflict.code === 'duplicate_image_stem'));
  assert.equal(result.pairs.length, 0);

  const manualConflict = scan(page, [
    fakeFile('Images/duplicate.png'), fakeFile('Images/duplicate.jpg'),
    fakeFile('Images/duplicate.json'),
  ], [{ imagePath: 'Images/duplicate.png', jsonPath: 'Images/duplicate.json' }]);
  assert.ok(manualConflict.errors.some(error => error.code === 'manual_pair_conflict'));
});

test('review store resumes exact datasets and retains only unchanged pair signatures', () => {
  const page = loadHtmlScript(pagePath);
  const firstScan = scan(page, [
    fakeFile('Images/a.png', { size: 10 }), fakeFile('Images/a.json', { size: 20 }),
    fakeFile('Images/b.png', { size: 30 }), fakeFile('Images/b.json', { size: 40 }),
  ]);
  page.context.__scan = firstScan;
  page.evaluate(`
    __store = new ReviewStore(localStorage);
    __manifest = __store.createManifest(__scan, {
      [__scan.pairs[0].id]: { status: 'match', reason: '', note: '', reviewedAt: '2026-08-02T00:00:00.000Z' },
      [__scan.pairs[1].id]: { status: 'mismatch', reason: 'image_patient_mismatch', note: 'wrong patient', reviewedAt: '2026-08-02T00:01:00.000Z' }
    }, [], __scan.pairs[1].id);
    __store.save(__scan, __manifest);
  `);
  const exact = page.snapshot('__store.load(__scan)');
  assert.equal(exact.status, 'exact');
  assert.equal(Object.keys(exact.manifest.reviews).length, 2);
  assert.deepEqual(exact.manifest.summary, {
    totalPairs: 2,
    reviewed: 2,
    match: 1,
    mismatch: 1,
    pending: 0,
    implant: 0,
  });

  const changedScan = scan(page, [
    fakeFile('Images/a.png', { size: 10 }), fakeFile('Images/a.json', { size: 20 }),
    fakeFile('Images/b.png', { size: 999 }), fakeFile('Images/b.json', { size: 40 }),
  ]);
  page.context.__changedScan = changedScan;
  const resumed = page.snapshot('__store.resume(__changedScan, __manifest)');
  assert.equal(Object.keys(resumed.manifest.reviews).length, 1);
  assert.deepEqual(resumed.stalePairIds, [firstScan.pairs[1].id]);
});

test('mismatch reason validation and CSV export are doctor-friendly', () => {
  const page = loadHtmlScript(pagePath);
  assert.deepEqual(page.snapshot("ReviewStore.validateDecision('mismatch', '', '')"), {
    ok: false,
    message: '不相符必須選擇原因。',
  });
  assert.equal(page.evaluate("ReviewStore.validateDecision('mismatch', 'other', '').ok"), false);
  assert.equal(page.evaluate("ReviewStore.validateDecision('mismatch', 'other', 'free text').ok"), true);

  page.context.__rows = [{
    imagePath: 'Images/a.png',
    jsonPath: 'Images/a.json',
    status: 'mismatch',
    reason: 'other',
    note: 'comma, quote " and newline\ntext',
    manual: true,
    reviewedAt: '2026-08-02T00:00:00.000Z',
  }];
  const csv = page.evaluate('ReviewStore.toCsv(__rows)');
  assert.ok(csv.startsWith('\uFEFFimage_path,json_path,status,reason,note,manual,reviewed_at'));
  assert.ok(csv.includes('"comma, quote "" and newline\ntext"'));
  assert.ok(csv.includes(',true,2026-08-02T00:00:00.000Z'));
});

test('batch auto mode scales coordinates for a near-uniform resized image', () => {
  const page = loadHtmlScript(pagePath);
  page.evaluate(`
    originalImage = { width: 1200, height: 2386 };
    annotationData = { imageInfo: { width: 1279, height: 2543 } };
    batchReviewMode = true;
    coordinateDisplayMode = 'auto';
    updateScaleFactors();
  `);
  const scales = page.snapshot('({ scaleX, scaleY })');
  assert.ok(Math.abs(scales.scaleX - (1200 / 1279)) < 1e-12);
  assert.ok(Math.abs(scales.scaleY - (2386 / 2543)) < 1e-12);
});

test('batch auto mode keeps raw coordinates for an aspect-ratio mismatch', () => {
  const page = loadHtmlScript(pagePath);
  page.evaluate(`
    originalImage = { width: 1200, height: 2200 };
    annotationData = { imageInfo: { width: 1200, height: 2400 } };
    batchReviewMode = true;
    coordinateDisplayMode = 'auto';
    updateScaleFactors();
  `);
  assert.deepEqual(page.snapshot('({ scaleX, scaleY })'), { scaleX: 1, scaleY: 1 });
});

test('loading a batch pair applies the selected coordinate display mode', async () => {
  const page = loadHtmlScript(pagePath);
  page.initialize();
  const currentScan = scan(page, [
    fakeFile('Images/a.png'),
    fakeFile('Images/a.json'),
  ]);
  page.context.__scan = currentScan;
  page.evaluate(`
    decodeImageFile = async () => ({ width: 1200, height: 2386 });
    readTextFile = async () => JSON.stringify({
      version: '2.3',
      spineType: 'L',
      imageInfo: { width: 1279, height: 2543 },
      vertebrae: [],
      discs: [],
      abnormalities: {},
    });
    coordinateDisplayMode = 'auto';
    __loadController = new BatchReviewController(new ReviewStore(localStorage));
    __loadController.scan = __scan;
    __loadController.manifest = __loadController.store.createManifest(__scan, {}, [], __scan.pairs[0].id);
  `);

  await page.evaluate('__loadController.loadCurrentPair()');
  const scales = page.snapshot('({ scaleX, scaleY })');
  assert.ok(Math.abs(scales.scaleX - (1200 / 1279)) < 1e-12);
  assert.ok(Math.abs(scales.scaleY - (2386 / 2543)) < 1e-12);
});

test('doctor can switch a uniform resize between raw and auto-scaled coordinates', () => {
  const page = loadHtmlScript(pagePath);
  page.initialize();
  assert.equal(page.evaluate('typeof setCoordinateDisplayMode'), 'function');
  page.evaluate(`
    originalImage = { width: 1200, height: 2386 };
    annotationData = {
      imageInfo: { width: 1279, height: 2543 },
      vertebrae: [],
      discs: [],
      abnormalities: {},
    };
    batchReviewMode = true;
    setCoordinateDisplayMode('raw');
  `);
  assert.deepEqual(page.snapshot('({ scaleX, scaleY, coordinateDisplayMode })'), {
    scaleX: 1,
    scaleY: 1,
    coordinateDisplayMode: 'raw',
  });

  page.evaluate("setCoordinateDisplayMode('auto')");
  const scales = page.snapshot('({ scaleX, scaleY, coordinateDisplayMode })');
  assert.equal(scales.coordinateDisplayMode, 'auto');
  assert.ok(Math.abs(scales.scaleX - (1200 / 1279)) < 1e-12);
  assert.ok(Math.abs(scales.scaleY - (2386 / 2543)) < 1e-12);
});

test('sidebar identifies a near-uniform resize separately from a crop mismatch', () => {
  const page = loadHtmlScript(pagePath);
  page.initialize();
  page.evaluate(`
    originalImage = { width: 1200, height: 2386 };
    annotationData = {
      version: '2.3',
      spineType: 'L',
      imageInfo: { width: 1279, height: 2543 },
      vertebrae: [],
      discs: [],
      abnormalities: {},
    };
    batchReviewMode = true;
    coordinateDisplayMode = 'auto';
    updateScaleFactors();
    renderSidebar();
  `);

  const info = page.document.getElementById('fileInfo').innerHTML;
  assert.match(info, /等比例縮放/);
  assert.match(info, /自動校正/);
});

test('controller records validated decisions and ignores stale async load tokens', () => {
  const page = loadHtmlScript(pagePath);
  const currentScan = scan(page, [fakeFile('Images/a.png'), fakeFile('Images/a.json')]);
  page.context.__scan = currentScan;
  page.evaluate(`
    __controller = new BatchReviewController(new ReviewStore(localStorage));
    __controller.scan = __scan;
    __controller.manifest = __controller.store.createManifest(__scan, {}, [], __scan.pairs[0].id);
    __firstToken = __controller.beginLoad();
    __secondToken = __controller.beginLoad();
  `);
  assert.equal(page.evaluate('__controller.isCurrentLoad(__firstToken)'), false);
  assert.equal(page.evaluate('__controller.isCurrentLoad(__secondToken)'), true);
  assert.equal(page.evaluate("__controller.recordDecision('mismatch', '', '', { render: false, advance: false })"), false);
  assert.equal(page.evaluate("__controller.recordDecision('match', '', '', { render: false, advance: false })"), true);
  assert.equal(page.evaluate("__controller.manifest.reviews[__scan.pairs[0].id].status"), 'match');
});

test('skip leaves the pair pending and advances without writing a decision', () => {
  const page = loadHtmlScript(pagePath);
  const currentScan = scan(page, [
    fakeFile('Images/a.png'), fakeFile('Images/a.json'),
    fakeFile('Images/b.png'), fakeFile('Images/b.json'),
  ]);
  page.context.__scan = currentScan;
  page.evaluate(`
    __skipController = new BatchReviewController(new ReviewStore(localStorage));
    __skipController.scan = __scan;
    __skipController.manifest = __skipController.store.createManifest(__scan, {}, [], __scan.pairs[0].id);
  `);
  assert.equal(page.evaluate('__skipController.skip({ render: false })'), true);
  assert.equal(page.evaluate('__skipController.currentIndex'), 1);
  assert.equal(page.evaluate('Object.keys(__skipController.manifest.reviews).length'), 0);
});

test('batch review controls are present in the standalone HTML', () => {
  const html = fs.readFileSync(pagePath, 'utf8');
  for (const id of [
    'reviewFolderInput', 'manifestInput', 'batchReviewPanel', 'batchProgress',
    'batchPairName', 'mismatchReason', 'mismatchNote', 'manualImageSelect',
    'manualJsonSelect', 'batchIssues', 'coordinateDisplayMode'
  ]) {
    assert.match(html, new RegExp(`id=["']${id}["']`));
  }
  assert.match(html, /webkitdirectory/);
  assert.match(html, /pair_review\.json/);
  assert.match(html, /pair_review\.csv/);
});

test('single-file mode cancels pending batch loads and restores coordinate scaling', () => {
  const page = loadHtmlScript(pagePath);
  page.evaluate('batchReviewMode = true; batchReviewController.loadToken = 5; exitBatchReviewMode()');
  assert.equal(page.evaluate('batchReviewMode'), false);
  assert.equal(page.evaluate('batchReviewController.loadToken'), 6);
});

test('imported review manifest from a different root folder is rejected', () => {
  const page = loadHtmlScript(pagePath);
  page.context.__files = [fakeFile('Images/a.png'), fakeFile('Images/a.json')];
  page.evaluate('batchReviewController.files = __files');
  page.context.__manifest = {
    version: '1.0',
    dataset: { rootName: 'OtherImages', formalPaths: ['OtherImages/a.png', 'OtherImages/a.json'] },
    pairs: [], reviews: {}, manualPairs: [], currentPairId: null,
  };
  assert.equal(page.evaluate('batchReviewController.loadImportedManifest(__manifest)'), false);
});

test('scanner excludes exported case_flags.json from annotation pairing', () => {
  const page = loadHtmlScript(pagePath);
  const result = scan(page, [
    fakeFile('Images/a.png'), fakeFile('Images/a.json'),
    fakeFile('Images/case_flags.json'),
  ]);
  assert.equal(result.pairs.length, 1);
  assert.deepEqual(result.excluded.map(item => item.path), ['Images/case_flags.json']);
});

test('implant case flags are recorded without a decision, survive resume, and export as case_flags.json', () => {
  const page = loadHtmlScript(pagePath);
  page.initialize();
  const currentScan = scan(page, [
    fakeFile('Images/a.png'), fakeFile('Images/a.json'),
    fakeFile('Images/b.png'), fakeFile('Images/b.json'),
  ]);
  page.context.__scan = currentScan;
  page.evaluate(`
    __flagController = new BatchReviewController(new ReviewStore(localStorage));
    __flagController.scan = __scan;
    __flagController.manifest = __flagController.store.createManifest(__scan, {}, [], __scan.pairs[0].id);
    batchReviewMode = true;
  `);
  assert.deepEqual(page.snapshot('__flagController.currentCaseFlags()'), { hasImplant: false, implantNote: '', source: null });
  assert.equal(page.evaluate("__flagController.setCaseFlags({ hasImplant: true, implantNote: ' L4-L5 cage ' })"), true);
  assert.equal(page.evaluate('Object.keys(__flagController.manifest.reviews).length'), 0, 'no decision written');
  assert.deepEqual(page.snapshot('__flagController.currentCaseFlags()'), { hasImplant: true, implantNote: 'L4-L5 cage', source: 'review' });
  assert.equal(page.evaluate('__flagController.manifest.summary.implant'), 1);
  assert.equal(page.evaluate('__flagController.toggleImplant() && __flagController.currentCaseFlags().hasImplant'), false);
  page.evaluate('__flagController.toggleImplant()');

  // JSON 自帶的 caseFlags 會顯示為預設值（來源 annotation）
  page.evaluate(`
    __flagController.currentIndex = 1;
    annotationData = { caseFlags: { hasImplant: true, implantNote: 'from json' } };
  `);
  assert.deepEqual(page.snapshot('__flagController.currentCaseFlags()'), { hasImplant: true, implantNote: 'from json', source: 'annotation' });

  const resumed = page.snapshot('__flagController.store.resume(__scan, __flagController.manifest).manifest.caseFlags');
  assert.equal(Object.keys(resumed).length, 1);
  const exported = page.snapshot('__flagController.caseFlagsExport()');
  assert.equal(exported.kind, 'case_flags');
  assert.deepEqual(exported.entries.map(e => [e.imagePath, e.jsonPath, e.hasImplant, e.implantNote]),
    [['Images/a.png', 'Images/a.json', true, 'L4-L5 cage']]);
});
