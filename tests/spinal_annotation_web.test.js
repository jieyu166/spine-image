'use strict';

const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');
const { loadHtmlScript } = require('./html_script_harness');

const pagePath = path.join(__dirname, '..', 'spinal-annotation-web.html');

test('annotation page script loads and initializes without a browser', () => {
  const page = loadHtmlScript(pagePath);
  page.initialize();
  assert.equal(page.evaluate('typeof exportData'), 'function');
  assert.equal(page.evaluate('typeof importAnnotationJson'), 'function');
});

test('annotation page exposes a single atomic case reset helper', () => {
  const page = loadHtmlScript(pagePath);
  assert.equal(page.evaluate('typeof resetCaseState'), 'function');
});

test('atomic reset clears annotations, interactions, image state, and pending imports', () => {
  const page = loadHtmlScript(pagePath);
  page.evaluate(`
    originalImage = { width: 640, height: 480 };
    currentImageMeta = { filename: 'old.png' };
    vertebrae = [{ name: 'L1', points: [] }];
    currentPoints = [{ x: 1, y: 2 }];
    annotationFinished = true;
    annotationDirty = true;
    isDragging = true;
    wasDragging = true;
    dragInfo = { vertebraIdx: 0, pointIdx: 0 };
    groupDrag = { lastImgX: 1, lastImgY: 2 };
    _justDragged = true;
    selectBox = { startX: 0, startY: 0, endX: 2, endY: 2 };
    selectedVertebrae.add(0);
    contrastEnhanced = true;
    processedImage = { cached: true };
    zoom = 2;
    panX = 20;
    panY = 30;
    window._pendingSpineFM = { vertebrae: [] };
    window._spinefmJsonName = 'old.json';
    resetCaseState();
  `);

  assert.deepEqual(page.snapshot(`({
    originalImage,
    currentImageMeta,
    vertebrae,
    currentPoints,
    annotationFinished,
    annotationDirty,
    isDragging,
    wasDragging,
    dragInfo,
    groupDrag,
    justDragged: _justDragged,
    selectBox,
    selectedCount: selectedVertebrae.size,
    contrastEnhanced,
    processedImage,
    zoom,
    panX,
    panY,
    pending: window._pendingSpineFM,
    pendingName: window._spinefmJsonName
  })`), {
    originalImage: null,
    currentImageMeta: null,
    vertebrae: [],
    currentPoints: [],
    annotationFinished: false,
    annotationDirty: false,
    isDragging: false,
    wasDragging: false,
    dragInfo: null,
    groupDrag: null,
    justDragged: false,
    selectBox: null,
    selectedCount: 0,
    contrastEnhanced: false,
    processedImage: null,
    zoom: 1,
    panX: 0,
    panY: 0,
    pending: null,
    pendingName: null,
  });
});

test('clear all removes interaction state but keeps the current image', () => {
  const page = loadHtmlScript(pagePath);
  page.initialize();
  page.evaluate(`
    originalImage = { width: 640, height: 480 };
    currentImageMeta = { filename: 'case.png' };
    vertebrae = [{ name: 'L1', points: [] }];
    dragInfo = { vertebraIdx: 0, pointIdx: 0 };
    selectedVertebrae.add(0);
    clearAll();
  `);
  assert.deepEqual(page.snapshot('({ filename: currentImageMeta.filename, hasImage: Boolean(originalImage), vertebrae, dragInfo, selectedCount: selectedVertebrae.size })'), {
    filename: 'case.png',
    hasImage: true,
    vertebrae: [],
    dragInfo: null,
    selectedCount: 0,
  });
});

test('choosing the same image file twice is possible because the input value is cleared', () => {
  const page = loadHtmlScript(pagePath);
  page.initialize();
  const event = { target: { files: [{ name: 'case.png' }], value: 'C:\\fakepath\\case.png' } };
  page.context.handleFileSelect(event);
  assert.equal(event.target.value, '');
});

test('unsaved annotation guard can cancel replacing the current case', () => {
  const page = loadHtmlScript(pagePath, { confirm: () => false });
  page.evaluate('annotationDirty = true');
  assert.equal(page.evaluate('confirmDiscardUnsavedChanges()'), false);
  assert.equal(page.evaluate('annotationDirty'), true);
});

test('committing a decoded image atomically clears the old case and records image metadata', () => {
  const page = loadHtmlScript(pagePath);
  page.initialize();
  page.evaluate(`
    vertebrae = [{ name: 'L1', points: [] }];
    currentPoints = [{ x: 4, y: 5 }];
    selectedVertebrae.add(0);
    annotationDirty = true;
    commitLoadedImage(
      { width: 800, height: 1000, naturalWidth: 800, naturalHeight: 1000 },
      { name: 'new-case.png', size: 1234, lastModified: 42 },
      'abc123'
    );
  `);
  assert.deepEqual(page.snapshot('({ vertebrae, currentPoints, selectedCount: selectedVertebrae.size, annotationDirty, currentImageMeta, canvasWidth: canvas.width, canvasHeight: canvas.height })'), {
    vertebrae: [],
    currentPoints: [],
    selectedCount: 0,
    annotationDirty: false,
    currentImageMeta: {
      filename: 'new-case.png',
      width: 800,
      height: 1000,
      size: 1234,
      lastModified: 42,
      sha256: 'abc123',
    },
    canvasWidth: 800,
    canvasHeight: 1000,
  });
});

function matchingAnnotation(overrides = {}) {
  const { imageInfo = {}, ...rest } = overrides;
  return {
    version: '2.3',
    spineType: 'L',
    imageInfo: {
      filename: 'case.png',
      width: 800,
      height: 1000,
      sha256: 'abc123',
      ...imageInfo,
    },
    vertebrae: [{
      name: 'S1',
      boundaryType: 'upper',
      points: {
        anteriorSuperior: { x: 10, y: 20 },
        middleSuperior: { x: 20, y: 20 },
        posteriorSuperior: { x: 30, y: 20 },
      },
    }],
    ...rest,
  };
}

function loadCase(page) {
  page.initialize();
  page.evaluate(`commitLoadedImage(
    { width: 800, height: 1000, naturalWidth: 800, naturalHeight: 1000 },
    { name: 'case.png', size: 1234, lastModified: 42 },
    'abc123'
  )`);
}

test('matching filename, dimensions, and hash pass strict annotation identity validation', () => {
  const page = loadHtmlScript(pagePath);
  loadCase(page);
  page.context.__annotation = matchingAnnotation();
  const result = page.snapshot("validateAnnotationForCurrentImage(__annotation, 'case.json')");
  assert.equal(result.ok, true);
  assert.deepEqual(result.errors, []);
});

test('dimension mismatch rejects import without mutating the current annotations', () => {
  const page = loadHtmlScript(pagePath);
  loadCase(page);
  page.evaluate("vertebrae = [{ name: 'OLD', points: [] }]");
  page.context.__annotation = matchingAnnotation({ imageInfo: { height: 999 } });
  assert.equal(page.evaluate("importAnnotationJson(__annotation, { jsonFilename: 'case.json' })"), false);
  assert.deepEqual(page.snapshot('vertebrae.map(v => v.name)'), ['OLD']);
});

test('legacy JSON is accepted only with exact stem and exact dimensions', () => {
  const page = loadHtmlScript(pagePath);
  loadCase(page);
  page.context.__legacy = matchingAnnotation({ imageInfo: { filename: undefined, sha256: undefined } });

  const matching = page.snapshot("validateAnnotationForCurrentImage(__legacy, 'case.json')");
  const mismatching = page.snapshot("validateAnnotationForCurrentImage(__legacy, 'other.json')");
  assert.equal(matching.ok, true);
  assert.ok(matching.warnings.some(message => message.includes('舊版')));
  assert.equal(mismatching.ok, false);
});

test('export uses the image stem and includes image identity metadata', async () => {
  const page = loadHtmlScript(pagePath);
  loadCase(page);
  page.context.__annotation = matchingAnnotation();
  assert.equal(page.evaluate("importAnnotationJson(__annotation, { jsonFilename: 'case.json', silent: true })"), true);
  page.evaluate('annotationDirty = true; exportData()');

  assert.equal(page.downloads.length, 1);
  assert.equal(page.downloads[0].download, 'case.json');
  const exported = JSON.parse(await page.downloads[0].blob.text());
  assert.deepEqual(exported.imageInfo, {
    filename: 'case.png',
    width: 800,
    height: 1000,
    size: 1234,
    lastModified: 42,
    sha256: 'abc123',
  });
  assert.equal(page.evaluate('annotationDirty'), false);
});

test('placing an annotation point marks the case as unsaved', () => {
  const page = loadHtmlScript(pagePath);
  loadCase(page);
  page.evaluate('annotationDirty = false');
  page.context.handleCanvasClick({ clientX: 100, clientY: 100, shiftKey: false });
  assert.equal(page.evaluate('annotationDirty'), true);
  assert.equal(page.evaluate('currentPoints.length'), 1);
});

test('SpineFM dimension mismatch is rejected and kept pending for the correct image', () => {
  const page = loadHtmlScript(pagePath);
  loadCase(page);
  page.evaluate("vertebrae = [{ name: 'OLD', points: [] }]; window._spinefmJsonName = 'case.json'");
  page.context.__annotation = matchingAnnotation({ imageInfo: { height: 999 } });
  assert.equal(page.evaluate('_applySpineFMData(__annotation)'), false);
  assert.deepEqual(page.snapshot('vertebrae.map(v => v.name)'), ['OLD']);
  assert.equal(page.evaluate('window._pendingSpineFM === __annotation'), true);
});

test('JSON import cannot overwrite unsaved edits when the doctor cancels', () => {
  const page = loadHtmlScript(pagePath, { confirm: () => false });
  loadCase(page);
  page.evaluate("vertebrae = [{ name: 'UNSAVED', points: [] }]; annotationDirty = true");
  page.context.__annotation = matchingAnnotation();
  assert.equal(page.evaluate("importAnnotationJson(__annotation, { jsonFilename: 'case.json', silent: true })"), false);
  assert.deepEqual(page.snapshot('vertebrae.map(v => v.name)'), ['UNSAVED']);
  assert.equal(page.evaluate('annotationDirty'), true);
});

test('changing spine type clears stale selections and drag state with the annotations', () => {
  const page = loadHtmlScript(pagePath);
  loadCase(page);
  page.evaluate(`
    vertebrae = [{ name: 'L1', points: [] }];
    selectedVertebrae.add(0);
    dragInfo = { vertebraIdx: 0, pointIdx: 0 };
    setSpineType('C');
  `);
  assert.deepEqual(page.snapshot('({ spineType, vertebrae, selectedCount: selectedVertebrae.size, dragInfo })'), {
    spineType: 'C',
    vertebrae: [],
    selectedCount: 0,
    dragInfo: null,
  });
});
