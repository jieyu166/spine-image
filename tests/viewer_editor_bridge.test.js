'use strict';

// annotation-viewer.html ⇄ spinal-annotation-web.html：「送到標註工具修正」與修正版回傳。
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const path = require('node:path');
const test = require('node:test');
const { loadHtmlScript } = require('./html_script_harness');

const viewerPath = path.join(__dirname, '..', 'annotation-viewer.html');
const editorPath = path.join(__dirname, '..', 'spinal-annotation-web.html');


function fakeWindow() {
  const messages = [];
  return {
    closed: false,
    focused: 0,
    messages,
    postMessage(data, target) { messages.push({ data, target }); },
    focus() { this.focused += 1; },
  };
}

function jsonFile(name, payload) {
  const text = JSON.stringify(payload);
  return { name, textContent: text, text: async () => text };
}

function annotation(extra = {}) {
  return {
    version: '2.3',
    spineType: 'L',
    imageInfo: { filename: 'case.png', width: 800, height: 1000, ...(extra.imageInfo || {}) },
    vertebrae: [{
      name: 'S1',
      boundaryType: 'upper',
      points: {
        anteriorSuperior: { x: 10, y: 20 },
        middleSuperior: { x: 20, y: 20 },
        posteriorSuperior: { x: 30, y: 20 },
      },
    }],
  };
}

// ── viewer 端 ───────────────────────────────────────────────────────
function loadViewer() {
  const opened = [];
  const editorWindow = fakeWindow();
  const page = loadHtmlScript(viewerPath, {
    open(url, name) {
      opened.push({ url, name });
      return page.context.__blockPopup ? null : editorWindow;
    },
  });
  page.initialize();
  return { page, opened, editorWindow };
}

function setViewerCase(page, imageName = 'Images/202609/case.png', json = annotation()) {
  page.context.__image = { name: 'case.png', size: 1234 };
  page.context.__json = jsonFile('case.json', json);
  page.context.__imagePath = imageName;
  page.evaluate(`setCurrentSource({
    imageFile: __image, imagePath: __imagePath,
    jsonFile: __json, jsonPath: 'Images/202609/case.json',
  }, true)`);
}

test('viewer opens the editor in a named tab and sends the case after the editor is ready', async () => {
  const { page, opened, editorWindow } = loadViewer();
  setViewerCase(page);

  assert.equal(await page.evaluate('sendToAnnotationEditor()'), true);
  assert.deepEqual(opened, [{ url: 'spinal-annotation-web.html?import=viewer', name: 'spinalAnnotationEditor' }]);
  assert.equal(editorWindow.messages.length, 0, 'waits for the ready handshake');

  page.context.__src = editorWindow;
  assert.equal(page.evaluate("handleEditorMessage({ data: { type: 'spine-editor-ready' }, source: __src })"), true);
  assert.equal(editorWindow.messages.length, 1);
  const sent = editorWindow.messages[0].data;
  assert.equal(sent.type, 'spine-viewer-case');
  assert.equal(sent.imageFile, page.context.__image, 'the original File object is passed, not a re-encoded copy');
  assert.deepEqual(JSON.parse(sent.jsonText), annotation());
  assert.equal(sent.jsonName, 'case.json');
  assert.equal(sent.sourceImagePath, 'Images/202609/case.png');
  assert.equal(sent.sourceJsonPath, 'Images/202609/case.json');
  // 重複 ready 不會再送一次
  assert.equal(page.evaluate("handleEditorMessage({ data: { type: 'spine-editor-ready' }, source: __src })"), false);
});

test('a second send reuses the open editor tab without reopening it', async () => {
  const { page, opened, editorWindow } = loadViewer();
  setViewerCase(page);
  await page.evaluate('sendToAnnotationEditor()');
  setViewerCase(page, 'Images/202609/other.png');
  await page.evaluate('sendToAnnotationEditor()');
  assert.equal(opened.length, 1);
  assert.equal(editorWindow.messages.length, 1);
  assert.equal(editorWindow.messages[0].data.sourceImagePath, 'Images/202609/other.png');
  assert.equal(editorWindow.focused, 1);
});

test('viewer refuses to send without an image, for DICOM, or when the popup is blocked', async () => {
  const { page, opened } = loadViewer();
  assert.equal(await page.evaluate('sendToAnnotationEditor()'), false);
  assert.match(page.alerts.at(-1), /請先載入影像/);

  page.context.__dcm = { name: 'case.dcm' };
  page.evaluate("setCurrentSource({ imageFile: __dcm, imagePath: 'Images/case.dcm' }, true)");
  assert.equal(await page.evaluate('sendToAnnotationEditor()'), false);
  assert.match(page.alerts.at(-1), /DICOM/);

  setViewerCase(page);
  page.context.__blockPopup = true;
  assert.equal(await page.evaluate('sendToAnnotationEditor()'), false);
  assert.match(page.alerts.at(-1), /彈出式視窗/);
  assert.equal(page.evaluate('pendingEditorPayload'), null);
  assert.equal(opened.length, 1);
});

test('viewer shows the corrected annotation only for the image it is displaying', async () => {
  const { page, editorWindow } = loadViewer();
  setViewerCase(page);
  await page.evaluate('sendToAnnotationEditor()');
  page.context.__src = editorWindow;

  const corrected = annotation();
  corrected.vertebrae.push({
    name: 'L5',
    points: {
      anteriorSuperior: { x: 10, y: 1 }, posteriorSuperior: { x: 30, y: 1 },
      posteriorInferior: { x: 30, y: 15 }, anteriorInferior: { x: 10, y: 15 },
    },
  });
  page.context.__msg = {
    type: 'spine-editor-exported', jsonText: JSON.stringify(corrected), filename: 'case.json',
    sourceImagePath: 'Images/202609/case.png', sourceJsonPath: 'Images/202609/case.json',
  };
  assert.equal(page.evaluate('handleEditorMessage({ data: __msg, source: __src })'), true);
  assert.equal(page.evaluate('annotationData._correctedFromEditor'), true);
  assert.equal(page.evaluate('annotationData.vertebrae.length'), 2);
  assert.match(page.evaluate("document.getElementById('statusText').textContent"), /覆蓋 Images\/202609\/case\.json/);

  page.context.__other = { ...page.context.__msg, sourceImagePath: 'Images/202609/someone_else.png' };
  assert.equal(page.evaluate('handleEditorMessage({ data: __other, source: __src })'), false);
  page.context.__stranger = { postMessage() {} };
  assert.equal(page.evaluate('handleEditorMessage({ data: __msg, source: __stranger })'), false);
});

// ── editor 端 ───────────────────────────────────────────────────────
class SizedImage {
  constructor() { this.width = 800; this.height = 1000; this.naturalWidth = 800; this.naturalHeight = 1000; }
  set src(value) { this._src = value; queueMicrotask(() => this.onload?.()); }
  get src() { return this._src; }
}

function loadEditor(search = '?import=viewer') {
  const viewerWindow = fakeWindow();
  const page = loadHtmlScript(editorPath, { location: { search }, opener: viewerWindow, Image: SizedImage });
  page.initialize();
  return { page, viewerWindow };
}

function pngFile() {
  const bytes = Buffer.from('fake-png-bytes');
  const file = new File([bytes], 'case.png', { type: 'image/png', lastModified: 42 });
  return { file, sha: createHash('sha256').update(bytes).digest('hex') };
}

test('editor announces it is ready to the viewer only when opened with ?import=viewer', () => {
  const { viewerWindow } = loadEditor();
  // 物件建立在 sandbox realm，用 JSON 比值避免跨 realm 原型差異
  assert.equal(JSON.stringify(viewerWindow.messages.map(m => m.data)), JSON.stringify([{ type: 'spine-editor-ready' }]));
  const plain = loadEditor('');
  assert.equal(plain.viewerWindow.messages.length, 0);
});

test('editor loads the viewer case with identity checks and returns the export to the viewer', async () => {
  const { page, viewerWindow } = loadEditor();
  const { file, sha } = pngFile();
  page.context.__msg = {
    type: 'spine-viewer-case', imageFile: file, imageName: 'case.png',
    jsonText: JSON.stringify(annotation({ imageInfo: { sha256: sha } })), jsonName: 'case.json',
    sourceImagePath: 'Images/202609/case.png', sourceJsonPath: 'Images/202609/case.json',
  };
  page.context.__src = viewerWindow;

  assert.equal(await page.evaluate('handleViewerMessage({ data: __msg, source: __src })'), true);
  assert.deepEqual(page.snapshot('currentImageMeta'), {
    filename: 'case.png', width: 800, height: 1000, size: file.size, lastModified: 42, sha256: sha,
    pixelSha256: null,   // 測試環境的 canvas 讀回全 0，常數影像不給像素雜湊

  });
  assert.equal(page.evaluate('vertebrae.length'), 1);
  assert.equal(page.evaluate('viewerLink.sourceJsonPath'), 'Images/202609/case.json');
  assert.match(page.evaluate("document.getElementById('taskText').textContent"), /已從 viewer 載入/);

  viewerWindow.messages.length = 0;
  page.evaluate('annotationDirty = true; exportData()');
  assert.equal(page.downloads.length, 1);
  assert.equal(viewerWindow.messages.length, 1);
  const back = viewerWindow.messages[0].data;
  assert.equal(back.type, 'spine-editor-exported');
  assert.equal(back.filename, 'case.json');
  assert.equal(back.sourceImagePath, 'Images/202609/case.png');
  assert.equal(JSON.parse(back.jsonText).imageInfo.sha256, sha);
});

test('editor accepts a re-saved image (file hash differs) when the JSON and image stems match', async () => {
  const { page, viewerWindow } = loadEditor();
  const { file } = pngFile();
  page.context.__msg = {
    type: 'spine-viewer-case', imageFile: file, imageName: 'case.png',
    jsonText: JSON.stringify(annotation({ imageInfo: { filename: 'image.png', sha256: 'deadbeef' } })), jsonName: 'case.json',
    sourceImagePath: 'Images/case.png',
  };
  page.context.__src = viewerWindow;
  assert.equal(await page.evaluate('handleViewerMessage({ data: __msg, source: __src })'), true);
  assert.equal(page.evaluate('vertebrae.length'), 1);
});

test('editor rejects a JSON whose pixels and filename do not match the sent image', async () => {
  const { page, viewerWindow } = loadEditor();
  const { file } = pngFile();
  page.context.__msg = {
    type: 'spine-viewer-case', imageFile: file, imageName: 'case.png',
    jsonText: JSON.stringify(annotation({ imageInfo: { filename: 'other.png', sha256: 'deadbeef', pixelSha256: 'beefdead' } })),
    jsonName: 'other.json', sourceImagePath: 'Images/case.png',
  };
  page.context.__src = viewerWindow;
  assert.equal(await page.evaluate('handleViewerMessage({ data: __msg, source: __src })'), false);
  assert.match(page.alerts.at(-1), /檔名不符/);
  assert.equal(page.evaluate('vertebrae.length'), 0);
  assert.match(page.evaluate("document.getElementById('taskText').textContent"), /標註未載入/);
});

test('editor ignores messages that are not from its opener or are malformed', async () => {
  const { page } = loadEditor();
  const { file } = pngFile();
  page.context.__msg = { type: 'spine-viewer-case', imageFile: file, sourceImagePath: 'Images/case.png' };
  page.context.__stranger = fakeWindow();
  assert.equal(await page.evaluate('handleViewerMessage({ data: __msg, source: __stranger })'), false);
  assert.equal(await page.evaluate("handleViewerMessage({ data: { type: 'spine-viewer-case', imageFile: 'nope' }, source: window.opener })"), false);
  assert.equal(await page.evaluate("handleViewerMessage({ data: 'hello', source: window.opener })"), false);
  assert.equal(page.evaluate('originalImage'), null);
});

test('without a viewer link, exporting does not try to message anyone', () => {
  const { page, viewerWindow } = loadEditor('');
  assert.equal(page.evaluate("notifyViewerOfExport({}, 'x.json')"), false);
  assert.equal(viewerWindow.messages.length, 0);
});

test('viewer batch scanner ignores the _json_backups .bak files written by save-to-original', () => {
  const page = loadHtmlScript(viewerPath);
  const file = (p) => ({ name: p.split('/').pop(), webkitRelativePath: p, size: 10, lastModified: 1 });
  page.context.__files = [
    file('Images/202609/case.png'), file('Images/202609/case.json'),
    file('Images/_json_backups/202609__case.json.20261029-110025.bak'),
  ];
  const scan = page.snapshot('PairScanner.scan(__files, [])');
  assert.equal(scan.pairs.length, 1);
  assert.equal(scan.unmatchedJson.length, 0);
  assert.equal(scan.unmatchedImages.length, 0);
});
