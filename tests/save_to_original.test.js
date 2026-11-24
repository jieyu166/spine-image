'use strict';

// spinal-annotation-web.html「存回原檔」：以 File System Access API 覆蓋原 JSON，覆蓋前備份。
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const path = require('node:path');
const test = require('node:test');
const { loadHtmlScript } = require('./html_script_harness');

const editorPath = path.join(__dirname, '..', 'spinal-annotation-web.html');

// ── 模擬資料夾（FileSystemDirectoryHandle 的子集） ─────────────────────
class FakeDir {
  constructor(name) {
    this.kind = 'directory';
    this.name = name;
    this.files = new Map();
    this.dirs = new Map();
    this.permission = 'granted';
    this.requested = 0;
  }
  sub(name) {
    if (!this.dirs.has(name)) this.dirs.set(name, new FakeDir(name));
    return this.dirs.get(name);
  }
  async getDirectoryHandle(name, options = {}) {
    if (!this.dirs.has(name)) {
      if (!options.create) throw new Error(`NotFoundError: ${name}`);
      this.dirs.set(name, new FakeDir(name));
    }
    return this.dirs.get(name);
  }
  async getFileHandle(name, options = {}) {
    if (!this.files.has(name)) {
      if (!options.create) throw new Error(`NotFoundError: ${name}`);
      this.files.set(name, '');
    }
    return this.fileHandle(name);
  }
  fileHandle(name) {
    const dir = this;
    return {
      kind: 'file',
      name,
      async getFile() { return { text: async () => dir.files.get(name) }; },
      async createWritable() {
        let buffer = '';
        return { async write(text) { buffer += text; }, async close() { dir.files.set(name, buffer); } };
      },
    };
  }
  async *values() {
    for (const name of this.files.keys()) yield this.fileHandle(name);
    for (const dir of this.dirs.values()) yield dir;
  }
  async queryPermission() { return this.permission; }
  async requestPermission() {
    this.requested += 1;
    return this.permission === 'prompt' ? 'granted' : this.permission;
  }
}

class SizedImage {
  constructor() { this.width = 800; this.height = 1000; this.naturalWidth = 800; this.naturalHeight = 1000; }
  set src(value) { this._src = value; queueMicrotask(() => this.onload?.()); }
  get src() { return this._src; }
}

const bytes = Buffer.from('fake-png-bytes');
const sha = createHash('sha256').update(bytes).digest('hex');

function original() {
  return JSON.stringify({
    version: '2.3', spineType: 'L',
    imageInfo: { filename: 'case.png', width: 800, height: 1000, sha256: sha },
    vertebrae: [{
      name: 'S1', boundaryType: 'upper',
      points: { anteriorSuperior: { x: 10, y: 20 }, middleSuperior: { x: 20, y: 20 }, posteriorSuperior: { x: 30, y: 20 } },
    }],
  }, null, 2);
}

function imagesTree() {
  const root = new FakeDir('Images');
  root.sub('202609').files.set('case.json', original());
  root.sub('202609').files.set('case.png', '<png>');
  return root;
}

async function loadCase({ root = imagesTree(), relPath = 'Images/202609/case.json', picker = null } = {}) {
  const viewerWindow = { closed: false, messages: [], postMessage(data) { this.messages.push(data); } };
  const page = loadHtmlScript(editorPath, {
    location: { search: '?import=viewer' },
    opener: viewerWindow,
    Image: SizedImage,
    showDirectoryPicker: picker || (async () => root),
  });
  page.initialize();
  page.context.__msg = {
    type: 'spine-viewer-case',
    imageFile: new File([bytes], 'case.png', { type: 'image/png', lastModified: 42 }),
    imageName: 'case.png', jsonText: original(), jsonName: 'case.json',
    sourceImagePath: 'Images/202609/case.png', sourceJsonPath: relPath,
  };
  page.context.__src = viewerWindow;
  assert.equal(await page.evaluate('handleViewerMessage({ data: __msg, source: __src })'), true);
  return { page, root, viewerWindow };
}

function moveS1(page, dx = 40) {
  page.evaluate(`vertebrae.find(v => v.name === 'S1').points[0].x += ${dx}; annotationDirty = true;`);
}

test('saves over the exact JSON path from the viewer and backs up the old version first', async () => {
  const { page, root, viewerWindow } = await loadCase();
  page.context.__root = root;
  page.evaluate('imagesDirHandle = __root');
  moveS1(page);
  viewerWindow.messages.length = 0;

  const result = await page.evaluate('saveToOriginal()');
  assert.equal(result.path, '202609/case.json');
  assert.match(result.backup, /^_json_backups\/202609__case\.json\.\d{8}-\d{6}\.bak$/);

  const saved = JSON.parse(root.sub('202609').files.get('case.json'));
  assert.equal(saved.vertebrae[0].points.anteriorSuperior.x, 50);
  const backups = [...root.sub('_json_backups').files.entries()];
  assert.equal(backups.length, 1);
  assert.equal(backups[0][1], original(), 'backup holds the exact previous content');
  assert.equal(page.downloads.length, 0, 'nothing goes to Downloads');
  assert.equal(page.evaluate('annotationDirty'), false);
  assert.equal(viewerWindow.messages.length, 1);
  assert.equal(viewerWindow.messages[0].savedToOriginal, true);
  assert.equal(viewerWindow.messages[0].savedPath, '202609/case.json');
  assert.match(page.evaluate("document.getElementById('taskText').textContent"), /已存回 Images\/202609\/case\.json/);
});

test('saving unchanged content writes nothing and makes no backup', async () => {
  const { page, root } = await loadCase();
  page.context.__root = root;
  page.evaluate('imagesDirHandle = __root');
  moveS1(page);
  await page.evaluate('saveToOriginal()');
  // 第二次存檔的 exportDate 一定不同；內容沒變就不能重寫（真實 Chrome 抓到的 bug）
  await new Promise(resolve => setTimeout(resolve, 5));
  const second = await page.evaluate('saveToOriginal()');
  assert.equal(second.unchanged, true);
  assert.equal(root.sub('_json_backups').files.size, 1);
});

test('without a granted folder it asks once, grants via the picker, then saves', async () => {
  const root = imagesTree();
  const { page } = await loadCase({ root });
  assert.equal(page.evaluate('imagesDirHandle'), null);
  moveS1(page);
  const result = await page.evaluate('saveToOriginal()');
  assert.equal(result.path, '202609/case.json');
  assert.match(page.confirms.at(-1), /尚未授權 Images 資料夾/);
  assert.match(page.evaluate("document.getElementById('saveTargetInfo').textContent"), /已授權資料夾：Images/);
});

test('a standalone import is found by name, but duplicates and a stem mismatch are refused', async () => {
  const { page, root } = await loadCase({ relPath: null });
  page.context.__root = root;
  page.evaluate("imagesDirHandle = __root; currentJsonSource = { name: 'case.json', relPath: null }");
  root.sub('_json_backups').files.set('case.json', 'ignored');  // 備份資料夾不列入搜尋
  moveS1(page);
  assert.equal((await page.evaluate('saveToOriginal()')).path, '202609/case.json');

  root.sub('202610').files.set('case.json', original());
  moveS1(page);
  assert.equal(await page.evaluate('saveToOriginal()'), false);
  assert.match(page.alerts.at(-1), /2 個同名的 case\.json/);

  root.sub('202609').files.set('other.json', original());
  page.evaluate("currentJsonSource = { name: 'other.json', relPath: 'Images/202609/other.json' }");
  assert.equal(await page.evaluate('saveToOriginal()'), false);
  assert.match(page.alerts.at(-1), /主檔名不同/);
  assert.equal(root.sub('202609').files.get('other.json'), original());
});

test('a denied write permission saves nothing', async () => {
  const { page, root } = await loadCase();
  root.permission = 'denied';
  page.context.__root = root;
  page.evaluate('imagesDirHandle = __root');
  moveS1(page);
  assert.equal(await page.evaluate('saveToOriginal()'), false);
  assert.match(page.alerts.at(-1), /沒有取得寫入權限/);
  assert.equal(root.sub('202609').files.get('case.json'), original());
  assert.equal(root.dirs.has('_json_backups'), false);
});

test('a folder granted one level above Images still resolves by searching', async () => {
  const parent = new FakeDir('Spine');
  parent.dirs.set('Images', imagesTree());
  const { page } = await loadCase({ root: parent });
  page.context.__root = parent;
  page.evaluate('imagesDirHandle = __root');
  moveS1(page);
  assert.equal((await page.evaluate('saveToOriginal()')).path, 'Images/202609/case.json');
});

test('Ctrl+S saves to the original file', async () => {
  const { page, root } = await loadCase();
  page.context.__root = root;
  page.evaluate('imagesDirHandle = __root');
  moveS1(page);
  page.evaluate("handleKeyDown({ ctrlKey: true, key: 's', preventDefault() {} })");
  await new Promise(resolve => setTimeout(resolve, 20));
  assert.equal(JSON.parse(root.sub('202609').files.get('case.json')).vertebrae[0].points.anteriorSuperior.x, 50);
});

test('download export still works and is reported to the viewer as not saved in place', async () => {
  const { page, viewerWindow } = await loadCase();
  viewerWindow.messages.length = 0;
  page.evaluate('annotationDirty = true; exportData()');
  assert.equal(page.downloads.length, 1);
  assert.equal(viewerWindow.messages[0].savedToOriginal, false);
});

test('an unchanged AI pre-annotation is still written once so the physician review is recorded', async () => {
  const root = imagesTree();
  const ai = JSON.parse(original());
  ai.annotationReview = { method: 'AI visual annotation', physicianReviewed: false, note: 'L5/S1 需複核' };
  root.sub('202609').files.set('case.json', JSON.stringify(ai, null, 2));
  const { page } = await loadCase({ root });
  page.context.__msg.jsonText = JSON.stringify(ai);
  await page.evaluate('handleViewerMessage({ data: __msg, source: __src })');
  page.context.__root = root;
  page.evaluate('imagesDirHandle = __root');

  const first = await page.evaluate('saveToOriginal()');  // 沒有改任何點
  assert.equal(first.unchanged, false);
  const saved = JSON.parse(root.sub('202609').files.get('case.json'));
  assert.equal(saved.annotationReview.physicianReviewed, true);
  assert.equal(saved.annotationReview.method, 'AI visual annotation');
  assert.equal(saved.annotationReview.priorNote, 'L5/S1 需複核');
  assert.equal(saved.annotationReview.previouslyUnreviewed, true);
  assert.equal(root.sub('_json_backups').files.size, 1);

  await new Promise(resolve => setTimeout(resolve, 5));
  assert.equal((await page.evaluate('saveToOriginal()')).unchanged, true, 'once reviewed, an identical save is a no-op');
});

test('a brand-new annotation is saved as <stem>.json beside the image, without a backup', async () => {
  const root = imagesTree();
  const { page } = await loadCase({ root });
  root.sub('202609').files.delete('case.json');
  page.context.__root = root;
  page.evaluate('imagesDirHandle = __root; currentJsonSource = null; importedAnnotationReview = null');
  const result = await page.evaluate('saveToOriginal()');
  assert.equal(result.created, true);
  assert.equal(result.path, '202609/case.json');
  const saved = JSON.parse(root.sub('202609').files.get('case.json'));
  assert.equal(saved.annotationReview.method, 'physician annotation');
  assert.equal(saved.annotationReview.physicianReviewed, true);
  assert.equal(root.dirs.has('_json_backups'), false);
  assert.equal(page.evaluate('currentJsonSource.relPath'), 'Images/202609/case.json', 'later saves target the new file');
  await new Promise(resolve => setTimeout(resolve, 5));
  assert.equal((await page.evaluate('saveToOriginal()')).unchanged, true);
});

test('a new annotation asks before overwriting an existing JSON that was not loaded', async () => {
  const root = imagesTree();
  const { page } = await loadCase({ root });
  page.context.__root = root;
  page.evaluate('imagesDirHandle = __root; currentJsonSource = null');
  const result = await page.evaluate('saveToOriginal()');
  assert.match(page.confirms.at(-1), /case\.json 已經存在/);
  assert.equal(result.path, '202609/case.json');
  assert.equal(root.sub('_json_backups').files.size, 1);
});

test('downloaded exports also carry the physician review record', async () => {
  const { page } = await loadCase();
  page.evaluate('annotationDirty = true; exportData()');
  const exported = JSON.parse(await page.downloads[0].blob.text());
  assert.equal(exported.annotationReview.physicianReviewed, true);
  assert.equal(exported.annotationReview.reviewedWith, 'spinal-annotation-web.html');
});
