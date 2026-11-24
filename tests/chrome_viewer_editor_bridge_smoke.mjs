// 端到端：annotation-viewer.html「送到標註工具修正」→ spinal-annotation-web.html → 匯出回傳 viewer。
// 以醫師實際使用的 file:// 開啟，headless Chrome 透過 DevTools 協定操作；需要真正的多分頁環境
// （內建預覽窗格會把 window.open 導到同一分頁，無法驗證）。
//
// 用法：node tests/chrome_viewer_editor_bridge_smoke.mjs <chrome.exe> <image.png> <annotation.json> <download-dir>
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawn } from 'node:child_process';

const [chromePath, imagePath, jsonPath, downloadPath] = process.argv.slice(2);
if (![chromePath, imagePath, jsonPath, downloadPath].every(Boolean)) {
  throw new Error('Usage: node chrome_viewer_editor_bridge_smoke.mjs <chrome> <image.png> <annotation.json> <download-dir>');
}
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const viewerUrl = pathToFileURL(path.join(root, 'annotation-viewer.html')).href;
const port = 9338;
fs.mkdirSync(downloadPath, { recursive: true });
const chrome = spawn(chromePath, [
  '--headless=new', '--disable-gpu', '--window-size=1600,1000',
  `--remote-debugging-port=${port}`, `--user-data-dir=${path.join(downloadPath, 'chrome-profile')}`,
  '--allow-file-access-from-files', 'about:blank',
], { windowsHide: true, stdio: 'ignore' });

const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function poll(fn, timeoutMs = 20000, what = 'condition') {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const result = await fn();
      if (result) return result;
    } catch (error) {
      lastError = error;
    }
    await delay(150);
  }
  throw lastError || new Error(`Timed out waiting for ${what}`);
}

async function connect(target) {
  const socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });
  let nextId = 0;
  const pending = new Map();
  const dialogs = [];
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (message.method === 'Page.javascriptDialogOpening') {
      dialogs.push(message.params.message);
      send('Page.handleJavaScriptDialog', { accept: true }).catch(() => {});
      return;
    }
    if (!message.id || !pending.has(message.id)) return;
    const { resolve, reject } = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) reject(new Error(message.error.message));
    else resolve(message.result);
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++nextId;
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async (expression, userGesture = false) => {
    const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, userGesture });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    return result.result.value;
  };
  await send('Page.enable');
  await send('Runtime.enable');
  await send('DOM.enable');
  return { socket, send, evaluate, dialogs };
}

const listTargets = async () => (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
const sockets = [];
try {
  const first = await poll(async () => (await listTargets()).find(t => t.type === 'page'), 20000, 'chrome');
  const viewer = await connect(first);
  sockets.push(viewer.socket);
  await viewer.send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: path.resolve(downloadPath) });
  await viewer.send('Page.navigate', { url: viewerUrl });
  await poll(() => viewer.evaluate("document.readyState === 'complete' && typeof sendToAnnotationEditor === 'function'"), 20000, 'viewer');

  // 1. 醫師在 viewer 用檔案選擇器載入影像與 JSON
  const doc = await viewer.send('DOM.getDocument', { depth: -1 });
  for (const [selector, file] of [['#imageInput', imagePath], ['#jsonInput', jsonPath]]) {
    const node = await viewer.send('DOM.querySelector', { nodeId: doc.root.nodeId, selector });
    await viewer.send('DOM.setFileInputFiles', { nodeId: node.nodeId, files: [path.resolve(file)] });
    await viewer.evaluate(`document.querySelector('${selector}').dispatchEvent(new Event('change', { bubbles: true }))`);
  }
  const loaded = await poll(() => viewer.evaluate(`originalImage && annotationData && currentSource.imageFile && currentSource.jsonFile && ({
    image: [originalImage.width, originalImage.height], vertebrae: annotationData.vertebrae.length,
    imagePath: currentSource.imagePath, jsonPath: currentSource.jsonPath })`), 20000, 'viewer load');

  // 2. 按「送到標註工具修正」（帶使用者手勢，與實際點擊相同）
  assert.equal(await viewer.evaluate('sendToAnnotationEditor()', true), true);
  const editorTarget = await poll(async () => (await listTargets()).find(t => t.type === 'page' && t.url.includes('spinal-annotation-web.html')), 20000, 'editor tab');
  assert.ok(editorTarget.url.endsWith('?import=viewer'));
  const editor = await connect(editorTarget);
  sockets.push(editor.socket);

  const received = await poll(() => editor.evaluate(`originalImage && vertebrae.length && viewerLink && ({
    filename: currentImageMeta.filename, size: [currentImageMeta.width, currentImageMeta.height],
    sha256: currentImageMeta.sha256, vertebrae: vertebrae.length, names: vertebrae.map(v => v.name),
    sourceImagePath: viewerLink.sourceImagePath, title: document.title,
    task: document.getElementById('taskText').textContent })`), 20000, 'editor import');
  const original = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
  assert.equal(received.filename, path.basename(imagePath));
  assert.deepEqual(received.size, loaded.image);
  assert.equal(received.vertebrae, original.vertebrae.length);
  assert.match(received.sha256, /^[0-9a-f]{64}$/);
  if (original.imageInfo?.sha256) assert.equal(received.sha256, original.imageInfo.sha256);
  assert.equal(editor.dialogs.length, 0, `unexpected editor dialogs: ${editor.dialogs.join(' | ')}`);

  // 3. 醫師在標註工具把 L3 前上角往右移 40 px 後匯出
  const target = original.vertebrae.find(v => v.points?.anteriorSuperior && v.name !== 'S1');
  const moved = await editor.evaluate(`(() => {
    const v = vertebrae.find(item => item.name === ${JSON.stringify(target.name)});
    v.points[0].x += 40; annotationDirty = true; exportData();
    return { name: v.name, x: v.points[0].x };
  })()`);
  assert.equal(moved.x, target.points.anteriorSuperior.x + 40);
  const stem = path.basename(imagePath).replace(/\.[^.]+$/, '');
  await poll(() => fs.existsSync(path.join(downloadPath, `${stem}.json`)), 20000, 'download');

  // 4. viewer 自動收到修正版並重畫
  const corrected = await poll(() => viewer.evaluate(`annotationData._correctedFromEditor && ({
    x: annotationData.vertebrae.find(v => v.name === ${JSON.stringify(target.name)}).points.anteriorSuperior.x,
    fileName: annotationData._fileName, status: document.getElementById('statusText').textContent })`), 20000, 'viewer update');
  assert.equal(corrected.x, moved.x);
  assert.match(corrected.status, /覆蓋/);
  const downloaded = JSON.parse(fs.readFileSync(path.join(downloadPath, `${stem}.json`), 'utf8'));
  assert.equal(downloaded.vertebrae.find(v => v.name === target.name).points.anteriorSuperior.x, moved.x);

  // 5. 再送一次：沿用同一個標註工具分頁，不會多開
  assert.equal(await viewer.evaluate('sendToAnnotationEditor()', true), true);
  await delay(1500);
  const editorTabs = (await listTargets()).filter(t => t.type === 'page' && t.url.includes('spinal-annotation-web.html'));
  assert.equal(editorTabs.length, 1);
  // 匯入後標註工具會標成「未存檔」（既有設計），這裡只核對內容換回原檔
  const reloaded = await poll(() => editor.evaluate(`(() => {
    const v = vertebrae.find(item => item.name === ${JSON.stringify(target.name)});
    return v && v.points[0].x === ${target.points.anteriorSuperior.x} && v.points[0].x;
  })()`), 20000, 'resend');
  assert.equal(reloaded, target.points.anteriorSuperior.x, 'the resent original replaces the edited copy');
  assert.equal(viewer.dialogs.length, 0, `unexpected viewer dialogs: ${viewer.dialogs.join(' | ')}`);

  process.stdout.write(JSON.stringify({ loaded, received: { ...received, sha256: received.sha256.slice(0, 12) + '…' }, moved, corrected, editorTabs: editorTabs.length }, null, 2) + '\n');
} finally {
  sockets.forEach(socket => socket.close());
  chrome.kill();
}
