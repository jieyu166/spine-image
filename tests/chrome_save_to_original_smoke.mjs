// 「存回原檔」在真實 Chrome 的 File System Access 介面上實跑。
// 真實資料夾需要醫師在選擇器按允許，自動化做不到；改用 Origin Private File System（OPFS）——
// 它提供與 showDirectoryPicker 相同的 FileSystemDirectoryHandle 介面，可驗證寫入、走訪、備份流程。
// OPFS 在 file:// 會被擋，所以這裡由本機 http 提供頁面（只綁 127.0.0.1）。
//
// 用法：node tests/chrome_save_to_original_smoke.mjs <chrome.exe> <image.png> <annotation.json> <work-dir>
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const [chromePath, imagePath, jsonPath, workDir] = process.argv.slice(2);
if (![chromePath, imagePath, jsonPath, workDir].every(Boolean)) {
  throw new Error('Usage: node chrome_save_to_original_smoke.mjs <chrome> <image.png> <annotation.json> <work-dir>');
}
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const httpPort = 8766;
const cdpPort = 9340;
fs.mkdirSync(workDir, { recursive: true });
const server = spawn('python', ['-m', 'http.server', String(httpPort), '--bind', '127.0.0.1'], { cwd: root, stdio: 'ignore', windowsHide: true });
const chrome = spawn(chromePath, ['--headless=new', '--disable-gpu', `--remote-debugging-port=${cdpPort}`,
  `--user-data-dir=${path.join(workDir, 'chrome-profile')}`, 'about:blank'], { stdio: 'ignore', windowsHide: true });
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function poll(fn, timeoutMs = 20000, what = 'condition') {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try { const result = await fn(); if (result) return result; } catch (error) { lastError = error; }
    await delay(150);
  }
  throw lastError || new Error(`Timed out waiting for ${what}`);
}

let socket;
try {
  const target = await poll(async () => (await (await fetch(`http://127.0.0.1:${cdpPort}/json/list`)).json()).find(t => t.type === 'page'), 20000, 'chrome');
  await poll(async () => (await fetch(`http://127.0.0.1:${httpPort}/spinal-annotation-web.html`)).ok, 20000, 'http server');
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.addEventListener('open', resolve, { once: true }); socket.addEventListener('error', reject, { once: true }); });
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
    message.error ? reject(new Error(message.error.message)) : resolve(message.result);
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++nextId;
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async expression => {
    const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    return result.result.value;
  };
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Page.navigate', { url: `http://127.0.0.1:${httpPort}/spinal-annotation-web.html` });
  await poll(() => evaluate("document.readyState === 'complete' && typeof saveToOriginal === 'function'"), 20000, 'editor');

  const rel = path.relative(root, imagePath).split(path.sep).join('/');
  const relJson = path.relative(root, jsonPath).split(path.sep).join('/');
  const jsonName = path.basename(jsonPath);
  const sub = path.dirname(relJson).split('/').slice(1);   // Images/202609/x.json → ['202609']

  const result = await evaluate(`(async () => {
    // OPFS 裡建一個 Images/<子資料夾>/<x>.json，內容是真實原檔
    const opfs = await navigator.storage.getDirectory();
    for await (const entry of opfs.values()) await opfs.removeEntry(entry.name, { recursive: true });
    const images = await opfs.getDirectoryHandle('Images', { create: true });
    let dir = images;
    for (const part of ${JSON.stringify(sub)}) dir = await dir.getDirectoryHandle(part, { create: true });
    const originalText = await (await fetch(${JSON.stringify(relJson)})).text();
    const fh = await dir.getFileHandle(${JSON.stringify(jsonName)}, { create: true });
    const w = await fh.createWritable(); await w.write(originalText); await w.close();

    // 模擬 viewer 送來的案例（真實影像與 JSON）
    const blob = await (await fetch(${JSON.stringify(rel)})).blob();
    const imageFile = new File([blob], ${JSON.stringify(path.basename(imagePath))}, { type: 'image/png' });
    const ok = await handleViewerMessage({ data: { type: 'spine-viewer-case', imageFile, jsonText: originalText,
      jsonName: ${JSON.stringify(jsonName)}, sourceImagePath: ${JSON.stringify(rel)}, sourceJsonPath: ${JSON.stringify(relJson)} }, source: null });

    imagesDirHandle = images;
    const v = vertebrae.find(item => item.name !== 'S1' && item.points.length === 6);
    v.points[0].x += 40; annotationDirty = true;
    const saved = await saveToOriginal();
    const after = JSON.parse(await (await fh.getFile()).text());
    const backups = await images.getDirectoryHandle('_json_backups');
    const names = []; for await (const entry of backups.values()) names.push(entry.name);
    const backupText = await (await (await backups.getFileHandle(names[0])).getFile()).text();
    const again = await saveToOriginal();
    return { ok, saved, again: again.unchanged, name: v.name, newX: v.points[0].x,
      writtenX: after.vertebrae.find(item => item.name === v.name).points.anteriorSuperior.x,
      backupNames: names, backupMatchesOriginal: backupText === originalText, dirty: annotationDirty,
      task: document.getElementById('taskText').textContent };
  })()`);

  assert.equal(result.ok, true);
  assert.equal(result.saved.path, relJson.split('/').slice(1).join('/'));
  assert.equal(result.writtenX, result.newX);
  assert.equal(result.backupNames.length, 1);
  assert.match(result.backupNames[0], /\.\d{8}-\d{6}\.bak$/);
  assert.equal(result.backupMatchesOriginal, true);
  assert.equal(result.again, true, 'second save with no change is a no-op');
  assert.equal(result.dirty, false);
  assert.equal(dialogs.length, 0, `unexpected dialogs: ${dialogs.join(' | ')}`);
  process.stdout.write(JSON.stringify(result, null, 2) + '\n');
} finally {
  socket?.close();
  chrome.kill();
  server.kill();
}
