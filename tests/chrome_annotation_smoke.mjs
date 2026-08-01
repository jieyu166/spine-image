import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { spawn } from 'node:child_process';

const [chromePath, htmlPath, firstImage, secondImage, downloadPath] = process.argv.slice(2);
if (![chromePath, htmlPath, firstImage, secondImage, downloadPath].every(Boolean)) {
  throw new Error('Usage: node chrome_annotation_smoke.mjs <chrome> <html> <image-1> <image-2> <download-dir>');
}

const port = 9338;
fs.mkdirSync(downloadPath, { recursive: true });
const chrome = spawn(chromePath, [
  '--headless=new', '--disable-gpu', `--remote-debugging-port=${port}`,
  `--user-data-dir=${path.join(downloadPath, 'chrome-profile')}`, 'about:blank'
], { windowsHide: true, stdio: 'ignore' });
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

async function poll(fn, timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const value = await fn();
      if (value) return value;
    } catch (error) { lastError = error; }
    await delay(100);
  }
  throw lastError || new Error('Timed out');
}

let socket;
try {
  const targets = await poll(async () => {
    const response = await fetch(`http://127.0.0.1:${port}/json/list`);
    const items = await response.json();
    return items.length ? items : null;
  });
  socket = new WebSocket((targets.find(item => item.type === 'page') || targets[0]).webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });
  let nextId = 0;
  const pending = new Map();
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (!message.id || !pending.has(message.id)) return;
    const item = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) item.reject(new Error(message.error.message));
    else item.resolve(message.result);
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++nextId;
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async expression => {
    const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
    return result.result.value;
  };
  const setInputFile = async (selector, filePath) => {
    const documentNode = await send('DOM.getDocument', { depth: -1, pierce: true });
    const input = await send('DOM.querySelector', { nodeId: documentNode.root.nodeId, selector });
    assert.ok(input.nodeId, `${selector} must exist`);
    await send('DOM.setFileInputFiles', { nodeId: input.nodeId, files: [path.resolve(filePath)] });
    await evaluate(`document.querySelector(${JSON.stringify(selector)}).dispatchEvent(new Event('change', { bubbles: true })); true`);
  };

  await send('Page.enable');
  await send('DOM.enable');
  await send('Runtime.enable');
  await send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath });
  await send('Page.navigate', { url: pathToFileURL(path.resolve(htmlPath)).href });
  await poll(() => evaluate("document.readyState === 'complete'"));
  await evaluate("window.__alerts = []; window.alert = message => __alerts.push(String(message)); window.confirm = () => true; true");

  await setInputFile('#fileInput', firstImage);
  const firstMeta = await poll(() => evaluate(`currentImageMeta?.sha256?.length === 64 && ({
    filename: currentImageMeta.filename,
    width: currentImageMeta.width,
    height: currentImageMeta.height,
    sha256: currentImageMeta.sha256,
    points: currentPoints.length,
    vertebrae: vertebrae.length
  })`));
  assert.equal(firstMeta.filename, path.basename(firstImage));
  assert.equal(firstMeta.points, 0);
  assert.equal(firstMeta.vertebrae, 0);

  await evaluate(`
    vertebrae = [{
      name: 'S1', boundaryType: 'upper', hasMiddlePoints: true,
      points: [{x:100,y:100},{x:150,y:105},{x:200,y:110}],
      anteriorHeight: null, middleHeight: null, posteriorHeight: null
    }];
    annotationDirty = true;
    exportData();
    true
  `);
  const firstStem = path.basename(firstImage).replace(/\.[^.]+$/, '');
  const exportedPath = path.join(downloadPath, `${firstStem}.json`);
  await poll(() => fs.existsSync(exportedPath));
  const exported = JSON.parse(fs.readFileSync(exportedPath, 'utf8'));
  assert.equal(exported.imageInfo.filename, path.basename(firstImage));
  assert.equal(exported.imageInfo.width, firstMeta.width);
  assert.equal(exported.imageInfo.height, firstMeta.height);
  assert.equal(exported.imageInfo.sha256.length, 64);

  const importedLiteral = JSON.stringify(exported);
  const roundTrip = await evaluate(`importAnnotationJson(${importedLiteral}, { jsonFilename: ${JSON.stringify(`${firstStem}.json`)}, silent: true })`);
  assert.equal(roundTrip, true);
  assert.equal(await evaluate('vertebrae.length'), 1);

  await setInputFile('#fileInput', secondImage);
  const secondMeta = await poll(() => evaluate(`currentImageMeta?.filename === ${JSON.stringify(path.basename(secondImage))} && ({
    filename: currentImageMeta.filename,
    width: currentImageMeta.width,
    height: currentImageMeta.height,
    points: currentPoints.length,
    vertebrae: vertebrae.length,
    selected: selectedVertebrae.size,
    dirty: annotationDirty
  })`));
  assert.equal(secondMeta.points, 0);
  assert.equal(secondMeta.vertebrae, 0);
  assert.equal(secondMeta.selected, 0);
  assert.equal(secondMeta.dirty, false);

  await evaluate("currentPoints = [{x:9,y:9}]; annotationDirty = false; true");
  const rejected = await evaluate(`importAnnotationJson(${importedLiteral}, { jsonFilename: ${JSON.stringify(`${firstStem}.json`)}, silent: true })`);
  assert.equal(rejected, false);
  assert.equal(await evaluate('currentPoints[0].x'), 9);
  assert.ok((await evaluate('window.__alerts')).some(message => message.includes('拒絕載入標註')));

  process.stdout.write(JSON.stringify({ firstMeta, secondMeta, exportedPath, roundTrip, rejected }, null, 2));
} finally {
  socket?.close();
  chrome.kill();
}
