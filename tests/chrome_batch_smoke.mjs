import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { spawn } from 'node:child_process';

const [chromePath, htmlPath, imagesPath, screenshotPath, downloadPath] = process.argv.slice(2);
if (![chromePath, htmlPath, imagesPath, screenshotPath, downloadPath].every(Boolean)) {
  throw new Error('Usage: node chrome_batch_smoke.mjs <chrome> <html> <Images> <screenshot> <download-dir>');
}

const port = 9337;
fs.mkdirSync(downloadPath, { recursive: true });
const profilePath = path.join(downloadPath, 'chrome-profile');
const chrome = spawn(chromePath, [
  '--headless=new',
  '--disable-gpu',
  '--hide-scrollbars',
  '--window-size=1800,1200',
  `--remote-debugging-port=${port}`,
  `--user-data-dir=${profilePath}`,
  'about:blank',
], { windowsHide: true, stdio: 'ignore' });

const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

async function poll(fn, timeoutMs = 15000) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const result = await fn();
      if (result) return result;
    } catch (error) {
      lastError = error;
    }
    await delay(100);
  }
  throw lastError || new Error('Timed out waiting for Chrome');
}

let socket;
try {
  const targets = await poll(async () => {
    const response = await fetch(`http://127.0.0.1:${port}/json/list`);
    const items = await response.json();
    return items.length ? items : null;
  });
  const target = targets.find(item => item.type === 'page') || targets[0];
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });

  let nextId = 0;
  const pending = new Map();
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
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
  const evaluate = async expression => {
    const result = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
    return result.result.value;
  };

  await send('Page.enable');
  await send('DOM.enable');
  await send('Runtime.enable');
  await send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath });
  await send('Page.navigate', { url: pathToFileURL(path.resolve(htmlPath)).href });
  await poll(() => evaluate("document.readyState === 'complete'"));

  const documentNode = await send('DOM.getDocument', { depth: -1, pierce: true });
  const inputNode = await send('DOM.querySelector', {
    nodeId: documentNode.root.nodeId,
    selector: '#reviewFolderInput',
  });
  assert.ok(inputNode.nodeId, 'reviewFolderInput must exist');
  await send('DOM.setFileInputFiles', { nodeId: inputNode.nodeId, files: [path.resolve(imagesPath)] });
  await evaluate("document.getElementById('reviewFolderInput').dispatchEvent(new Event('change', { bubbles: true }))");

  const scan = await poll(() => evaluate(`batchReviewController.scan && ({
    pairs: batchReviewController.scan.pairs.length,
    excluded: batchReviewController.scan.excluded.length,
    unmatchedImages: batchReviewController.scan.unmatchedImages.length,
    unmatchedJson: batchReviewController.scan.unmatchedJson.length,
    conflicts: batchReviewController.scan.conflicts.length,
    errors: batchReviewController.scan.errors.length
  })`), 30000);
  assert.equal(scan.pairs, 74);
  assert.equal(scan.excluded, 15);
  assert.equal(scan.unmatchedImages, 0);
  assert.equal(scan.unmatchedJson, 0);
  assert.equal(scan.conflicts, 0);
  assert.equal(scan.errors, 0);

  const loaded = await poll(() => evaluate(`originalImage && annotationData && ({
    imageWidth: originalImage.width,
    imageHeight: originalImage.height,
    jsonName: annotationData._fileName,
    scaleX,
    scaleY
  })`), 30000);
  assert.ok(loaded.imageWidth > 0 && loaded.imageHeight > 0);
  assert.equal(loaded.scaleX, 1);
  assert.equal(loaded.scaleY, 1);

  await evaluate("markBatchMatch(); true");
  await poll(() => evaluate("batchReviewController.currentIndex === 1 && annotationData?._fileName === 'Images/12056134.json'"));
  await evaluate(`
    document.getElementById('mismatchReason').value = 'image_patient_mismatch';
    document.getElementById('mismatchNote').value = '病人錯配，請確認';
    markBatchMismatch();
    true
  `);
  const reviewCount = await poll(() => evaluate('Object.keys(batchReviewController.manifest.reviews).length'));
  assert.equal(reviewCount, 2);

  await send('Page.reload', { ignoreCache: true });
  await poll(() => evaluate("document.readyState === 'complete'"));
  const reloadedDocument = await send('DOM.getDocument', { depth: -1, pierce: true });
  const reloadedInput = await send('DOM.querySelector', {
    nodeId: reloadedDocument.root.nodeId,
    selector: '#reviewFolderInput',
  });
  await send('DOM.setFileInputFiles', { nodeId: reloadedInput.nodeId, files: [path.resolve(imagesPath)] });
  await evaluate("document.getElementById('reviewFolderInput').dispatchEvent(new Event('change', { bubbles: true })); true");
  const resumedCount = await poll(() => evaluate('batchReviewController.manifest && Object.keys(batchReviewController.manifest.reviews).length'));
  assert.equal(resumedCount, 2);

  const dicomIndex = await evaluate("batchReviewController.scan.pairs.findIndex(pair => pair.imagePath.toLowerCase().endsWith('.dcm'))");
  assert.ok(dicomIndex >= 0, 'inventory must include a DICOM review pair');
  await evaluate(`batchReviewController.currentIndex = ${dicomIndex}; batchReviewController.loadCurrentPair(); true`);
  const dicomLoaded = await poll(() => evaluate(`originalImage?._dicomCanvas && annotationData && ({
    imageWidth: originalImage.width,
    imageHeight: originalImage.height,
    jsonName: annotationData._fileName,
    scaleX,
    scaleY
  })`), 30000);
  assert.ok(dicomLoaded.imageWidth > 0 && dicomLoaded.imageHeight > 0);
  assert.ok(dicomLoaded.jsonName.toLowerCase().endsWith('.json'));
  assert.equal(dicomLoaded.scaleX, 1);
  assert.equal(dicomLoaded.scaleY, 1);

  await evaluate('exportPairReviewJson(); exportPairReviewCsv(); true');
  await poll(() => fs.existsSync(path.join(downloadPath, 'pair_review.json')) && fs.existsSync(path.join(downloadPath, 'pair_review.csv')));
  const manifest = JSON.parse(fs.readFileSync(path.join(downloadPath, 'pair_review.json'), 'utf8'));
  assert.equal(manifest.version, '1.0');
  assert.equal(Object.keys(manifest.reviews).length, 2);
  assert.equal(manifest.summary.match, 1);
  assert.equal(manifest.summary.mismatch, 1);
  assert.ok(Object.values(manifest.reviews).some(review => review.note === '病人錯配，請確認'));
  const csv = fs.readFileSync(path.join(downloadPath, 'pair_review.csv'));
  assert.deepEqual([...csv.subarray(0, 3)], [0xef, 0xbb, 0xbf]);
  assert.ok(csv.toString('utf8').includes('病人錯配，請確認'));

  const screenshot = await send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: false });
  fs.writeFileSync(screenshotPath, Buffer.from(screenshot.data, 'base64'));

  process.stdout.write(JSON.stringify({ scan, loaded, dicomLoaded, reviewCount, resumedCount, screenshotPath, downloadPath }, null, 2));
} finally {
  socket?.close();
  chrome.kill();
}
