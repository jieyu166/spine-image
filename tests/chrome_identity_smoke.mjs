// 影像與 JSON 配對：真實 Chrome、file:// 開啟標註工具，重現醫院的建檔流程。
//
//   A. 一般載入：檔案選擇框選影像與 JSON，必須成功匯入。
//   B. 剪貼簿貼上（檔名 image.png）→ 匯出 JSON → 另存（重新編碼、改名為 <主檔名>.png）→ 載入另存檔＋匯出的 JSON：
//      檔案 SHA-256 與記錄的檔名都變了，必須靠像素雜湊直接通過，不跳任何確認。
//   C. 另存檔＋把 JSON 的 pixelSha256 拿掉（模擬本次修改前匯出的 JSON）：必須靠同主檔名通過。
//
// 用法：node tests/chrome_identity_smoke.mjs <chrome.exe> <work-dir> <a.png> <a.json> [<b.png> <b.json> ...]
// 結果寫到 <work-dir>/identity_result.json（含每張影像在瀏覽器算出的像素雜湊，供和 Python 比對）。
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawn } from 'node:child_process';

const [chromePath, workDir, ...pairArgs] = process.argv.slice(2);
if (!chromePath || !workDir || pairArgs.length < 2 || pairArgs.length % 2) {
  throw new Error('Usage: node chrome_identity_smoke.mjs <chrome> <work-dir> <a.png> <a.json> [...]');
}
const pairs = [];
for (let i = 0; i < pairArgs.length; i += 2) pairs.push({ png: pairArgs[i], json: pairArgs[i + 1] });
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const cdpPort = Number(process.env.CDP_PORT || 9242);   // 9305–9404 在這台 Windows 被保留
fs.mkdirSync(workDir, { recursive: true });
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function poll(fn, timeoutMs, what) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try { const r = await fn(); if (r) return r; } catch (e) { lastError = e; }
    await delay(200);
  }
  throw lastError || new Error(`Timed out waiting for ${what}`);
}

const chrome = spawn(chromePath, ['--headless=new', `--remote-debugging-port=${cdpPort}`, '--window-size=1400,1000',
  `--user-data-dir=${path.join(workDir, 'chrome-profile-identity')}`, 'about:blank'], { stdio: 'ignore', windowsHide: true });

let socket;
try {
  const target = await poll(async () => (await (await fetch(`http://127.0.0.1:${cdpPort}/json/list`)).json()).find(t => t.type === 'page'), 20000, 'chrome');
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.addEventListener('open', resolve, { once: true }); socket.addEventListener('error', reject, { once: true }); });
  let nextId = 0;
  const pending = new Map();
  let dialogs = [];
  socket.addEventListener('message', event => {
    const m = JSON.parse(event.data);
    if (m.method === 'Page.javascriptDialogOpening') {
      dialogs.push(m.params.message);
      send('Page.handleJavaScriptDialog', { accept: true }).catch(() => {});
      return;
    }
    if (!m.id || !pending.has(m.id)) return;
    const { resolve, reject } = pending.get(m.id);
    pending.delete(m.id);
    m.error ? reject(new Error(m.error.message)) : resolve(m.result);
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++nextId;
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async expression => {
    const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || r.exceptionDetails.text);
    return r.result.value;
  };
  const setFiles = async (selector, files) => {
    const { root: doc } = await send('DOM.getDocument', { depth: -1, pierce: true });
    const { nodeId } = await send('DOM.querySelector', { nodeId: doc.nodeId, selector });
    await send('DOM.setFileInputFiles', { nodeId, files: files.map(f => path.resolve(f)) });
  };
  const rejected = () => dialogs.filter(d => /拒絕|無法確認|像素內容/.test(d));
  await send('Page.enable');
  await send('Runtime.enable');
  await send('DOM.enable');

  const results = [];
  for (const [i, pair] of pairs.entries()) {
    const gt = JSON.parse(fs.readFileSync(pair.json, 'utf8'));
    const n = gt.vertebrae.length;
    const stem = path.basename(pair.png, path.extname(pair.png));
    await send('Page.navigate', { url: pathToFileURL(path.join(root, 'spinal-annotation-web.html')).href });
    await poll(() => evaluate("document.readyState === 'complete' && typeof validateAnnotationForCurrentImage === 'function'"), 20000, 'editor');
    // 攔截匯出的 JSON（不經下載資料夾）
    await evaluate(`window.__blobs = []; { const o = URL.createObjectURL.bind(URL); URL.createObjectURL = b => { window.__blobs.push(b); return o(b); }; } true`);

    // A. 一般載入
    dialogs = [];
    await setFiles('#fileInput', [pair.png]);
    await poll(() => evaluate('!!originalImage && !!(currentImageMeta && currentImageMeta.pixelSha256)'), 30000, 'image');
    const pixelSha256 = await evaluate('currentImageMeta.pixelSha256');
    const fileSha256 = await evaluate('currentImageMeta.sha256');
    await setFiles('#jsonInput', [pair.json]);
    await poll(() => evaluate(`vertebrae.length === ${n}`), 10000, 'json import').catch(() => {});
    const a = { imported: await evaluate('vertebrae.length'), rejected: rejected(), jsonShaMatches: (gt.imageInfo?.sha256 || '') === fileSha256 };

    // B. 貼上 → 匯出 → 另存 → 載入另存檔＋匯出的 JSON
    dialogs = [];
    const b = await evaluate(`(async () => {
      const original = medsamSource.blob;
      const pasted = new File([await original.arrayBuffer()], 'image.png', { type: 'image/png' });   // 剪貼簿給的檔名
      annotationDirty = false;
      loadImageFromFile(pasted);
      const until = async f => { for (let k = 0; k < 300 && !f(); k++) await new Promise(r => setTimeout(r, 100)); };
      await until(() => currentImageMeta && currentImageMeta.filename === 'image.png' && currentImageMeta.pixelSha256);
      importAnnotationJson(${JSON.stringify(gt)}, { skipIdentityValidation: true, replaceConfirmed: true, silent: true });
      annotationDirty = true;
      exportData();
      const exported = JSON.parse(await window.__blobs.at(-1).text());
      // 另存：重新編碼（位元組不同、像素相同），檔名改成主檔名
      const c = document.createElement('canvas');
      c.width = originalImage.naturalWidth; c.height = originalImage.naturalHeight;
      c.getContext('2d').drawImage(originalImage, 0, 0);
      // Chrome 的 canvas 編碼可能和原檔逐位元組相同，檔尾再加幾個位元組（解碼器會忽略）確保位元組不同
      const resaved = new File([await new Promise(r => c.toBlob(r, 'image/png')), new TextEncoder().encode('resaved')],
                               ${JSON.stringify(stem + '.png')}, { type: 'image/png' });
      annotationDirty = false;
      loadImageFromFile(resaved);
      await until(() => currentImageMeta && currentImageMeta.filename === ${JSON.stringify(stem + '.png')} && currentImageMeta.pixelSha256);
      const resavedMeta = { ...currentImageMeta };
      const ok = importAnnotationJson(exported, { jsonFilename: ${JSON.stringify(stem + '.json')} });
      const count = vertebrae.length;
      const exportedPixel = exported.imageInfo.pixelSha256 || null, exportedMethod = exported.imageInfo.pixelHashMethod || null;
      // C. 同一張另存檔＋拿掉 pixelSha256 的 JSON（本次修改前的格式）
      delete exported.imageInfo.pixelSha256;
      delete exported.imageInfo.pixelHashMethod;
      annotationDirty = false;
      const okLegacy = importAnnotationJson(exported, { jsonFilename: ${JSON.stringify(stem + '.json')} });
      return { exportedFilename: exported.imageInfo.filename, exportedPixel, exportedMethod, resavedPixel: resavedMeta.pixelSha256,
               ok, count, okLegacy, countLegacy: vertebrae.length, bytesDiffer: resavedMeta.sha256 !== exported.imageInfo.sha256 };
    })()`);
    b.rejected = rejected();
    results.push({ pair: i + 1, vertebrae: n, pixelSha256, fileSha256, a, b, dialogs });

    assert.equal(a.imported, n, `pair ${i + 1} A：一般載入應匯入 ${n} 節；dialogs ${JSON.stringify(dialogs)}`);
    assert.deepEqual(a.rejected, [], `pair ${i + 1} A：不應拒絕`);
    assert.equal(b.exportedFilename, 'image.png');
    assert.equal(b.exportedPixel, pixelSha256, '匯出的像素雜湊應與原檔相同');
    assert.equal(b.exportedMethod, 'r-channel-v1');
    assert.equal(b.resavedPixel, pixelSha256, '另存後像素雜湊應不變');
    assert.equal(b.bytesDiffer, true, '另存後檔案位元組應不同');
    assert.equal(b.ok && b.count === n, true, `pair ${i + 1} B：另存檔應以像素雜湊通過`);
    assert.equal(b.okLegacy && b.countLegacy === n, true, `pair ${i + 1} C：舊格式 JSON 應以主檔名通過`);
    assert.deepEqual(b.rejected, [], `pair ${i + 1} B/C：不應拒絕或要求確認`);
  }
  fs.writeFileSync(path.join(workDir, 'identity_result.json'), JSON.stringify(results, null, 2));
  process.stdout.write(JSON.stringify(results.map(r => ({ pair: r.pair, vertebrae: r.vertebrae, a: r.a, b: r.b })), null, 2) + '\n');
} finally {
  try { socket?.close(); } catch { /* 忽略 */ }
  chrome.kill();
}
