// 「📦 瀏覽器模型」在真實 Chrome＋真實 ONNX 模型上實跑（不需要 Python 服務）。
//
// 以 file:// 開啟標註工具（與醫院電腦直接打開 HTML 相同），經由真實的檔案選擇框載入影像、醫師 JSON
// 與模型資料夾 → 「修整全部」→ 與 Python（tests/medsam_reference_points.py）的結果逐點比對 → 截圖。
// 再載入外部測試集影像（原始檔、以及沒有原始檔雜湊的貼上影像），兩種都必須被拒。
//
// 用法：node tests/chrome_medsam_browser_smoke.mjs <chrome.exe> <onnx 資料夾> <image.png> <annotation.json>
//         <python 參考答案.json> <work-dir> [holdout.png]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawn } from 'node:child_process';

const [chromePath, onnxDir, imagePath, jsonPath, refPath, workDir, holdoutPath] = process.argv.slice(2);
if (![chromePath, onnxDir, imagePath, jsonPath, refPath, workDir].every(Boolean)) {
  throw new Error('Usage: node chrome_medsam_browser_smoke.mjs <chrome> <onnx-dir> <image.png> <annotation.json> <ref.json> <work-dir> [holdout.png]');
}
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const cdpPort = Number(process.env.CDP_PORT || 9243);   // 9305–9404 在這台 Windows 被保留
fs.mkdirSync(workDir, { recursive: true });
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function poll(fn, timeoutMs, what) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try { const r = await fn(); if (r) return r; } catch (e) { lastError = e; }
    await delay(300);
  }
  throw lastError || new Error(`Timed out waiting for ${what}`);
}

const chrome = spawn(chromePath, ['--headless=new', '--enable-unsafe-webgpu', `--remote-debugging-port=${cdpPort}`, '--window-size=1400,1000',
  `--user-data-dir=${path.join(workDir, 'chrome-profile')}`, 'about:blank'], { stdio: 'ignore', windowsHide: true });

let socket;
try {
  const target = await poll(async () => (await (await fetch(`http://127.0.0.1:${cdpPort}/json/list`)).json()).find(t => t.type === 'page'), 20000, 'chrome');
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.addEventListener('open', resolve, { once: true }); socket.addEventListener('error', reject, { once: true }); });
  let nextId = 0;
  const pending = new Map();
  const dialogs = [];
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
  await send('Page.enable');
  await send('Runtime.enable');
  await send('DOM.enable');
  await send('Page.navigate', { url: pathToFileURL(path.join(root, 'spinal-annotation-web.html')).href });
  await poll(() => evaluate("document.readyState === 'complete' && typeof medsamLoadBrowserModel === 'function'"), 20000, 'editor');

  // 影像與醫師 JSON：走真實的檔案選擇框
  const gt = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
  await evaluate(`setSpineType(${JSON.stringify(gt.spineType || 'L')})`);
  await setFiles('#fileInput', [imagePath]);
  await poll(() => evaluate('!!originalImage && !!(currentImageMeta && currentImageMeta.sha256)'), 20000, 'image');
  await setFiles('#jsonInput', [jsonPath]);
  await poll(() => evaluate(`vertebrae.length === ${gt.vertebrae.length}`), 20000, 'json import');

  // 模型資料夾
  const modelFiles = ['meta.json', 'decoder.onnx', 'encoder_fp16.onnx', 'encoder.onnx', 'holdout_hashes.json']
    .map(f => path.join(onnxDir, f)).filter(f => fs.existsSync(f));
  const t0 = Date.now();
  await setFiles('#medsamModelInput', modelFiles);
  try {
    await poll(() => evaluate('!!medsamWeb && medsamMode'), 180000, 'browser model');
  } catch (error) {
    const s = await evaluate(`({ web: !!medsamWeb, mode: medsamMode, ort: !!window.ort, gpu: !!navigator.gpu,
      status: document.getElementById('medsamStatus').textContent })`);
    throw new Error(`${error.message}；${JSON.stringify(s)}；dialogs ${JSON.stringify(dialogs)}；files ${JSON.stringify(modelFiles.map(f => path.basename(f)))}`);
  }
  const loadSeconds = (Date.now() - t0) / 1000;
  const backend = await evaluate('({ backend: medsamBackend, device: medsamWeb.device, status: document.getElementById("medsamStatus").textContent })');
  assert.match(backend.device, /外部測試集 \d+ 張（清單 /, '狀態要顯示外部測試集阻擋清單的筆數');

  // 修整全部（confirm 由 CDP 自動接受）
  const t1 = Date.now();
  const refined = await evaluate('medsamRefineAll()');
  const refineSeconds = (Date.now() - t1) / 1000;
  const ref = JSON.parse(fs.readFileSync(refPath, 'utf8'));
  const keys = ['anteriorSuperior', 'middleSuperior', 'posteriorSuperior', 'posteriorInferior', 'middleInferior', 'anteriorInferior'];
  const state = await evaluate('JSON.stringify(vertebrae.map(v => ({ name: v.name, boundaryType: v.boundaryType || null, points: v.points, assist: v.assist || null })))');
  const verts = JSON.parse(state);
  const parity = {}, gtErrors = [];
  for (const v of verts.filter(x => !x.boundaryType)) {
    const r = ref.vertebrae[v.name];
    const g = gt.vertebrae.find(x => x.name === v.name);
    parity[v.name] = keys.map((k, i) => Math.round(Math.hypot(v.points[i].x - r.points[k].x, v.points[i].y - r.points[k].y) * 10) / 10);
    [0, 2, 3, 5].forEach(i => gtErrors.push(Math.hypot(v.points[i].x - g.points[keys[i]].x, v.points[i].y - g.points[keys[i]].y)));
  }
  gtErrors.sort((a, b) => a - b);
  const clip = await evaluate(`(() => { resetZoom(); selectedVertebrae.clear(); redraw();
    const r = canvas.getBoundingClientRect(); return { x: r.left + scrollX, y: r.top + scrollY, width: r.width, height: r.height, scale: 1 }; })()`);
  const shot = await send('Page.captureScreenshot', { format: 'png', clip, captureBeyondViewport: true });
  fs.writeFileSync(path.join(workDir, 'medsam_browser.png'), Buffer.from(shot.data, 'base64'));
  const mainDialogs = dialogs.slice();

  // 外部測試集：原始檔（檔案雜湊）與貼上的影像（只有像素指紋），都必須被拒
  let holdout = null;
  if (holdoutPath) {
    holdout = {};
    await evaluate('annotationDirty = false');
    await setFiles('#fileInput', [holdoutPath]);
    await poll(() => evaluate('vertebrae.length === 0 && !!(currentImageMeta && currentImageMeta.sha256)'), 20000, 'holdout image');
    holdout.original = await evaluate(`(async () => { await toggleMedsamMode(true); const ok = await medsamApplyBox([100, 100, 400, 400]);
      return { ok, count: vertebrae.length, status: document.getElementById('medsamStatus').textContent }; })()`);
    const b64 = fs.readFileSync(holdoutPath).toString('base64');
    holdout.pasted = await evaluate(`(async () => {
      const bytes = Uint8Array.from(atob(${JSON.stringify(b64)}), c => c.charCodeAt(0));
      const img = new Image();
      await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = URL.createObjectURL(new Blob([bytes], { type: 'image/png' })); });
      annotationDirty = false;
      commitLoadedImage(img, { name: 'pasted.png' }, null);
      await toggleMedsamMode(true);
      const ok = await medsamApplyBox([100, 100, 400, 400]);
      return { ok, count: vertebrae.length, status: document.getElementById('medsamStatus').textContent };
    })()`);
  }

  const maxParity = Math.max(...Object.values(parity).flat());
  const result = { backend, loadSeconds, refined, refineSeconds, secondsPerVertebra: +(refineSeconds / Math.max(refined, 1)).toFixed(2),
    parityPxVsPython: parity, maxParityPx: maxParity, cornerMedianVsGtPx: Math.round(gtErrors[Math.floor(gtErrors.length / 2)]),
    assist: verts.filter(v => v.assist).length, holdout, dialogs };
  fs.writeFileSync(path.join(workDir, 'result.json'), JSON.stringify(result, null, 2));
  process.stdout.write(JSON.stringify(result, null, 2) + '\n');

  assert.equal(backend.backend, 'browser');
  assert.equal(refined, verts.filter(v => !v.boundaryType).length);
  assert.ok(maxParity <= 6, `瀏覽器與 Python 點位差太多：${JSON.stringify(parity)}`);
  assert.ok(result.cornerMedianVsGtPx < 80, `角點中位數 ${result.cornerMedianVsGtPx}`);
  assert.ok(mainDialogs.some(d => /完整椎體/.test(d)), '應出現「修整全部」的確認');
  assert.ok(!mainDialogs.some(d => /失敗|外部測試集|連不到/.test(d)), `不應有錯誤對話框：${mainDialogs.join(' | ')}`);
  if (holdout) {
    for (const k of ['original', 'pasted']) {
      assert.equal(holdout[k].ok, false, `${k}：外部測試集不應產生點位`);
      assert.equal(holdout[k].count, 0);
      assert.match(holdout[k].status, /外部測試集/, `${k}：${holdout[k].status}`);
    }
  }
} finally {
  socket?.close();
  chrome.kill();
}
