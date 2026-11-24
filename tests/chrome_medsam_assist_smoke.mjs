// 「🧠 MedSAM 框選」在真實 Chrome＋真實 medsam_server.py（真實模型）上實跑。
//
// 流程：本機 http 提供標註工具 → 載入真實影像與醫師 JSON → 選取一節、用真實滑鼠事件拉框 →
// 「修整全部」→ 匯出含 assist／assistedBy → 截圖。另外直接對服務上傳外部測試集影像，確認被拒。
// 若 127.0.0.1:8003 已有服務就沿用，否則由本腳本啟動並在結束時關閉。
//
// 用法：node tests/chrome_medsam_assist_smoke.mjs <chrome.exe> <image.png> <annotation.json> <work-dir> [holdout.png]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const [chromePath, imagePath, jsonPath, workDir, holdoutPath] = process.argv.slice(2);
if (![chromePath, imagePath, jsonPath, workDir].every(Boolean)) {
  throw new Error('Usage: node chrome_medsam_assist_smoke.mjs <chrome> <image.png> <annotation.json> <work-dir> [holdout.png]');
}
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const httpPort = 8767;
// 9305–9404 在這台 Windows 被 Hyper-V 保留（netsh interface ipv4 show excludedportrange protocol=tcp）
const cdpPort = Number(process.env.CDP_PORT || 9241);
const medsamUrl = 'http://127.0.0.1:8003';
fs.mkdirSync(workDir, { recursive: true });
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function poll(fn, timeoutMs = 20000, what = 'condition') {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try { const result = await fn(); if (result) return result; } catch (error) { lastError = error; }
    await delay(200);
  }
  throw lastError || new Error(`Timed out waiting for ${what}`);
}
const healthy = async () => { try { return (await fetch(medsamUrl + '/health')).ok; } catch { return false; } };

let medsam = null;
if (!(await healthy())) {
  medsam = spawn('python', ['medsam_server.py'], { cwd: root, stdio: 'ignore', windowsHide: true });
}
const server = spawn('python', ['-m', 'http.server', String(httpPort), '--bind', '127.0.0.1'], { cwd: root, stdio: 'ignore', windowsHide: true });
const chrome = spawn(chromePath, ['--headless=new', '--disable-gpu', `--remote-debugging-port=${cdpPort}`, '--window-size=1400,1000',
  `--user-data-dir=${path.join(workDir, 'chrome-profile')}`, 'about:blank'], { stdio: 'ignore', windowsHide: true });

let socket;
try {
  await poll(healthy, 120000, 'medsam_server');
  const health = await (await fetch(medsamUrl + '/health')).json();

  // 外部測試集影像：服務必須拒絕
  let holdout = null;
  if (holdoutPath) {
    const form = new FormData();
    form.append('file', new Blob([fs.readFileSync(holdoutPath)], { type: 'image/png' }), 'h.png');
    const r = await fetch(medsamUrl + '/image', { method: 'POST', body: form });
    holdout = { status: r.status, detail: (await r.json()).detail };
  }

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
  await poll(() => evaluate("document.readyState === 'complete' && typeof medsamApplyBox === 'function'"), 20000, 'editor');

  const rel = path.relative(root, imagePath).split(path.sep).join('/');
  const relJson = path.relative(root, jsonPath).split(path.sep).join('/');
  const gt = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));

  // 載入真實影像與醫師標註
  await evaluate(`(async () => {
    const blob = await (await fetch(${JSON.stringify(rel)})).blob();
    const file = new File([blob], ${JSON.stringify(path.basename(imagePath))}, { type: 'image/png', lastModified: 1 });
    const sha = await calculateFileSha256(file);
    const img = new Image();
    await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = URL.createObjectURL(blob); });
    commitLoadedImage(img, file, sha);
    const data = await (await fetch(${JSON.stringify(relJson)})).json();
    setSpineType(data.spineType || 'L');
    return importAnnotationJson(data, { jsonFilename: ${JSON.stringify(path.basename(jsonPath))}, silent: true });
  })()`);
  assert.equal(await evaluate('toggleMedsamMode(true)'), true, 'MedSAM 模式應可開啟（服務已連線）');

  // 選一節完整椎體，用真實滑鼠事件拉框（框 = 醫師點位外接框 + 4%）
  const pick = await evaluate(`(() => {
    const i = vertebrae.findIndex(v => !v.boundaryType && v.points.length === 6 && v.name !== 'L1');
    selectedVertebrae.clear(); selectedVertebrae.add(i);
    const box = medsamBoxFromPoints(vertebrae[i].points);
    // 讓整節出現在「視窗內可見的畫布區域」中央（畫布可能比視窗高）
    const r1 = canvas.getBoundingClientRect();
    const vx0 = Math.max(r1.left, 0), vx1 = Math.min(r1.right, innerWidth);
    const vy0 = Math.max(r1.top, 0), vy1 = Math.min(r1.bottom, innerHeight);
    const sx = canvas.width / r1.width, sy = canvas.height / r1.height;
    const visW = (vx1 - vx0) * sx, visH = (vy1 - vy0) * sy;
    zoom = Math.min(visW, visH) / Math.max(box[2] - box[0], box[3] - box[1]) / 2.5;
    panX = ((vx0 + vx1) / 2 - r1.left) * sx - (box[0] + box[2]) / 2 * zoom;
    panY = ((vy0 + vy1) / 2 - r1.top) * sy - (box[1] + box[3]) / 2 * zoom;
    redraw();
    const r2 = canvas.getBoundingClientRect();
    const c = (x, y) => ({ x: r2.left + (x * zoom + panX) * r2.width / canvas.width, y: r2.top + (y * zoom + panY) * r2.height / canvas.height });
    // 從框的邊緣外一點點開始，避免一開始就點到既有角點
    return { index: i, name: vertebrae[i].name, before: vertebrae[i].points.map(p => ({ ...p })),
             start: c(box[0], box[1]), end: c(box[2], box[3]) };
  })()`);
  const mouse = (type, p, buttons = 1) => send('Input.dispatchMouseEvent', { type, x: p.x, y: p.y, button: 'left', buttons, clickCount: 1 });
  await mouse('mousePressed', pick.start);
  for (let t = 0.25; t <= 1.001; t += 0.25) {
    await mouse('mouseMoved', { x: pick.start.x + (pick.end.x - pick.start.x) * t, y: pick.start.y + (pick.end.y - pick.start.y) * t });
  }
  await mouse('mouseReleased', pick.end, 0);
  try {
    await poll(() => evaluate(`!medsamBusy && !!vertebrae[${pick.index}].assist`), 60000, 'medsam drag result');
  } catch (error) {
    const state = await evaluate(`({ mode: medsamMode, busy: medsamBusy, box: medsamBox, selected: [...selectedVertebrae],
      count: vertebrae.length, status: document.getElementById('medsamStatus').textContent })`);
    throw new Error(`${error.message}；狀態 ${JSON.stringify(state)}；pick ${JSON.stringify(pick)}；dialogs ${JSON.stringify(dialogs)}`);
  }
  const single = await evaluate(`(() => {
    const v = vertebrae[${pick.index}];
    const before = ${JSON.stringify(pick.before)};
    const d = v.points.map((p, k) => Math.hypot(p.x - before[k].x, p.y - before[k].y));
    return { name: v.name, count: vertebrae.length, assist: v.assist, errors: d.map(x => Math.round(x)),
             status: document.getElementById('medsamStatus').textContent };
  })()`);

  // 修整全部（confirm 由 CDP 自動接受）
  const refined = await evaluate('medsamRefineAll()');
  const summary = await evaluate(`(() => {
    const gtv = ${JSON.stringify(gt.vertebrae)};
    const keys = ['anteriorSuperior','middleSuperior','posteriorSuperior','posteriorInferior','middleInferior','anteriorInferior'];
    const corner = [0, 2, 3, 5];
    const errs = [];
    for (const v of vertebrae.filter(v => !v.boundaryType)) {
      const g = gtv.find(x => x.name === v.name);
      if (!g) continue;
      corner.forEach(k => errs.push(Math.hypot(v.points[k].x - g.points[keys[k]].x, v.points[k].y - g.points[keys[k]].y)));
    }
    errs.sort((a, b) => a - b);
    const data = buildExportData();
    return { refined: vertebrae.filter(v => v.assist).length, cornerMedianPx: Math.round(errs[Math.floor(errs.length / 2)]),
             cornerMaxPx: Math.round(errs[errs.length - 1]), assistedBy: data.annotationReview.assistedBy,
             exportedAssist: data.vertebrae.filter(v => v.assist).length, dirty: annotationDirty,
             status: document.getElementById('medsamStatus').textContent };
  })()`);

  // 截圖（整張影像縮到畫面內）
  const clip = await evaluate(`(() => { resetZoom(); selectedVertebrae.clear(); redraw();
    const r = canvas.getBoundingClientRect();
    return { x: r.left + scrollX, y: r.top + scrollY, width: r.width, height: r.height, scale: 1 }; })()`);
  const shot = await send('Page.captureScreenshot', { format: 'png', clip, captureBeyondViewport: true });
  const shotPath = path.join(workDir, 'medsam_assist.png');
  fs.writeFileSync(shotPath, Buffer.from(shot.data, 'base64'));

  const mainDialogs = dialogs.slice();

  // 外部測試集：走標註工具本身的上傳路徑（原始檔、以及沒有原始檔時的 canvas 重新編碼），都必須被拒
  let browserHoldout = null;
  if (holdoutPath) {
    const b64 = fs.readFileSync(holdoutPath).toString('base64');
    browserHoldout = await evaluate(`(async () => {
      const bytes = Uint8Array.from(atob(${JSON.stringify(b64)}), c => c.charCodeAt(0));
      const blob = new Blob([bytes], { type: 'image/png' });
      const out = {};
      for (const mode of ['original', 'reencoded']) {
        const img = new Image();
        await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = URL.createObjectURL(blob); });
        annotationDirty = false;
        commitLoadedImage(img, mode === 'original' ? new File([blob], 'h.png', { type: 'image/png' }) : { name: 'pasted.png' }, null);
        await toggleMedsamMode(true);
        const ok = await medsamApplyBox([100, 100, 400, 400]);
        out[mode] = { ok, count: vertebrae.length, status: document.getElementById('medsamStatus').textContent };
      }
      return out;
    })()`);
  }

  const result = { health, holdout, browserHoldout, single, refinedAll: refined, summary, dialogs, screenshot: shotPath };
  assert.equal(single.count, gt.vertebrae.length, '修正選取的那一節，不應新增椎體');
  assert.equal(single.assist.method, 'MedSAM');
  assert.ok(Math.max(...single.errors) < 150, `拉框結果離醫師點位太遠：${single.errors}`);
  assert.equal(summary.refined, refined);
  assert.equal(summary.exportedAssist, refined);
  assert.equal(summary.assistedBy.method, 'MedSAM');
  assert.ok(summary.cornerMedianPx < 80, `修整全部角點中位數 ${summary.cornerMedianPx} px`);
  assert.match(mainDialogs[0] || '', /完整椎體/, '第一個對話框應是「修整全部」的確認');
  assert.ok(!mainDialogs.some(d => /連不到|失敗|外部測試集/.test(d)), `不應出現錯誤對話框：${mainDialogs.join(' | ')}`);
  if (holdoutPath) {
    assert.equal(holdout.status, 403);
    assert.match(holdout.detail, /外部測試集/);
    for (const mode of ['original', 'reencoded']) {
      assert.equal(browserHoldout[mode].ok, false, `${mode}：外部測試集影像不應產生點位`);
      assert.equal(browserHoldout[mode].count, 0);
      assert.match(browserHoldout[mode].status, /外部測試集/, `${mode}：${browserHoldout[mode].status}`);
    }
  }
  process.stdout.write(JSON.stringify(result, null, 2) + '\n');
} finally {
  socket?.close();
  chrome.kill();
  server.kill();
  medsam?.kill();
}
