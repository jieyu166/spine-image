// medsam_onnx_bench.html 在真實 Chrome 上自動測速（CDP 選檔、按「開始測速」、讀 window.benchResult）。
//
// 預設以 file:// 開啟（與醫院電腦「直接打開本機 HTML」相同：WASM 只能單執行緒）。
// --isolated：改由本機 http 伺服器加上 COOP/COEP 標頭開啟，WASM 可多執行緒（對照用）。
//
// 用法：node tests/chrome_medsam_onnx_bench.mjs <chrome.exe> <onnx 資料夾> <work-dir>
//         [--encoder encoder.onnx] [--modes wasm,webgpu,webgpu-low] [--runs 3] [--isolated]
import assert from 'node:assert/strict';
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { spawn } from 'node:child_process';

const argv = process.argv.slice(2);
const flag = (name, fallback) => { const i = argv.indexOf(name); return i >= 0 ? argv[i + 1] : fallback; };
const [chromePath, onnxDir, workDir] = argv;   // 三個位置參數在前，選項在後
if (![chromePath, onnxDir, workDir].every(Boolean)) {
  throw new Error('Usage: node chrome_medsam_onnx_bench.mjs <chrome> <onnx-dir> <work-dir> [--encoder f] [--modes wasm,webgpu] [--runs 3] [--isolated]');
}
const encoderName = flag('--encoder', 'encoder.onnx');
const modes = flag('--modes', 'wasm,webgpu').split(',');
const runs = Number(flag('--runs', 3));
const isolated = argv.includes('--isolated');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
// 9305–9404 在這台 Windows 被 Hyper-V 保留
const cdpPort = Number(process.env.CDP_PORT || 9242);
const httpPort = 8768;
fs.mkdirSync(workDir, { recursive: true });
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function poll(fn, timeoutMs, what) {
  const deadline = Date.now() + timeoutMs;
  let lastError;
  while (Date.now() < deadline) {
    try { const r = await fn(); if (r) return r; } catch (e) { lastError = e; }
    await delay(500);
  }
  throw lastError || new Error(`Timed out waiting for ${what}`);
}

let server = null;
let pageUrl = pathToFileURL(path.join(root, 'medsam_onnx_bench.html')).href;
if (isolated) {
  server = http.createServer((req, res) => {
    const file = path.join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname));
    if (!file.startsWith(root) || !fs.existsSync(file)) { res.writeHead(404); res.end(); return; }
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8', 'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless' });
    fs.createReadStream(file).pipe(res);
  }).listen(httpPort, '127.0.0.1');
  pageUrl = `http://127.0.0.1:${httpPort}/medsam_onnx_bench.html`;
}

const chrome = spawn(chromePath, ['--headless=new', '--enable-unsafe-webgpu', `--remote-debugging-port=${cdpPort}`, '--window-size=1200,900',
  `--user-data-dir=${path.join(workDir, 'chrome-profile')}`, 'about:blank'], { stdio: 'ignore', windowsHide: true });

let socket;
try {
  const target = await poll(async () => (await (await fetch(`http://127.0.0.1:${cdpPort}/json/list`)).json()).find(t => t.type === 'page'), 20000, 'chrome');
  socket = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.addEventListener('open', resolve, { once: true }); socket.addEventListener('error', reject, { once: true }); });
  let nextId = 0;
  const pending = new Map();
  socket.addEventListener('message', event => {
    const m = JSON.parse(event.data);
    if (m.method === 'Page.javascriptDialogOpening') { send('Page.handleJavaScriptDialog', { accept: true }).catch(() => {}); return; }
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
  await send('Page.enable');
  await send('Runtime.enable');
  await send('DOM.enable');
  await send('Page.navigate', { url: pageUrl });
  await poll(() => evaluate("document.readyState === 'complete' && !!window.benchEnv"), 60000, 'bench page');
  assert.ok(await evaluate('!!self.ort'), 'onnxruntime-web 沒從 CDN 載入');

  const { root: doc } = await send('DOM.getDocument', {});
  for (const [selector, file] of [['#encFile', encoderName], ['#decFile', 'decoder.onnx']]) {
    const { nodeId } = await send('DOM.querySelector', { nodeId: doc.nodeId, selector });
    await send('DOM.setFileInputFiles', { nodeId, files: [path.resolve(onnxDir, file)] });
  }
  await evaluate(`(() => {
    $('useWasm').checked = ${modes.includes('wasm')};
    $('useGpu').checked = ${modes.includes('webgpu')};
    $('useGpuLow').checked = ${modes.includes('webgpu-low')};
    $('runs').value = ${runs};
    window.benchResult = null;
    $('run').click();
    return true;
  })()`);
  const result = await poll(() => evaluate('window.benchResult'), 40 * 60 * 1000, 'bench result');
  result.pageUrl = pageUrl;
  result.log = await evaluate("$('log').textContent");
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  fs.writeFileSync(path.join(workDir, `bench_${path.parse(encoderName).name}${isolated ? '_isolated' : ''}.png`), Buffer.from(shot.data, 'base64'));
  process.stdout.write(JSON.stringify(result, null, 2) + '\n');
  assert.ok(result.results.length > 0, `沒有任何模式成功：${JSON.stringify(result.errors)}`);
} finally {
  socket?.close();
  chrome.kill();
  server?.close();
}
