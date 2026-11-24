'use strict';

// spinal-annotation-web.html「🧠 MedSAM 框選」：拉框 → 本機 medsam_server.py → 該節點位。
const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');
const { loadHtmlScript } = require('./html_script_harness');

const editorPath = path.join(__dirname, '..', 'spinal-annotation-web.html');
const FULL = ['anteriorSuperior', 'middleSuperior', 'posteriorSuperior', 'posteriorInferior', 'middleInferior', 'anteriorInferior'];

function vertebraPoints(cx, cy, keys = FULL) {
  const local = {
    anteriorSuperior: [-150, -100], middleSuperior: [0, -95], posteriorSuperior: [150, -100],
    posteriorInferior: [150, 100], middleInferior: [0, 95], anteriorInferior: [-150, 100],
  };
  return Object.fromEntries(keys.map(k => [k, { x: cx + local[k][0], y: cy + local[k][1] }]));
}

function fakeServer({ uploadStatus = 200, uploadDetail = '', segmentFailOnce = 0, health = true, onSegment = null } = {}) {
  const calls = [];
  let failures = segmentFailOnce;
  const reply = (status, body) => ({ ok: status >= 200 && status < 300, status, json: async () => body });
  const fetch = async (url, opts = {}) => {
    const route = url.replace('http://127.0.0.1:8003', '');
    const body = opts.body && typeof opts.body === 'string' ? JSON.parse(opts.body) : opts.body;
    calls.push({ route, method: opts.method || 'GET', body });
    if (route === '/health') {
      if (!health) throw new TypeError('Failed to fetch');
      return reply(200, { ok: true, fine_tuned: true });
    }
    if (route === '/image') {
      return uploadStatus === 200 ? reply(200, { image_id: 'img-1', width: 1000, height: 2000 })
        : reply(uploadStatus, { detail: uploadDetail });
    }
    if (route === '/segment') {
      if (failures > 0) { failures -= 1; return reply(404, { detail: '找不到影像，請重新上傳' }); }
      if (onSegment) await onSegment(body);
      const [x0, y0, x1, y1] = body.box;
      const keys = body.boundary === 'upper' ? FULL.slice(0, 3)
        : body.boundary === 'lower' ? ['anteriorInferior', 'middleInferior', 'posteriorInferior'] : FULL;
      return reply(200, { points: vertebraPoints((x0 + x1) / 2, (y0 + y1) / 2, keys), iou: 0.91, outline: [], boundary: body.boundary });
    }
    return reply(404, {});
  };
  return { fetch, calls };
}

async function openPage(server, spineType = 'L') {
  const page = loadHtmlScript(editorPath, { fetch: server.fetch, FormData });
  page.initialize();
  page.evaluate(`setSpineType && setSpineType(${JSON.stringify(spineType)})`);
  page.evaluate(`commitLoadedImage({width:1000,height:2000,naturalWidth:1000,naturalHeight:2000},{name:'case.png',size:10,lastModified:1},'${'c'.repeat(64)}')`);
  // harness 的 canvas 沒有 toBlob：換成固定 Blob
  page.evaluate(`medsamImageBlob = async () => new Blob(['png-bytes'], { type: 'image/png' })`);
  return page;
}

test('拉框依序產生 S1（上終板 3 點）與 L5（6 點），並記錄 assist', async () => {
  const server = fakeServer();
  const page = await openPage(server);
  assert.equal(await page.evaluate('toggleMedsamMode(true)'), true);
  assert.equal(await page.evaluate('medsamApplyBox([400, 1700, 600, 1500])'), true);   // 反向拖曳也可
  assert.equal(await page.evaluate('medsamApplyBox([380, 1200, 620, 1420])'), true);
  const verts = JSON.parse(page.evaluate('JSON.stringify(vertebrae)'));
  assert.deepEqual(verts.map(v => v.name), ['S1', 'L5']);
  assert.equal(verts[0].boundaryType, 'upper');
  assert.equal(verts[0].points.length, 3);
  assert.deepEqual(verts[0].points[0], { x: 350, y: 1500 });          // 前上角 = 中心 (500,1600) + (-150,-100)
  assert.equal(verts[1].points.length, 6);
  assert.deepEqual(verts[1].points[3], { x: 650, y: 1410 });          // 後下角
  assert.deepEqual(verts[1].assist, { method: 'MedSAM', iou: 0.91, edited: false });
  const segs = server.calls.filter(c => c.route === '/segment');
  assert.equal(segs[0].body.boundary, 'upper');
  assert.deepEqual(segs[0].body.box, [400, 1500, 600, 1700]);
  assert.equal(segs[1].body.boundary, null);
  assert.equal(server.calls.filter(c => c.route === '/image').length, 1, '同一張影像只上傳一次');
  const upload = server.calls.find(c => c.route === '/image').body;
  assert.equal(upload.get('sha256'), 'c'.repeat(64));
});

test('先選取一節再拉框＝只修正該節，名稱不變', async () => {
  const server = fakeServer();
  const page = await openPage(server);
  await page.evaluate('toggleMedsamMode(true)');
  await page.evaluate('medsamApplyBox([400, 1500, 600, 1700])');
  await page.evaluate('medsamApplyBox([380, 1200, 620, 1420])');
  page.evaluate('selectedVertebrae.clear(); selectedVertebrae.add(1)');
  assert.equal(await page.evaluate('medsamApplyBox([300, 1000, 500, 1200])'), true);
  const verts = JSON.parse(page.evaluate('JSON.stringify(vertebrae)'));
  assert.equal(verts.length, 2);
  assert.equal(verts[1].name, 'L5');
  assert.deepEqual(verts[1].points[0], { x: 250, y: 1000 });
});

test('外部測試集影像被拒：顯示原因、不新增椎體', async () => {
  const server = fakeServer({ uploadStatus: 403, uploadDetail: '這張是外部測試集（holdout_20261029）的影像，不提供 MedSAM 輔助' });
  const page = await openPage(server);
  await page.evaluate('toggleMedsamMode(true)');
  assert.equal(await page.evaluate('medsamApplyBox([400, 1500, 600, 1700])'), false);
  assert.equal(page.evaluate('vertebrae.length'), 0);
  assert.match(page.alerts.at(-1), /外部測試集/);
  assert.match(page.document.getElementById('medsamStatus').textContent, /外部測試集/);
});

test('服務重啟（影像快取 404）時自動重新上傳再試一次', async () => {
  const server = fakeServer({ segmentFailOnce: 1 });
  const page = await openPage(server);
  await page.evaluate('toggleMedsamMode(true)');
  assert.equal(await page.evaluate('medsamApplyBox([400, 1500, 600, 1700])'), true);
  assert.equal(server.calls.filter(c => c.route === '/image').length, 2);
  assert.equal(page.evaluate('vertebrae.length'), 1);
});

test('服務沒開：無法進入 MedSAM 模式並提示啟動指令', async () => {
  const page = await openPage(fakeServer({ health: false }));
  assert.equal(await page.evaluate('toggleMedsamMode(true)'), false);
  assert.equal(page.evaluate('medsamMode'), false);
  assert.match(page.alerts.at(-1), /python medsam_server\.py/);
});

test('太小的框與點到一半的椎體不送出', async () => {
  const server = fakeServer();
  const page = await openPage(server);
  await page.evaluate('toggleMedsamMode(true)');
  assert.equal(await page.evaluate('medsamApplyBox([10, 10, 15, 200])'), false);
  page.evaluate('currentPoints.push({x: 1, y: 2})');
  assert.equal(await page.evaluate('medsamApplyBox([400, 1500, 600, 1700])'), false);
  assert.match(page.alerts.at(-1), /部分點/);
  assert.equal(server.calls.filter(c => c.route === '/segment').length, 0);
});

test('MedSAM 模式下單擊畫布不新增點', async () => {
  const page = await openPage(fakeServer());
  await page.evaluate('toggleMedsamMode(true)');
  page.evaluate(`handleCanvasClick({ clientX: 500, clientY: 500, shiftKey: false })`);
  assert.equal(page.evaluate('currentPoints.length'), 0);
});

test('匯出帶 assist 與 assistedBy；拖曳後標記 edited；匯入保留 assist', async () => {
  const server = fakeServer();
  const page = await openPage(server);
  await page.evaluate('toggleMedsamMode(true)');
  await page.evaluate('medsamApplyBox([400, 1500, 600, 1700])');
  await page.evaluate('medsamApplyBox([380, 1200, 620, 1420])');
  page.evaluate('medsamMarkEdited([1])');
  const data = JSON.parse(page.evaluate('JSON.stringify(buildExportData())'));
  const l5 = data.vertebrae.find(v => v.name === 'L5');
  assert.deepEqual(l5.assist, { method: 'MedSAM', iou: 0.91, edited: true });
  assert.deepEqual(data.annotationReview.assistedBy, { method: 'MedSAM', vertebrae: 2, editedAfterAssist: 1 });

  const fresh = await openPage(fakeServer());
  fresh.context.__data = data;
  assert.equal(fresh.evaluate(`importAnnotationJson(__data, { jsonFilename: 'case.json', silent: true })`), true);
  const imported = JSON.parse(fresh.evaluate('JSON.stringify(vertebrae)'));
  assert.deepEqual(imported.find(v => v.name === 'L5').assist, { method: 'MedSAM', iou: 0.91, edited: true });
});

test('沒有 MedSAM 輔助時，annotationReview 不出現 assistedBy', async () => {
  const page = await openPage(fakeServer());
  page.context.__data = {
    version: '2.3', spineType: 'L', imageInfo: { filename: 'case.png', width: 1000, height: 2000 },
    vertebrae: [{ name: 'L3', points: vertebraPoints(500, 800) }],
  };
  page.evaluate(`importAnnotationJson(__data, { jsonFilename: 'case.json', silent: true })`);
  const data = JSON.parse(page.evaluate('JSON.stringify(buildExportData())'));
  assert.equal(data.annotationReview.assistedBy, undefined);
  assert.equal(data.vertebrae[0].assist, undefined);
});

test('修整全部：只改完整椎體，送外接框＋提示點位', async () => {
  const server = fakeServer();
  const page = await openPage(server);
  page.context.__data = {
    version: '2.3', spineType: 'L', imageInfo: { filename: 'case.png', width: 1000, height: 2000 },
    vertebrae: [
      { name: 'L4', points: vertebraPoints(500, 1100) },
      { name: 'L5', points: vertebraPoints(510, 1400) },
      { name: 'S1', boundaryType: 'upper', points: vertebraPoints(520, 1700, FULL.slice(0, 3)) },
    ],
  };
  page.evaluate(`importAnnotationJson(__data, { jsonFilename: 'case.json', silent: true })`);
  const s1Before = page.evaluate(`JSON.stringify(vertebrae.find(v => v.name === 'S1').points)`);
  assert.equal(await page.evaluate('medsamRefineAll()'), 2);
  assert.match(page.confirms.at(-1), /2 節完整椎體/);
  const segs = server.calls.filter(c => c.route === '/segment');
  assert.equal(segs.length, 2);
  const box = segs[0].body.box;
  // 300×200 的點位外接框，外擴 4% × 300 = 12
  assert.deepEqual(box, [338, 988, 662, 1212]);
  assert.deepEqual(Object.keys(segs[0].body.hint_points), FULL);
  assert.equal(page.evaluate(`JSON.stringify(vertebrae.find(v => v.name === 'S1').points)`), s1Before);
  assert.equal(page.evaluate(`vertebrae.filter(v => v.assist).length`), 2);
  assert.equal(page.evaluate('annotationDirty'), true);
});

test('修整全部：取消確認就不呼叫服務', async () => {
  const server = fakeServer();
  const page = loadHtmlScript(editorPath, { fetch: server.fetch, FormData, confirm: () => false });
  page.initialize();
  page.evaluate(`commitLoadedImage({width:1000,height:2000,naturalWidth:1000,naturalHeight:2000},{name:'case.png'},null)`);
  page.context.__data = { version: '2.3', spineType: 'L', imageInfo: { filename: 'case.png', width: 1000, height: 2000 },
    vertebrae: [{ name: 'L4', points: vertebraPoints(500, 1100) }] };
  page.evaluate(`importAnnotationJson(__data, { jsonFilename: 'case.json', silent: true })`);
  assert.equal(await page.evaluate('medsamRefineAll()'), 0);
  assert.equal(server.calls.length, 0);
});

// ── 審查後補強 ────────────────────────────────────────────────────────
test('新增時送鄰近椎體當方向參考；修正時送該節現有點位', async () => {
  const server = fakeServer();
  const page = await openPage(server);
  await page.evaluate('toggleMedsamMode(true)');
  await page.evaluate('medsamApplyBox([400, 1500, 600, 1700])');   // S1：沒有鄰節
  await page.evaluate('medsamApplyBox([380, 1200, 620, 1420])');   // L5：鄰節 = S1
  page.evaluate('selectedVertebrae.add(1)');
  await page.evaluate('medsamApplyBox([380, 1200, 620, 1420])');   // 修正 L5
  const segs = server.calls.filter(c => c.route === '/segment').map(c => c.body);
  assert.equal(segs[0].neighbor_points, null);
  assert.deepEqual(Object.keys(segs[1].neighbor_points), FULL.slice(0, 3));
  assert.equal(segs[1].hint_points, null);
  assert.deepEqual(Object.keys(segs[2].hint_points), FULL);
  assert.equal(segs[2].neighbor_points, null);
});

test('修正選取的那一節後取消選取，下一個框改為新增下一節', async () => {
  const page = await openPage(fakeServer());
  await page.evaluate('toggleMedsamMode(true)');
  await page.evaluate('medsamApplyBox([400, 1500, 600, 1700])');
  page.evaluate('selectedVertebrae.add(0)');
  await page.evaluate('medsamApplyBox([400, 1500, 600, 1700])');
  assert.equal(page.evaluate('selectedVertebrae.size'), 0);
  await page.evaluate('medsamApplyBox([380, 1200, 620, 1420])');
  assert.deepEqual(JSON.parse(page.evaluate('JSON.stringify(vertebrae.map(v => v.name))')), ['S1', 'L5']);
});

test('計算途中換了影像：結果丟棄、不寫進新影像、不跳錯誤視窗', async () => {
  let page;
  const server = fakeServer({
    onSegment: async () => {
      page.evaluate(`commitLoadedImage({width:900,height:1800,naturalWidth:900,naturalHeight:1800},{name:'other.png'},null)`);
    },
  });
  page = await openPage(server);
  await page.evaluate('toggleMedsamMode(true)');
  const alertsBefore = page.alerts.length;
  assert.equal(await page.evaluate('medsamApplyBox([400, 1500, 600, 1700])'), false);
  assert.equal(page.evaluate('vertebrae.length'), 0);
  assert.equal(page.alerts.length, alertsBefore);
  assert.match(page.document.getElementById('medsamStatus').textContent, /已捨棄/);
  // 新影像要重新上傳，不能沿用舊影像的 image_id
  await page.evaluate('medsamApplyBox([400, 1500, 600, 1700])').catch(() => {});
  assert.equal(server.calls.filter(c => c.route === '/image').length, 2);
});

test('計算中不能刪除、撤銷或清除', async () => {
  let page;
  const server = fakeServer({
    onSegment: async () => {
      page.evaluate('deleteCurrentVertebra(); undoPoint(); clearAll(); deleteSelectedVertebrae()');
    },
  });
  page = await openPage(server);
  page.context.__data = { version: '2.3', spineType: 'L', imageInfo: { filename: 'case.png', width: 1000, height: 2000 },
    vertebrae: [{ name: 'L4', points: vertebraPoints(500, 1100) }, { name: 'L5', points: vertebraPoints(510, 1400) }] };
  page.evaluate(`importAnnotationJson(__data, { jsonFilename: 'case.json', silent: true })`);
  page.evaluate('selectedVertebrae.add(0)');
  const confirmsBefore = page.confirms.length;
  assert.equal(await page.evaluate('medsamRefineAll()'), 2);
  assert.equal(page.evaluate('vertebrae.length'), 2);
  assert.equal(page.confirms.length, confirmsBefore + 1, '只有修整全部本身的確認，刪除／清除都被擋下');
  assert.equal(page.evaluate('vertebrae.filter(v => v.assist).length'), 2);
});

test('有原始檔案時上傳原始位元組，不用 canvas 重新編碼', async () => {
  const server = fakeServer();
  const page = loadHtmlScript(editorPath, { fetch: server.fetch, FormData });
  page.initialize();
  page.context.__file = new Blob(['original-png-bytes'], { type: 'image/png' });
  page.evaluate(`commitLoadedImage({width:1000,height:2000,naturalWidth:1000,naturalHeight:2000}, __file, '${'d'.repeat(64)}')`);
  await page.evaluate('toggleMedsamMode(true)');
  assert.equal(await page.evaluate('medsamApplyBox([400, 1500, 600, 1700])'), true);
  const form = server.calls.find(c => c.route === '/image').body;
  assert.equal(await form.get('file').text(), 'original-png-bytes');
  assert.equal(form.get('sha256'), 'd'.repeat(64));
});

// ── 瀏覽器內推論（不需要 Python）────────────────────────────────────────
// 假 ort：遮罩 = 提示框，用來檢查標註工具的瀏覽器分支（真正的模型在 Chrome 端對端測試）
function fakeBrowserModel(page, { holdoutFiles = {}, holdoutPixels = {} } = {}) {
  const runs = [];
  page.context.__fakeModel = {
    holdoutFiles, holdoutPixels, runs,
    make(ctxTensor) {
      const S = 512, L = 128;
      return {
        ort: { Tensor: ctxTensor }, side: S, context: 2.2, boxMargin: 0.04, normalize: 'imagenet', device: 'fake',
        holdoutFiles, holdoutPixels,
        encoder: { run: async () => ({ image_embeddings: new ctxTensor('float32', new Float32Array(1), [1]) }) },
        decoder: { run: async ({ input_boxes }) => {
          const b = Array.from(input_boxes.data);
          runs.push(b);
          const logits = new Float32Array(L * L);
          for (let y = 0; y < L; y++) for (let x = 0; x < L; x++) {
            const cx = (x + 0.5) * 4 - 0.5, cy = (y + 0.5) * 4 - 0.5;
            logits[y * L + x] = (cx >= b[0] && cx <= b[2] && cy >= b[1] && cy <= b[3]) ? 10 : -10;
          }
          return { low_res_masks: { data: logits }, iou_scores: { data: Float32Array.of(0.88) } };
        } },
      };
    },
  };
  page.evaluate(`medsamWeb = __fakeModel.make(class { constructor(t, d, s) { this.type = t; this.data = d; this.dims = s; } })`);
  return runs;
}

test('瀏覽器模式：不呼叫服務，拉框產生 S1、L5 並記錄 assist', async () => {
  const server = fakeServer();
  const page = await openPage(server);
  const runs = fakeBrowserModel(page);
  assert.equal(await page.evaluate('toggleMedsamMode(true)'), true);
  assert.equal(page.evaluate('medsamBackend'), 'browser');
  assert.match(page.document.getElementById('medsamStatus').textContent, /瀏覽器模式/);
  assert.equal(await page.evaluate('medsamApplyBox([400, 1500, 600, 1700])'), true);
  assert.equal(await page.evaluate('medsamApplyBox([380, 1200, 620, 1420])'), true);
  const verts = JSON.parse(page.evaluate('JSON.stringify(vertebrae)'));
  assert.deepEqual(verts.map(v => v.name), ['S1', 'L5']);
  assert.equal(verts[0].points.length, 3);
  assert.equal(verts[1].points.length, 6);
  // 角點回到框角（遮罩 = 框）：L5 前上角 ≈ (380, 1200)
  assert.ok(Math.hypot(verts[1].points[0].x - 380, verts[1].points[0].y - 1200) < 8, JSON.stringify(verts[1].points[0]));
  assert.equal(verts[1].assist.method, 'MedSAM');
  assert.equal(runs.length, 2);
  assert.equal(server.calls.filter(c => c.route !== '/health').length, 0, '瀏覽器模式不能上傳影像到服務');
});

test('瀏覽器模式：外部測試集依檔案雜湊拒絕', async () => {
  const page = await openPage(fakeServer());
  fakeBrowserModel(page, { holdoutFiles: { ['c'.repeat(64)]: 'holdout_t' } });
  await page.evaluate('toggleMedsamMode(true)');
  assert.equal(await page.evaluate('medsamApplyBox([400, 1500, 600, 1700])'), false);
  assert.equal(page.evaluate('vertebrae.length'), 0);
  assert.match(page.alerts.at(-1), /外部測試集（holdout_t）/);
});

test('瀏覽器模式：外部測試集依像素指紋拒絕（沒有原始檔雜湊時）', async () => {
  const page = await openPage(fakeServer());
  page.evaluate(`commitLoadedImage({width:40,height:30,naturalWidth:40,naturalHeight:30},{name:'pasted.png'},null)`);
  // harness 的 getImageData 全為 0：指紋 = SHA-256("30x40:" + 1200 個 0)
  const { createHash } = require('node:crypto');
  const fp = createHash('sha256').update(Buffer.concat([Buffer.from('30x40:'), Buffer.alloc(1200)])).digest('hex');
  fakeBrowserModel(page, { holdoutPixels: { [fp]: 'holdout_p' } });
  await page.evaluate('toggleMedsamMode(true)');
  assert.equal(await page.evaluate('medsamApplyBox([5, 5, 30, 25])'), false);
  assert.match(page.alerts.at(-1), /外部測試集（holdout_p）/);
});

test('瀏覽器模型資料夾缺少外部測試集清單：不啟用', async () => {
  const page = await openPage(fakeServer({ health: false }));
  const file = (name, text) => ({ name, text: async () => text, arrayBuffer: async () => new ArrayBuffer(4) });
  page.context.__files = [file('meta.json', '{"preprocess":{"side":512,"context":2.2,"box_margin":0.04}}'),
                          file('decoder.onnx', ''), file('encoder_fp16.onnx', ''), file('encoder.onnx', '')];
  assert.equal(await page.evaluate('medsamLoadBrowserModel(__files)'), false);
  assert.match(page.alerts.at(-1), /holdout_hashes\.json/);
  assert.equal(page.evaluate('medsamWeb'), null);
});

test('瀏覽器模式：修整全部不檢查本機服務，直接用瀏覽器模型', async () => {
  const server = fakeServer({ health: false });
  const page = await openPage(server);
  const runs = fakeBrowserModel(page);
  page.context.__data = { version: '2.3', spineType: 'L', imageInfo: { filename: 'case.png', width: 1000, height: 2000 },
    vertebrae: [{ name: 'L4', points: vertebraPoints(500, 1100) }, { name: 'L5', points: vertebraPoints(510, 1400) }] };
  page.evaluate(`importAnnotationJson(__data, { jsonFilename: 'case.json', silent: true })`);
  assert.equal(await page.evaluate('medsamRefineAll()'), 2);
  assert.equal(runs.length, 2);
  assert.equal(page.evaluate('medsamBackend'), 'browser');
  assert.ok(!page.alerts.some(a => /還沒準備好/.test(a)), page.alerts.join(' | '));
});

// ── 瀏覽器模型載入的防護（審查後補強）────────────────────────────────────
function modelFolder({ holdout, metaHoldout, encoderBytes = 'enc', recordedEncoder = 'enc' }) {
  const { createHash } = require('node:crypto');
  const sha = s => createHash('sha256').update(Buffer.from(s)).digest('hex');
  const holdText = JSON.stringify(holdout);
  const meta = {
    normalize: 'imagenet', preprocess: { side: 512, context: 2.2, box_margin: 0.04 },
    holdout: metaHoldout || { files: Object.keys(holdout.files).length, pixels: Object.keys(holdout.pixels).length,
                              sha256: sha(holdText), created: '2026-11-24T00:00:00' },
    files: { encoder_fp16: { name: 'encoder_fp16.onnx', sha256: sha(recordedEncoder) }, decoder: { name: 'decoder.onnx', sha256: sha('dec') } },
  };
  const file = (name, text) => ({ name, text: async () => text, arrayBuffer: async () => new Uint8Array(Buffer.from(text)).buffer });
  return [file('meta.json', JSON.stringify(meta)), file('holdout_hashes.json', holdText),
          file('encoder_fp16.onnx', encoderBytes), file('decoder.onnx', 'dec')];
}

function fakeOrt(page, { failWebgpu = false } = {}) {
  const created = [];
  page.context.navigator.gpu = { requestAdapter: async () => ({}) };
  page.context.ort = {
    env: { wasm: {}, webgpu: {} },
    Tensor: class { constructor(t, d, s) { this.type = t; this.data = d; this.dims = s; } },
    InferenceSession: { create: async (_bytes, opts) => {
      const ep = opts.executionProviders[0];
      if (failWebgpu && ep === 'webgpu') throw new Error('shader-f16 not supported');
      const s = { ep, released: false, run: async () => ({}), release: async () => { s.released = true; } };
      created.push(s);
      return s;
    } },
  };
  return created;
}

const HOLD = { created: '2026-11-24T00:00:00', files: { ['a'.repeat(64)]: 'holdout_t' }, pixels: { ['b'.repeat(64)]: 'holdout_t' } };

test('瀏覽器模型：清單、meta 與模型檔一致才載入，顯示外部測試集筆數', async () => {
  const page = await openPage(fakeServer({ health: false }));
  const created = fakeOrt(page);
  page.context.__files = modelFolder({ holdout: HOLD });
  assert.equal(await page.evaluate('medsamLoadBrowserModel(__files)'), true, page.alerts.join(' | '));
  assert.equal(page.evaluate('medsamBackend'), 'browser');
  assert.match(page.evaluate('medsamWeb.device'), /WebGPU.*外部測試集 1 張（清單 2026-11-24）/);
  assert.deepEqual(created.map(s => s.ep), ['webgpu', 'webgpu']);
  // 再載入一次：舊的 session 要先釋放
  page.context.__files = modelFolder({ holdout: HOLD });
  assert.equal(await page.evaluate('medsamLoadBrowserModel(__files)'), true);
  assert.equal(created.slice(0, 2).every(s => s.released), true);
});

test('瀏覽器模型：WebGPU 建立失敗時改用 CPU', async () => {
  const page = await openPage(fakeServer({ health: false }));
  const created = fakeOrt(page, { failWebgpu: true });
  page.context.__files = modelFolder({ holdout: HOLD });
  assert.equal(await page.evaluate('medsamLoadBrowserModel(__files)'), true, page.alerts.join(' | '));
  assert.match(page.evaluate('medsamWeb.device'), /CPU/);
  assert.deepEqual(created.map(s => s.ep), ['wasm', 'wasm']);
});

for (const [label, opts, pattern] of [
  ['外部測試集清單是空的', { holdout: { created: 'x', files: {}, pixels: {} } }, /空的/],
  ['清單和 meta.json 不是同一次匯出', { holdout: HOLD, metaHoldout: { files: 1, pixels: 1, sha256: '0'.repeat(64) } }, /不是同一次匯出/],
  ['encoder 不是 meta.json 記錄的那一個', { holdout: HOLD, encoderBytes: 'old-encoder' }, /encoder_fp16\.onnx 與 meta\.json 記錄的不同/],
]) {
  test(`瀏覽器模型：${label}就不啟用`, async () => {
    const page = await openPage(fakeServer({ health: false }));
    fakeOrt(page);
    page.context.__files = modelFolder(opts);
    assert.equal(await page.evaluate('medsamLoadBrowserModel(__files)'), false);
    assert.match(page.alerts.at(-1), pattern);
    assert.equal(page.evaluate('medsamWeb'), null);
  });
}
