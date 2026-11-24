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
