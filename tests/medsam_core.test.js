'use strict';

// 標註工具內 MedsamCore（瀏覽器推論）與 Python／OpenCV／torch 的結果比對。
// 標準答案由 python tests/make_medsam_core_fixtures.py 產生。
const assert = require('node:assert/strict');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');
const { extractScriptById } = require('./html_script_harness');

const editorPath = path.join(__dirname, '..', 'spinal-annotation-web.html');
const cases = require('./fixtures/medsam_core_cases.json');
const ctx = { module: { exports: {} } };
vm.createContext(ctx);
vm.runInContext(extractScriptById(editorPath, 'medsam-core'), ctx);
const C = ctx.module.exports;

function unrle(runs, n) {
  const out = new Uint8Array(n);
  let p = 0, v = 0;
  for (const r of runs) { if (v) out.fill(1, p, p + r); p += r; v ^= 1; }
  assert.equal(p, n);
  return out;
}
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b}`);
const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);

test('裁切仿射矩陣與座標轉換 = refine_stage2.crop_matrix', () => {
  for (const c of cases.affine) {
    const m = C.cropMatrix(c.center[0], c.center[1], c.side, c.crop);
    m.forEach((v, i) => close(v, c.matrix[i], 1e-9, `m[${i}]`));
    c.points.forEach((p, i) => {
      const q = C.applyAffine(m, p[0], p[1]);
      close(q[0], c.mapped[i][0], 1e-9, 'x'); close(q[1], c.mapped[i][1], 1e-9, 'y');
      const back = C.applyAffine(C.invertAffine(m), q[0], q[1]);
      close(back[0], p[0], 1e-6, 'inv x'); close(back[1], p[1], 1e-6, 'inv y');
    });
  }
});

test('warpGray = cv2.warpAffine(INTER_LINEAR, 邊界 0)，誤差 ≤ 1 灰階', () => {
  for (const c of cases.warp) {
    const out = C.warpGray(Uint8Array.from(c.gray), c.w, c.h, c.matrix, c.S);
    let maxDiff = 0;
    out.forEach((v, i) => { maxDiff = Math.max(maxDiff, Math.abs(v - c.out[i])); });
    assert.ok(maxDiff <= 1, `最大差 ${maxDiff}`);
  }
});

test('upsampleMask = torch bilinear(align_corners=False) > 0', () => {
  for (const c of cases.upsample) {
    const mask = C.upsampleMask(Float32Array.from(c.logits), c.L, c.S);
    const ref = unrle(c.mask_rle, c.S * c.S);
    let diff = 0;
    mask.forEach((v, i) => { if (v !== ref[i]) diff++; });
    assert.ok(diff <= c.S * c.S * 1e-4, `不同像素 ${diff}`);    // 只容許 0 附近的浮點誤差
  }
});

test('preprocess = medsam_refine.preprocess（imagenet、minmax）', () => {
  const c = cases.preprocess;
  for (const mode of ['imagenet', 'minmax']) {
    const out = C.preprocess(Uint8Array.from(c.patch), c.S, mode);
    out.forEach((v, i) => close(v, c[mode][i], 1e-5, `${mode}[${i}]`));
  }
});

test('local_axes：完整椎體、只有上終板、只有下終板、四角', () => {
  for (const c of cases.local_axes) {
    const [u, v] = C.localAxes(c.points);
    close(u[0], c.axes[0][0], 1e-9, 'u0'); close(u[1], c.axes[0][1], 1e-9, 'u1');
    close(v[0], c.axes[1][0], 1e-9, 'v0'); close(v[1], c.axes[1][1], 1e-9, 'v1');
  }
});

test('最大區塊、外輪廓 = cv2.connectedComponents + findContours', () => {
  for (const [i, c] of cases.masks.entries()) {
    const lc = C.largestComponent(unrle(c.mask_rle, c.S * c.S), c.S, c.S);
    assert.equal(lc.area, c.area, `case ${i} 面積`);
    const contour = C.traceContour(lc.comp, c.S, c.S);
    const set = new Set(contour.map(p => p.join(',')));
    const ref = new Set(c.contour_set.map(p => p.join(',')));
    assert.deepEqual([...set].sort(), [...ref].sort(), `case ${i} 輪廓點集合`);
    assert.equal(contour.length, c.contour_len, `case ${i} 輪廓長度`);
  }
});

test('axes_from_mask（含鄰近椎體方向）與 mask_to_points 誤差 ≤ 1 px', () => {
  for (const [i, c] of cases.masks.entries()) {
    const lc = C.largestComponent(unrle(c.mask_rle, c.S * c.S), c.S, c.S);
    const contour = C.traceContour(lc.comp, c.S, c.S);
    const plain = C.axesFromContour(contour);
    const pref = C.axesFromContour(contour, c.prefer_v);
    for (const [got, want, name] of [[plain, c.axes_plain, 'plain'], [pref, c.axes_pref, 'prefer']]) {
      assert.ok(got[0][0] * want[0][0] + got[0][1] * want[0][1] > 0.999, `case ${i} ${name} u ${got[0]} vs ${want[0]}`);
      assert.ok(got[1][0] * want[1][0] + got[1][1] * want[1][1] > 0.999, `case ${i} ${name} v ${got[1]} vs ${want[1]}`);
    }
    for (const [axes, want, name] of [[c.axes_points, c.points_with_point_axes, 'point'],
                                      [plain, c.points_with_mask_axes, 'mask'],
                                      [pref, c.points_with_prefer_axes, 'prefer']]) {
      const got = C.maskToPoints(contour, lc.area, axes);
      assert.ok(got, `case ${i} ${name} 求點失敗`);
      for (const k of C.POINTS) assert.ok(dist(got[k], want[k]) <= 1.0, `case ${i} ${name} ${k}: ${got[k]} vs ${want[k]}`);
    }
  }
});

test('segment：假 ort 讓遮罩 = 提示框，角點回到原圖框角', async () => {
  const S = 512, L = 128;
  class Tensor { constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; } }
  let box;
  const model = {
    ort: { Tensor }, side: S, normalize: 'imagenet', context: 2.2, boxMargin: 0.04,
    encoder: { run: async ({ pixel_values }) => { assert.deepEqual(Array.from(pixel_values.dims), [1, 3, S, S]); return { image_embeddings: new Tensor('float32', new Float32Array(1), [1]) }; } },
    decoder: { run: async ({ input_boxes }) => {
      box = Array.from(input_boxes.data);
      const logits = new Float32Array(L * L);
      for (let y = 0; y < L; y++) for (let x = 0; x < L; x++) {
        const cx = (x + 0.5) * 4 - 0.5, cy = (y + 0.5) * 4 - 0.5;
        logits[y * L + x] = (cx >= box[0] && cx <= box[2] && cy >= box[1] && cy <= box[3]) ? 10 : -10;
      }
      return { low_res_masks: new Tensor('float32', logits, [1, L, L]), iou_scores: new Tensor('float32', Float32Array.of(0.9), [1]) };
    } },
  };
  const w = 2500, h = 3000;
  const res = await C.segment(model, new Uint8Array(w * h), w, h, [1000, 1500, 1360, 1760]);
  close(res.iou, 0.9, 1e-6, 'iou');
  const expect = { anteriorSuperior: [1000, 1500], posteriorSuperior: [1360, 1500], posteriorInferior: [1360, 1760], anteriorInferior: [1000, 1760] };
  for (const [k, p] of Object.entries(expect)) assert.ok(dist([res.points[k].x, res.points[k].y], p) < 8, `${k}: ${JSON.stringify(res.points[k])}`);
});

test('boxGeometry 反推訓練時的框外擴', () => {
  const g = C.boxGeometry([338, 988, 662, 1212], 0.04);
  close(g.cx, 500, 1e-9, 'cx'); close(g.cy, 1100, 1e-9, 'cy'); close(g.size, 324 / 1.08, 1e-9, 'size');
  assert.throws(() => C.boxGeometry([10, 10, 5, 20], 0.04));
});

test('segment：模型輸出尺寸與 side 不符時報錯，不默默產生錯的點', async () => {
  class Tensor { constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; } }
  const model = {
    ort: { Tensor }, side: 512, normalize: 'imagenet', context: 2.2, boxMargin: 0.04,
    encoder: { run: async () => ({ image_embeddings: new Tensor('float32', new Float32Array(1), [1]) }) },
    decoder: { run: async () => ({ low_res_masks: new Tensor('float32', new Float32Array(256 * 256), [1, 256, 256]),
                                   iou_scores: new Tensor('float32', Float32Array.of(0.9), [1]) }) },
  };
  await assert.rejects(C.segment(model, new Uint8Array(100 * 100), 100, 100, [10, 10, 60, 60]), /不符/);
});
