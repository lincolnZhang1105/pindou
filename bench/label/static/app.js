'use strict';
/*
 * 豆伴 · 真值标注工具（bench/label，T0.4）。原生 JS，不需要构建。
 *
 * 坐标约定：x_lines / y_lines 是原图像素坐标（像素 i 占 [i, i+1)，方向已按 EXIF 校正），
 * 画布用 view = {s, tx, ty} 缩放平移：屏幕坐标 = 图片坐标 × s + t。
 * 格子 cells[row][col] 是整张图纸（全局）坐标；分页图每页用 col_offset / row_offset 映射到全局。
 * 色号统一按 normCode 规范化（与 bench/common/formats.py 的 code_key 一致：A01 → A1，zg:zg01 → ZG:ZG1）。
 */

const KNOWN_TAGS = ['similar_colors', 'single_pixel_detail', 'gradient_skin', 'white_vs_empty', 'watermark', 'blurry'];
const TYPES = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7'];
const USER_KEY = 'pindou.label.user';
const UNDO_LIMIT = 200;

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const S = {
  tasks: [], key: null, task: null, imgs: [], d: null, pi: 0,
  view: { s: 1, tx: 0, ty: 0 },
  tool: 'paint', prevTool: 'paint', brush: null, recent: [],
  alpha: 0.45, showText: true, showGrid: true, highlight: false,
  undo: [], redo: [],
  counts: new Map(), flags: new Map(), flagKey: null,
  dirty: false, saveTimer: null, saving: null,
  palettes: [], palMap: null, colorCache: new Map(),
  sampleTarget: null, legendSel: -1, legendPending: false,
  drag: null, hover: null, spaceDown: false,
  rafPending: false, needLegend: false,
  dpr: window.devicePixelRatio || 1, cw: 0, ch: 0, needFit: false,
};

// --------------------------------------------------------------------------- 小工具

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function normCode(s) {
  if (s === null || s === undefined) return null;
  s = String(s).replace(/\s+/g, '').toUpperCase();
  if (!s) return null;
  let pal = null;
  const i = s.indexOf(':');
  if (i >= 0) { pal = s.slice(0, i); s = s.slice(i + 1); }
  const m = /^([A-Z]+)0*(\d+)$/.exec(s);
  if (m) s = m[1] + String(parseInt(m[2], 10));
  return pal ? `${pal}:${s}` : s;
}

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const r2 = (v) => Math.round(v * 100) / 100;
const clone = (o) => JSON.parse(JSON.stringify(o));
const nowText = () => new Date().toLocaleTimeString();

function linspace(a, b, n) {
  const out = [];
  for (let i = 0; i <= n; i++) out.push(r2(a + (b - a) * i / n));
  return out;
}

function range(n) { return Array.from({ length: n }, (_, i) => i); }

function hashColor(code) {
  let h = 0;
  for (const ch of code) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  const hue = h % 360;
  // HSL → RGB（饱和度 0.6，亮度 0.55）
  const s = 0.6, l = 0.55, k = (n) => (n + hue / 30) % 12, a = s * Math.min(l, 1 - l);
  const f = (n) => l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
  return [f(0), f(8), f(4)].map((v) => Math.round(v * 255));
}

function luminance(rgb) { return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255; }

// lines[i] <= v < lines[i+1] 的 i；v 在范围外返回 -1
function findIdx(lines, v) {
  const n = lines.length - 1;
  if (n < 1 || v < lines[0] || v >= lines[n]) return -1;
  let lo = 0, hi = n;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (lines[mid] <= v) lo = mid; else hi = mid; }
  return lo;
}

function clampIdx(lines, v) {
  const n = lines.length - 1;
  if (v < lines[0]) return 0;
  if (v >= lines[n]) return n - 1;
  return findIdx(lines, v);
}

function nearestLine(lines, v) {
  const i = clampIdx(lines, v);
  return Math.abs(lines[i] - v) <= Math.abs(lines[i + 1] - v) ? i : i + 1;
}

// --------------------------------------------------------------------------- 服务端

async function api(method, url, body) {
  const opt = { method, headers: {} };
  if (body !== undefined) {
    opt.headers['Content-Type'] = 'application/json';
    opt.body = JSON.stringify(body);
  }
  const r = await fetch(url, opt);
  let data;
  try { data = await r.json(); } catch (e) { data = { error: `HTTP ${r.status}` }; }
  if (!r.ok && !(data && 'ok' in data)) throw new Error((data && data.error) || `HTTP ${r.status}`);
  return data;
}

function msg(text, isErr) {
  const el = $('#st-msg');
  el.textContent = text || '';
  el.classList.toggle('err', !!isErr);
}

function dialog(html) {
  $('#dialog-body').innerHTML = html;
  $('#dialog').classList.remove('hidden');
}

function listHtml(items, cls) {
  return items && items.length ? `<ul class="${cls}">${items.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>` : '';
}

// --------------------------------------------------------------------------- 任务列表

async function loadTasks() {
  const r = await api('GET', '/api/tasks');
  S.tasks = r.tasks;
  renderTaskList();
}

function renderTaskList() {
  const fs = $('#f-split').value, fst = $('#f-status').value, q = $('#f-search').value.trim().toLowerCase();
  const rows = S.tasks.filter((t) => (!fs || (fs === '-' ? !t.split : t.split === fs))
    && (!fst || t.status.code === fst)
    && (!q || t.key.toLowerCase().includes(q) || t.image_ids.some((i) => i.toLowerCase().includes(q))));
  $('#task-table tbody').innerHTML = rows.map((t) => `
    <tr data-key="${esc(t.key)}" class="${t.key === S.key ? 'current' : ''}">
      <td title="${esc(t.image_ids.join(', '))}">${esc(t.key)}${t.n_images > 1 ? ` <span class="muted">×${t.n_images}</span>` : ''}</td>
      <td>${esc(t.split || '-')}</td><td>${esc(t.type || '')}</td><td>${esc(t.size_bucket || '')}</td>
      <td><span class="st st-${esc(t.status.code)}" title="${esc(t.status.text)}">${esc(t.status.text)}</span>${t.needs_review ? ' <span class="muted" title="XL 档或分页图，需要第二人复核">复</span>' : ''}</td>
    </tr>`).join('');
  $('#task-count').textContent = `${rows.length} / ${S.tasks.length} 个任务`;
}

function updateTaskStatus(key, status) {
  const t = S.tasks.find((x) => x.key === key);
  if (t && status) { t.status = status; renderTaskList(); }
}

// --------------------------------------------------------------------------- 打开任务

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error(`图片加载失败：${url}`));
    im.src = url;
  });
}

async function openTask(key) {
  if (!key) return;
  if (S.dirty) await saveDraft();
  let t;
  try {
    t = await api('GET', `/api/task/${encodeURIComponent(key)}`);
  } catch (e) {
    msg(e.message, true);
    return;
  }
  S.key = key;
  S.task = t;
  const errs = [];
  S.imgs = await Promise.all(t.images.map((im) => (im.readable
    ? loadImage(im.url).catch((e) => { errs.push(e.message); return null; })
    : (errs.push(`${im.image_id}：${im.format} 还没转换成 PNG（运行 python -m bench.collect convert）`), null))));
  S.d = normalizeDraft(t.draft, t);
  S.pi = 0;
  S.undo = []; S.redo = [];
  S.flags = new Map((S.d.flags || []).map(([r, c, why]) => [r * S.d.grid.cols + c, why]));
  S.flagKey = null;
  S.recent = (S.d.recent || []).filter(Boolean).slice(0, 9);
  S.sampleTarget = null; S.legendSel = -1;
  S.dirty = false;
  if (S.d.annotator === undefined) S.d.annotator = null;
  setPalette(S.d.palette_hint);
  recount();
  location.hash = encodeURIComponent(key);
  renderTaskList();
  refreshAll();
  fitView(true);
  setSaveStatus(t.draft_source === 'draft' ? `草稿 rev ${t.draft.rev || 0}` : (t.draft_source === 'label' ? '从已导出的真值载入' : '新任务'));
  if (errs.length) dialog(`<p class="err">图片加载问题：</p>${listHtml(errs, 'err')}`);
  loadPredRuns();
  if (!S.d.grid.cols) { switchTab('grid'); msg('先在「网格」页填写行列数'); } else msg('');
}

function makeCells(rows, cols, old) {
  const out = [];
  for (let r = 0; r < rows; r++) {
    const row = new Array(cols).fill(null);
    if (old && old[r]) for (let c = 0; c < Math.min(cols, old[r].length); c++) row[c] = normCode(old[r][c]);
    out.push(row);
  }
  return out;
}

function defaultBBox(ii) {
  const img = S.imgs[ii];
  if (!img) return [0, 0, 100, 100];
  const w = img.naturalWidth, h = img.naturalHeight;
  return [r2(w * 0.05), r2(h * 0.05), r2(w * 0.95), r2(h * 0.95)];
}

function bboxOf(p) {
  if (p.x_lines.length >= 2 && p.y_lines.length >= 2) return [p.x_lines[0], p.y_lines[0], p.x_lines[p.x_lines.length - 1], p.y_lines[p.y_lines.length - 1]];
  return defaultBBox(p.image_index);
}

function setUniform(p, bb) {
  p.x_lines = linspace(bb[0], bb[2], p.cols);
  p.y_lines = linspace(bb[1], bb[3], p.rows);
  p.x_anchors = [0, p.cols];
  p.y_anchors = [0, p.rows];
}

function normalizePage(p) {
  const q = {
    image_index: p.image_index | 0, cols: p.cols | 0, rows: p.rows | 0,
    col_offset: p.col_offset | 0, row_offset: p.row_offset | 0,
    x_lines: (p.x_lines || []).map(Number), y_lines: (p.y_lines || []).map(Number),
  };
  if (q.cols < 1 || q.rows < 1) { q.cols = Math.max(1, q.cols); q.rows = Math.max(1, q.rows); }
  if (q.x_lines.length !== q.cols + 1 || q.y_lines.length !== q.rows + 1) {
    setUniform(q, defaultBBox(q.image_index));
  } else {
    const okA = (a, n) => Array.isArray(a) && a.length >= 2 && a.every((v) => Number.isInteger(v) && v >= 0 && v <= n);
    // 外部来的网格线（预测结果、已导出的真值）默认每条都是锚点：拖动时只动这一条
    q.x_anchors = okA(p.x_anchors, q.cols) ? [...new Set([0, q.cols, ...p.x_anchors])].sort((a, b) => a - b) : range(q.cols + 1);
    q.y_anchors = okA(p.y_anchors, q.rows) ? [...new Set([0, q.rows, ...p.y_anchors])].sort((a, b) => a - b) : range(q.rows + 1);
  }
  return q;
}

function newPage(ii, cols, rows) {
  const p = { image_index: ii, cols, rows, col_offset: 0, row_offset: 0, x_lines: [], y_lines: [] };
  setUniform(p, defaultBBox(ii));
  return p;
}

function normalizeDraft(d, t) {
  if (!d) {
    const g = t.meta.grid || { cols: 0, rows: 0 };
    d = {
      type: t.meta.type || 'T1', palette_hint: t.meta.palette_hint || '', tags: [...(t.meta.tags || [])], notes: '',
      grid: { cols: g.cols | 0, rows: g.rows | 0 }, pages: [], cells: [], legend: [], empty_sample: null,
      no_legend: false, flags: [],
    };
    if (d.grid.cols > 0 && d.grid.rows > 0) d.pages = [newPage(0, d.grid.cols, d.grid.rows)];
  } else {
    d = clone(d);
  }
  d.grid = { cols: (d.grid && d.grid.cols) | 0, rows: (d.grid && d.grid.rows) | 0 };
  d.cells = makeCells(d.grid.rows, d.grid.cols, d.cells);
  d.pages = (d.pages || []).map(normalizePage);
  d.legend = (d.legend || []).map((r) => ({
    code: normCode(r.code) || '', count: (r.count === null || r.count === undefined || r.count === '') ? null : Number(r.count),
    rgb: r.rgb || null, sample: r.sample || null,
  }));
  d.tags = d.tags || [];
  d.flags = d.flags || [];
  d.no_legend = !!d.no_legend;
  d.empty_sample = d.empty_sample || null;
  d.type = d.type || t.meta.type || 'T1';
  d.palette_hint = d.palette_hint || t.meta.palette_hint || '';
  d.notes = d.notes || '';
  return d;
}

function curPage() { return S.d && S.d.pages[S.pi] ? S.d.pages[S.pi] : null; }

// --------------------------------------------------------------------------- 颜色和色卡

function setPalette(pid) {
  S.palMap = null;
  const pal = S.palettes.find((p) => p.id.toUpperCase() === String(pid || '').toUpperCase());
  if (pal) S.palMap = new Map(pal.colors.map(([code, r, g, b]) => [normCode(code), [r, g, b]]));
  S.colorCache.clear();
  updateCodeList();
}

function updateCodeList() {
  const codes = new Set();
  if (S.palMap) for (const c of S.palMap.keys()) codes.add(c);
  if (S.d) for (const r of S.d.legend) if (r.code) codes.add(r.code);
  $('#code-list').innerHTML = [...codes].map((c) => `<option value="${esc(c)}">`).join('');
}

function colorOf(code) {
  let c = S.colorCache.get(code);
  if (c) return c;
  let rgb = null;
  if (S.d) {
    const row = S.d.legend.find((r) => r.code === code && r.rgb);
    if (row) rgb = row.rgb;
  }
  if (!rgb && S.palMap) rgb = S.palMap.get(code) || S.palMap.get(code.includes(':') ? code.split(':')[1] : code) || null;
  if (!rgb) rgb = hashColor(code);
  c = { css: `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`, rgb, light: luminance(rgb) > 0.55 };
  S.colorCache.set(code, c);
  return c;
}

// --------------------------------------------------------------------------- 格子编辑

function recount() {
  S.counts = new Map();
  if (!S.d) return;
  for (const row of S.d.cells) for (const v of row) if (v !== null) S.counts.set(v, (S.counts.get(v) || 0) + 1);
  S.needLegend = true;
}

function bump(code, delta) {
  if (code === null) return;
  const n = (S.counts.get(code) || 0) + delta;
  if (n > 0) S.counts.set(code, n); else S.counts.delete(code);
}

function setCell(r, c, v, rec) {
  const row = S.d.cells[r];
  const old = row[c];
  if (old === v) return false;
  const k = r * S.d.grid.cols + c;
  if (rec && !rec.has(k)) rec.set(k, old);
  row[c] = v;
  bump(old, -1);
  bump(v, 1);
  S.flags.delete(k);
  S.needLegend = true;
  return true;
}

function commitStroke(rec) {
  if (!rec || !rec.size) return;
  const cols = S.d.grid.cols;
  const ch = [];
  for (const [k, old] of rec) ch.push([k, old, S.d.cells[Math.floor(k / cols)][k % cols]]);
  pushUndo({ kind: 'cells', ch });
}

// 图片坐标 → 当前页的格子（全局坐标）
function cellAt(x, y) {
  const p = curPage();
  if (!p) return null;
  const c = findIdx(p.x_lines, x), r = findIdx(p.y_lines, y);
  if (c < 0 || r < 0) return null;
  const gr = r + p.row_offset, gc = c + p.col_offset;
  if (gr < 0 || gc < 0 || gr >= S.d.grid.rows || gc >= S.d.grid.cols) return null;
  return { r, c, gr, gc };
}

function paintAt(x, y, rec) {
  const cell = cellAt(x, y);
  if (cell && setCell(cell.gr, cell.gc, S.brush, rec)) requestRender();
}

function paintLine(x0, y0, x1, y1, rec) {
  const p = curPage();
  if (!p) return;
  const cw = Math.max(1, Math.min((p.x_lines[p.cols] - p.x_lines[0]) / p.cols, (p.y_lines[p.rows] - p.y_lines[0]) / p.rows));
  const steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / (cw * 0.4)));
  for (let i = 1; i <= steps; i++) paintAt(x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps, rec);
}

function fillAt(x, y) {
  const cell = cellAt(x, y);
  if (!cell) return;
  const { rows, cols } = S.d.grid;
  const target = S.d.cells[cell.gr][cell.gc];
  if (target === S.brush) return;
  const rec = new Map();
  const seen = new Uint8Array(rows * cols);
  const stack = [cell.gr * cols + cell.gc];
  seen[stack[0]] = 1;
  while (stack.length) {
    const k = stack.pop();
    const r = Math.floor(k / cols), c = k % cols;
    setCell(r, c, S.brush, rec);
    for (const [dr, dc] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      const nr = r + dr, nc = c + dc;
      if (nr < 0 || nc < 0 || nr >= rows || nc >= cols) continue;
      const nk = nr * cols + nc;
      if (!seen[nk] && S.d.cells[nr][nc] === target) { seen[nk] = 1; stack.push(nk); }
    }
  }
  commitStroke(rec);
  msg(`填充了 ${rec.size} 格`);
  requestRender();
}

function pickAt(x, y) {
  const cell = cellAt(x, y);
  if (!cell) return;
  setBrush(S.d.cells[cell.gr][cell.gc]);
  msg(`吸管：${S.brush === null ? '空格' : S.brush}`);
}

async function sampleAt(x, y) {
  const p = curPage();
  const ii = p ? p.image_index : 0;
  let target = S.sampleTarget;
  if (target === null) {
    if (S.legendSel >= 0) target = S.legendSel;
    else { msg('先在「图例」页点某一行的「取色」，或点「空格颜色」的取色', true); return; }
  }
  const row = target === 'empty' ? null : S.d.legend[target];
  if (target !== 'empty' && !row) return;
  // 先同步推进到下一个要取色的行（请求返回之前用户可能已经点了下一个色块）
  if (row) row._pending = true;
  let next = -1;
  if (row) {
    for (let i = 1; i <= S.d.legend.length; i++) {
      const j = (target + i) % S.d.legend.length;
      const r = S.d.legend[j];
      if (!r.rgb && !r._pending) { next = j; break; }
    }
  }
  if (next >= 0) {
    S.sampleTarget = next;
    S.legendSel = next;
  } else {
    S.sampleTarget = null;
    setTool(S.prevTool === 'sample' ? 'paint' : S.prevTool);
  }
  S.needLegend = true;
  let res;
  try {
    res = await api('GET', `/api/sample/${encodeURIComponent(S.key)}?image_index=${ii}&x=${x.toFixed(2)}&y=${y.toFixed(2)}&r=4`);
  } catch (e) {
    if (row) delete row._pending;
    msg(e.message, true);
    return;
  }
  const sample = { image_index: ii, x: r2(x), y: r2(y) };
  if (target === 'empty') {
    S.d.empty_sample = { rgb: res.rgb, sample };
    msg(`空格颜色 rgb(${res.rgb.join(', ')})`);
  } else {
    delete row._pending;
    row.rgb = res.rgb;
    row.sample = sample;
    const nx = next >= 0 ? S.d.legend[next] : null;
    msg(`${row.code || '（未填色号）'} 取色 rgb(${res.rgb.join(', ')})${nx ? `，接着点 ${nx.code || '下一行'} 的色块` : '，全部取完'}`);
  }
  S.colorCache.clear();
  S.needLegend = true;
  renderEmptySwatch();
  renderBrush();
  markDirty();
  requestRender();
}

function setBrush(code) {
  S.brush = code === undefined ? null : code;
  // 已在「最近」里的色号保持原位，数字键 1–9 对应的色号才稳定
  if (S.brush !== null && !S.recent.includes(S.brush)) S.recent = [S.brush, ...S.recent].slice(0, 9);
  renderBrush();
  S.needLegend = true;
  requestRender();
}

function renderBrush() {
  const chip = $('#brush-chip');
  if (S.brush === null) {
    chip.textContent = '空格';
    chip.style.background = '#fff';
    chip.style.color = '#888';
  } else {
    const c = colorOf(S.brush);
    chip.textContent = S.brush;
    chip.style.background = c.css;
    chip.style.color = c.light ? '#000' : '#fff';
  }
  $('#recent').innerHTML = S.recent.map((c, i) => {
    const col = colorOf(c);
    return `<span class="chip" data-code="${esc(c)}" style="background:${col.css};color:${col.light ? '#000' : '#fff'}"><sup>${i + 1}</sup>${esc(c)}</span>`;
  }).join('');
}

// --------------------------------------------------------------------------- 撤销 / 重做

function pageGeom(p) {
  return { x_lines: [...p.x_lines], y_lines: [...p.y_lines], x_anchors: [...p.x_anchors], y_anchors: [...p.y_anchors] };
}

function snapshot() {
  const d = S.d;
  return JSON.stringify({ grid: d.grid, pages: d.pages, cells: d.cells, legend: d.legend, no_legend: d.no_legend,
    empty_sample: d.empty_sample, flags: [...S.flags] });
}

function restoreSnapshot(json) {
  const o = JSON.parse(json);
  Object.assign(S.d, { grid: o.grid, pages: o.pages, cells: o.cells, legend: o.legend, no_legend: o.no_legend,
    empty_sample: o.empty_sample });
  S.flags = new Map(o.flags);
  S.pi = clamp(S.pi, 0, Math.max(0, S.d.pages.length - 1));
  S.colorCache.clear();
  recount();
  refreshAll();
}

function withSnapshot(fn) {
  const before = snapshot();
  const r = fn();
  pushUndo({ kind: 'all', before, after: snapshot() });
  return r;
}

function pushUndo(e) {
  S.undo.push(e);
  if (S.undo.length > UNDO_LIMIT) S.undo.shift();
  S.redo = [];
  markDirty();
}

function applyEntry(e, dir) {
  if (e.kind === 'cells') {
    const cols = S.d.grid.cols;
    for (const [k, o, n] of e.ch) setCell(Math.floor(k / cols), k % cols, dir < 0 ? o : n, null);
  } else if (e.kind === 'page') {
    const p = S.d.pages[e.pi];
    if (p) Object.assign(p, clone(dir < 0 ? e.before : e.after));
    S.pi = e.pi;
    refreshPagePanel();
  } else if (e.kind === 'all') {
    restoreSnapshot(dir < 0 ? e.before : e.after);
  }
  markDirty();
  requestRender();
}

function undo() {
  const e = S.undo.pop();
  if (!e) { msg('没有可撤销的操作'); return; }
  applyEntry(e, -1);
  S.redo.push(e);
}

function redo() {
  const e = S.redo.pop();
  if (!e) { msg('没有可重做的操作'); return; }
  applyEntry(e, 1);
  S.undo.push(e);
}

// --------------------------------------------------------------------------- 网格线编辑

function remap(base, a, b) {
  const a0 = base[0], b0 = base[base.length - 1];
  return base.map((v) => r2(a + (v - a0) * (b - a) / (b0 - a0)));
}

// 拖动第 i 条线。锚点模式：相邻锚点之间的线按比例重新均分；single：只动这一条
function dragLine(p, axis, i, pos, single) {
  const L = axis === 'x' ? p.x_lines : p.y_lines;
  const A = axis === 'x' ? p.x_anchors : p.y_anchors;
  const n = L.length - 1;
  if (single) {
    const lo = i > 0 ? L[i - 1] + 0.5 : -1e9, hi = i < n ? L[i + 1] - 0.5 : 1e9;
    L[i] = r2(clamp(pos, lo, hi));
  } else {
    let prev = -1, next = -1;
    for (const a of A) { if (a < i) prev = a; if (a > i && next < 0) next = a; }
    const lo = prev >= 0 ? L[prev] + (i - prev) * 0.5 : -1e9;
    const hi = next >= 0 ? L[next] - (next - i) * 0.5 : 1e9;
    L[i] = r2(clamp(pos, lo, hi));
    if (prev >= 0) for (let j = prev + 1; j < i; j++) L[j] = r2(L[prev] + (L[i] - L[prev]) * (j - prev) / (i - prev));
    if (next >= 0) for (let j = i + 1; j < next; j++) L[j] = r2(L[i] + (L[next] - L[i]) * (j - i) / (next - i));
  }
  if (!A.includes(i)) { A.push(i); A.sort((a, b) => a - b); }
}

function dragCorner(p, before, w, x, y) {
  const bx = before.x_lines, by = before.y_lines;
  const n = bx.length - 1, m = by.length - 1;
  let x0 = bx[0], x1 = bx[n], y0 = by[0], y1 = by[m];
  if (w[1] === 'l') x0 = Math.min(x, x1 - n * 2); else x1 = Math.max(x, x0 + n * 2);
  if (w[0] === 't') y0 = Math.min(y, y1 - m * 2); else y1 = Math.max(y, y0 + m * 2);
  p.x_lines = remap(bx, x0, x1);
  p.y_lines = remap(by, y0, y1);
}

function hitGrid(sx, sy) {
  const p = curPage();
  if (!p) return null;
  const xl = p.x_lines, yl = p.y_lines, n = xl.length - 1, m = yl.length - 1;
  for (const [w, i, j] of [['tl', 0, 0], ['tr', n, 0], ['bl', 0, m], ['br', n, m]]) {
    if (Math.abs(X(xl[i]) - sx) <= 9 && Math.abs(Y(yl[j]) - sy) <= 9) return { kind: 'corner', w };
  }
  let best = null;
  if (sy >= Y(yl[0]) - 6 && sy <= Y(yl[m]) + 6) {
    const i = nearestLine(xl, IX(sx));
    const d = Math.abs(X(xl[i]) - sx);
    if (d <= 5) best = { kind: 'line', axis: 'x', i, d };
  }
  if (sx >= X(xl[0]) - 6 && sx <= X(xl[n]) + 6) {
    const j = nearestLine(yl, IY(sy));
    const d = Math.abs(Y(yl[j]) - sy);
    if (d <= 5 && (!best || d < best.d)) best = { kind: 'line', axis: 'y', i: j, d };
  }
  return best;
}

// --------------------------------------------------------------------------- 画布

const canvas = $('#canvas');
const ctx = canvas.getContext('2d', { alpha: false });

const X = (x) => x * S.view.s + S.view.tx;
const Y = (y) => y * S.view.s + S.view.ty;
const IX = (sx) => (sx - S.view.tx) / S.view.s;
const IY = (sy) => (sy - S.view.ty) / S.view.s;

function resizeCanvas() {
  const r = canvas.getBoundingClientRect();
  S.dpr = window.devicePixelRatio || 1;
  S.cw = r.width; S.ch = r.height;
  canvas.width = Math.max(1, Math.round(r.width * S.dpr));
  canvas.height = Math.max(1, Math.round(r.height * S.dpr));
  if (S.needFit) fitView(S.needFit === 'whole');
  requestRender();
}

function requestRender() {
  if (S.rafPending) return;
  S.rafPending = true;
  requestAnimationFrame(() => {
    S.rafPending = false;
    render();
    if (S.needLegend) renderLegend();
  });
}

function fitView(whole) {
  const p = curPage();
  const img = S.imgs[p ? p.image_index : 0];
  let bb;
  if (!whole && p) bb = bboxOf(p);
  else if (img) bb = [0, 0, img.naturalWidth, img.naturalHeight];
  else if (p) bb = bboxOf(p);
  else return;
  const w = bb[2] - bb[0], h = bb[3] - bb[1];
  if (!(w > 0 && h > 0) || !S.cw || !S.ch) { S.needFit = whole ? 'whole' : 'page'; return; }
  S.needFit = false;
  const s = Math.min(S.cw / w, S.ch / h) * 0.96;
  S.view = { s, tx: (S.cw - w * s) / 2 - bb[0] * s, ty: (S.ch - h * s) / 2 - bb[1] * s };
  requestRender();
}

function zoomAt(sx, sy, factor) {
  const s = clamp(S.view.s * factor, 0.03, 80);
  const x = IX(sx), y = IY(sy);
  S.view.s = s;
  S.view.tx = sx - x * s;
  S.view.ty = sy - y * s;
  requestRender();
}

function render() {
  const dpr = S.dpr;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = '#2b2b2b';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  if (!S.d) return;
  const p = curPage();
  const img = S.imgs[p ? p.image_index : 0];
  if (img) {
    ctx.setTransform(dpr * S.view.s, 0, 0, dpr * S.view.s, dpr * S.view.tx, dpr * S.view.ty);
    ctx.imageSmoothingEnabled = S.view.s < 1;
    ctx.drawImage(img, 0, 0);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (p && p.x_lines.length > 1 && S.d.cells.length) {
    drawCells(p);
    if (S.showGrid || S.tool === 'grid') drawGrid(p);
  }
  drawSamples(p ? p.image_index : 0);
}

function visibleRange(p) {
  return {
    c0: clampIdx(p.x_lines, IX(0)), c1: clampIdx(p.x_lines, IX(S.cw)),
    r0: clampIdx(p.y_lines, IY(0)), r1: clampIdx(p.y_lines, IY(S.ch)),
  };
}

function drawCells(p) {
  const { c0, c1, r0, r1 } = visibleRange(p);
  const xl = p.x_lines, yl = p.y_lines, cells = S.d.cells, cols = S.d.grid.cols;
  const hl = S.highlight;
  const cellPx = (xl[p.cols] - xl[0]) / p.cols * S.view.s;
  const drawText = S.showText && cellPx >= 18;
  const fs = Math.min(14, Math.max(8, cellPx * (S.alpha >= 0.6 ? 0.32 : 0.24)));
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = `${fs}px Consolas, monospace`;
  for (let r = r0; r <= r1; r++) {
    const gr = r + p.row_offset;
    if (gr < 0 || gr >= S.d.grid.rows) continue;
    const sy0 = Y(yl[r]), sy1 = Y(yl[r + 1]);
    for (let c = c0; c <= c1; c++) {
      const gc = c + p.col_offset;
      if (gc < 0 || gc >= cols) continue;
      const code = cells[gr][gc];
      const sx0 = X(xl[c]), sx1 = X(xl[c + 1]);
      const w = sx1 - sx0, h = sy1 - sy0;
      if (hl) {
        if (code !== S.brush) {
          ctx.globalAlpha = 0.62;
          ctx.fillStyle = '#000';
          ctx.fillRect(sx0, sy0, w, h);
          ctx.globalAlpha = 1;
          continue;
        }
      }
      if (code !== null && S.alpha > 0) {
        ctx.globalAlpha = S.alpha;
        ctx.fillStyle = colorOf(code).css;
        ctx.fillRect(sx0, sy0, w, h);
        ctx.globalAlpha = 1;
      }
      if (hl) {
        ctx.strokeStyle = '#ffd400';
        ctx.lineWidth = 2;
        ctx.strokeRect(sx0 + 1, sy0 + 1, w - 2, h - 2);
      }
      if (drawText && code !== null) {
        const col = colorOf(code);
        const cx = (sx0 + sx1) / 2, cy = (sy0 + sy1) / 2;
        if (S.alpha >= 0.6) {
          // 色块基本盖住原图：文字放在格子中央
          ctx.fillStyle = col.light ? '#000' : '#fff';
          ctx.fillText(code, cx, cy);
        } else {
          // 半透明：原图的色号在格子中央，标注的色号放到格子下沿，便于上下对照
          const ty = sy0 + h * 0.82;
          ctx.lineWidth = 3;
          ctx.strokeStyle = 'rgba(0,0,0,0.85)';
          ctx.strokeText(code, cx, ty);
          ctx.fillStyle = '#fff';
          ctx.fillText(code, cx, ty);
        }
      }
    }
  }
  // 可疑格
  if (S.flags.size) {
    ctx.lineWidth = 2;
    for (const [k, why] of S.flags) {
      const gr = Math.floor(k / cols), gc = k % cols;
      const r = gr - p.row_offset, c = gc - p.col_offset;
      if (r < r0 || r > r1 || c < c0 || c > c1) continue;
      ctx.strokeStyle = k === S.flagKey ? '#ff00ff' : '#ff2a2a';
      ctx.lineWidth = k === S.flagKey ? 3 : 2;
      ctx.strokeRect(X(xl[c]) + 1, Y(yl[r]) + 1, X(xl[c + 1]) - X(xl[c]) - 2, Y(yl[r + 1]) - Y(yl[r]) - 2);
    }
  }
  // 鼠标所在格
  if (S.hover && S.hover.cell && S.tool !== 'grid') {
    const { r, c } = S.hover.cell;
    ctx.lineWidth = 1;
    ctx.strokeStyle = '#fff';
    ctx.setLineDash([3, 2]);
    ctx.strokeRect(X(xl[c]), Y(yl[r]), X(xl[c + 1]) - X(xl[c]), Y(yl[r + 1]) - Y(yl[r]));
    ctx.setLineDash([]);
  }
}

function drawGrid(p) {
  const xl = p.x_lines, yl = p.y_lines, n = xl.length - 1, m = yl.length - 1;
  const gridMode = S.tool === 'grid';
  const top = Y(yl[0]), bottom = Y(yl[m]), left = X(xl[0]), right = X(xl[n]);
  const { c0, c1, r0, r1 } = visibleRange(p);
  const xa = new Set(p.x_anchors), ya = new Set(p.y_anchors);
  const line = (x0, y0, x1, y1, color, w) => {
    ctx.strokeStyle = color; ctx.lineWidth = w;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  };
  // 吸附后每条线都是锚点：画细一点，免得满屏橙色
  const aw = (A, total) => (A.size >= total ? 1 : 2);
  const xw = aw(xa, n + 1), yw = aw(ya, m + 1);
  for (let i = c0; i <= Math.min(n, c1 + 1); i++) {
    const sx = Math.round(X(xl[i])) + 0.5;
    if (gridMode && xa.has(i)) line(sx, top, sx, bottom, '#ff9500', xw);
    else line(sx, top, sx, bottom, i % 5 === 0 ? 'rgba(0,229,255,0.95)' : 'rgba(0,229,255,0.55)', 1);
  }
  for (let j = r0; j <= Math.min(m, r1 + 1); j++) {
    const sy = Math.round(Y(yl[j])) + 0.5;
    if (gridMode && ya.has(j)) line(left, sy, right, sy, '#ff9500', yw);
    else line(left, sy, right, sy, j % 5 === 0 ? 'rgba(0,229,255,0.95)' : 'rgba(0,229,255,0.55)', 1);
  }
  if (gridMode) {
    ctx.fillStyle = '#ff9500';
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    for (const [i, j] of [[0, 0], [n, 0], [0, m], [n, m]]) {
      ctx.fillRect(X(xl[i]) - 6, Y(yl[j]) - 6, 12, 12);
      ctx.strokeRect(X(xl[i]) - 6, Y(yl[j]) - 6, 12, 12);
    }
    if (S.hover && S.hover.grid && S.hover.grid.kind === 'line') {
      const h = S.hover.grid;
      if (h.axis === 'x') line(X(xl[h.i]), top, X(xl[h.i]), bottom, '#fff', 2);
      else line(left, Y(yl[h.i]), right, Y(yl[h.i]), '#fff', 2);
    }
  }
}

function drawSamples(ii) {
  const marks = [];
  for (const r of S.d.legend) if (r.sample && r.sample.image_index === ii) marks.push([r.sample, r.code || '?', r.rgb]);
  const es = S.d.empty_sample;
  if (es && es.sample && es.sample.image_index === ii) marks.push([es.sample, '空', es.rgb]);
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'middle';
  for (const [s, label, rgb] of marks) {
    const sx = X(s.x), sy = Y(s.y);
    if (sx < -20 || sy < -20 || sx > S.cw + 20 || sy > S.ch + 20) continue;
    ctx.beginPath();
    ctx.arc(sx, sy, 5, 0, Math.PI * 2);
    ctx.fillStyle = rgb ? `rgb(${rgb.join(',')})` : '#fff';
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#ff00ff';
    ctx.stroke();
    ctx.lineWidth = 3;
    ctx.strokeStyle = '#000';
    ctx.strokeText(label, sx + 8, sy);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, sx + 8, sy);
  }
}

// --------------------------------------------------------------------------- 鼠标

function canvasPos(e) {
  const r = canvas.getBoundingClientRect();
  return [e.clientX - r.left, e.clientY - r.top];
}

function onMouseDown(e) {
  if (!S.d) return;
  canvas.focus();
  const [sx, sy] = canvasPos(e);
  if (e.button === 1 || e.button === 2 || (e.button === 0 && S.spaceDown)) {
    e.preventDefault();
    S.drag = { kind: 'pan', sx, sy, tx: S.view.tx, ty: S.view.ty };
    canvas.style.cursor = 'grabbing';
    return;
  }
  if (e.button !== 0) return;
  const x = IX(sx), y = IY(sy);
  const tool = (e.altKey && S.tool === 'paint') ? 'pick' : S.tool;
  if (tool === 'grid') {
    const p = curPage();
    const h = hitGrid(sx, sy);
    if (p && h) S.drag = { kind: 'grid', h, before: pageGeom(p), shift: e.shiftKey, moved: false };
    else S.drag = { kind: 'pan', sx, sy, tx: S.view.tx, ty: S.view.ty };
  } else if (tool === 'paint') {
    S.drag = { kind: 'paint', rec: new Map(), last: [x, y] };
    paintAt(x, y, S.drag.rec);
  } else if (tool === 'fill') {
    fillAt(x, y);
  } else if (tool === 'pick') {
    pickAt(x, y);
    if (S.tool === 'pick') setTool(S.prevTool === 'pick' ? 'paint' : S.prevTool);
  } else if (tool === 'sample') {
    sampleAt(x, y);
  }
  requestRender();
}

function onMouseMove(e) {
  if (!S.d) return;
  const [sx, sy] = canvasPos(e);
  const x = IX(sx), y = IY(sy);
  const d = S.drag;
  if (d) {
    if (d.kind === 'pan') {
      S.view.tx = d.tx + (sx - d.sx);
      S.view.ty = d.ty + (sy - d.sy);
    } else if (d.kind === 'paint') {
      paintLine(d.last[0], d.last[1], x, y, d.rec);
      d.last = [x, y];
    } else if (d.kind === 'grid') {
      const p = curPage();
      if (d.h.kind === 'corner') dragCorner(p, d.before, d.h.w, x, y);
      else dragLine(p, d.h.axis, d.h.i, d.h.axis === 'x' ? x : y, d.shift || e.shiftKey);
      d.moved = true;
    }
    requestRender();
  }
  // 悬停信息
  const inside = sx >= 0 && sy >= 0 && sx <= S.cw && sy <= S.ch;
  const cell = inside ? cellAt(x, y) : null;
  const grid = inside && S.tool === 'grid' && !d ? hitGrid(sx, sy) : null;
  const prev = S.hover;
  S.hover = { cell, grid };
  if (!d) {
    canvas.style.cursor = S.spaceDown ? 'grab' : (S.tool === 'grid'
      ? (grid ? (grid.kind === 'corner' ? 'move' : (grid.axis === 'x' ? 'ew-resize' : 'ns-resize')) : 'default')
      : 'crosshair');
  }
  if (!prev || (prev.cell && cell ? (prev.cell.gr !== cell.gr || prev.cell.gc !== cell.gc) : prev.cell !== cell)
      || JSON.stringify(prev.grid) !== JSON.stringify(grid)) {
    requestRender();
    renderHoverStatus();
  }
}

function onMouseUp() {
  const d = S.drag;
  if (!d) return;
  S.drag = null;
  if (d.kind === 'paint') commitStroke(d.rec);
  else if (d.kind === 'grid' && d.moved) {
    pushUndo({ kind: 'page', pi: S.pi, before: d.before, after: pageGeom(curPage()) });
    refreshPagePanel();
  }
  canvas.style.cursor = S.tool === 'grid' ? 'default' : 'crosshair';
  requestRender();
}

function renderHoverStatus() {
  const h = S.hover;
  if (!h || !h.cell) { $('#st-pos').textContent = ''; return; }
  const { gr, gc } = h.cell;
  const code = S.d.cells[gr][gc];
  let s = `第 ${gr + 1} 行 第 ${gc + 1} 列　${code === null ? '空格' : code}`;
  if (code !== null) {
    const row = S.d.legend.find((r) => r.code === code);
    s += `（图例 ${row && row.count !== null ? row.count : '-'} / 实际 ${S.counts.get(code) || 0}）`;
  }
  const why = S.flags.get(gr * S.d.grid.cols + gc);
  if (why) s += `　可疑：${{ far: '离所有色块都远', ambiguous: '两个色号难区分', 'text?': '有没有文字拿不准', uncertain: '识别不确定' }[why] || why}`;
  $('#st-pos').textContent = s;
}

// --------------------------------------------------------------------------- 可疑格导航

function centerOnCell(gr, gc) {
  let pi = S.pi;
  const inPage = (p) => p && gr >= p.row_offset && gr < p.row_offset + p.rows && gc >= p.col_offset && gc < p.col_offset + p.cols;
  if (!inPage(S.d.pages[pi])) {
    pi = S.d.pages.findIndex(inPage);
    if (pi < 0) return;
    S.pi = pi;
    refreshPagePanel();
  }
  const p = curPage();
  const r = gr - p.row_offset, c = gc - p.col_offset;
  const cx = (p.x_lines[c] + p.x_lines[c + 1]) / 2, cy = (p.y_lines[r] + p.y_lines[r + 1]) / 2;
  const cw = p.x_lines[c + 1] - p.x_lines[c];
  S.view.s = Math.max(S.view.s, 40 / cw);
  S.view.tx = S.cw / 2 - cx * S.view.s;
  S.view.ty = S.ch / 2 - cy * S.view.s;
  requestRender();
}

function gotoFlag(dir) {
  const keys = [...S.flags.keys()].sort((a, b) => a - b);
  updateFlagCount();
  if (!keys.length) { msg('没有可疑格'); return; }
  const cur = S.flagKey === null ? (dir > 0 ? -1 : Infinity) : S.flagKey;
  let k = dir > 0 ? keys.find((v) => v > cur) : [...keys].reverse().find((v) => v < cur);
  if (k === undefined) k = dir > 0 ? keys[0] : keys[keys.length - 1];
  S.flagKey = k;
  const cols = S.d.grid.cols;
  centerOnCell(Math.floor(k / cols), k % cols);
  S.hover = { cell: null, grid: null };
  const idx = keys.indexOf(k) + 1;
  msg(`可疑格 ${idx} / ${keys.length}：第 ${Math.floor(k / cols) + 1} 行 第 ${k % cols + 1} 列（${S.flags.get(k)}）。改格子或按 X 标为已处理`);
}

function resolveFlag() {
  if (S.flagKey === null || !S.flags.has(S.flagKey)) { gotoFlag(1); return; }
  S.flags.delete(S.flagKey);
  markDirty();
  gotoFlag(1);
}

function updateFlagCount() { $('#flag-count').textContent = String(S.flags.size); }

// --------------------------------------------------------------------------- 图例面板

function legendState() {
  const d = S.d;
  if (!d) return { ok: false, text: '', cls: 'na' };
  if (d.no_legend) return { ok: true, text: '无图例（跳过校验）', cls: 'na' };
  if (!d.legend.length) return { ok: false, text: '图例未录入', cls: 'bad' };
  const expected = new Map();
  let unfilled = 0, empty = 0;
  for (const r of d.legend) {
    if (!r.code) { empty++; continue; }
    if (r.count === null || Number.isNaN(r.count)) unfilled++;
    expected.set(r.code, (expected.get(r.code) || 0) + (r.count || 0));
  }
  const codes = new Set([...expected.keys(), ...S.counts.keys()]);
  let bad = 0, total = 0;
  for (const c of codes) {
    if ((expected.get(c) || 0) !== (S.counts.get(c) || 0)) bad++;
    total += S.counts.get(c) || 0;
  }
  if (unfilled) return { ok: false, text: `图例：${unfilled} 行没填颗数`, cls: 'bad' };
  if (bad) return { ok: false, text: `图例：${bad} 个色号不一致`, cls: 'bad' };
  if (empty) return { ok: false, text: '图例有空行', cls: 'bad' };
  return { ok: true, text: `图例校验通过（${total} 颗）`, cls: 'ok' };
}

function renderLegend() {
  S.needLegend = false;
  const tbody = $('#legend-table tbody');
  if (tbody.contains(document.activeElement)) { S.legendPending = true; updateLegendStatus(); return; }
  S.legendPending = false;
  if (!S.d) { tbody.innerHTML = ''; return; }
  const rows = S.d.legend;
  const inLegend = new Set();
  let html = '', sumLegend = 0, sumActual = 0;
  rows.forEach((r, i) => {
    const code = r.code || null;
    if (code) inLegend.add(code);
    const actual = code ? (S.counts.get(code) || 0) : 0;
    const exp = r.count;
    const diff = exp === null ? null : actual - exp;
    sumLegend += exp || 0;
    sumActual += actual;
    const cls = [diff !== 0 ? 'mismatch' : '', code !== null && code === S.brush ? 'brush' : '', S.sampleTarget === i ? 'target' : ''].join(' ');
    html += `<tr data-i="${i}" class="${cls}">
      <td><span class="swatch" style="${r.rgb ? `background:rgb(${r.rgb.join(',')})` : ''}" title="${r.rgb ? `取色 rgb(${r.rgb.join(', ')})` : '还没取色'}"></span></td>
      <td><input class="code" data-i="${i}" value="${esc(r.code || '')}" list="code-list"></td>
      <td><input class="count" data-i="${i}" type="number" min="0" value="${exp === null ? '' : exp}"></td>
      <td class="actual">${actual}</td>
      <td class="diff">${diff === null ? '?' : (diff > 0 ? '+' : '') + diff}</td>
      <td><button class="sample" data-i="${i}" title="在图上点这个色号的图例色块">取色</button></td>
      <td><button class="del" data-i="${i}" title="删除这一行">×</button></td></tr>`;
  });
  const extra = [...S.counts].filter(([c, n]) => n > 0 && !inLegend.has(c)).sort();
  for (const [code, n] of extra) {
    sumActual += n;
    html += `<tr class="mismatch extra" data-code="${esc(code)}">
      <td><span class="swatch" style="background:${colorOf(code).css}"></span></td><td>${esc(code)}</td>
      <td class="muted">不在图例</td><td class="actual">${n}</td><td class="diff">+${n}</td><td></td><td></td></tr>`;
  }
  tbody.innerHTML = html;
  $('#legend-sum').textContent = String(sumLegend);
  $('#actual-sum').textContent = String(sumActual);
  $('#diff-sum').textContent = String(sumActual - sumLegend);
  $('#no-legend').checked = S.d.no_legend;
  updateLegendStatus();
  updateFlagCount();
}

function updateLegendStatus() {
  const st = legendState();
  const el = $('#legend-status');
  el.textContent = st.text;
  el.className = `pill ${st.cls}`;
  const gridOk = S.d && S.d.grid.cols > 0 && S.d.pages.length > 0;
  const btn = $('#btn-export');
  btn.disabled = !(st.ok && gridOk);
  btn.title = !gridOk ? '先设置网格' : (st.ok ? '校验并导出到 labels/<id>.json' : `不能导出：${st.text}（04 §2.2：图例校验不通过禁止导出）`);
  $('#btn-review').disabled = btn.disabled;
}

function renderEmptySwatch() {
  const es = S.d && S.d.empty_sample;
  const el = $('#empty-swatch');
  el.style.background = es && es.rgb ? `rgb(${es.rgb.join(',')})` : '';
  el.title = es && es.rgb ? `rgb(${es.rgb.join(', ')})` : '还没取色';
}

function parseLegendText(text) {
  const out = [];
  const re = /((?:[A-Za-z]+:)?[A-Za-z]*\d+)\s*[x×X*:：=]?\s*(\d+)/g;
  let m;
  while ((m = re.exec(text)) !== null) out.push([normCode(m[1]), parseInt(m[2], 10)]);
  return out;
}

function legendChanged() {
  S.colorCache.clear();
  updateCodeList();
  renderBrush();
  markDirty();
  renderLegend();
  requestRender();
}

// --------------------------------------------------------------------------- 网格面板

function refreshPagePanel() {
  const d = S.d;
  const sel = $('#page-select');
  sel.innerHTML = d ? d.pages.map((p, i) => `<option value="${i}">第 ${i + 1} 页 · 图 ${p.image_index + 1}</option>`).join('') : '';
  sel.value = String(S.pi);
  sel.disabled = !d || d.pages.length <= 1;
  if (!d) return;
  $('#g-cols').value = d.grid.cols || '';
  $('#g-rows').value = d.grid.rows || '';
  const imgs = S.task ? S.task.images : [];
  $('#p-image').innerHTML = imgs.map((im, i) => `<option value="${i}">${i + 1}：${esc(im.image_id)}（${im.width}×${im.height}）</option>`).join('');
  const p = curPage();
  $('#page-label').textContent = p ? `（第 ${S.pi + 1} / ${d.pages.length} 页）` : '（还没有页）';
  for (const id of ['#p-image', '#p-cols', '#p-rows', '#p-coff', '#p-roff', '#btn-page-apply', '#btn-page-del', '#btn-even', '#btn-snap', '#btn-anchor-reset']) {
    $(id).disabled = !p;
  }
  if (p) {
    $('#p-image').value = String(p.image_index);
    $('#p-cols').value = p.cols;
    $('#p-rows').value = p.rows;
    $('#p-coff').value = p.col_offset;
    $('#p-roff').value = p.row_offset;
  }
  const multi = imgs.length > 1;
  $('#l-cols').value = $('#l-cols').value || (p ? p.cols : '');
  $('#l-rows').value = $('#l-rows').value || (p ? p.rows : '');
  $('#l-per-row').value = $('#l-per-row').value || (multi ? Math.ceil(Math.sqrt(imgs.length)) : 1);
}

function applyGridSize() {
  const cols = parseInt($('#g-cols').value, 10), rows = parseInt($('#g-rows').value, 10);
  if (!(cols > 0 && rows > 0)) { msg('列数和行数必须是正整数', true); return; }
  const d = S.d;
  withSnapshot(() => {
    d.grid = { cols, rows };
    d.cells = makeCells(rows, cols, d.cells);
    S.flags = new Map(); // 尺寸变了，可疑格坐标失效
    if (d.pages.length === 0) d.pages.push(newPage(0, cols, rows));
    else if (d.pages.length === 1 && S.task.images.length === 1) {
      const p = d.pages[0];
      const bb = bboxOf(p);
      p.cols = cols; p.rows = rows; p.col_offset = 0; p.row_offset = 0;
      setUniform(p, bb);
    }
  });
  recount();
  refreshAll();
  msg(`网格 ${cols}×${rows}。用「网格」工具（G）拖四个角对准图纸外框`);
  setTool('grid');
}

function applyPage() {
  const p = curPage();
  if (!p) return;
  const ii = parseInt($('#p-image').value, 10) || 0;
  const cols = parseInt($('#p-cols').value, 10), rows = parseInt($('#p-rows').value, 10);
  const co = parseInt($('#p-coff').value, 10) || 0, ro = parseInt($('#p-roff').value, 10) || 0;
  if (!(cols > 0 && rows > 0) || co < 0 || ro < 0) { msg('页的行列数必须是正整数，偏移不能为负', true); return; }
  withSnapshot(() => {
    const changed = cols !== p.cols || rows !== p.rows || ii !== p.image_index;
    const bb = ii !== p.image_index ? defaultBBox(ii) : bboxOf(p);
    Object.assign(p, { image_index: ii, cols, rows, col_offset: co, row_offset: ro });
    if (changed) setUniform(p, bb);
  });
  if (p.col_offset + p.cols > S.d.grid.cols || p.row_offset + p.rows > S.d.grid.rows) msg('注意：这一页超出了整张图纸的范围', true);
  refreshAll();
}

function addPage() {
  const d = S.d;
  const cur = curPage();
  const nimg = S.task.images.length;
  const ii = cur ? Math.min(cur.image_index + 1, nimg - 1) : 0;
  withSnapshot(() => {
    const p = newPage(ii, cur ? cur.cols : d.grid.cols, cur ? cur.rows : d.grid.rows);
    d.pages.push(p);
  });
  S.pi = d.pages.length - 1;
  refreshAll();
  fitView(true);
}

function deletePage() {
  if (!S.d || S.d.pages.length <= 1) { msg('至少要保留一页', true); return; }
  withSnapshot(() => { S.d.pages.splice(S.pi, 1); });
  S.pi = Math.max(0, S.pi - 1);
  refreshAll();
}

function applyLayout() {
  const pc = parseInt($('#l-cols').value, 10), pr = parseInt($('#l-rows').value, 10), k = parseInt($('#l-per-row').value, 10);
  const n = S.task.images.length;
  if (!(pc > 0 && pr > 0 && k > 0)) { msg('每页行列数和每行页数必须是正整数', true); return; }
  const d = S.d;
  withSnapshot(() => {
    const old = d.pages;
    d.pages = range(n).map((i) => {
      const prev = old.find((p) => p.image_index === i);
      const p = { image_index: i, cols: pc, rows: pr, col_offset: (i % k) * pc, row_offset: Math.floor(i / k) * pr, x_lines: [], y_lines: [] };
      setUniform(p, prev ? bboxOf(prev) : defaultBBox(i));
      return p;
    });
    d.grid = { cols: Math.min(k, n) * pc, rows: Math.ceil(n / k) * pr };
    d.cells = makeCells(d.grid.rows, d.grid.cols, d.cells);
    S.flags = new Map();
  });
  S.pi = 0;
  recount();
  refreshAll();
  msg(`分页布局：${n} 页，整张图纸 ${d.grid.cols}×${d.grid.rows}。逐页用「网格」工具对准外框`);
}

async function snapGrid() {
  const p = curPage();
  if (!p) return;
  let res;
  try {
    res = await api('POST', `/api/snap/${encodeURIComponent(S.key)}`, { page: { image_index: p.image_index, x_lines: p.x_lines, y_lines: p.y_lines } });
  } catch (e) { msg(e.message, true); return; }
  const before = pageGeom(p);
  p.x_lines = res.x_lines.map(r2);
  p.y_lines = res.y_lines.map(r2);
  p.x_anchors = range(p.cols + 1);
  p.y_anchors = range(p.rows + 1);
  pushUndo({ kind: 'page', pi: S.pi, before, after: pageGeom(p) });
  msg(`吸附：移动了 ${res.moved} 条线（没有找到明显网格线的保持原位）。不满意可以撤销`);
  requestRender();
}

// --------------------------------------------------------------------------- 初稿

async function loadPredRuns() {
  const sel = $('#pred-runs');
  sel.innerHTML = '<option value="">（没有预测结果）</option>';
  try {
    const r = await api('GET', `/api/preds/${encodeURIComponent(S.key)}`);
    if (r.runs.length) sel.innerHTML = r.runs.map((x) => `<option value="${esc(x.run)}">${esc(x.run)}（${esc(x.mtime.slice(0, 16))}）</option>`).join('');
  } catch (e) { /* 忽略 */ }
}

async function importPred() {
  const run = $('#pred-runs').value;
  if (!run) { msg('没有可导入的预测结果', true); return; }
  let r;
  try { r = await api('GET', `/api/pred/${encodeURIComponent(S.key)}/${encodeURIComponent(run)}`); } catch (e) { msg(e.message, true); return; }
  if (r.error) { dialog(`<p class="err">这次运行识别失败：${esc(r.error)}</p>`); return; }
  const useGrid = $('#pi-grid').checked, useCells = $('#pi-cells').checked, useLegend = $('#pi-legend').checked;
  const d = S.d;
  withSnapshot(() => {
    if (useGrid || useCells) {
      d.grid = { cols: r.grid.cols, rows: r.grid.rows };
      d.cells = makeCells(d.grid.rows, d.grid.cols, d.cells);
    }
    if (useGrid && r.pages.length) d.pages = r.pages.map((p) => normalizePage(p));
    if (useCells) {
      d.cells = makeCells(d.grid.rows, d.grid.cols, r.cells);
      S.flags = new Map(r.flags.map(([rr, cc, why]) => [rr * d.grid.cols + cc, why]));
    }
    if (useLegend && r.legend.length) {
      const old = new Map(d.legend.map((x) => [x.code, x]));
      d.legend = r.legend.map((x) => {
        const code = normCode(x.code);
        const o = old.get(code);
        return { code, count: x.count === undefined ? null : x.count, rgb: o ? o.rgb : null, sample: o ? o.sample : null };
      });
    }
    if (!d.palette_hint && r.palette_guess) d.palette_hint = r.palette_guess;
  });
  S.pi = 0;
  setPalette(d.palette_hint);
  recount();
  refreshAll();
  fitView(false);
  msg(`已导入 ${run}（${r.engine_version || ''}），可疑格 ${S.flags.size} 个（N 逐个检查）。图例颗数请照图核对`);
}

async function autofill() {
  const d = S.d;
  const p = curPage();
  if (!p) { msg('先设置网格', true); return; }
  const sw = d.legend.filter((r) => r.code && r.rgb).map((r) => ({ code: r.code, rgb: r.rgb }));
  const missing = d.legend.filter((r) => r.code && !r.rgb).map((r) => r.code);
  if (d.empty_sample && d.empty_sample.rgb) sw.push({ code: null, rgb: d.empty_sample.rgb });
  if (!sw.length) { msg('先在「图例」页给色号取色', true); switchTab('legend'); return; }
  if (missing.length && !confirm(`这些色号还没取色，自动填充时不会用到：${missing.join(', ')}\n继续？`)) return;
  if (!d.empty_sample && !confirm('还没给「空格颜色」取色：空格会被填成最接近的色号。继续？')) return;
  const body = {
    grid: d.grid, pages: d.pages, cells: d.cells, swatches: sw,
    options: {
      inner: parseFloat($('#af-inner').value) || 0.5, ink_frac: parseFloat($('#af-ink').value) || 0.04,
      max_de: parseFloat($('#af-maxde').value) || 12, overwrite: $('#af-overwrite').checked,
    },
  };
  msg('自动填充中…');
  let r;
  try { r = await api('POST', `/api/autofill/${encodeURIComponent(S.key)}`, body); } catch (e) { msg(e.message, true); return; }
  withSnapshot(() => {
    d.cells = makeCells(d.grid.rows, d.grid.cols, r.cells);
    S.flags = new Map(r.flags.map(([rr, cc, why]) => [rr * d.grid.cols + cc, why]));
  });
  recount();
  S.flagKey = null;
  refreshAll();
  const st = legendState();
  msg(`自动填充 ${r.filled} 格，可疑格 ${S.flags.size} 个（N 逐个检查）。${st.text}`);
}

// --------------------------------------------------------------------------- 信息面板

function renderMeta() {
  const d = S.d, t = S.task;
  if (!d || !t) return;
  $('#m-type').innerHTML = TYPES.map((x) => `<option ${x === d.type ? 'selected' : ''}>${x}</option>`).join('');
  $('#m-palette').value = d.palette_hint || '';
  $('#m-tags').innerHTML = KNOWN_TAGS.map((tg) => `<label><input type="checkbox" value="${tg}" ${d.tags.includes(tg) ? 'checked' : ''}>${tg}</label>`).join('');
  $('#m-tags-extra').value = d.tags.filter((x) => !KNOWN_TAGS.includes(x)).join(', ');
  $('#m-notes').value = d.notes || '';
  const src = t.meta.source || {};
  $('#m-source').innerHTML = Object.entries(src).map(([k, v]) => `<div>${esc(k)}：${k === 'url' && v ? `<a href="${esc(v)}" target="_blank" rel="noreferrer">${esc(v)}</a>` : esc(v ?? '-')}</div>`).join('')
    + `<div>划分：${esc(t.split || '未划分')}　manifest 预估：${esc(t.meta.type || '')} / ${esc(t.meta.size_bucket || '')}</div>`;
  $('#m-images').innerHTML = t.images.map((im, i) => `<div>${i + 1}. ${esc(im.image_id)}　${im.width}×${im.height} ${esc(im.format)}${im.exif_orientation ? `（EXIF 方向 ${im.exif_orientation}）` : ''}${im.readable ? '' : ' <span class="err">未转换</span>'}</div>`).join('');
  const lab = t.label;
  $('#m-label').innerHTML = lab
    ? `<div>标注：${esc(lab.annotator || '-')}（${esc(lab.labeled_at || '')}）</div><div>复核：${esc(lab.reviewed_by || (lab.needs_review ? '待复核' : '不需要'))}${lab.reviewed_at ? `（${esc(lab.reviewed_at)}）` : ''}</div>`
    : '<div>还没有导出过</div>';
  const needReview = t.needs_review || (lab && lab.needs_review);
  $('#btn-review').classList.toggle('hidden', !(needReview && lab));
}

function readMeta() {
  const d = S.d;
  d.type = $('#m-type').value;
  const pid = $('#m-palette').value.trim().toUpperCase();
  if (pid !== d.palette_hint) { d.palette_hint = pid; setPalette(pid); renderBrush(); requestRender(); }
  const extra = $('#m-tags-extra').value.split(/[,，;；\s]+/).map((x) => x.trim()).filter(Boolean);
  d.tags = [...$$('#m-tags input:checked').map((x) => x.value), ...extra.filter((x) => !KNOWN_TAGS.includes(x))];
  d.notes = $('#m-notes').value;
  markDirty();
}

// --------------------------------------------------------------------------- 保存 / 导出

function toDraft() {
  const d = S.d, cols = d.grid.cols;
  return {
    type: d.type, palette_hint: d.palette_hint, tags: d.tags, notes: d.notes,
    grid: d.grid,
    pages: d.pages.map((p) => ({ image_index: p.image_index, cols: p.cols, rows: p.rows, col_offset: p.col_offset,
      row_offset: p.row_offset, x_lines: p.x_lines, y_lines: p.y_lines, x_anchors: p.x_anchors, y_anchors: p.y_anchors })),
    cells: d.cells,
    legend: d.legend.map((r) => ({ code: r.code || null, count: r.count, rgb: r.rgb, sample: r.sample })),
    empty_sample: d.empty_sample, no_legend: d.no_legend,
    flags: [...S.flags].map(([k, why]) => [Math.floor(k / cols), k % cols, why]),
    annotator: d.annotator || null, recent: S.recent,
  };
}

function setSaveStatus(text) { $('#save-status').textContent = text; }

function markDirty() {
  if (!S.d) return;
  S.dirty = true;
  setSaveStatus('未保存');
  clearTimeout(S.saveTimer);
  S.saveTimer = setTimeout(() => saveDraft(), 1200);
  updateLegendStatus();
}

async function saveDraft(force) {
  if (!S.d || !S.key) return;
  clearTimeout(S.saveTimer);
  if (S.saving) await S.saving;
  if (!S.dirty && !force) return;
  const key = S.key;
  const body = toDraft();
  S.dirty = false;
  setSaveStatus('保存中…');
  S.saving = api('PUT', `/api/draft/${encodeURIComponent(key)}`, body)
    .then((r) => { if (key === S.key && !S.dirty) setSaveStatus(`已保存 ${nowText()}（rev ${r.rev}）`); updateTaskStatus(key, r.status); })
    .catch((e) => { if (key === S.key) S.dirty = true; setSaveStatus('保存失败'); msg(`保存失败：${e.message}`, true); })
    .finally(() => { S.saving = null; });
  await S.saving;
}

async function doExport(review) {
  if (!S.d) return;
  const user = $('#user-name').value.trim();
  if (!user) { dialog('<p>先在右上角「我是」里填写你的名字（导出时记为标注人 / 复核人）。</p>'); $('#user-name').focus(); return; }
  const st = legendState();
  if (!st.ok) { dialog(`<p class="err">不能导出：${esc(st.text)}</p><p>04 §2.2：图例校验不通过时禁止导出。</p>`); return; }
  await saveDraft(true);
  let res;
  try {
    res = await api('POST', `/api/export/${encodeURIComponent(S.key)}`, { draft: toDraft(), user, review: !!review });
  } catch (e) { dialog(`<p class="err">${esc(e.message)}</p>`); return; }
  updateTaskStatus(S.key, res.status);
  if (!res.ok) {
    dialog(`<p class="err">导出失败：</p>${listHtml(res.errors, 'err')}${res.warnings.length ? `<p class="warn">提示：</p>${listHtml(res.warnings, 'warn')}` : ''}`);
    return;
  }
  S.d.annotator = review ? S.d.annotator : user;
  try {
    const t = await api('GET', `/api/task/${encodeURIComponent(S.key)}`);
    S.task.label = t.label;
    S.task.status = t.status;
  } catch (e) { /* 忽略 */ }
  renderMeta();
  const needReview = res.needs_review && !res.reviewed_by;
  dialog(`<p>${review ? '复核通过，' : ''}已写入 <code>${esc(res.path)}</code>。</p>`
    + (needReview ? '<p class="warn">这张图是 XL 档或分页图，需要另一个人打开它点「复核通过」（04 §2.2）。</p>' : '')
    + (res.warnings.length ? `<p class="warn">提示：</p>${listHtml(res.warnings, 'warn')}` : ''));
}

// --------------------------------------------------------------------------- 界面刷新、工具切换

function setTool(t) {
  if (t !== S.tool) S.prevTool = S.tool;
  S.tool = t;
  for (const b of $$('#tools button')) b.classList.toggle('active', b.dataset.tool === t);
  if (t !== 'sample') { S.sampleTarget = null; S.needLegend = true; }
  canvas.style.cursor = t === 'grid' ? 'default' : 'crosshair';
  if (t === 'sample') {
    const tgt = S.sampleTarget === 'empty' ? '空格' : (S.sampleTarget !== null && S.d ? (S.d.legend[S.sampleTarget] || {}).code || '这一行' : null);
    msg(tgt ? `取色：在图上点 ${tgt} 的图例色块（或一个该色的格子）` : '取色：先在「图例」页选一行');
  }
  requestRender();
}

function switchTab(name) {
  for (const b of $$('#tabs button')) b.classList.toggle('active', b.dataset.tab === name);
  for (const t of $$('.tab')) t.classList.toggle('hidden', t.id !== `tab-${name}`);
}

function refreshAll() {
  if (!S.d) return;
  const t = S.task;
  $('#task-title').textContent = S.key;
  $('#task-meta').textContent = `${t.split || '未划分'} · ${S.d.type} · ${S.d.grid.cols}×${S.d.grid.rows} · ${t.images.length} 张图`;
  refreshPagePanel();
  renderLegend();
  renderEmptySwatch();
  renderMeta();
  renderBrush();
  updateCodeList();
  updateFlagCount();
  $('#draft-info').textContent = t.draft_source === 'draft' ? `草稿 rev ${t.draft.rev || 0}，${t.draft.updated_at || ''}` : '';
  requestRender();
}

// --------------------------------------------------------------------------- 事件绑定

function isTyping(e) {
  const t = e.target;
  return t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable);
}

function onKeyDown(e) {
  if (e.key === 'Escape') {
    for (const m of $$('.modal')) m.classList.add('hidden');
    if (isTyping(e)) e.target.blur();
    if (S.tool === 'sample') setTool(S.prevTool === 'sample' ? 'paint' : S.prevTool);
    return;
  }
  const ctrl = e.ctrlKey || e.metaKey;
  if (ctrl && e.key.toLowerCase() === 's') { e.preventDefault(); saveDraft(true); return; }
  if (isTyping(e) || !S.d) return;
  if (ctrl) {
    const k = e.key.toLowerCase();
    if (k === 'z' && !e.shiftKey) { e.preventDefault(); undo(); } else if (k === 'y' || (k === 'z' && e.shiftKey)) { e.preventDefault(); redo(); }
    return;
  }
  if (e.altKey) return;
  const k = e.key;
  if (k === ' ') { e.preventDefault(); S.spaceDown = true; canvas.style.cursor = 'grab'; return; }
  if (/^[1-9]$/.test(k)) { const c = S.recent[parseInt(k, 10) - 1]; if (c) setBrush(c); return; }
  switch (k.toLowerCase()) {
    case 'e': setBrush(null); break;
    case 'b': setTool('paint'); break;
    case 'f': setTool('fill'); break;
    case 'i': setTool('pick'); break;
    case 'g': setTool('grid'); break;
    case 's': if (S.legendSel >= 0 && S.sampleTarget === null) S.sampleTarget = S.legendSel; setTool('sample'); break;
    case 'h': S.highlight = !S.highlight; $('#show-hl').checked = S.highlight; requestRender(); break;
    case 't': S.showText = !S.showText; $('#show-text').checked = S.showText; requestRender(); break;
    case 'n': gotoFlag(e.shiftKey ? -1 : 1); break;
    case 'x': resolveFlag(); break;
    case '0': fitView(false); break;
    case '[': setAlpha(S.alpha - 0.1); break;
    case ']': setAlpha(S.alpha + 0.1); break;
    case '/': e.preventDefault(); $('#code-input').focus(); break;
    case '?': $('#help').classList.remove('hidden'); break;
    case 'l': toggleList(); break;
    case 'pageup': gotoPage(S.pi - 1); break;
    case 'pagedown': gotoPage(S.pi + 1); break;
    default: return;
  }
  e.preventDefault();
}

function setAlpha(a) {
  S.alpha = clamp(Math.round(a * 100) / 100, 0, 1);
  $('#alpha').value = String(Math.round(S.alpha * 100));
  requestRender();
}

function gotoPage(i) {
  if (!S.d || i < 0 || i >= S.d.pages.length) return;
  S.pi = i;
  refreshPagePanel();
  fitView(false);
}

function toggleList() {
  $('#list-panel').classList.toggle('hidden');
  resizeCanvas();
}

function bind() {
  canvas.tabIndex = 0;
  canvas.addEventListener('mousedown', onMouseDown);
  window.addEventListener('mousemove', onMouseMove);
  window.addEventListener('mouseup', onMouseUp);
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('mouseleave', () => { S.hover = null; renderHoverStatus(); requestRender(); });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const [sx, sy] = canvasPos(e);
    zoomAt(sx, sy, Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015)));
  }, { passive: false });
  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', (e) => { if (e.key === ' ') { S.spaceDown = false; canvas.style.cursor = S.tool === 'grid' ? 'default' : 'crosshair'; } });
  new ResizeObserver(resizeCanvas).observe(canvas);
  window.addEventListener('beforeunload', (e) => {
    if (S.dirty) { saveDraft(true); e.preventDefault(); e.returnValue = ''; }
  });

  // 顶栏
  $('#btn-list').onclick = toggleList;
  $('#btn-help').onclick = () => $('#help').classList.remove('hidden');
  for (const m of $$('.modal')) m.addEventListener('click', (e) => { if (e.target === m || e.target.classList.contains('close')) m.classList.add('hidden'); });
  const user = $('#user-name');
  user.value = localStorage.getItem(USER_KEY) || '';
  user.addEventListener('change', () => localStorage.setItem(USER_KEY, user.value.trim()));
  $('#btn-export').onclick = () => doExport(false);
  $('#btn-review').onclick = () => {
    if (confirm('确认已经逐格复核过这张图，并以你的名字记为复核人？')) doExport(true);
  };

  // 任务列表
  for (const id of ['#f-split', '#f-status']) $(id).addEventListener('change', renderTaskList);
  $('#f-search').addEventListener('input', renderTaskList);
  $('#task-table tbody').addEventListener('click', (e) => {
    const tr = e.target.closest('tr[data-key]');
    if (tr) openTask(tr.dataset.key);
  });

  // 工具栏
  for (const b of $$('#tools button')) b.onclick = () => {
    if (b.dataset.tool === 'sample' && S.legendSel >= 0 && S.sampleTarget === null) S.sampleTarget = S.legendSel;
    setTool(b.dataset.tool);
  };
  const ci = $('#code-input');
  ci.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      const v = normCode(ci.value);
      if (v) {
        setBrush(v);
        if (S.palMap && !v.includes(':') && !S.palMap.has(v)) msg(`注意：${v} 不在 ${S.d ? S.d.palette_hint : ''} 色卡里（混用其他品牌写成 品牌:色号）`, true);
        if (S.tool !== 'paint' && S.tool !== 'fill') setTool('paint');
      }
      ci.value = '';
      canvas.focus();
      e.preventDefault();
    }
  });
  $('#btn-empty').onclick = () => setBrush(null);
  $('#recent').addEventListener('click', (e) => { const c = e.target.closest('.chip'); if (c) setBrush(c.dataset.code); });
  $('#alpha').addEventListener('input', (e) => { S.alpha = parseInt(e.target.value, 10) / 100; requestRender(); });
  $('#show-text').addEventListener('change', (e) => { S.showText = e.target.checked; requestRender(); });
  $('#show-grid').addEventListener('change', (e) => { S.showGrid = e.target.checked; requestRender(); });
  $('#show-hl').addEventListener('change', (e) => { S.highlight = e.target.checked; requestRender(); });
  $('#btn-flag-next').onclick = () => gotoFlag(1);
  $('#btn-flag-prev').onclick = () => gotoFlag(-1);
  $('#page-select').addEventListener('change', (e) => gotoPage(parseInt(e.target.value, 10)));
  $('#btn-fit').onclick = () => fitView(true);
  $('#btn-undo').onclick = undo;
  $('#btn-redo').onclick = redo;

  // 侧栏
  for (const b of $$('#tabs button')) b.onclick = () => switchTab(b.dataset.tab);
  $('#btn-grid-apply').onclick = applyGridSize;
  $('#btn-page-apply').onclick = applyPage;
  $('#btn-page-add').onclick = addPage;
  $('#btn-page-del').onclick = deletePage;
  $('#btn-even').onclick = () => {
    const p = curPage();
    if (!p) return;
    const before = pageGeom(p);
    setUniform(p, bboxOf(p));
    pushUndo({ kind: 'page', pi: S.pi, before, after: pageGeom(p) });
    requestRender();
  };
  $('#btn-snap').onclick = snapGrid;
  $('#btn-anchor-reset').onclick = () => {
    const p = curPage();
    if (!p) return;
    const before = pageGeom(p);
    p.x_anchors = [0, p.cols];
    p.y_anchors = [0, p.rows];
    pushUndo({ kind: 'page', pi: S.pi, before, after: pageGeom(p) });
    requestRender();
  };
  $('#btn-layout').onclick = applyLayout;

  // 图例
  const lt = $('#legend-table tbody');
  lt.addEventListener('click', (e) => {
    const tr = e.target.closest('tr');
    if (!tr) return;
    if (e.target.classList.contains('sample')) {
      const i = parseInt(e.target.dataset.i, 10);
      S.legendSel = i;
      S.sampleTarget = i;
      setTool('sample');
      S.needLegend = true;
      requestRender();
      return;
    }
    if (e.target.classList.contains('del')) {
      const i = parseInt(e.target.dataset.i, 10);
      S.d.legend.splice(i, 1);
      S.legendSel = -1;
      legendChanged();
      return;
    }
    if (e.target.tagName === 'INPUT') return;
    if (tr.dataset.code) { setBrush(tr.dataset.code); return; }
    const i = parseInt(tr.dataset.i, 10);
    S.legendSel = i;
    const code = S.d.legend[i] && S.d.legend[i].code;
    if (code) setBrush(code); else { S.needLegend = true; requestRender(); }
  });
  lt.addEventListener('change', (e) => {
    const i = parseInt(e.target.dataset.i, 10);
    const row = S.d.legend[i];
    if (!row) return;
    if (e.target.classList.contains('code')) row.code = normCode(e.target.value) || '';
    if (e.target.classList.contains('count')) {
      const v = e.target.value.trim();
      row.count = v === '' ? null : Math.max(0, parseInt(v, 10) || 0);
    }
    S.needLegend = true;
    legendChanged();
  });
  lt.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' || e.target.tagName !== 'INPUT') return;
    e.preventDefault();
    const cls = e.target.classList.contains('code') ? 'count' : 'code';
    const i = parseInt(e.target.dataset.i, 10) + (cls === 'code' ? 1 : 0);
    e.target.blur();
    setTimeout(() => { const nx = $(`#legend-table input.${cls}[data-i="${i}"]`); if (nx) { nx.focus(); nx.select(); } }, 0);
  });
  lt.addEventListener('focusout', () => setTimeout(() => { if (S.legendPending && !lt.contains(document.activeElement)) renderLegend(); }, 0));
  $('#btn-legend-add').onclick = () => {
    S.d.legend.push({ code: '', count: null, rgb: null, sample: null });
    legendChanged();
    setTimeout(() => { const ins = $$('#legend-table input.code'); if (ins.length) ins[ins.length - 1].focus(); }, 0);
  };
  $('#btn-legend-parse').onclick = () => {
    const items = parseLegendText($('#legend-bulk').value);
    if (!items.length) { msg('没有解析出「色号 颗数」', true); return; }
    for (const [code, count] of items) {
      const row = S.d.legend.find((r) => r.code === code);
      if (row) row.count = count; else S.d.legend.push({ code, count, rgb: null, sample: null });
    }
    $('#legend-bulk').value = '';
    legendChanged();
    msg(`图例：解析了 ${items.length} 项`);
  };
  $('#btn-legend-fromcells').onclick = () => {
    const have = new Set(S.d.legend.map((r) => r.code));
    let n = 0;
    for (const code of [...S.counts.keys()].sort()) if (!have.has(code)) { S.d.legend.push({ code, count: null, rgb: null, sample: null }); n++; }
    legendChanged();
    msg(n ? `补了 ${n} 个色号，颗数请照图上的图例填写` : '格子里的色号都已在图例里');
  };
  $('#no-legend').addEventListener('change', (e) => { S.d.no_legend = e.target.checked; markDirty(); renderLegend(); });
  $('#btn-empty-sample').onclick = () => { S.sampleTarget = 'empty'; setTool('sample'); };

  // 初稿
  $('#btn-pred-import').onclick = importPred;
  $('#btn-autofill').onclick = autofill;
  $('#btn-reload').onclick = async () => {
    if (S.dirty && !confirm('有未保存的修改，重新加载会丢掉它们。继续？')) return;
    S.dirty = false;
    clearTimeout(S.saveTimer);
    await openTask(S.key);
  };

  // 信息
  for (const id of ['#m-type', '#m-palette', '#m-tags-extra', '#m-notes']) $(id).addEventListener('change', readMeta);
  $('#m-tags').addEventListener('change', readMeta);
}

async function boot() {
  bind();
  setTool('paint');
  resizeCanvas();
  try {
    const [pal] = await Promise.all([api('GET', '/api/palettes'), loadTasks()]);
    S.palettes = pal.palettes || [];
    $('#palette-list').innerHTML = S.palettes.map((p) => `<option value="${esc(p.id)}">`).join('');
  } catch (e) {
    msg(`加载失败：${e.message}`, true);
    return;
  }
  const key = decodeURIComponent(location.hash.slice(1));
  if (key) await openTask(key);
  else msg('从左侧列表选一个任务开始。按 ? 查看快捷键');
}

boot();
