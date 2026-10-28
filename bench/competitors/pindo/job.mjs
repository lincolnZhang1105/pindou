// Node 入口（entry.mjs）和浏览器对照入口（browser_entry.mjs）共用：用 Pindo 识别一张图，整理成统一的输出行。
// 识别参数照 Pindo 网页的默认状态：preferAutoGrid = true，色卡不过滤（整套色卡）。

import { recognizePatternFromImage } from 'pindo-src/lib/engine/pattern-recognition.ts';
import { matchColor } from 'pindo-src/lib/engine/color-matcher.ts';
import { loadPalette } from 'pindo-src/lib/data/palettes/loader.ts';

export async function runJob(file, job) {
  const palette = await loadPalette(job.brand);
  const t0 = performance.now();
  const res = await recognizePatternFromImage(file, {
    brand: job.brand, width: job.width, height: job.height, palette, preferAutoGrid: true,
  });
  const ms = performance.now() - t0;
  const byId = new Map(palette.map((c) => [c.id, c.code]));
  const { width, height } = res.pattern.metadata;
  const codes = [];
  for (const row of res.pattern.cells) {
    for (const cell of row) codes.push(cell.isEmpty ? null : byId.get(cell.colorId) ?? `?${cell.colorId}`);
  }
  // 纯白底在 Pindo 里会被归到哪个色号（--white-as-empty 变体用）：直接用它自己的配色函数算
  const white = matchColor({ r: 255, g: 255, b: 255 }, palette).code;
  return {
    key: job.key, ok: true, grid_source: res.gridSource, cols: width, rows: height, codes,
    white_code: white, palette_size: palette.length, ms: Math.round(ms * 10) / 10,
  };
}

export async function runJobSafe(file, job) {
  try {
    return await runJob(file, job);
  } catch (e) {
    return { key: job.key, ok: false, error: String(e?.message ?? e) };
  }
}
