// T0.8 Pindo baseline 的 Node 端入口（我们自己写的胶水代码，不含 Pindo 的代码）。
//
// build.mjs 把它和本地下载的 Pindo 源码一起打包成 .work/bundle-<commit>.mjs（只在本地，不进 git、不分发）。
// 运行：stdin 读一个 JSON {"jobs": [{"key", "path", "mime", "brand", "width", "height"}]}，
// 每处理完一张图往 stdout 写一行 JSON（格式见 job.mjs）：
//   {"key", "ok": true, "grid_source", "cols", "rows", "codes": [行优先的色号], "white_code", "palette_size", "ms"}
//   {"key", "ok": false, "error": "..."}

import { readFileSync } from 'node:fs';
import { createCanvas, Image, ImageData } from '@napi-rs/canvas';
import { runJobSafe } from './job.mjs';

// ---- 浏览器接口的最小替身：Pindo 读图只用到 Image、URL.createObjectURL、document.createElement('canvas')
globalThis.Image = Image;
globalThis.ImageData = ImageData;
globalThis.document = {
  createElement(tag) {
    if (tag !== 'canvas') throw new Error(`替身不支持 document.createElement('${tag}')`);
    return createCanvas(1, 1);
  },
};
// 传给 Pindo 的「File」是 {bytes, type}；对象 URL 直接换成 data URL，Image 解码时不经过文件系统
URL.createObjectURL = (file) => `data:${file.type};base64,${file.bytes.toString('base64')}`;
URL.revokeObjectURL = () => {};

const { jobs } = JSON.parse(readFileSync(0, 'utf8'));
for (const job of jobs) {
  const file = { bytes: readFileSync(job.path), type: job.mime, name: job.path };
  process.stdout.write(JSON.stringify(await runJobSafe(file, job)) + '\n');
}
