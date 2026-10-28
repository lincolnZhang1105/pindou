// 把我们的入口和本地的 Pindo 源码打包成一个文件（T0.8）。
//
//   node build.mjs <Pindo 源码目录> <输出文件>            Node 版（entry.mjs）
//   node build.mjs <Pindo 源码目录> <输出文件> --browser  浏览器对照版（browser_entry.mjs，IIFE）
//
// 入口里的 `pindo-src/...` 指向 Pindo 源码目录；Pindo 自己的 `@/...` 别名由 esbuild 按它的 tsconfig.json 解析。
// Node 版不打包 @napi-rs/canvas（原生模块），运行时从本目录的 node_modules 加载。

import { build } from 'esbuild';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const [srcDir, outFile, flag] = process.argv.slice(2);
if (!srcDir || !outFile) {
  console.error('用法：node build.mjs <Pindo 源码目录> <输出文件> [--browser]');
  process.exit(2);
}
const browser = flag === '--browser';
const here = path.dirname(fileURLToPath(import.meta.url));
const pindoRoot = path.resolve(srcDir);

const pindoSrc = {
  name: 'pindo-src',
  setup(b) {
    b.onResolve({ filter: /^pindo-src\// }, (args) => ({ path: path.join(pindoRoot, args.path.slice('pindo-src/'.length)) }));
  },
};

await build({
  entryPoints: [path.join(here, browser ? 'browser_entry.mjs' : 'entry.mjs')],
  outfile: path.resolve(outFile),
  bundle: true,
  platform: browser ? 'browser' : 'node',
  format: browser ? 'iife' : 'esm',
  target: browser ? 'es2022' : 'node20',
  external: browser ? [] : ['@napi-rs/canvas'],
  plugins: [pindoSrc],
  logLevel: 'warning',
  legalComments: 'inline',
});
console.log(`打包完成：${outFile}`);
