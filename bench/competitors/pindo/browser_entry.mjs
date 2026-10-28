// 浏览器对照入口（check-browser 用）：在真浏览器里跑和 Node 端同一份 Pindo 代码，验证 Node 替身层没有引入偏差。
// 页面里放一个 <script id="jobs" type="application/json">，每个 job 带 base64 图片；结果写进 <pre id="out">。

import { runJobSafe } from './job.mjs';

async function main() {
  const jobs = JSON.parse(document.getElementById('jobs').textContent);
  const results = [];
  for (const job of jobs) {
    const bytes = Uint8Array.from(atob(job.b64), (c) => c.charCodeAt(0));
    const file = new File([bytes], job.key, { type: job.mime });
    results.push(await runJobSafe(file, job));
  }
  document.getElementById('out').textContent = JSON.stringify(results);
  document.title = 'done';
}

main().catch((e) => {
  document.getElementById('out').textContent = JSON.stringify({ fatal: String(e?.message ?? e) });
});
