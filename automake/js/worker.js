// The page's model thread: loads a network's weights, picks the fastest way to run it, and writes walls on request, posting every
// pass, part and pick as it happens so the page can draw the wall and the network live. A new run cancels the one before;
// loading another network replaces the current one.
//
// Backends: WebAssembly SIMD kernels (wasm.js) with helper threads on shared memory when the page is cross-origin isolated,
// on this thread alone when it is not; plain JavaScript (model.js, helper threads with copies) where WebAssembly SIMD is missing
// or with ?backend=js.
//
// in:  {type: "load", url, file, format, backend, threads}  -> {type: "progress", loaded, total} ..., {type: "ready", backend, ...}
//      {type: "run", id, wall, ops, start, brief, reject}   -> {type: "encode"|"encoded"|"part"|"pass", id, ...}, {type: "done", id, ...}
//      {type: "cancel"}
import { M0, PROF, profReset } from "./model.js";
import { EncoderPool } from "./pool.js";
import { M0Wasm, WasmPool, simdSupported } from "./wasm.js";
import { writeWall } from "./writer.js";

let model = null, pool = null, current = 0, loading = 0, backend = "";
const kernels = {};

const channel = new MessageChannel(), wake = [];
channel.port1.onmessage = () => { const r = wake.shift(); if (r) r(); };
const nextTask = () => new Promise(r => { wake.push(r); channel.port2.postMessage(0); });   // let queued messages in

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = +res.headers.get("content-length") || 0;
  if (!res.body) return res.arrayBuffer();
  const reader = res.body.getReader(), parts = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    onProgress(loaded, Math.max(total, loaded));
  }
  const out = new Uint8Array(loaded);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out.buffer;
}

function compileKernels(shared) {
  const url = new URL(shared ? "./kernels-shared.wasm" : "./kernels.wasm", import.meta.url);
  return (kernels[shared] ??= WebAssembly.compileStreaming(fetch(url))
    .catch(async () => WebAssembly.compile(await (await fetch(url)).arrayBuffer())));
}

async function load(msg) {
  const token = ++loading;
  current = 0;
  const t0 = performance.now();
  const cores = self.navigator && navigator.hardwareConcurrency || 2;
  const useWasm = msg.backend !== "js" && simdSupported();
  const shared = useWasm && self.crossOriginIsolated === true;
  const kernelsReady = useWasm ? compileKernels(shared) : null;          // compiles while the weights download
  const manifest = await (await fetch(msg.url + `${msg.file}.${msg.format}.json`)).json();
  const buffer = await fetchWithProgress(msg.url + `${msg.file}.${msg.format}.bin`,
    (loaded, total) => { if (token === loading) self.postMessage({ type: "progress", file: msg.file, loaded, total }); });
  if (token !== loading) return;
  const tFetch = performance.now() - t0;
  const threads = msg.threads ? Math.max(1, +msg.threads) : (shared ? Math.max(1, Math.min(6, Math.floor(cores / 2))) : 1);
  let next = null, nextBackend = "";
  if (useWasm) {
    try {
      next = M0Wasm.create(manifest, buffer, await kernelsReady, { shared, helpers: shared ? threads - 1 : 0 });
    } catch (err) {
      next = null;                                                         // e.g. not enough memory: plain JavaScript below
    }
  }
  if (!next) next = M0.fromBuffers(manifest, buffer);
  const tParse = performance.now() - t0 - tFetch;
  if (pool) { pool.terminate(); pool = null; }
  model = null;
  let nextPool;
  if (next instanceof M0Wasm) {
    nextPool = shared && threads > 1 ? await WasmPool.create(next, threads - 1) : null;
    nextBackend = `WebAssembly SIMD, ${(nextPool ? nextPool.size : 0) + 1} thread${nextPool ? "s" : ""}`;
  } else {
    nextPool = await EncoderPool.create(next, msg.threads ? Math.max(0, +msg.threads - 1) : Math.max(0, Math.min(3, Math.floor(cores / 2) - 1)));
    nextBackend = `JavaScript, ${(nextPool ? nextPool.size : 0) + 1} thread${nextPool ? "s" : ""}`;
  }
  if (token !== loading) { if (nextPool) nextPool.terminate(); return; }
  model = next; pool = nextPool; backend = nextBackend;
  const { config, items, params, trained, commit, name, run, format, minutes, script } = manifest;
  self.postMessage({ type: "ready", file: msg.file, fetchMs: tFetch, parseMs: tParse, poolMs: performance.now() - t0 - tFetch - tParse,
    helpers: pool ? pool.size : 0, bytes: buffer.byteLength, backend, isolated: self.crossOriginIsolated === true,
    manifest: { config, items, params, trained, commit, name, run, format, minutes, script } });
}

async function run(msg) {
  const id = msg.id;
  current = id;
  profReset(!!msg.prof);
  const t0 = performance.now();
  let firstPart = null, parts = 0;
  const gen = writeWall(model, { wall: msg.wall, ops: msg.ops, start: msg.start, brief: msg.brief, reject: msg.reject, detail: true, pool });
  let r;
  for (;;) {
    r = await gen.next();
    if (current !== id) return;
    if (r.done) break;
    const ev = r.value;
    if (ev.type === "part") { parts += 1; if (firstPart === null) firstPart = performance.now() - t0; }
    self.postMessage({ ...ev, id, ms: performance.now() - t0 });
    if (ev.type === "part" || ev.type === "encoded") {
      await nextTask();
      if (current !== id) return;
    }
  }
  self.postMessage({ type: "done", id, ...r.value, ms: performance.now() - t0, firstPartMs: firstPart, parts, backend,
    prof: PROF.on ? { t: PROF.t, n: PROF.n } : null });
}

self.onmessage = e => {
  const msg = e.data;
  if (msg.type === "load") load(msg).catch(err => self.postMessage({ type: "error", message: String(err && err.message || err) }));
  else if (msg.type === "run") {
    if (!model) { self.postMessage({ type: "error", id: msg.id, message: "the network is not loaded yet" }); return; }
    run(msg).catch(err => self.postMessage({ type: "error", id: msg.id, message: String(err && err.stack || err) }));
  } else if (msg.type === "cancel") current = 0;
};
