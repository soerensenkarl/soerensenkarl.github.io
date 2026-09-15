// A helper thread of wasm.js: instantiates the kernels on the model's shared memory and runs encoder and key/value rows.
import { stageA, stageB, stageKV } from "./wasm.js";

let k = null, F = null, encP, kvP, B, Sc, d, heads, tmp64;

self.onmessage = e => {
  const m = e.data, t0 = performance.now();
  try {
    if (m.op === "init") {
      k = new WebAssembly.Instance(m.module, { env: { memory: m.memory } }).exports;
      F = new Float32Array(m.memory.buffer);
      ({ encP, kvP, B, Sc, d, heads } = m);
      tmp64 = new Float64Array(m.Mmax);
    } else if (m.op === "A") stageA(k, encP[m.l], B, m.lo, m.hi, d);
    else if (m.op === "B") stageB(k, F, encP[m.l], B, Sc, tmp64, m.lo, m.hi, m.M, d, heads);
    else if (m.op === "KV") stageKV(k, kvP, B, m.lo, m.hi, m.N, d);
    self.postMessage({ id: m.id, ms: performance.now() - t0 });
  } catch (err) {
    self.postMessage({ id: m.id, error: String(err && err.stack || err) });
  }
};
