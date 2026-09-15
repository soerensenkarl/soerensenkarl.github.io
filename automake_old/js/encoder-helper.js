// A helper thread of pool.js: encoder rows and key/value rows for M0 (see model.js stageA, stageB, stageKV).
import { encoderLayer, kvWeights, makeScratch, stageA, stageB, stageKV } from "./model.js";

let layers = null, kv = null, d = 0, heads = 0, scratch = null;

self.onmessage = e => {
  const m = e.data, t0 = performance.now();
  try {
    if (m.type === "init") {
      d = m.config.d; heads = m.config.heads;
      layers = Array.from({ length: m.config.enc_layers }, (_, l) => encoderLayer(m.tensors, l, d));
      kv = kvWeights(m.tensors, m.config.dec_layers, d);
      scratch = makeScratch(d, 512);
      self.postMessage({ id: m.id });
    } else if (m.type === "A") {
      const n = m.hi - m.lo, H = new Float32Array(n * d), Q = new Float32Array(n * d), K = new Float32Array(n * d), V = new Float32Array(n * d);
      stageA(layers[m.layer], m.X, H, Q, K, V, 0, n, d);
      self.postMessage({ id: m.id, lo: m.lo, Q, K, V, ms: performance.now() - t0 }, [Q.buffer, K.buffer, V.buffer]);
    } else if (m.type === "B") {
      const n = m.hi - m.lo, H = new Float32Array(n * d);
      stageB(layers[m.layer], m.X, H, m.Q, m.K, m.V, m.M, 0, n, d, heads, scratch);
      self.postMessage({ id: m.id, lo: m.lo, X: m.X, ms: performance.now() - t0 }, [m.X.buffer]);
    } else if (m.type === "KV") {
      const n = m.hi - m.lo, box = Math.max(0, Math.min(m.hi, m.boxRows) - m.lo);
      const crossK = kv.cross.map(() => new Float32Array(n * d)), crossV = kv.cross.map(() => new Float32Array(n * d));
      const encKey = [new Float32Array(box * d), new Float32Array(box * d)];
      stageKV(kv, m.mem, 0, n, box, crossK, crossV, encKey, d);
      self.postMessage({ id: m.id, lo: m.lo, crossK, crossV, encKey, ms: performance.now() - t0 }, [...crossK, ...crossV, ...encKey].map(a => a.buffer));
    }
  } catch (err) {
    self.postMessage({ id: m.id, error: String(err && err.stack || err) });
  }
};
