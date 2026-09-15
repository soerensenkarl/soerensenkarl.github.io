// Helper threads for M0's encoder: each holds a copy of the encoder layers and of the per-row key/value weights, and works
// on a range of rows (model.js stageA, stageB, stageKV). The model's own thread does the first range. Same numbers as one thread.
import { HELPER_TENSOR } from "./model.js";

export class EncoderPool {
  constructor(workers) {
    this.workers = workers;
    this.size = workers.length;
    this.pending = new Map();
    this.next = 0;
    workers.forEach(w => {
      w.onmessage = e => {
        const p = this.pending.get(e.data.id);
        this.pending.delete(e.data.id);
        if (p) e.data.error ? p.reject(new Error(e.data.error)) : p.resolve(e.data);
      };
      w.onerror = e => { for (const p of this.pending.values()) p.reject(new Error(e.message || "helper failed")); this.pending.clear(); };
    });
  }

  // n helper threads for `model` (null when n < 1 or threads cannot start here)
  static async create(model, n, url = new URL("./encoder-helper.js", import.meta.url)) {
    if (n < 1 || typeof Worker === "undefined") return null;
    const workers = [];
    try {
      for (let h = 0; h < n; h++) workers.push(new Worker(url, { type: "module" }));
      const pool = new EncoderPool(workers);
      await Promise.all(workers.map((w, h) => {
        const tensors = {}, transfer = [];
        for (const [name, a] of Object.entries(model.W)) if (HELPER_TENSOR(name)) { tensors[name] = a.slice(); transfer.push(tensors[name].buffer); }
        return pool.run(h, { type: "init", tensors, config: model.manifest.config }, transfer);
      }));
      return pool;
    } catch (e) {
      workers.forEach(w => w.terminate());
      return null;
    }
  }

  run(h, msg, transfer = []) {
    const id = ++this.next;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.workers[h].postMessage({ ...msg, id }, transfer);
    });
  }

  terminate() { this.workers.forEach(w => w.terminate()); }
}
