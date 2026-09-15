// M0 on WebAssembly SIMD kernels (kernels.wasm / kernels-shared.wasm, built by web/tools/build-kernels.mjs).
//
// Every weight and activation lives in one WebAssembly memory; the kernels work on byte offsets into it. The model's thread
// runs the decoder; the encoder and the pass's key/value set-up are split by rows over helper threads (wasm-helper.js) when
// the page is cross-origin isolated (shared memory; coi-serviceworker.min.js provides it on hosts that cannot send the headers),
// else everything runs on the model's thread. Softmax stays in JS (WebAssembly has no exp). The picks, the anchor mixture,
// the canvas, free space and the world are model.js / wall.js as before.
import { EncoderPool } from "./pool.js";
import { M0, PROF, profAdd, ranges, tensorData } from "./model.js";
import { NX } from "./wall.js";

const SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);
export function simdSupported() { try { return WebAssembly.validate(SIMD_PROBE); } catch { return false; } }
const CH = 32;                                       // encoder rows per attention block (bounds the scores scratch)
const now = () => performance.now();
const P = a => (a ? a.byteOffset : 0);

// softmax over `groups` runs of S float32 numbers starting at index `off` of F (computed in float64, as model.js does)
export function softmaxGroups(F, off, groups, S, tmp) {
  for (let gi = 0, b = off; gi < groups; gi++, b += S) {
    let m = -Infinity;
    for (let s = 0; s < S; s++) if (F[b + s] > m) m = F[b + s];
    let z = 0;
    for (let s = 0; s < S; s++) { const e = Math.exp(F[b + s] - m); tmp[s] = e; z += e; }
    for (let s = 0; s < S; s++) F[b + s] = tmp[s] / z;
  }
}

// encoder rows lo..hi of one layer (E: the layer's weight offsets; B: the activations' offsets). Same order as model.js.
export function stageA(k, E, B, lo, hi, d) {
  const rows = hi - lo, o = lo * d * 4;
  k.layernorm(B.X + o, E.n1w, E.n1b, B.H + o, rows, d);
  k.linear(B.H + o, E.qW, E.qB, B.Q + o, rows, d, d);
  k.linear(B.H + o, E.kW, E.kB, B.K + o, rows, d, d);
  k.linear(B.H + o, E.vW, E.vB, B.V + o, rows, d, d);
}

export function stageB(k, F, E, B, Sc, tmp64, lo, hi, M, d, heads) {
  const hd = d / heads, scale = 1 / Math.sqrt(hd);
  for (let r0 = lo; r0 < hi; r0 += CH) {
    const rows = Math.min(CH, hi - r0), o = r0 * d * 4;
    k.attn_scores(B.Q + o, B.K, rows, M, heads, hd, scale, Sc.scores);
    softmaxGroups(F, Sc.scores >> 2, rows * heads, M, tmp64);
    k.attn_mix(Sc.scores, B.V, rows, M, heads, hd, Sc.att);
    k.linear(Sc.att, E.outW, E.outB, Sc.tmp, rows, d, d);
    k.add(B.X + o, Sc.tmp, rows * d * 4);
    k.layernorm(B.X + o, E.n2w, E.n2b, B.H + o, rows, d);
    k.ffn_add(B.H + o, E.w1, E.b1, E.w2c, E.b2, B.X + o, rows, d, 4 * d, Sc.hid);
  }
}

// the pass's set-up for memory rows lo..hi: every decoder layer's cross keys and values, the anchor keys of the box rows (< N)
export function stageKV(k, KV, B, lo, hi, N, d) {
  const rows = hi - lo, o = lo * d * 4;
  KV.cross.forEach((c, l) => {
    k.linear(B.mem + o, c.kW, c.kB, B.crossK[l] + o, rows, d, d);
    k.linear(B.mem + o, c.vW, c.vB, B.crossV[l] + o, rows, d, d);
  });
  const box = Math.max(0, Math.min(hi, N) - lo);
  if (box) for (let e = 0; e < 2; e++) k.linear(B.mem + o, KV.kEnc[e], 0, B.encKey[e] + o, box, d, d);
}

class Arena {
  constructor(memory) { this.memory = memory; this.top = 16; }
  alloc(n) {
    const off = (this.top + 3) & ~3;
    this.top = off + 4 * n;
    if (this.top > this.memory.buffer.byteLength) throw new Error("the model's memory is too small");
    return new Float32Array(this.memory.buffer, off, n);
  }
}

export class M0Wasm extends M0 {
  // manifest + weight buffer as for M0.fromBuffers; module: the compiled kernels (kernels-shared.wasm when shared)
  static create(manifest, buffer, module, { shared = false, helpers = 0, Mmax = 1536 } = {}) {
    const c = manifest.config, d = c.d, dh = 4 * d, heads = c.heads, maxT = 1 + 5 * c.chunk, threads = helpers + 1;
    const tensorFloats = manifest.tensors.reduce((a, t) => a + t.shape.reduce((x, y) => x * y, 1), 0);
    const floats = tensorFloats + (c.enc_layers + c.dec_layers) * d * dh + 2 * c.dec_layers * maxT * d
      + (6 + 2 * c.dec_layers + 2) * Mmax * d + threads * (CH * heads * Mmax + 2 * CH * d + dh)
      + 10 * d + dh + heads * Math.max(Mmax, maxT) + NX + Mmax + (4 * c.chunk + 1) * d + (1 << 16);
    const pages = Math.ceil((4 * floats + 256) / 65536) + 1;
    const memory = new WebAssembly.Memory({ initial: pages, maximum: pages, shared });
    const instance = new WebAssembly.Instance(module, { env: { memory } });
    const arena = new Arena(memory);
    const W = {};
    for (const t of manifest.tensors) { const src = tensorData(t, buffer); const v = arena.alloc(src.length); v.set(src); W[t.name] = v; }
    return new M0Wasm(W, manifest, { memory, instance, module, arena, Mmax, threads, shared, pages });
  }

  constructor(W, manifest, rt) {
    super(W, manifest);
    const A = rt.arena, d = this.d, heads = this.heads;
    this.rt = rt; this.k = rt.instance.exports; this.memory = rt.memory; this.F32 = new Float32Array(rt.memory.buffer);
    this.Mmax = rt.Mmax; this.threads = rt.threads;
    const toArena = a => { const v = A.alloc(a.length); v.set(a); return v; };
    for (const L of [...this.enc, ...this.dec]) L.w2c = toArena(L.w2c);
    this.cacheK = this.dec.map(() => A.alloc(this.maxT * d));
    this.cacheV = this.dec.map(() => A.alloc(this.maxT * d));
    this.encP = this.enc.map(L => ({ n1w: P(L.n1w), n1b: P(L.n1b), n2w: P(L.n2w), n2b: P(L.n2b), qW: P(L.qW), qB: P(L.qB), kW: P(L.kW),
      kB: P(L.kB), vW: P(L.vW), vB: P(L.vB), outW: P(L.outW), outB: P(L.outB), w1: P(L.w1), b1: P(L.b1), w2c: P(L.w2c), b2: P(L.b2) }));
    this.decP = this.dec.map((L, l) => ({ n1w: P(L.n1w), n1b: P(L.n1b), n2w: P(L.n2w), n2b: P(L.n2b), n3w: P(L.n3w), n3b: P(L.n3b),
      qkvW: P(W[`decoder.layers.${l}.self_attn.in_proj_weight`]), qkvB: P(W[`decoder.layers.${l}.self_attn.in_proj_bias`]),
      outW: P(L.outW), outB: P(L.outB), cqW: P(L.cqW), cqB: P(L.cqB), coW: P(L.coW), coB: P(L.coB), w1: P(L.w1), b1: P(L.b1), w2c: P(L.w2c), b2: P(L.b2) }));
    this.kvP = { cross: this.kv.cross.map(c => ({ kW: P(c.kW), kB: P(c.kB), vW: P(c.vW), vB: P(c.vB) })), kEnc: this.kv.kEnc.map(P) };
    const Mm = this.Mmax, act = this.act = {
      X: A.alloc(Mm * d), H: A.alloc(Mm * d), Q: A.alloc(Mm * d), K: A.alloc(Mm * d), V: A.alloc(Mm * d), mem: A.alloc(Mm * d),
      crossK: this.dec.map(() => A.alloc(Mm * d)), crossV: this.dec.map(() => A.alloc(Mm * d)), encKey: [A.alloc(Mm * d), A.alloc(Mm * d)] };
    this.actP = { X: P(act.X), H: P(act.H), Q: P(act.Q), K: P(act.K), V: P(act.V), mem: P(act.mem),
      crossK: act.crossK.map(P), crossV: act.crossV.map(P), encKey: act.encKey.map(P) };
    this.scP = Array.from({ length: this.threads }, () => ({ scores: P(A.alloc(CH * heads * Mm)), att: P(A.alloc(CH * d)), tmp: P(A.alloc(CH * d)), hid: P(A.alloc(this.dh)) }));
    this.tmp64 = new Float64Array(Math.max(Mm, this.maxT));
    this.D = { y: A.alloc(d), h1: A.alloc(d), qkv: A.alloc(3 * d), q: A.alloc(d), att: A.alloc(d), tmp: A.alloc(d), h: A.alloc(d),
      hid: A.alloc(this.dh), scores: A.alloc(heads * Math.max(Mm, this.maxT)) };
    this.rulerF32 = A.alloc(NX); this.keyF32 = A.alloc(Mm); this.keys = A.alloc((4 * this.chunk + 1) * d);
    this.bufs = {};
    this.encNormP = [P(W["enc_norm.weight"]), P(W["enc_norm.bias"])];
    this.decNormP = [P(W["dec_norm.weight"]), P(W["dec_norm.bias"])];
  }

  get backend() { return "wasm"; }

  buf(name, n) {
    const b = this.bufs[name];
    if (b && b.length === n) return b;
    return (this.bufs[name] = this.rt.arena.alloc(n));
  }
  keyBuf(i) { return this.keys.subarray(i * this.d, (i + 1) * this.d); }
  lin(x, xo, Wt, b, y, yo, din, dout) { this.k.linear(x.byteOffset + 4 * xo, Wt.byteOffset, P(b), y.byteOffset + 4 * yo, 1, din, dout); }
  rulerLogits(u, tab, n, ruler) {
    this.k.linear(P(u), P(tab), 0, P(this.rulerF32), 1, this.d, n);
    const r = this.rulerF32;
    for (let t = 0; t < n; t++) ruler[t] = r[t];
  }
  keyDots(qa, kmat, N, out, off) {
    this.k.linear(P(qa), P(kmat), 0, P(this.keyF32), 1, this.d, N);
    const r = this.keyF32;
    for (let i = 0; i < N; i++) out[off + i] = r[i];
  }

  makeStep(enc) {
    const { d, heads, hd, k, F32: F, D, tmp64 } = this;
    const M = enc.M, scale = 1 / Math.sqrt(hd), prof = PROF.on;
    const crossK = enc.crossK.map(P), crossV = enc.crossV.map(P), cK = this.cacheK, cV = this.cacheV;
    const y = P(D.y), h1 = P(D.h1), qkv = P(D.qkv), q = P(D.q), att = P(D.att), tmp = P(D.tmp), h = P(D.h), hid = P(D.hid), sc = P(D.scores);
    let T = 0;
    return inp => {
      D.y.set(inp);
      let ts = prof ? now() : 0;
      const seg = name => { if (prof) { const t2 = now(); profAdd(name, t2 - ts); ts = t2; } };
      for (let l = 0; l < this.decP.length; l++) {
        const E = this.decP[l];
        k.layernorm(y, E.n1w, E.n1b, h1, 1, d);
        k.linear(h1, E.qkvW, E.qkvB, qkv, 1, d, 3 * d);
        cK[l].set(D.qkv.subarray(d, 2 * d), T * d);
        cV[l].set(D.qkv.subarray(2 * d, 3 * d), T * d);
        k.attn_scores(qkv, P(cK[l]), 1, T + 1, heads, hd, scale, sc);
        softmaxGroups(F, sc >> 2, heads, T + 1, tmp64);
        k.attn_mix(sc, P(cV[l]), 1, T + 1, heads, hd, att);
        k.linear(att, E.outW, E.outB, tmp, 1, d, d);
        k.add(y, tmp, 4 * d);
        seg("dec.self-attention (norm, qkv, attend, out)");
        k.layernorm(y, E.n2w, E.n2b, h1, 1, d);
        k.linear(h1, E.cqW, E.cqB, q, 1, d, d);
        k.attn_scores(q, crossK[l], 1, M, heads, hd, scale, sc);
        softmaxGroups(F, sc >> 2, heads, M, tmp64);
        k.attn_mix(sc, crossV[l], 1, M, heads, hd, att);
        k.linear(att, E.coW, E.coB, tmp, 1, d, d);
        k.add(y, tmp, 4 * d);
        seg("dec.cross-attention (norm, q, attend, out)");
        k.layernorm(y, E.n3w, E.n3b, h1, 1, d);
        k.ffn_add(h1, E.w1, E.b1, E.w2c, E.b2, y, 1, d, 4 * d, hid);
        seg("dec.feed-forward (norm, 1024 relu, 256)");
      }
      T += 1;
      k.layernorm(y, this.decNormP[0], this.decNormP[1], h, 1, d);
      seg("dec.final norm");
      if (prof) { profAdd("steps", 0, 1); profAdd("steps.self positions", 0, T); profAdd("steps.cross positions", 0, M); }
      return D.h;
    };
  }

  async encode(tok, brief, pool = null, split = 1) {
    const { d, heads, k } = this, prof = PROF.on;
    let t = prof ? now() : 0;
    const tick = name => { if (prof) { const t2 = now(); profAdd(name, t2 - t); t = t2; } };
    const { X, N, M, bins, cover } = this.embed(tok, brief);
    if (M > this.Mmax) throw new Error(`the wall has ${M} tokens, more than the ${this.Mmax} the page makes room for`);
    this.act.X.set(X.subarray(0, M * d));
    tick("enc.embed+canvas");
    const B = this.actP;
    const rr = ranges(M, pool ? pool.size + 1 : split);
    const mine = pool ? [rr[0]] : [...rr].reverse(), helper = pool ? rr.slice(1) : [];
    const wait = async jobs => { for (const r of await Promise.all(jobs)) if (prof) profAdd("helpers.compute", r.ms); tick("enc.waiting for helpers"); };
    for (let l = 0; l < this.encP.length; l++) {
      let jobs = helper.map(([lo, hi], i) => pool.run(i, { op: "A", l, lo, hi }));
      for (const [lo, hi] of mine) stageA(k, this.encP[l], B, lo, hi, d);
      tick("enc.norm1+qkv");
      await wait(jobs);
      jobs = helper.map(([lo, hi], i) => pool.run(i, { op: "B", l, lo, hi, M }));
      for (const [lo, hi] of mine) stageB(k, this.F32, this.encP[l], B, this.scP[0], this.tmp64, lo, hi, M, d, heads);
      tick("enc.attention+out+norm2+ff");
      await wait(jobs);
    }
    k.layernorm(B.X, this.encNormP[0], this.encNormP[1], B.mem, M, d);
    tick("enc.final norm");
    const jobs = helper.map(([lo, hi], i) => pool.run(i, { op: "KV", lo, hi, N }));
    for (const [lo, hi] of mine) stageKV(k, this.kvP, B, lo, hi, N, d);
    tick("pass.cross keys/values + anchor keys");
    await wait(jobs);
    return { mem: this.act.mem, N, M, bins, cover, crossK: this.act.crossK, crossV: this.act.crossV, encKey: this.act.encKey };
  }
}

// helper threads sharing the model's memory: each runs encoder and key/value rows on its own scratch
export class WasmPool extends EncoderPool {
  static async create(model, n, url = new URL("./wasm-helper.js", import.meta.url)) {
    if (n < 1 || typeof Worker === "undefined" || !model.rt.shared) return null;
    const workers = [];
    try {
      for (let h = 0; h < n; h++) workers.push(new Worker(url, { type: "module" }));
      const pool = new WasmPool(workers);
      await Promise.all(workers.map((w, h) => pool.run(h, { op: "init", module: model.rt.module, memory: model.memory, encP: model.encP,
        kvP: model.kvP, B: model.actP, Sc: model.scP[h + 1], d: model.d, heads: model.heads, Mmax: model.Mmax })));
      return pool;
    } catch (e) {
      workers.forEach(w => w.terminate());
      return null;
    }
  }
}
