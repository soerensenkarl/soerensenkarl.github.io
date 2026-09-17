// M0 in plain JavaScript: the forward pass of automake/mvp/network.py (MVPEditor, encoder-decoder, anchored picks, briefs,
// free space tokens, the 40 cm canvas, no pick position vectors, passes of 16), written from the state dict as
// automake/app/m0_facts.py ReimplM0 does, with the decoder's key/value cache of MVPEditor._stepper.
//
//   const model = M0.fromBuffers(manifest, arrayBuffer);
//   const enc = await model.encode(tokens, brief, pool);   // tokens from wall.js tokens(); pool optional (pool.js)
//   for (const part of model.writePass(enc)) ...          // up to 16 parts: {elem: [item, x0, y0, x1, y1], picks: [...]}
//
// The encoder and the pass's key/value set-up work row by row (every row's numbers depend only on its own inputs and on the
// layer's keys and values), so they can be split over worker threads (encoder-helper.js) and still give the same numbers.
// No dependencies. Weights are float32 in memory whatever the file stores (f32, f16, or int8 with a scale per row).

import { NX, NY, POINT_KN, Q, SECTIONS, X_MAX, Y_MAX, rint } from "./wall.js";
import { IMPLIED, MAX_POLY, N_CODE, classOf, itemOf, segBox } from "./segment.js";

const f32 = Math.fround;

// encode.sec_features: what an item is to the network - across the member's axis what it shows in the elevation,
// along the wall's normal what goes through it, each read as its log against the one timber width. Logs because the
// two numbers span a factor of six and a linear map of raw metres has almost nothing to spread (round G2: the item
// head learnt nothing while the class and the cuts learnt everything). Through float32, as numpy holds them.
const SEC_REF = 0.045;
export function itemSec(name) {
  const [w, depth, lay] = SECTIONS[name], a = lay === "flat" ? w : depth, b = lay === "flat" ? depth : w;
  return [f32(Math.log(f32(f32(a) / f32(SEC_REF)))), f32(Math.log(f32(f32(b) / f32(SEC_REF))))];
}

// erf to double precision, by the series with no cancellation in it - erf(z) = 2/sqrt(pi) e^(-z^2) sum 2^n z^(2n+1) /
// (1.3.5...(2n+1)), every term positive, where the alternating series loses its digits - and 1 beyond |z| 6. It is
// here for the exact GELU (torch's default, the error function and not the tanh) that the section map goes through.
function erf(z) {
  const a = Math.abs(z);
  if (a > 6) return Math.sign(z);
  let term = a, sum = a;
  for (let n = 1; n < 200 && term > 1e-18 * sum; n++) {
    term *= 2 * a * a / (2 * n + 1);
    sum += term;
  }
  return Math.sign(z) * 1.1283791670955126 * Math.exp(-a * a) * sum;   // 2 / sqrt(pi)
}
const gelu = x => x * 0.5 * (1 + erf(x / Math.SQRT2));

// What the encoder hands the pass, and the one shape both backends must return: `wasm.js` overrides `encode()`
// wholesale, so a field added here would otherwise go missing there in silence - which is how the item mask reached
// one backend and not the other. `itemOk` is the one optional field (null when the design named no stock).
const ENCODED = ["mem", "N", "M", "bins", "cover", "crossK", "crossV", "encKey"];
export function encoded(e) {
  const missing = ENCODED.filter(k => e[k] === undefined);
  if (missing.length) throw new Error(`the encoder returned no ${missing.join(", ")} (model.js encoded)`);
  return { itemOk: null, ...e };
}

// optional timing of every stage (web/tests/profile.js, the page with ?prof): PROF.t[name] ms, PROF.n[name] calls
export const PROF = { on: false, t: {}, n: {} };
export function profAdd(name, ms, calls = 1) { PROF.t[name] = (PROF.t[name] || 0) + ms; PROF.n[name] = (PROF.n[name] || 0) + calls; }
export function profReset(on) { PROF.on = on; PROF.t = {}; PROF.n = {}; }
const now = () => performance.now();

let F16 = null;
function f16Table() {                                   // every half-precision bit pattern as a float32, built once
  if (F16) return F16;
  F16 = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    F16[h] = e === 0 ? s * m * 2 ** -24 : (e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15));
  }
  return F16;
}

export function tensorData(t, buffer) {
  const n = t.shape.reduce((a, b) => a * b, 1);
  if (t.dtype === "f32") return new Float32Array(buffer, t.offset, n);
  if (t.dtype === "f16") {
    const src = new Uint16Array(buffer, t.offset, n), out = new Float32Array(n), table = f16Table();
    for (let i = 0; i < n; i++) out[i] = table[src[i]];
    return out;
  }
  if (t.dtype === "int8") {
    const rows = t.shape[0], cols = n / rows;
    const scale = new Float32Array(buffer, t.scale_offset, rows), q = new Int8Array(buffer, t.offset, n), out = new Float32Array(n);
    for (let r = 0; r < rows; r++) { const s = scale[r], o = r * cols; for (let c = 0; c < cols; c++) out[o + c] = q[o + c] * s; }
    return out;
  }
  throw new Error("unknown dtype " + t.dtype);
}

// ---------------------------------------------------------------- kernels
function linear(x, xo, W, b, y, yo, din, dout) {       // y = W x + b for one row
  for (let j = 0, wo = 0; j < dout; j++, wo += din) {
    let s = b ? b[j] : 0, k = 0;
    for (; k + 3 < din; k += 4) s += W[wo + k] * x[xo + k] + W[wo + k + 1] * x[xo + k + 1] + W[wo + k + 2] * x[xo + k + 2] + W[wo + k + 3] * x[xo + k + 3];
    for (; k < din; k++) s += W[wo + k] * x[xo + k];
    y[yo + j] = s;
  }
}

function layerNorm(x, xo, w, b, y, yo, d) {
  let mean = 0;
  for (let k = 0; k < d; k++) mean += x[xo + k];
  mean /= d;
  let v = 0;
  for (let k = 0; k < d; k++) { const t = x[xo + k] - mean; v += t * t; }
  const inv = 1 / Math.sqrt(v / d + 1e-5);
  for (let k = 0; k < d; k++) y[yo + k] = (x[xo + k] - mean) * inv * w[k] + b[k];
}

function softmaxInPlace(a, o, n) {                     // a[o..o+n) -> probabilities (float64 buffer)
  let m = -Infinity;
  for (let i = 0; i < n; i++) if (a[o + i] > m) m = a[o + i];
  let s = 0;
  for (let i = 0; i < n; i++) { const e = Math.exp(a[o + i] - m); a[o + i] = e; s += e; }
  for (let i = 0; i < n; i++) a[o + i] /= s;
}

// feed-forward: y += W2 relu(W1 x + b1) + b2, with W2 by columns so zero activations cost nothing
function feedForward(x, xo, L, hid, y, yo, d, dh) {
  linear(x, xo, L.w1, L.b1, hid, 0, d, dh);
  for (let j = 0; j < d; j++) y[yo + j] += L.b2[j];
  const W2c = L.w2c;
  for (let k = 0; k < dh; k++) {
    const h = hid[k];
    if (h <= 0) continue;
    const co = k * d;
    for (let j = 0; j < d; j++) y[yo + j] += h * W2c[co + j];
  }
}

// multi-head attention of one query row over S keys/values (heads split along the vector), before out_proj
function attendRow(q, qo, K, V, S, heads, hd, out, oo, scores) {
  const scale = 1 / Math.sqrt(hd), d = hd * heads;
  for (let h = 0; h < heads; h++) {
    const ho = h * hd;
    let m = -Infinity;
    for (let s = 0, so = ho; s < S; s++, so += d) {
      let dot = 0;
      for (let t = 0; t < hd; t++) dot += q[qo + ho + t] * K[so + t];
      dot *= scale;
      scores[s] = dot;
      if (dot > m) m = dot;
    }
    let z = 0;
    for (let s = 0; s < S; s++) { const e = Math.exp(scores[s] - m); scores[s] = e; z += e; }
    for (let t = 0; t < hd; t++) out[oo + ho + t] = 0;
    for (let s = 0, so = ho; s < S; s++, so += d) {
      const w = scores[s] / z;
      if (w < 1e-12) continue;
      for (let t = 0; t < hd; t++) out[oo + ho + t] += w * V[so + t];
    }
  }
}

// ---------------------------------------------------------------- weights by layer
const byColumns = (w, rows, cols) => {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) for (let k = 0; k < cols; k++) out[k * rows + r] = w[r * cols + k];
  return out;
};

export function encoderLayer(W, l, d) {
  const p = `encoder.layers.${l}`, inW = W[p + ".self_attn.in_proj_weight"], inB = W[p + ".self_attn.in_proj_bias"];
  return { n1w: W[p + ".norm1.weight"], n1b: W[p + ".norm1.bias"], n2w: W[p + ".norm2.weight"], n2b: W[p + ".norm2.bias"],
    qW: inW.subarray(0, d * d), qB: inB.subarray(0, d), kW: inW.subarray(d * d, 2 * d * d), kB: inB.subarray(d, 2 * d),
    vW: inW.subarray(2 * d * d), vB: inB.subarray(2 * d),
    outW: W[p + ".self_attn.out_proj.weight"], outB: W[p + ".self_attn.out_proj.bias"],
    w1: W[p + ".linear1.weight"], b1: W[p + ".linear1.bias"], b2: W[p + ".linear2.bias"], w2c: byColumns(W[p + ".linear2.weight"], d, 4 * d) };
}

// what a pass needs from the memory, per row: every decoder layer's cross-attention keys and values, the anchor's box-edge keys
export function kvWeights(W, decLayers, d) {
  return {
    cross: Array.from({ length: decLayers }, (_, l) => {
      const p = `decoder.layers.${l}.multihead_attn`, cw = W[p + ".in_proj_weight"], cb = W[p + ".in_proj_bias"];
      return { kW: cw.subarray(d * d, 2 * d * d), kB: cb.subarray(d, 2 * d), vW: cw.subarray(2 * d * d, 3 * d * d), vB: cb.subarray(2 * d, 3 * d) };
    }),
    kEnc: [W["anchor.k_enc.0.weight"], W["anchor.k_enc.1.weight"]],
  };
}

// the tensors a helper thread needs (encoder layers, cross-attention keys/values, anchor box keys)
export const HELPER_TENSOR = name => name.startsWith("encoder.layers.") || /^decoder\.layers\.\d+\.multihead_attn\.in_proj/.test(name)
  || name.startsWith("anchor.k_enc.");

export function makeScratch(d, M) {
  return { att: new Float32Array(d), tmp: new Float32Array(d), hid: new Float32Array(4 * d), scores: new Float64Array(Math.max(M, 1)) };
}

// stage A of an encoder layer for rows lo..hi: norm1, then queries, keys, values
export function stageA(L, X, H, Qm, Km, Vm, lo, hi, d) {
  for (let i = lo; i < hi; i++) {
    const o = i * d;
    layerNorm(X, o, L.n1w, L.n1b, H, o, d);
    linear(H, o, L.qW, L.qB, Qm, o, d, d);
    linear(H, o, L.kW, L.kB, Km, o, d, d);
    linear(H, o, L.vW, L.vB, Vm, o, d, d);
  }
}

// stage B for rows lo..hi (of X, H, Qm) over all M keys and values: attention, residual, norm2, feed-forward, residual
export function stageB(L, X, H, Qm, Km, Vm, M, lo, hi, d, heads, s) {
  const hd = d / heads, prof = PROF.on;
  if (s.scores.length < M) s.scores = new Float64Array(M);
  let tA = 0, tF = 0;
  for (let i = lo; i < hi; i++) {
    const o = i * d;
    const t0 = prof ? now() : 0;
    attendRow(Qm, o, Km, Vm, M, heads, hd, s.att, 0, s.scores);
    const t1 = prof ? now() : 0;
    linear(s.att, 0, L.outW, L.outB, s.tmp, 0, d, d);
    for (let k = 0; k < d; k++) X[o + k] += s.tmp[k];
    layerNorm(X, o, L.n2w, L.n2b, H, o, d);
    feedForward(H, o, L, s.hid, X, o, d, 4 * d);
    if (prof) { tA += t1 - t0; tF += now() - t1; }
  }
  if (prof) { profAdd("enc.attention", tA, hi - lo); profAdd("enc.out+norm2+ff", tF, hi - lo); }
}

// the pass's set-up for rows lo..hi of the memory: cross keys and values of every decoder layer; anchor keys for box rows (< boxRows)
export function stageKV(KV, mem, lo, hi, boxRows, crossK, crossV, encKey, d) {
  for (let i = lo; i < hi; i++) {
    const o = i * d;
    KV.cross.forEach((c, l) => { linear(mem, o, c.kW, c.kB, crossK[l], o, d, d); linear(mem, o, c.vW, c.vB, crossV[l], o, d, d); });
    if (i < boxRows) for (let e = 0; e < 2; e++) linear(mem, o, KV.kEnc[e], null, encKey[e], o, d, d);
  }
}

export const ranges = (n, parts) => {                          // split 0..n into `parts` contiguous ranges
  const out = [], step = Math.ceil(n / parts);
  for (let lo = 0; lo < n; lo += step) out.push([lo, Math.min(n, lo + step)]);
  return out;
};

// ---------------------------------------------------------------- the network
export class M0 {
  static fromBuffers(manifest, buffer) {
    const W = {};
    for (const t of manifest.tensors) W[t.name] = tensorData(t, buffer);
    return new M0(W, manifest);
  }

  constructor(W, manifest) {
    const c = manifest.config;
    this.manifest = manifest;
    this.W = W;
    this.items = manifest.items;
    this.d = c.d; this.heads = c.heads; this.hd = c.d / c.heads; this.dh = 4 * c.d;
    this.K = c.anchor_k; this.P = c.canvas; this.chunk = c.chunk;
    this.nItems = manifest.items.length;
    // The token format the checkpoint was trained in (automake/mvp/legacy.py TOKENS). "sections": a member is a
    // segment, a class and the ends it declares cut, an item is its own section rather than a row of weights, the
    // skin is a polygon and the y ruler is 4.8 m, not 3.2 (js/segment.js). "rect", or none, is every network
    // published before 2026-09-16. Anything else is refused outright rather than read as if it were one of these:
    // the page then leaves that network out (app.js dropModel) instead of drawing a wall off the wrong weights.
    const tokens = c.tokens || "rect";
    if (tokens !== "rect" && tokens !== "sections")
      throw new Error(`this build does not speak the "${tokens}" token format (js/segment.js does "sections")`);
    this.segment = tokens === "sections";
    // and it must carry the weights that format implies. A checkpoint exported before the section map went deep has
    // `sec_emb.weight` where this wants `sec_emb.0.weight`, and read its items as raw metres rather than logs. It is
    // refused here, while the network loads, so the page leaves that artifact out (app.js dropModel) - the same safe
    // failure as a format it does not speak. Refused later, on the first wall, the page would keep a broken artifact.
    if (this.segment && !W["sec_emb.0.weight"])
      throw new Error("this checkpoint predates the deeper section map (no sec_emb.0.weight): re-export it from a "
        + "Python tree that has encode.sec_features");
    // whether it was built to read the items a design may be built from: if it was, an inventory also puts every
    // other item out of reach of the first head (MVPEditor.item_mask), which is what `itemMask` below builds.
    this.inventory = c.inventory === true;
    this.nCodes = this.segment ? this.nItems * N_CODE : this.nItems;
    this.STOP = this.nCodes;
    this.ruler = this.segment ? [X_MAX, 2.4, X_MAX, 2.4] : [X_MAX, Y_MAX, X_MAX, Y_MAX];
    this.rulerNy = this.segment ? 960 : NY;
    const d = this.d;
    this.enc = Array.from({ length: c.enc_layers }, (_, l) => encoderLayer(W, l, d));
    this.kv = kvWeights(W, c.dec_layers, d);
    this.dec = Array.from({ length: c.dec_layers }, (_, l) => {
      const p = `decoder.layers.${l}`, inW = W[p + ".self_attn.in_proj_weight"], inB = W[p + ".self_attn.in_proj_bias"];
      const cw = W[p + ".multihead_attn.in_proj_weight"], cb = W[p + ".multihead_attn.in_proj_bias"];
      return { n1w: W[p + ".norm1.weight"], n1b: W[p + ".norm1.bias"], n2w: W[p + ".norm2.weight"], n2b: W[p + ".norm2.bias"],
        n3w: W[p + ".norm3.weight"], n3b: W[p + ".norm3.bias"],
        qW: inW.subarray(0, d * d), qB: inB.subarray(0, d), kW: inW.subarray(d * d, 2 * d * d), kB: inB.subarray(d, 2 * d),
        vW: inW.subarray(2 * d * d), vB: inB.subarray(2 * d), outW: W[p + ".self_attn.out_proj.weight"], outB: W[p + ".self_attn.out_proj.bias"],
        cqW: cw.subarray(0, d * d), cqB: cb.subarray(0, d), coW: W[p + ".multihead_attn.out_proj.weight"], coB: W[p + ".multihead_attn.out_proj.bias"],
        w1: W[p + ".linear1.weight"], b1: W[p + ".linear1.bias"], b2: W[p + ".linear2.bias"], w2c: byColumns(W[p + ".linear2.weight"], d, this.dh) };
    });
    this.rulers = [W["x_emb.weight"], W["y_emb.weight"], W["x_emb.weight"], W["y_emb.weight"]];   // by edge x0, y0, x1, y1
    this.rulerN = [NX, this.rulerNy, NX, this.rulerNy];
    this.projCache = [0, 1, 2, 3].map(f => ({ v: new Float32Array(this.rulerN[f] * d), ok: new Uint8Array(this.rulerN[f]) }));
    this.maxT = 1 + 5 * this.chunk;
    this.cacheK = this.dec.map(() => new Float32Array(this.maxT * d));
    this.cacheV = this.dec.map(() => new Float32Array(this.maxT * d));
    this.scratch = makeScratch(d, 512);
  }

  // scratch and small operations a faster backend overrides (wasm.js)
  buf(name, n) { return new Float32Array(n); }
  keyBuf(i) { return new Float32Array(this.d); }
  lin(x, xo, Wt, b, y, yo, din, dout) { linear(x, xo, Wt, b, y, yo, din, dout); }
  rulerLogits(u, tab, n, ruler) {                     // the tied ruler head: u . tick row, for every tick
    const d = this.d;
    for (let t = 0, to = 0; t < n; t++, to += d) {
      let s = 0, k = 0;
      for (; k + 3 < d; k += 4) s += u[k] * tab[to + k] + u[k + 1] * tab[to + k + 1] + u[k + 2] * tab[to + k + 2] + u[k + 3] * tab[to + k + 3];
      ruler[t] = s;
    }
  }
  keyDots(qa, kmat, N, out, off) {                    // the anchor query against every box-edge key
    const d = this.d;
    for (let i = 0, ko = 0; i < N; i++, ko += d) {
      let dot = 0, k = 0;
      for (; k + 3 < d; k += 4) dot += qa[k] * kmat[ko + k] + qa[k + 1] * kmat[ko + k + 1] + qa[k + 2] * kmat[ko + k + 2] + qa[k + 3] * kmat[ko + k + 3];
      out[off + i] = dot;
    }
  }

  // edge_proj[f](ruler_f[bin]), cached per tick
  edgeVec(f, bin) {
    const c = this.projCache[f], d = this.d;
    if (!c.ok[bin]) { linear(this.rulers[f], bin * d, this.W[`edge_proj.${f}.weight`], null, c.v, bin * d, d, d); c.ok[bin] = 1; }
    return bin * d;
  }

  rectBins(r) {                                        // MVPEditor.rect_bins on a float32 rectangle
    const scale = this.ruler, n = this.rulerN;
    return r.map((v, f) => Math.min(Math.max(rint(f32(f32(v + f32(scale[f])) / f32(Q))), 0), n[f] - 1));
  }

  // MVPEditor.code_table: one vector per thing the first field can pick - the item's section through one map
  // (`sec_emb`), its class and its cuts through one embedding (`code_emb`). Shared by the input tokens, the decoder's
  // inputs and the output head, exactly as a ruler tick is, and nothing at all is learned per item: a section the
  // network never saw in training is read and written like one it did. Built once, from weights that never change.
  codeTable() {
    if (this.code) return this.code;
    const { d, W } = this;
    const table = this.buf("codeTable", this.nCodes * d);          // in the arena, for the WebAssembly backend
    const sec = new Float32Array(2), mid = new Float32Array(d), row = new Float32Array(d), ce = W["code_emb.weight"];
    for (let i = 0; i < this.nItems; i++) {
      const s = itemSec(this.items[i]);
      sec[0] = s[0]; sec[1] = s[1];
      // sec_emb: Linear(2, d), exact GELU, Linear(d, d), LayerNorm(d). Read in JavaScript on either backend - the
      // first map's input is two floats and the SIMD kernel takes eight at a time - and built once, so it costs
      // nothing to do here. The code's own row is added last, per row of the repeated table.
      linear(sec, 0, W["sec_emb.0.weight"], W["sec_emb.0.bias"], mid, 0, 2, d);
      for (let k = 0; k < d; k++) mid[k] = gelu(mid[k]);
      linear(mid, 0, W["sec_emb.2.weight"], W["sec_emb.2.bias"], row, 0, d, d);
      layerNorm(row, 0, W["sec_emb.3.weight"], W["sec_emb.3.bias"], row, 0, d);
      for (let c = 0; c < N_CODE; c++) {
        const o = (i * N_CODE + c) * d, co = c * d;
        for (let k = 0; k < d; k++) table[o + k] = row[k] + ce[co + k];
      }
    }
    return (this.code = table);
  }

  // MVPEditor.item_mask: which of the first head's codes the design allows - every code of every item in its own
  // inventory, and STOP, always. The inventory tokens say which items those are (type 6, the item on the same code a
  // member carries), so nothing beyond the tokens has to travel. null when there is nothing to mask.
  itemMask(tok) {
    if (!this.inventory) return null;
    const want = new Set();
    for (let i = 0; i < tok.types.length; i++) if (tok.types[i] === 6) want.add(itemOf(tok.items[i] - 1));
    if (!want.size) return null;                        // a design with no inventory of its own may use every item
    const ok = new Uint8Array(this.nCodes + 1);         // whatever a code packs besides the item, the format says so
    for (let c = 0; c < this.nCodes; c++) if (want.has(itemOf(c))) ok[c] = 1;
    ok[this.nCodes] = 1;
    return ok;
  }

  // the box tokens and the canvas as vectors (M x d), their tick bins, the canvas cover
  embed(tok, brief) {
    const { d, W, P } = this;
    const NYr = this.rulerNy;
    const N = tok.types.length, gx = NX / P, gy = NYr / P, G = gx * gy, M = N + G;
    const X = new Float32Array(M * d);
    const bins = tok.rects.map(r => this.rectBins(r));
    // what each token fills: a segment token grown by its thickness (sequence.token_boxes), any other one as it is
    const boxBins = this.segment ? tok.rects.map((r, i) => this.rectBins(segBox(r, tok.thick[i]))) : bins;
    const inside = this.segment ? this.insidePoly(tok, bins, gx, gy, P) : null;
    const add = (dst, o, src, so) => { for (let k = 0; k < d; k++) dst[o + k] += src[so + k]; };
    for (let i = 0; i < N; i++) {
      const o = i * d;
      add(X, o, W["type_emb.weight"], tok.types[i] * d);
      // the code a token carries, on the same table the head picks from; token item 0 ("none") is a zero row
      if (!this.segment) add(X, o, W["item_in.weight"], tok.items[i] * d);
      else if (tok.items[i] > 0) add(X, o, this.codeTable(), (tok.items[i] - 1) * d);
      add(X, o, W["brief_emb.weight"], (i === 0 ? brief : 0) * d);
      // how heavy a load is (network.embed_rects): load_mag * log(kN / POINT_KN), which a POINT_KN load makes exactly
      // zero, so a wall of ordinary loads is bit-for-bit what it was before the channel existed
      const kn = tok.kn ? tok.kn[i] : 0;
      if (kn > 0 && W["load_mag.weight"]) {
        const m = f32(Math.log(f32(kn / POINT_KN)));
        if (m !== 0) { const g = W["load_mag.weight"]; for (let k = 0; k < d; k++) X[o + k] = f32(X[o + k] + f32(g[k] * m)); }
      }
      for (let f = 0; f < 4; f++) add(X, o, this.projCache[f].v, this.edgeVec(f, bins[i][f]));
    }
    // canvas: per patch, the share covered by parts, openings and the wall (row by row from the bottom left)
    const cover = new Float32Array(G * 3);
    const ci = W["canvas_in.weight"], cb = W["canvas_in.bias"], ct = W["canvas_type"];
    for (let row = 0, g = 0; row < gy; row++) for (let col = 0; col < gx; col++, g++) {
      const px0 = col * P, py0 = row * P;
      let cp = 0, co = 0, cw = 0;
      for (let i = 0; i < N; i++) {
        const t = tok.types[i];
        if (t > 2 || (this.segment && t === 0)) continue;   // the wall's own share is the outline's, measured below
        const b = boxBins[i];
        const ox = Math.max(Math.min(b[2], px0 + P) - Math.max(b[0], px0), 0);
        if (!ox) continue;
        const oy = Math.max(Math.min(b[3], py0 + P) - Math.max(b[1], py0), 0);
        if (!oy) continue;
        const share = f32(f32(ox * oy) / (P * P));
        if (t === 2) cp = f32(cp + share); else if (t === 1) co = f32(co + share); else cw = f32(cw + share);
      }
      if (this.segment) cw = inside[g];
      cover[g * 3] = cp; cover[g * 3 + 1] = co; cover[g * 3 + 2] = cw;
      const o = (N + g) * d;
      for (let k = 0; k < d; k++) X[o + k] = ct[k] + (ci[k * 3] * cp + ci[k * 3 + 1] * co + ci[k * 3 + 2] * cw + cb[k]);
      const edges = [px0, py0, Math.min(px0 + P, NX - 1), Math.min(py0 + P, NYr - 1)];
      for (let f = 0; f < 4; f++) add(X, o, this.projCache[f].v, this.edgeVec(f, edges[f]));
    }
    return { X, N, M, bins, cover };
  }

  // MVPEditor._inside_poly: the share of each patch that lies inside the skin's outline, by ray crossing over the
  // outline's edge tokens (the first MAX_POLY, type 0), sampled CANVAS_SUB x CANVAS_SUB per patch
  insidePoly(tok, bins, gx, gy, P) {
    const S = 4, G = gx * gy, out = new Float32Array(G);
    const e = [];
    for (let i = 0; i < Math.min(MAX_POLY, tok.types.length); i++) if (tok.types[i] === 0) e.push(bins[i]);
    const step = Array.from({ length: S }, (_, i) => (i + 0.5) * (P / S));
    for (let row = 0, g = 0; row < gy; row++) for (let col = 0; col < gx; col++, g++) {
      let hit = 0;
      for (const dx of step) for (const dy of step) {
        const px = col * P + dx, py = row * P + dy;
        let c = 0;
        for (const [x0, y0, x1, y1] of e) {
          if ((y0 <= py) === (y1 <= py)) continue;
          const dyy = Math.abs(y1 - y0) < 1e-9 ? 1.0 : y1 - y0;
          if (px < x0 + (py - y0) * (x1 - x0) / dyy) c++;
        }
        if (c % 2 === 1) hit++;
      }
      out[g] = f32(hit / (S * S));
    }
    return out;
  }

  // the encoder, then the pass's keys and values. pool: an EncoderPool (pool.js) or null; split (tests only): process the rows
  // in this many ranges, last range first, to show the split changes nothing
  async encode(tok, brief, pool = null, split = 1) {
    const { d, heads } = this, prof = PROF.on;
    let t = prof ? now() : 0;
    const tick = name => { if (prof) { const t2 = now(); profAdd(name, t2 - t); t = t2; } };
    const { X, N, M, bins, cover } = this.embed(tok, brief);
    tick("enc.embed+canvas");
    const H = new Float32Array(M * d), Qm = new Float32Array(M * d), Km = new Float32Array(M * d), Vm = new Float32Array(M * d);
    const parts = pool ? pool.size + 1 : split;
    const rr = ranges(M, parts);
    const mine = pool ? [rr[0]] : [...rr].reverse();      // with a pool, this thread does the first range, the helpers the rest
    const helperRanges = pool ? rr.slice(1) : [];
    for (let l = 0; l < this.enc.length; l++) {
      const L = this.enc[l];
      let jobs = helperRanges.map(([lo, hi], h) => pool.run(h, { type: "A", layer: l, lo, hi, X: X.slice(lo * d, hi * d) }));
      tick("enc.messages");
      for (const [lo, hi] of mine) stageA(L, X, H, Qm, Km, Vm, lo, hi, d);
      tick("enc.norm1+qkv");
      for (const r of await Promise.all(jobs)) { Qm.set(r.Q, r.lo * d); Km.set(r.K, r.lo * d); Vm.set(r.V, r.lo * d); if (prof) profAdd("helpers.compute", r.ms); }
      tick("enc.waiting for helpers");
      jobs = helperRanges.map(([lo, hi], h) => pool.run(h, { type: "B", layer: l, lo, hi, M, X: X.slice(lo * d, hi * d),
        Q: Qm.slice(lo * d, hi * d), K: Km, V: Vm }));
      tick("enc.messages");
      for (const [lo, hi] of mine) stageB(L, X, H, Qm, Km, Vm, M, lo, hi, d, heads, this.scratch);
      t = prof ? now() : 0;
      for (const r of await Promise.all(jobs)) { X.set(r.X, r.lo * d); if (prof) profAdd("helpers.compute", r.ms); }
      tick("enc.waiting for helpers");
    }
    const mem = new Float32Array(M * d);
    for (let i = 0; i < M; i++) layerNorm(X, i * d, this.W["enc_norm.weight"], this.W["enc_norm.bias"], mem, i * d, d);
    tick("enc.final norm");
    const crossK = this.dec.map(() => new Float32Array(M * d)), crossV = this.dec.map(() => new Float32Array(M * d));
    const encKey = [new Float32Array(N * d), new Float32Array(N * d)];
    const jobs = helperRanges.map(([lo, hi], h) => pool.run(h, { type: "KV", lo, hi, boxRows: N, mem: mem.slice(lo * d, hi * d) }));
    tick("enc.messages");
    for (const [lo, hi] of mine) stageKV(this.kv, mem, lo, hi, N, crossK, crossV, encKey, d);
    tick("pass.cross keys/values + anchor keys");
    for (const r of await Promise.all(jobs)) {
      r.crossK.forEach((a, l) => crossK[l].set(a, r.lo * d));
      r.crossV.forEach((a, l) => crossV[l].set(a, r.lo * d));
      r.encKey.forEach((a, e) => { if (a.length) encKey[e].set(a, r.lo * d); });
      if (prof) profAdd("helpers.compute", r.ms);
    }
    tick("enc.waiting for helpers");
    return encoded({ mem, N, M, bins, cover, crossK, crossV, encKey, itemOk: this.itemMask(tok) });
  }

  // one decoder step with the key/value cache: the embedding of the next position in, its final state out
  makeStep(enc) {
    const { d, W, heads, hd } = this;
    const { M, crossK, crossV } = enc;
    let T = 0;
    const y = new Float32Array(d), h1 = new Float32Array(d), q = new Float32Array(d);
    const att = new Float32Array(d), tmp = new Float32Array(d), h = new Float32Array(d), hid = new Float32Array(this.dh);
    const selfScores = new Float64Array(this.maxT), crossScores = new Float64Array(M);
    const prof = PROF.on;
    const step = (inp) => {
      y.set(inp);
      let ts = prof ? now() : 0;
      const seg = name => { if (prof) { const t2 = now(); profAdd(name, t2 - ts); ts = t2; } };
      for (let l = 0; l < this.dec.length; l++) {
        const L = this.dec[l];
        layerNorm(y, 0, L.n1w, L.n1b, h1, 0, d);
        linear(h1, 0, L.qW, L.qB, q, 0, d, d);
        linear(h1, 0, L.kW, L.kB, this.cacheK[l], T * d, d, d);
        linear(h1, 0, L.vW, L.vB, this.cacheV[l], T * d, d, d);
        attendRow(q, 0, this.cacheK[l], this.cacheV[l], T + 1, heads, hd, att, 0, selfScores);
        linear(att, 0, L.outW, L.outB, tmp, 0, d, d);
        for (let k = 0; k < d; k++) y[k] += tmp[k];
        seg("dec.self-attention (norm, qkv, attend, out)");
        layerNorm(y, 0, L.n2w, L.n2b, h1, 0, d);
        linear(h1, 0, L.cqW, L.cqB, q, 0, d, d);
        attendRow(q, 0, crossK[l], crossV[l], M, heads, hd, att, 0, crossScores);
        linear(att, 0, L.coW, L.coB, tmp, 0, d, d);
        for (let k = 0; k < d; k++) y[k] += tmp[k];
        seg("dec.cross-attention (norm, q, attend, out)");
        layerNorm(y, 0, L.n3w, L.n3b, h1, 0, d);
        feedForward(h1, 0, L, hid, y, 0, d, this.dh);
        seg("dec.feed-forward (norm, 1024 relu, 256)");
      }
      T += 1;
      layerNorm(y, 0, W["dec_norm.weight"], W["dec_norm.bias"], h, 0, d);
      seg("dec.final norm");
      if (prof) { profAdd("steps", 0, 1); profAdd("steps.self positions", 0, T); profAdd("steps.cross positions", 0, M); }
      return h;
    };
    return step;
  }

  // one pass: greedy picks until STOP or `chunk` parts; yields each part with what decided its picks
  *writePass(enc, detail = true) {
    const { d, W, K, heads, hd } = this;
    const { N, M, bins, crossK, crossV, encKey } = enc;
    const written = { x: [], y: [] };                  // same-axis picks of this pass: {v, key, j, f}
    let nKeys = 0;
    const prof = PROF.on;
    const step = this.makeStep(enc);
    const fe = W["field_emb.weight"];
    // MVPEditor._value_emb: a pick's own vector, from the table its field is scored against - the code table for the
    // first field, the field's ruler for the other four - plus that field's mark
    const codes = this.segment ? this.codeTable() : W["item_out.weight"];
    const valueEmb = (f, v, out) => {
      const table = f === 0 ? codes : this.rulers[f - 1];
      for (let k = 0; k < d; k++) out[k] = table[v * d + k] + fe[f * d + k];
      return out;
    };
    const itemLogits = new Float64Array(this.nCodes + 1), u = this.buf("u", d), qa = this.buf("qa", d);
    const rectLogits = this.segment ? null : this.buf("itemLogits", this.nCodes + 1), stopL = this.buf("stopL", 1);
    const offL = this.buf("offL", 2 * K + 1), off = new Float64Array(2 * K + 1), gateL = this.buf("gateL", 1);
    const ruler = new Float64Array(NX), copy = new Float64Array(NX), hist = new Float64Array(NX);
    const cap = 2 * N + 2 * this.chunk + 2, sc = new Float64Array(cap), vals = new Int32Array(cap);
    const inp = this.buf("inp", d);
    inp.set(W["bos"]);
    for (let j = 0; j < this.chunk; j++) {
      const elem = [0, 0, 0, 0, 0], picks = [];
      for (let f = 0; f < 5; f++) {
        const hs = step(inp);
        let th = prof ? now() : 0;
        const hseg = name => { if (prof) { const t2 = now(); profAdd(name, t2 - th); th = t2; } };
        if (f === 0) {
          if (this.segment) {
            // the head is tied to the code table, as the ruler heads are to their ticks; STOP is the one class with
            // no section behind it and has a head of its own (MVPEditor._head)
            this.lin(hs, 0, W["item_proj.weight"], W["item_proj.bias"], u, 0, d, d);
            this.rulerLogits(u, codes, this.nCodes, itemLogits);
            this.lin(hs, 0, W["stop_head.weight"], W["stop_head.bias"], stopL, 0, d, 1);
            itemLogits[this.nCodes] = stopL[0];
          } else {
            this.lin(hs, 0, W["item_head.weight"], W["item_head.bias"], rectLogits, 0, d, this.nCodes + 1);
            for (let i = 0; i <= this.nCodes; i++) itemLogits[i] = rectLogits[i];
          }
          // what the design has not got is out of reach, for the pick and for the probabilities alike (_masked)
          if (enc.itemOk) for (let i = 0; i <= this.nCodes; i++) if (!enc.itemOk[i]) itemLogits[i] = -Infinity;
          let v = 0;
          for (let i = 1; i <= this.nCodes; i++) if (itemLogits[i] > itemLogits[v]) v = i;
          hseg("head.item");
          if (v === this.STOP) return;
          elem[0] = v;
          if (detail) {
            const p = Float64Array.from(itemLogits);
            softmaxInPlace(p, 0, p.length);
            picks.push({ f, v, probs: Array.from(p, x => +x.toFixed(4)) });
          }
          valueEmb(0, v, inp);
          hseg("head.item detail + next input");
          continue;
        }
        // the ruler: pick_proj then the tied tick table
        const n = this.rulerN[f - 1], tab = this.rulers[f - 1];
        this.lin(hs, 0, W[`pick_proj.${f - 1}.weight`], W[`pick_proj.${f - 1}.bias`], u, 0, d, d);
        this.rulerLogits(u, tab, n, ruler);
        softmaxInPlace(ruler, 0, n);
        hseg("head.ruler (proj, tick logits, softmax)");
        if (prof) profAdd("head.ruler ticks", 0, n);
        // the anchor: attention over the same-axis edges of the box tokens and the picks written so far, an offset, a gate
        const axis = f === 1 || f === 3 ? "x" : "y", lowF = axis === "x" ? 0 : 1;
        this.lin(hs, 0, W["anchor.q.weight"], W["anchor.q.bias"], qa, 0, d, d);
        const inv = 1 / Math.sqrt(d);
        const wr = written[axis];
        const S = 2 * N + wr.length;
        let m = -Infinity;
        for (let e = 0; e < 2; e++) {
          this.keyDots(qa, encKey[e], N, sc, e * N);
          for (let i = 0; i < N; i++) {
            const s = e * N + i;
            sc[s] = sc[s] * inv;
            vals[s] = bins[i][lowF + 2 * e];
            if (sc[s] > m) m = sc[s];
          }
        }
        for (let i = 0; i < wr.length; i++) {
          const key = wr[i].key;
          let dot = 0;
          for (let k = 0; k < d; k++) dot += qa[k] * key[k];
          sc[2 * N + i] = dot * inv; vals[2 * N + i] = wr[i].v;
          if (sc[2 * N + i] > m) m = sc[2 * N + i];
        }
        let z = 0;
        for (let s = 0; s < S; s++) { sc[s] = Math.exp(sc[s] - m); z += sc[s]; }
        for (let s = 0; s < S; s++) sc[s] /= z;
        hseg("head.anchor sources (q, scores, softmax)");
        if (prof) profAdd("head.anchor source count", 0, S);
        this.lin(hs, 0, W["anchor.offset.weight"], W["anchor.offset.bias"], offL, 0, d, 2 * K + 1);
        off.set(offL);
        softmaxInPlace(off, 0, off.length);
        this.lin(hs, 0, W["anchor.gate.weight"], W["anchor.gate.bias"], gateL, 0, d, 1);
        const g = 1 / (1 + Math.exp(-gateL[0]));
        hseg("head.anchor offset + gate");
        let lo = n, hi = -1;
        for (let s = 0; s < S; s++) { hist[vals[s]] += sc[s]; if (vals[s] < lo) lo = vals[s]; if (vals[s] > hi) hi = vals[s]; }
        const a0 = Math.max(0, lo - K), a1 = Math.min(n - 1, hi + K);
        for (let t = a0; t <= a1; t++) copy[t] = 0;
        for (let src = lo; src <= hi; src++) {
          const mass = hist[src];
          if (!mass) continue;
          const t0 = Math.max(0, src - K), t1 = Math.min(n - 1, src + K);
          for (let t = t0; t <= t1; t++) copy[t] += mass * off[t - src + K];
        }
        let v = 0, best = -Infinity;
        const g1 = 1 - g;
        for (let t = 0; t < a0; t++) { const p = g1 * ruler[t]; if (p > best) { best = p; v = t; } }
        for (let t = a0; t <= a1; t++) { const p = g1 * ruler[t] + g * copy[t]; if (p > best) { best = p; v = t; } }
        for (let t = a1 + 1; t < n; t++) { const p = g1 * ruler[t]; if (p > best) { best = p; v = t; } }
        hseg("head.mixture (histogram, offsets, argmax)");
        // the coordinate this class implies is copied, not picked (MVPEditor._implied): a horizontal member's y1 is
        // its y0, a vertical member's x1 is its x0. Three picks for an axis-aligned member, four for a rotated one.
        const imp = this.segment ? IMPLIED[classOf(elem[0])] : null;
        const implied = imp && imp[0] === f;
        if (implied) v = elem[imp[1]];
        if (detail) picks.push(implied ? { f, v, implied: true }
          : this.pickDetail(f, v, g, ruler, copy, a0, a1, sc, vals, S, N, off, wr));
        hseg("head.pick detail for the page");
        for (let s = 0; s < S; s++) hist[vals[s]] = 0;
        elem[f] = v;
        valueEmb(f, v, inp);
        const key = this.keyBuf(nKeys++);
        this.lin(inp, 0, W["anchor.k_dec.weight"], null, key, 0, d, d);
        wr.push({ v, key, j, f });
        hseg("head.next input + source key");
      }
      yield { j, elem, picks };
    }
  }

  // what decided an edge pick: the gate, the ruler's and the copy's share of the chosen tick, the source that explains the tick
  // best (largest attention x offset term), the offset from it, and the pick's distribution around the tick
  pickDetail(f, v, g, ruler, copy, a0, a1, sc, vals, S, N, off, wr) {
    const K = this.K, n = this.rulerN[f - 1];
    let sBest = -1, cBest = -1;
    for (let s = 0; s < S; s++) {
      const o = v - vals[s];
      if (o < -K || o > K) continue;
      const c = sc[s] * off[o + K];
      if (c > cBest) { cBest = c; sBest = s; }
    }
    const pAt = t => (1 - g) * ruler[t] + (t >= a0 && t <= a1 ? g * copy[t] : 0);
    const pr = (1 - g) * ruler[v], pc = v >= a0 && v <= a1 ? g * copy[v] : 0;
    let src = null;
    if (sBest >= 0) {
      src = sBest < 2 * N ? { kind: "box", token: sBest % N, side: sBest < N ? 0 : 1, tick: vals[sBest], offset: v - vals[sBest], weight: +sc[sBest].toFixed(4) }
        : { kind: "written", part: wr[sBest - 2 * N].j, field: wr[sBest - 2 * N].f, tick: vals[sBest], offset: v - vals[sBest], weight: +sc[sBest].toFixed(4) };
    }
    const total = pr + pc || 1, w0 = Math.max(0, v - K), w1 = Math.min(n - 1, v + K), win = [];
    let z = 0;
    for (let t = 0; t < n; t++) z += pAt(t);
    for (let t = w0; t <= w1; t++) win.push(+(pAt(t) / z).toFixed(4));
    return { f, v, g: +g.toFixed(4), p: +(total / z).toFixed(4), copyShare: +(pc / total).toFixed(4), fromRuler: pr >= pc, src, win: [w0, win] };
  }
}
