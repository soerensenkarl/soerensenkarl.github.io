// Automake, the drawing tool. Drag the wall and its openings on a white canvas; a neural network fills the wall in, part by
// part, in this browser. Everything the page does is here: the design and its rules, the drawing, the dragging and the model
// thread. The wall and its openings are drawn the way a CAD window draws a selection - a dotted rectangle with a cross on
// each corner - and the parts the network writes are the only solid things on the page.
//
// The engine is untouched: wall.js (scenes, encoding, the world's refusal), worker.js, writer.js, model.js, wasm.js.
import { BRIEFS, elementRect, ITEMS, makeScene, refuseOverlaps, skinOf, skinRects } from "./wall.js";

const qs = new URLSearchParams(location.search);
const FORMAT = qs.get("format") || "f16";

// the networks on offer: model/<file>.<format>.json/.bin, written by scripts/export_web_model.py
const MODELS = [
  { id: "o4", file: "o4", name: "O4", title: "Frames, then blocks", scripts: ["frame", "block"],
    blurb: "The framing network, then 50 minutes on concrete-block walls. It kept its framing by rehearsing framed walls it had written itself and the world had accepted: no framing script or framing data in that stage." },
  // `loads: true` - the network reads the loads on the wall's top edge (sequence.py TYPES index 4), point and line
  // alike. Only these networks are given load tokens, and only for them does the page draw, drag or link any load.
  // To put a newer checkpoint behind this artifact, export it (scripts/export_web_model.py --name x9) and change `file`.
  { id: "x8", file: "x8", name: "X8", title: "Studs under loads", scripts: ["frame"], loads: true },
  { id: "n0", file: "n0", name: "N0", label: "N0 · frames only", scripts: ["frame"] },
  { id: "m0", file: "m0", name: "M0", label: "M0 · both together", scripts: ["frame", "block"] },
];
const DEFAULT_MODEL = "o4";

// what the rulers and the dataset allow (metres)
const LMIN = 1.2, LMAX = 7.9, HMIN = 2.0, HMAX = 3.15;
const SIDE = 0.2, GAP = 0.3, MINW = 0.4, MINH = 0.4, HEAD = 0.35, MINSILL = 0.3, MAXOPS = 4;
// loads on the top edge (automake/mvp/dataset.py: LOAD_CLEAR_END, LOAD_SPACING, load_grid). A point load (10 kN)
// stands at one x, over an opening if you like, and `+ load` lays them out as the same regular 600 mm line the data
// uses, from GRID0; a line load (10 kN/m) spans LINEMIN to LINEMAX of the edge and keeps LLCLEAR clear of every
// point load.
const LEND = 0.2, LAPART = 0.4, LGRID = 0.6, GRID0 = 0.3, MAXLOADS = 16;
const LLCLEAR = 0.2, LINEMIN = 0.6, LINEMAX = 2.4, LINEDEF = 1.2, MAXLINES = 4;

// what each artifact opens with, and what it says about itself
const WALLS = {
  n0: () => ({ script: "frame", L: 5.18, H: 2.63, openings: [door(0.535, 0.935, 2.08), win(2.695, 1.29, 1.0, 0.85)] }),
  o4: () => ({ script: "block", L: 4.97, H: 2.63, openings: [door(1.12, 0.945, 2.0), win(2.625, 1.47, 1.115, 0.83)] }),
  x8: () => ({ script: "frame", L: 5.4, H: 2.7, openings: [door(0.6, 0.9, 2.05), win(2.7, 1.2, 1.1, 0.9)],
                loads: [0.45, 1.05, 1.65, 2.25, 2.85, 3.45, 4.05, 4.65] }),
};
const ABOUT = {
  n0: "An 8.8-million-parameter encoder-decoder transformer that has learned light timber framing by imitating a simple framing script, judged only by geometry. It reads the wall, its openings and the parts already there as boxes and writes each part as an item and four edges on a 5 mm ruler, one part at a time, with no framing rules built in. It runs entirely in your browser on WebAssembly; nothing is sent anywhere. Trained on 40,000 walls 2.4-6 m long; on walls it has not seen it writes 88% of the script's parts with 91% of its parts right.",
  x8: "The framing network after nine rounds of learning from the world's physics alone: a search that only knows 'slide a box, copy one, cut one, take one away' improved walls under load by the strain energy the world measures, and the network learned to reproduce them. It now frames every opening on every side and stands a stud at each end of the wall, carries the load over a door or a window on a header, reads a line load (a blue bar) as well as a point load, and leaves nothing hanging – and was never told what a stud, a header or a jamb is. Runs entirely in your browser; nothing is sent anywhere.",
  o4: "The same network after it had learned timber framing, then trained for 50 minutes on concrete-block walls. It kept its framing by rehearsing framed walls it had written itself and the world had accepted, with no framing script or framing data in that stage. Its framed walls are as good as before (88% of the script's parts, 91% right); its block walls get about 7 in 10 blocks right. It runs entirely in your browser on WebAssembly; nothing is sent anywhere.",
};

const $ = id => document.getElementById(id);
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const q = v => Math.round(v / 0.005) * 0.005;            // the 5 mm ruler the network writes on
const r3 = v => Number(v.toFixed(3));

// ---------------------------------------------------------------- the design and its rules
const door = (x, w, h) => ({ kind: "door", x, w, h, sill: 0 });
const win = (x, w, h, sill) => ({ kind: "window", x, w, h, sill });
let modelId = DEFAULT_MODEL;
let design = WALLS[DEFAULT_MODEL]();

const sillOf = o => o.kind === "door" ? 0 : o.sill;
const minLength = ops => ops.length ? 2 * SIDE + ops.reduce((s, o) => s + o.w, 0) + GAP * (ops.length - 1) : LMIN;
const minHeight = ops => ops.reduce((m, o) => Math.max(m, sillOf(o) + o.h + HEAD), HMIN);

// everything inside the rulers and the dataset's rules: 0.2 m from the ends, 0.3 m between openings, doors on the floor
function fit(d) {
  d.openings.sort((a, b) => a.x - b.x);
  d.openings = d.openings.slice(0, MAXOPS);
  d.H = clamp(q(d.H), HMIN, HMAX);
  for (const o of d.openings) {
    o.w = clamp(q(o.w), MINW, LMAX - 2 * SIDE);
    if (o.kind === "door") { o.sill = 0; o.h = clamp(q(o.h), 1.2, d.H - HEAD); }
    else { o.h = clamp(q(o.h), MINH, d.H - MINSILL - HEAD); o.sill = clamp(q(o.sill), MINSILL, d.H - HEAD - o.h); }
  }
  d.H = clamp(Math.max(d.H, minHeight(d.openings)), HMIN, HMAX);
  d.L = clamp(Math.max(q(d.L), minLength(d.openings)), LMIN, LMAX);
  let cur = SIDE;                                        // left to right, each clear of the one before
  for (const o of d.openings) { o.x = q(Math.max(q(o.x), cur)); cur = o.x + o.w + GAP; }
  let lim = d.L - SIDE;                                  // then back from the far end
  for (let i = d.openings.length - 1; i >= 0; i--) { const o = d.openings[i]; if (o.x + o.w > lim + 1e-9) o.x = q(lim - o.w); lim = o.x - GAP; }
  for (const o of d.openings) { o.x = r3(clamp(o.x, SIDE, Math.max(SIDE, d.L - SIDE - o.w))); o.w = r3(o.w); o.h = r3(o.h); o.sill = r3(o.sill); }
  d.L = r3(d.L); d.H = r3(d.H);
  return fitLines(fitLoads(d));
}

// ---------------------------------------------------------------- the loads on the top edge
// Only a network built with them reads loads; for the others the page has none at all.
const readsLoads = () => !!MODELS.find(m => m.id === modelId).loads;
const qUp = v => Math.ceil(v / 0.005 - 1e-9) * 0.005, qDn = v => Math.floor(v / 0.005 + 1e-9) * 0.005;

// the top edge LEND from each end, with `blocks` cut out of it
function edgeBands(d, blocks) {
  const out = [];
  let a = LEND;
  for (const [lo, hi] of [...blocks].sort((p, r) => p[0] - r[0])) {
    if (hi <= a) { a = Math.max(a, hi); continue; }
    if (lo > a) out.push([a, lo]);
    a = hi;
  }
  out.push([a, d.L - LEND]);
  return out.map(([lo, hi]) => [qUp(Math.max(lo, LEND)), qDn(Math.min(hi, d.L - LEND))]).filter(b => b[1] >= b[0] - 1e-9);
}
// where a point load may stand: clear of the line loads (an opening below it no longer matters)
const loadBands = d => edgeBands(d, (d.lines || []).map(s => [s[0] - LLCLEAR, s[1] + LLCLEAR]));
// where a line load may lie: clear of the point loads and of the line loads `others`
const lineBands = (d, others) => edgeBands(d, [...(d.loads || []).map(v => [v - LLCLEAR, v + LLCLEAR]), ...others]);

const snapTo = (x, bands) => {                           // the nearest allowed x, null when there is nowhere to go
  let best = null, bd = Infinity;
  for (const [a, b] of bands) { const v = clamp(x, a, b), e = Math.abs(v - x); if (e < bd) { bd = e; best = q(v); } }
  return best;
};

// loads on the 5 mm ruler, in order, inside the bands, LAPART apart; one with nowhere left to stand is dropped
function fitLoads(d) {
  if (!readsLoads()) { d.loads = []; return d; }
  const bands = loadBands(d);
  let xs = (d.loads || []).map(q).sort((a, b) => a - b).slice(0, MAXLOADS);
  let cur = -Infinity;
  xs = xs.map(x => { const v = snapTo(Math.max(x, cur + LAPART), bands); if (v !== null) cur = v; return v; });
  let lim = Infinity;
  for (let i = xs.length - 1; i >= 0; i--) {
    const v = xs[i] === null ? null : snapTo(Math.min(xs[i], lim - LAPART), bands);
    xs[i] = (v === null || v > lim - LAPART + 1e-9) ? null : v;
    if (xs[i] !== null) lim = xs[i];
  }
  d.loads = xs.filter(v => v !== null).map(r3);
  return d;
}

// where the next load goes: the next place on the 600 mm line, GRID0 from the left end and then LGRID to the right of
// the rightmost load there (dataset.load_grid). Null when the line has run off the wall or a line load holds every
// place left - the button is then gone, never greyed.
function loadSpot(d) {
  if (d.loads.length >= MAXLOADS) return null;
  const bands = loadBands(d);
  for (let x = q(d.loads.length ? d.loads[d.loads.length - 1] + LGRID : GRID0); x <= d.L - LEND + 1e-9; x = q(x + LGRID))
    if (x >= LEND - 1e-9 && bands.some(([a, b]) => x >= a - 1e-9 && x <= b + 1e-9)) return r3(x);
  return null;
}

function addLoad() {
  const x = loadSpot(design);
  if (x === null) return;
  design.loads.push(x);
  fit(design);
  selL = design.loads.indexOf(r3(x));
  sel = selN = -1;
  changed();
}

// line loads on the 5 mm ruler, each in a stretch clear of the point loads and of the lines before it, LINEMIN to
// LINEMAX long; one with no room left is dropped
function fitLines(d) {
  if (!readsLoads()) { d.lines = []; return d; }
  const out = [];
  for (const s of (d.lines || []).slice(0, MAXLINES).sort((a, b) => a[0] - b[0])) {
    const len = clamp(q(s[1] - s[0]), LINEMIN, LINEMAX);
    let best = null, bd = Infinity;
    for (const [a, b] of lineBands(d, out)) {
      if (b - a < LINEMIN - 1e-9) continue;
      const l = Math.min(len, qDn(b - a)), x = clamp(q(s[0]), a, qDn(b - l)), e = Math.abs(x - s[0]);
      if (e < bd) { bd = e; best = [r3(x), r3(x + l)]; }
    }
    if (best) out.push(best);
  }
  d.lines = out;
  return d;
}

function lineSpot(d) {                                   // where a new line load would go: the widest stretch left
  if ((d.lines || []).length >= MAXLINES) return null;
  let best = null;
  for (const b of lineBands(d, d.lines || [])) if (b[1] - b[0] >= LINEMIN - 1e-9 && (!best || b[1] - b[0] > best[1] - best[0])) best = b;
  if (!best) return null;
  const l = Math.min(LINEDEF, qDn(best[1] - best[0])), x = q(best[0] + (best[1] - best[0] - l) / 2);
  return [r3(x), r3(x + l)];
}

function addLine() {
  const s = lineSpot(design);
  if (!s) return;
  design.lines.push(s);
  fit(design);
  selN = design.lines.findIndex(t => t[0] === s[0]);
  sel = selL = -1;
  changed();
}

// the loads as the network reads them: one segment [x0, x1] each in the wall frame, a point load being x0 = x1
// (dataset.loads_in_wall_frame); point loads first, then the line loads, as sequence.load_rects stacks them
const loadSegs = d => (readsLoads() ? [...d.loads.map(v => [v, v]), ...(d.lines || [])] : [])
  .map(s => [s[0] - d.L / 2, s[1] - d.L / 2]);

function freeSpan(d) {                                   // the widest stretch of bare wall, for a new opening
  const edges = [[SIDE - GAP, SIDE], ...d.openings.map(o => [o.x, o.x + o.w]), [d.L - SIDE, d.L - SIDE + GAP]];
  let best = null;
  for (let i = 0; i + 1 < edges.length; i++) {
    const lo = edges[i][1] + GAP, hi = edges[i + 1][0] - GAP;
    if (hi - lo > (best ? best[1] - best[0] : MINW - 1e-9)) best = [lo, hi];
  }
  return best;
}

function addOpening(kind) {
  const span = design.openings.length < MAXOPS && freeSpan(design);
  if (!span) return;
  const room = span[1] - span[0], w = Math.min(kind === "door" ? 0.9 : 1.2, room), x = q(span[0] + (room - w) / 2);
  const o = kind === "door" ? door(x, w, Math.min(2.1, design.H - HEAD)) : win(x, w, 1.2, 0.9);
  design.openings.push(o);
  fit(design);
  sel = design.openings.indexOf(o);
  changed();
}

// a wall drawn the way the training data is drawn (automake/mvp/dataset.py: train_spec, extended_spec, _draw, _place):
// the training domain, and one wall in four from the extended one (3 or 4 openings, longer, taller, or a wide opening)
function randomDesign() {
  const U = (a, b) => a + Math.random() * (b - a), step = (v, k) => Math.round(v / k) * k;
  const nTrain = () => { const r = Math.random(); return r < 0.2 ? 0 : r < 0.65 ? 1 : 2; };
  for (;;) {
    const extended = Math.random() < 0.25;
    const tags = extended ? ["openings", "long", "tall", "wide"].filter(() => Math.random() < 0.5) : [];
    if (extended && !tags.length) continue;                             // an extended wall is extended in some way
    let n = tags.includes("openings") ? 3 + Math.round(Math.random()) : nTrain();
    if (tags.includes("wide")) n = Math.max(n, 1);
    const L = step(U(...(tags.includes("long") ? [6.2, 7.9] : [2.4, 6.0])), 0.01);
    const H = step(U(...(tags.includes("tall") ? [2.85, 3.1] : [2.2, 2.8])), 0.01);
    const wideI = tags.includes("wide") && n ? Math.floor(Math.random() * n) : -1;
    const ops = [];
    for (let i = 0; i < n; i++) {
      const [wLo, wHi] = i === wideI ? [2.0, 3.0] : [0.4, 1.8];
      if (Math.random() < 0.4) ops.push(door(0, step(U(Math.max(0.6, wLo), wHi), 0.01), step(U(1.8, Math.max(1.8, Math.min(2.4, H - HEAD))), 0.01)));
      else {
        const w = step(U(wLo, wHi), 0.01), sill = step(U(MINSILL, Math.min(1.5, H - HEAD - MINH)), 0.01);
        ops.push(win(0, w, step(U(MINH, Math.max(MINH, Math.min(1.8, H - HEAD - sill))), 0.01), sill));
      }
    }
    const spare = L - 2 * SIDE - ops.reduce((s, o) => s + o.w, 0) - GAP * Math.max(0, n - 1);
    if (spare < 0) continue;                                            // the openings do not fit: draw again
    const slack = ops.map(() => U(0, spare)).sort((a, b) => a - b);
    let a = SIDE, used = 0;
    ops.forEach((o, i) => { a += slack[i] - used; used = slack[i]; o.x = q(a); a += o.w + GAP; });
    const d = fit({ script: design.script, L, H, openings: ops, loads: [], lines: [] });
    if (readsLoads()) {                  // the 600 mm line from a random offset (dataset.load_grid), now and then
      if (Math.random() < 0.3) {         // under a line load, as the "mixed" sampler draws one (dataset.line_random)
        const w = q(U(LINEMIN, LINEMAX));
        const room = lineBands(d, []).filter(b => b[1] - b[0] >= w - 1e-9);
        if (room.length) {
          const b = room[Math.floor(Math.random() * room.length)], x = q(U(b[0], b[1] - w));
          d.lines = [[r3(x), r3(x + w)]];
        }
      }
      const bands = loadBands(d);
      for (let x = q(U(0.1, 0.5)); x <= d.L - LEND + 1e-9 && d.loads.length < MAXLOADS; x = q(x + LGRID))
        if (x >= LEND - 1e-9 && bands.some(([a, b]) => x >= a - 1e-9 && x <= b + 1e-9)) d.loads.push(r3(x));
      fit(d);
    }
    return d;
  }
}

// ---------------------------------------------------------------- the link, so a wall can be shared
function writeHash() {
  const o = design.openings.map(o => o.kind === "door" ? `d${r3(o.x)}_${r3(o.w)}_${r3(o.h)}` : `w${r3(o.x)}_${r3(o.w)}_${r3(o.h)}_${r3(o.sill)}`).join("~");
  const p = new URLSearchParams({ n: modelId, t: design.script, L: design.L, H: design.H, o });
  if (readsLoads()) {                                    // metres from the wall's left end: point loads `l`, line loads `q`
    p.set("l", design.loads.map(r3).join("_"));
    if (design.lines.length) p.set("q", design.lines.map(s => `${r3(s[0])}-${r3(s[1])}`).join("~"));
  }
  history.replaceState(null, "", "#" + p);
}
if (location.hash.length > 2) try {
  const p = new URLSearchParams(location.hash.slice(1));
  if (MODELS.some(m => m.id === p.get("n"))) modelId = p.get("n");
  if (WALLS[modelId]) design = WALLS[modelId]();         // each artifact opens with its own wall
  const ops = (p.get("o") || "").split("~").filter(Boolean).map(s => {
    const val = s.slice(1).split("_").map(Number);
    return s[0] === "d" ? door(val[0], val[1], val[2]) : win(val[0], val[1], val[2], val[3]);
  });
  const len = p.get("L");
  if (len !== null || p.has("o")) design = { script: p.get("t") === "block" ? "block" : "frame", L: +len || 5.8, H: +p.get("H") || 2.7, openings: ops, loads: [], lines: [] };   // a link with only a network keeps that artifact's wall
  else if (p.has("t")) design.script = p.get("t") === "block" ? "block" : "frame";
  if (p.has("l")) design.loads = (p.get("l") || "").split("_").filter(Boolean).map(Number).filter(v => isFinite(v));
  if (p.has("q")) design.lines = p.get("q").split("~").filter(Boolean).map(v => v.split("-").map(Number)).filter(s => s.length === 2 && s.every(isFinite));
} catch { /* keep the default wall */ }
if (!MODELS.find(m => m.id === modelId).scripts.includes(design.script)) design.script = "frame";
fit(design);

// ---------------------------------------------------------------- state
let sel = -1, selL = -1, selN = -1, hover = null, drag = null;
let scene = null, obs = [], parts = [], pending = [], hot = null;
let runId = 1, timer = null, ready = false, wantRun = true;
window.__automake = { loads: [], runs: [] };

// ---------------------------------------------------------------- the model thread
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });

function loadModel() {
  ready = false;
  const m = MODELS.find(m => m.id === modelId);
  window.__automake.loadStart = performance.now();
  worker.postMessage({ type: "load", url: new URL("../model/", import.meta.url).href, file: m.file, format: FORMAT,
    backend: qs.get("backend"), threads: qs.get("threads") });
  status("loading neural net…");
}

worker.onmessage = e => {
  const ev = e.data, m = MODELS.find(m => m.file === ev.file);
  if (ev.type === "progress") {
    if (m && m.id === modelId) status(`loading neural net · ${(ev.loaded / 1e6).toFixed(1)} of ${(ev.total / 1e6).toFixed(1)} MB`);
    if (m && m.id === modelId) { $("load").hidden = false; $("load").firstElementChild.style.width = `${Math.min(100, 100 * ev.loaded / Math.max(1, ev.total))}%`; }
    return;
  }
  if (ev.type === "ready") {
    if (!m || m.id !== modelId) return;
    ready = true;
    window.__automake.loads.push({ model: m.id, ms: performance.now() - window.__automake.loadStart, fetchMs: ev.fetchMs,
      parseMs: ev.parseMs, poolMs: ev.poolMs, helpers: ev.helpers, bytes: ev.bytes, backend: ev.backend });
    $("load").hidden = true;
    if (wantRun) startRun(); else status("ready");
    return;
  }
  if (ev.id !== runId) return;
  if (ev.type === "error") status("something went wrong: " + String(ev.message).split("\n")[0]);
  else if (ev.type === "part") {
    if (!refuseOverlaps(scene.wall, obs, [ev.elem]).kept.length) return;   // the world refuses a part that overlaps or leaves the wall
    obs.push(elementRect(ev.elem));
    pending.push(ev.elem);
    tick();
  } else if (ev.type === "done") {
    window.__automake.runs.push({ model: modelId, script: design.script, parts: ev.parts, passes: ev.passes, term: ev.term,
      ms: ev.ms, firstPartMs: ev.firstPartMs, backend: ev.backend });
    status(`${obs.length - scene.ops.length} parts in ${(ev.ms / 1000).toFixed(1)} s`);
  }
};

function startRun() {
  if (!ready) { wantRun = true; return; }
  wantRun = false;
  const sc = makeScene({ L: design.L, H: design.H, openings: design.openings.map(o => [o.x, o.w, sillOf(o), o.h]) });
  scene = skinRects(skinOf(sc));
  obs = scene.ops.map(r => r.slice());
  parts = []; pending = []; hot = null;
  worker.postMessage({ type: "run", id: runId, wall: scene.wall, ops: scene.ops, start: [], brief: BRIEFS[design.script],
    reject: true, loads: loadSegs(design) });
  status("writing…");
  tick();
}

// a change cancels the run in flight, empties the wall and starts again once the hand rests
function changed() {
  runId++;
  worker.postMessage({ type: "cancel" });
  parts = []; pending = []; hot = null;
  if (!drag) writeHash();                                // during a drag the link is written once, when the hand lets go
  sync();
  tick();
  clearTimeout(timer);
  timer = setTimeout(startRun, 150);
}

const status = t => { $("status").textContent = t; };

// ---------------------------------------------------------------- the view: metres to pixels
// fixed to the whole ruler, so dragging an edge moves it exactly under the pointer and the drawing never rescales
const WX0 = -0.28, WX1 = 8.28, WY0 = -0.3, WY1 = 3.46;
const canvas = $("canvas"), ctx = canvas.getContext("2d");
let v = null, raf = 0, last = 0;

function measure() {
  const cw = Math.max(240, canvas.clientWidth), ch = Math.max(1, canvas.clientHeight), dpr = Math.min(3, devicePixelRatio || 1);
  const w = Math.round(cw * dpr), h = Math.round(ch * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const s = Math.min(cw / (WX1 - WX0), ch / (WY1 - WY0));
  v = { s, dpr, cw, ch, ox: (cw - s * (WX1 - WX0)) / 2 - s * WX0, oy: (ch - s * (WY1 - WY0)) / 2 + s * WY1 };
}
const px = x => v.ox + v.s * x, py = y => v.oy - v.s * y;
const mx = p => (p - v.ox) / v.s, my = p => (v.oy - p) / v.s;
const obox = o => [o.x, sillOf(o), o.x + o.w, sillOf(o) + o.h];
// an element [item, x0, y0, x1, y1] in 5 mm ticks as a box in metres from the wall's bottom-left corner
const ebox = e => [e[1] * .005 - 4 + design.L / 2, e[2] * .005 - 1.6 + design.H / 2, e[3] * .005 - 4 + design.L / 2, e[4] * .005 - 1.6 + design.H / 2];
const BLUE = "#3b8cff";                                  // the outline on a part the moment the network writes it
// bright amber timber, grey blocks: the only fills on the page
const FILL = { block: ["#c9c9c9", "#6b6b6b"], lintel: ["#8d8d8d", "#3a3a3a"], timber: ["#e7c993", "#a67c48"] };   // pine, grey blocks
function grain(x, y, w, h, seed) {                       // pine: a few faint grain streaks along the member, fixed per part
  const along = w >= h, n = along ? h : w, len = along ? w : h;
  if (n < 3 || len < 8) return;
  let r = seed * 9301 + 49297;
  const rnd = () => (r = (r * 9301 + 49297) % 233280) / 233280;
  ctx.save(); ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
  ctx.lineWidth = 1;
  for (let k = 0, m = 2 + Math.floor(n / 4); k < m; k++) {
    const off = (k + .5 + (rnd() - .5) * .8) * n / m, amp = .6 + rnd() * 1.2, ph = rnd() * 6.3, dark = .12 + rnd() * .16;
    ctx.strokeStyle = `rgba(120,78,30,${dark})`; ctx.beginPath();
    for (let t = 0; t <= len; t += 6) {
      const wob = Math.sin(t / 40 + ph) * amp;
      const px_ = along ? x + t : x + off + wob, py_ = along ? y + off + wob : y + t;
      t ? ctx.lineTo(px_, py_) : ctx.moveTo(px_, py_);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function frame(b, dash, colour, w) {                     // a rectangle in metres, drawn as a line
  ctx.save(); ctx.setLineDash(dash); ctx.strokeStyle = colour; ctx.lineWidth = w;
  ctx.strokeRect(px(b[0]) + .5, py(b[3]) + .5, Math.max(0, v.s * (b[2] - b[0]) - 1), Math.max(0, v.s * (b[3] - b[1]) - 1));
  ctx.restore();
}
function cross(x, y, r, on) {                            // the CAD corner mark, and the handle you drag
  ctx.strokeStyle = on ? "#000" : "#111"; ctx.lineWidth = on ? 2 : 1;
  ctx.beginPath(); ctx.moveTo(px(x) - r, py(y)); ctx.lineTo(px(x) + r, py(y));
  ctx.moveTo(px(x), py(y) - r); ctx.lineTo(px(x), py(y) + r); ctx.stroke();
}

// a point load: a blue arrow standing on the wall's top edge at its x, pointing down. A line load: a bar above the edge
// over the stretch it covers, with a small arrow every LSTEP along it.
const LARROW = 0.30, LFOOT = 0.045, LSTEP = 0.2;         // metres above the top edge: the tail, the tip; the arrow spacing
function arrow(X, yTail, yTip, head) {
  ctx.beginPath(); ctx.moveTo(X, yTail); ctx.lineTo(X, yTip - 3); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(X, yTip + 5); ctx.lineTo(X - head, yTip - 5); ctx.lineTo(X + head, yTip - 5); ctx.closePath(); ctx.fill();
}
function del(X, Y, lit) {                                // the × that removes a load
  ctx.lineWidth = lit ? 2 : 1;
  ctx.strokeStyle = lit ? "#000" : "#9ab";
  ctx.beginPath(); ctx.moveTo(X - 4, Y - 4); ctx.lineTo(X + 4, Y + 4); ctx.moveTo(X + 4, Y - 4); ctx.lineTo(X - 4, Y + 4); ctx.stroke();
}
function drawLoad(x, H, i, on) {
  ctx.save();
  ctx.strokeStyle = ctx.fillStyle = on ? "#1668d8" : BLUE;
  ctx.lineWidth = on ? 2.4 : 1.6;
  arrow(px(x), py(H + LARROW), py(H + LFOOT), 4.5);
  if (on) del(px(x) + 14, py(H + LARROW) + 5, hover === `l:${i}:x`);
  ctx.restore();
}
function drawLine(s, H, i, on) {
  const a = px(s[0]), b = px(s[1]), yTail = py(H + LARROW), yTip = py(H + LFOOT);
  ctx.save();
  ctx.strokeStyle = ctx.fillStyle = on ? "#1668d8" : BLUE;
  ctx.lineWidth = on ? 2.4 : 1.6;
  ctx.beginPath(); ctx.moveTo(a, yTail); ctx.lineTo(b, yTail); ctx.stroke();
  ctx.lineWidth = 1.1;
  const n = Math.max(1, Math.round((s[1] - s[0]) / LSTEP));
  for (let k = 0; k <= n; k++) arrow(a + (b - a) * k / n, yTail, yTip, 3.5);
  if (on) {                                              // a handle at each end, to stretch it
    [a, b].forEach((X, k) => {
      const lit = hover === `n:${i}:${k}`;
      ctx.lineWidth = lit ? 2 : 1.4;
      ctx.fillStyle = lit ? "#1668d8" : "#fff";
      ctx.beginPath(); ctx.rect(X - 3.5, yTail - 3.5, 7, 7); ctx.fill(); ctx.stroke();
    });
    del(b + 15, yTail + 5, hover === `n:${i}:x`);
  }
  ctx.restore();
}

function paint() {
  measure();
  const { L, H, openings } = design;
  ctx.setTransform(v.dpr, 0, 0, v.dpr, 0, 0);
  ctx.clearRect(0, 0, v.cw, v.ch);

  for (const e of parts) {                               // what the network has written so far
    const [fill, stroke] = FILL[ITEMS[e[0]] === "block" ? "block" : ITEMS[e[0]] === "lintel" ? "lintel" : "timber"];
    const b = ebox(e), x = px(b[0]), y = py(b[3]), w = Math.max(.7, v.s * (b[2] - b[0])), h = Math.max(.7, v.s * (b[3] - b[1]));
    ctx.fillStyle = fill; ctx.fillRect(x, y, w, h);
    if (fill === FILL.timber[0]) grain(x, y, w, h, (e[1] * 7 + e[2] * 13 + e[3] * 3 + e[4]) % 1000);
    ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.strokeRect(x + .5, y + .5, Math.max(0, w - 1), Math.max(0, h - 1));
  }
  if (hot) { ctx.save(); ctx.globalAlpha = hot.a; ctx.shadowColor = "#6fb2ff"; ctx.shadowBlur = 14; frame(ebox(hot.e), [], BLUE, 2.5); frame(ebox(hot.e), [], BLUE, 2.5); ctx.restore(); }   // a lit, glowing outline   // the part just written

  const lit = (k, i) => hover === `${k}:${i}` || (hover || "").startsWith(`${k}:${i}:`);
  (design.loads || []).forEach((x, i) => drawLoad(x, H, i, selL === i || lit("l", i)));
  (design.lines || []).forEach((s, i) => drawLine(s, H, i, selN === i || lit("n", i)));

  frame([0, 0, L, H], [1, 3], "#555", 1);                // the wall, and the three crosses that size it
  cross(0, 0, 7, false);
  cross(L, 0, 7, hover === "w:r" || hover === "w:c");
  cross(0, H, 7, hover === "w:t" || hover === "w:c");
  cross(L, H, 7, hover === "w:c");
  openings.forEach((o, i) => {
    const b = obox(o), on = sel === i || (hover || "").startsWith(`o:${i}:`);
    frame(b, [1, 3], on ? "#333" : "#999", 1);
    if (!on) return;
    for (const [cx, cy, k] of [[b[0], b[1], "lb"], [b[2], b[1], "rb"], [b[0], b[3], "lt"], [b[2], b[3], "rt"]])
      cross(cx, cy, 5, hover === `o:${i}:${k}`);
    const dx = px(b[2]) + 15, dy = py(b[3]) - 15;        // the × that removes it
    ctx.strokeStyle = hover === `o:${i}:x` ? "#000" : "#999"; ctx.lineWidth = hover === `o:${i}:x` ? 2 : 1;
    ctx.beginPath(); ctx.moveTo(dx - 4, dy - 4); ctx.lineTo(dx + 4, dy + 4); ctx.moveTo(dx + 4, dy - 4); ctx.lineTo(dx - 4, dy + 4); ctx.stroke();
  });
}

// parts appear one by one, never all at once: one a frame, a little faster when many are waiting
function tick() {
  if (raf) return;
  raf = requestAnimationFrame(now => {
    raf = 0;
    const dt = last ? now - last : 16;
    last = now;
    for (let i = pending.length > 64 ? 4 : pending.length > 24 ? 2 : 1; i-- > 0 && pending.length;) {
      const e = pending.shift(); parts.push(e); hot = { e, a: 1 };
    }
    if (hot && (hot.a -= dt / 420) <= 0) hot = null;
    paint();
    if (pending.length || hot) tick();
  });
}

// ---------------------------------------------------------------- dragging, with a finger or a mouse
function hit(p) {
  const { L, H, openings } = design;
  const near = (x, y, r) => Math.abs(p.x - px(x)) <= r && Math.abs(p.y - py(y)) <= r;
  const loads = design.loads || [];                      // the arrows live above the wall, clear of its top edge
  for (let i = loads.length - 1; i >= 0; i--) {
    if (selL === i && Math.abs(p.x - px(loads[i]) - 14) <= 10 && Math.abs(p.y - py(H + LARROW) - 5) <= 10) return `l:${i}:x`;
    if (Math.abs(p.x - px(loads[i])) <= 11 && p.y >= py(H + LARROW) - 8 && p.y <= py(H + LFOOT) + 5) return `l:${i}`;
  }
  const lines = design.lines || [];                      // a line load: its × and its two end handles, then its bar
  for (let i = lines.length - 1; i >= 0; i--) {
    const a = px(lines[i][0]), b = px(lines[i][1]), yb = py(H + LARROW);
    if (selN === i && Math.abs(p.x - b - 15) <= 10 && Math.abs(p.y - yb - 5) <= 10) return `n:${i}:x`;
    if (p.y < yb - 9 || p.y > py(H + LFOOT) + 5) continue;
    if (Math.abs(p.x - a) <= 8) return `n:${i}:0`;
    if (Math.abs(p.x - b) <= 8) return `n:${i}:1`;
    if (p.x > a && p.x < b) return `n:${i}`;
  }
  for (let i = openings.length - 1; i >= 0; i--) {
    const b = obox(openings[i]);
    if (sel === i && Math.abs(p.x - px(b[2]) - 15) <= 11 && Math.abs(p.y - py(b[3]) + 15) <= 11) return `o:${i}:x`;
    for (const [cx, cy, k] of [[b[0], b[1], "lb"], [b[2], b[1], "rb"], [b[0], b[3], "lt"], [b[2], b[3], "rt"]])
      if (near(cx, cy, 12)) return `o:${i}:${k}`;
  }
  for (let i = openings.length - 1; i >= 0; i--) {       // an opening's four edges, the corners having had first refusal
    const b = obox(openings[i]);
    const onY = p.y <= py(b[1]) + 9 && p.y >= py(b[3]) - 9, onX = p.x >= px(b[0]) - 9 && p.x <= px(b[2]) + 9;
    if (onY && Math.abs(p.x - px(b[0])) <= 9) return `o:${i}:l.`;
    if (onY && Math.abs(p.x - px(b[2])) <= 9) return `o:${i}:r.`;
    if (onX && Math.abs(p.y - py(b[3])) <= 9) return `o:${i}:.t`;
    if (onX && openings[i].kind !== "door" && Math.abs(p.y - py(b[1])) <= 9) return `o:${i}:.b`;
  }
  if (near(L, H, 14)) return "w:c";
  if (near(L, 0, 14)) return "w:r";
  if (near(0, H, 14)) return "w:t";
  if (Math.abs(p.x - px(L)) <= 9 && p.y <= py(0) + 9 && p.y >= py(H) - 9) return "w:r";   // the wall's right edge: its length
  if (Math.abs(p.y - py(H)) <= 9 && p.x >= px(0) - 9 && p.x <= px(L) + 9) return "w:t";   // its top edge: its height
  for (let i = openings.length - 1; i >= 0; i--) {
    const b = obox(openings[i]);
    if (p.x >= px(b[0]) && p.x <= px(b[2]) && p.y <= py(b[1]) && p.y >= py(b[3])) return `o:${i}:move`;
  }
  return null;
}
const CURSOR = { "w:r": "ew-resize", "w:t": "ns-resize", "w:c": "nwse-resize", lb: "nesw-resize", rb: "nwse-resize",
  lt: "nwse-resize", rt: "nesw-resize", "l.": "ew-resize", "r.": "ew-resize", ".t": "ns-resize", ".b": "ns-resize",
  move: "move", x: "pointer", 0: "ew-resize", 1: "ew-resize" };
const cursorFor = h => !h ? "default"
  : (CURSOR[h] || CURSOR[h.split(":")[2]] || (h.startsWith("l:") ? "ew-resize" : h.startsWith("n:") ? "move" : "default"));
const at = e => { const r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };

canvas.addEventListener("pointerdown", e => {
  const p = at(e), h = hit(p);
  canvas.focus({ preventScroll: true });
  if (h && h.startsWith("l:")) {
    const [, i, edge] = h.split(":");
    if (edge === "x") { design.loads.splice(+i, 1); selL = -1; fit(design); changed(); return; }
    selL = +i; sel = selN = -1;
  } else if (h && h.startsWith("n:")) {
    const [, i, edge] = h.split(":");
    if (edge === "x") { design.lines.splice(+i, 1); selN = -1; fit(design); changed(); return; }
    selN = +i; sel = selL = -1;
  } else if (h && h.startsWith("o:")) {
    const [, i, edge] = h.split(":");
    if (edge === "x") { design.openings.splice(+i, 1); sel = -1; fit(design); changed(); return; }
    sel = +i; selL = selN = -1;
  } else if (!h) { sel = selL = selN = -1; }
  if (!h) { paint(); return; }
  e.preventDefault();
  canvas.setPointerCapture(e.pointerId);
  drag = { h, p, d0: structuredClone(design) };
  paint();
});

canvas.addEventListener("pointermove", e => {
  const p = at(e);
  if (!drag) {
    const h = hit(p);
    if (h !== hover) { hover = h; canvas.style.cursor = cursorFor(h); paint(); }
    return;
  }
  e.preventDefault();
  const dx = mx(p.x) - mx(drag.p.x), dy = my(p.y) - my(drag.p.y);
  const O = drag.d0, d = design, h = drag.h.split(":"), edge = h[h.length - 1];
  if (h[0] === "l") {                                    // a load slides along the top edge, between its neighbours
    const i = +h[1];
    if (O.loads[i] === undefined || d.loads[i] === undefined) return;
    const lo = i > 0 ? d.loads[i - 1] + LAPART : LEND, hi = i + 1 < d.loads.length ? d.loads[i + 1] - LAPART : d.L - LEND;
    const bands = loadBands(d).map(b => [Math.max(b[0], lo), Math.min(b[1], hi)]).filter(b => b[1] >= b[0] - 1e-9);
    const v = snapTo(q(O.loads[i] + dx), bands);
    if (v === null) return;
    d.loads[i] = r3(v);
  } else if (h[0] === "n") {                             // a line load slides as a whole, or stretches by one end
    const i = +h[1], s0 = O.lines[i];
    if (!s0 || !d.lines[i]) return;
    const gaps = lineBands(d, d.lines.filter((_, k) => k !== i));
    if (h.length < 3) {
      const len = s0[1] - s0[0];
      const bs = gaps.map(g => [g[0], qDn(g[1] - len)]).filter(g => g[1] >= g[0] - 1e-9);
      const x = snapTo(q(s0[0] + dx), bs);
      if (x === null) return;
      d.lines[i] = [r3(x), r3(x + len)];
    } else {
      const k = +h[2], end = s0[1 - k], g = gaps.find(g => end >= g[0] - 1e-9 && end <= g[1] + 1e-9);
      if (!g) return;
      const x = k ? clamp(q(s0[1] + dx), end + LINEMIN, Math.min(end + LINEMAX, g[1]))
                  : clamp(q(s0[0] + dx), Math.max(end - LINEMAX, g[0]), end - LINEMIN);
      d.lines[i] = k ? [r3(end), r3(x)] : [r3(x), r3(end)];
    }
  } else if (h[0] === "w") {
    if (edge !== "t") d.L = clamp(q(O.L + dx), minLength(d.openings), LMAX);
    if (edge !== "r") d.H = clamp(q(O.H + dy), minHeight(d.openings), HMAX);
  } else {
    const i = +h[1], o = d.openings[i], o0 = O.openings[i];
    if (!o || !o0) return;
    const prev = d.openings[i - 1], next = d.openings[i + 1];
    const lo = prev ? prev.x + prev.w + GAP : SIDE, hi = next ? next.x - GAP : d.L - SIDE;
    const top = d.H - HEAD;
    if (edge === "move") {
      o.x = q(clamp(o0.x + dx, lo, Math.max(lo, hi - o.w)));
      if (o.kind !== "door") o.sill = q(clamp(o0.sill + dy, MINSILL, Math.max(MINSILL, top - o.h)));
    } else {
      const ex = edge[0], ey = edge[1];                  // a corner moves both, an edge only its own ("." = leave alone)
      if (ex === "l") { const x = q(clamp(o0.x + dx, lo, o0.x + o0.w - MINW)); o.w = q(o0.w + (o0.x - x)); o.x = x; }
      else if (ex === "r") o.w = q(clamp(o0.w + dx, MINW, hi - o.x));
      if (ey === "t") o.h = q(clamp(o0.h + dy, o.kind === "door" ? 1.2 : MINH, top - sillOf(o)));
      else if (ey === "b" && o.kind !== "door") { const s = q(clamp(o0.sill + dy, MINSILL, o0.sill + o0.h - MINH)); o.h = q(o0.h - (s - o0.sill)); o.sill = s; }
    }
  }
  fit(d);
  changed();
});

const stop = e => { if (drag) { drag = null; try { canvas.releasePointerCapture(e.pointerId); } catch {} writeHash(); paint(); } };
canvas.addEventListener("pointerup", stop);
canvas.addEventListener("pointercancel", stop);
canvas.addEventListener("pointerleave", () => { if (!drag && hover) { hover = null; canvas.style.cursor = "default"; paint(); } });
canvas.addEventListener("keydown", e => {
  const del_ = e.key === "Delete" || e.key === "Backspace";
  if (del_ && design.loads && design.loads[selL] !== undefined) {
    e.preventDefault(); design.loads.splice(selL, 1); selL = -1; fit(design); changed();
  } else if (del_ && design.lines && design.lines[selN] !== undefined) {
    e.preventDefault(); design.lines.splice(selN, 1); selN = -1; fit(design); changed();
  } else if (del_ && design.openings[sel]) {
    e.preventDefault(); design.openings.splice(sel, 1); sel = -1; fit(design); changed();
  } else if (e.key === "Escape") { sel = selL = selN = -1; paint(); }
});

// ---------------------------------------------------------------- the two controls
function sync() {
  const m = MODELS.find(m => m.id === modelId);
  $("caption").textContent = `${design.script === "block" ? "Concrete blocks" : "Timber frame"} · ${design.L.toFixed(2)} × ${design.H.toFixed(2)} m`;
  const sw = $("sw");                                    // a control that can do nothing is removed, never greyed
  if (sw) {
    if (m.scripts.length < 2) sw.remove();
    else { $("script").checked = design.script === "block"; sw.className = `sw ${design.script}`; }
  }
  $("addDoor").disabled = $("addWindow").disabled = design.openings.length >= MAXOPS || !freeSpan(design);
  const bl = $("addLoad"), bn = $("addLine");             // only a network that reads loads has these buttons at all
  if (bl) { if (!m.loads) bl.remove(); else bl.disabled = loadSpot(design) === null; }
  if (bn) { if (!m.loads) bn.remove(); else bn.disabled = lineSpot(design) === null; }
}
$("script")?.addEventListener("change", e => { design.script = e.target.checked ? "block" : "frame"; changed(); });
$("addDoor").addEventListener("click", () => { selL = selN = -1; addOpening("door"); });
$("addWindow").addEventListener("click", () => addOpening("window"));
$("addLoad")?.addEventListener("click", addLoad);
$("addLine")?.addEventListener("click", addLine);
$("random").addEventListener("click", () => { design = randomDesign(); sel = selL = selN = -1; changed(); });
if (ABOUT[modelId]) { $("about").textContent = ABOUT[modelId]; $("aboutLink").hidden = false; }
$("aboutLink").addEventListener("click", e => { e.preventDefault(); $("about").hidden = !$("about").hidden; });
addEventListener("resize", paint);
if (window.ResizeObserver) new ResizeObserver(paint).observe(canvas);

sync();
writeHash();
paint();                                                 // the wall is on screen and draggable before the weights arrive
loadModel();
