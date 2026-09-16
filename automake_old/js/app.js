// The page: design a wall, pick a network, watch it write the wall live next to the script's, look inside the network.
import { BRIEFS, ITEMS, buildWall, scoreElements } from "./wall.js";
import { FIELD_NAMES, FIELD_WORDS, NET_SVG, distMarkup, elemBox, highlightNet, partLabel, partStyle, tokensMarkup, viewBox, wallMarkup,
  wrongStyle, BLOCK_FILL, BLOCK_STROKE, ITEM_COLOURS, WRONG_STROKE } from "./draw.js";

const qs = new URLSearchParams(location.search);
const FORMAT = qs.get("format") || "f16";

// ---------------------------------------------------------------- the networks on offer: one line each
// file: web/automake/model/<file>.<format>.json/.bin, written by scripts/export_web_model.py
const MODELS = [
  { id: "n0", file: "n0", name: "N0", title: "Frames", scripts: ["frame"],
    blurb: "Trained on framed walls from the framing script only. It has no block parts in its vocabulary.",
    numbers: "N0 on framed walls it had not seen: precision 0.91, recall 0.88 beyond the training range (120 runs); exactly right in 62% of runs in the training range (48) and 52% beyond it up to 6 m (42)." },
  { id: "o4", file: "o4", name: "O4", title: "Frames, then blocks", scripts: ["frame", "block"],
    blurb: "N0, then 50 minutes on concrete block walls. It kept its framing by rehearsing framed walls it had written itself and the world kept: no framing script or framing data after the first stage.",
    numbers: "O4 on walls it had not seen, beyond the training range: framed walls precision 0.91, recall 0.88 (N0: 0.91 / 0.88); concrete block walls precision 0.68, recall 0.72, and 0.73 / 0.71 with the world refusing overlaps (120 runs each)." },
  { id: "m0", file: "m0", name: "M0", title: "Both together", scripts: ["frame", "block"],
    blurb: "Trained on both scripts together from the start, over several rounds: the best block walls so far.",
    numbers: "M0 on walls it had not seen: framed walls exactly right in 77% of runs in the training range (48) and 55% beyond it up to 6 m (42); block walls beyond the training range precision 0.79, recall 0.82 as written, 0.84 / 0.81 with the world refusing overlaps (120 runs each)." },
];
const DEFAULT_MODEL = "o4";

// ---------------------------------------------------------------- designs and presets
const door = (x, w, h) => ({ kind: "door", x, w, h, sill: 0 });
const win = (x, w, h, sill) => ({ kind: "window", x, w, h, sill });
const D = (script, L, H, openings, more = {}) => ({ script, L, H, openings, start: "empty", share: 0.5, seed: 7, reject: false, ...more });
const THREE = [door(0.5, 0.9, 2.1), win(2.0, 1.2, 1.2, 0.9), win(3.9, 1.4, 1.2, 0.9)];
// chosen examples, each run first with the Python network (2026-09-15); the tiles show how each goes
const PRESETS = [
  { label: "Framed wall, three openings", sub: "N0 · beyond training", model: "n0", design: D("frame", 5.8, 2.7, THREE) },
  { label: "The same wall after learning blocks", sub: "O4 · its framing kept", model: "o4", design: D("frame", 5.8, 2.7, THREE) },
  { label: "Tall wall, 2.6 m window", sub: "N0 · beyond training", model: "n0", design: D("frame", 4.8, 3.05, [win(1.1, 2.6, 1.4, 0.8)]) },
  { label: "Fill the holes", sub: "N0 · 60% of the parts given", model: "n0", design: D("frame", 6.0, 2.6, [door(0.8, 1.0, 2.1), win(3.0, 1.8, 1.2, 0.8)], { start: "subset", share: 0.6 }) },
  { label: "Finish a half-built block wall", sub: "O4 · its blocks after 50 minutes", model: "o4", design: D("block", 5.4, 2.7, [door(0.6, 0.9, 2.1), win(2.8, 1.6, 1.2, 0.9)], { start: "prefix", share: 0.5 }) },
  { label: "The world refuses overlaps", sub: "O4 · blocks with holes, a rule outside the network", model: "o4", design: D("block", 5.8, 2.7, [door(0.3, 0.9, 2.1), win(1.7, 0.8, 1.2, 0.9), win(3.0, 1.0, 1.2, 0.9), door(4.5, 1.0, 2.1)], { start: "subset", share: 0.6, reject: true }) },
  { label: "A 7.2 m block wall", sub: "M0 · both scripts from the start", model: "m0", design: D("block", 7.2, 2.8, [door(0.6, 1.0, 2.1), win(2.6, 1.2, 1.2, 0.9), win(4.8, 1.6, 1.2, 0.9)]) },
];
const TRAIN = { L: [2.4, 6.0], H: [2.2, 2.8], ops: 2, w: [0.4, 1.8], doorW: [0.6, 1.8], doorH: [1.8, 2.4], winH: [0.4, 1.8], sill: [0.3, 1.5], side: 0.2, gap: 0.3, above: 0.35 };

const $ = id => document.getElementById(id);
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const r2 = v => Number((+v).toFixed(2));
const q05 = v => r2(Math.round(v / 0.05) * 0.05);
const pct = v => `${Math.round(100 * v)}%`;
const clone = o => JSON.parse(JSON.stringify(o));

function fit(d, sort = true) {
  d.L = r2(clamp(q05(d.L), 1.2, 7.9));
  d.H = r2(clamp(q05(d.H), 2.0, 3.1));
  d.openings = d.openings.slice(0, 4);
  for (const o of d.openings) {
    o.w = r2(clamp(q05(o.w), 0.3, Math.max(0.3, Math.min(3.0, d.L - 0.2))));
    if (o.kind === "door") { o.sill = 0; o.h = r2(clamp(q05(o.h), 0.6, d.H - 0.15)); }
    else { o.h = r2(clamp(q05(o.h), 0.3, d.H - 0.45)); o.sill = r2(clamp(q05(o.sill), 0.15, d.H - o.h - 0.25)); }
    o.x = r2(clamp(q05(o.x), 0.05, Math.max(0.05, d.L - o.w - 0.05)));
  }
  if (sort) {                                           // left to right, each clear of the one before; what no longer fits is dropped
    d.openings.sort((a, b) => a.x - b.x);
    const kept = [];
    for (const o of d.openings) {
      const prev = kept[kept.length - 1];
      if (prev && o.x < prev.x + prev.w + 0.1) o.x = r2(q05(prev.x + prev.w + 0.1 + 0.024));
      if (o.x + o.w <= d.L - 0.05 + 1e-9) kept.push(o);
    }
    d.openings = kept;
  }
  d.share = r2(clamp(+d.share, 0.1, 0.9));
  return d;
}

function beyondTraining(d) {
  const out = [];
  if (d.openings.length > TRAIN.ops) out.push(`${d.openings.length} openings`);
  if (d.L > TRAIN.L[1] + 1e-6) out.push(`${d.L.toFixed(1)} m long`); else if (d.L < TRAIN.L[0] - 1e-6) out.push(`${d.L.toFixed(1)} m short`);
  if (d.H > TRAIN.H[1] + 1e-6) out.push(`${d.H.toFixed(2)} m tall`); else if (d.H < TRAIN.H[0] - 1e-6) out.push(`${d.H.toFixed(2)} m low`);
  if (d.openings.some(o => o.w > TRAIN.w[1] + 1e-6)) out.push("a wide opening");
  if (d.openings.some(o => o.kind === "door" ? (o.h < TRAIN.doorH[0] - 1e-6 || o.h > TRAIN.doorH[1] + 1e-6) : (o.h > TRAIN.winH[1] + 1e-6 || o.sill < TRAIN.sill[0] - 1e-6 || o.sill > TRAIN.sill[1] + 1e-6))) out.push("an unusual opening");
  return out;
}

// the script's parts already there at the start
function mulberry(seed) { let a = seed >>> 0; return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
function startElements(d, elems) {
  if (d.start === "prefix") return elems.slice(0, Math.round(d.share * elems.length));
  if (d.start === "subset") { const rnd = mulberry(d.seed * 7919 + 1); return elems.filter(() => rnd() < d.share); }
  return [];
}

// ---------------------------------------------------------------- link
function writeHash() {
  const d = design;
  const o = d.openings.map(o => o.kind === "door" ? `d${o.x}_${o.w}_${o.h}` : `w${o.x}_${o.w}_${o.h}_${o.sill}`).join("~");
  const p = new URLSearchParams({ n: modelId, t: d.script, L: d.L, H: d.H, o, s: d.start, f: d.share, seed: d.seed, w: d.reject ? 1 : 0 });
  history.replaceState(null, "", "#" + p.toString());
}
function readHash() {
  if (location.hash.length < 3) return null;
  try {
    const p = new URLSearchParams(location.hash.slice(1));
    const ops = (p.get("o") || "").split("~").filter(Boolean).map(s => {
      const v = s.slice(1).split("_").map(Number);
      return s[0] === "d" ? door(v[0], v[1], v[2]) : win(v[0], v[1], v[2], v[3]);
    });
    const m = MODELS.find(m => m.id === p.get("n")) || MODELS.find(m => m.id === DEFAULT_MODEL);
    const d = D(p.get("t") === "block" ? "block" : "frame", +p.get("L") || 5.8, +p.get("H") || 2.7, ops,
      { start: ["empty", "prefix", "subset"].includes(p.get("s")) ? p.get("s") : "empty", share: +p.get("f") || 0.5, seed: +p.get("seed") || 7, reject: p.get("w") === "1" });
    if (!m.scripts.includes(d.script)) d.script = m.scripts[0];
    return { model: m.id, design: fit(d) };
  } catch { return null; }
}

// ---------------------------------------------------------------- state
const fromHash = readHash();
let modelId = fromHash ? fromHash.model : PRESETS[0].model;
let design = fromHash ? fromHash.design : fit(clone(PRESETS[0].design));
let presetOn = fromHash ? -1 : 0;
let script = null, start = [];
let modelReady = false, modelMeta = null, wantRun = false;
let runSeq = 0, run = null;
let cur = { k: -1, f: 4 }, follow = true, selF = null;
const play = { on: false, acc: 0, last: 0 };
let drag = null, dragView = null, dirty = false, runTimer = null;
window.__automake = { loads: [], runs: [] };

function newRun() { return { id: ++runSeq, parts: [], passes: [], passEnd: [], done: false, result: null, encoding: -1, error: null, sent: false }; }
run = newRun();

// ---------------------------------------------------------------- the model thread
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
const modelUrl = new URL("../../automake/model/", import.meta.url).href;   // the weights live with the current page; this is the frozen old one
function loadModel() {
  modelReady = false; modelMeta = null;
  const m = MODELS.find(m => m.id === modelId);
  worker.postMessage({ type: "load", url: modelUrl, file: m.file, format: FORMAT, backend: qs.get("backend"), threads: qs.get("threads") });
  setStatus(`Loading ${m.name}…`);
  window.__automake.loadStart = performance.now();
}
worker.onmessage = e => {
  const ev = e.data;
  if (ev.type === "progress") {
    const m = MODELS.find(m => m.file === ev.file);
    if (!m || m.id !== modelId) return;
    setStatus(`Loading ${m.name} · ${(ev.loaded / 1e6).toFixed(1)} of ${(ev.total / 1e6).toFixed(1)} MB`, ev.loaded / ev.total);
    return;
  }
  if (ev.type === "ready") {
    const m = MODELS.find(m => m.file === ev.file);
    if (!m || m.id !== modelId) return;
    modelReady = true; modelMeta = ev.manifest;
    const ms = performance.now() - window.__automake.loadStart;
    window.__automake.loads.push({ model: m.id, ms, fetchMs: ev.fetchMs, parseMs: ev.parseMs, poolMs: ev.poolMs, helpers: ev.helpers, bytes: ev.bytes, backend: ev.backend });
    setStatus(`${m.name} ready · ${(ev.bytes / 1e6).toFixed(1)} MB · ${ev.backend}`);
    $("meta").textContent = `${m.name} (run ${ev.manifest.run}, trained ${ev.manifest.trained}, ${(ev.manifest.params / 1e6).toFixed(2)}M parameters, weights exported at commit ${ev.manifest.commit}) · runs entirely in your browser; nothing is sent anywhere · © 2026 Karl-Johan Sørensen. All rights reserved.`;
    if (wantRun) sendRun();
    return;
  }
  if (ev.id !== run.id) return;
  if (ev.type === "error") { run.error = ev.message; setStatus("Something went wrong: " + ev.message.split("\n")[0]); invalidate(); return; }
  if (ev.type === "encode") { run.encoding = ev.pass; run.passes[ev.pass] = { tokens: ev.tokens, present: ev.present, finished: false }; }
  else if (ev.type === "encoded") { run.passes[ev.pass].cover = ev.cover; }
  else if (ev.type === "part") {
    run.encoding = -1;
    run.parts.push({ pass: ev.pass, j: ev.j, elem: ev.elem, picks: ev.picks, refused: false, ms: ev.ms });
    if (follow) { cur = { k: run.parts.length - 1, f: 4 }; }
  } else if (ev.type === "pass") {
    run.encoding = -1;
    const P = run.passes[ev.pass];
    Object.assign(P, { finished: true, written: ev.written.length, kept: ev.kept.length, refused: ev.refused });
    run.passEnd[ev.pass] = run.parts.length - 1;
    if (ev.refused) {
      let q = 0;
      run.parts.forEach(p => {
        if (p.pass !== ev.pass) return;
        if (q < ev.kept.length && ev.kept[q].every((v, i) => v === p.elem[i])) q++; else p.refused = true;
      });
    }
  } else if (ev.type === "done") {
    run.done = true; run.result = ev;
    window.__automake.runs.push({ model: modelId, script: design.script, parts: ev.parts, passes: ev.passes, term: ev.term, ms: ev.ms, firstPartMs: ev.firstPartMs, reject: design.reject, start: design.start, prof: ev.prof });
    const m = MODELS.find(m => m.id === modelId);
    setStatus(`${m.name} wrote ${ev.parts} parts in ${(ev.ms / 1000).toFixed(1)} s (first part after ${((ev.firstPartMs || 0) / 1000).toFixed(2)} s)`);
  }
  invalidate();
};

function sendRun() {
  if (!modelReady) { wantRun = true; return; }
  wantRun = false;
  run.sent = true;
  worker.postMessage({ type: "run", id: run.id, wall: script.wall, ops: script.ops, start, brief: BRIEFS[design.script], reject: design.reject, prof: qs.has("prof") });
  const m = MODELS.find(m => m.id === modelId);
  setStatus(`${m.name} is writing…`);
}

function setStatus(text, frac = null) {
  $("status").innerHTML = text.replace(/&/g, "&amp;").replace(/</g, "&lt;") + (frac === null ? "" : `<span class="bar"><i style="width:${Math.round(100 * frac)}%"></i></span>`);
}

// ---------------------------------------------------------------- design changes
function rebuild() {
  script = buildWall({ L: design.L, H: design.H, openings: design.openings.map(o => [o.x, o.w, o.kind === "door" ? 0 : o.sill, o.h]) }, design.script);
  start = startElements(design, script.elements);
}

function changed({ immediate = false, keepPreset = false } = {}) {
  fit(design);
  if (!keepPreset) presetOn = -1;
  rebuild();
  worker.postMessage({ type: "cancel" });
  run = newRun();
  cur = { k: -1, f: 4 }; follow = true; selF = null; play.on = false;
  syncControls();
  writeHash();
  invalidate();
  clearTimeout(runTimer);
  runTimer = setTimeout(sendRun, immediate ? 0 : 180);
}

function selectModel(id, { run: go = true } = {}) {
  const m = MODELS.find(m => m.id === id);
  if (!m) return;
  const switching = id !== modelId || !modelReady;
  modelId = id;
  if (!m.scripts.includes(design.script)) design.script = m.scripts[0];
  if (switching) loadModel();
  if (go) changed({ immediate: true, keepPreset: true });
}

function applyPreset(i) {
  const p = PRESETS[i];
  presetOn = i;
  design = fit(clone(p.design));
  renderOps();
  selectModel(p.model);
}

function surprise() {
  const u = (a, b) => a + Math.random() * (b - a);
  const m = MODELS.find(m => m.id === modelId);
  const scriptKind = m.scripts[Math.floor(Math.random() * m.scripts.length)];
  for (let tries = 0; tries < 100; tries++) {
    const ext = Math.random() < 0.5;
    const tags = ext ? ["openings", "long", "tall", "wide"].filter(() => Math.random() < 0.5) : [];
    if (ext && !tags.length) continue;
    const L = q05(tags.includes("long") ? u(6.2, 7.9) : u(2.4, 6.0)), H = q05(tags.includes("tall") ? u(2.85, 3.1) : u(2.2, 2.8));
    let n = tags.includes("openings") ? 3 + Math.floor(Math.random() * 2) : [0, 1, 2][Math.random() < 0.2 ? 0 : (Math.random() < 0.56 ? 1 : 2)];
    if (tags.includes("wide")) n = Math.max(n, 1);
    const wide = tags.includes("wide") ? Math.floor(Math.random() * n) : -1;
    const ops = Array.from({ length: n }, (_, i) => {
      const isDoor = Math.random() < 0.4, w = q05(i === wide ? u(2.0, 3.0) : u(isDoor ? 0.6 : 0.4, 1.8));
      if (isDoor) return door(0, w, q05(u(1.8, Math.max(1.8, Math.min(2.4, H - 0.35)))));
      const sill = q05(u(0.3, Math.min(1.5, H - 0.75)));
      return win(0, w, q05(u(0.4, Math.max(0.4, Math.min(1.8, H - 0.35 - sill)))), sill);
    });
    const free = L - 2 * TRAIN.side - ops.reduce((a, o) => a + o.w, 0) - TRAIN.gap * Math.max(0, n - 1);
    if (free < 0) continue;
    const slack = ops.map(() => u(0, free)).sort((a, b) => a - b);
    let a = TRAIN.side, used = 0;
    ops.forEach((o, i) => { a += slack[i] - used; used = slack[i]; o.x = q05(a); a += o.w + TRAIN.gap; });
    design = fit(D(scriptKind, L, H, ops, { reject: design.reject }));
    renderOps();
    changed({ immediate: true });
    return;
  }
}

// ---------------------------------------------------------------- controls
function renderModels() {
  $("modelSeg").style.gridTemplateColumns = `repeat(${MODELS.length}, 1fr)`;
  $("modelSeg").innerHTML = MODELS.map(m => `<button data-model="${m.id}" title="${m.blurb}"><b>${m.name}</b><small>${m.title}</small></button>`).join("");
  $("modelSeg").onclick = e => { const b = e.target.closest("button"); if (b) { presetOn = -1; selectModel(b.dataset.model); } };
}

function renderPresets() {
  $("presets").innerHTML = PRESETS.map((p, i) => `<button class="chip" data-p="${i}">${p.label}<small>${p.sub}</small></button>`).join("")
    + `<button class="chip surprise" data-p="surprise">Surprise me<small>a random wall</small></button>`;
  $("presets").onclick = e => { const b = e.target.closest("button"); if (!b) return; b.dataset.p === "surprise" ? surprise() : applyPreset(+b.dataset.p); };
}

const OPF = [["x", "From left", 0.05], ["w", "Width", 0.05], ["h", "Height", 0.05], ["sill", "Sill", 0.05]];
function renderOps() {
  $("ops").innerHTML = design.openings.map((o, i) => `<div class="op"><div class="ophd"><span>${o.kind === "door" ? "Door" : "Window"} ${i + 1}</span><button data-rm="${i}" title="Remove">×</button></div><div class="opgrid">`
    + OPF.filter(f => f[0] !== "sill" || o.kind !== "door").map(([f, label, step]) => `<div class="row"><label for="o${i}${f}">${label} <b id="o${i}${f}v"></b></label>`
      + `<input type="range" id="o${i}${f}" data-op="${i}" data-f="${f}" step="${step}"><div class="hint" id="o${i}${f}h"></div></div>`).join("") + "</div></div>").join("")
    || `<div class="hint">No openings: a plain wall.</div>`;
  syncControls();
}

function syncControls() {
  const d = design, m = MODELS.find(m => m.id === modelId);
  document.querySelectorAll("#modelSeg button").forEach(b => b.classList.toggle("on", b.dataset.model === modelId));
  $("modelBlurb").textContent = m.blurb;
  document.querySelectorAll("#scriptSeg button").forEach(b => {
    b.classList.toggle("on", b.dataset.script === d.script);
    b.disabled = !m.scripts.includes(b.dataset.script);
  });
  $("scriptHint").textContent = m.scripts.includes("block") ? "" : `${m.name} learned framed walls only, so it cannot be asked for concrete blocks.`;
  document.querySelectorAll(".presets .chip").forEach(b => b.classList.toggle("on", +b.dataset.p === presetOn));
  const setRange = (id, lo, hi, v) => { const s = $(id); if (!s) return; s.min = lo; s.max = Math.max(lo, hi); s.value = v; };
  setRange("L", 1.2, 7.9, d.L); setRange("H", 2.0, 3.1, d.H);
  $("Lv").textContent = `${d.L.toFixed(2)} m`; $("Hv").textContent = `${d.H.toFixed(2)} m`;
  const hint = (id, text, out) => { const h = $(id); if (h) { h.textContent = text; h.classList.toggle("beyond", !!out); } };
  hint("Lh", "trained on 2.4 to 6 m", d.L < TRAIN.L[0] - 1e-6 || d.L > TRAIN.L[1] + 1e-6);
  hint("Hh", "trained on 2.2 to 2.8 m", d.H < TRAIN.H[0] - 1e-6 || d.H > TRAIN.H[1] + 1e-6);
  $("opCount").textContent = d.openings.length ? `(${d.openings.length} of 4)` : "";
  $("addDoor").disabled = $("addWindow").disabled = d.openings.length >= 4;
  hint("opsHint", d.openings.length > TRAIN.ops ? "trained with up to 2 openings" : "Drag openings and the wall's edges on the drawings, too.", d.openings.length > TRAIN.ops);
  d.openings.forEach((o, i) => {
    const s = o.kind === "door" ? 0 : o.sill;
    setRange(`o${i}x`, 0.05, d.L - o.w - 0.05, o.x);
    setRange(`o${i}w`, 0.3, Math.min(3.0, d.L - 0.2), o.w);
    setRange(`o${i}h`, o.kind === "door" ? 0.6 : 0.3, o.kind === "door" ? d.H - 0.15 : d.H - s - 0.25, o.h);
    if (o.kind !== "door") setRange(`o${i}sill`, 0.15, d.H - o.h - 0.25, o.sill);
    for (const f of ["x", "w", "h", "sill"]) if ($(`o${i}${f}v`)) $(`o${i}${f}v`).textContent = `${o[f].toFixed(2)}`;
    const W = o.kind === "door" ? TRAIN.doorW : TRAIN.w;
    hint(`o${i}wh`, `trained ${W.join("–")}`, o.w > W[1] + 1e-6 || o.w < W[0] - 1e-6);
    const T = o.kind === "door" ? TRAIN.doorH : TRAIN.winH;
    hint(`o${i}hh`, `trained ${T.join("–")}`, o.h < T[0] - 1e-6 || o.h > T[1] + 1e-6);
    if (o.kind !== "door") hint(`o${i}sillh`, `trained ${TRAIN.sill.join("–")}`, o.sill < TRAIN.sill[0] - 1e-6 || o.sill > TRAIN.sill[1] + 1e-6);
    hint(`o${i}xh`, "metres", false);
  });
  $("start").value = d.start;
  $("shareRow").hidden = d.start === "empty";
  $("share").value = d.share; $("sharev").textContent = pct(d.share);
  $("reject").checked = d.reject;
  $("numbers").innerHTML = `<b>How well.</b> ${m.numbers} A part counts as right within 30 mm of the script's, lengths within 12 mm.`;
}

function bindControls() {
  $("scriptSeg").onclick = e => { const b = e.target.closest("button"); if (!b || b.disabled) return; design.script = b.dataset.script; changed({ immediate: true }); };
  for (const id of ["L", "H"]) $(id).addEventListener("input", e => { design[id] = +e.target.value; changed(); });
  $("ops").addEventListener("input", e => { const s = e.target; if (s.dataset.op === undefined) return; design.openings[+s.dataset.op][s.dataset.f] = +s.value; fitKeepingOrder(); changed(); });
  $("ops").addEventListener("change", () => renderOps());
  $("ops").addEventListener("click", e => { const b = e.target.closest("[data-rm]"); if (!b) return; design.openings.splice(+b.dataset.rm, 1); renderOps(); changed({ immediate: true }); });
  const add = kind => {
    const d = design;
    let a = 0, best = [0, 0];
    for (const o of [...d.openings].sort((p, q) => p.x - q.x)) { if (o.x - a > best[1] - best[0]) best = [a, o.x]; a = Math.max(a, o.x + o.w); }
    if (d.L - a > best[1] - best[0]) best = [a, d.L];
    const w = clamp(best[1] - best[0] - 0.6, 0.3, kind === "door" ? 0.9 : 1.2);
    const x = (best[0] + best[1]) / 2 - w / 2;
    d.openings.push(kind === "door" ? door(x, w, 2.1) : win(x, w, 1.2, 0.9));
    fit(d); renderOps(); changed({ immediate: true });
  };
  $("addDoor").onclick = () => add("door"); $("addWindow").onclick = () => add("window");
  $("start").onchange = e => { design.start = e.target.value; changed({ immediate: true }); };
  $("share").addEventListener("input", e => { design.share = +e.target.value; changed(); });
  $("reject").onchange = e => { design.reject = e.target.checked; changed({ immediate: true }); };
  $("copyLink").onclick = async () => {
    writeHash();
    try { await navigator.clipboard.writeText(location.href); $("copyLink").textContent = "Link copied"; } catch { $("copyLink").textContent = "Copy the address bar"; }
    setTimeout(() => { $("copyLink").textContent = "Copy a link to this wall"; }, 1800);
  };
  $("showScript").onchange = e => { $("walls").classList.toggle("solo", !e.target.checked); invalidate(); };
  if (matchMedia("(max-width: 760px)").matches) { $("showScript").checked = false; $("walls").classList.add("solo"); }
  // player
  $("replay").onclick = () => { if (!run.parts.length) return; follow = false; selF = null; cur = { k: 0, f: 0 }; startPlay(); };
  $("pause").onclick = () => { if (play.on) { play.on = false; invalidate(); } else if (run.parts.length) { if (cur.k * 5 + cur.f >= run.parts.length * 5 - 1) cur = { k: 0, f: 0 }; follow = false; startPlay(); } };
  $("scrub").addEventListener("input", e => { play.on = false; follow = false; selF = null; const v = +e.target.value; cur = { k: Math.floor(v / 5), f: v % 5 }; invalidate(); });
  $("seq").onclick = e => { const li = e.target.closest("li[data-k]"); if (!li) return; play.on = false; follow = false; selF = null; cur = { k: +li.dataset.k, f: 4 }; $("net").open = true; invalidate(); };
  $("pickCard").onclick = e => { const b = e.target.closest("button[data-f]"); if (!b) return; selF = +b.dataset.f; if (!follow) cur.f = selF; invalidate(); };
  for (const id of ["showFree", "showCanvas"]) $(id).onchange = invalidate;
  $("net").addEventListener("toggle", invalidate);
  $("tokDetails").addEventListener("toggle", invalidate);
  for (const svg of [$("svgScript"), $("svgModel")]) attachDrag(svg);
}

function fitKeepingOrder() { fit(design, false); }

function startPlay() { play.on = true; play.acc = 0; play.last = performance.now(); requestAnimationFrame(tick); invalidate(); }
function tick(ts) {
  if (!play.on) return;
  const dt = Math.min(0.25, (ts - play.last) / 1000);
  play.last = ts;
  play.acc += dt * +$("speed").value;
  const steps = Math.floor(play.acc);
  play.acc -= steps;
  if (steps) {
    const max = run.parts.length * 5 - 1;
    let pos = Math.min(max, cur.k * 5 + cur.f + steps);
    if (pos >= max) { pos = max; play.on = false; if (!run.done) follow = true; }
    cur = { k: Math.floor(pos / 5), f: pos % 5 };
    invalidate();
  }
  if (play.on) requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- dragging on the drawings
function wallPoint(svg, e) {
  const pt = svg.createSVGPoint();
  pt.x = e.clientX; pt.y = e.clientY;
  const p = pt.matrixTransform(svg.getScreenCTM().inverse());
  return { x: p.x, y: -p.y };
}
function attachDrag(svg) {
  svg.addEventListener("pointerdown", e => {
    const h = e.target.closest("[data-h]");
    if (!h) return;
    e.preventDefault();
    svg.setPointerCapture(e.pointerId);
    const [what, idx, edge] = h.dataset.h.split(":");
    design.openings.sort((a, b) => a.x - b.x);
    const growX = what === "wall" && edge === "r", growY = what === "wall" && edge === "t";
    dragView = viewBox(growX ? 7.9 : design.L, growY ? 3.1 : design.H);
    drag = { what, i: +idx, edge, p0: null, orig: clone(design), svg };
    svg.setAttribute("viewBox", dragView);
    drag.p0 = wallPoint(svg, e);
    worker.postMessage({ type: "cancel" });
    run = newRun(); cur = { k: -1, f: 4 }; play.on = false;
    invalidate();
  });
  svg.addEventListener("pointermove", e => {
    if (!drag || drag.svg !== svg) return;
    const p = wallPoint(svg, e), dx = p.x - drag.p0.x, dy = p.y - drag.p0.y, O = drag.orig, d = design;
    if (drag.what === "wall") {
      if (drag.edge === "r") d.L = O.L + dx; else d.H = O.H + dy;
      d.openings = clone(O.openings);
    } else {
      const o0 = O.openings[drag.i], o = d.openings[drag.i];
      Object.assign(o, clone(o0));
      const prev = O.openings[drag.i - 1], next = O.openings[drag.i + 1];
      const lo = prev ? prev.x + prev.w + 0.1 : 0.05, hi = next ? next.x - 0.1 : O.L - 0.05;
      if (drag.edge === "move") { o.x = clamp(o0.x + dx, lo, Math.max(lo, hi - o0.w)); if (o.kind !== "door") o.sill = o0.sill + dy; }
      else if (drag.edge === "l") { const x = clamp(o0.x + dx, lo, o0.x + o0.w - 0.3); o.w = o0.w + (o0.x - x); o.x = x; }
      else if (drag.edge === "r") o.w = clamp(o0.w + dx, 0.3, hi - o0.x);
      else if (drag.edge === "t") o.h = o0.h + dy;
      else if (drag.edge === "b") { const s = clamp(o0.sill + dy, 0.15, o0.sill + o0.h - 0.3); o.h = o0.h - (s - o0.sill); o.sill = s; }
    }
    fit(d, false);
    rebuild();
    presetOn = -1;
    syncControls();
    invalidate();
  });
  const end = e => {
    if (!drag || drag.svg !== svg) return;
    drag = null; dragView = null;
    renderOps();
    changed({ immediate: true });
  };
  svg.addEventListener("pointerup", end);
  svg.addEventListener("pointercancel", end);
}

// ---------------------------------------------------------------- drawing
function invalidate() { if (!dirty) { dirty = true; requestAnimationFrame(() => { dirty = false; render(); }); } }

function stateAt(k) {
  const drawn = start.map(e => ({ e, kind: "given" }));
  for (let i = 0; i <= k && i < run.parts.length; i++) {
    const p = run.parts[i], P = run.passes[p.pass];
    drawn.push({ e: p.elem, kind: "part", i, refused: p.refused && P && P.finished && k >= run.passEnd[p.pass] });
  }
  return drawn;
}

function edgeLine(part, f, L, H, cls) {
  const b = elemBox(part.elem, L, H);
  if (f === 1 || f === 3) { const x = f === 1 ? b[0] : b[2]; return { cls, x1: x, y1: b[1] - 0.06, x2: x, y2: b[3] + 0.06 }; }
  const y = f === 2 ? b[1] : b[3];
  return { cls, x1: b[0] - 0.06, y1: y, x2: b[2] + 0.06, y2: y };
}

function sourceLine(part, pk, L, H) {
  if (!pk || pk.f === 0 || !pk.src || pk.fromRuler) return null;
  const axisX = pk.f === 1 || pk.f === 3, P = run.passes[part.pass];
  let r, side;
  if (pk.src.kind === "box") {
    const t = P.tokens.rects[pk.src.token];
    r = [t[0] + L / 2, t[1] + H / 2, t[2] + L / 2, t[3] + H / 2]; side = pk.src.side;
  } else {
    const other = run.parts.find(p => p.pass === part.pass && p.j === pk.src.part);
    if (!other) return null;
    r = elemBox(other.elem, L, H); side = pk.src.field >= 3 ? 1 : 0;
  }
  if (axisX) { const x = side ? r[2] : r[0]; return { cls: "srcline", x1: x, y1: r[1], x2: x, y2: r[3] }; }
  const y = side ? r[3] : r[1];
  return { cls: "srcline", x1: r[0], y1: y, x2: r[2], y2: y };
}

function describeSource(part, pk) {
  if (!pk.src || pk.fromRuler) return `picked a tick on the ruler: the gate put ${pct(1 - pk.g)} on the ruler, and no visible edge explains this tick`;
  const axisX = pk.f === 1 || pk.f === 3;
  const sideWord = s => axisX ? (s ? "right" : "left") : (s ? "top" : "bottom");
  const off = pk.src.offset;
  const offText = off === 0 ? "exactly on" : `${Math.abs(off)} tick${Math.abs(off) === 1 ? "" : "s"} (${Math.abs(off) * 5} mm) ${axisX ? (off > 0 ? "right of" : "left of") : (off > 0 ? "above" : "below")}`;
  let what;
  if (pk.src.kind === "box") {
    const tok = run.passes[part.pass].tokens, t = tok.types[pk.src.token];
    if (t === 0) what = "the wall";
    else if (t === 1) what = `opening ${tok.types.slice(0, pk.src.token + 1).filter(x => x === 1).length}`;
    else if (t === 2) what = `a ${partLabel([tok.items[pk.src.token] - 1, 0, 0, 78, 0]) === "block" ? "block" : ITEMS[tok.items[pk.src.token] - 1]} already there`;
    else what = "a free-space box";
    return `copied an edge: ${offText} the ${sideWord(pk.src.side)} edge of ${what}`;
  }
  const own = pk.src.part === part.j;
  return `copied an edge: ${offText} ${own ? "this part's own" : `the ${sideWord(pk.src.field >= 3 ? 1 : 0)} edge of part ${pk.src.part + 1} of this pass`}${own ? ` ${sideWord(pk.src.field >= 3 ? 1 : 0)} edge` : ""}`;
}

function pickCard(part, f, L, H) {
  const m = MODELS.find(m => m.id === modelId);
  const items = (modelMeta && modelMeta.items) || ITEMS;
  const chips = FIELD_NAMES.map((n, i) => `<button data-f="${i}" class="${i === f ? "on" : ""}">${n} ${i === 0 ? items[part.elem[0]] : part.elem[i]}</button>`).join("");
  const pk = part.picks && part.picks[f];
  if (!pk) return `<div class="chips">${chips}</div>`;
  if (f === 0) {
    const names = [...items, "STOP"];
    const rows = pk.probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p).slice(0, 5)
      .map(({ p, i }) => `<span>${names[i]}</span><span class="b"><i class="${i === pk.v ? "on" : ""}" style="width:${Math.max(1, 100 * p)}%"></i></span><span>${pct(p)}</span>`).join("");
    return `<div class="chips">${chips}</div><div class="line">Pick 1 of 5: <b>which item</b>. ${m.name} chose <b>${partLabel(part.elem)}</b>.</div><div class="bars">${rows}</div>`;
  }
  const axisX = f === 1 || f === 3;
  const metres = pk.v * 0.005 - (axisX ? 4 : 1.6) + (axisX ? L / 2 : H / 2);
  return `<div class="chips">${chips}</div>
    <div class="line">Pick ${f + 1} of 5: <b>the ${FIELD_WORDS[f]}</b> at tick ${pk.v}, ${metres.toFixed(3)} m from the wall's ${axisX ? "left end" : "bottom"} (p = ${pk.p.toFixed(2)}).</div>
    <div class="line">It ${describeSource(part, pk)}.</div>
    <div class="mix" title="how the chosen tick's probability splits between copying an edge and the ruler"><span class="c" style="width:${100 * pk.copyShare}%"></span><span class="r" style="width:${100 * (1 - pk.copyShare)}%"></span></div>
    <div class="hint">orange: copy an edge + offset · grey: the ruler · below: the pick's probabilities within 40 ticks (200 mm)</div>${distMarkup(pk)}`;
}

function render() {
  if (!script) return;
  const d = design, L = d.L, H = d.H, m = MODELS.find(m => m.id === modelId);
  const view = dragView || viewBox(L, H);
  const svgS = $("svgScript"), svgM = $("svgModel");
  svgS.setAttribute("viewBox", view); svgM.setAttribute("viewBox", view);
  svgS.innerHTML = wallMarkup({ L, H, openings: d.openings, handles: true,
    parts: script.elements.map(e => ({ box: elemBox(e, L, H), ...partStyle(e), cls: start.includes(e) ? "given" : "" })) });
  $("scriptCap").textContent = `${script.elements.length} parts · the reference the network is judged against`;
  const why = beyondTraining(d);
  $("beyond").hidden = !why.length;
  $("beyond").textContent = why.length ? `beyond training: ${why.slice(0, 2).join(", ")}` : "";
  $("modelTitle").textContent = `${m.name}'s wall`;

  if (drag) {
    svgM.innerHTML = wallMarkup({ L, H, openings: d.openings, handles: true, note: "let go and it writes" });
    return;
  }
  const k = Math.min(cur.k, run.parts.length - 1);
  const drawn = stateAt(k);
  const scored = drawn.filter(x => !x.refused);
  const s = scoreElements(scored.map(x => x.e), script.elements);
  const matched = new Set(s.pairs.map(p => p[0]));
  const atEnd = run.done && k === run.parts.length - 1;
  let si = 0;
  const parts = drawn.map(x => {
    let st, cls = x.kind === "given" ? "given" : "";
    if (x.refused) { st = partStyle(x.e); cls = "refused"; }
    else { st = matched.has(si) ? partStyle(x.e) : wrongStyle(x.e); si++; }
    if (x.kind === "part" && x.i === k && !x.refused && (!run.done || !follow || play.on)) cls += " just";
    return { box: elemBox(x.e, L, H), ...st, cls };
  });
  if (atEnd) for (const j of s.missingIdx) parts.push({ box: elemBox(script.elements[j], L, H), fill: "none", stroke: "", cls: "missing" });
  const part = k >= 0 ? run.parts[k] : null;
  const fSel = selF ?? cur.f;
  const lines = [];
  if (part && ($("net").open || !follow)) {
    const sl = sourceLine(part, part.picks[fSel], L, H);
    if (sl) lines.push(sl);
    if (fSel > 0) lines.push(edgeLine(part, fSel, L, H, "picktick"));
  }
  let note = "";
  if (!modelReady && !run.parts.length) note = `loading ${m.name}…`;
  else if (!run.parts.length && run.encoding >= 0) note = "reading the wall…";
  svgM.innerHTML = wallMarkup({ L, H, openings: d.openings, parts, lines, handles: true, note });

  // captions, player
  const refusedN = run.parts.filter(p => p.refused).length;
  const pass = part ? part.pass : Math.max(0, run.passes.length - 1);
  if (run.error) $("modelCap").textContent = "stopped: " + run.error.split("\n")[0];
  else if (!run.done) $("modelCap").textContent = run.encoding >= 0 ? `reading the wall for pass ${run.encoding + 1} · ${run.parts.length} parts so far` : `writing pass ${pass + 1} · ${run.parts.length} parts`;
  else $("modelCap").textContent = `${run.parts.length - refusedN} parts in ${run.result.passes} passes · ${(run.result.ms / 1000).toFixed(1)} s${refusedN ? ` · ${refusedN} refused by the world` : ""}`;
  const total = run.parts.length * 5;
  $("scrub").max = Math.max(0, total - 1); $("scrub").value = Math.max(0, k * 5 + cur.f);
  $("scrub").disabled = !total;
  $("pos").textContent = k < 0 ? "nothing written yet" : `part ${k + 1} of ${run.parts.length} · ${FIELD_NAMES[cur.f]}`;
  $("pause").textContent = play.on ? "❚❚" : "▶";

  // tiles
  const secs = run.done ? run.result.ms / 1000 : null;
  const tile = (key, v, sub, cls = "") => `<div class="tile ${cls}"><div class="k">${key}</div><div class="v">${v}</div><div class="s">${sub}</div></div>`;
  const written = scored.length;
  $("tiles").innerHTML = tile("Script's parts rebuilt", pct(s.recall), `recall · ${s.matched} of ${script.elements.length}`, atEnd ? (s.recall > 0.99 ? "good" : (s.recall < 0.8 ? "bad" : "")) : "")
    + tile("Written parts right", written ? pct(s.precision) : "–", `precision · ${s.extra} not in the script's wall`, atEnd && written ? (s.precision > 0.99 ? "good" : (s.precision < 0.8 ? "bad" : "")) : "")
    + tile("Exact wall", s.exact ? "yes" : "no", "every part matched, none extra", atEnd ? (s.exact ? "good" : "bad") : "")
    + tile("Overlapping pairs", `${s.overlaps}`, d.reject ? `${refusedN} parts refused by the world` : "the world's check is off", atEnd ? (s.overlaps ? "bad" : "good") : "")
    + tile("Time", secs === null ? (run.parts.length ? `${(run.parts[run.parts.length - 1].ms / 1000).toFixed(1)} s` : "–") : `${secs.toFixed(1)} s`,
      run.parts.length ? `first part after ${(run.parts[0].ms / 1000).toFixed(2)} s · ${5 * run.parts.length} picks` : "in your browser");
  renderLegend();
  if ($("net").open) renderNet(part, fSel, L, H, k);
}

let legendKey = "";
function renderLegend() {
  const key = `${design.script}:${design.reject}:${design.start}`;
  if (key === legendKey) return;
  legendKey = key;
  const sw = (fill, stroke, extra = "") => `<i style="background:${fill};border-color:${stroke};${extra}"></i>`;
  const right = design.script === "block"
    ? `<span>${sw(BLOCK_FILL.block, BLOCK_STROKE.block)}${sw(BLOCK_FILL["cut block"], BLOCK_STROKE["cut block"])}${sw(BLOCK_FILL.lintel, BLOCK_STROKE.lintel)}right (block, cut block, lintel)</span>`
    : `<span>${sw(ITEM_COLOURS["2x4"], "#7a5230")}${sw(ITEM_COLOURS["2x8"], "#7a5230")}right (2x4, 2x8 header)</span>`;
  const wrongFill = wrongStyle(design.script === "block" ? [5, 0, 0, 78, 38] : [0, 0, 0, 9, 400]).fill;
  $("legend").innerHTML = right + `<span>${sw(wrongFill, WRONG_STROKE)}not in the script's wall</span>`
    + `<span>${sw("transparent", "var(--ink2)", "border-style:dashed")}in the script's wall, not written</span>`
    + (design.start !== "empty" ? `<span>${sw("transparent", "#111", "border-width:2px")}given at the start</span>` : "")
    + `<span>${sw("transparent", "var(--accent)", "border-width:2px")}just written</span>`
    + (design.reject ? `<span>${sw("transparent", "var(--bad)", "border-style:dashed")}refused by the world</span>` : "")
    + `<span>${sw("transparent", "var(--src)", "border-width:2px")}the edge a pick copied</span>`;
}

let seqKey = "";
function renderNet(part, fSel, L, H, k) {
  if (!$("netDiagram").firstChild) $("netDiagram").innerHTML = NET_SVG;
  const passIdx = follow && run.encoding >= 0 ? run.encoding : (part ? part.pass : run.passes.length - 1);
  const P = run.passes[passIdx];
  const svgT = $("svgTokens");
  svgT.setAttribute("viewBox", viewBox(L, H));
  if (P && P.tokens) {
    const lines = [];
    if (part && part.pass === passIdx) { const sl = sourceLine(part, part.picks[fSel], L, H); if (sl) lines.push(sl); }
    svgT.innerHTML = tokensMarkup({ L, H, tokens: P.tokens, cover: P.cover, showFree: $("showFree").checked, showCanvas: $("showCanvas").checked, lines });
    const c = [0, 0, 0, 0];
    P.tokens.types.forEach(t => c[t]++);
    $("tokCap").textContent = `pass ${passIdx + 1}: 1 wall + ${c[1]} opening${c[1] === 1 ? "" : "s"} + ${c[2]} part${c[2] === 1 ? "" : "s"} + ${c[3]} free-space box${c[3] === 1 ? "" : "es"} + 160 grid patches = ${P.tokens.types.length + 160} tokens`;
    if ($("tokDetails").open) {
      const names = ["wall", "opening", "part", "free"], items = (modelMeta && modelMeta.items) || ITEMS;
      const bins = r => [r[0] + 4, r[1] + 1.6, r[2] + 4, r[3] + 1.6].map(v => Math.round(v / 0.005));
      $("tokTable").innerHTML = `<table class="tok"><tr><th>#</th><th>type</th><th>item</th><th>x0</th><th>y0</th><th>x1</th><th>y1</th></tr>`
        + P.tokens.rects.slice(0, 400).map((r, i) => `<tr><td>${i}</td><td>${names[P.tokens.types[i]]}${i === 0 ? ` (brief: ${design.script})` : ""}</td><td>${P.tokens.items[i] ? items[P.tokens.items[i] - 1] : ""}</td>${bins(r).map(b => `<td>${b}</td>`).join("")}</tr>`).join("") + "</table>";
    }
  } else {
    svgT.innerHTML = wallMarkup({ L, H, openings: design.openings, note: modelReady ? "" : "loading…" });
    $("tokCap").textContent = "";
  }
  const stage = run.done && follow ? null : (run.encoding >= 0 && follow ? "encode" : (part ? "write" : null));
  highlightNet($("netDiagram"), stage || (part ? "write" : null), fSel);
  $("netCap").textContent = stage === "encode" ? "the encoder is reading the wall" : (part ? `part ${k + 1}, pick ${FIELD_NAMES[fSel]}` : "");
  $("pickCard").innerHTML = part ? pickCard(part, fSel, L, H) : `<div class="hint">The picks of each part appear here as the network writes them.</div>`;
  // the sequence of picks, pass by pass
  const key = `${run.id}:${run.parts.length}:${run.passes.filter(p => p && p.finished).length}:${k}`;
  if (key !== seqKey) {
    seqKey = key;
    const items = (modelMeta && modelMeta.items) || ITEMS;
    let html = "", lastPass = -1;
    run.parts.forEach((p, i) => {
      if (p.pass !== lastPass) { lastPass = p.pass; const Pp = run.passes[p.pass]; html += `<li class="pass">pass ${p.pass + 1} · reads ${Pp ? Pp.present : 0} parts${Pp && Pp.refused ? ` · world refused ${Pp.refused}` : ""}</li>`; }
      const cls = [i === k ? "on" : "", i > k ? "future" : "", p.refused ? "refused" : ""].join(" ");
      html += `<li data-k="${i}" class="${cls}"><span>${i + 1}</span><span>${items[p.elem[0]] === "block" ? partLabel(p.elem) : items[p.elem[0]]}</span><span>${p.elem[1]}</span><span>${p.elem[2]}</span><span>${p.elem[3]}</span><span>${p.elem[4]}</span></li>`;
    });
    if (run.done) html += `<li class="pass">STOP · a pass wrote nothing${run.result.term === "refused" ? " the world kept" : ""}</li>`;
    const seq = $("seq");
    seq.innerHTML = html;
    const on = seq.querySelector("li.on");
    if (on) { const top = on.offsetTop - seq.offsetTop; if (top < seq.scrollTop || top > seq.scrollTop + seq.clientHeight - 20) seq.scrollTop = top - seq.clientHeight / 2; }
  }
}

// ---------------------------------------------------------------- start
renderModels();
renderPresets();
renderOps();
bindControls();
if (matchMedia("(min-width: 1250px)").matches) $("net").open = true;
selectModel(modelId, { run: false });
rebuild();
syncControls();
invalidate();
changed({ immediate: true, keepPreset: true });
