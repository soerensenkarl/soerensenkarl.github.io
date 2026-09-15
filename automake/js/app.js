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
  { id: "o1", file: "o1", name: "O1", label: "O1 · frames, then blocks", scripts: ["frame", "block"] },
  { id: "n0", file: "n0", name: "N0", label: "N0 · frames only", scripts: ["frame"] },
  { id: "m0", file: "m0", name: "M0", label: "M0 · both together", scripts: ["frame", "block"] },
];
const DEFAULT_MODEL = "o1";

// what the rulers and the dataset allow (metres)
const LMIN = 1.2, LMAX = 7.9, HMIN = 2.0, HMAX = 3.15;
const SIDE = 0.2, GAP = 0.3, MINW = 0.4, MINH = 0.4, HEAD = 0.35, MINSILL = 0.3, MAXOPS = 4;

const $ = id => document.getElementById(id);
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const q = v => Math.round(v / 0.005) * 0.005;            // the 5 mm ruler the network writes on
const r3 = v => Number(v.toFixed(3));

// ---------------------------------------------------------------- the design and its rules
const door = (x, w, h) => ({ kind: "door", x, w, h, sill: 0 });
const win = (x, w, h, sill) => ({ kind: "window", x, w, h, sill });
let design = { script: "frame", L: 5.8, H: 2.7, openings: [door(0.6, 0.9, 2.1), win(2.6, 1.4, 1.2, 0.9)] };
let modelId = DEFAULT_MODEL;

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
  return d;
}

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
    return fit({ script: design.script, L, H, openings: ops });
  }
}

// ---------------------------------------------------------------- the link, so a wall can be shared
function writeHash() {
  const o = design.openings.map(o => o.kind === "door" ? `d${r3(o.x)}_${r3(o.w)}_${r3(o.h)}` : `w${r3(o.x)}_${r3(o.w)}_${r3(o.h)}_${r3(o.sill)}`).join("~");
  history.replaceState(null, "", "#" + new URLSearchParams({ n: modelId, t: design.script, L: design.L, H: design.H, o }));
}
if (location.hash.length > 2) try {
  const p = new URLSearchParams(location.hash.slice(1));
  if (MODELS.some(m => m.id === p.get("n"))) modelId = p.get("n");
  const ops = (p.get("o") || "").split("~").filter(Boolean).map(s => {
    const val = s.slice(1).split("_").map(Number);
    return s[0] === "d" ? door(val[0], val[1], val[2]) : win(val[0], val[1], val[2], val[3]);
  });
  design = { script: p.get("t") === "block" ? "block" : "frame", L: +p.get("L") || 5.8, H: +p.get("H") || 2.7, openings: ops };
} catch { /* keep the default wall */ }
if (!MODELS.find(m => m.id === modelId).scripts.includes(design.script)) design.script = "frame";
fit(design);

// ---------------------------------------------------------------- state
let sel = -1, hover = null, drag = null;
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
  status(`loading ${m.name}…`);
}

worker.onmessage = e => {
  const ev = e.data, m = MODELS.find(m => m.file === ev.file);
  if (ev.type === "progress") {
    if (m && m.id === modelId) status(`loading ${m.name} · ${(ev.loaded / 1e6).toFixed(1)} of ${(ev.total / 1e6).toFixed(1)} MB`);
    return;
  }
  if (ev.type === "ready") {
    if (!m || m.id !== modelId) return;
    ready = true;
    window.__automake.loads.push({ model: m.id, ms: performance.now() - window.__automake.loadStart, fetchMs: ev.fetchMs,
      parseMs: ev.parseMs, poolMs: ev.poolMs, helpers: ev.helpers, bytes: ev.bytes, backend: ev.backend });
    if (wantRun) startRun(); else status(`${m.name} ready`);
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
  worker.postMessage({ type: "run", id: runId, wall: scene.wall, ops: scene.ops, start: [], brief: BRIEFS[design.script], reject: true });
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
const BLUE = "#1d5fbf";                                  // the outline on a part the moment the network writes it
// one neutral timber, one neutral grey: the only fills on the page
const FILL = { block: ["#dedcd6", "#a6a39b"], lintel: ["#d5d5d2", "#9b9b96"], timber: ["#e1d8c9", "#b0a181"] };

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

function paint() {
  measure();
  const { L, H, openings } = design;
  ctx.setTransform(v.dpr, 0, 0, v.dpr, 0, 0);
  ctx.clearRect(0, 0, v.cw, v.ch);

  for (const e of parts) {                               // what the network has written so far
    const [fill, stroke] = FILL[ITEMS[e[0]] === "block" ? "block" : ITEMS[e[0]] === "lintel" ? "lintel" : "timber"];
    const b = ebox(e), x = px(b[0]), y = py(b[3]), w = Math.max(.7, v.s * (b[2] - b[0])), h = Math.max(.7, v.s * (b[3] - b[1]));
    ctx.fillStyle = fill; ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.strokeRect(x + .5, y + .5, Math.max(0, w - 1), Math.max(0, h - 1));
  }
  if (hot) { ctx.save(); ctx.globalAlpha = hot.a; frame(ebox(hot.e), [], BLUE, 1.6); ctx.restore(); }   // the part just written

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
  for (let i = openings.length - 1; i >= 0; i--) {
    const b = obox(openings[i]);
    if (sel === i && Math.abs(p.x - px(b[2]) - 15) <= 11 && Math.abs(p.y - py(b[3]) + 15) <= 11) return `o:${i}:x`;
    for (const [cx, cy, k] of [[b[0], b[1], "lb"], [b[2], b[1], "rb"], [b[0], b[3], "lt"], [b[2], b[3], "rt"]])
      if (near(cx, cy, 12)) return `o:${i}:${k}`;
  }
  if (near(L, H, 14)) return "w:c";
  if (near(L, 0, 14)) return "w:r";
  if (near(0, H, 14)) return "w:t";
  for (let i = openings.length - 1; i >= 0; i--) {
    const b = obox(openings[i]);
    if (p.x >= px(b[0]) && p.x <= px(b[2]) && p.y <= py(b[1]) && p.y >= py(b[3])) return `o:${i}:move`;
  }
  return null;
}
const CURSOR = { "w:r": "ew-resize", "w:t": "ns-resize", "w:c": "nwse-resize", lb: "nesw-resize", rb: "nwse-resize",
  lt: "nwse-resize", rt: "nesw-resize", move: "move", x: "pointer" };
const at = e => { const r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };

canvas.addEventListener("pointerdown", e => {
  const p = at(e), h = hit(p);
  canvas.focus({ preventScroll: true });
  if (h && h.startsWith("o:")) {
    const [, i, edge] = h.split(":");
    if (edge === "x") { design.openings.splice(+i, 1); sel = -1; fit(design); changed(); return; }
    sel = +i;
  } else if (!h) sel = -1;
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
    if (h !== hover) { hover = h; canvas.style.cursor = h ? CURSOR[h] || CURSOR[h.split(":")[2]] || "default" : "default"; paint(); }
    return;
  }
  e.preventDefault();
  const dx = mx(p.x) - mx(drag.p.x), dy = my(p.y) - my(drag.p.y);
  const O = drag.d0, d = design, h = drag.h.split(":"), edge = h[h.length - 1];
  if (h[0] === "w") {
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
      if (edge[0] === "l") { const x = q(clamp(o0.x + dx, lo, o0.x + o0.w - MINW)); o.w = q(o0.w + (o0.x - x)); o.x = x; }
      else o.w = q(clamp(o0.w + dx, MINW, hi - o.x));
      if (edge[1] === "t") o.h = q(clamp(o0.h + dy, o.kind === "door" ? 1.2 : MINH, top - sillOf(o)));
      else if (o.kind !== "door") { const s = q(clamp(o0.sill + dy, MINSILL, o0.sill + o0.h - MINH)); o.h = q(o0.h - (s - o0.sill)); o.sill = s; }
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
  if ((e.key === "Delete" || e.key === "Backspace") && design.openings[sel]) {
    e.preventDefault(); design.openings.splice(sel, 1); sel = -1; fit(design); changed();
  } else if (e.key === "Escape") { sel = -1; paint(); }
});

// ---------------------------------------------------------------- the two controls
function sync() {
  const m = MODELS.find(m => m.id === modelId);
  $("caption").textContent = `${design.script === "block" ? "Concrete blocks" : "Timber frame"} · ${design.L.toFixed(2)} × ${design.H.toFixed(2)} m`;
  $("model").value = modelId;
  $("script").checked = design.script === "block";
  $("script").disabled = m.scripts.length < 2;
  $("sw").className = `sw ${design.script}${m.scripts.length < 2 ? " off" : ""}`;
  $("addDoor").disabled = $("addWindow").disabled = design.openings.length >= MAXOPS || !freeSpan(design);
}
$("model").innerHTML = MODELS.map(m => `<option value="${m.id}">${m.label}</option>`).join("");
$("model").addEventListener("change", e => {
  modelId = e.target.value;
  const m = MODELS.find(m => m.id === modelId);
  if (!m.scripts.includes(design.script)) design.script = m.scripts[0];
  runId++;
  worker.postMessage({ type: "cancel" });
  parts = []; pending = []; hot = null; wantRun = true;
  writeHash(); sync(); paint(); loadModel();
});
$("script").addEventListener("change", e => { design.script = e.target.checked ? "block" : "frame"; changed(); });
$("addDoor").addEventListener("click", () => addOpening("door"));
$("addWindow").addEventListener("click", () => addOpening("window"));
$("random").addEventListener("click", () => { design = randomDesign(); sel = -1; changed(); });
addEventListener("resize", paint);
if (window.ResizeObserver) new ResizeObserver(paint).observe(canvas);

sync();
writeHash();
paint();                                                 // the wall is on screen and draggable before the weights arrive
loadModel();
