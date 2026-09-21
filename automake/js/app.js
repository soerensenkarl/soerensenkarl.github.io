// Automake, the drawing tool. Drag the wall and its openings on a white canvas; a neural network fills the wall in, part by
// part, in this browser. Everything the page does is here: the design and its rules, the drawing, the dragging and the model
// thread. The wall and its openings are drawn the way a CAD window draws a selection - a dotted outline with a cross on
// each corner, five-sided when the wall is raked - and the parts the network writes are the only solid things on the page.
//
// The engine is untouched: wall.js (scenes, encoding, the world's refusal), worker.js, writer.js, model.js, wasm.js.
import { api, BRIEFS, ITEMS, makeScene, setItems, skinOf, TIMBER } from "./wall.js";

const qs = new URLSearchParams(location.search);
const FORMAT = qs.get("format") || "f16";

// the networks on offer: model/<file>.<format>.json/.bin, written by scripts/export_web_model.py
const MODELS = [
  { id: "o6", file: "o6", name: "O6", title: "Frames, then blocks", scripts: ["frame", "block"],
    blurb: "The framing network, then 50 minutes on concrete-block walls. It kept its framing by rehearsing framed walls it had written itself and the world had accepted: no framing script or framing data in that stage." },
  // `loads: true` - the network reads the loads on the wall's top edge (sequence.py TYPES index 4). Only these
  // networks are given load tokens, and only for them does the page draw, drag or link a load. (The network reads a
  // line load too; the page offers only the point-load array, which is what a wall is designed for.)
  // To put a newer checkpoint behind this artifact, export it (scripts/export_web_model.py --name x12), change this
  // entry's `id` and `file` with the `WALLS` and `ABOUT` keys, and pick its wall again with web/tests/pick_wall.mjs.
  // `heavy: true` - the network was trained with one load of its own four times as heavy (dataset "heavy"),
  // so the page draws it, drags it and links it (`k=`). The others are given POINT_KN loads only.
  { id: "x11", file: "x11", name: "X11", title: "Designs with the forces", scripts: ["frame"], loads: true, heavy: true },
  { id: "n0", file: "n0", name: "N0", label: "N0 · frames only", scripts: ["frame"] },
  { id: "m0", file: "m0", name: "M0", label: "M0 · both together", scripts: ["frame", "block"] },
  // It reads a stock as well: it was trained with one on every wall, and given no inventory tokens it is off its own
  // distribution and writes a quarter of a wall (recall 0.38 against 0.83 with them). This artifact is about the
  // outline, so the page hands it the script's own two sections and offers no control over them - `inventory` is
  // what the network is given, `picker` is whether the page lets anyone change it.
  { id: "g2", file: "g2", name: "G2", title: "Neural net that frames a wall of any outline", scripts: ["frame"],
    segment: true, inventory: true, rake: true },
  // `template: true` - the network reads the pattern cell the wall is asked to follow (sequence.py TYPES index 5), so
  // the page draws the 2 x 4 window beside the wall and the user fills it in. Weights land with round T0.
  { id: "t0", file: "t0", name: "T0", title: "Neural net that follows a bond you draw", scripts: ["block"],
    segment: true, template: true },
  // `inventory: true` - the network reads the sections the design may be built from (TYPES index 6) and may write
  // nothing else (MVPEditor.item_mask); `picker: true` puts the catalogue on the page as a row of toggles, which is
  // this artifact's whole point. `view3d: true` gives it the 2D/3D switch, where a deep section stands visibly deep.
  // No `rake`: I0_stock was trained on rectangles alone, and on a raked outline it writes a third of the wall. The
  // page therefore offers it no rake at all - a control that cannot act is absent - until I1 has learnt one.
  { id: "i0", file: "i0", name: "I0", title: "Neural net that adapts the frame to your inventory", scripts: ["frame"],
    segment: true, inventory: true, picker: true, view3d: true },
];
const DEFAULT_MODEL = "o6";

// what the rulers and the dataset allow (metres)
const LMIN = 1.2, LMAX = 7.9, HMIN = 2.0, HMAX = 3.15;
const SIDE = 0.2, GAP = 0.3, MINW = 0.4, MINH = 0.4, HEAD = 0.35, MINSILL = 0.3, MAXOPS = 4;
// the load on the top edge (automake/mvp/dataset.py: LOAD_CLEAR_END, LOAD_SPACING, load_grid). One distributed load:
// point loads of 10 kN standing every `sp` metres from `off`, LEND clear of each end of the wall, over an opening if
// they fall there. The standard is the data's own 600 mm line.
const LEND = 0.2, SPDEF = 0.6, OFFDEF = 0.3, SPMIN = 0.3, SPMAX = 2.4, MAXLOADS = 16;
// and one load of its own, four times as heavy (dataset.HEAVY_KN), which a `heavy: true` network was trained with:
// it stands anywhere along the top edge, openings included, and is dragged there. Its tail starts higher than the
// array's line, so a heavier load is a bigger arrow, and it says its own weight.
const HEAVY_KN = 40, HARROW = 0.46;
const SPACINGS = [0.4, 0.6, 0.6, 0.6, 0.9, 1.2];         // what "random" draws the interval from
// the magnet: 15 mm either side of a value the trade builds to, and nothing outside that window. The interval holds at
// the 600 mm line and at 400; the offset holds where an arrow stands on a stud of the script's 600 mm grid from the
// left end of the wall (0 mod 0.6) and where it stands exactly halfway between two of them (0.3 mod 0.6).
const SNAP = 0.015, STUD = 0.6, SPSNAP = [0.6, 0.4], EXACT = 5e-4;
// the raked top a segment network reads (automake/mvp/dataset.py gable_spec): the pitch, where along the wall the
// ridge may stand, and the tallest apex the y ruler holds. With a rake, `H` is the eaves and the ridge stands above it.
// The ridge may stand anywhere along the wall, its own ends included: at an end the outline is a quadrilateral with
// one sloping top edge - a single slope - and anywhere else a pentagon (gable_framer.gable_poly drops the corner that
// adds nothing). A single slope is the shallower band the data draws it in (dataset.MONO_PITCH).
const PMIN = 10, PMAX = 40, MONOMIN = 5, MONOMAX = 30, GMAXH = 4.6, RISE = 0.01, MSNAP = 0.02;
// the pattern cell a template network reads (automake/mvp/template.py): a window two blocks wide and four courses
// tall, on the half-module lattice. It stands beside the wall, drawn at `cs()` times the wall's own scale so a cell is
// big enough to hit - larger where the canvas is narrow, at the same breakpoint the page's own layout turns on. The
// blocks keep their true proportions (390 x 190 with 10 mm joints) whatever that scale is.
const CELLX = 8.42, CELLY = 0.0;
const cs = () => (canvas.clientWidth < 620 ? 3 : 1.5);
const RUNNING = [[0, 0, 2], [0, 2, 2], [1, 1, 2], [1, 3, 2], [2, 0, 2], [2, 2, 2], [3, 1, 2], [3, 3, 2]];  // Karl's drawing 1
// the timber catalogue as the designer picks it (automake/world/materials.py TIMBER, dataset.sample_inventory): one
// toggle per section, each section available both ways up, which is the pair of items the sampler hands out
const CATALOGUE = TIMBER.map(([depth, flat, edge]) => ({ key: `45x${Math.round(depth * 1000)}`,
  label: `45×${Math.round(depth * 1000)}`, items: [flat, edge] }));
const INVDEF = ["45x95", "45x195"];                      // the sections the framing script itself reaches for

// what each artifact opens with, and what it says about itself
const WALLS = {
  n0: () => ({ script: "frame", L: 5.18, H: 2.63, openings: [door(0.535, 0.935, 2.08), win(2.695, 1.29, 1.0, 0.85)] }),
  // Karl's own block wall, kept unchanged when O4 gave way to O6 on 2026-09-17: the artifact is a better network
  // behind the same wall, so the two can be told apart by what they lay on it and by nothing else.
  o6: () => ({ script: "block", L: 5.23, H: 2.69, openings: [door(0.965, 0.9, 2.1), win(2.65, 0.89, 1.2, 0.59)] }),
  // picked by trying: web/tests/pick_wall.mjs ranked 16 candidates and this one shows the answer to the loads most
  // plainly - every stud within 6 mm of an arrow, 8 mm at worst, against 266 mm for the script's load-blind 600 mm
  // grid on the same wall; both openings headed, jacked and silled, nothing hanging, all five arrows over an opening
  // carried on a header, 25 parts
  x11: () => ({ script: "frame", L: 5.4, H: 2.7, openings: [door(0.6, 0.9, 2.05), win(2.7, 1.2, 1.1, 0.9)],
                off: 0.3, sp: 0.6, kx: 2.4 }),   // the heavy load starts on solid wall, midway between two arrows
  // a gable to open on; drag the ridge to either end of the wall and the same network frames a single slope
  g2: () => ({ script: "frame", L: 4.6, H: 2.45, pitch: 26, ridge: 2.1, openings: [door(0.6, 0.9, 2.05), win(2.5, 1.2, 1.0, 0.9)] }),
  // the block wall Karl picked for the second artifact, with the window opening on the running bond it is laid in
  t0: () => ({ script: "block", L: 5.23, H: 2.69, openings: [door(0.965, 0.9, 2.1), win(2.65, 0.89, 1.2, 0.59)] }),
  i0: () => ({ script: "frame", L: 5.4, H: 2.7, openings: [door(0.6, 0.9, 2.05), win(2.7, 1.2, 1.1, 0.9)] }),
};
const ABOUT = {
  t0: "You draw a bond in the window on the right - two blocks wide, four courses tall, any fill you like - and the network lays the wall to it: it repeats your pattern across the wall, cuts it where it meets the ends and the openings, puts a lintel over every opening and leaves out what could not stand. What you leave empty stays empty, so a sparse drawing is a perforated screen wall you can see through. Nothing in it knows what running bond is: it is one drawing among the rest, and the network was trained across the whole domain of drawings, the common bonds weighted up. It reads your drawing as tokens, and each part's place in the drawing's own 800 mm period as a coordinate, the way a mason sets out the bond along a wall before laying. Held out on 64 walls and drawings it had not seen: 78% of the parts of the wall a bricklaying script lays to the same drawing, 75% of its own parts right, 28% of the walls exact part for part. Known gaps: cut pieces beside a window are sometimes left out, a block is sometimes laid with nothing under it, and an unusual drawing on a large wall can lose its rhythm. Runs entirely in your browser; nothing is sent anywhere.",
  g2: "It frames the wall you outline - gable, off-centre ridge or single slope - with a plate along each slope and the studs cut to it. An 8.9-million-parameter encoder-decoder transformer trained from scratch on 40,000 walls whose top edge is not level. It writes each part as the two ends of its own edge and says which of its ends are cut; the world saws them flush, and does it the moment the thing they are cut to arrives. Held out on 40 raked walls it had not seen: a plate on every rake on 98% of them, 91% of the script's members, and one wall in five written exactly as the script would. It runs entirely in your browser; nothing is sent anywhere.",
  i0: "The designer says what timber is in the yard and the wall is framed from that and nothing else. The network reads the available sections as tokens and may write no other: an item is not a name to it but its two numbers, so a section it never saw in training is read like one it did. Turn the wall to 3D and a 45 x 245 stands visibly deeper through the wall than a 45 x 95, which is the whole of what a stock changes and the drawing cannot show. Held out on 8 walls it had not seen: 98% of the script's parts, 99% of its own parts right, 5 of the 8 walls exact. The known gap is the built-up header on a shallow stock, where nothing on its own is deep enough to span. Runs entirely in your browser; nothing is sent anywhere.",
  n0: "An 8.8-million-parameter encoder-decoder transformer that has learned light timber framing by imitating a simple framing script, judged only by geometry. It reads the wall, its openings and the parts already there as boxes and writes each part as an item and four edges on a 5 mm ruler, one part at a time, with no framing rules built in. It runs entirely in your browser on WebAssembly; nothing is sent anywhere. Trained on 40,000 walls 2.4-6 m long; on walls it has not seen it writes 88% of the script's parts with 91% of its parts right.",
  x11: "The framing network after twelve rounds of learning from the world's physics alone: a search that only knows 'slide a box, copy one, cut one, take one away' improved walls under load by the strain energy the world measures, and the network learned to reproduce them. It reads where the loads stand and designs with them: it puts a stud under each of them – 6 mm away on the wall this page opens on, where the script's 600 mm grid leaves 266 – frames every opening on every side, carries a load standing over a door or a window on a header, and leaves nothing hanging, and was never told what a stud, a header or a jamb is. A stiffer wall is the aim; a stud under a load is one of the ways it gets there. Drag the 40 kN load along the top edge and watch it answer that too: a column of studs packs together under it, and over a door or a window the header deepens to carry it. Runs entirely in your browser; nothing is sent anywhere.",
  o6: "The same network after it had learned timber framing, then trained for 50 minutes on concrete-block walls. It kept its framing by rehearsing framed walls it had written itself and the world had accepted, with no framing script or framing data in that stage. Its framing came out better than before (89% of the script's parts, 93% of its own right, and 79% of training-domain walls exactly), and its block walls get about 8 in 10 blocks right - better than the network trained on both kinds of wall together. What it still gets wrong is finishing: on the longest walls it can stop before the wall is full. It runs entirely in your browser on WebAssembly; nothing is sent anywhere.",
};

const $ = id => document.getElementById(id);
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);
const q = v => Math.round(v / 0.005) * 0.005;            // the 5 mm ruler the network writes on
const r3 = v => Number(v.toFixed(3));
// the nearest of `targets` within the window, or the value untouched: a number outside the window passes through freely
const magnet = (v, targets, win = SNAP) => {
  let best = null;
  for (const t of targets) if (Math.abs(v - t) <= win && (best === null || Math.abs(v - t) < Math.abs(v - best))) best = t;
  return best === null ? q(v) : best;
};
const offTargets = v => [Math.round(v / STUD) * STUD, Math.round((v - STUD / 2) / STUD) * STUD + STUD / 2];
const modDist = (v, phase) => { const r = (((v - phase) % STUD) + STUD) % STUD; return Math.min(r, STUD - r); };
// whether the array as it ended up is held by the magnet - asked of the laid-out design, so the mark never lies
const offHeld = d => modDist(d.off, 0) < EXACT || modDist(d.off, STUD / 2) < EXACT;
const spHeld = d => SPSNAP.some(t => Math.abs(d.sp - t) < EXACT);

// the token format the chosen network speaks, and the geometry that goes with it (wall.js api): "rect" for every
// network published before 2026-09-16, "segment" for the ones after it, which are the ones that read a raked top.
const SEGA = api("segment");                             // the gable's own arithmetic, whichever format is in use
const isSegment = () => !!MODELS.find(m => m.id === modelId).segment;
const fmt = () => api(isSegment() ? "segment" : "rect");
const rad = d => d * Math.PI / 180, deg = r => r * 180 / Math.PI;
const snap10 = v => Math.round(v / RISE) * RISE;         // the 10 mm step the data puts a ridge and a rise on
const ridgeRun = d => Math.max(d.ridge, d.L - d.ridge);  // the longer half of the wall: the one the pitch is measured on
const isMono = d => d.ridge < 1e-9 || d.ridge > d.L - 1e-9;   // the ridge on an end corner: one sloping top edge
const ridgeH = d => SEGA.ridgeY(d.L, d.H, d.pitch, d.ridge);
const wallTop = d => d.pitch ? ridgeH(d) : d.H;          // the wall's full height: its ridge, or its flat top

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
  return fitInv(fitCell(layLoads(rake(d))));
}

// the raked top, on a network that reads one (dataset.gable_spec): the ridge in the middle of the wall, the pitch
// between PMIN and PMAX, and never so steep that the apex leaves the y ruler. The rise is put on a 10 mm step and the
// pitch follows from it, as the data does it, so the wall's own corners sit on the ruler the network writes on.
function rake(d) {
  if (!readsRake() || !(d.pitch > 0)) { d.pitch = 0; d.ridge = null; return d; }
  // the ridge holds at either end of the wall and at mid-span, and is free between them
  d.ridge = r3(clamp(magnet(snap10(d.ridge == null ? d.L / 2 : d.ridge), [0, d.L / 2, d.L], MSNAP), 0, d.L));
  const mono = isMono(d), lo = mono ? MONOMIN : PMIN, up = mono ? MONOMAX : PMAX;
  const run = ridgeRun(d), hi = Math.max(lo, Math.min(up, deg(Math.atan((GMAXH - d.H) / run))));
  d.pitch = r3(deg(Math.atan(snap10(run * Math.tan(rad(clamp(d.pitch, lo, hi)))) / run)));
  return d;
}

// ---------------------------------------------------------------- the load array on the top edge
// Only a network built with them reads loads; for the others the page has none at all. The load is one array, drawn
// the way a distributed load is drawn - a line with arrows hanging from it - and set by the two numbers that describe
// it: `off`, where the first arrow stands, and `sp`, the interval between arrows (dataset.load_grid).
const readsLoads = () => !!MODELS.find(m => m.id === modelId).loads;
const readsHeavy = () => !!MODELS.find(m => m.id === modelId).heavy;

// the array laid out: an arrow at off + k*sp for every k that stands LEND clear of both ends of the wall. `off` is
// wrapped into [LEND, LEND + sp), so an arrow always stands within one interval of the left end; a wall too short for
// even that keeps its first arrow at LEND, so there is always one arrow to take hold of.
function layLoads(d) {
  d.kx = readsHeavy() && d.kx !== undefined ? r3(clamp(q(d.kx), LEND, Math.max(LEND, d.L - LEND))) : undefined;
  if (!readsLoads()) { d.loads = []; return d; }
  d.sp = r3(clamp(q(d.sp === undefined ? SPDEF : d.sp), SPMIN, SPMAX));
  let off = q(d.off === undefined ? OFFDEF : d.off);
  off = r3(LEND + (((off - LEND) % d.sp) + d.sp) % d.sp);
  d.off = off > d.L - LEND + 1e-9 ? LEND : off;
  d.loads = [];
  for (let x = d.off; x <= d.L - LEND + 1e-9 && d.loads.length < MAXLOADS; x = q(x + d.sp)) d.loads.push(r3(x));
  return d;
}

// the loads as the network reads them: one segment [x0, x1] each in the wall frame, a point load being x0 = x1
// (dataset.loads_in_wall_frame), exactly as sequence.load_rects stacks them
const loadSegs = d => !readsLoads() ? []
  : [...d.loads.map(v => [v - d.L / 2, v - d.L / 2]),
     ...(d.kx === undefined ? [] : [[d.kx - d.L / 2, d.kx - d.L / 2, HEAVY_KN]])];   // heavy last, and it says its kN

// ---------------------------------------------------------------- the pattern cell, and the inventory
// Karl: the most general way to say interlocking is not a rule about offsets but a picture - here is a window two
// blocks wide and four courses tall, I fill it in, and the wall repeats it. A block is [course, first step, steps]:
// one step is a half block, two a whole one, and a block that runs off the right edge wraps round to the left, which
// is what the tiling does by itself. Any fill is valid, sparse or dense; what a course may not have is two blocks on
// the same step, since the cell wraps and the wall would then be asked for two blocks in one place (template._free).
const readsTemplate = () => !!MODELS.find(m => m.id === modelId).template;
const readsInv = () => !!MODELS.find(m => m.id === modelId).inventory;
// whether the page lets anyone change that stock; a network may read one without the artifact being about it
const showsPicker = () => !!MODELS.find(m => m.id === modelId).picker;
// whether this network was taught a top edge that is not level; one that was not is given no rake to drag
const readsRake = () => !!MODELS.find(m => m.id === modelId).rake;
const TPL = SEGA.TEMPLATE;
const steps = ([c, u, n]) => Array.from({ length: n }, (_, i) => (u + i) % TPL.COURSES);

function fitCell(d) {
  if (!readsTemplate()) { d.cell = null; return d; }
  const out = [];
  for (const b of (d.cell == null ? RUNNING : d.cell)) {
    const one = [clamp(Math.round(b[0]), 0, TPL.COURSES - 1), ((Math.round(b[1]) % 4) + 4) % 4, clamp(Math.round(b[2]), 1, 2)];
    const taken = new Set(out.filter(o => o[0] === one[0]).flatMap(steps));
    if (!steps(one).some(s => taken.has(s))) out.push(one);
  }
  d.cell = out.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  return d;
}

// the cell as the network reads it: (x0, x1, course) in metres, which is exactly template.Cell.blocks
const cellBlocks = d => (d.cell || []).map(([c, u, n]) => [u * TPL.STEP, u * TPL.STEP + (n > 1 ? TPL.FULL : TPL.HALF), c]);

// The sections the design may be built from: at least one, in the catalogue's order, each of them available both ways
// up. The network is given the item names; the world would refuse anything else in any case.
function fitInv(d) {
  if (!readsInv()) { d.inv = null; return d; }
  const want = d.inv == null ? INVDEF : d.inv;
  d.inv = CATALOGUE.filter(s => want.includes(s.key)).map(s => s.key);
  if (!d.inv.length) d.inv = INVDEF.slice();
  return d;
}
const invItems = d => (d.inv ? d.inv.flatMap(k => CATALOGUE.find(s => s.key === k).items) : null);

// a cell from the whole domain, as template.random_cell draws one: the running bond now and then, otherwise any fill
// of the window - whole and half blocks, sparse, dense or ugly, with a gap in a course as often as not
function randomCell() {
  if (Math.random() < 0.3) return RUNNING.map(b => b.slice());
  const out = [];
  for (let c = 0; c < TPL.COURSES; c++) {
    const taken = new Set();
    let u = Math.floor(Math.random() * 4);
    for (let k = 0; k < 4; k++) {
      const n = Math.random() < 0.65 ? 2 : 1;
      if (steps([c, u, n]).some(s => taken.has(s))) break;
      if (Math.random() < 0.75) { out.push([c, u % 4, n]); steps([c, u, n]).forEach(s => taken.add(s)); }
      else taken.add(u % 4);
      u += n + Math.floor(Math.random() * 2);
    }
  }
  return out;
}

// an inventory as dataset.sample_inventory draws a rich one: three to five sections, at least one deep enough to be a
// header on its own (HEADER_DEPTH, 145 mm)
function randomInv() {
  for (;;) {
    const n = 3 + Math.floor(Math.random() * 3);
    const pick = CATALOGUE.map(s => s.key).sort(() => Math.random() - 0.5).slice(0, n);
    if (pick.some(k => +k.split("x")[1] >= 145)) return CATALOGUE.filter(s => pick.includes(s.key)).map(s => s.key);
  }
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
    // the 600 mm line of the data most of the time, now and then a closer or a wider one, from a random offset
    const sp = SPACINGS[Math.floor(Math.random() * SPACINGS.length)];
    // half of a segment network's walls are gables (dataset.train_gable_spec), pitch 15 to 35 degrees
    // half of a raked network's walls are raked; three in ten of those are a single slope, as the data draws them
    const mono = Math.random() < 0.3;
    const rk = readsRake() && design.script !== "block" && Math.random() < 0.5
      ? { pitch: mono ? U(5, 30) : U(15, 35), ridge: mono ? (Math.random() < 0.5 ? 0 : L) : L * U(0.2, 0.8) } : {};
    return fit({ script: design.script, L, H, openings: ops, sp, off: q(U(LEND, LEND + sp)),
                kx: readsHeavy() ? q(U(LEND, L - LEND)) : undefined, ...rk,
      ...(readsTemplate() ? { cell: randomCell() } : {}), ...(readsInv() ? { inv: randomInv() } : {}) });
  }
}

// ---------------------------------------------------------------- the link, so a wall can be shared
function writeHash() {
  const o = design.openings.map(o => o.kind === "door" ? `d${r3(o.x)}_${r3(o.w)}_${r3(o.h)}` : `w${r3(o.x)}_${r3(o.w)}_${r3(o.h)}_${r3(o.sill)}`).join("~");
  const p = new URLSearchParams({ n: modelId, t: design.script, L: design.L, H: design.H, o });
  if (readsLoads()) p.set("l", `${r3(design.off)}_${r3(design.sp)}`);   // the load array: first arrow, then interval, in metres
  if (readsHeavy() && design.kx !== undefined) p.set("k", r3(design.kx));   // where the heavy load stands; no `k`, no heavy load
  if (design.pitch) p.set("r", `${r3(design.pitch)},${r3(design.ridge)}`);   // the rake: the pitch in degrees, then the ridge
  // the pattern cell: one block per `~`, as `<course>:<first step>-<last step>` on the half-module lattice, so a block
  // that ends past step 4 is one that wraps round the cell's left edge. An empty `c` is a wall asked to follow nothing.
  if (readsTemplate()) p.set("c", design.cell.map(([c, u, n]) => `${c}:${u}-${u + n}`).join("~"));
  if (showsPicker()) p.set("v", design.inv.join("_"));   // the sections the design may be built from
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
  if (len !== null || p.has("o")) design = { script: p.get("t") === "block" ? "block" : "frame", L: +len || 5.8, H: +p.get("H") || 2.7, openings: ops };   // a link with only a network keeps that artifact's wall
  else if (p.has("t")) design.script = p.get("t") === "block" ? "block" : "frame";
  if (p.has("l")) {                                      // `l=<offset>_<interval>`; an older link listed every load,
    const val = (p.get("l") || "").split("_").map(Number).filter(v => isFinite(v));   // so read the array it lay on
    if (val.length > 2) { design.off = val[0]; design.sp = val[1] - val[0]; }
    else if (val.length) { design.off = val[0]; if (val.length > 1) design.sp = val[1]; }
  }
  if (p.has("k")) { const v = +p.get("k"); design.kx = isFinite(v) ? v : undefined; }   // `k=<x>`; no `k`, no heavy load
  else if (p.has("l")) design.kx = undefined;            // a link that sets the loads and no heavy one has none
  if (p.has("c")) design.cell = (p.get("c") || "").split("~").filter(Boolean).map(s => {
    const [c, span] = s.split(":"), [a, b] = (span || "").split("-").map(Number);
    return [+c, a, b - a];
  });
  if (p.has("v")) design.inv = (p.get("v") || "").split("_").filter(Boolean);
  if (p.has("r")) {                                      // `r=<pitch>,<ridge>`; a flat top has no `r` at all
    const [pitch, ridge] = (p.get("r") || "").split(",").map(Number);
    design.pitch = isFinite(pitch) ? pitch : 0;
    if (isFinite(ridge)) design.ridge = ridge;
  }
} catch { /* keep the default wall */ }
if (!MODELS.find(m => m.id === modelId).scripts.includes(design.script)) design.script = "frame";
fit(design);

// ---------------------------------------------------------------- state
let sel = -1, hover = null, drag = null, held = null;    // `held`: "sp" or "off" while the magnet holds this drag
let mode3 = false, three = null;                         // the 3D view, built the first time it is asked for
let scene = null, parts = [], pending = [], hot = null;
const written = () => parts.concat(pending);             // every part the world has kept: drawn, or waiting its turn
let runId = 1, timer = null, ready = false, wantRun = true;
window.__automake = { loads: [], runs: [] };

// ---------------------------------------------------------------- the model thread
const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });

let asking = null;                                       // the network the page has asked the thread for

function loadModel() {
  ready = false;
  const m = MODELS.find(m => m.id === modelId);
  asking = m;
  window.__automake.loadStart = performance.now();
  worker.postMessage({ type: "load", url: new URL("../model/", import.meta.url).href, file: m.file, format: FORMAT,
    backend: qs.get("backend"), threads: qs.get("threads") });
  status("loading neural net…");
}

// a network whose weights are not published yet: its manifest does not fetch, so it leaves the list and the page falls
// back to the one it opens with. An artifact can then be listed here before the round behind it has trained.
function dropModel(m, why) {
  console.warn(`automake: ${m.name} is not published yet (${why}); leaving it out`);
  MODELS.splice(MODELS.indexOf(m), 1);
  asking = null;
  const back = MODELS.find(x => x.id === DEFAULT_MODEL) || MODELS[0];
  if (!back) { status("no network is published here yet"); return; }
  modelId = back.id;
  design = WALLS[modelId] ? WALLS[modelId]() : design;
  if (!back.scripts.includes(design.script)) design.script = "frame";
  sel = -1;
  fit(design);
  sync();
  writeHash();
  paint();
  loadModel();
}

worker.onmessage = e => {
  const ev = e.data, m = MODELS.find(m => m.file === ev.file);
  if (ev.type === "error" && ev.id === undefined) { if (asking) dropModel(asking, ev.message); return; }   // no weights there
  if (ev.type === "progress") {
    if (m && m.id === modelId) status(`loading neural net · ${(ev.loaded / 1e6).toFixed(1)} of ${(ev.total / 1e6).toFixed(1)} MB`);
    if (m && m.id === modelId) { $("load").hidden = false; $("load").firstElementChild.style.width = `${Math.min(100, 100 * ev.loaded / Math.max(1, ev.total))}%`; }
    return;
  }
  if (ev.type === "ready") {
    if (!m || m.id !== modelId) return;
    ready = true; asking = null;
    // The page draws and measures the parts too, and a worker module is its own copy of wall.js: the vocabulary the
    // thread was given has to be given to this side as well, or every part is read as the wrong item.
    setItems(ev.manifest.items, ev.manifest.sections);
    window.__automake.loads.push({ model: m.id, ms: performance.now() - window.__automake.loadStart, fetchMs: ev.fetchMs,
      parseMs: ev.parseMs, poolMs: ev.poolMs, helpers: ev.helpers, bytes: ev.bytes, backend: ev.backend });
    $("load").hidden = true;
    if (wantRun) startRun(); else status("ready");
    return;
  }
  if (ev.id !== runId) return;
  if (ev.type === "error") status("something went wrong: " + String(ev.message).split("\n")[0]);
  else if (ev.type === "part") {
    const F = fmt(), cur = written();                    // the world refuses a part that overlaps or leaves the wall
    const { kept } = F.segment ? F.refuseOverlaps(scene.wall, scene.ops, [ev.elem], undefined, scene.poly, cur)
      : F.refuseOverlaps(scene.wall, [...scene.ops, ...cur.map(F.elementRect)], [ev.elem]);
    if (!kept.length) return;
    pending.push(ev.elem);
    tick();
  } else if (ev.type === "done") {
    window.__automake.runs.push({ model: modelId, script: design.script, parts: ev.parts, passes: ev.passes, term: ev.term,
      ms: ev.ms, firstPartMs: ev.firstPartMs, backend: ev.backend });
    status(`${written().length} parts in ${(ev.ms / 1000).toFixed(1)} s`);
  }
};

function startRun() {
  if (!ready) { wantRun = true; return; }
  wantRun = false;
  const F = fmt();
  const sc = makeScene({ L: design.L, H: design.H, pitch: design.pitch, ridge: design.ridge,
    openings: design.openings.map(o => [o.x, o.w, sillOf(o), o.h]) });
  const skin = F.segment ? F.skinFeatures(sc) : skinOf(sc);   // the wall as the network reads it: a polygon, or a box
  scene = { ...F.skinRects(skin), skin, poly: F.segment ? F.skinPoly(skin) : null };
  parts = []; pending = []; hot = null;
  worker.postMessage({ type: "run", id: runId, wall: scene.wall, ops: scene.ops, skin, poly: scene.poly, start: [],
    brief: BRIEFS[design.script], reject: true, loads: loadSegs(design), cell: cellBlocks(design), inv: invItems(design) });
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
// A raked wall stands higher than a flat one, so the rake has a taller view of its own; it changes when the rake
// comes and goes, never under the hand.
// The pattern cell stands to the right of the longest wall the page can draw, so the view is that much wider whenever
// a network reads one; like the rake's taller view it changes only when the artifact does, never under the hand.
const WX0 = -0.28, WX1 = 8.28, WY0 = -0.3, WY1 = 3.46, WY1R = 4.9;
const wx1 = () => readsTemplate() ? CELLX + TPL.WIDTH * cs() + 0.2 : WX1;
const wy1 = () => design.pitch ? WY1R : WY1;
const canvas = $("canvas"), ctx = canvas.getContext("2d");
let v = null, raf = 0, last = 0;

function measure() {
  const cw = Math.max(240, canvas.clientWidth), ch = Math.max(1, canvas.clientHeight), dpr = Math.min(3, devicePixelRatio || 1);
  const w = Math.round(cw * dpr), h = Math.round(ch * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const x1 = wx1(), y1 = wy1(), s = Math.min(cw / (x1 - WX0), ch / (y1 - WY0));
  v = { s, dpr, cw, ch, ox: (cw - s * (x1 - WX0)) / 2 - s * WX0, oy: (ch - s * (y1 - WY0)) / 2 + s * y1 };
}
const px = x => v.ox + v.s * x, py = y => v.oy - v.s * y;
const mx = p => (p - v.ox) / v.s, my = p => (v.oy - p) / v.s;
const obox = o => [o.x, sillOf(o), o.x + o.w, sillOf(o) + o.h];
const boxPoly = b => [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]];
// a part as the quadrilateral it really covers, in metres from the wall's bottom-left corner: the format's own
// elementPoly - the rectangle's four corners in the rect format, the member at its true angle in the segment one
const shift = q => q.map(([x, y]) => [x + design.L / 2, y + wallTop(design) / 2]);
const epoly = e => shift(fmt().elementPoly(e));
// every part at once, so a member that declared an end cut is drawn as the saw leaves it - against the parts around
// it, which is the only way its bevel can be known (segment.elementPolys)
const epolys = es => (fmt().elementPolys ? fmt().elementPolys(es) : es.map(e => fmt().elementPoly(e))).map(shift);
const itemName = e => ITEMS[fmt().segment ? fmt().itemOf(e[0]) : e[0]];
// the wall's outline, from the scene's own gable_poly: four corners, or five when it is raked
const wallPoly = () => {
  const { L, H } = design, top = wallTop(design);
  const p = design.pitch ? SEGA.gablePoly(L, top, design.ridge, H) : boxPoly([-L / 2, -H / 2, L / 2, H / 2]);
  return p.map(([x, y]) => [x + L / 2, y + top / 2]);
};
const BLUE = "#3b8cff";                                  // the outline on a part the moment the network writes it
// bright amber timber, grey blocks: the only fills on the page
const FILL = { block: ["#c9c9c9", "#6b6b6b"], lintel: ["#8d8d8d", "#3a3a3a"], timber: ["#e7c993", "#a67c48"] };   // pine, grey blocks
// pine: a few faint grain streaks along the member, fixed per part. The member may lie at any angle, so the streaks
// are drawn in its own frame - the first edge of its quadrilateral is its length, the second its thickness.
function grain(poly, seed) {
  const P = poly.map(([x, y]) => [px(x), py(y)]);
  // the part's own frame, from its longest side: a member lies at any angle, and a declared cut may have taken a
  // corner off it, so nothing here may assume the four corners of a rectangle
  let a = P[0], len = 0, ux = 1, uy = 0;
  for (let i = 0; i < P.length; i++) {
    const q = P[(i + 1) % P.length], dx = q[0] - P[i][0], dy = q[1] - P[i][1], L = Math.hypot(dx, dy);
    if (L > len) { len = L; a = P[i]; ux = dx / L; uy = dy / L; }
  }
  if (len < 8) return;
  let s0 = 0, s1 = 0, t0 = 0, t1 = 0;                    // its extent along that side and across it
  for (const q of P) {
    const dx = q[0] - a[0], dy = q[1] - a[1];
    const alo = dx * ux + dy * uy, acr = dx * -uy + dy * ux;
    s0 = Math.min(s0, alo); s1 = Math.max(s1, alo); t0 = Math.min(t0, acr); t1 = Math.max(t1, acr);
  }
  const span = s1 - s0, n = t1 - t0;
  if (n < 3) return;
  let r = seed * 9301 + 49297;
  const rnd = () => (r = (r * 9301 + 49297) % 233280) / 233280;
  ctx.save();
  polyPath(poly); ctx.clip();                            // clipped to the part as it is really drawn, cut and all
  ctx.translate(a[0] + ux * (s0 + s1) / 2 - uy * (t0 + t1) / 2, a[1] + uy * (s0 + s1) / 2 + ux * (t0 + t1) / 2);
  ctx.rotate(Math.atan2(uy, ux));
  ctx.lineWidth = 1;
  for (let k = 0, m = 2 + Math.floor(n / 4); k < m; k++) {
    const off = (k + .5 + (rnd() - .5) * .8) * n / m - n / 2, amp = .6 + rnd() * 1.2, ph = rnd() * 6.3, dark = .12 + rnd() * .16;
    ctx.strokeStyle = `rgba(120,78,30,${dark})`; ctx.beginPath();
    for (let t = 0; t <= span; t += 6) {
      const wob = Math.sin(t / 40 + ph) * amp;
      t ? ctx.lineTo(t - span / 2, off + wob) : ctx.moveTo(t - span / 2, off + wob);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function polyPath(poly) {                                // a polygon in metres, as a path on the canvas
  ctx.beginPath();
  poly.forEach(([x, y], i) => (i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y))));
  ctx.closePath();
}
function outline(poly, dash, colour, w) {                // a boundary in metres, drawn as a line
  ctx.save(); ctx.setLineDash(dash); ctx.strokeStyle = colour; ctx.lineWidth = w; ctx.translate(.5, .5);
  polyPath(poly); ctx.stroke(); ctx.restore();
}
function cross(x, y, r, on) {                            // the CAD corner mark, and the handle you drag
  ctx.strokeStyle = on ? "#000" : "#111"; ctx.lineWidth = on ? 2 : 1;
  ctx.beginPath(); ctx.moveTo(px(x) - r, py(y)); ctx.lineTo(px(x) + r, py(y));
  ctx.moveTo(px(x), py(y) - r); ctx.lineTo(px(x), py(y) + r); ctx.stroke();
}

// the load array, drawn as a distributed load is drawn: a horizontal line above the top edge spanning the wall, with
// an arrow hanging from it at every place a load stands. Drag the line to shift the array, an arrow to set the interval.
const LARROW = 0.30, LFOOT = 0.045;                      // metres above the top edge: the line, the arrow tips
function arrow(X, yTail, yTip, head) {
  ctx.beginPath(); ctx.moveTo(X, yTail); ctx.lineTo(X, yTip - 3); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(X, yTip + 5); ctx.lineTo(X - head, yTip - 5); ctx.lineTo(X + head, yTip - 5); ctx.closePath(); ctx.fill();
}
// the interval between the first two arrows, dimensioned the way a drawing dimensions it: a thin line with tick ends
// just above the load line and the number of millimetres in a gap at its middle. While the magnet holds the drag, the
// number goes blue for an interval and the ticks for an offset - the arrow standing on a stud line is the tick's own end.
function drawDim(d, held) {
  if (d.loads.length < 2) return;
  const x0 = px(d.loads[0]), x1 = px(d.loads[1]), y = Math.max(11, py(d.H + LARROW) - 16), mid = (x0 + x1) / 2;
  const txt = String(Math.round((d.loads[1] - d.loads[0]) * 1000));
  ctx.save();
  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
  if ("letterSpacing" in ctx) ctx.letterSpacing = "1.4px";
  ctx.textAlign = "center";
  ctx.lineWidth = 1;
  const w = ctx.measureText(txt).width + 9, gap = x1 - x0 > w + 10;
  ctx.strokeStyle = held === "off" ? BLUE : "#111";      // the ends of the dimension are the two arrows
  ctx.beginPath();
  ctx.moveTo(x0 + .5, y - 4); ctx.lineTo(x0 + .5, y + 4);
  ctx.moveTo(x1 + .5, y - 4); ctx.lineTo(x1 + .5, y + 4);
  ctx.stroke();
  ctx.strokeStyle = "#111";
  ctx.beginPath();
  if (gap) { ctx.moveTo(x0, y + .5); ctx.lineTo(mid - w / 2, y + .5); ctx.moveTo(mid + w / 2, y + .5); ctx.lineTo(x1, y + .5); }
  else { ctx.moveTo(x0, y + .5); ctx.lineTo(x1, y + .5); }
  ctx.stroke();
  ctx.fillStyle = held === "sp" ? BLUE : "#111";
  ctx.textBaseline = gap ? "middle" : "bottom";
  ctx.fillText(txt, mid, gap ? y : y - 5);
  ctx.restore();
}

function drawLoads(d, on, onK) {
  const yTail = py(d.H + LARROW), yTip = py(d.H + LFOOT);
  ctx.save();
  ctx.strokeStyle = ctx.fillStyle = on ? "#1668d8" : BLUE;
  ctx.lineWidth = on ? 2.4 : 1.6;
  ctx.beginPath(); ctx.moveTo(px(0), yTail); ctx.lineTo(px(d.L), yTail); ctx.stroke();
  ctx.lineWidth = on ? 1.9 : 1.4;
  for (const x of d.loads) arrow(px(x), yTail, yTip, 4.5);
  if (d.kx !== undefined) {                             // the heavy load: a bigger arrow, from higher up, labelled
    const yk = py(d.H + HARROW);
    ctx.strokeStyle = ctx.fillStyle = onK ? "#1668d8" : BLUE;
    ctx.lineWidth = onK ? 3.4 : 2.6;
    arrow(px(d.kx), yk, yTip, 8);
    ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';
    if ("letterSpacing" in ctx) ctx.letterSpacing = "1.4px";
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    ctx.fillText(`${HEAVY_KN} kN`, px(d.kx), yk - 5);
  }
  ctx.restore();
}

// The pattern cell: the window the wall is asked to repeat, standing beside the wall at the blocks' true proportions
// (390 x 190 with 10 mm joints). Click a cell for a half block, drag across two for a whole one, click a block to take
// it away; drag off the right edge and the block wraps round to the left, where its stub is drawn.
const cellPx = u => CELLX + u * TPL.STEP * cs();
const cellPy = c => CELLY + c * TPL.COURSE * cs();
const cellW = () => TPL.WIDTH * cs(), cellH = () => TPL.COURSES * TPL.COURSE * cs();
const cellCol = p => clamp(Math.floor((mx(p.x) - CELLX) / (TPL.STEP * cs())), 0, 4);   // 4: one step past the edge

function blockRects(b) {                                 // one block in page metres; two pieces when it wraps
  const [c, u, n] = b, y0 = cellPy(c), y1 = y0 + TPL.HEIGHT * cs();
  const x0 = cellPx(u), len = (n > 1 ? TPL.FULL : TPL.HALF) * cs(), over = x0 + len - (CELLX + cellW());
  return over > 1e-9 ? [[x0, y0, CELLX + cellW(), y1], [CELLX, y0, CELLX + over, y1]] : [[x0, y0, x0 + len, y1]];
}

function litCells() {                                    // the cells the hand is holding, or the one under it
  const h = drag && drag.h[0] === "t" ? drag.h : (hover || "").startsWith("t:") ? hover : null;
  if (!h) return null;
  const [, c, u0] = h.split(":").map(Number);
  const n = clamp((drag && drag.h === h && drag.col != null ? drag.col : u0) - u0 + 1, 1, 2);
  return { c, us: Array.from({ length: n }, (_, i) => (u0 + i) % 4) };
}

function drawCell() {
  const xr = CELLX + cellW(), yt = CELLY + cellH();
  ctx.save(); ctx.strokeStyle = "#ececE8"; ctx.lineWidth = 1; ctx.beginPath();
  for (let u = 1; u < 4; u++) { const X = px(cellPx(u)) + .5; ctx.moveTo(X, py(CELLY)); ctx.lineTo(X, py(yt)); }
  for (let c = 1; c < TPL.COURSES; c++) { const Y = py(cellPy(c)) + .5; ctx.moveTo(px(CELLX), Y); ctx.lineTo(px(xr), Y); }
  ctx.stroke(); ctx.restore();
  const lit = litCells();
  if (lit) {
    ctx.fillStyle = "#eef3fc";
    for (const u of lit.us)
      ctx.fillRect(px(cellPx(u)), py(cellPy(lit.c) + TPL.COURSE * cs()), v.s * TPL.STEP * cs(), v.s * TPL.COURSE * cs());
  }
  for (const b of design.cell) for (const [x0, y0, x1, y1] of blockRects(b)) {
    ctx.beginPath(); ctx.rect(px(x0), py(y1), v.s * (x1 - x0), v.s * (y1 - y0));
    ctx.fillStyle = FILL.block[0]; ctx.fill();
    ctx.strokeStyle = FILL.block[1]; ctx.lineWidth = 1; ctx.stroke();
  }
  outline(boxPoly([CELLX, CELLY, xr, yt]), [1, 3], "#555", 1);
}

// one gesture on the window: a click on a block takes it away, a click on an empty cell lays a half block, and a drag
// across two cells lays a whole one, over whatever stood there
function editCell(t) {
  const [, c, u] = t.h.split(":").map(Number);
  const n = clamp((t.col == null ? u : t.col) - u + 1, 1, 2);
  const at = design.cell.findIndex(b => b[0] === c && steps(b).includes(u));
  if (n === 1 && at >= 0) { design.cell.splice(at, 1); return; }
  const want = steps([c, u, n]);
  design.cell = design.cell.filter(b => b[0] !== c || !steps(b).some(s => want.includes(s)));
  design.cell.push([c, u, n]);
}

// a part as the 3D view reads it: the member its format gives, moved to the wall's bottom-left corner like everything
// else the page draws. The 3D module knows no token format, only members and an outline.
const member = e => {
  const m = fmt().elementMember(e);
  return { ...m, x: m.x + design.L / 2, y: m.y + wallTop(design) / 2 };
};

function paint() {
  if (mode3 && three) {                                  // the other view of the same wall; the 2D canvas is hidden
    three.draw({ design, parts, hot, member, poly: wallPoly(), top: wallTop(design), showLoads: readsLoads() });
    return;
  }
  measure();
  const { L, H, openings } = design;
  ctx.setTransform(v.dpr, 0, 0, v.dpr, 0, 0);
  ctx.clearRect(0, 0, v.cw, v.ch);

  const partPolys = epolys(parts);
  parts.forEach((e, pi) => {                             // what the network has written so far
    const item = itemName(e);
    const [fill, stroke] = FILL[item === "block" ? "block" : item === "lintel" ? "lintel" : "timber"];
    const poly = partPolys[pi];
    polyPath(poly); ctx.fillStyle = fill; ctx.fill();
    if (fill === FILL.timber[0]) grain(poly, (e[1] * 7 + e[2] * 13 + e[3] * 3 + e[4]) % 1000);
    polyPath(poly); ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke();
  });
  if (hot) {                                             // the part just written, lit: the shape it really is, cut and all
    const q = partPolys[hot.i] || epoly(hot.e);
    ctx.save(); ctx.globalAlpha = hot.a; ctx.shadowColor = "#6fb2ff"; ctx.shadowBlur = 14;
    outline(q, [], BLUE, 2.5); outline(q, [], BLUE, 2.5); ctx.restore();
  }

  if (readsLoads()) { drawLoads(design, (hover || "")[0] === "l", hover === "k"); drawDim(design, held); }

  outline(wallPoly(), [1, 3], "#555", 1);                // the wall's own outline, and the three crosses that size it
  cross(0, 0, 7, false);
  cross(L, 0, 7, hover === "w:r" || hover === "w:c");
  cross(0, H, 7, hover === "w:t" || hover === "w:c");
  cross(L, H, 7, hover === "w:c");
  if (design.pitch) drawRake();
  openings.forEach((o, i) => {
    const b = obox(o), on = sel === i || (hover || "").startsWith(`o:${i}:`);
    outline(boxPoly(b), [1, 3], on ? "#333" : "#999", 1);
    if (!on) return;
    for (const [cx, cy, k] of [[b[0], b[1], "lb"], [b[2], b[1], "rb"], [b[0], b[3], "lt"], [b[2], b[3], "rt"]])
      cross(cx, cy, 5, hover === `o:${i}:${k}`);
    const dx = px(b[2]) + 15, dy = py(b[3]) - 15;        // the × that removes it
    ctx.strokeStyle = hover === `o:${i}:x` ? "#000" : "#999"; ctx.lineWidth = hover === `o:${i}:x` ? 2 : 1;
    ctx.beginPath(); ctx.moveTo(dx - 4, dy - 4); ctx.lineTo(dx + 4, dy + 4); ctx.moveTo(dx + 4, dy - 4); ctx.lineTo(dx - 4, dy + 4); ctx.stroke();
  });
  if (readsTemplate()) drawCell();
}

// the rake: the ridge stands on a dotted line above the eaves, with the same crosses the wall is sized by. The foot
// slides the ridge along the wall, the apex sets the pitch.
function drawRake() {
  const x = design.ridge, y = ridgeH(design);
  ctx.save(); ctx.setLineDash([1, 3]); ctx.strokeStyle = "#555"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(px(x) + .5, py(design.H)); ctx.lineTo(px(x) + .5, py(y)); ctx.stroke();
  ctx.restore();
  cross(x, y, 7, hover === "g:a");
  cross(x, design.H, 7, hover === "g:r");
}

// parts appear one by one, never all at once: one a frame, a little faster when many are waiting
function tick() {
  if (raf) return;
  raf = requestAnimationFrame(now => {
    raf = 0;
    const dt = last ? now - last : 16;
    last = now;
    for (let i = pending.length > 64 ? 4 : pending.length > 24 ? 2 : 1; i-- > 0 && pending.length;) {
      const e = pending.shift(); parts.push(e); hot = { e, i: parts.length - 1, a: 1 };
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
  if (readsTemplate()) {                                 // the window stands clear of the longest wall, so it goes first
    const c = Math.floor((my(p.y) - CELLY) / (TPL.COURSE * cs())), u = Math.floor((mx(p.x) - CELLX) / (TPL.STEP * cs()));
    if (c >= 0 && c < TPL.COURSES && u >= 0 && u < 4) return `t:${c}:${u}`;
  }
  if (readsHeavy() && design.kx !== undefined                  // the heavy load hangs from higher up than the array
      && p.y >= py(H + HARROW) - 9 && p.y <= py(H + LFOOT) + 5 && Math.abs(p.x - px(design.kx)) <= 12) return "k";
  if (readsLoads()) {                                    // the array lives above the wall, clear of its top edge:
    const yTail = py(H + LARROW), yTip = py(H + LFOOT);  // the line at the top, the arrows hanging below it
    if (Math.abs(p.y - yTail) <= 7 && p.x >= px(0) - 9 && p.x <= px(L) + 9) return "l";
    if (p.y > yTail && p.y <= yTip + 5)
      for (let i = design.loads.length - 1; i >= 0; i--)
        if (Math.abs(p.x - px(design.loads[i])) <= 11) return `l:${i}`;
  }
  if (design.pitch) {                                    // the rake's two handles, above the wall's own corners
    if (near(design.ridge, ridgeH(design), 14)) return "g:a";
    if (near(design.ridge, H, 14)) return "g:r";
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
  move: "move", x: "pointer", l: "move", "l:0": "move", k: "ew-resize", "g:a": "ns-resize", "g:r": "ew-resize" };                  // the line, and the arrow it turns about
const cursorFor = h => !h ? "default" : h[0] === "t" ? "pointer"
  : (CURSOR[h] || CURSOR[h.split(":")[2]] || (h[0] === "l" ? "ew-resize" : "default"));
const at = e => { const r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };

canvas.addEventListener("pointerdown", e => {
  const p = at(e), h = hit(p);
  canvas.focus({ preventScroll: true });
  if (h && h.startsWith("o:")) {
    const [, i, edge] = h.split(":");
    if (edge === "x") { design.openings.splice(+i, 1); sel = -1; fit(design); changed(); return; }
    sel = +i;
  } else if (!h || h[0] === "t") { sel = -1; }
  if (!h) { paint(); return; }
  e.preventDefault();
  canvas.setPointerCapture(e.pointerId);
  drag = { h, p, d0: structuredClone(design), col: h[0] === "t" ? +h.split(":")[2] : null };
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
  // the window is edited on the release, not under the hand: until then the drag only says how far it has reached
  if (drag.h[0] === "t") { const c = cellCol(p); if (c !== drag.col) { drag.col = c; paint(); } return; }
  const dx = mx(p.x) - mx(drag.p.x), dy = my(p.y) - my(drag.p.y);
  const O = drag.d0, d = design, h = drag.h.split(":"), edge = h[h.length - 1];
  if (h[0] === "k") {                                    // the heavy load slides along the top edge, openings included
    d.kx = q(clamp(O.kx + dx, LEND, Math.max(LEND, d.L - LEND)));
  } else if (h[0] === "l") {                             // the line shifts the whole array; an arrow sets the interval,
    const k = h.length > 1 ? +h[1] : 0;                  // following the cursor while the first arrow stays put
    if (k === 0) { const raw = O.off + dx; d.off = magnet(raw, offTargets(raw)); }
    else { d.off = O.off; d.sp = clamp(magnet((mx(p.x) - O.off) / k, SPSNAP), SPMIN, SPMAX); }   // wrapping keeps that arrow in the array
  } else if (h[0] === "g") {                              // the ridge slides along the wall; the apex sets the pitch
    if (edge === "r") d.ridge = clamp(O.ridge + dx, 0, d.L);   // to either end, where the wall becomes a single slope
    else d.pitch = clamp(deg(Math.atan(Math.max(0, ridgeH(O) + dy - d.H) / ridgeRun(d))), PMIN, PMAX);
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
  held = h[0] !== "l" ? null : +(h[1] || 0) ? (spHeld(d) ? "sp" : null) : (offHeld(d) ? "off" : null);
  changed();
});

const stop = e => {
  if (!drag) return;
  const t = drag.h[0] === "t" ? drag : null;
  drag = held = null;
  try { canvas.releasePointerCapture(e.pointerId); } catch {}
  if (t) { editCell(t); fit(design); changed(); return; }
  writeHash(); paint();
};
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
  // a control that cannot act is absent from the page, never greyed out
  const sw = $("sw");
  sw.hidden = m.scripts.length < 2;
  if (!sw.hidden) { $("script").checked = design.script === "block"; sw.className = `sw ${design.script}`; }
  $("addDoor").hidden = $("addWindow").hidden = design.openings.length >= MAXOPS || !freeSpan(design);
  const rk = $("rake");                                  // only a segment network reads a raked top, and only on timber
  rk.hidden = !readsRake() || design.script === "block";
  if (!rk.hidden) rk.textContent = design.pitch ? "− rake" : "+ rake";
  const iv = $("inv");                                   // the catalogue, where the artifact is about the stock
  iv.hidden = !showsPicker();
  if (!iv.hidden) for (const b of iv.children) b.classList.toggle("on", design.inv.includes(b.dataset.k));
  const vs = $("vsw");                                   // and the 3D switch, only where the entry asks for it
  vs.hidden = !m.view3d;
  if (vs.hidden && mode3) { mode3 = false; canvas.hidden = false; $("gl").hidden = true; }
  const about = ABOUT[modelId];
  $("about").textContent = about || "";
  $("aboutLink").hidden = !about;
  if (!about) $("about").hidden = true;
}
$("script")?.addEventListener("change", e => { design.script = e.target.checked ? "block" : "frame"; changed(); });
$("addDoor").addEventListener("click", () => addOpening("door"));
$("addWindow").addEventListener("click", () => addOpening("window"));
$("random").addEventListener("click", () => { design = randomDesign(); sel = -1; changed(); });
$("rake")?.addEventListener("click", () => { design.pitch = design.pitch ? 0 : 25; fit(design); changed(); });
// the inventory: one toggle per section of the catalogue, and never none - a wall must have something to be built from
for (const s of CATALOGUE) {
  const b = document.createElement("button");
  b.type = "button"; b.dataset.k = s.key; b.textContent = s.label;
  b.setAttribute("aria-label", `${s.label} mm timber`);
  b.addEventListener("click", () => {
    const has = design.inv.includes(s.key);
    if (has && design.inv.length < 2) return;
    design.inv = has ? design.inv.filter(k => k !== s.key) : [...design.inv, s.key];
    fit(design); changed();
  });
  $("inv").append(b);
}
$("aboutLink").addEventListener("click", e => { e.preventDefault(); $("about").hidden = !$("about").hidden; });
// the same wall, turned: the module and the three.js it needs are fetched the first time the switch is thrown, so an
// artifact without the switch pays nothing for it
$("vsw").addEventListener("change", async e => {
  const on = e.target.checked;
  const show = ok => { mode3 = ok; $("vsw").className = `sw ${ok ? "on" : "off"}`; $("view").checked = ok;
    canvas.hidden = ok; $("gl").hidden = !ok;
    $("hint").textContent = ok ? "Drag to turn the wall." : "Drag the wall and its openings."; };
  show(on);
  if (on && !three) {
    const was = $("status").textContent;
    status("opening the 3D view…");
    try { three = await (await import("./view3d.js")).init($("gl")); status(was); }
    catch { show(false); status("the 3D view could not load"); }
  }
  paint();
});
addEventListener("resize", paint);
if (window.ResizeObserver) { const ro = new ResizeObserver(paint); ro.observe(canvas); ro.observe($("gl")); }

sync();
writeHash();
paint();                                                 // the wall is on screen and draggable before the weights arrive
loadModel();
