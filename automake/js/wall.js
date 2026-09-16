// The wall side of Automake's M0 demo, ported line by line from the Python research code so the browser builds the same
// walls, tokens and scores (web/tests checks this against fixtures made by scripts/web_fixtures.py).
//
//   scenes      automake/data/framer.py make_wall_scene + automake/mvp/dataset.py make_scene (MultiSpec)
//   scripts     automake/mvp/naive_framer.py frame_wall, automake/mvp/naive_bricklayer.py lay_bricks,
//               build_wall's overlap drop (automake/world/contacts.py find_overlaps: later part first, keep-outs)
//   encoding    automake/mvp/encode.py (5 mm rulers, skin_features / skin_rects, member_rect / rect_member)
//   sequence    automake/mvp/sequence.py (elements_of, canonical_order, element_rects, tokens, free_rects)
//   judging     automake/mvp/metrics.py (match_members, score), count_overlaps, evaluate.refuse_overlaps
//
// Numbers follow the Python arithmetic, including where numpy works in float32 (Math.fround), so edges round to the same ticks.

export const Q = 0.005, X_MAX = 4.0, Y_MAX = 1.6, NX = 1600, NY = 640;
export const ITEMS = ["2x4", "2x6", "2x8", "2x10", "2x12", "block", "lintel"];
export const SECTIONS = { "2x4": [0.045, 0.095], "2x6": [0.045, 0.145], "2x8": [0.045, 0.195], "2x10": [0.045, 0.245],
  "2x12": [0.045, 0.295], block: [0.19, 0.19], lintel: [0.19, 0.19] };
const LENGTHS = { "2x4": [0.05, 6.0], "2x6": [0.05, 6.0], "2x8": [0.05, 6.0], "2x10": [0.05, 6.0], "2x12": [0.05, 6.0],
  block: [0.095, 0.39], lintel: [0.1, 8.0] };
export const OVERLAP_TOL = 0.003, FREE_GAP = 0.006, FREE_MIN = 0.03;
export const TYPES = ["wall", "opening", "part", "free", "load"];   // "load": a point or line load on the top edge (sequence.TYPES)
export const LOAD = 4;
export const BRIEFS = { frame: 1, block: 2 };
const TOL_CONTACT = 0.003, KEEPOUT_EXTRA = 0.05, WALL_DEPTH = 0.095;
const f32 = Math.fround;

// ---------------------------------------------------------------- rounding as Python and numpy round
export function rint(x) {                       // numpy rint / torch.round: half to even
  const f = Math.floor(x), d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}
const pyRound = (x, n) => Number(x.toFixed(n));  // Python round(x, n) (differs only on exact binary ties, which walls on the 5 mm grid never hit)
const npRound6 = x => rint(x * 1e6) / 1e6;       // numpy round(x, 6)

// ---------------------------------------------------------------- the scene: a wall and its openings
// spec: {L, H, openings: [[x from the left end, width, sill, height], ...]} (sill 0 = a door)
export function makeScene(spec) {
  const L = spec.L, H = spec.H;
  const raw = (spec.openings || []).map(o => o.slice(0, 4).map(Number));
  raw.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2] || a[3] - b[3]);
  const oz = H / 2;                             // the wall frame's origin height in the world
  const openings = raw.map((o, k) => {
    let [x0, w, sill, h] = o;
    if (k === 0) {                              // make_wall_scene clamps the first opening (clamp_opening)
      w = Math.max(0.1, Math.min(w, L));
      x0 = Math.min(Math.max(x0, 0.0), L - w);
      sill = Math.max(0.0, Math.min(sill, H - 0.1));
      h = Math.max(0.1, Math.min(h, H - sill));
      [x0, w, sill, h] = [x0, w, sill, h].map(v => pyRound(v, 4));
    }
    const cx = -L / 2 + x0 + w / 2, cy = -H / 2 + sill + h / 2;
    const wz = cy + oz;                         // world height of the centre
    return { spec: [x0, w, sill, h], cx, cy: wz - oz, wz, hw: w / 2, hh: h / 2, w, h };
  });
  return { L, H, hx: L / 2, hy: H / 2, oz, openings };
}

// ---------------------------------------------------------------- parts as placed by a script
function makePart(sc, item, orient, x, y, length, label) {
  const [lo, hi] = LENGTHS[item];
  const len = Math.min(Math.max(length, lo), hi);          // Item.resolve clamps free lengths
  const [w, d] = SECTIONS[item];
  // half extents along the wall's x, its y (up) and through it, as the object's box sees them
  const half = orient === 0 ? [w / 2, len / 2, d / 2] : (orient === 1 ? [len / 2, w / 2, d / 2] : [len / 2, d / 2, w / 2]);
  return { item, orient, x, wz: y + sc.oz, length: len, half, label };
}

function arange(start, stop, step) {            // numpy.arange for floats (length by ceil, then start + i * delta)
  const n = Math.ceil((stop - start) / step);
  const out = [];
  if (n <= 0) return out;
  out.push(start);
  if (n === 1) return out;
  const next = start + step;
  out.push(next);
  const delta = next - start;
  for (let i = 2; i < n; i++) out.push(start + i * delta);
  return out;
}

function openingsLocal(sc, r = v => v) {         // x0, x1, y0, y1 in the wall frame, left to right
  const ops = sc.openings.map(o => [o.cx - o.hw, o.cx + o.hw, o.cy - o.hh, o.cy + o.hh].map(r));
  ops.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2] || a[3] - b[3]);
  return ops;
}

// naive_framer.frame_wall
export function frameWall(sc) {
  const parts = [];
  const hx = sc.hx, hy = sc.hy, t = 0.045, spacing = 0.6;
  const put = (item, orient, x, y, length, label) => {
    if (length > LENGTHS[item][1]) {            // longer than the timber: two pieces (along the wall: plates only)
      const half = length / 2;
      for (const sign of [-1, 1]) {
        const off = sign * half / 2;
        if (orient === 0) put(item, orient, x, (y + sc.oz + off) - sc.oz, half, label);
        else put(item, orient, x + off, y, half, label);
      }
      return;
    }
    parts.push(makePart(sc, item, orient, x, y, length, label));
  };
  const ops = openingsLocal(sc);
  const yBot = -hy + t, yTop = hy - t;
  const doors = ops.filter(op => op[2] <= -hy + 1e-6);
  if (doors.length) {
    let a = -hx;
    for (const op of doors) { put("2x4", 1, 0.5 * (a + op[0]), -hy + t / 2, op[0] - a, "bottom plate"); a = op[1]; }
    put("2x4", 1, 0.5 * (a + hx), -hy + t / 2, hx - a, "bottom plate");
  } else {
    put("2x4", 1, 0.0, -hy + t / 2, 2 * hx, "bottom plate");
  }
  put("2x4", 1, 0.0, hy - t / 2, 2 * hx, "top plate");
  const xs = arange(-hx + t / 2, hx - t / 2, spacing).concat([hx - t / 2]);
  for (const x of xs) {
    if (ops.some(op => op[0] - 2 * t < x && x < op[1] + 2 * t)) continue;
    put("2x4", 0, x, 0.5 * (yBot + yTop), yTop - yBot, "stud");
  }
  for (const [x0, x1, y0, y1] of ops) {
    for (const x of [x0 - 1.5 * t, x1 + 1.5 * t]) put("2x4", 0, x, 0.5 * (yBot + yTop), yTop - yBot, "king stud");
    for (const x of [x0 - t / 2, x1 + t / 2]) put("2x4", 0, x, 0.5 * (yBot + y1), y1 - yBot, "jack stud");
    const hd = 0.195;
    put("2x8", 2, 0.5 * (x0 + x1), y1 + hd / 2, x1 - x0 + 2 * t, "header");
    for (const x of xs) if (x0 < x && x < x1 && yTop - (y1 + hd) > 0.05) put("2x4", 0, x, 0.5 * (y1 + hd + yTop), yTop - (y1 + hd), "cripple");
    if (y0 > yBot + 0.05) {
      put("2x4", 1, 0.5 * (x0 + x1), y0 - t / 2, x1 - x0, "sill");
      for (const x of xs) if (x0 < x && x < x1 && y0 - t - yBot > 0.05) put("2x4", 0, x, 0.5 * (yBot + y0 - t), y0 - t - yBot, "cripple");
    }
  }
  return parts;
}

// naive_bricklayer.lay_bricks (BrickParams defaults)
export function layBlocks(sc) {
  const EPS = 1e-6, P = { length: 0.390, height: 0.190, joint: 0.010, minPiece: 0.095, bearing: 0.200 };
  const module = P.length + P.joint, course = P.height + P.joint;
  const r6 = v => pyRound(v, 6);
  const hx = sc.hx, hy = sc.hy;
  const parts = [];
  const put = (item, x0, x1, y0, label) => parts.push(makePart(sc, item, 1, r6(0.5 * (x0 + x1)), r6(y0 + P.height / 2), r6(x1 - x0), label));
  const ops = openingsLocal(sc, r6);
  const h2 = 2 * hy;
  const nCourses = h2 < P.height - EPS ? 0 : Math.floor((h2 - P.height) / course + EPS) + 1;
  const ys = [];
  for (let k = 0; k < nCourses; k++) ys.push(r6(-hy + k * course));
  const lintels = new Map();
  ops.forEach(([x0, x1, y0, y1], i) => {
    const k = ys.findIndex(y => y >= y1 - EPS);
    if (k < 0) return;
    let a = Math.max(x0 - P.bearing, -hx, i > 0 ? ops[i - 1][1] : -hx);
    const b = Math.min(x1 + P.bearing, hx, i + 1 < ops.length ? ops[i + 1][0] : hx);
    for (const [, d] of lintels.get(k) || []) a = Math.max(a, d);
    if (!lintels.has(k)) lintels.set(k, []);
    lintels.get(k).push([r6(a), r6(b)]);
  });
  const cut = (a, b, obstacles) => {
    let pieces = [[a, b]];
    for (const [c, d] of obstacles) {
      const nxt = [];
      for (const [u, v] of pieces) {
        if (d <= u + EPS || c >= v - EPS) { nxt.push([u, v]); continue; }
        if (c > u + EPS) nxt.push([u, c]);
        if (d < v - EPS) nxt.push([d, v]);
      }
      pieces = nxt;
    }
    return pieces;
  };
  ys.forEach((y0, k) => {
    const y1 = y0 + P.height;
    const blocked = ops.filter(([, , oy0, oy1]) => y0 < oy1 - EPS && y1 > oy0 + EPS).map(([x0, x1]) => [x0, x1]);
    for (const [a, b] of lintels.get(k) || []) { put("lintel", a, b, y0, "lintel"); blocked.push([a, b]); }
    const start = -hx - (k % 2 ? module / 2 : 0.0);
    let j = 0;
    while (start + j * module < hx - EPS) {
      const x = start + j * module;
      j += 1;
      for (const [u, v] of cut(r6(Math.max(x, -hx)), r6(Math.min(x + P.length, hx)), blocked)) {
        if (v - u >= P.minPiece - EPS) put("block", u, v, y0, Math.abs(v - u - P.length) < EPS ? "block" : "cut block");
      }
    }
  });
  return parts;
}

// build_wall's clean-up: find_overlaps on the parts' boxes (every later part of an overlapping pair, every part in a keep-out)
function dropOverlaps(sc, parts) {
  const drop = new Set();
  const pen = (a, b) => {                        // separating-axis penetration of two boxes that share the wall's axes
    const ox = (a.half[0] + b.half[0]) - Math.abs(b.x - a.x);
    const oy = (a.half[1] + b.half[1]) - Math.abs(b.wz - a.wz);
    const oz = a.half[2] + b.half[2];
    return Math.min(ox, oy, oz);
  };
  for (let i = 0; i < parts.length; i++)
    for (let j = i + 1; j < parts.length; j++) if (pen(parts[i], parts[j]) > TOL_CONTACT) drop.add(j);
  const keepouts = sc.openings.map(o => ({ x: o.cx, wz: o.wz, half: [o.w / 2, o.h / 2, WALL_DEPTH / 2 + KEEPOUT_EXTRA] }));
  parts.forEach((p, i) => { if (keepouts.some(k => pen(p, k) > TOL_CONTACT)) drop.add(i); });
  return { kept: parts.filter((_, i) => !drop.has(i)), dropped: drop.size };
}

// ---------------------------------------------------------------- encoding: rulers, rectangles, elements
const toBin = (v, vmax, n) => Math.min(Math.max(rint(f32(f32(v + f32(vmax)) / f32(Q))), 0), n - 1);
export const fromBin = (b, vmax) => b * Q - vmax;

function memberRectF32(orient, x, y, L, w, d) {  // encode.member_rect on float32 numbers (the walls dataset's dtype)
  if (orient === 0) return [f32(x - f32(w / 2)), f32(y - f32(L / 2)), f32(x + f32(w / 2)), f32(y + f32(L / 2))];
  const h = orient === 1 ? w : d;
  return [f32(x - f32(L / 2)), f32(y - f32(h / 2)), f32(x + f32(L / 2)), f32(y + f32(h / 2))];
}

export function partElement(sc, p) {             // wall_members + elements_of: [item, x0, y0, x1, y1] in ticks
  const [w, d] = SECTIONS[p.item];
  const r = memberRectF32(p.orient, f32(p.x), f32(p.wz - sc.oz), f32(p.length), w, d);
  return [ITEMS.indexOf(p.item), toBin(r[0], X_MAX, NX), toBin(r[1], Y_MAX, NY), toBin(r[2], X_MAX, NX), toBin(r[3], Y_MAX, NY)];
}

export function canonicalOrder(elems) {          // bottom to top, then left to right (y0, x0, x1, y1, item); stable
  return elems.map((e, i) => i).sort((a, b) => {
    const p = elems[a], q = elems[b];
    return p[2] - q[2] || p[1] - q[1] || p[3] - q[3] || p[4] - q[4] || p[0] - q[0];
  });
}

// skin_features then skin_rects: the wall and its openings as the network reads them (through float32)
export function skinOf(sc) {
  const ops = [...sc.openings].sort((a, b) => a.cx - b.cx).slice(0, 4);
  const f = new Float32Array(19);
  f[0] = sc.L / 8.0; f[1] = sc.H / 3.2; f[2] = ops.length;
  ops.forEach((o, i) => { const b = 3 + 4 * i; f[b] = o.cx / 4.0; f[b + 1] = o.cy / 1.6; f[b + 2] = o.w / 8.0; f[b + 3] = o.h / 3.2; });
  return Array.from(f);
}

export function skinRects(skin) {
  const L = f32(skin[0] * 8.0), H = f32(skin[1] * f32(3.2));
  const wall = [-L / 2, -H / 2, L / 2, H / 2];
  const ops = [];
  for (let i = 0; i < Math.round(skin[2]); i++) {
    const b = 3 + 4 * i;
    const cx = f32(skin[b] * 4.0), cy = f32(skin[b + 1] * f32(1.6)), w = f32(skin[b + 2] * 8.0), h = f32(skin[b + 3] * f32(3.2));
    ops.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]);
  }
  return { wall, ops };
}

// build_wall + wall_members + elements_of + canonical order: the script's wall as elements
export function buildWall(spec, script) {
  const sc = makeScene(spec);
  const placed = script === "block" ? layBlocks(sc) : frameWall(sc);
  const { kept, dropped } = dropOverlaps(sc, placed);
  const elems = kept.map(p => partElement(sc, p));
  const order = canonicalOrder(elems);
  const skin = skinOf(sc);
  return { scene: sc, skin, ...skinRects(skin), elements: order.map(i => elems[i]), labels: order.map(i => kept[i].label), dropped };
}

export function elementRect(e) {                 // element_rects: metres, through float32
  return [f32(e[1] * 0.005 - X_MAX), f32(e[2] * 0.005 - Y_MAX), f32(e[3] * 0.005 - X_MAX), f32(e[4] * 0.005 - Y_MAX)];
}

export function elementMember(e) {               // elements_to_members: item, orient, centre, length
  const item = ITEMS[e[0]], [w, d] = SECTIONS[item];
  const x0 = fromBin(e[1], X_MAX), y0 = fromBin(e[2], Y_MAX), x1 = fromBin(e[3], X_MAX), y1 = fromBin(e[4], Y_MAX);
  const dx = x1 - x0, dy = y1 - y0, x = (x0 + x1) / 2, y = (y0 + y1) / 2;
  const fits = [[npRound6(Math.abs(dy - w)), 1], [npRound6(Math.abs(dy - d)), 2], [npRound6(Math.abs(dx - w)), 0]];
  let best = fits[0];
  for (const fit of fits) if (fit[0] < best[0]) best = fit;
  const orient = best[1];
  return { item, orient, x, y, L: orient === 0 ? dy : dx, depth: d };
}

// ---------------------------------------------------------------- judging
export function matchMembers(pred, ref, posTol = 0.03, lenTol = 0.012) {
  const cand = [];
  pred.forEach((p, i) => ref.forEach((r, j) => {
    if (p.item !== r.item || p.orient !== r.orient) return;
    const d = Math.hypot(p.x - r.x, p.y - r.y);
    if (d <= posTol && Math.abs(p.L - r.L) <= lenTol) cand.push([d, i, j]);
  }));
  cand.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  const usedP = new Set(), usedR = new Set(), pairs = [];
  for (const [, i, j] of cand) {
    if (usedP.has(i) || usedR.has(j)) continue;
    usedP.add(i); usedR.add(j); pairs.push([i, j]);
  }
  return { pairs, extra: pred.map((_, i) => i).filter(i => !usedP.has(i)), missing: ref.map((_, j) => j).filter(j => !usedR.has(j)) };
}

export function countOverlaps(rects, tol = OVERLAP_TOL) {
  const t = f32(tol);
  let n = 0;
  for (let i = 0; i < rects.length; i++) for (let j = i + 1; j < rects.length; j++) {
    const a = rects[i], b = rects[j];
    if (f32(Math.min(a[2], b[2]) - Math.max(a[0], b[0])) > t && f32(Math.min(a[3], b[3]) - Math.max(a[1], b[1])) > t) n++;
  }
  return n;
}

export function scoreElements(elems, refElems) { // evaluate.judge: metrics.score plus overlapping pairs
  const pred = elems.map(elementMember), ref = refElems.map(elementMember);
  const { pairs, extra, missing } = matchMembers(pred, ref);
  const m = pairs.length;
  return { precision: pred.length ? m / pred.length : 1.0, recall: ref.length ? m / ref.length : 1.0, matched: m, extra: extra.length,
    missing: missing.length, exact: extra.length === 0 && missing.length === 0, overlaps: countOverlaps(elems.map(elementRect)),
    pairs, extraIdx: extra, missingIdx: missing };
}

// evaluate.refuse_overlaps: each new part in writing order is kept if it is inside the wall and overlaps no opening, no part
// already there and no new part kept before it
export function refuseOverlaps(wall, obstacles, news, tol = OVERLAP_TOL) {
  const obs = obstacles.map(r => r.slice());
  const kept = [];
  for (const e of news) {
    const r = elementRect(e);
    const inside = r[0] >= wall[0] - tol && r[1] >= wall[1] - tol && r[2] <= wall[2] + tol && r[3] <= wall[3] + tol;
    const hit = obs.some(o => Math.min(r[2], o[2]) - Math.max(r[0], o[0]) > tol && Math.min(r[3], o[3]) - Math.max(r[1], o[1]) > tol);
    if (inside && !hit) { kept.push(e); obs.push(r); }
  }
  return { kept, refused: news.length - kept.length };
}

// sequence.load_rects: one token per load, a segment [x0, x1] of the wall's top edge (wall frame, metres) drawn as the
// rect (x0 - Q/2, top - Q, x1 + Q/2, top). A point load is x0 = x1 (a tick square); a line load spans [x0, x1]. The
// magnitude is implicit - 10 kN for a point load, 10 kN/m for a line load - so the width says which it is. A load is not
// an obstacle, so free space (types 1 and 2) and the canvas cover (types 0 to 2) never see it.
export function loadRects(wall, loads) {
  const top = f32(wall[3]), h = f32(Q / 2), q = f32(Q);
  return (loads || []).map(v => {
    const s = Array.isArray(v) ? v : [v, v];
    return [f32(f32(s[0]) - h), f32(top - q), f32(f32(s[1]) + h), top];
  });
}

// ---------------------------------------------------------------- the network's input tokens
// sequence.tokens + sequence.free_rects: rects (float32 metres), types (0 wall, 1 opening, 2 part, 4 load, 3 free),
// items (0 none, 1 + item). Loads (segments, one token each) go after the parts and before the free space, as Python
// builds them; a network that does not read loads must never be given any (its type embedding has no row for them).
export function tokens(wall, ops, present, loads = []) {
  const rects = [wall.map(f32), ...ops.map(o => o.map(f32)), ...present.map(elementRect), ...loadRects(wall, loads)];
  const types = [0, ...ops.map(() => 1), ...present.map(() => 2), ...(loads || []).map(() => LOAD)];
  const items = [0, ...ops.map(() => 0), ...present.map(e => 1 + e[0]), ...(loads || []).map(() => 0)];
  for (const r of freeRects(rects, types)) { rects.push(r); types.push(3); items.push(0); }
  return { rects, types, items };
}

export function freeRects(rects, types) {
  const wall = rects[types.indexOf(0)];
  const g = FREE_GAP, tol = 2e-6;
  let o = [];
  rects.forEach((r, i) => {
    if (types[i] !== 1 && types[i] !== 2) return;
    const q = [Math.max(r[0] - g, wall[0]), Math.max(r[1], wall[1]), Math.min(r[2] + g, wall[2]), Math.min(r[3], wall[3])];
    if (q[2] > q[0] && q[3] > q[1]) o.push(q);
  });
  const ysAll = [wall[1], wall[3], ...o.map(q => q[1]), ...o.map(q => q[3])].map(npRound6).sort((a, b) => a - b);
  const ys = ysAll.filter((v, i) => i === 0 || v !== ysAll[i - 1]);
  const S = ys.length - 1;
  const iv = [];
  for (let k = 0; k < S; k++) {
    const c = o.filter(q => q[1] <= ys[k] + tol && q[3] >= ys[k + 1] - tol).sort((a, b) => a[0] - b[0]);
    let x = wall[0];
    for (const q of c) {
      if (q[0] > x + tol) iv.push([k, x, q[0]]);
      x = Math.max(x, q[2]);
    }
    if (wall[2] > x + tol) iv.push([k, x, wall[2]]);
  }
  if (!iv.length) return [];
  const K = iv.length;
  let out = [];
  for (let k = 0; k < K; k++) {
    const [st, xa, xb] = iv[k];
    const free = new Uint8Array(S);
    for (let k2 = 0; k2 < K; k2++) if (iv[k2][1] <= xa + tol && iv[k2][2] >= xb - tol) free[iv[k2][0]] = 1;
    let up = S, down = -1;
    for (let s = st + 1; s < S; s++) if (!free[s]) { up = s; break; }
    for (let s = st - 1; s >= 0; s--) if (!free[s]) { down = s; break; }
    const r = [xa, ys[down + 1], xb, ys[up]];
    const shift = [-g, 0.0, g, 0.0];
    for (let m = 0; m < 4; m++) r[m] = r[m] + (Math.abs(r[m] - wall[m]) <= 1e-5 + 1e-5 * Math.abs(wall[m]) ? 0.0 : shift[m]);
    if (r[2] - r[0] >= FREE_MIN && r[3] - r[1] >= FREE_MIN) out.push(r.map(npRound6));
  }
  out.sort((a, b) => a[1] - b[1] || a[0] - b[0] || a[2] - b[2] || a[3] - b[3]);
  out = out.filter((r, i) => i === 0 || r.some((v, m) => v !== out[i - 1][m]));
  return out.map(r => r.map(f32));
}
