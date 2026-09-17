// The segment token format in the browser, ported line by line from the Python research code (2026-09-16), so the page
// writes the same walls as a network trained in it. The rectangle format that `wall.js` still carries is what every
// published network before that date speaks; this module is what the ones after it speak.
//
//   a member    automake/mvp/encode.py: the two ends of its reference edge (the centreline offset by half the section
//               onto a face the 5 mm ruler can name) plus a class - horizontal, vertical or rotated - carried with the
//               item on one head: code = item * 3 + class
//   a skin      an outline polygon (encode.skin_features / skin_poly / poly_top / poly_outside); a rectangle is its
//               four-edge case, a gable its five
//   tokens      sequence.tokens: one per outline edge, then the openings, the parts, the loads, the template cell's
//               blocks and the inventory's sections, each two points, with a thickness beside it (sequence.token_boxes
//               turns a segment back into the area it fills)
//   refusal     sequence.polys_overlap: the separating-axis test on what a member really covers
//   the script  automake/mvp/gable_framer.py: a plate along each rake, every stud cut to the plate as placed
//
// Numbers follow the Python arithmetic, including where numpy works in float32 (Math.fround).

import { ITEMS, OVERLAP_TOL, Q, SECTIONS, rint } from "./wall.js";

// the segment format's own rulers: x 8 m as before, y 4.8 m because a gable's ridge needs the height
export const RULER = { X_MAX: 4.0, Y_MAX: 2.4, NX: 1600, NY: 960 };

export const segment = true;
export const N_CLASS = 3;
export const N_CUTS = 4;                           // the two ends' cut declarations, as a 2-bit mask
export const N_CODE = N_CLASS * N_CUTS;            // what one item is worth on the first head
export const NPOLY = 6;                            // corners a part is carried as: four as written, two for a cut
export const CUT_LO = 1, CUT_HI = 2;               // bit 0 the -x end of the member's own axis, bit 1 the +x end
export const CLASSES = ["horizontal", "vertical", "rotated"];
export const MAX_POLY = 6, MAX_OPENINGS = 4;
export const SKIN_DIM = 2 + 2 * MAX_POLY + 4 * MAX_OPENINGS;      // 30
const f32 = Math.fround;
const npRound6 = x => rint(x * 1e6) / 1e6;

export const codeOf = (item, cls, cuts = 0) => (item * N_CLASS + cls) * N_CUTS + cuts;
export const itemOf = code => Math.floor(code / (N_CLASS * N_CUTS));
export const classOf = code => Math.floor(code / N_CUTS) % N_CLASS;
export const cutsOf = code => code % N_CUTS;
// which coordinate field a class implies, and which one it copies (encode.IMPLIED)
export const IMPLIED = [[4, 2], [3, 1], [0, 0]];

export function thickOf(item) {                    // the member's size across its own axis: its lay says which
  const [w, d, lay] = SECTIONS[item];
  return lay === "flat" ? w : d;
}

export function normalOf(u) { return [-u[1], u[0]]; }   // the reference edge's side: the left normal of the axis

export function segClass(x0, y0, x1, y1, tol = 1e-6) {
  if (Math.abs(y1 - y0) <= tol) return 0;
  if (Math.abs(x1 - x0) <= tol) return 1;
  return 2;
}

export function closeElement(e) {                  // sequence.close_elements: fill in the coordinate the class implies
  const out = e.slice();
  const [f, s] = IMPLIED[classOf(out[0])];
  if (f) out[f] = out[s];
  return out;
}

const toBin = (v, vmax, n) => Math.min(Math.max(rint(f32(f32(v + f32(vmax)) / f32(Q))), 0), n - 1);
export const fromBin = (b, vmax) => b * Q - vmax;

export function elementSeg(e) {                    // sequence.element_segs: the reference edge in metres, through float32
  const c = closeElement(e), { X_MAX, Y_MAX } = RULER;
  return [f32(c[1] * 0.005 - X_MAX), f32(c[2] * 0.005 - Y_MAX), f32(c[3] * 0.005 - X_MAX), f32(c[4] * 0.005 - Y_MAX)];
}

export function elementMember(e) {                 // encode.seg_member, off `from_bin`: float64, not the tokens' float32
  const item = ITEMS[itemOf(e[0])], [w, d, lay] = SECTIONS[item], t = lay === "flat" ? w : d;
  const c = closeElement(e), { X_MAX, Y_MAX } = RULER;
  const x0 = fromBin(c[1], X_MAX), y0 = fromBin(c[2], Y_MAX), x1 = fromBin(c[3], X_MAX), y1 = fromBin(c[4], Y_MAX);
  const dx = x1 - x0, dy = y1 - y0, L = Math.hypot(dx, dy);
  const u = L > 1e-9 ? [dx / L, dy / L] : [1, 0], n = normalOf(u);
  return { item, orient: classOf(e[0]) === 1 ? 0 : (lay === "flat" ? 1 : 2),
    x: (x0 + x1) / 2 + n[0] * t / 2, y: (y0 + y1) / 2 + n[1] * t / 2,
    L, depth: d, w, thick: t, ang: Math.atan2(dy, dx) };
}

export function memberPoly(m) {                    // encode.member_corners: the four corners it really covers
  const u = [Math.cos(m.ang), Math.sin(m.ang)], n = normalOf(u);
  const a = [u[0] * m.L / 2, u[1] * m.L / 2], b = [n[0] * m.thick / 2, n[1] * m.thick / 2];
  return [[m.x - a[0] - b[0], m.y - a[1] - b[1]], [m.x + a[0] - b[0], m.y + a[1] - b[1]],
    [m.x + a[0] + b[0], m.y + a[1] + b[1]], [m.x - a[0] + b[0], m.y - a[1] + b[1]]];
}

export const elementPoly = e => memberPoly(elementMember(e));

// Polygons of three to NPOLY corners, the last repeated to fill (sequence.pad_polys). A repeated corner is a side of
// zero length: it changes no projection, and `normals` hands it the first side's axis, so the padding separates
// nothing and the test reads a padded polygon exactly as it reads the polygon itself.
export const padPoly = q => q.concat(Array(Math.max(0, NPOLY - q.length)).fill(q[q.length - 1]));

// a padded polygon's real corners: the repeats at its end dropped (sequence.unpad)
export function unpad(q) {
  let n = q.length;
  while (n > 3 && Math.abs(q[n - 1][0] - q[n - 2][0]) < 1e-12 && Math.abs(q[n - 1][1] - q[n - 2][1]) < 1e-12) n -= 1;
  return q.slice(0, n);
}

// sequence._same_axes: two parts that share their axes are never sawn - a plumb stud under a level plate has no
// bevel to cut, so that overlap stays refused
function sameAxes(a, b, tol = 1e-6) {
  const ang = q => ((Math.atan2(q[1][1] - q[0][1], q[1][0] - q[0][0]) % (Math.PI / 2)) + Math.PI / 2) % (Math.PI / 2);
  const d = Math.abs(ang(a) - ang(b));
  return Math.min(d, Math.PI / 2 - d) < tol;
}

// sequence._cut_plane: the side of `o` one declared end is sawn on - whichever leaves most of the part, among those
// that take material off that end (`along` is each corner's distance towards it)
// sequence.shortest_side: how much member a cut leaves where it takes most - the shorter of the two long sides
export function shortestSide(m, n, d) {
  const u = [Math.cos(m.ang), Math.sin(m.ang)], c = [m.x, m.y], t = n[0] * u[0] + n[1] * u[1];
  let worst = m.L;
  for (const k of [-1, 1]) {
    const p0 = [c[0] - k * (m.thick / 2) * u[1], c[1] + k * (m.thick / 2) * u[0]];
    let lo = -m.L / 2, hi = m.L / 2;
    if (Math.abs(t) >= 1e-12) {
      const x = (d - (n[0] * p0[0] + n[1] * p0[1])) / t;
      if (t > 0) hi = Math.min(hi, x); else lo = Math.max(lo, x);
    }
    worst = Math.min(worst, Math.max(0, hi - lo));
  }
  return worst;
}

function cutPlane(q, o, m, along, tol) {
  let best = null;
  for (let i = 0; i < o.length; i++) {
    const a = o[i], b = o[(i + 1) % o.length];
    let n = [b[1] - a[1], a[0] - b[0]];            // the side's outward normal (the corners run counter-clockwise)
    const L = Math.hypot(n[0], n[1]);
    if (L < 1e-12) continue;
    n = [n[0] / L, n[1] / L];
    const d = n[0] * a[0] + n[1] * a[1];
    const out = q.map(r => r[0] * n[0] + r[1] * n[1] - d);
    const bites = out.map(v => v < -tol);
    if (Math.max.apply(null, out) <= tol || !bites.some(Boolean) || bites.some((v, k) => v && along[k] <= 0)) continue;
    const keep = shortestSide(m, [-n[0], -n[1]], -d);
    if (best === null || keep > best[0]) best = [keep, n, d];
  }
  return best === null ? null : [best[1], best[2]];
}

// sequence._clip: `q` cut to the outside of n.x = d, corner for corner as the world cuts the solid
function clipPoly(q, n, d) {
  const e = q.map(r => r[0] * n[0] + r[1] * n[1] - d);
  const out = [];
  for (let i = 0; i < q.length; i++) {
    const j = (i + 1) % q.length;
    if (e[i] >= 0) out.push(q[i]);
    if ((e[i] > 0 && e[j] < 0) || (e[i] < 0 && e[j] > 0)) {
      const t = e[i] / (e[i] - e[j]);
      out.push([q[i][0] + (q[j][0] - q[i][0]) * t, q[i][1] + (q[j][1] - q[i][1]) * t]);
    }
  }
  return out.length >= 3 && out.length <= NPOLY ? out : q;
}

// sequence.cut_poly: the part as the saw leaves it, given the ends it declared cut, the member it is and the raw
// parts around it - the world's resolve_cuts, in plan
export function cutPoly(q, cutBits, m, others, tol = OVERLAP_TOL) {
  const u = [Math.cos(m.ang), Math.sin(m.ang)], c = [m.x, m.y];
  let w = unpad(q);                                // the corners it really has: a padded one is no corner to cut
  for (const be of [[CUT_LO, -1], [CUT_HI, 1]]) {
    if (!(cutBits & be[0])) continue;
    for (const o of others) {
      const padded = padPoly(w);
      if (sameAxes(padded, o) || !polysOverlap(padded, o, tol)) continue;
      const along = w.map(r => be[1] * ((r[0] - c[0]) * u[0] + (r[1] - c[1]) * u[1]));
      const pl = cutPlane(w, unpad(o), m, along, tol);
      if (pl) w = clipPoly(w, pl[0], pl[1]);
    }
  }
  return padPoly(w);
}

// sequence.element_polys: the corners each member covers, the cuts it declared taken off it, so the shape the page
// draws, the shape the overlap test measures and the solid the world builds are one shape
export function elementPolys(elems) {
  const mem = elems.map(elementMember), polys = mem.map(m => padPoly(memberPoly(m)));
  if (polys.length < 2 || !elems.some(e => cutsOf(e[0]))) return polys;
  return polys.map((q, i) => cutsOf(elems[i][0])
    ? cutPoly(q, cutsOf(elems[i][0]), mem[i], polys.filter((_, j) => j !== i)) : q);
}

export function segBox(seg, thick) {               // sequence.token_boxes: a segment grown by its thickness, as extents
  const d = [seg[2] - seg[0], seg[3] - seg[1]], L = Math.hypot(d[0], d[1]) || 1e-12;
  const n = normalOf([d[0] / L, d[1] / L]).map(v => v * thick);
  const xs = [seg[0], seg[2], seg[0] + n[0], seg[2] + n[0]], ys = [seg[1], seg[3], seg[1] + n[1], seg[3] + n[1]];
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

export const elementRect = e => segBox(elementSeg(e), thickOf(ITEMS[itemOf(e[0])]));
export const rectPoly = r => padPoly([[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]]);

// ---------------------------------------------------------------- the separating-axis test (sequence.polys_overlap)
function normals(p) {
  const out = [];
  for (let i = 0; i < p.length; i++) {
    const q = p[(i + 1) % p.length], e = [q[0] - p[i][0], q[1] - p[i][1]];
    const n = [-e[1], e[0]], L = Math.hypot(n[0], n[1]);
    out.push(L > 1e-12 ? [n[0] / L, n[1] / L] : null);      // a padded corner has no side, so no axis of its own
  }
  const first = out.find(v => v !== null) || [1, 0];
  return out.map(v => v || first);                          // it is given the first side's, which separates nothing
}

export function polysOverlap(a, b, tol = OVERLAP_TOL) {
  for (const ax of [...normals(a), ...normals(b)]) {
    let amin = Infinity, amax = -Infinity, bmin = Infinity, bmax = -Infinity;
    for (const p of a) { const v = p[0] * ax[0] + p[1] * ax[1]; amin = Math.min(amin, v); amax = Math.max(amax, v); }
    for (const p of b) { const v = p[0] * ax[0] + p[1] * ax[1]; bmin = Math.min(bmin, v); bmax = Math.max(bmax, v); }
    if (Math.min(amax, bmax) - Math.max(amin, bmin) <= tol) return false;
  }
  return true;
}

export function countOverlaps(polys, tol = OVERLAP_TOL) {
  let n = 0;
  for (let i = 0; i < polys.length; i++) for (let j = i + 1; j < polys.length; j++) if (polysOverlap(polys[i], polys[j], tol)) n++;
  return n;
}

// ---------------------------------------------------------------- the skin as a polygon
export function skinFeatures(sc) {                 // encode.skin_features
  const { X_MAX, Y_MAX } = RULER;
  const poly = sc.poly || [[-sc.hx, -sc.hy], [sc.hx, -sc.hy], [sc.hx, sc.hy], [-sc.hx, sc.hy]];
  const ops = [...sc.openings].sort((a, b) => a.cx - b.cx).slice(0, MAX_OPENINGS);
  const f = new Float32Array(SKIN_DIM);
  f[0] = poly.length; f[1] = ops.length;
  poly.forEach((v, i) => { f[2 + 2 * i] = v[0] / X_MAX; f[3 + 2 * i] = v[1] / Y_MAX; });
  ops.forEach((o, i) => {
    const b = 2 + 2 * MAX_POLY + 4 * i;
    f[b] = o.cx / X_MAX; f[b + 1] = o.cy / Y_MAX; f[b + 2] = o.w / (2 * X_MAX); f[b + 3] = o.h / (2 * Y_MAX);
  });
  return Array.from(f);
}

export function skinPoly(skin) {
  const { X_MAX, Y_MAX } = RULER, n = Math.round(skin[0]), out = [];
  for (let i = 0; i < n; i++) out.push([skin[2 + 2 * i] * X_MAX, skin[3 + 2 * i] * Y_MAX]);   // float64, as numpy
  return out;
}

export function skinRects(skin) {                  // encode.skin_rects: the outline's bounding box and the openings
  const { X_MAX, Y_MAX } = RULER, poly = skinPoly(skin);
  const wall = [Math.min(...poly.map(p => p[0])), Math.min(...poly.map(p => p[1])),
    Math.max(...poly.map(p => p[0])), Math.max(...poly.map(p => p[1]))];
  const ops = [];
  for (let i = 0; i < Math.round(skin[1]); i++) {
    const b = 2 + 2 * MAX_POLY + 4 * i;
    // numpy scalar promotion (NEP 50): a float32 times a plain float stays float32, so these are float32 multiplies
    const cx = f32(skin[b] * f32(X_MAX)), cy = f32(skin[b + 1] * f32(Y_MAX));
    const w = f32(skin[b + 2] * f32(2 * X_MAX)), h = f32(skin[b + 3] * f32(2 * Y_MAX));
    ops.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]);
  }
  return { wall, ops };
}

export const polyEdges = poly => poly.map((a, i) => [a[0], a[1], poly[(i + 1) % poly.length][0], poly[(i + 1) % poly.length][1]]);

export function polyTop(poly, x) {                 // encode.poly_top: the outline's upper boundary above x
  let out = -Infinity;
  for (const [x0, y0, x1, y1] of polyEdges(poly)) {
    const lo = Math.min(x0, x1), hi = Math.max(x0, x1);
    if (x < lo - 1e-9 || x > hi + 1e-9) continue;
    out = Math.max(out, hi - lo < 1e-12 ? Math.max(y0, y1) : y0 + (x - x0) * (y1 - y0) / (x1 - x0));
  }
  return out;
}

function cuts(edges, y) {
  const xs = [];
  for (const [x0, y0, x1, y1] of edges) {
    if (Math.min(y0, y1) - 1e-9 <= y && y <= Math.max(y0, y1) + 1e-9)
      Math.abs(y1 - y0) < 1e-12 ? xs.push(x0, x1) : xs.push(x0 + (y - y0) * (x1 - x0) / (y1 - y0));
  }
  return xs;
}

export function polyOutside(poly) {                // encode.poly_outside: the bounding box less the outline
  const xLo = Math.min(...poly.map(p => p[0])), xHi = Math.max(...poly.map(p => p[0]));
  const all = poly.map(p => rint(p[1] * 1e9) / 1e9).sort((a, b) => a - b);   // numpy round(y, 9), as Python
  const ys = all.filter((v, i) => i === 0 || v !== all[i - 1]);
  const e = polyEdges(poly), out = [];
  for (let i = 0; i + 1 < ys.length; i++) {
    const [ya, yb] = [ys[i], ys[i + 1]];
    let left = xLo, right = xHi;
    for (const y of [ya, yb]) {
      const xs = cuts(e, y);
      if (xs.length) { left = Math.max(left, Math.min(...xs)); right = Math.min(right, Math.max(...xs)); }
    }
    if (left > xLo + 1e-9) out.push([xLo, ya, left, yb]);
    if (right < xHi - 1e-9) out.push([right, ya, xHi, yb]);
  }
  return out;
}

// ---------------------------------------------------------------- tokens
export function loadRects(poly, loads) {           // sequence.load_rects: on the outline's upper boundary, not a flat top
  const h = f32(Q / 2), q = f32(Q);
  return (loads || []).map(v => {
    const s = Array.isArray(v) ? v : [v, v];
    const x0 = f32(s[0]), x1 = f32(s[1]), top = f32(Math.min(polyTop(poly, x0), polyTop(poly, x1)));
    return [f32(x0 - h), f32(top - q), f32(x1 + h), top];
  });
}

// The period cell of automake/mvp/template.py: a window two blocks wide and four courses tall, filled on the half
// module lattice, which the wall is asked to repeat. A block may run off the cell's right edge and wrap round to its
// left - the tiling does that by itself, so a block is simply (x0, x1, course) with x1 free to pass WIDTH.
export const TEMPLATE = { STEP: 0.2, FULL: 0.39, HALF: 0.195, COURSE: 0.2, HEIGHT: 0.19, WIDTH: 0.8, COURSES: 4 };

export function cellRects(wall, cell) {            // sequence.cell_rects: the cell drawn once in the wall's corner
  const { COURSE, HEIGHT } = TEMPLATE, x0 = f32(wall[0]), y0 = f32(wall[1]);
  return (cell || []).map(([a, b, c]) => {         // float32 throughout, as numpy promotes a weak Python float (NEP 50)
    const yc = f32(y0 + f32(c * COURSE));
    return [f32(x0 + f32(a)), yc, f32(x0 + f32(b)), f32(yc + f32(HEIGHT))];
  });
}

export function invRects(wall, inv) {              // sequence.inv_rects: each available item's own section, as a cut
  const x0 = f32(wall[0]), y0 = f32(wall[1]);      // through the member would show it - thickness across, depth along
  return (inv || []).map(i => {
    const [w, d, lay] = SECTIONS[ITEMS[i]], thick = lay === "flat" ? w : d, through = lay === "flat" ? d : w;
    return [x0, y0, f32(x0 + f32(thick)), f32(y0 + f32(through))];
  });
}

// `cell`: the template cell as (x0, x1, course) in metres; `inv`: the items the design may be built from, as indices
// into ITEMS. Both are facts about the brief rather than places, and only a network built with them is ever given
// them (its type embedding has no row for a kind it never saw), exactly as with the loads.
export function tokens(skin, present, loads = [], cell = null, inv = null) {
  const poly = skinPoly(skin), { wall, ops } = skinRects(skin);
  const rects = [...polyEdges(poly).map(r => r.map(f32)), ...ops.map(o => o.map(f32))];
  const types = [...poly.map(() => 0), ...ops.map(() => 1)];
  const items = rects.map(() => 0);
  const thick = rects.map(() => 0.0);
  for (const e of present) {
    rects.push(elementSeg(e)); types.push(2); items.push(1 + e[0]); thick.push(thickOf(ITEMS[itemOf(e[0])]));
  }
  for (const r of loadRects(poly, loads)) { rects.push(r); types.push(4); items.push(0); thick.push(0.0); }
  for (const r of cellRects(wall, cell)) { rects.push(r); types.push(5); items.push(0); thick.push(0.0); }
  invRects(wall, inv).forEach((r, i) => {          // an inventory token names its item on the same code a member does
    rects.push(r); types.push(6); items.push(1 + codeOf(inv[i], 0)); thick.push(0.0);
  });
  for (const r of freeRects(rects, types, thick, poly)) { rects.push(r); types.push(3); items.push(0); thick.push(0.0); }
  return { rects, types, items, thick };
}

export function freeRects(rects, types, thick, poly) {   // sequence.free_rects, the outline's own wedges as obstacles
  const wall = [Math.min(...poly.map(p => p[0])), Math.min(...poly.map(p => p[1])),
    Math.max(...poly.map(p => p[0])), Math.max(...poly.map(p => p[1]))];
  const g = 0.006, tol = 2e-6, FREE_MIN = 0.03;
  const boxes = [];
  rects.forEach((r, i) => { if (types[i] === 1 || types[i] === 2) boxes.push(segBox(r, thick[i])); });
  boxes.push(...polyOutside(poly));
  const o = [];
  for (const r of boxes) {
    const q = [Math.max(r[0] - g, wall[0]), Math.max(r[1], wall[1]), Math.min(r[2] + g, wall[2]), Math.min(r[3], wall[3])];
    if (q[2] > q[0] && q[3] > q[1]) o.push(q);
  }
  const ysAll = [wall[1], wall[3], ...o.map(q => q[1]), ...o.map(q => q[3])].map(npRound6).sort((a, b) => a - b);
  const ys = ysAll.filter((v, i) => i === 0 || v !== ysAll[i - 1]);
  const S = ys.length - 1, iv = [];
  for (let k = 0; k < S; k++) {
    const c = o.filter(q => q[1] <= ys[k] + tol && q[3] >= ys[k + 1] - tol).sort((a, b) => a[0] - b[0]);
    let x = wall[0];
    for (const q of c) { if (q[0] > x + tol) iv.push([k, x, q[0]]); x = Math.max(x, q[2]); }
    if (wall[2] > x + tol) iv.push([k, x, wall[2]]);
  }
  if (!iv.length) return [];
  let out = [];
  for (let k = 0; k < iv.length; k++) {
    const [st, xa, xb] = iv[k];
    const free = new Uint8Array(S);
    for (const w of iv) if (w[1] <= xa + tol && w[2] >= xb - tol) free[w[0]] = 1;
    let up = S, down = -1;
    for (let s = st + 1; s < S; s++) if (!free[s]) { up = s; break; }
    for (let s = st - 1; s >= 0; s--) if (!free[s]) { down = s; break; }
    const r = [xa, ys[down + 1], xb, ys[up]], shift = [-g, 0.0, g, 0.0];
    for (let m = 0; m < 4; m++) r[m] = r[m] + (Math.abs(r[m] - wall[m]) <= 1e-5 + 1e-5 * Math.abs(wall[m]) ? 0.0 : shift[m]);
    if (r[2] - r[0] >= FREE_MIN && r[3] - r[1] >= FREE_MIN) out.push(r.map(npRound6));
  }
  out.sort((a, b) => a[1] - b[1] || a[0] - b[0] || a[2] - b[2] || a[3] - b[3]);
  out = out.filter((r, i) => i === 0 || r.some((v, m) => v !== out[i - 1][m]));
  return out.map(r => r.map(f32));
}

// ---------------------------------------------------------------- judging and the world's refusal
export function matchMembers(pred, ref, posTol = 0.03, lenTol = 0.012, angTol = 0.02) {
  const cand = [];
  pred.forEach((p, i) => ref.forEach((r, j) => {
    if (p.item !== r.item || p.orient !== r.orient) return;
    let da = Math.abs((p.ang || 0) - (r.ang || 0)) % Math.PI;
    da = Math.min(da, Math.PI - da);
    const d = Math.hypot(p.x - r.x, p.y - r.y);
    if (d <= posTol && Math.abs(p.L - r.L) <= lenTol && da <= angTol) cand.push([d, i, j]);
  }));
  cand.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  const usedP = new Set(), usedR = new Set(), pairs = [];
  for (const [, i, j] of cand) {
    if (usedP.has(i) || usedR.has(j)) continue;
    usedP.add(i); usedR.add(j); pairs.push([i, j]);
  }
  return { pairs, extra: pred.map((_, i) => i).filter(i => !usedP.has(i)), missing: ref.map((_, j) => j).filter(j => !usedR.has(j)) };
}

export function scoreElements(elems, refElems) {
  const pred = elems.map(elementMember), ref = refElems.map(elementMember);
  const polys = elementPolys(elems);
  const { pairs, extra, missing } = matchMembers(pred, ref);
  const m = pairs.length;
  return { precision: pred.length ? m / pred.length : 1.0, recall: ref.length ? m / ref.length : 1.0, matched: m,
    extra: extra.length, missing: missing.length, exact: extra.length === 0 && missing.length === 0,
    overlaps: countOverlaps(polys), pairs, extraIdx: extra, missingIdx: missing };
}

// evaluate.world_reject: the openings and whatever lies outside the outline are obstacles, and the test is the
// separating-axis one on what each part really covers
export function refuseOverlaps(wall, obstacles, news, tol = OVERLAP_TOL, poly = null, present = []) {
  const parts = elementPolys(present);
  const obs = [...obstacles.map(rectPoly), ...(poly ? polyOutside(poly).map(rectPoly) : []), ...parts];
  const kept = [];
  for (const e of news) {
    const m = elementMember(e);
    let q = padPoly(memberPoly(m));
    if (cutsOf(e[0])) q = cutPoly(q, cutsOf(e[0]), m, parts);   // sawn on what is already there, then tested
    const r = [Math.min(...q.map(p => p[0])), Math.min(...q.map(p => p[1])),
      Math.max(...q.map(p => p[0])), Math.max(...q.map(p => p[1]))];
    const inside = r[0] >= wall[0] - tol && r[1] >= wall[1] - tol && r[2] <= wall[2] + tol && r[3] <= wall[3] + tol;
    if (inside && !obs.some(o => polysOverlap(q, o, tol))) { kept.push(e); obs.push(q); parts.push(q); }
  }
  return { kept, refused: news.length - kept.length };
}

// ---------------------------------------------------------------- a part as the scripts place it, in this language
export function segPart(item, cls, p0, p1, label, cutBits = 0) {
  return { item, cls, cuts: cutBits, p0: [f32(p0[0]), f32(p0[1])], p1: [f32(p1[0]), f32(p1[1])], label };
}

export function partElement(p) {                   // dataset.wall_members + sequence.elements_of
  const { X_MAX, Y_MAX } = RULER;
  const e = [codeOf(ITEMS.indexOf(p.item), p.cls, p.cuts | 0), toBin(p.p0[0], X_MAX, RULER.NX), toBin(p.p0[1], Y_MAX, RULER.NY),
    toBin(p.p1[0], X_MAX, RULER.NX), toBin(p.p1[1], Y_MAX, RULER.NY)];
  return closeElement(e);
}

export function partPolys(parts) {                 // what they cover, cuts taken off, for the clean-up and the drawing
  return elementPolys(parts.map(partElement));
}

// encode.object_segment: a placed member's reference edge, the axis canonicalised so the same member reads one way
export function memberSeg(item, ang, x, y, L) {
  let u = [Math.cos(ang), Math.sin(ang)], flip = false;
  if (u[0] < -1e-9 || (Math.abs(u[0]) <= 1e-9 && u[1] < 0)) { u = [-u[0], -u[1]]; flip = true; }
  const n = normalOf(u), t = thickOf(item), c = [x - n[0] * t / 2, y - n[1] * t / 2];
  return { flip, cls: Math.abs(u[1]) <= 1e-9 ? 0 : (Math.abs(u[0]) <= 1e-9 ? 1 : 2),
    p0: [c[0] - u[0] * L / 2, c[1] - u[1] * L / 2], p1: [c[0] + u[0] * L / 2, c[1] + u[1] * L / 2] };
}

export function placed(item, ang, x, y, L, label, cutBits = 0) {
  const s = memberSeg(item, ang, x, y, L);          // the axis is canonicalised, and the two ends swap with it
  const c = s.flip ? ((cutBits & CUT_LO) << 1) | ((cutBits & CUT_HI) >> 1) : cutBits;
  return segPart(item, s.cls, s.p0, s.p1, label, c);
}

// ---------------------------------------------------------------- the gable script (automake/mvp/gable_framer.py)
const snap = v => rint(v / Q) * Q;

export function gablePoly(L, H, ridgeX, eaves) {   // gable_framer.gable_poly, in the wall frame
  const hx = L / 2, hy = H / 2, ex = eaves - hy;
  const p = [[-hx, -hy], [hx, -hy], [hx, ex], [ridgeX - hx, hy], [-hx, ex]];
  // a ridge standing on an end corner leaves that corner in line with its neighbours: it is no corner, and the
  // outline is a quadrilateral with one sloping top edge (gable_framer._straight)
  const straight = (a, b2, c) => Math.abs((b2[0] - a[0]) * (c[1] - a[1]) - (b2[1] - a[1]) * (c[0] - a[0])) <= 1e-9;
  return p.filter((v, i) => !straight(p[(i + p.length - 1) % p.length], v, p[(i + 1) % p.length]));
}

export function ridgeY(L, eaves, pitch, ridgeX) {  // GableSpec.ridge_y
  return eaves + Math.max(ridgeX, L - ridgeX) * Math.tan(pitch * Math.PI / 180);
}

export function rakeSegments(poly) {               // gable_framer.rake_segments: the upper boundary, left to right
  let iLo = 0;
  poly.forEach((p, i) => { if (p[0] + 1e-6 * p[1] < poly[iLo][0] + 1e-6 * poly[iLo][1]) iLo = i; });
  const p = poly.map((_, i) => poly[(i + iLo) % poly.length]);
  const top = [];
  for (let i = 0; i < p.length; i++) {
    const a = p[i], b = p[(i + 1) % p.length];
    if (b[0] < a[0] - 1e-9) top.push([b.slice(), a.slice()]);
  }
  top.sort((s, t) => s[0][0] - t[0][0]);
  return top;
}

function lineAt(segs, x, reach = 0.1) {            // gable_framer._line_at: held level beyond a segment's own ends
  let out = -Infinity;
  for (const [a, b] of segs) {
    const lo = Math.min(a[0], b[0]), hi = Math.max(a[0], b[0]);
    if (hi - lo < 1e-12 || x < lo - reach || x > hi + reach) continue;
    out = Math.max(out, a[1] + (Math.min(Math.max(x, lo), hi) - a[0]) * (b[1] - a[1]) / (b[0] - a[0]));
  }
  return out;
}

export function gableWall(sc, arange, openingsLocal) {
  const t = 0.045, spacing = 0.6, hx = sc.hx, hy = sc.hy, parts = [], under = [];
  const segs = rakeSegments(sc.poly);
  const put = (item, ang, x, y, L, label, cutBits = 0) => {
    if (L > 0.01) parts.push(placed(item, ang, x, y, L, label, cutBits));
  };
  const putSegment = (item, a, b, label) => {
    const p = [snap(a[0]), snap(a[1])], q = [snap(b[0]), snap(b[1])];
    const L = Math.hypot(q[0] - p[0], q[1] - p[1]);
    if (L <= 0.01) return null;
    const u = [(q[0] - p[0]) / L, (q[1] - p[1]) / L], n = normalOf(u), th = thickOf(item);
    put(item, Math.atan2(u[1], u[0]), (p[0] + q[0]) / 2 + n[0] * th / 2, (p[1] + q[1]) / 2 + n[1] * th / 2, L, label);
    return [p, q];
  };
  // up to the HIGHEST point of the plate's underside over the stud's own width, so the box runs into the plate: the
  // stud declares that end cut and the world saws it flush (gable_framer.top_of_stud)
  const topOfStud = x => snap(Math.max(lineAt(under, x - t / 2), lineAt(under, x), lineAt(under, x + t / 2)));

  const ops = openingsLocal(sc);
  const yBot = -hy + t;
  let a = -hx;
  for (const op of ops.filter(o => o[2] <= -hy + 1e-6)) {
    put("2x4", 0, 0.5 * (a + op[0]), -hy + t / 2, op[0] - a, "bottom plate");
    a = op[1];
  }
  put("2x4", 0, 0.5 * (a + hx), -hy + t / 2, hx - a, "bottom plate");
  segs.forEach(([A, B], k) => {
    const L = Math.hypot(B[0] - A[0], B[1] - A[1]);
    const u = [(B[0] - A[0]) / L, (B[1] - A[1]) / L], ang = Math.atan2(u[1], u[0]);
    const n = [u[1], -u[0]], back = t * Math.abs(Math.tan(ang));
    const a0 = k > 0 ? [A[0] + u[0] * back, A[1] + u[1] * back] : A.slice();
    const b0 = k + 1 < segs.length ? [B[0] - u[0] * back, B[1] - u[1] * back] : B.slice();
    const laid = putSegment("2x4", [a0[0] + n[0] * t, a0[1] + n[1] * t], [b0[0] + n[0] * t, b0[1] + n[1] * t], "rake plate");
    if (laid) under.push(laid);
  });
  const xs = arange(-hx + t / 2, hx - t / 2, spacing);
  if (xs[xs.length - 1] < hx - 1.5 * t) xs.push(hx - t / 2);   // the stud at the far end, unless the run put one there
  for (const x of xs) {
    if (ops.some(op => op[0] - 2 * t < x && x < op[1] + 2 * t)) continue;
    const yTop = topOfStud(x);
    put("2x4", Math.PI / 2, x, 0.5 * (yBot + yTop), yTop - yBot, "stud", CUT_HI);
  }
  for (const [x0, x1, y0, y1] of ops) {
    for (const x of [x0 - 1.5 * t, x1 + 1.5 * t]) {
      const yTop = topOfStud(x);
      put("2x4", Math.PI / 2, x, 0.5 * (yBot + yTop), yTop - yBot, "king stud", CUT_HI);
    }
    for (const x of [x0 - t / 2, x1 + t / 2]) put("2x4", Math.PI / 2, x, 0.5 * (yBot + y1), y1 - yBot, "jack stud");
    const hd = 0.195;
    put("2x8", 0, 0.5 * (x0 + x1), y1 + hd / 2, x1 - x0 + 2 * t, "header");
    for (const x of xs) if (x0 < x && x < x1) {
      const yTop = topOfStud(x);
      put("2x4", Math.PI / 2, x, 0.5 * (y1 + hd + yTop), yTop - (y1 + hd), "cripple", CUT_HI);
    }
    if (y0 > yBot + 0.05) {
      put("2x4", 0, 0.5 * (x0 + x1), y0 - t / 2, x1 - x0, "sill");
      for (const x of xs) if (x0 < x && x < x1 && y0 - t - yBot > 0.05)
        put("2x4", Math.PI / 2, x, 0.5 * (yBot + y0 - t), y0 - t - yBot, "cripple");
    }
  }
  return parts;
}

// build_wall's clean-up for parts of any angle: the world's separating-axis penetration, later part first
export function dropOverlapsPoly(sc, parts, tol = 0.003) {
  const polys = partPolys(parts), drop = new Set();
  const deeper = (a, b) => polysOverlap(a, b, tol);
  for (let i = 0; i < polys.length; i++) for (let j = i + 1; j < polys.length; j++) if (deeper(polys[i], polys[j])) drop.add(j);
  const keep = sc.openings.map(o => rectPoly([o.cx - o.w / 2, o.cy - o.h / 2, o.cx + o.w / 2, o.cy + o.h / 2]));
  polys.forEach((p, i) => { if (keep.some(k => deeper(p, k))) drop.add(i); });
  return { kept: parts.filter((_, i) => !drop.has(i)), dropped: drop.size };
}
