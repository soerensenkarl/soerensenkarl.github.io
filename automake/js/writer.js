// How M0 writes a wall: pass by pass (up to 16 parts each) until a pass writes nothing, the parts of each pass added to the
// wall before the next; `loads` are the loads on the top edge, each a segment [x0, x1] in the wall frame (a point load
// is x0 = x1, a line load spans [x0, x1]); `cell` is the template cell the wall is asked to follow, as (x0, x1, course)
// in metres, and `inv` the items the design may be built from, as indices into ITEMS - each of the three read only by a
// network built with it. With `reject`, the world first refuses every new part that overlaps something or leaves the wall
// (automake/mvp/evaluate.py: the passes loop and world_reject). An async generator of events, for the worker and the tests;
// its return value is {final, passes, term}.
import { api } from "./wall.js";
import { PROF, profAdd } from "./model.js";

export const MAX_PASSES = 40;

export async function* writeWall(model, { wall, ops, start = [], brief, reject = false, maxPasses = MAX_PASSES, detail = true,
  pool = null, split = 1, loads = [], cell = null, inv = null, skin = null, poly = null }) {
  const A = api(model.segment ? "segment" : "rect");     // the token format the network speaks (wall.js api)
  let cur = start.map(e => e.slice());
  for (let p = 0; p < maxPasses; p++) {
    const t0 = performance.now();
    const tok = A.segment ? A.tokens(skin, cur, loads, cell, inv) : A.tokens(wall, ops, cur, loads);
    if (PROF.on) profAdd("tokens + free space (JS)", performance.now() - t0);
    yield { type: "encode", pass: p, tokens: tok, present: cur.length };
    const enc = await model.encode(tok, brief, pool, split);
    yield { type: "encoded", pass: p, cover: enc.cover, bins: enc.bins };
    const written = [];
    for (const part of model.writePass(enc, detail)) {
      written.push(part.elem);
      yield { type: "part", pass: p, j: part.j, elem: part.elem, picks: part.picks };
    }
    let kept = written, refused = 0;
    if (reject && written.length) ({ kept, refused } = A.segment
      ? A.refuseOverlaps(wall, ops, written, undefined, poly, cur)
      : A.refuseOverlaps(wall, [...ops, ...cur.map(A.elementRect)], written));
    yield { type: "pass", pass: p, written, kept, refused };
    if (!kept.length) return { final: cur, passes: p + 1, term: refused ? "refused" : "converged" };
    cur = cur.concat(kept);
  }
  return { final: cur, passes: maxPasses, term: "budget" };
}
