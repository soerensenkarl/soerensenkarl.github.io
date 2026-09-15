// Drawing for the M0 demo: walls, parts, the network's input tokens and a sketch of the network, as SVG markup.
// Walls are drawn with the origin at the wall's bottom-left corner, x along the wall, y up (metres).
import { ITEMS, SECTIONS } from "./wall.js";

// the platform's colours (automake/app/elevation.py): framing timber by section, concrete block walls by piece
export const ITEM_COLOURS = { "2x4": "rgb(214,166,92)", "2x6": "rgb(190,130,70)", "2x8": "rgb(165,105,55)", "2x10": "rgb(140,80,45)", "2x12": "rgb(110,60,35)" };
export const BLOCK_FILL = { block: "#8d8a82", "cut block": "#c9c5bb", lintel: "#5d6b7a" };
export const BLOCK_STROKE = { block: "#4f4c46", "cut block": "#8a857a", lintel: "#2f3a45" };
export const WRONG_STROKE = "#94766d";
export const FIELD_NAMES = ["item", "x0", "y0", "x1", "y1"];
export const FIELD_WORDS = ["item", "left edge", "bottom edge", "right edge", "top edge"];
const n4 = v => +v.toFixed(4);

export function partLabel(e) {
  const item = ITEMS[e[0]];
  return item === "block" && e[3] - e[1] < 78 ? "cut block" : item;       // a whole block is 78 ticks (390 mm) long
}
export function partStyle(e) {
  const item = ITEMS[e[0]];
  if (item === "block" || item === "lintel") { const l = partLabel(e); return { fill: BLOCK_FILL[l], stroke: BLOCK_STROKE[l] }; }
  return { fill: ITEM_COLOURS[item] || "grey", stroke: "#7a5230" };
}
function tint(c, t) {                                   // elevation.py WRONG_JS partTint: the part's own colour, a soft red
  let r = 150, g = 145, b = 138;
  const m = /rgba?\((\d+),\s*(\d+),\s*(\d+)/.exec(c || "");
  if (m) { r = +m[1]; g = +m[2]; b = +m[3]; } else if (/^#[0-9a-f]{6}$/i.test(c || "")) { r = parseInt(c.slice(1, 3), 16); g = parseInt(c.slice(3, 5), 16); b = parseInt(c.slice(5, 7), 16); }
  const mix = (v, w) => Math.round(v * (1 - t) + w * t);
  return `rgb(${mix(r, 200)},${mix(g, 105)},${mix(b, 90)})`;
}
export function wrongStyle(e) { return { fill: tint(partStyle(e).fill, 0.14), stroke: WRONG_STROKE }; }

// an element [item, x0, y0, x1, y1] (ticks) as a rectangle from the wall's bottom-left corner
export function elemBox(e, L, H) {
  return [e[1] * 0.005 - 4 + L / 2, e[2] * 0.005 - 1.6 + H / 2, e[3] * 0.005 - 4 + L / 2, e[4] * 0.005 - 1.6 + H / 2];
}
export function openingBox(o) { const s = o.kind === "door" ? 0 : o.sill; return [o.x, s, o.x + o.w, s + o.h]; }

export function viewBox(L, H, pad = 0.2) {
  return `${n4(-pad)} ${n4(-(H + pad))} ${n4(L + 2 * pad)} ${n4(H + 2 * pad + 0.12)}`;
}
const rect = (b, attrs) => `<rect x="${n4(b[0])}" y="${n4(-b[3])}" width="${n4(Math.max(0, b[2] - b[0]))}" height="${n4(Math.max(0, b[3] - b[1]))}" ${attrs}/>`;

// the wall with its openings, parts ({box, cls, fill, stroke, title}), marks (lines) and drag handles
export function wallMarkup({ L, H, openings, parts = [], lines = [], handles = false, note = "" }) {
  const out = [rect([0, 0, L, H], 'class="wallbg"')];
  openings.forEach(o => out.push(rect(openingBox(o), 'class="opening"')));
  for (const p of parts) out.push(rect(p.box, `class="part ${p.cls || ""}" style="fill:${p.fill};stroke:${p.stroke}"${p.title ? ` data-tip="${p.title}"` : ""}`));
  for (const l of lines) out.push(`<line class="${l.cls}" x1="${n4(l.x1)}" y1="${n4(-l.y1)}" x2="${n4(l.x2)}" y2="${n4(-l.y2)}"/>`);
  out.push(`<text class="dim" x="${n4(L / 2)}" y="0.14" text-anchor="middle">${L.toFixed(2)} m × ${H.toFixed(2)} m</text>`);
  if (note) out.push(`<text class="note" x="${n4(L / 2)}" y="${n4(-H / 2)}" text-anchor="middle">${note}</text>`);
  if (handles) {
    const t = Math.max(0.06, Math.min(0.12, L / 60));
    openings.forEach((o, i) => {
      const [x0, y0, x1, y1] = openingBox(o);
      out.push(rect([x0 + t, y0 + t, x1 - t, y1 - t], `class="h move" data-h="op:${i}:move"`));
      out.push(rect([x0 - t, y0, x0 + t, y1], `class="h ew" data-h="op:${i}:l"`));
      out.push(rect([x1 - t, y0, x1 + t, y1], `class="h ew" data-h="op:${i}:r"`));
      out.push(rect([x0 + t, y1 - t, x1 - t, y1 + t], `class="h ns" data-h="op:${i}:t"`));
      if (o.kind !== "door") out.push(rect([x0 + t, y0 - t, x1 - t, y0 + t], `class="h ns" data-h="op:${i}:b"`));
      out.push(rect([x0 + (x1 - x0) / 2 - 0.12, y1 - 0.02, x0 + (x1 - x0) / 2 + 0.12, y1 + 0.02], 'class="grip"'));
      out.push(rect([x1 - 0.02, y0 + (y1 - y0) / 2 - 0.12, x1 + 0.02, y0 + (y1 - y0) / 2 + 0.12], 'class="grip"'));
    });
    out.push(rect([L - t, 0, L + t, H], 'class="h ew" data-h="wall:0:r"'));
    out.push(rect([0, H - t, L - t, H + t], 'class="h ns" data-h="wall:0:t"'));
    out.push(rect([L - 0.02, H / 2 - 0.2, L + 0.02, H / 2 + 0.2], 'class="grip"'));
    out.push(rect([L / 2 - 0.2, H - 0.02, L / 2 + 0.2, H + 0.02], 'class="grip"'));
  }
  return out.join("");
}

// the network's input for one pass: every box token and the canvas patches, in the same wall drawing
export function tokensMarkup({ L, H, tokens, cover, showFree, showCanvas, lines = [] }) {
  const out = [rect([0, 0, L, H], 'class="wallbg"')];
  const sx = L / 2, sy = H / 2;
  if (showCanvas && cover) {
    for (let row = 0, g = 0; row < 8; row++) for (let col = 0; col < 20; col++, g++) {
      const b = [col * 0.4 - 4 + sx, row * 0.4 - 1.6 + sy, (col + 1) * 0.4 - 4 + sx, (row + 1) * 0.4 - 1.6 + sy];
      if (b[2] <= 0 || b[0] >= L || b[3] <= 0 || b[1] >= H) continue;
      const cp = Math.min(1, cover[g * 3]), co = Math.min(1, cover[g * 3 + 1]);
      out.push(rect(b, `style="fill:rgba(var(--grid),${n4(0.04 + 0.42 * cp)});stroke:rgba(var(--grid),0.28);stroke-width:0.6;vector-effect:non-scaling-stroke"`));
      if (co > 0.02) out.push(rect(b, `style="fill:rgba(128,128,128,${n4(0.25 * co)});stroke:none"`));
    }
  }
  tokens.rects.forEach((r, i) => {
    const t = tokens.types[i];
    if (t === 0) return;
    const b = [r[0] + sx, r[1] + sy, r[2] + sx, r[3] + sy];
    if (t === 1) out.push(rect(b, 'class="opening"'));
    else if (t === 2) out.push(rect(b, 'style="fill:rgba(128,128,128,0.18);stroke:currentColor;stroke-width:0.7;vector-effect:non-scaling-stroke;color:var(--ink2)"'));
    else if (t === 3 && showFree) out.push(rect([b[0] + 0.012, b[1] + 0.012, b[2] - 0.012, b[3] - 0.012],
      'style="fill:rgba(47,158,107,0.07);stroke:var(--free);stroke-width:1.1;stroke-dasharray:4 2;vector-effect:non-scaling-stroke"'));
  });
  for (const l of lines) out.push(`<line class="${l.cls}" x1="${n4(l.x1)}" y1="${n4(-l.y1)}" x2="${n4(l.x2)}" y2="${n4(-l.y2)}"/>`);
  return out.join("");
}

// a sketch of the network; parts with ids light up as it reads and writes
export const NET_SVG = `<svg class="netsvg" viewBox="0 0 400 178" role="img" aria-label="The network: boxes in, encoder, decoder, five picks per part">
<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="currentColor" style="color:var(--ink3)"/></marker></defs>
<rect class="blk" id="nd-in" x="4" y="14" width="92" height="58" rx="6"/><text x="50" y="34" text-anchor="middle">Boxes in</text>
<text class="sub" x="50" y="48" text-anchor="middle">wall, openings, parts,</text><text class="sub" x="50" y="60" text-anchor="middle">free space, 40 cm grid</text>
<path class="arrow" d="M96,43 H122"/>
<rect class="blk" id="nd-enc" x="124" y="14" width="96" height="58" rx="6"/><text x="172" y="36" text-anchor="middle">Encoder</text>
<text class="sub" x="172" y="50" text-anchor="middle">4 layers, d = 256</text><text class="sub" x="172" y="62" text-anchor="middle">every box sees every box</text>
<path class="arrow" d="M220,43 H246"/>
<rect class="blk" id="nd-dec" x="248" y="14" width="148" height="58" rx="6"/><text x="322" y="36" text-anchor="middle">Decoder</text>
<text class="sub" x="322" y="50" text-anchor="middle">4 layers, one pick per step,</text><text class="sub" x="322" y="62" text-anchor="middle">reads the boxes and its picks</text>
<path class="arrow" d="M322,72 V92"/>
<g id="nd-heads">
<rect class="head" id="nd-h0" x="180" y="96" width="44" height="22" rx="4"/><text x="202" y="111" text-anchor="middle">item</text>
<rect class="head" id="nd-h1" x="228" y="96" width="40" height="22" rx="4"/><text x="248" y="111" text-anchor="middle">x0</text>
<rect class="head" id="nd-h2" x="272" y="96" width="40" height="22" rx="4"/><text x="292" y="111" text-anchor="middle">y0</text>
<rect class="head" id="nd-h3" x="316" y="96" width="40" height="22" rx="4"/><text x="336" y="111" text-anchor="middle">x1</text>
<rect class="head" id="nd-h4" x="360" y="96" width="36" height="22" rx="4"/><text x="378" y="111" text-anchor="middle">y1</text>
</g>
<text class="sub" x="4" y="104">each part: 5 picks</text><text class="sub" x="4" y="116">(STOP ends the wall)</text>
<rect class="blk" id="nd-anchor" x="4" y="130" width="392" height="44" rx="6"/>
<text x="12" y="147">An edge pick</text><text class="sub" x="12" y="162">= gate · (copy a visible edge + offset) + (1 − gate) · a tick on the 5 mm ruler</text>
<path class="arrow" d="M300,118 V128"/>
</svg>`;

export function highlightNet(root, stage, field) {
  const set = (id, on) => { const el = root.querySelector("#" + id); if (el) el.classList.toggle("on", !!on); };
  set("nd-in", stage === "encode"); set("nd-enc", stage === "encode"); set("nd-dec", stage === "write");
  for (let f = 0; f < 5; f++) set("nd-h" + f, stage === "write" && field === f);
  set("nd-anchor", stage === "write" && field > 0);
}

// the pick's distribution around the chosen tick (±40 ticks), with the source tick marked
export function distMarkup(pick) {
  if (!pick || !pick.win) return "";
  const [w0, vals] = pick.win, n = vals.length, max = Math.max(...vals, 1e-9), bw = 400 / 81;
  const bars = vals.map((p, i) => { const h = Math.max(0.5, 46 * p / max); return `<rect x="${n4(i * bw)}" y="${n4(50 - h)}" width="${n4(bw * 0.8)}" height="${n4(h)}"${w0 + i === pick.v ? ' class="on"' : ""}/>`; });
  const src = pick.src && !pick.fromRuler ? pick.src.tick - w0 : null;
  const mark = src !== null && src >= 0 && src < n ? `<line x1="${n4((src + 0.4) * bw)}" x2="${n4((src + 0.4) * bw)}" y1="0" y2="52"/>` : "";
  return `<svg class="dist" viewBox="0 0 400 54" preserveAspectRatio="none">${bars.join("")}${mark}</svg>`;
}
