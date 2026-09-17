// A second view of the same wall. The network computes in 2D - a piece of timber is 45 mm wide and the drawing never
// shows how deep it is - so here every part it writes is extruded to its true section: the depth the inventory gives
// the item (a 2x4 is 45 x 95, a 2x8 45 x 195, a block 190 x 190), across the wall, on the midplane the network sees.
// The wall is built part by part exactly as the 2D view builds it, with the same blue flash on each new part.
//
// The 2D drawing stays the editing surface; here the wall only turns. three.js (UMD, pinned) is fetched the first time
// this view is opened and never again, so an artifact that does not offer 3D pays nothing for it.
import { ITEMS, SECTIONS } from "./wall.js";

const SRC = "https://cdnjs.cloudflare.com/ajax/libs/three.js/0.159.0/three.min.js";
const BLUE = 0x3b8cff;
const CAP = 640;                                         // instances per material: more parts than any wall has
const LARROW = 0.30, LFOOT = 0.045;                      // the load array, at the heights the 2D view draws it

let T = null;                                            // the library, once
const loadThree = () => new Promise((res, rej) => {
  if (window.THREE) return res(window.THREE);
  const s = document.createElement("script");
  s.src = SRC;
  s.onload = () => (window.THREE ? res(window.THREE) : rej(new Error("three.js")));
  s.onerror = () => rej(new Error("three.js"));
  document.head.appendChild(s);
});

// pine, a few faint streaks along the length of the piece (u), as the 2D view's grain: warm, matt, not photographic
function woodTexture() {
  const c = document.createElement("canvas");
  c.width = 256; c.height = 64;
  const g = c.getContext("2d");
  g.fillStyle = "#dcb87e"; g.fillRect(0, 0, 256, 64);
  let r = 20250916;
  const rnd = () => (r = (r * 9301 + 49297) % 233280) / 233280;
  for (let k = 0; k < 26; k++) {
    g.strokeStyle = `rgba(112,70,26,${0.07 + rnd() * 0.15})`;
    g.lineWidth = 0.5 + rnd() * 1.5;
    const y0 = rnd() * 64, amp = 0.8 + rnd() * 3, ph = rnd() * 6.3;
    g.beginPath();
    for (let x = 0; x <= 256; x += 8) { const y = y0 + Math.sin(x / 33 + ph) * amp; x ? g.lineTo(x, y) : g.moveTo(x, y); }
    g.stroke();
  }
  const t = new T.CanvasTexture(c);
  t.wrapS = t.wrapT = T.RepeatWrapping;
  if ("colorSpace" in t) t.colorSpace = T.SRGBColorSpace;
  return t;
}

// an element as a box: its length, the face the drawing shows, and the depth the drawing cannot show. The section says
// which is which - the in-plane dimension is one of the item's two, so the other one goes through the wall.
function boxOf(e, b) {
  const [w, d] = SECTIONS[ITEMS[e[0]]];
  const dx = b[2] - b[0], dy = b[3] - b[1], up = dy > dx;
  const len = up ? dy : dx, face = up ? dx : dy;
  return { x: (b[0] + b[2]) / 2, y: (b[1] + b[3]) / 2, len, face, deep: Math.abs(face - d) < Math.abs(face - w) ? w : d, up };
}
const kindOf = e => (ITEMS[e[0]] === "block" ? "block" : ITEMS[e[0]] === "lintel" ? "lintel" : "timber");

export async function init(canvas) {
  T = await loadThree();
  const renderer = new T.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setClearAlpha(0);
  const scene = new T.Scene();
  const camera = new T.PerspectiveCamera(24, 2.3, 0.05, 120);

  // soft daylight: a bright sky, one low sun from the left and a weak fill, so a stud's two faces read apart
  scene.add(new T.HemisphereLight(0xf4f7ff, 0xd6cec2, 1.5));
  const sun = new T.DirectionalLight(0xfffaf0, 3.0); sun.position.set(-5, 6, 8); scene.add(sun);
  const fill = new T.DirectionalLight(0xffffff, 0.55); fill.position.set(6, 1, -6); scene.add(fill);

  const box = new T.BoxGeometry(1, 1, 1);
  const bin = mat => {
    const m = new T.InstancedMesh(box, mat, CAP);
    m.count = 0; m.frustumCulled = false; m.instanceMatrix.setUsage(T.DynamicDrawUsage);
    scene.add(m); return m;
  };
  const bins = {
    timber: bin(new T.MeshStandardMaterial({ map: woodTexture(), roughness: 0.88, metalness: 0 })),
    block: bin(new T.MeshStandardMaterial({ color: 0xcccdca, roughness: 0.97, metalness: 0 })),
    lintel: bin(new T.MeshStandardMaterial({ color: 0x9c9c9a, roughness: 0.97, metalness: 0 })),
  };

  // the skin: a faint pane of the wall with its openings cut out of it, just behind the parts
  const skin = new T.Group(); scene.add(skin);
  const skinMat = new T.MeshBasicMaterial({ color: 0xf0eee9, transparent: true, opacity: 0.32, side: T.DoubleSide, depthWrite: false });
  const edgeMat = new T.LineBasicMaterial({ color: 0xc2c0bb, transparent: true, opacity: 0.85 });

  // the load array: the line, the shafts, and a cone on each arrow
  const loads = new T.Group(); scene.add(loads);
  const blueLine = new T.LineBasicMaterial({ color: BLUE });
  const blueSolid = new T.MeshBasicMaterial({ color: BLUE });
  const cone = new T.ConeGeometry(0.036, 0.1, 10).rotateX(Math.PI).translate(0, -0.05, 0);
  const tips = new T.InstancedMesh(cone, blueSolid, 16); tips.count = 0; tips.frustumCulled = false; loads.add(tips);
  let shafts = null;

  // the part just written, lit from within: a blue skin over the box and its edges, fading as the 2D outline fades
  const hotMat = new T.MeshBasicMaterial({ color: BLUE, transparent: true, depthWrite: false });
  const hotBox = new T.Mesh(box, hotMat); hotBox.visible = false; hotBox.frustumCulled = false; scene.add(hotBox);
  const hotEdgeMat = new T.LineBasicMaterial({ color: BLUE, transparent: true });
  const hotEdge = new T.LineSegments(new T.EdgesGeometry(box), hotEdgeMat);
  hotEdge.visible = false; hotEdge.frustumCulled = false; scene.add(hotEdge);
  hotBox.matrixAutoUpdate = hotEdge.matrixAutoUpdate = false;

  const m4 = new T.Matrix4(), pos = new T.Vector3(), scl = new T.Vector3(), rot = new T.Quaternion();
  const UP = new T.Vector3(0, 0, 1), target = new T.Vector3();
  // a part's place in the world: its length along its own axis, its drawn face, and its depth through the wall
  const boxMatrix = (b, grow) => {
    pos.set(b.x, b.y, 0);
    rot.setFromAxisAngle(UP, b.up ? Math.PI / 2 : 0);
    scl.set(b.len + grow, b.face + grow, b.deep + grow);
    return m4.compose(pos, rot, scl);
  };

  let theta = -0.52, phi = 0.17, view = { L: 5, H: 2.7, top: 2.8 };
  function place() {
    const lo = -0.08, hi = view.top, cy = (lo + hi) / 2;
    const halfW = Math.max(view.L, 3.0) / 2 * 1.14, halfH = (hi - lo) / 2 * 1.12;
    const tanV = Math.tan(camera.fov * Math.PI / 360);
    const d = Math.max(halfH / tanV, halfW / (tanV * Math.max(0.6, camera.aspect))) + 0.55;
    target.set(view.L / 2, cy, 0);
    camera.position.set(target.x + d * Math.sin(theta) * Math.cos(phi), target.y + d * Math.sin(phi),
      target.z + d * Math.cos(theta) * Math.cos(phi));
    camera.lookAt(target);
  }

  function resize() {
    const w = Math.max(1, canvas.clientWidth), h = Math.max(1, canvas.clientHeight);
    const dpr = Math.min(2, devicePixelRatio || 1);
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      renderer.setPixelRatio(dpr);
      renderer.setSize(w, h, false);
    }
    if (camera.aspect !== w / h) { camera.aspect = w / h; camera.updateProjectionMatrix(); }
  }

  // the wall's pane and the holes in it, rebuilt only when the design changes
  let skinKey = "";
  function setSkin(L, H, openings, back) {
    const key = `${L}|${H}|${back}|${openings.map(o => [o.x, o.w, o.sill, o.h].join()).join("~")}`;
    if (key === skinKey) return;
    skinKey = key;
    skin.clear();
    const shape = new T.Shape([[0, 0], [L, 0], [L, H], [0, H]].map(([x, y]) => new T.Vector2(x, y)));
    const rects = [[0, 0, L, H]];
    for (const o of openings) {
      const y0 = o.kind === "door" ? 0 : o.sill, r = [o.x, y0, o.x + o.w, y0 + o.h];
      rects.push(r);
      shape.holes.push(new T.Path([[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]].map(([x, y]) => new T.Vector2(x, y))));
    }
    const pane = new T.Mesh(new T.ShapeGeometry(shape), skinMat);
    pane.position.z = -back;
    skin.add(pane);
    for (const r of rects) {
      const pts = [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]], [r[0], r[1]]]
        .map(([x, y]) => new T.Vector3(x, y, -back + 0.001));
      skin.add(new T.Line(new T.BufferGeometry().setFromPoints(pts), edgeMat));
    }
  }

  let loadKey = "";
  function setLoads(d, on) {
    const key = on ? `${d.L}|${d.H}|${d.loads.join()}` : "";
    if (key === loadKey) return;
    loadKey = key;
    loads.visible = !!on;
    if (shafts) { shafts.geometry.dispose(); loads.remove(shafts); shafts = null; }
    tips.count = 0;
    if (!on) return;
    const yTail = d.H + LARROW, yTip = d.H + LFOOT;
    const pts = [new T.Vector3(0, yTail, 0), new T.Vector3(d.L, yTail, 0)];
    d.loads.forEach((x, i) => {
      pts.push(new T.Vector3(x, yTail, 0), new T.Vector3(x, yTip + 0.08, 0));
      m4.compose(pos.set(x, yTip + 0.1, 0), rot.identity(), scl.set(1, 1, 1));
      tips.setMatrixAt(i, m4);
    });
    tips.count = Math.min(16, d.loads.length);
    tips.instanceMatrix.needsUpdate = true;
    shafts = new T.LineSegments(new T.BufferGeometry().setFromPoints(pts), blueLine);
    shafts.frustumCulled = false;
    loads.add(shafts);
  }

  function draw(st) {
    const { design, parts, hot, ebox, showLoads } = st;
    resize();
    const deep = design.script === "block" ? 0.19 : 0.095;
    setSkin(design.L, design.H, design.openings, deep / 2 + 0.004);
    setLoads(design, showLoads);
    view = { L: design.L, H: design.H, top: showLoads ? design.H + LARROW + 0.18 : design.H + 0.14 };

    const n = { timber: 0, block: 0, lintel: 0 };
    for (const e of parts) {
      const k = kindOf(e);
      bins[k].setMatrixAt(n[k]++, boxMatrix(boxOf(e, ebox(e)), 0));
    }
    for (const k of Object.keys(bins)) { bins[k].count = Math.min(CAP, n[k]); bins[k].instanceMatrix.needsUpdate = true; }

    hotBox.visible = hotEdge.visible = !!hot;
    if (hot) {
      hotBox.matrix.copy(boxMatrix(boxOf(hot.e, ebox(hot.e)), 0.012));
      hotEdge.matrix.copy(hotBox.matrix);
      hotBox.matrixWorldNeedsUpdate = hotEdge.matrixWorldNeedsUpdate = true;
      hotMat.opacity = 0.38 * hot.a;
      hotEdgeMat.opacity = hot.a;
    }
    place();
    renderer.render(scene, camera);
  }

  // the wall only turns here: drag anywhere on it, with a finger or a mouse
  let drag = null;
  canvas.style.cursor = "grab";
  canvas.addEventListener("pointerdown", e => {
    drag = { x: e.clientX, y: e.clientY };
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = "grabbing";
    e.preventDefault();
  });
  canvas.addEventListener("pointermove", e => {
    if (!drag) return;
    theta -= (e.clientX - drag.x) * 0.006;
    phi = Math.min(1.15, Math.max(-0.22, phi + (e.clientY - drag.y) * 0.005));
    drag = { x: e.clientX, y: e.clientY };
    place();
    renderer.render(scene, camera);
  });
  const up = e => { if (drag) { drag = null; canvas.style.cursor = "grab"; try { canvas.releasePointerCapture(e.pointerId); } catch {} } };
  canvas.addEventListener("pointerup", up);
  canvas.addEventListener("pointercancel", up);

  return { draw };
}
