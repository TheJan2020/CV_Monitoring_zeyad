// Bbox labeling page: load the class's image list, render the current
// image, let the user draw rectangles, save in YOLO normalized format,
// keyboard-driven prev/next workflow.

const CID = window.__CLASS_ID;
const START_IMG = window.__START_IMG || "";

const img       = document.getElementById("cl-img");
const canvas    = document.getElementById("cl-canvas");
const stage     = document.getElementById("cl-stage");
const ctx       = canvas.getContext("2d");
const titleEl   = document.getElementById("cl-title");
const classNameEl = document.getElementById("cl-class-name");
const progressEl  = document.getElementById("cl-progress");
const boxCountEl  = document.getElementById("cl-box-count");
const dimsEl      = document.getElementById("cl-dims");
const saveStatus  = document.getElementById("cl-save-status");

const btnPrev = document.getElementById("cl-prev");
const btnNext = document.getElementById("cl-next");
const btnNextUnlabeled = document.getElementById("cl-next-unlabeled");
const btnClear = document.getElementById("cl-clear");
const btnSave  = document.getElementById("cl-save");

let className = "";
let modelReady = false;
let images = [];            // [{id, labeled, box_count, ...}, ...]
let cursor = -1;            // index into images
let curImgNatural = { w: 0, h: 0 };

// boxes are stored as YOLO-normalized {cx, cy, w, h} ∈ [0,1], plus
// a transient pixel-rect cache rebuilt on every render so we don't
// recompute during drags.
let boxes = [];             // [{cx, cy, w, h, _px}]
let selectedIdx = -1;
// One unified interaction state across all mouse modes — null when idle.
//   {type:"new",    x0, y0, x1, y1}
//   {type:"move",   idx, sBox:{cx,cy,w,h}, sMouseX, sMouseY}
//   {type:"resize", idx, handle:"nw|ne|sw|se|n|s|e|w",
//                   sBox:{cx,cy,w,h}, sMouseX, sMouseY}
let interaction = null;
let dirty = false;
const HANDLE_SIZE = 9;      // px — slightly larger than the line width for grabbability

// ---- bootstrap -------------------------------------------------------

async function loadClassMeta() {
  const r = await fetch("/api/custom-classes", { cache: "no-store" });
  const all = r.ok ? await r.json() : [];
  const me = all.find((c) => c.id === CID);
  if (!me) {
    document.querySelector(".hist-section").innerHTML =
      `<p class="muted">Class not found. <a href="/classes">← Back</a></p>`;
    return false;
  }
  className = me.name;
  modelReady = !!me.model_ready;
  titleEl.textContent = `Labeling: ${me.name}`;
  classNameEl.textContent = me.name;
  document.title = `Label ${me.name} — PrimeAnalyze`;
  return true;
}

async function loadImages() {
  const r = await fetch(`/api/custom-classes/${encodeURIComponent(CID)}/images`,
    { cache: "no-store" });
  images = r.ok ? await r.json() : [];
  if (!images.length) {
    document.querySelector(".hist-section").innerHTML =
      `<p class="muted">No images to label. <a href="/classes/${encodeURIComponent(CID)}">← Add some first</a></p>`;
    return false;
  }
  // The hub returns newest-first; reverse so we label in capture order.
  images.reverse();
  // Pick starting position: explicit ?img= wins, else first unlabeled, else first.
  if (START_IMG) {
    cursor = images.findIndex((x) => x.id === START_IMG);
  }
  if (cursor < 0) {
    cursor = images.findIndex((x) => !x.labeled);
  }
  if (cursor < 0) cursor = 0;
  return true;
}

// ---- per-image render ------------------------------------------------

async function loadCurrent() {
  saveStatus.textContent = "";
  selectedIdx = -1;
  boxes = [];
  const cur = images[cursor];
  progressEl.textContent = `${cursor + 1} / ${images.length}`;

  // Use a query-string cache-buster so re-labeling the same image after
  // delete refetches its label record without a hard reload.
  const r = await fetch(`/api/custom-classes/${encodeURIComponent(CID)}/images/${cur.id}/label`,
    { cache: "no-store" });
  if (r.ok) {
    const data = await r.json();
    boxes = (data.boxes || []).map(([cx, cy, w, h]) => ({ cx, cy, w, h }));
  }
  // Auto-label: if the class has a trained model and this image is
  // currently unlabeled, fetch the model's suggestions and pre-fill
  // them as a starting point. The user accepts (just save) or edits
  // before saving. We don't run inference on already-labeled images —
  // the human labels are ground truth, no point re-predicting them.
  if (modelReady && !cur.labeled && boxes.length === 0) {
    saveStatus.textContent = "Running model prediction…";
    try {
      const pr = await fetch(
        `/api/custom-classes/${encodeURIComponent(CID)}/predict/${cur.id}`,
        { cache: "no-store" });
      if (pr.ok) {
        const data = await pr.json();
        const suggested = (data.boxes || [])
          .filter((b) => b.confidence >= 0.25)
          .map((b) => ({ cx: b.cx, cy: b.cy, w: b.w, h: b.h, _suggested: true }));
        if (suggested.length) {
          boxes = suggested;
          dirty = true;   // saving these confirms them
          saveStatus.textContent = `Model suggested ${suggested.length} box${suggested.length === 1 ? "" : "es"}. Adjust and Save.`;
        } else {
          saveStatus.textContent = "Model found nothing — draw a box.";
        }
      }
    } catch (e) {
      saveStatus.textContent = "Prediction failed; draw manually.";
    }
  }

  // Display image. We render the canvas only once we know its natural
  // pixel dims (so coordinate conversion works).
  await new Promise((res, rej) => {
    img.onload = () => res();
    img.onerror = () => rej(new Error("image load failed"));
    img.src = `/api/custom-classes/${encodeURIComponent(CID)}/images/${cur.id}`;
  });
  curImgNatural = { w: img.naturalWidth, h: img.naturalHeight };
  dimsEl.textContent = `${curImgNatural.w} × ${curImgNatural.h} px`;

  // Size the canvas to overlay the displayed (CSS-sized) image exactly.
  // We track CSS pixels for drawing math, then output normalized YOLO
  // floats independent of the display size.
  resizeCanvas();
  redraw();
  dirty = false;
}

function resizeCanvas() {
  const rect = img.getBoundingClientRect();
  // Use the actual rendered size of the <img>, not its natural pixels.
  canvas.width  = Math.round(rect.width);
  canvas.height = Math.round(rect.height);
  canvas.style.width  = rect.width + "px";
  canvas.style.height = rect.height + "px";
}

// ---- drawing ---------------------------------------------------------

const BOX_COLOR_NORMAL    = "#0bb0ba";
const BOX_COLOR_SELECTED  = "#f08a1f";
const BOX_COLOR_DRAGGING  = "#18a052";

function normToPx(box) {
  const W = canvas.width, H = canvas.height;
  const w = box.w * W;
  const h = box.h * H;
  const x = box.cx * W - w / 2;
  const y = box.cy * H - h / 2;
  return { x, y, w, h };
}

function pxToNorm(rect) {
  const W = canvas.width, H = canvas.height;
  const cx = (rect.x + rect.w / 2) / W;
  const cy = (rect.y + rect.h / 2) / H;
  return { cx, cy, w: rect.w / W, h: rect.h / H };
}

function handlePositions(px) {
  // 8 handles: 4 corners + 4 edge midpoints.
  const cx = px.x + px.w / 2;
  const cy = px.y + px.h / 2;
  return {
    nw: { x: px.x,        y: px.y },
    n:  { x: cx,           y: px.y },
    ne: { x: px.x + px.w, y: px.y },
    e:  { x: px.x + px.w, y: cy },
    se: { x: px.x + px.w, y: px.y + px.h },
    s:  { x: cx,           y: px.y + px.h },
    sw: { x: px.x,        y: px.y + px.h },
    w:  { x: px.x,        y: cy },
  };
}

function drawHandle(p) {
  const s = HANDLE_SIZE;
  ctx.fillStyle = BOX_COLOR_SELECTED;
  ctx.fillRect(p.x - s / 2, p.y - s / 2, s, s);
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(p.x - s / 2, p.y - s / 2, s, s);
}

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let i = 0; i < boxes.length; i++) {
    const px = normToPx(boxes[i]);
    boxes[i]._px = px;
    const sel = i === selectedIdx;
    const sug = boxes[i]._suggested;
    ctx.lineWidth = 3;
    ctx.strokeStyle = sel ? BOX_COLOR_SELECTED : BOX_COLOR_NORMAL;
    if (sug) ctx.setLineDash([8, 4]);
    ctx.strokeRect(px.x, px.y, px.w, px.h);
    if (sug) ctx.setLineDash([]);
    const badge = `${className} #${i + 1}${sug ? " (suggested)" : ""}`;
    ctx.font = "13px ui-monospace, monospace";
    const tw = ctx.measureText(badge).width + 10;
    ctx.fillStyle = sel ? BOX_COLOR_SELECTED : BOX_COLOR_NORMAL;
    ctx.fillRect(px.x, Math.max(0, px.y - 20), tw, 20);
    ctx.fillStyle = "#fff";
    ctx.fillText(badge, px.x + 5, Math.max(14, px.y - 6));
    // Draw resize handles on the selected box only.
    if (sel) {
      const hs = handlePositions(px);
      for (const k of Object.keys(hs)) drawHandle(hs[k]);
    }
  }

  if (interaction && interaction.type === "new") {
    const r = normalizeDrag(interaction);
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = BOX_COLOR_DRAGGING;
    ctx.strokeRect(r.x, r.y, r.w, r.h);
    ctx.setLineDash([]);
  }

  boxCountEl.textContent = `${boxes.length} box${boxes.length === 1 ? "" : "es"}`;
}

function normalizeDrag(d) {
  return {
    x: Math.min(d.x0, d.x1),
    y: Math.min(d.y0, d.y1),
    w: Math.abs(d.x1 - d.x0),
    h: Math.abs(d.y1 - d.y0),
  };
}

// ---- mouse events ----------------------------------------------------

function evtPoint(e) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: e.clientX - rect.left,
    y: e.clientY - rect.top,
  };
}

function hitTestBox(p) {
  // Top-most box (later index = drawn last) wins.
  for (let i = boxes.length - 1; i >= 0; i--) {
    const px = boxes[i]._px;
    if (!px) continue;
    if (p.x >= px.x && p.x <= px.x + px.w && p.y >= px.y && p.y <= px.y + px.h) {
      return i;
    }
  }
  return -1;
}

function hitTestHandle(p, idx) {
  // Only the selected box exposes handles.
  if (idx < 0) return null;
  const px = boxes[idx]._px;
  if (!px) return null;
  const hs = handlePositions(px);
  const s = HANDLE_SIZE + 2;     // a touch larger than the visual square
  for (const k of Object.keys(hs)) {
    const h = hs[k];
    if (Math.abs(p.x - h.x) <= s / 2 && Math.abs(p.y - h.y) <= s / 2) {
      return k;
    }
  }
  return null;
}

function cursorForHandle(k) {
  return {
    nw: "nwse-resize", se: "nwse-resize",
    ne: "nesw-resize", sw: "nesw-resize",
    n:  "ns-resize",   s:  "ns-resize",
    e:  "ew-resize",   w:  "ew-resize",
  }[k] || "default";
}

canvas.addEventListener("mousedown", (e) => {
  const p = evtPoint(e);
  // 1. Handle on currently-selected box → resize.
  const handle = hitTestHandle(p, selectedIdx);
  if (handle) {
    const b = boxes[selectedIdx];
    interaction = {
      type: "resize", idx: selectedIdx, handle,
      sBox: { cx: b.cx, cy: b.cy, w: b.w, h: b.h },
      sMouseX: p.x, sMouseY: p.y,
    };
    return;
  }
  // 2. Inside an existing box → select + start move.
  const hit = hitTestBox(p);
  if (hit >= 0) {
    selectedIdx = hit;
    const b = boxes[hit];
    interaction = {
      type: "move", idx: hit,
      sBox: { cx: b.cx, cy: b.cy, w: b.w, h: b.h },
      sMouseX: p.x, sMouseY: p.y,
    };
    redraw();
    return;
  }
  // 3. Empty space → drag to draw a new box.
  selectedIdx = -1;
  interaction = { type: "new", x0: p.x, y0: p.y, x1: p.x, y1: p.y };
  redraw();
});

canvas.addEventListener("mousemove", (e) => {
  const p = evtPoint(e);

  // Hover affordance: when idle, show resize cursor over handles and
  // move cursor inside the selected box.
  if (!interaction) {
    const handle = hitTestHandle(p, selectedIdx);
    if (handle) {
      canvas.style.cursor = cursorForHandle(handle);
    } else if (selectedIdx >= 0) {
      const px = boxes[selectedIdx]._px;
      const inside = px && p.x >= px.x && p.x <= px.x + px.w
                          && p.y >= px.y && p.y <= px.y + px.h;
      canvas.style.cursor = inside ? "move" : "crosshair";
    } else {
      canvas.style.cursor = "crosshair";
    }
    return;
  }

  if (interaction.type === "new") {
    interaction.x1 = p.x;
    interaction.y1 = p.y;
    redraw();
    return;
  }

  const W = canvas.width, H = canvas.height;
  const dx = (p.x - interaction.sMouseX) / W;
  const dy = (p.y - interaction.sMouseY) / H;
  const s = interaction.sBox;
  const b = boxes[interaction.idx];

  if (interaction.type === "move") {
    // Clamp the box inside [0, 1] so it never leaves the image area.
    const halfW = s.w / 2;
    const halfH = s.h / 2;
    b.cx = Math.max(halfW, Math.min(1 - halfW, s.cx + dx));
    b.cy = Math.max(halfH, Math.min(1 - halfH, s.cy + dy));
    delete b._suggested;
    dirty = true;
    redraw();
    return;
  }

  if (interaction.type === "resize") {
    // Convert center-form to edges, mutate edges per handle, recompute.
    let left   = s.cx - s.w / 2;
    let right  = s.cx + s.w / 2;
    let top    = s.cy - s.h / 2;
    let bottom = s.cy + s.h / 2;
    const k = interaction.handle;
    if (k.includes("w")) left   = Math.max(0, Math.min(right  - 0.01, left   + dx));
    if (k.includes("e")) right  = Math.min(1, Math.max(left   + 0.01, right  + dx));
    if (k.includes("n")) top    = Math.max(0, Math.min(bottom - 0.01, top    + dy));
    if (k.includes("s")) bottom = Math.min(1, Math.max(top    + 0.01, bottom + dy));
    b.cx = (left + right) / 2;
    b.cy = (top + bottom) / 2;
    b.w  = right - left;
    b.h  = bottom - top;
    delete b._suggested;
    dirty = true;
    redraw();
    return;
  }
});

window.addEventListener("mouseup", () => {
  if (!interaction) return;
  if (interaction.type === "new") {
    const r = normalizeDrag(interaction);
    if (r.w >= 8 && r.h >= 8) {
      boxes.push(pxToNorm(r));
      selectedIdx = boxes.length - 1;
      dirty = true;
    }
  }
  interaction = null;
  canvas.style.cursor = "crosshair";
  redraw();
});

window.addEventListener("resize", () => {
  resizeCanvas();
  redraw();
});

// ---- save / navigate -------------------------------------------------

async function saveCurrent() {
  if (cursor < 0) return false;
  const cur = images[cursor];
  const payload = boxes.map((b) => [b.cx, b.cy, b.w, b.h]);
  saveStatus.textContent = "Saving…";
  const r = await fetch(
    `/api/custom-classes/${encodeURIComponent(CID)}/images/${cur.id}/label`,
    { method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ boxes: payload }) },
  );
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    saveStatus.textContent = "Save failed: " + (err.error || r.status);
    return false;
  }
  // Update the in-memory image record so prev/next badges stay current.
  cur.labeled = true;
  cur.box_count = boxes.length;
  saveStatus.textContent = `Saved ${boxes.length} box${boxes.length === 1 ? "" : "es"}.`;
  dirty = false;
  return true;
}

async function go(delta) {
  if (dirty) {
    const ok = await saveCurrent();
    if (!ok) return;
  }
  const next = cursor + delta;
  if (next < 0 || next >= images.length) {
    saveStatus.textContent = next < 0 ? "At first image." : "At last image.";
    return;
  }
  cursor = next;
  loadCurrent();
}

function goNextUnlabeled() {
  // Save current first if dirty; the async result is fine to discard since
  // we look at in-memory state.
  if (dirty) saveCurrent();
  for (let i = cursor + 1; i < images.length; i++) {
    if (!images[i].labeled) {
      cursor = i;
      loadCurrent();
      return;
    }
  }
  for (let i = 0; i <= cursor; i++) {
    if (!images[i].labeled) {
      cursor = i;
      loadCurrent();
      return;
    }
  }
  saveStatus.textContent = "All images are labeled. 🎉";
}

btnPrev.addEventListener("click", () => go(-1));
btnNext.addEventListener("click", () => go(1));
btnNextUnlabeled.addEventListener("click", goNextUnlabeled);
btnSave.addEventListener("click", async () => {
  const ok = await saveCurrent();
  if (ok) goNextUnlabeled();
});
btnClear.addEventListener("click", () => {
  if (!boxes.length) return;
  boxes = [];
  selectedIdx = -1;
  dirty = true;
  redraw();
});

document.addEventListener("keydown", (e) => {
  if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
  if (e.key === "ArrowLeft")        { e.preventDefault(); go(-1); }
  else if (e.key === "ArrowRight")  { e.preventDefault(); go(1); }
  else if (e.key === "u" || e.key === "U") { e.preventDefault(); goNextUnlabeled(); }
  else if (e.key === "Enter")       { e.preventDefault(); btnSave.click(); }
  else if (e.key === "Delete" || e.key === "Backspace") {
    if (selectedIdx >= 0) {
      e.preventDefault();
      boxes.splice(selectedIdx, 1);
      selectedIdx = -1;
      dirty = true;
      redraw();
    }
  } else if (e.key === "Escape") {
    if (dragging) { dragging = null; redraw(); }
  }
});

// ---- run -------------------------------------------------------------

(async () => {
  if (!(await loadClassMeta())) return;
  if (!(await loadImages())) return;
  await loadCurrent();
})();
