// Polygon / rectangle ROI editor.
// Points are stored as normalized [x, y] in 0..1. Saved to the hub via
// PATCH /api/cameras/<id> with either roi_polygon (list) or roi (rect).

const wrap    = document.getElementById("canvas-wrap");
const img     = document.getElementById("snap");
const canvas  = document.getElementById("canvas");
const loading = document.getElementById("loading");
const ctx     = canvas.getContext("2d");

const POINT_HIT_RADIUS = 0.025;   // normalized — pick up clicks within this
const POINT_SIZE       = 7;       // pixels

let mode    = "polygon";          // "polygon" | "rect"
let polygon = [];                 // [[x, y], ...]
let rect    = [0, 0, 1, 1];       // [x1, y1, x2, y2]
let dragIdx = -1;
let dragStart = null;             // rect: pointerdown corner
let dragStartedFromExisting = false;

// === init from server state =============================================
if (initialPoly && Array.isArray(initialPoly) && initialPoly.length >= 3) {
  polygon = initialPoly.map((p) => [Number(p[0]), Number(p[1])]);
  mode = "polygon";
  document.getElementById("mode-polygon").checked = true;
} else if (initialRect && Array.isArray(initialRect) && initialRect.length === 4) {
  rect = initialRect.map(Number);
  // If existing rect isn't the full frame, default to rectangle mode so the
  // user can refine what they already have.
  if (!(rect[0] <= 0.01 && rect[1] <= 0.01 && rect[2] >= 0.99 && rect[3] >= 0.99)) {
    mode = "rect";
    document.getElementById("mode-rect").checked = true;
  }
}

// === snapshot loading ===================================================
function loadSnapshot(bust = false) {
  loading.style.display = "";
  const u = "/api/cameras/" + camId + "/snapshot" + (bust ? "?t=" + Date.now() : "");
  img.src = u;
}
img.addEventListener("load", () => {
  loading.style.display = "none";
  fitCanvas();
  draw();
});
img.addEventListener("error", () => {
  loading.textContent = "snapshot unavailable (worker not ready)";
});
loadSnapshot();

// === canvas sizing ======================================================
function fitCanvas() {
  const w = img.clientWidth, h = img.clientHeight;
  if (!w || !h) return;
  canvas.width = w;
  canvas.height = h;
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
}
window.addEventListener("resize", () => { fitCanvas(); draw(); });

// === mode switching =====================================================
document.querySelectorAll("input[name=mode]").forEach((el) =>
  el.addEventListener("change", () => {
    mode = el.value;
    draw();
    renderPointsList();
  }),
);

// === drawing ============================================================
function clamp01(v) { return Math.max(0, Math.min(1, v)); }

function px(p) {
  return [p[0] * canvas.width, p[1] * canvas.height];
}

function draw() {
  if (!canvas.width) return;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "rgba(0, 0, 0, 0.55)";
  ctx.fillRect(0, 0, W, H);

  if (mode === "polygon" && polygon.length >= 3) {
    // Cut a transparent hole
    ctx.save();
    ctx.globalCompositeOperation = "destination-out";
    ctx.beginPath();
    polygon.forEach((p, i) => {
      const [x, y] = px(p);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  } else if (mode === "rect") {
    const [x1, y1] = px([rect[0], rect[1]]);
    const [x2, y2] = px([rect[2], rect[3]]);
    ctx.save();
    ctx.globalCompositeOperation = "destination-out";
    ctx.fillRect(
      Math.min(x1, x2), Math.min(y1, y2),
      Math.abs(x2 - x1), Math.abs(y2 - y1),
    );
    ctx.restore();
  }

  // Draw outline + handles
  ctx.strokeStyle = "#0bb0ba";
  ctx.lineWidth = 2;

  if (mode === "polygon") {
    if (polygon.length >= 2) {
      ctx.beginPath();
      polygon.forEach((p, i) => {
        const [x, y] = px(p);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      if (polygon.length >= 3) ctx.closePath();
      ctx.stroke();
    }
    polygon.forEach((p, i) => {
      const [x, y] = px(p);
      ctx.fillStyle = "#0bb0ba";
      ctx.beginPath();
      ctx.arc(x, y, POINT_SIZE, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "bold 10px -apple-system, Segoe UI, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(i + 1), x, y);
    });
  } else if (mode === "rect") {
    const [x1, y1] = px([rect[0], rect[1]]);
    const [x2, y2] = px([rect[2], rect[3]]);
    ctx.strokeRect(Math.min(x1, x2), Math.min(y1, y2),
                   Math.abs(x2 - x1), Math.abs(y2 - y1));
    [[x1, y1], [x2, y1], [x1, y2], [x2, y2]].forEach(([x, y]) => {
      ctx.fillStyle = "#0bb0ba";
      ctx.fillRect(x - 5, y - 5, 10, 10);
    });
  }
}

// === pointer interaction ================================================
function eventToNorm(e) {
  const r = canvas.getBoundingClientRect();
  return [clamp01((e.clientX - r.left) / r.width),
          clamp01((e.clientY - r.top) / r.height)];
}

function hitVertex(nx, ny) {
  for (let i = polygon.length - 1; i >= 0; i--) {
    const dx = polygon[i][0] - nx;
    const dy = polygon[i][1] - ny;
    if (Math.hypot(dx, dy) < POINT_HIT_RADIUS) return i;
  }
  return -1;
}

canvas.addEventListener("pointerdown", (e) => {
  if (e.button === 2) return;     // right-click handled below
  const [nx, ny] = eventToNorm(e);
  if (mode === "polygon") {
    const idx = hitVertex(nx, ny);
    if (idx >= 0) {
      dragIdx = idx;
      dragStartedFromExisting = true;
      canvas.setPointerCapture(e.pointerId);
    } else {
      polygon.push([nx, ny]);
      dragIdx = polygon.length - 1;
      dragStartedFromExisting = false;
      canvas.setPointerCapture(e.pointerId);
    }
    draw();
    renderPointsList();
  } else {
    dragStart = [nx, ny];
    rect = [nx, ny, nx, ny];
    canvas.setPointerCapture(e.pointerId);
    draw();
  }
});

canvas.addEventListener("pointermove", (e) => {
  const [nx, ny] = eventToNorm(e);
  if (mode === "polygon" && dragIdx >= 0) {
    polygon[dragIdx] = [nx, ny];
    draw();
    renderPointsList();
  } else if (mode === "rect" && dragStart) {
    rect = [dragStart[0], dragStart[1], nx, ny];
    draw();
  }
});

canvas.addEventListener("pointerup", () => {
  if (mode === "rect" && dragStart) {
    rect = [Math.min(rect[0], rect[2]), Math.min(rect[1], rect[3]),
            Math.max(rect[0], rect[2]), Math.max(rect[1], rect[3])];
  }
  dragIdx = -1;
  dragStart = null;
  dragStartedFromExisting = false;
});

canvas.addEventListener("contextmenu", (e) => {
  e.preventDefault();
  if (mode !== "polygon") return;
  const [nx, ny] = eventToNorm(e);
  const idx = hitVertex(nx, ny);
  if (idx >= 0) {
    polygon.splice(idx, 1);
    draw();
    renderPointsList();
  }
});

// === buttons + points list ==============================================
document.getElementById("btn-reset").addEventListener("click", () => {
  polygon = [];
  rect = [0, 0, 1, 1];
  draw(); renderPointsList();
});
document.getElementById("btn-clear").addEventListener("click", () => {
  polygon = [];
  draw(); renderPointsList();
});
document.getElementById("btn-refresh").addEventListener("click", () => loadSnapshot(true));

function renderPointsList() {
  document.getElementById("points-count").textContent =
    mode === "polygon" ? polygon.length : "rect";
  const el = document.getElementById("points-list");
  if (mode !== "polygon") { el.innerHTML = ""; return; }
  if (polygon.length === 0) {
    el.innerHTML = '<div class="cfg-empty">Click on the snapshot to add the first point.</div>';
    return;
  }
  el.innerHTML = polygon.map((p, i) =>
    `<div class="cfg-point-row"><span>${i + 1}</span><code>${p[0].toFixed(3)}, ${p[1].toFixed(3)}</code></div>`
  ).join("");
}

renderPointsList();

// === save ===============================================================
document.getElementById("btn-save").addEventListener("click", async () => {
  const status = document.getElementById("status");
  status.className = "cfg-status info";
  status.textContent = "Saving…";
  let body;
  if (mode === "polygon") {
    if (polygon.length < 3) {
      status.className = "cfg-status err";
      status.textContent = "A polygon needs at least 3 points.";
      return;
    }
    body = { roi_polygon: polygon, roi: bboxOfPolygon(polygon) };
  } else {
    body = { roi: rect, roi_polygon: null };
  }
  try {
    const r = await fetch("/api/cameras/" + camId, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if (!r.ok) {
      status.className = "cfg-status err";
      status.textContent = d.error || ("HTTP " + r.status);
      return;
    }
    status.className = "cfg-status ok";
    status.textContent = "✓ Saved. Worker is restarting with the new region…";
  } catch (e) {
    status.className = "cfg-status err";
    status.textContent = "Save failed: " + e.message;
  }
});

function bboxOfPolygon(pts) {
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}
