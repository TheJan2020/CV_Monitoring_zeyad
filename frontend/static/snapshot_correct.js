// Standalone full-page correction drawing UI. Opened in its own tab
// from the History / Summary lightboxes when the operator marks a
// snapshot incorrect on an out_of_frame frame.

const SNAP_ID = Number(window.__SNAP_ID);

const img        = document.getElementById("sc-img");
const canvas     = document.getElementById("sc-canvas");
const ctx        = canvas.getContext("2d");
const titleEl    = document.getElementById("sc-title");
const metaEl     = document.getElementById("sc-meta");
const statusEl   = document.getElementById("sc-status");
const btnClear   = document.getElementById("sc-clear");
const btnSave    = document.getElementById("sc-save");
const btnClose   = document.getElementById("sc-close");

let boxes = [];
let dragging = null;
let snap = null;

function resizeCanvas() {
  const r = img.getBoundingClientRect();
  canvas.width = Math.round(r.width);
  canvas.height = Math.round(r.height);
  canvas.style.width = r.width + "px";
  canvas.style.height = r.height + "px";
}

function redraw() {
  if (!canvas.width || !canvas.height) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (const b of boxes) {
    const w = b.w * canvas.width, h = b.h * canvas.height;
    const x = b.cx * canvas.width - w / 2;
    const y = b.cy * canvas.height - h / 2;
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#18a052";
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = "#18a052";
    ctx.font = "13px ui-monospace, monospace";
    const label = "baby (correction)";
    const tw = ctx.measureText(label).width + 10;
    ctx.fillRect(x, Math.max(0, y - 20), tw, 20);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x + 5, Math.max(14, y - 6));
  }
  if (dragging) {
    const rx = Math.min(dragging.x0, dragging.x1);
    const ry = Math.min(dragging.y0, dragging.y1);
    const rw = Math.abs(dragging.x1 - dragging.x0);
    const rh = Math.abs(dragging.y1 - dragging.y0);
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = "#18a052";
    ctx.lineWidth = 2;
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.setLineDash([]);
  }
}

function evtPoint(e) {
  const r = canvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

canvas.addEventListener("mousedown", (e) => {
  const p = evtPoint(e);
  dragging = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
  redraw();
});
canvas.addEventListener("mousemove", (e) => {
  if (!dragging) return;
  const p = evtPoint(e);
  dragging.x1 = p.x;
  dragging.y1 = p.y;
  redraw();
});
window.addEventListener("mouseup", () => {
  if (!dragging) return;
  const rx = Math.min(dragging.x0, dragging.x1);
  const ry = Math.min(dragging.y0, dragging.y1);
  const rw = Math.abs(dragging.x1 - dragging.x0);
  const rh = Math.abs(dragging.y1 - dragging.y0);
  dragging = null;
  if (rw >= 8 && rh >= 8) {
    // Single box per snapshot — replace any existing.
    boxes = [{
      cx: (rx + rw / 2) / canvas.width,
      cy: (ry + rh / 2) / canvas.height,
      w: rw / canvas.width,
      h: rh / canvas.height,
    }];
    statusEl.textContent = "Drawn — click Save correction.";
  }
  redraw();
});

window.addEventListener("resize", () => { resizeCanvas(); redraw(); });

btnClear.addEventListener("click", () => {
  boxes = [];
  statusEl.textContent = "";
  redraw();
});

btnSave.addEventListener("click", async () => {
  statusEl.textContent = "Saving…";
  try {
    const r = await fetch(`/api/snapshots/${SNAP_ID}/correction`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        boxes: boxes.map((b) => [b.cx, b.cy, b.w, b.h]),
      }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      statusEl.textContent = "Save failed: " + (err.error || r.status);
      return;
    }
    statusEl.textContent = boxes.length
      ? `Saved ${boxes.length} box. You can close this tab.`
      : "Correction cleared. You can close this tab.";
    // Cross-tab nudge: any open History / Summary tab listens to this
    // localStorage key and refreshes the matching thumbnail badge.
    try {
      localStorage.setItem(
        "primeanalyze.snapshotCorrectionSaved",
        JSON.stringify({ snap_id: SNAP_ID, box_count: boxes.length, t: Date.now() }),
      );
    } catch (e) {}
  } catch (e) {
    statusEl.textContent = "Save failed: " + e.message;
  }
});

btnClose.addEventListener("click", () => {
  // Browsers only allow scripts to close tabs they opened. If the
  // tab was opened with target=_blank from another tab it works;
  // otherwise we fall back to leaving a "you can close this tab" hint.
  window.close();
  setTimeout(() => {
    statusEl.textContent = "Couldn't auto-close — close the tab manually.";
  }, 100);
});

function fmtClock(ts) {
  const d = new Date(ts * 1000);
  return d.toTimeString().slice(0, 8);
}
function fmtDate(ts) {
  const d = new Date(ts * 1000);
  return d.toISOString().slice(0, 10);
}

(async function bootstrap() {
  if (!SNAP_ID) {
    statusEl.textContent = "Missing snapshot id.";
    return;
  }
  let r;
  try {
    r = await fetch(`/api/snapshots/${SNAP_ID}`, { cache: "no-store" });
  } catch (e) {
    statusEl.textContent = "Failed to load snapshot.";
    return;
  }
  if (!r.ok) {
    statusEl.textContent = "Snapshot not found.";
    return;
  }
  snap = await r.json();
  titleEl.textContent = `Draw the baby's location — ${fmtClock(snap.captured_at)}`;
  metaEl.textContent = `${snap.camera_id} · ${fmtDate(snap.captured_at)} ${fmtClock(snap.captured_at)}`;
  boxes = (snap.correction_boxes || []).map(([cx, cy, w, h]) => ({ cx, cy, w, h }));
  if (boxes.length) statusEl.textContent = "Existing correction loaded — adjust and re-save if needed.";

  const annotated = `/api/snapshots/${snap.file_rel}`;
  const raw = annotated.replace(/\.jpg$/, "_raw.jpg");
  img.onerror = () => { img.onerror = null; img.src = annotated; };
  await new Promise((res) => {
    img.onload = () => res();
    img.src = raw;
  });
  resizeCanvas();
  redraw();
})();
