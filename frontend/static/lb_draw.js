// Lightbox Draw-box tab. Shared by History and Summary because they
// both include _lightbox.html. Owns the canvas, mouse handlers, and
// the Save / Clear buttons; expects the host page to call
//
//   LbDraw.show(snap)   — when the Draw tab is activated for a snap
//   LbDraw.hide()       — when the lightbox closes or the user navigates
//   LbDraw.attach({ onSaved })  — once on page load
//
// Persists to PUT /api/snapshots/<id>/correction (single box per snap;
// drawing again replaces). Sends a localStorage nudge so the open
// History / Summary grid can update the affected cell's badge without
// a refresh.

window.LbDraw = (function () {
  const stage   = document.getElementById("lb-draw-stage");
  const img     = document.getElementById("lb-draw-img");
  const canvas  = document.getElementById("lb-draw-canvas");
  const btnClear = document.getElementById("lb-draw-clear");
  const btnSave  = document.getElementById("lb-draw-save");
  const statusEl = document.getElementById("lb-draw-status");
  if (!stage) return { attach() {}, show() {}, hide() {} }; // page with no lightbox

  const ctx = canvas.getContext("2d");
  let openSnap = null;
  let boxes = [];                     // [{cx, cy, w, h}] normalized 0..1
  let dragging = null;                // {x0,y0,x1,y1} canvas pixels
  let onSavedFn = null;

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
      ctx.font = "12px ui-monospace, monospace";
      const label = "baby (correction)";
      const tw = ctx.measureText(label).width + 10;
      ctx.fillRect(x, Math.max(0, y - 18), tw, 18);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x + 5, Math.max(13, y - 5));
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

  // Resize listener — only re-render when the draw tab is the visible one,
  // otherwise the modal resize triggers a useless redraw.
  window.addEventListener("resize", () => {
    if (stage.closest(".lb-pane")?.classList.contains("active")) {
      resizeCanvas();
      redraw();
    }
  });

  btnClear.addEventListener("click", () => {
    boxes = [];
    statusEl.textContent = "";
    redraw();
  });

  btnSave.addEventListener("click", async () => {
    if (!openSnap || openSnap.id == null) return;
    statusEl.textContent = "Saving…";
    try {
      const r = await fetch(`/api/snapshots/${openSnap.id}/correction`, {
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
        ? `Saved ${boxes.length} box.`
        : "Correction cleared.";
      const stored = boxes.map((b) => [b.cx, b.cy, b.w, b.h]);
      openSnap.correction_boxes = stored;
      if (onSavedFn) onSavedFn(openSnap, stored);
      // Same-tab nudge to refresh the grid badge.
      try {
        localStorage.setItem(
          "primeanalyze.snapshotCorrectionSaved",
          JSON.stringify({
            snap_id: openSnap.id, box_count: boxes.length, t: Date.now(),
          }),
        );
      } catch (e) {}
    } catch (e) {
      statusEl.textContent = "Save failed: " + e.message;
    }
  });

  async function show(snap) {
    openSnap = snap;
    boxes = (snap.correction_boxes || []).map(([cx, cy, w, h]) => ({ cx, cy, w, h }));
    statusEl.textContent = boxes.length
      ? "Existing correction loaded — adjust or re-save."
      : "Drag on the image to draw a box around the baby.";

    // Prefer the raw frame; fall back to annotated if no raw was saved.
    const annotated = `/api/snapshots/${snap.file_rel}`;
    const raw = annotated.replace(/\.jpg$/, "_raw.jpg");
    await new Promise((res) => {
      img.onerror = () => { img.onerror = null; img.src = annotated; img.onload = () => res(); };
      img.onload = () => res();
      img.src = raw;
    });
    resizeCanvas();
    redraw();
  }

  function hide() {
    openSnap = null;
    boxes = [];
    dragging = null;
    statusEl.textContent = "";
    img.removeAttribute("src");
  }

  function attach({ onSaved } = {}) {
    onSavedFn = onSaved || null;
  }

  return { attach, show, hide };
})();
