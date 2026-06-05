// Shared lightbox correction-drawing helper. Loaded by both History and
// Summary because they share _lightbox.html. Exposes:
//
//   LbCorrection.attach({ getOpenSnap, onSaved })
//
// where getOpenSnap() returns the snapshot object currently shown in
// the lightbox (must have .id and .file_rel) and onSaved(snap, boxes)
// is called after a successful PUT so the page can update its in-
// memory copy.
//
// The correction stage only appears when the operator marks the
// current snapshot as "incorrect" — that's when a drawn box is
// information-rich enough to feed back into training.

window.LbCorrection = (function () {
  const root        = document.getElementById("lb-correction");
  if (!root) return { attach() {}, hide() {}, show() {} };  // page without lightbox
  const stage       = document.getElementById("lb-correction-stage");
  const img         = document.getElementById("lb-correction-img");
  const canvas      = document.getElementById("lb-correction-canvas");
  const btnClear    = document.getElementById("lb-correction-clear");
  const btnSave     = document.getElementById("lb-correction-save");
  const statusEl    = document.getElementById("lb-correction-status");
  const ctx         = canvas.getContext("2d");

  let openSnap = null;
  let boxes = [];                 // [{cx, cy, w, h}] normalized
  let dragging = null;            // {x0, y0, x1, y1}
  let getOpenSnapFn = null;
  let onSavedFn = null;
  let imgLoadPromise = null;

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
      const tw = ctx.measureText(label).width + 8;
      ctx.fillRect(x, Math.max(0, y - 18), tw, 18);
      ctx.fillStyle = "#fff";
      ctx.fillText(label, x + 4, Math.max(13, y - 5));
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
      // Only one correction box per snapshot — replace any existing.
      boxes = [{
        cx: (rx + rw / 2) / canvas.width,
        cy: (ry + rh / 2) / canvas.height,
        w: rw / canvas.width,
        h: rh / canvas.height,
      }];
      statusEl.textContent = "Drawn — click Save correction to keep it.";
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
      statusEl.textContent = `Saved ${boxes.length} box${boxes.length === 1 ? "" : "es"}. Re-train will pick this up.`;
      openSnap.correction_boxes = boxes.map((b) => [b.cx, b.cy, b.w, b.h]);
      if (onSavedFn) onSavedFn(openSnap, openSnap.correction_boxes);
    } catch (e) {
      statusEl.textContent = "Save failed: " + e.message;
    }
  });

  async function show(snap) {
    openSnap = snap;
    boxes = (snap.correction_boxes || []).map(([cx, cy, w, h]) => ({ cx, cy, w, h }));
    statusEl.textContent = "";
    root.hidden = false;
    // Use the raw frame if it exists so the dashed YOLO overlay
    // doesn't distract while you draw.
    const annotated = `/api/snapshots/${snap.file_rel}`;
    const raw = annotated.replace(/\.jpg$/, "_raw.jpg");
    img.src = raw;
    img.onerror = () => { img.onerror = null; img.src = annotated; };
    imgLoadPromise = new Promise((res) => { img.onload = () => res(); });
    await imgLoadPromise;
    resizeCanvas();
    redraw();
  }

  function hide() {
    root.hidden = true;
    openSnap = null;
    boxes = [];
    dragging = null;
    statusEl.textContent = "";
  }

  function attach({ getOpenSnap, onSaved } = {}) {
    getOpenSnapFn = getOpenSnap || null;
    onSavedFn = onSaved || null;
  }

  return { attach, show, hide };
})();
