// Class workspace — upload images, list grid, delete.

const CID = window.__CLASS_ID;
const grid = document.getElementById("cw-grid");
const empty = document.getElementById("cw-empty");
const stats = document.getElementById("cw-stats");
const titleEl = document.getElementById("cw-title");
const drop = document.getElementById("cw-drop");
const fileInput = document.getElementById("cw-file");
const btnBrowse = document.getElementById("cw-browse");
const btnDelete = document.getElementById("cw-delete");
const btnLabel  = document.getElementById("cw-label");
const btnTrain  = document.getElementById("cw-train");
const trainCard   = document.getElementById("cw-train-card");
const trainTitle  = document.getElementById("cw-train-title");
const trainBar    = document.getElementById("cw-train-bar-fill");
const trainMeta   = document.getElementById("cw-train-meta");
const trainLog    = document.getElementById("cw-train-log");
const MIN_LABELED_TO_TRAIN = 20;
let trainPollHandle = null;
const uploadStatus = document.getElementById("cw-upload-status");
const gridCount = document.getElementById("cw-grid-count");

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

async function loadMeta() {
  const r = await fetch("/api/custom-classes", { cache: "no-store" });
  const all = r.ok ? await r.json() : [];
  const me = all.find((c) => c.id === CID);
  if (!me) {
    document.querySelector(".hist-section").innerHTML =
      `<p class="muted">Class not found. <a href="/classes">← Back</a></p>`;
    return null;
  }
  titleEl.textContent = me.name;
  document.title = `${me.name} — PrimeAnalyze`;
  stats.textContent =
    `${me.image_count} image${me.image_count === 1 ? "" : "s"} · ${me.labeled_count} labeled`
    + (me.model_ready ? " · model ready" : "");
  // Enable Train when there's enough labeled data. The action label
  // flips between Train / Re-train depending on whether a model is
  // already on disk.
  if (btnTrain) {
    const ok = me.labeled_count >= MIN_LABELED_TO_TRAIN;
    btnTrain.disabled = !ok;
    btnTrain.title = ok
      ? "Spawn a fine-tune job on the RTX 5000"
      : `Need at least ${MIN_LABELED_TO_TRAIN} labeled images (have ${me.labeled_count})`;
    btnTrain.textContent = me.model_ready ? "Re-train" : "Train on GPU";
  }
  return me;
}

async function loadImages() {
  const r = await fetch(`/api/custom-classes/${encodeURIComponent(CID)}/images`, { cache: "no-store" });
  const imgs = r.ok ? await r.json() : [];
  gridCount.textContent = imgs.length ? `(${imgs.length})` : "";
  empty.hidden = imgs.length > 0;
  grid.innerHTML = imgs.map((im) => `
    <a class="cw-thumb" data-id="${im.id}"
       href="/classes/${encodeURIComponent(CID)}/label?img=${encodeURIComponent(im.id)}"
       title="Click to label this image">
      <img src="/api/custom-classes/${encodeURIComponent(CID)}/images/${im.id}"
           alt="" loading="lazy">
      <div class="cw-thumb-meta">
        <span class="cw-thumb-badge cw-badge-${im.labeled ? "labeled" : "unlabeled"}">
          ${im.labeled ? `${im.box_count} box${im.box_count === 1 ? "" : "es"}` : "unlabeled"}
        </span>
        <button class="cw-thumb-del" data-id="${im.id}" title="Delete">✕</button>
      </div>
    </a>
  `).join("");
  grid.querySelectorAll(".cw-thumb-del").forEach((b) =>
    b.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      deleteImage(b.dataset.id);
    })
  );
}

if (btnLabel) {
  btnLabel.href = `/classes/${encodeURIComponent(CID)}/label`;
}

async function deleteImage(imgId) {
  await fetch(`/api/custom-classes/${encodeURIComponent(CID)}/images/${imgId}`, { method: "DELETE" });
  refresh();
}

async function refresh() {
  await loadMeta();
  await loadImages();
}

async function upload(files) {
  if (!files || !files.length) return;
  uploadStatus.textContent = `Uploading ${files.length}…`;
  const fd = new FormData();
  for (const f of files) fd.append("file", f, f.name);
  try {
    const r = await fetch(`/api/custom-classes/${encodeURIComponent(CID)}/images`, {
      method: "POST",
      body: fd,
    });
    const data = await r.json().catch(() => ({}));
    if (r.ok) {
      const okN = (data.saved || []).length;
      const errN = (data.errors || []).length;
      uploadStatus.textContent = errN
        ? `Saved ${okN}, skipped ${errN}: ${data.errors.map((e) => e.error).join(", ")}`
        : `Saved ${okN}.`;
    } else {
      uploadStatus.textContent = `Upload failed: ${data.error || r.status}`;
    }
  } catch (e) {
    uploadStatus.textContent = "Upload failed: " + e.message;
  }
  refresh();
}

btnBrowse.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => upload(fileInput.files));

["dragenter", "dragover"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("dragging"); })
);
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, () => drop.classList.remove("dragging"))
);
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  const files = [...(e.dataTransfer?.files || [])].filter((f) => f.type.startsWith("image/"));
  upload(files);
});

// ---- webcam capture --------------------------------------------------

const camStage    = document.getElementById("cw-cam-stage");
const camVideo    = document.getElementById("cw-cam-video");
const camFlash    = document.getElementById("cw-cam-flash");
const camStart    = document.getElementById("cw-cam-start");
const camCapture  = document.getElementById("cw-cam-capture");
const camStop     = document.getElementById("cw-cam-stop");
const camPick     = document.getElementById("cw-cam-pick");
const camDevice   = document.getElementById("cw-cam-device");
const camStatus   = document.getElementById("cw-cam-status");

let camStream = null;
let camSessionCount = 0;

async function camListDevices() {
  const devs = await navigator.mediaDevices.enumerateDevices();
  const cams = devs.filter((d) => d.kind === "videoinput");
  camDevice.innerHTML = cams.map((c, i) =>
    `<option value="${c.deviceId}">${c.label || `Camera ${i + 1}`}</option>`
  ).join("");
  camPick.hidden = cams.length < 2;
}

async function camStartFn() {
  if (!navigator.mediaDevices?.getUserMedia) {
    camStatus.textContent = "This browser doesn't support camera access.";
    return;
  }
  try {
    const constraints = {
      audio: false,
      video: camDevice.value
        ? { deviceId: { exact: camDevice.value }, width: 1280, height: 720 }
        : { width: 1280, height: 720, facingMode: "environment" },
    };
    camStream = await navigator.mediaDevices.getUserMedia(constraints);
    camVideo.srcObject = camStream;
    await camVideo.play();
    await camListDevices();
    camStage.hidden = false;
    camStart.hidden = true;
    camCapture.hidden = false;
    camStop.hidden = false;
    camStatus.textContent = "Aim at the object and click Capture. Press space to grab a frame.";
  } catch (e) {
    camStatus.textContent = "Camera blocked: " + e.message;
  }
}

function camStopFn() {
  if (camStream) {
    camStream.getTracks().forEach((t) => t.stop());
    camStream = null;
  }
  camVideo.srcObject = null;
  camStage.hidden = true;
  camStart.hidden = false;
  camCapture.hidden = true;
  camStop.hidden = true;
  camStatus.textContent = camSessionCount
    ? `Captured ${camSessionCount} this session.`
    : "Click Start, allow camera access, then aim and click Capture.";
}

async function camCaptureFn() {
  if (!camStream || camVideo.readyState < 2) return;
  // Brief screen flash so the user gets a tactile "I captured a frame" signal.
  camFlash.classList.add("flashing");
  setTimeout(() => camFlash.classList.remove("flashing"), 130);

  const canvas = document.createElement("canvas");
  canvas.width = camVideo.videoWidth;
  canvas.height = camVideo.videoHeight;
  canvas.getContext("2d").drawImage(camVideo, 0, 0);
  const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.9));
  camCapture.disabled = true;
  try {
    const r = await fetch(`/api/custom-classes/${encodeURIComponent(CID)}/images`, {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
      body: blob,
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      camStatus.textContent = "Capture failed: " + (err.error || r.status);
      return;
    }
    camSessionCount += 1;
    camStatus.textContent = `Captured ${camSessionCount} this session.`;
    refresh();
  } catch (e) {
    camStatus.textContent = "Capture failed: " + e.message;
  } finally {
    camCapture.disabled = false;
  }
}

camStart.addEventListener("click", camStartFn);
camStop.addEventListener("click", camStopFn);
camCapture.addEventListener("click", camCaptureFn);
camDevice.addEventListener("change", () => {
  if (camStream) { camStopFn(); setTimeout(camStartFn, 200); }
});
// Space bar captures while the camera is live (and we aren't typing in a field).
document.addEventListener("keydown", (e) => {
  if (e.code !== "Space") return;
  if (!camStream) return;
  if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA")) return;
  e.preventDefault();
  camCaptureFn();
});
window.addEventListener("pagehide", camStopFn);

// ---- import from event cameras -------------------------------------------

const evCamSel    = document.getElementById("cw-event-cam");
const evHoursSel  = document.getElementById("cw-event-hours");
const evLoadBtn   = document.getElementById("cw-event-load");
const evStatus    = document.getElementById("cw-event-status");
const evGrid      = document.getElementById("cw-event-grid");
const evActions   = document.getElementById("cw-event-actions");
const evSelCount  = document.getElementById("cw-event-selcount");
const evSelAll    = document.getElementById("cw-event-selectall");
const evSelNone   = document.getElementById("cw-event-selectnone");
const evImport    = document.getElementById("cw-event-import");

let evSelected = new Set();   // snapshot ids the user has ticked
let evAvailable = [];         // {id, captured_at, file_rel, person_count}[]

function evRenderGrid() {
  evGrid.innerHTML = evAvailable.map((s) => `
    <label class="cw-event-thumb" data-id="${s.id}">
      <input type="checkbox" data-id="${s.id}" ${evSelected.has(s.id) ? "checked" : ""}>
      <img src="/api/snapshots/${s.file_rel}" loading="lazy" alt="">
      <span class="cw-event-meta">
        ${new Date(s.captured_at * 1000).toLocaleString()}
        ${s.person_count ? ` · ${s.person_count} person${s.person_count === 1 ? "" : "s"}` : ""}
      </span>
    </label>
  `).join("");
  evGrid.querySelectorAll('input[type="checkbox"]').forEach((cb) => {
    cb.addEventListener("change", (e) => {
      const id = Number(cb.dataset.id);
      if (cb.checked) evSelected.add(id);
      else evSelected.delete(id);
      updateSelCount();
    });
  });
  updateSelCount();
}
function updateSelCount() {
  evSelCount.textContent = `${evSelected.size} selected`;
  evImport.disabled = evSelected.size === 0;
}

async function evLoadCameras() {
  let cams = [];
  try {
    const r = await fetch("/api/cameras", { cache: "no-store" });
    if (r.ok) cams = await r.json();
  } catch (e) {}
  const events = cams.filter((c) => (c.category || c.type) === "event");
  if (events.length === 0) {
    evStatus.textContent = "No event cameras configured. Add one via /cameras to start collecting snapshots.";
    evLoadBtn.disabled = true;
    return;
  }
  evCamSel.innerHTML = events.map((c) =>
    `<option value="${c.id}">${c.name || c.id}</option>`
  ).join("");
}

async function evLoadAvailable() {
  const camId = evCamSel.value;
  const hours = evHoursSel.value;
  if (!camId) return;
  evStatus.textContent = "Loading…";
  evGrid.hidden = true;
  evActions.hidden = true;
  evSelected.clear();
  try {
    const r = await fetch(
      `/api/event-snapshots?camera_id=${encodeURIComponent(camId)}`
      + `&cid=${encodeURIComponent(CID)}`
      + `&hours=${encodeURIComponent(hours)}`,
      { cache: "no-store" });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      evStatus.textContent = "Load failed: " + (err.error || r.status);
      return;
    }
    const data = await r.json();
    evAvailable = data.available || [];
    if (evAvailable.length === 0) {
      const prior = data.already_imported || 0;
      evStatus.textContent = prior > 0
        ? `No new snapshots in this window (${prior} already imported).`
        : "No snapshots captured in this window yet.";
      return;
    }
    evStatus.textContent = `${evAvailable.length} new snapshot${evAvailable.length === 1 ? "" : "s"} ready to import.`;
    evRenderGrid();
    evGrid.hidden = false;
    evActions.hidden = false;
  } catch (e) {
    evStatus.textContent = "Load failed: " + e.message;
  }
}

evLoadBtn.addEventListener("click", evLoadAvailable);
evCamSel.addEventListener("change", () => { /* hold until user clicks Load */ });

evSelAll.addEventListener("click", () => {
  evSelected = new Set(evAvailable.map((s) => s.id));
  evRenderGrid();
});
evSelNone.addEventListener("click", () => {
  evSelected.clear();
  evRenderGrid();
});

evImport.addEventListener("click", async () => {
  if (evSelected.size === 0) return;
  const ids = [...evSelected];
  evImport.disabled = true;
  evStatus.textContent = `Importing ${ids.length}…`;
  try {
    const r = await fetch(
      `/api/custom-classes/${encodeURIComponent(CID)}/import-snapshots`,
      { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ snapshot_ids: ids }) },
    );
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      evStatus.textContent = "Import failed: " + (err.error || r.status);
      evImport.disabled = false;
      return;
    }
    const data = await r.json();
    const okN = (data.saved || []).length;
    const errN = (data.errors || []).length;
    evStatus.textContent = errN
      ? `Imported ${okN}, skipped ${errN}.`
      : `Imported ${okN}. They appear in the image grid below — label them and train.`;
    // Re-load the available set so the imported ones disappear.
    evSelected.clear();
    evLoadAvailable();
    refresh();
  } catch (e) {
    evStatus.textContent = "Import failed: " + e.message;
    evImport.disabled = false;
  }
});

evLoadCameras();

btnDelete.addEventListener("click", async () => {
  if (!confirm("Delete this entire class and every image?\nThis cannot be undone.")) return;
  const r = await fetch(`/api/custom-classes/${encodeURIComponent(CID)}`, { method: "DELETE" });
  if (r.ok) location.href = "/classes";
  else alert("Delete failed");
});

// ---- training --------------------------------------------------------

async function pollTraining() {
  try {
    const r = await fetch(`/api/custom-classes/${encodeURIComponent(CID)}/train/status`,
      { cache: "no-store" });
    if (!r.ok) return;
    const s = await r.json();
    if (!s.running && !s.return_code && !s.epochs_done) {
      // No training started this session; hide the card.
      trainCard.hidden = true;
      return;
    }
    trainCard.hidden = false;
    const done = s.epochs_done || 0;
    const total = s.epochs_total || 0;
    const pct = total ? Math.min(100, Math.round((done / total) * 100)) : 0;
    trainBar.style.width = `${pct}%`;
    if (s.running) {
      const m = s.last_metrics || {};
      const loss = m["train/box_loss"] || m.box_loss || "";
      const map  = m["metrics/mAP50(B)"] || m["metrics/mAP50"] || "";
      const elapsed = s.started_at ? Math.round(Date.now() / 1000 - s.started_at) : 0;
      const mins = Math.floor(elapsed / 60), secs = elapsed % 60;
      trainTitle.textContent = `Training — epoch ${done} / ${total}`;
      trainMeta.textContent =
        `${pct}% · ${mins}m ${secs}s elapsed`
        + (loss ? ` · box loss ${(+loss).toFixed(4)}` : "")
        + (map ? ` · mAP@.5 ${(+map).toFixed(3)}` : "");
    } else if (s.return_code === 0) {
      trainTitle.textContent = "Training complete";
      trainMeta.textContent = `${done} / ${total} epochs · model saved to custom_classes/${CID}/model/best.pt`;
      trainBar.style.width = "100%";
      stopTrainPolling();
      refresh();
    } else if (s.return_code != null) {
      trainTitle.textContent = `Training failed (exit ${s.return_code})`;
      trainMeta.textContent = "Check the log below.";
      stopTrainPolling();
    }
    if (s.log_tail) trainLog.textContent = s.log_tail.join("\n");
  } catch (e) {
    // network blip — keep polling
  }
}

function startTrainPolling() {
  if (trainPollHandle) return;
  pollTraining();
  trainPollHandle = setInterval(pollTraining, 4000);
}
function stopTrainPolling() {
  if (trainPollHandle) clearInterval(trainPollHandle);
  trainPollHandle = null;
}

if (btnTrain) {
  btnTrain.addEventListener("click", async () => {
    if (!confirm(
      "Spawn a YOLO11s fine-tune on the GPU? Roughly 15-45 minutes "
      + "depending on dataset size. The live cameras keep running."
    )) return;
    btnTrain.disabled = true;
    const r = await fetch(`/api/custom-classes/${encodeURIComponent(CID)}/train`,
      { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ epochs: 50, imgsz: 640, batch: 8 }) });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      alert("Training failed to start: " + (err.error || r.status));
      btnTrain.disabled = false;
      return;
    }
    trainCard.hidden = false;
    trainTitle.textContent = "Training queued…";
    trainMeta.textContent = "Loading YOLO11s base weights…";
    trainBar.style.width = "1%";
    startTrainPolling();
  });
}

// On page load, pick up an in-progress training (e.g. user refreshed
// during a long run).
fetch(`/api/custom-classes/${encodeURIComponent(CID)}/train/status`, { cache: "no-store" })
  .then((r) => r.ok ? r.json() : null)
  .then((s) => { if (s && s.running) startTrainPolling(); });

refresh();
