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
  stats.textContent = `${me.image_count} image${me.image_count === 1 ? "" : "s"} · ${me.labeled_count} labeled`;
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

btnDelete.addEventListener("click", async () => {
  if (!confirm("Delete this entire class and every image?\nThis cannot be undone.")) return;
  const r = await fetch(`/api/custom-classes/${encodeURIComponent(CID)}`, { method: "DELETE" });
  if (r.ok) location.href = "/classes";
  else alert("Delete failed");
});

refresh();
