// Live camera demo — browser camera → POST → GPU YOLO → draw boxes.

const video   = document.getElementById("demo-video");
const overlay = document.getElementById("demo-overlay");
const ctx     = overlay.getContext("2d");
const status  = document.getElementById("demo-status");
const startBtn = document.getElementById("demo-start");
const stopBtn  = document.getElementById("demo-stop");
const camSelect = document.getElementById("demo-camera");
const camWrap   = document.getElementById("demo-cam-wrap");
const fpsSlider = document.getElementById("demo-fps");
const fpsVal    = document.getElementById("demo-fps-val");
const confSlider = document.getElementById("demo-conf");
const confVal    = document.getElementById("demo-conf-val");

const elRealFps = document.getElementById("demo-real-fps");
const elInfMs   = document.getElementById("demo-inf-ms");
const elRttMs   = document.getElementById("demo-rtt-ms");
const elPersons = document.getElementById("demo-persons");

// Class palette — pick distinct colours for the common ones.
const CLS_COLOR = {
  person: "#0bb0ba",   // accent teal
  chair: "#6342dd",
  laptop: "#6ba9e0",
  "cell phone": "#f08a1f",
  cup: "#18a052",
  book: "#d99300",
};
// Custom-class palette — cycle through high-contrast colours so every
// trained custom class gets its own, indexed by source (cid).
const CUSTOM_PALETTE = ["#f5c518", "#ff5d8f", "#21d4fd", "#a4e72b", "#ff8a3c"];
const _customColorByCid = new Map();
function colorForCustom(cid) {
  if (!_customColorByCid.has(cid)) {
    _customColorByCid.set(
      cid,
      CUSTOM_PALETTE[_customColorByCid.size % CUSTOM_PALETTE.length],
    );
  }
  return _customColorByCid.get(cid);
}
// A detection carries source=<cid> when it came from a trained custom
// model; "base" (or missing) means COCO. Give customs their own palette
// so the user instantly sees "the system found IQOS" vs a person.
function colorFor(d) {
  if (d.source && d.source !== "base") return colorForCustom(d.source);
  return CLS_COLOR[d.class] || "#a472d9";
}

let stream = null;
let running = false;
let inflight = false;
let frameTimes = [];

// ---- active-class filter -------------------------------------------------
//
// Enabled classes are stored as a Set<string>. Defaults: only "person"
// from COCO + every custom class returned by the backend (added in
// rebuildCustomChips on first response). User toggles persist to
// localStorage so the panel keeps state across reloads.
const LS_KEY = "demo.enabledClasses.v1";
const DEFAULTS = new Set(["person"]);     // COCO defaults; customs default-on
const POSE_LS  = "demo.showPose.v1";
let enabledClasses;
try {
  const raw = JSON.parse(localStorage.getItem(LS_KEY) || "null");
  enabledClasses = new Set(Array.isArray(raw) ? raw : Array.from(DEFAULTS));
} catch { enabledClasses = new Set(DEFAULTS); }
const knownCustoms = new Set();           // populated as the server reports them

function persistClasses() {
  localStorage.setItem(LS_KEY, JSON.stringify([...enabledClasses]));
}

function classCounterText() {
  return `(${enabledClasses.size} on)`;
}
function refreshClassChips() {
  document.querySelectorAll("[data-cls]").forEach((el) => {
    el.classList.toggle("on", enabledClasses.has(el.dataset.cls));
  });
  const counter = document.getElementById("demo-class-counter");
  if (counter) counter.textContent = classCounterText();
}
function toggleClass(name) {
  if (enabledClasses.has(name)) enabledClasses.delete(name);
  else enabledClasses.add(name);
  persistClasses();
  refreshClassChips();
}

// Custom classes arrive in the analyze response. Render them as a
// dedicated row above the COCO list and default-enable any new ones.
function rebuildCustomChips(customs) {
  if (!customs || !customs.length) return;
  const wrap = document.getElementById("demo-custom-group");
  const ul   = document.getElementById("demo-custom-chips");
  if (!wrap || !ul) return;
  let changed = false;
  for (const c of customs) {
    if (!knownCustoms.has(c.name)) {
      knownCustoms.add(c.name);
      if (!enabledClasses.has(c.name)) {
        enabledClasses.add(c.name);    // custom classes default-on
        changed = true;
      }
    }
  }
  // Re-render the custom chip list to reflect any new arrivals.
  ul.innerHTML = customs.map((c) => {
    const safe = c.name.replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[ch]));
    return `<li data-cls="${safe}" data-custom="1">${safe}</li>`;
  }).join("");
  wrap.hidden = false;
  if (changed) persistClasses();
  refreshClassChips();
}

document.addEventListener("click", (e) => {
  const chip = e.target.closest("[data-cls]");
  if (!chip) return;
  toggleClass(chip.dataset.cls);
});
const btnDefault = document.getElementById("demo-class-default");
const btnClear   = document.getElementById("demo-class-clear");
const btnAll     = document.getElementById("demo-class-all");
btnDefault.addEventListener("click", () => {
  enabledClasses = new Set([...DEFAULTS, ...knownCustoms]);
  persistClasses();
  refreshClassChips();
});
btnClear.addEventListener("click", () => {
  // Drop every COCO chip; keep customs.
  document.querySelectorAll("[data-cls]:not([data-custom])").forEach(
    (el) => enabledClasses.delete(el.dataset.cls),
  );
  persistClasses();
  refreshClassChips();
});
btnAll.addEventListener("click", () => {
  document.querySelectorAll("[data-cls]:not([data-custom])").forEach(
    (el) => enabledClasses.add(el.dataset.cls),
  );
  persistClasses();
  refreshClassChips();
});

// ---- pose toggle ---------------------------------------------------------
const poseToggle = document.getElementById("demo-pose");
poseToggle.checked = localStorage.getItem(POSE_LS) === "1";
poseToggle.addEventListener("change", () => {
  localStorage.setItem(POSE_LS, poseToggle.checked ? "1" : "0");
});

refreshClassChips();

fpsSlider.addEventListener("input", () => { fpsVal.textContent = fpsSlider.value; });
// Slider is now a percent (1-100) for readability. Internally we
// always compare against detection confidences which come back as
// 0..1 floats, so confThreshold() does the divide-by-100 in one place.
function confThreshold() { return (+confSlider.value || 0) / 100; }
confSlider.addEventListener("input", () => { confVal.textContent = `${+confSlider.value}%`; });

async function listCameras() {
  // Enumerate AFTER the user grants permission (labels are otherwise empty).
  const devs = await navigator.mediaDevices.enumerateDevices();
  const cams = devs.filter((d) => d.kind === "videoinput");
  camSelect.innerHTML = cams.map((c, i) =>
    `<option value="${c.deviceId}">${c.label || `Camera ${i + 1}`}</option>`
  ).join("");
  camWrap.hidden = cams.length < 2;
}

async function start() {
  if (!navigator.mediaDevices?.getUserMedia) {
    status.textContent = "your browser doesn't support getUserMedia";
    return;
  }
  try {
    const constraints = {
      audio: false,
      video: camSelect.value
        ? { deviceId: { exact: camSelect.value }, width: 640, height: 480 }
        : { width: 640, height: 480, facingMode: "user" },
    };
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    await video.play();
    // Re-list cameras now that labels are populated.
    await listCameras();
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;
    running = true;
    startBtn.hidden = true;
    stopBtn.hidden = false;
    status.textContent = "";
    status.style.display = "none";
    loop();
  } catch (e) {
    status.textContent = "camera blocked: " + e.message;
  }
}

function stop() {
  running = false;
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  startBtn.hidden = false;
  stopBtn.hidden = true;
  status.textContent = "camera off";
  status.style.display = "";
}

async function loop() {
  if (!running) return;
  const targetFps = Math.max(1, Math.min(10, +fpsSlider.value || 4));
  const minInterval = 1000 / targetFps;
  const tickStart = performance.now();

  if (!inflight && video.readyState >= 2) {
    inflight = true;
    try {
      // Draw current video frame to a hidden canvas and convert to JPEG.
      const capCanvas = document.createElement("canvas");
      capCanvas.width = video.videoWidth;
      capCanvas.height = video.videoHeight;
      capCanvas.getContext("2d").drawImage(video, 0, 0);
      const blob = await new Promise((res) =>
        capCanvas.toBlob(res, "image/jpeg", 0.7),
      );

      const t0 = performance.now();
      const url = poseToggle.checked
        ? "/api/demo/analyze?pose=1"
        : "/api/demo/analyze";
      const r = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "image/jpeg" },
        body: blob,
      });
      const rtt = performance.now() - t0;
      if (r.ok) {
        const data = await r.json();
        rebuildCustomChips(data.custom_classes);
        draw(data);
        elInfMs.textContent = `${Math.round(data.inference_ms || 0)} ms`;
        elRttMs.textContent = `${Math.round(rtt)} ms`;
        const persons = (data.detections || []).filter(
          (d) => d.class === "person" && d.confidence >= confThreshold(),
        ).length;
        elPersons.textContent = persons;
        frameTimes.push(performance.now());
        frameTimes = frameTimes.filter((t) => t > performance.now() - 1000);
        elRealFps.textContent = frameTimes.length.toFixed(0);
      } else {
        const err = await r.json().catch(() => ({ error: r.statusText }));
        status.textContent = "Analyze error: " + (err.error || r.status);
        status.style.display = "";
      }
    } catch (e) {
      // Network blip; just skip this frame.
    } finally {
      inflight = false;
    }
  }

  // Schedule the next tick so we stay close to the chosen fps even
  // when the network is slow.
  const elapsed = performance.now() - tickStart;
  const wait = Math.max(0, minInterval - elapsed);
  setTimeout(loop, wait);
}

function draw(data) {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const minConf = confThreshold();
  // The inference image may have been resized — scale boxes to the
  // displayed video resolution.
  const sx = overlay.width / (data.image_width || overlay.width);
  const sy = overlay.height / (data.image_height || overlay.height);

  for (const d of data.detections || []) {
    if (d.confidence < minConf) continue;
    if (!enabledClasses.has(d.class)) continue;
    const col = colorFor(d);
    const [x1, y1, x2, y2] = d.box;
    const X1 = x1 * sx, Y1 = y1 * sy, X2 = x2 * sx, Y2 = y2 * sy;
    ctx.strokeStyle = col;
    ctx.lineWidth = 3;
    ctx.strokeRect(X1, Y1, X2 - X1, Y2 - Y1);
    const label = `${d.class} ${(d.confidence * 100).toFixed(0)}%`;
    ctx.font = "13px ui-monospace, monospace";
    const tw = ctx.measureText(label).width + 12;
    ctx.fillStyle = col;
    ctx.fillRect(X1, Y1 - 20, tw, 18);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, X1 + 6, Y1 - 6);
  }

  // Pose skeleton — drawn in yellow over the boxes. COCO 17-keypoint
  // layout: nose 0, eyes 1-2, ears 3-4, shoulders 5-6, elbows 7-8,
  // wrists 9-10, hips 11-12, knees 13-14, ankles 15-16.
  if (poseToggle.checked && Array.isArray(data.poses)) {
    drawPoses(data.poses, sx, sy, minConf);
  }
}

const POSE_EDGES = [
  [5, 7], [7, 9], [6, 8], [8, 10],          // arms
  [11, 13], [13, 15], [12, 14], [14, 16],    // legs
  [5, 6], [5, 11], [6, 12], [11, 12],        // torso
  [0, 1], [0, 2], [1, 3], [2, 4],            // face
];
const POSE_COLOR = "#f5c518";       // yellow — replaces the old pink
const POSE_KP_CONF = 0.3;            // hide noisy low-confidence keypoints

function drawPoses(poses, sx, sy, minConf) {
  ctx.strokeStyle = POSE_COLOR;
  ctx.fillStyle = POSE_COLOR;
  for (const p of poses) {
    if (p.confidence != null && p.confidence < minConf) continue;
    const kps = p.keypoints || [];
    // Edges
    ctx.lineWidth = 2.5;
    for (const [a, b] of POSE_EDGES) {
      const A = kps[a], B = kps[b];
      if (!A || !B || A[2] < POSE_KP_CONF || B[2] < POSE_KP_CONF) continue;
      ctx.beginPath();
      ctx.moveTo(A[0] * sx, A[1] * sy);
      ctx.lineTo(B[0] * sx, B[1] * sy);
      ctx.stroke();
    }
    // Joints
    for (const k of kps) {
      if (!k || k[2] < POSE_KP_CONF) continue;
      ctx.beginPath();
      ctx.arc(k[0] * sx, k[1] * sy, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

startBtn.addEventListener("click", start);
stopBtn.addEventListener("click", stop);
camSelect.addEventListener("change", () => {
  if (running) {
    stop();
    setTimeout(start, 200);
  }
});

// Stop the stream cleanly when the user navigates away.
window.addEventListener("pagehide", stop);
