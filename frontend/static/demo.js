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

fpsSlider.addEventListener("input", () => { fpsVal.textContent = fpsSlider.value; });
confSlider.addEventListener("input", () => { confVal.textContent = parseFloat(confSlider.value).toFixed(2); });

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
      const r = await fetch("/api/demo/analyze", {
        method: "POST",
        headers: { "Content-Type": "image/jpeg" },
        body: blob,
      });
      const rtt = performance.now() - t0;
      if (r.ok) {
        const data = await r.json();
        draw(data);
        elInfMs.textContent = `${Math.round(data.inference_ms || 0)} ms`;
        elRttMs.textContent = `${Math.round(rtt)} ms`;
        const persons = (data.detections || []).filter(
          (d) => d.class === "person" && d.confidence >= +confSlider.value,
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
  const minConf = +confSlider.value;
  // The inference image may have been resized — scale boxes to the
  // displayed video resolution.
  const sx = overlay.width / (data.image_width || overlay.width);
  const sy = overlay.height / (data.image_height || overlay.height);

  for (const d of data.detections || []) {
    if (d.confidence < minConf) continue;
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
