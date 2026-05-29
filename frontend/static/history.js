// History page — 3 timeline tracks + snapshot gallery (baby cameras only).

const ACTIVITY_LABEL = {
  asleep: "Asleep", lying: "Lying", sitting: "Sitting",
  moving_a_lot: "Moving a lot", out_of_frame: "Out of frame",
  resting: "Resting", fidgeting: "Fidgeting", restless: "Restless",
  sitting_calm: "Sitting calm", playing: "Playing", very_active: "Very active",
  standing: "Standing", walking: "Walking", running: "Running",
  uncertain: "Uncertain",
};
const POSTURE_LABEL = {
  lying: "Lying", sitting: "Sitting", standing: "Standing",
  upright: "Upright", transitioning: "Transitioning", unknown: "Unknown",
};
const MOTION_LABEL = {
  still: "Still", moving: "Moving", active: "Active", unknown: "Unknown",
};

const TRACK_LABEL = { activity: ACTIVITY_LABEL, posture: POSTURE_LABEL, motion: MOTION_LABEL };

// Map every value to a CSS class so we can colour timelines + totals.
const VALUE_CLASS = (track, value) => `v-${track}-${value}`;

const camSelect  = document.getElementById("cam-select");
const dateInput  = document.getElementById("date-input");
const summary    = document.getElementById("hist-summary");
const totalsDiv  = document.getElementById("hist-totals");
const snapGrid   = document.getElementById("snap-grid");
const snapCount  = document.getElementById("snap-count");
const hoursDiv   = document.getElementById("hist-hours");

let currentCam  = null;
let currentDate = new Date().toISOString().slice(0, 10);
dateInput.value = currentDate;

renderHourLabels();

// ---- camera select ---------------------------------------------------

async function loadCameras() {
  let cams = [];
  try {
    const r = await fetch("/api/cameras", { cache: "no-store" });
    if (r.ok) cams = await r.json();
  } catch (e) {}
  const baby = cams.filter((c) => c.category === "baby" || c.type === "baby");
  if (baby.length === 0) {
    camSelect.innerHTML = '<option value="">No baby cameras configured</option>';
    summary.textContent = "Add a camera with category Baby to start recording history.";
    return;
  }
  camSelect.innerHTML = baby.map((c) =>
    `<option value="${c.id}">${escapeHtml(c.name)}</option>`,
  ).join("");
  currentCam = camSelect.value;
  loadHistory();
}

camSelect.addEventListener("change", () => {
  currentCam = camSelect.value;
  loadHistory();
});

dateInput.addEventListener("change", () => {
  currentDate = dateInput.value;
  loadHistory();
});

function shiftDay(days) {
  const d = new Date(currentDate + "T00:00:00");
  d.setDate(d.getDate() + days);
  currentDate = d.toISOString().slice(0, 10);
  dateInput.value = currentDate;
  loadHistory();
}
document.getElementById("btn-prev").addEventListener("click", () => shiftDay(-1));
document.getElementById("btn-next").addEventListener("click", () => shiftDay(1));
document.getElementById("btn-today").addEventListener("click", () => {
  currentDate = new Date().toISOString().slice(0, 10);
  dateInput.value = currentDate;
  loadHistory();
});

// ---- render hour labels (00:00 … 24:00) ------------------------------

function renderHourLabels() {
  let html = "";
  for (let h = 0; h <= 24; h += 3) {
    html += `<span>${String(h).padStart(2, "0")}:00</span>`;
  }
  hoursDiv.innerHTML = html;
}

// ---- load + render ---------------------------------------------------

async function loadHistory() {
  if (!currentCam) return;
  summary.textContent = "Loading…";
  let data;
  try {
    const r = await fetch(`/api/history/${currentCam}?date=${currentDate}`, { cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    data = await r.json();
  } catch (e) {
    summary.textContent = "Failed to load history: " + e.message;
    return;
  }

  const tracks = data.tracks || {};
  const totalSegments = (tracks.activity?.length || 0)
                       + (tracks.posture?.length || 0)
                       + (tracks.motion?.length || 0);
  const totalSnaps = (data.snapshots || []).length;
  summary.textContent =
    `${totalSegments} segment${totalSegments === 1 ? "" : "s"} · ` +
    `${totalSnaps} snapshot${totalSnaps === 1 ? "" : "s"} for ${data.date}`;

  ["activity", "posture", "motion"].forEach((track) =>
    renderBar(track, tracks[track] || [], data.date),
  );
  renderTotals(data.totals || {});
  renderSnapshots(data.snapshots || []);
}

function dayBounds(dateStr) {
  const d = new Date(dateStr + "T00:00:00");
  return [d.getTime() / 1000, d.getTime() / 1000 + 86400];
}

function renderBar(track, segments, date) {
  const bar = document.querySelector(`.hist-bar[data-track="${track}"]`);
  if (!bar) return;
  const tt = bar.querySelector(".hist-tooltip");
  bar.querySelectorAll(".seg, .nowmark").forEach((el) => el.remove());
  const [start_ts, end_ts] = dayBounds(date);
  for (const s of segments) {
    const left = ((s.start_ts - start_ts) / 86400) * 100;
    const width = ((s.end_ts - s.start_ts) / 86400) * 100;
    if (width < 0.05) continue;
    const el = document.createElement("div");
    el.className = `seg ${VALUE_CLASS(track, s.value)}`;
    el.style.left  = left + "%";
    el.style.width = width + "%";
    el.dataset.track = track;
    el.dataset.value = s.value;
    el.dataset.start = s.start_ts;
    el.dataset.end   = s.end_ts;
    el.dataset.duration = s.duration_s;
    bar.appendChild(el);
  }
  // "now" marker on today
  const today = new Date().toISOString().slice(0, 10);
  if (date === today) {
    const now = Date.now() / 1000;
    if (now >= start_ts && now <= end_ts) {
      const m = document.createElement("div");
      m.className = "nowmark";
      m.style.left = ((now - start_ts) / 86400) * 100 + "%";
      bar.appendChild(m);
    }
  }
  // mouse tooltip
  bar.onmousemove = (e) => {
    const t = e.target;
    if (!t.classList.contains("seg")) { tt.classList.remove("show"); return; }
    const lbl = TRACK_LABEL[t.dataset.track]?.[t.dataset.value] || t.dataset.value;
    tt.innerHTML =
      `<b>${lbl}</b><br>${fmtClock(t.dataset.start)} → ${fmtClock(t.dataset.end)}` +
      `<br><span class="muted">${fmtDur(t.dataset.duration)}</span>`;
    const r = bar.getBoundingClientRect();
    tt.style.left = (e.clientX - r.left + 10) + "px";
    tt.style.top  = "-46px";
    tt.classList.add("show");
  };
  bar.onmouseleave = () => tt.classList.remove("show");
}

function renderTotals(totals) {
  const tracks = ["activity", "posture", "motion"];
  totalsDiv.innerHTML = tracks.map((track) => {
    const entries = Object.entries(totals[track] || {}).sort((a, b) => b[1] - a[1]);
    const totalSec = entries.reduce((s, [, v]) => s + v, 0);
    if (totalSec === 0) {
      return `<div class="totals-col"><h4>${cap(track)}</h4><p class="muted">No data</p></div>`;
    }
    const rows = entries.map(([value, sec]) => {
      const lbl = TRACK_LABEL[track]?.[value] || value;
      const pct = ((sec / totalSec) * 100).toFixed(0);
      return `
        <div class="totals-row">
          <span class="totals-swatch ${VALUE_CLASS(track, value)}"></span>
          <span class="totals-name">${lbl}</span>
          <span class="totals-dur">${fmtDur(sec)}</span>
          <span class="totals-pct">${pct}%</span>
        </div>`;
    }).join("");
    return `<div class="totals-col"><h4>${cap(track)}</h4>${rows}</div>`;
  }).join("");
}

let _currentSnaps = [];

function renderSnapshots(snaps) {
  _currentSnaps = snaps;
  snapCount.textContent = snaps.length ? `· ${snaps.length}` : "";
  if (snaps.length === 0) {
    snapGrid.innerHTML = '<p class="muted">No snapshots saved for this day yet.</p>';
    return;
  }
  snapGrid.innerHTML = snaps.map((s, idx) => {
    const url = `/api/snapshots/${s.file_rel}`;
    return `
      <div class="snap-cell" data-idx="${idx}">
        <img loading="lazy" src="${url}" alt="${fmtClock(s.captured_at)}">
        <div class="snap-time">${fmtClock(s.captured_at)}</div>
      </div>`;
  }).join("");
  snapGrid.querySelectorAll(".snap-cell").forEach((el) => {
    el.addEventListener("click", () => {
      const s = _currentSnaps[Number(el.dataset.idx)];
      if (s) openLightbox(s);
    });
  });
}

// ---- lightbox --------------------------------------------------------

const lightbox = document.getElementById("lightbox");

function openLightbox(snap) {
  const annotated = `/api/snapshots/${snap.file_rel}`;
  const raw = annotated.replace(/\.jpg$/, "_raw.jpg");
  document.getElementById("lb-caption").textContent = fmtClock(snap.captured_at);
  // Reset raw column visibility (the onerror handler may have hidden it on a
  // previous snapshot that had no raw counterpart).
  const rawCol = document.getElementById("lb-img-raw").parentElement;
  rawCol.classList.remove("no-raw");
  document.getElementById("lb-img-raw").style.display = "";
  // Set sources fresh
  document.getElementById("lb-img-annotated").src = annotated;
  document.getElementById("lb-img-raw").src = raw;
  // Parameters
  document.getElementById("lb-params").innerHTML = renderParams(snap.state);
  lightbox.classList.remove("hidden");
}

function renderParams(state) {
  if (!state || typeof state !== "object") {
    return '<div class="muted">No parameters captured for this snapshot.</div>';
  }
  const rows = [];
  const push = (label, value, badge) => {
    if (value === null || value === undefined || value === "") return;
    rows.push(`
      <div class="param-row">
        <span class="param-k">${label}</span>
        <span class="param-v">${badge ? `<span class="badge-pill ${badge}">${escapeHtml(String(value))}</span>` : escapeHtml(String(value))}</span>
      </div>`);
  };
  const activity = state.activity;
  push("Activity",  activity ? (ACTIVITY_LABEL[activity] || activity) : null, activity ? `pill-act-${activity}` : "");
  push("Posture",   state.posture ? (POSTURE_LABEL[state.posture] || state.posture) : null);
  push("Motion",    state.motion ? (MOTION_LABEL[state.motion] || state.motion) : null);
  push("Persons",   state.person_count);
  push("Still for", state.still_seconds != null ? fmtDur(state.still_seconds) : null);
  push("Motion score", state.motion_score != null ? Number(state.motion_score).toFixed(3) : null);
  push("Body angle", state.posture_angle_deg != null ? Math.round(state.posture_angle_deg) + "°" : null);
  push("FPS", state.fps != null ? Number(state.fps).toFixed(1) : null);
  if (state.baby_lock) {
    push("Lock age", fmtDur(state.baby_lock.age_s));
    push("Lock confidence", state.baby_lock.confidence != null ? Number(state.baby_lock.confidence).toFixed(2) : null);
    if (state.baby_lock.box) {
      const b = state.baby_lock.box;
      push("Lock bbox", `(${b[0]}, ${b[1]}) → (${b[2]}, ${b[3]})`);
    }
  }
  if (state.camera_type) push("Camera type", state.camera_type);
  if (rows.length === 0) return '<div class="muted">No parameters captured.</div>';
  return rows.join("");
}
lightbox.querySelectorAll("[data-close]").forEach((el) =>
  el.addEventListener("click", () => lightbox.classList.add("hidden")),
);

// ---- helpers ---------------------------------------------------------

function fmtClock(unix) {
  const d = new Date(Number(unix) * 1000);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
}
function fmtDur(s) {
  s = Math.max(0, Math.round(Number(s) || 0));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}
function cap(s) { return s ? s[0].toUpperCase() + s.slice(1) : s; }
function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]),
  );
}

loadCameras();
// Auto-refresh "today" view every 15 s
setInterval(() => {
  const today = new Date().toISOString().slice(0, 10);
  if (currentDate === today && currentCam) loadHistory();
}, 15000);
