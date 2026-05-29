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
let _currentData = null;            // last /api/history payload
let _view = null;                   // {startTs, endTs} — shared zoom across tracks
const MIN_VIEW_SPAN = 60;           // 1 minute floor
dateInput.value = currentDate;

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

// ---- view (shared zoom/pan across the 3 tracks) ----------------------

function dayBoundsTs() {
  const d = new Date(currentDate + "T00:00:00");
  return [d.getTime() / 1000, d.getTime() / 1000 + 86400];
}

function setView(startTs, endTs, doRender = true) {
  const [dayStart, dayEnd] = dayBoundsTs();
  let span = Math.max(MIN_VIEW_SPAN, Math.min(86400, endTs - startTs));
  let s = Math.max(dayStart, startTs);
  let e = s + span;
  if (e > dayEnd) { e = dayEnd; s = e - span; if (s < dayStart) s = dayStart; }
  _view = { startTs: s, endTs: e };
  if (doRender) rerender();
}

function resetView() {
  const [dayStart] = dayBoundsTs();
  setView(dayStart, dayStart + 86400);
}

function rerender() {
  if (!_currentData || !_view) return;
  ["activity", "posture", "motion"].forEach((track) =>
    renderBar(track, (_currentData.tracks || {})[track] || []),
  );
  renderHourTicks();
  renderViewBadge();
  renderSnapshots(filterSnapsByView(_currentData.snapshots || []));
}

function pickTickStep(span) {
  if (span >= 18 * 3600) return 3 * 3600;
  if (span >=  9 * 3600) return 2 * 3600;
  if (span >=  4 * 3600) return 3600;
  if (span >=  2 * 3600) return 30 * 60;
  if (span >=      3600) return 15 * 60;
  if (span >=      1800) return 5 * 60;
  if (span >=       300) return 60;
  return 30;
}

function fmtTick(unix, step) {
  const d = new Date(unix * 1000);
  const HH = String(d.getHours()).padStart(2, "0");
  const MM = String(d.getMinutes()).padStart(2, "0");
  return step >= 3600 ? `${HH}:00` : `${HH}:${MM}`;
}

function fmtSpan(s) {
  if (s >= 3600) return `${(s / 3600).toFixed(s >= 36000 ? 0 : 1)}h`;
  if (s >= 60)   return `${Math.round(s / 60)}m`;
  return `${Math.round(s)}s`;
}

function renderHourTicks() {
  const span = _view.endTs - _view.startTs;
  const step = pickTickStep(span);
  hoursDiv.innerHTML = "";
  const firstTick = Math.ceil(_view.startTs / step) * step;
  for (let t = firstTick; t <= _view.endTs + 0.5; t += step) {
    const left = ((t - _view.startTs) / span) * 100;
    if (left < -0.5 || left > 100.5) continue;
    const el = document.createElement("span");
    el.className = "tick";
    el.style.left = left + "%";
    el.textContent = fmtTick(t, step);
    hoursDiv.appendChild(el);
  }
}

function renderViewBadge() {
  const span = _view.endTs - _view.startTs;
  const isFull = span >= 86400 - 1;
  const fromI = document.getElementById("view-from");
  const toI   = document.getElementById("view-to");
  // Don't yank a value out from under the user while they're typing.
  if (document.activeElement !== fromI) fromI.value = fmtHHMM(_view.startTs);
  if (document.activeElement !== toI)   toI.value   = fmtHHMM(_view.endTs);
  document.getElementById("view-span").textContent = fmtSpan(span);
  document.getElementById("zoomed-pill").style.display = isFull ? "none" : "";
}

function fmtHHMM(unix) {
  const d = new Date(Number(unix) * 1000);
  return String(d.getHours()).padStart(2, "0") + ":" +
         String(d.getMinutes()).padStart(2, "0");
}

function parseHHMM(s) {
  if (!s || !/^\d{1,2}:\d{2}$/.test(s)) return null;
  const [h, m] = s.split(":").map(Number);
  if (h < 0 || h > 23 || m < 0 || m > 59) return null;
  return h * 3600 + m * 60;
}

function filterSnapsByView(snaps) {
  if (!_view) return snaps;
  return snaps.filter((s) =>
    s.captured_at >= _view.startTs && s.captured_at < _view.endTs,
  );
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

  _currentData = data;
  // New date = reset zoom to full day. Re-fetches on the same date preserve view.
  const [dayStart] = dayBoundsTs();
  if (!_view
      || _view.startTs < dayStart
      || _view.endTs > dayStart + 86400 + 1) {
    setView(dayStart, dayStart + 86400, false);
  }

  const tracks = data.tracks || {};
  const totalSegments = (tracks.activity?.length || 0)
                       + (tracks.posture?.length || 0)
                       + (tracks.motion?.length || 0);
  const totalSnaps = (data.snapshots || []).length;
  summary.textContent =
    `${totalSegments} segment${totalSegments === 1 ? "" : "s"} · ` +
    `${totalSnaps} snapshot${totalSnaps === 1 ? "" : "s"} for ${data.date}`;

  rerender();
  renderTotals(data.totals || {});
}

function renderBar(track, segments) {
  if (!_view) return;
  const bar = document.querySelector(`.hist-bar[data-track="${track}"]`);
  if (!bar) return;
  const tt = bar.querySelector(".hist-tooltip");
  bar.querySelectorAll(".seg, .nowmark").forEach((el) => el.remove());
  const span = _view.endTs - _view.startTs;
  for (const s of segments) {
    // Clip segment to the visible window
    const cs = Math.max(s.start_ts, _view.startTs);
    const ce = Math.min(s.end_ts,   _view.endTs);
    if (ce <= cs) continue;
    const left  = ((cs - _view.startTs) / span) * 100;
    const width = ((ce - cs) / span) * 100;
    if (width < 0.02) continue;
    const el = document.createElement("div");
    el.className = `seg ${VALUE_CLASS(track, s.value)}`;
    el.style.left  = left + "%";
    el.style.width = width + "%";
    el.dataset.track = track;
    el.dataset.value = s.value;
    el.dataset.start = s.start_ts;     // keep TRUE bounds for tooltip
    el.dataset.end   = s.end_ts;
    el.dataset.duration = s.duration_s;
    bar.appendChild(el);
  }
  // "now" marker on today, only if it falls inside the current view
  const today = new Date().toISOString().slice(0, 10);
  if (currentDate === today) {
    const now = Date.now() / 1000;
    if (now >= _view.startTs && now <= _view.endTs) {
      const m = document.createElement("div");
      m.className = "nowmark";
      m.style.left = ((now - _view.startTs) / span) * 100 + "%";
      bar.appendChild(m);
    }
  }
  // mouse tooltip — show segment's true (un-clipped) bounds
  bar.onmousemove = (e) => {
    if (bar.classList.contains("dragging")) { tt.classList.remove("show"); return; }
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

// ---- mouse zoom + pan on all three bars (shared view) ----------------

function zoomAround(anchorPct, factor) {
  if (!_view) return;
  const span = _view.endTs - _view.startTs;
  const anchorTs = _view.startTs + span * anchorPct;
  const newSpan = Math.max(MIN_VIEW_SPAN, Math.min(86400, span * factor));
  setView(anchorTs - newSpan * anchorPct, anchorTs + newSpan * (1 - anchorPct));
}

function attachBarInteractions() {
  document.querySelectorAll(".hist-bar").forEach((bar) => {
    bar.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = bar.getBoundingClientRect();
      const anchorPct = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
      const factor = e.deltaY > 0 ? 1.25 : 0.8;  // out / in
      zoomAround(anchorPct, factor);
    }, { passive: false });

    bar.addEventListener("dblclick", (e) => {
      e.preventDefault();
      resetView();
    });

    let drag = null;
    bar.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      drag = {
        startX: e.clientX,
        startView: { ..._view },
        moved: false,
      };
      bar.setPointerCapture(e.pointerId);
    });
    bar.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.startX;
      if (Math.abs(dx) > 2) {
        drag.moved = true;
        bar.classList.add("dragging");
        const r = bar.getBoundingClientRect();
        const span = drag.startView.endTs - drag.startView.startTs;
        const shift = -(dx / r.width) * span;
        setView(drag.startView.startTs + shift, drag.startView.endTs + shift);
      }
    });
    bar.addEventListener("pointerup", () => {
      drag = null;
      bar.classList.remove("dragging");
    });
    bar.addEventListener("pointercancel", () => {
      drag = null;
      bar.classList.remove("dragging");
    });
  });

  document.getElementById("zoom-in").addEventListener("click", () => zoomAround(0.5, 0.5));
  document.getElementById("zoom-out").addEventListener("click", () => zoomAround(0.5, 2.0));
  document.getElementById("zoom-reset").addEventListener("click", resetView);

  // Manual time range inputs — typing/picking a value sets the view directly.
  document.getElementById("view-from").addEventListener("change", (e) => {
    const sec = parseHHMM(e.target.value);
    if (sec === null || !_view) return;
    const [dayStart] = dayBoundsTs();
    let newStart = dayStart + sec;
    let newEnd   = _view.endTs;
    if (newStart >= newEnd - MIN_VIEW_SPAN) {
      newEnd = newStart + MIN_VIEW_SPAN;          // push the right edge out
    }
    setView(newStart, newEnd);
  });
  document.getElementById("view-to").addEventListener("change", (e) => {
    const sec = parseHHMM(e.target.value);
    if (sec === null || !_view) return;
    const [dayStart] = dayBoundsTs();
    let newEnd   = dayStart + sec;
    let newStart = _view.startTs;
    if (newEnd <= newStart + MIN_VIEW_SPAN) {
      newStart = newEnd - MIN_VIEW_SPAN;          // pull the left edge back
    }
    if (newEnd > dayStart + 86400) newEnd = dayStart + 86400;
    setView(newStart, newEnd);
  });
}

attachBarInteractions();

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
    const clipDot = s.clip_rel
      ? '<span class="snap-clipdot" title="Has audio/video clip">▶</span>' : "";
    return `
      <div class="snap-cell" data-idx="${idx}">
        <img loading="lazy" src="${url}" alt="${fmtClock(s.captured_at)}">
        ${clipDot}
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

  // Video clip: show if the snapshot has one, otherwise hide the block.
  const clipWrap = document.getElementById("lb-clip-wrap");
  const clipEl   = document.getElementById("lb-clip");
  if (snap.clip_rel) {
    clipEl.src = `/api/snapshots/${snap.clip_rel}`;
    clipEl.load();
    clipWrap.style.display = "";
  } else {
    clipEl.removeAttribute("src");
    clipEl.load();
    clipWrap.style.display = "none";
  }

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
  el.addEventListener("click", () => {
    const clipEl = document.getElementById("lb-clip");
    try { clipEl.pause(); } catch (e) {}
    lightbox.classList.add("hidden");
  }),
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
