// History page — 3 timeline tracks + snapshot gallery (baby cameras only).

const ACTIVITY_LABEL = {
  in_crib: "In crib", out_of_frame: "Out of frame",
  // Legacy values — only seen on snapshots captured before the
  // 06-05 simplification. New snapshots only emit in_crib /
  // out_of_frame.
  asleep: "Asleep", lying: "Lying", sitting: "Sitting",
  moving_a_lot: "Moving a lot",
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

// Format a Date as YYYY-MM-DD in the BROWSER's local timezone.
// Using toISOString() would convert to UTC and silently roll the date
// back near midnight in +03 / +0X timezones.
function ymdLocal(d = new Date()) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

let currentCam  = null;
let currentDate = ymdLocal();
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
  currentDate = ymdLocal(d);
  dateInput.value = currentDate;
  loadHistory();
}
document.getElementById("btn-prev").addEventListener("click", () => shiftDay(-1));
document.getElementById("btn-next").addEventListener("click", () => shiftDay(1));
document.getElementById("btn-today").addEventListener("click", () => {
  currentDate = ymdLocal();
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
  const today = ymdLocal();
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
        <div class="totals-row" data-track="${track}" data-value="${value}" role="button"
             title="Click to filter snapshots by ${lbl}">
          <span class="totals-swatch ${VALUE_CLASS(track, value)}"></span>
          <span class="totals-name">${lbl}</span>
          <span class="totals-dur">${fmtDur(sec)}</span>
          <span class="totals-pct">${pct}%</span>
        </div>`;
    }).join("");
    return `<div class="totals-col"><h4>${cap(track)}</h4>${rows}</div>`;
  }).join("");

  // Wire row clicks for snapshot filtering.
  totalsDiv.querySelectorAll(".totals-row").forEach((el) =>
    el.addEventListener("click", () =>
      setSnapshotFilter(el.dataset.track, el.dataset.value),
    ),
  );
  updateTotalsHighlight();
}

// ---- snapshot filtering by clicking a totals row ---------------------

let _activeFilter = null; // { track: 'activity', value: 'asleep' } | null
const totalsClearBtn = document.getElementById("totals-clear");

function setSnapshotFilter(track, value) {
  if (_activeFilter && _activeFilter.track === track && _activeFilter.value === value) {
    _activeFilter = null;
  } else {
    _activeFilter = { track, value };
  }
  renderSnapshots(_allSnapsForFilter);  // re-render with the new filter
  updateTotalsHighlight();
}

function clearSnapshotFilter() {
  if (!_activeFilter) return;
  _activeFilter = null;
  renderSnapshots(_allSnapsForFilter);
  updateTotalsHighlight();
}

function updateTotalsHighlight() {
  totalsDiv.querySelectorAll(".totals-row").forEach((el) => {
    const t = el.dataset.track;
    const v = el.dataset.value;
    const isActive = !!_activeFilter && _activeFilter.track === t && _activeFilter.value === v;
    el.classList.toggle("active", isActive);
  });
  if (totalsClearBtn) {
    totalsClearBtn.style.display = _activeFilter ? "" : "none";
  }
}

if (totalsClearBtn) {
  totalsClearBtn.addEventListener("click", clearSnapshotFilter);
}

// ---- label-status filter chips --------------------------------------

function setLabelFilter(name) {
  _labelFilter = name;
  document.querySelectorAll("[data-label-filter]").forEach((b) => {
    b.classList.toggle("active", b.dataset.labelFilter === name);
  });
  renderSnapshots(_allSnapsForFilter);
}

document.querySelectorAll("[data-label-filter]").forEach((b) => {
  b.addEventListener("click", () => setLabelFilter(b.dataset.labelFilter));
});

let _currentSnaps = [];
let _allSnapsForFilter = [];
// Label-status filter — "all" | "unlabeled" | "correct" | "incorrect".
// Composes with _activeFilter (activity/posture/motion totals filter)
// so you can stack e.g. "asleep AND unlabeled" to find unreviewed
// sleep frames in one click.
let _labelFilter = "all";

function renderSnapshots(snaps) {
  // `snaps` here is the full set for the day; filters are applied at
  // render time so toggling/clearing reruns without a refetch.
  _allSnapsForFilter = snaps;
  let visible = snaps;
  if (_activeFilter) {
    visible = visible.filter(
      (s) => s.state && s.state[_activeFilter.track] === _activeFilter.value,
    );
  }
  if (_labelFilter !== "all") {
    visible = visible.filter((s) => {
      if (_labelFilter === "unlabeled") return !s.label;
      return s.label === _labelFilter;
    });
  }
  _currentSnaps = visible;

  // Build a compact summary of what filters are active for the count badge.
  const filterParts = [];
  if (_activeFilter) {
    const lbl = TRACK_LABEL[_activeFilter.track]?.[_activeFilter.value] || _activeFilter.value;
    filterParts.push(`${_activeFilter.track}: ${lbl}`);
  }
  if (_labelFilter !== "all") {
    filterParts.push(_labelFilter);
  }
  if (snaps.length === 0) {
    snapCount.textContent = "";
  } else if (filterParts.length > 0) {
    snapCount.textContent = `· ${visible.length} of ${snaps.length} · ${filterParts.join(" · ")}`;
  } else {
    snapCount.textContent = `· ${snaps.length}`;
  }

  if (visible.length === 0) {
    snapGrid.innerHTML = filterParts.length > 0
      ? '<p class="muted">No snapshots match this filter.</p>'
      : '<p class="muted">No snapshots saved for this day yet.</p>';
    return;
  }
  snapGrid.innerHTML = visible.map((s, idx) => {
    const url = `/api/snapshots/${s.file_rel}`;
    const clipDot = s.clip_rel
      ? '<span class="snap-clipdot" title="Has audio/video clip">▶</span>' : "";
    const act = (s.state && s.state.activity) || "";
    const actCls = act ? ` snap-act-${act}` : "";
    const actLabel = act ? (ACTIVITY_LABEL[act] || act) : "";
    const labelCls = s.label ? ` snap-labeled-${s.label}` : "";
    const labelDot = s.label === "correct"
      ? '<span class="snap-label-dot snap-label-correct" title="Marked correct">✓</span>'
      : s.label === "incorrect"
        ? '<span class="snap-label-dot snap-label-incorrect" title="Marked incorrect">✗</span>'
        : "";
    const hasDrawn = Array.isArray(s.correction_boxes) && s.correction_boxes.length > 0;
    const drawnDot = hasDrawn
      ? '<span class="snap-drawn-dot" title="Operator-drawn box on this snapshot (training signal)">▢</span>'
      : "";
    return `
      <div class="snap-cell${actCls}${labelCls}${hasDrawn ? " snap-drawn" : ""}" data-idx="${idx}" title="${actLabel}">
        <img loading="lazy" src="${url}" alt="${fmtClock(s.captured_at)}">
        ${clipDot}
        ${labelDot}
        ${drawnDot}
        ${actLabel ? `<div class="snap-act-tag">${actLabel}</div>` : ""}
        <div class="snap-time">${fmtClock(s.captured_at)}</div>
      </div>`;
  }).join("");
  snapGrid.querySelectorAll(".snap-cell").forEach((el) => {
    el.addEventListener("click", (ev) => {
      const idx = Number(el.dataset.idx);
      const s = _currentSnaps[idx];
      if (!s) return;
      if (_selectMode) {
        // Alt+click selects the entire visual row of the clicked cell.
        if (ev.altKey) {
          selectRowOf(el);
          _lastSelIdx = idx;
          return;
        }
        // Shift+click selects a range from the last anchor to this idx.
        if (ev.shiftKey && _lastSelIdx != null) {
          const [lo, hi] = idx < _lastSelIdx ? [idx, _lastSelIdx] : [_lastSelIdx, idx];
          for (let i = lo; i <= hi; i++) {
            const ss = _currentSnaps[i];
            if (ss) _selectedIds.add(ss.id);
          }
        } else {
          // Toggle this single cell.
          if (_selectedIds.has(s.id)) _selectedIds.delete(s.id);
          else _selectedIds.add(s.id);
          _lastSelIdx = idx;
        }
        updateSelectionUI();
      } else {
        openLightbox(s);
      }
    });
  });
  // Re-apply selected styling after a re-render (e.g. after a filter change).
  if (_selectMode) {
    updateSelectionUI();
    refreshRowHandles();
  }
}

// ---- lightbox --------------------------------------------------------

const lightbox = document.getElementById("lightbox");

function setLightboxTab(name) {
  document.querySelectorAll("#lightbox .lb-tab").forEach((el) =>
    el.classList.toggle("active", el.dataset.tab === name),
  );
  document.querySelectorAll("#lightbox .lb-pane").forEach((el) =>
    el.classList.toggle("active", el.dataset.pane === name),
  );
  // Pause video if leaving the video tab.
  if (name !== "video") {
    const clipEl = document.getElementById("lb-clip");
    try { if (clipEl) clipEl.pause(); } catch (e) {}
  }
}

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

  // Video clip + tab availability.
  const clipEl   = document.getElementById("lb-clip");
  const noclip   = document.getElementById("lb-noclip");
  const tabVideo = document.getElementById("lb-tab-video");
  if (snap.clip_rel) {
    clipEl.src = `/api/snapshots/${snap.clip_rel}`;
    clipEl.load();
    clipEl.style.display = "";
    noclip.style.display = "none";
    tabVideo.disabled = false;
    tabVideo.classList.remove("disabled");
  } else {
    clipEl.removeAttribute("src");
    clipEl.load();
    clipEl.style.display = "none";
    noclip.style.display = "";
    // Tab stays visible but disabled so users see "no clip recorded".
    tabVideo.disabled = false;
    tabVideo.classList.add("disabled");
  }

  // Parameters
  document.getElementById("lb-params").innerHTML = renderParams(snap.state);
  // Label row state — track which snapshot is open so the buttons know
  // what to PATCH against.
  _openSnap = snap;
  syncLabelRow(snap.label || null);
  syncCorrectionVisibility();
  // Always default to the Images tab when opening — least disruptive.
  setLightboxTab("images");
  lightbox.classList.remove("hidden");
}

// Reveal a CTA in the lightbox that opens the dedicated draw-box page
// in a new tab when the operator has marked Incorrect on an
// out_of_frame snapshot. The drawing UI lives on a standalone
// /snapshot/<id>/correct page so it isn't fighting the lightbox modal
// for screen real estate.
// When the standalone correction page saves, it pokes a localStorage
// key — picked up here so the affected snapshot's badge updates without
// requiring a full page refresh.
window.addEventListener("storage", (e) => {
  if (e.key !== "primeanalyze.snapshotCorrectionSaved" || !e.newValue) return;
  let payload;
  try { payload = JSON.parse(e.newValue); } catch { return; }
  if (!payload || payload.snap_id == null) return;
  const id = Number(payload.snap_id);
  for (const arr of [_currentSnaps, _allSnapsForFilter]) {
    const m = arr.find((s) => s.id === id);
    if (m) m.correction_boxes = payload.box_count ? [[0, 0, 0, 0]] : [];
  }
  refreshSnapDrawnBadge(id);
  if (_openSnap && _openSnap.id === id) {
    _openSnap.correction_boxes = payload.box_count ? [[0, 0, 0, 0]] : [];
    syncCorrectionVisibility();
  }
});

function refreshSnapDrawnBadge(id) {
  const idx = _currentSnaps.findIndex((s) => s.id === id);
  if (idx < 0) return;
  const cell = snapGrid.querySelector(`.snap-cell[data-idx="${idx}"]`);
  if (!cell) return;
  const has = Array.isArray(_currentSnaps[idx].correction_boxes)
              && _currentSnaps[idx].correction_boxes.length > 0;
  cell.classList.toggle("snap-drawn", has);
  let dot = cell.querySelector(".snap-drawn-dot");
  if (has && !dot) {
    dot = document.createElement("span");
    dot.className = "snap-drawn-dot";
    dot.title = "Operator-drawn box on this snapshot (training signal)";
    dot.textContent = "▢";
    cell.appendChild(dot);
  } else if (!has && dot) {
    dot.remove();
  }
}

function syncCorrectionVisibility() {
  const cta = document.getElementById("lb-correction-cta");
  if (!cta || !_openSnap) return;
  const act = (_openSnap.state && _openSnap.state.activity) || "";
  const eligible = _openSnap.label === "incorrect" && act === "out_of_frame";
  cta.hidden = !eligible;
  if (!eligible) return;
  const link = document.getElementById("lb-correction-open");
  link.href = `/snapshot/${_openSnap.id}/correct`;
  const existing = document.getElementById("lb-correction-existing");
  const has = Array.isArray(_openSnap.correction_boxes) && _openSnap.correction_boxes.length > 0;
  existing.hidden = !has;
  link.textContent = has ? "↗ Adjust drawn box (new tab)" : "↗ Draw box (new tab)";
}

// Tab click handlers (set up once).
document.querySelectorAll("#lightbox .lb-tab").forEach((el) =>
  el.addEventListener("click", () => setLightboxTab(el.dataset.tab)),
);

// ---- snapshot labeling ----------------------------------------------

let _openSnap = null;

function syncLabelRow(label) {
  document.querySelectorAll("#lb-label-row .lb-label-btn").forEach((b) => {
    const v = b.dataset.label;
    const isActive =
      (label === "correct" && v === "correct") ||
      (label === "incorrect" && v === "incorrect") ||
      (!label && v === "");
    b.classList.toggle("active", isActive);
  });
  const status = document.getElementById("lb-label-status");
  if (label === "correct") {
    status.textContent = "Marked correct";
    status.className = "lb-label-status status-correct";
  } else if (label === "incorrect") {
    status.textContent = "Marked incorrect";
    status.className = "lb-label-status status-incorrect";
  } else {
    status.textContent = "Unlabeled — Summary uses the system's own classification";
    status.className = "lb-label-status status-none";
  }
}

async function labelSnapshot(value) {
  if (!_openSnap || _openSnap.id == null) return;
  const newLabel = value || null;
  const prev = _openSnap.label || null;
  if (newLabel === prev) return;
  // Optimistic local update so the grid + lightbox feel snappy.
  _openSnap.label = newLabel;
  syncLabelRow(newLabel);
  // Persist the in-memory copy in the grid arrays too.
  for (const arr of [_currentSnaps, _allSnapsForFilter]) {
    const m = arr.find((s) => s.id === _openSnap.id);
    if (m) m.label = newLabel;
  }
  // Refresh the grid cell so the badge/border updates without a full
  // re-render of the whole grid.
  refreshSnapCell(_openSnap);
  syncCorrectionVisibility();
  try {
    const r = await fetch(`/api/snapshots/${_openSnap.id}/label`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: newLabel }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
  } catch (e) {
    // Roll back on failure.
    _openSnap.label = prev;
    syncLabelRow(prev);
    for (const arr of [_currentSnaps, _allSnapsForFilter]) {
      const m = arr.find((s) => s.id === _openSnap.id);
      if (m) m.label = prev;
    }
    refreshSnapCell(_openSnap);
    const status = document.getElementById("lb-label-status");
    status.textContent = "Save failed: " + e.message;
    status.className = "lb-label-status status-error";
  }
}

function refreshSnapCell(snap) {
  const idx = _currentSnaps.findIndex((s) => s.id === snap.id);
  if (idx < 0) return;
  const cell = snapGrid.querySelector(`.snap-cell[data-idx="${idx}"]`);
  if (!cell) return;
  cell.classList.remove("snap-labeled-correct", "snap-labeled-incorrect");
  if (snap.label) cell.classList.add(`snap-labeled-${snap.label}`);
  let dot = cell.querySelector(".snap-label-dot");
  if (snap.label) {
    if (!dot) {
      dot = document.createElement("span");
      cell.appendChild(dot);
    }
    dot.className = `snap-label-dot snap-label-${snap.label}`;
    dot.textContent = snap.label === "correct" ? "✓" : "✗";
    dot.title = snap.label === "correct" ? "Marked correct" : "Marked incorrect";
  } else if (dot) {
    dot.remove();
  }
}

document.querySelectorAll("#lb-label-row .lb-label-btn").forEach((b) =>
  b.addEventListener("click", () => labelSnapshot(b.dataset.label)),
);

// ---- snapshot multi-select + bulk labeling --------------------------

let _selectMode = false;
let _selectedIds = new Set();
let _lastSelIdx = null;

const selectToggleBtn = document.getElementById("snap-select-toggle");
const bulkBar         = document.getElementById("snap-bulk-bar");
const bulkCountEl     = document.getElementById("snap-bulk-count");
const bulkStatusEl    = document.getElementById("snap-bulk-status");

function setSelectMode(on) {
  _selectMode = on;
  if (!on) {
    _selectedIds.clear();
    _lastSelIdx = null;
  }
  selectToggleBtn.textContent = on ? "Selecting…" : "Select";
  selectToggleBtn.classList.toggle("active", on);
  snapGrid.classList.toggle("select-mode", on);
  bulkBar.hidden = !on;
  const hint = document.getElementById("snap-bulk-hint");
  if (hint) hint.hidden = !on;
  updateSelectionUI();
  refreshRowHandles();
}

// Visual-row select: in selection mode, the leftmost cell of every
// visual row gets a small "Row" handle. Clicking it (or Alt+clicking
// any cell) selects every other cell sharing that cell's offsetTop —
// i.e. the whole row at the current viewport width.
function selectRowOf(rowAnchorCell) {
  const top = rowAnchorCell.offsetTop;
  snapGrid.querySelectorAll(".snap-cell").forEach((el) => {
    if (el.offsetTop === top) {
      const idx = Number(el.dataset.idx);
      const s = _currentSnaps[idx];
      if (s) _selectedIds.add(s.id);
    }
  });
  updateSelectionUI();
}

function refreshRowHandles() {
  // Tear down any existing handles first — the grid may have re-flowed
  // (resize, filter change, mode toggle) and the row anchors changed.
  snapGrid.querySelectorAll(".snap-row-handle").forEach((h) => h.remove());
  if (!_selectMode) return;
  const cells = Array.from(snapGrid.querySelectorAll(".snap-cell"));
  if (cells.length === 0) return;
  // Group cells by offsetTop; the cell at the smallest offsetLeft in
  // each group is the row anchor.
  const minLeftByTop = new Map();
  for (const c of cells) {
    const t = c.offsetTop;
    const l = c.offsetLeft;
    if (!minLeftByTop.has(t) || l < minLeftByTop.get(t)) {
      minLeftByTop.set(t, l);
    }
  }
  for (const c of cells) {
    if (c.offsetLeft !== minLeftByTop.get(c.offsetTop)) continue;
    const handle = document.createElement("button");
    handle.type = "button";
    handle.className = "snap-row-handle";
    handle.title = "Select entire row (or Alt+click any cell)";
    handle.textContent = "Row";
    handle.addEventListener("click", (ev) => {
      ev.stopPropagation();
      ev.preventDefault();
      selectRowOf(c);
    });
    c.appendChild(handle);
  }
}

// Re-compute row anchors when the grid re-flows. Debounced so it
// doesn't fire on every pixel of a window drag.
let _resizeTimer = null;
window.addEventListener("resize", () => {
  if (!_selectMode) return;
  clearTimeout(_resizeTimer);
  _resizeTimer = setTimeout(refreshRowHandles, 80);
});

function updateSelectionUI() {
  bulkCountEl.textContent = `${_selectedIds.size} selected`;
  snapGrid.querySelectorAll(".snap-cell").forEach((el) => {
    const idx = Number(el.dataset.idx);
    const s = _currentSnaps[idx];
    el.classList.toggle("snap-selected", !!(s && _selectedIds.has(s.id)));
  });
}

selectToggleBtn.addEventListener("click", () => setSelectMode(!_selectMode));
document.getElementById("snap-bulk-cancel").addEventListener("click", () =>
  setSelectMode(false),
);
document.getElementById("snap-bulk-all").addEventListener("click", () => {
  _currentSnaps.forEach((s) => _selectedIds.add(s.id));
  updateSelectionUI();
});
document.getElementById("snap-bulk-none").addEventListener("click", () => {
  _selectedIds.clear();
  updateSelectionUI();
});
document.getElementById("snap-bulk-correct").addEventListener("click", () =>
  bulkLabel("correct"),
);
document.getElementById("snap-bulk-incorrect").addEventListener("click", () =>
  bulkLabel("incorrect"),
);
document.getElementById("snap-bulk-clear").addEventListener("click", () =>
  bulkLabel(null),
);

async function bulkLabel(label) {
  if (_selectedIds.size === 0) {
    bulkStatusEl.textContent = "Nothing selected.";
    bulkStatusEl.className = "snap-bulk-status status-error";
    return;
  }
  const ids = Array.from(_selectedIds);
  // Snapshot of previous labels so we can roll back on failure.
  const prevLabels = new Map();
  for (const arr of [_currentSnaps, _allSnapsForFilter]) {
    for (const s of arr) {
      if (_selectedIds.has(s.id) && !prevLabels.has(s.id)) {
        prevLabels.set(s.id, s.label || null);
      }
    }
  }
  // Optimistic UI: apply the new label everywhere it could appear.
  for (const arr of [_currentSnaps, _allSnapsForFilter]) {
    for (const s of arr) {
      if (_selectedIds.has(s.id)) s.label = label;
    }
  }
  _currentSnaps.forEach((s, i) => {
    if (_selectedIds.has(s.id)) refreshSnapCellByIdx(i, s.label);
  });
  bulkStatusEl.textContent = `Saving ${ids.length}…`;
  bulkStatusEl.className = "snap-bulk-status status-pending";
  try {
    const r = await fetch("/api/snapshots/labels", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ids, label }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const body = await r.json();
    const verb = label === null ? "Cleared" : `Marked ${label} on`;
    bulkStatusEl.textContent = `${verb} ${body.updated ?? ids.length} snapshot${(body.updated ?? ids.length) === 1 ? "" : "s"}.`;
    bulkStatusEl.className = "snap-bulk-status status-ok";
  } catch (e) {
    // Roll back local state on failure.
    for (const arr of [_currentSnaps, _allSnapsForFilter]) {
      for (const s of arr) {
        if (prevLabels.has(s.id)) s.label = prevLabels.get(s.id);
      }
    }
    _currentSnaps.forEach((s, i) => {
      if (prevLabels.has(s.id)) refreshSnapCellByIdx(i, s.label);
    });
    bulkStatusEl.textContent = "Save failed: " + e.message;
    bulkStatusEl.className = "snap-bulk-status status-error";
  }
}

// Same as refreshSnapCell but skips findIndex when idx is already known.
function refreshSnapCellByIdx(idx, label) {
  const cell = snapGrid.querySelector(`.snap-cell[data-idx="${idx}"]`);
  if (!cell) return;
  cell.classList.remove("snap-labeled-correct", "snap-labeled-incorrect");
  if (label) cell.classList.add(`snap-labeled-${label}`);
  let dot = cell.querySelector(".snap-label-dot");
  if (label) {
    if (!dot) {
      dot = document.createElement("span");
      cell.appendChild(dot);
    }
    dot.className = `snap-label-dot snap-label-${label}`;
    dot.textContent = label === "correct" ? "✓" : "✗";
    dot.title = label === "correct" ? "Marked correct" : "Marked incorrect";
  } else if (dot) {
    dot.remove();
  }
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
  // Crib motion is only meaningful while the baby is in the crib;
  // 'undetected' is the worker's signal that there's no lock, so we
  // hide it rather than show a meaningless value.
  if (state.crib_motion && state.crib_motion !== "undetected") {
    const cm = state.crib_motion;
    push("Crib motion", cm === "still" ? "Still" : "Moving",
         cm === "still" ? "pill-cribmotion-still" : "pill-cribmotion-moving");
  }
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
  const today = ymdLocal();
  if (currentDate === today && currentCam) loadHistory();
}, 15000);
