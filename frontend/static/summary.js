// Summary page — 2-state ("in bed" / "out of bed") view derived from
// the same /api/history payload the History page uses. Single timeline,
// daily totals + episode list.

const SUM_BED_IN = "in";    // value used in CSS class .v-bed-in
const SUM_BED_OUT = "out";  // value used in CSS class .v-bed-out

// Anything that isn't out_of_frame counts as "in bed" — asleep / lying /
// sitting / moving_a_lot all mean the baby is in the crib.
function _asBed(activity) {
  return activity === "out_of_frame" ? SUM_BED_OUT : SUM_BED_IN;
}

const camSelect = document.getElementById("sum-cam-select");
const dateInput = document.getElementById("sum-date-input");
const summary = document.getElementById("sum-summary");
const bar = document.querySelector('.sum-bar[data-track="bed"]');
const hoursDiv = document.getElementById("sum-hours");

const elTotalIn = document.getElementById("sum-total-in");
const elTotalOut = document.getElementById("sum-total-out");
const elOutCount = document.getElementById("sum-out-count");
const elLongestIn = document.getElementById("sum-longest-in");
const elTotalSnaps = document.getElementById("sum-total-snaps");
const elTotalScored = document.getElementById("sum-total-scored");
const elScoredPct = document.getElementById("sum-scored-pct");
const elAccuracy = document.getElementById("sum-accuracy");
const elAccuracyDetail = document.getElementById("sum-accuracy-detail");

const periodsDiv = document.getElementById("sum-periods");
const periodsCount = document.getElementById("sum-periods-count");

let currentCam = null;
let currentDate = new Date().toISOString().slice(0, 10);
// Latest history payload — periods read snapshots from here so we
// don't re-fetch when a row is expanded.
let _snapshots = [];
dateInput.value = currentDate;

// ---- camera select ---------------------------------------------------

async function loadCameras() {
  let cams = [];
  try {
    const r = await fetch("/api/cameras", { cache: "no-store" });
    if (r.ok) cams = await r.json();
  } catch (e) { /* ignore */ }
  // Baby cameras only — Summary is the baby-monitor lens.
  cams = (cams || []).filter(
    (c) => (c.category || c.type || "").toLowerCase() === "baby" && c.enabled !== false
  );
  camSelect.innerHTML = "";
  if (cams.length === 0) {
    const opt = document.createElement("option");
    opt.textContent = "No baby cameras";
    camSelect.appendChild(opt);
    summary.textContent = "Add a camera with category Baby to start recording in-bed history.";
    return;
  }
  cams.forEach((c) => {
    const opt = document.createElement("option");
    opt.value = c.id;
    opt.textContent = c.name || c.id;
    camSelect.appendChild(opt);
  });
  currentCam = cams[0].id;
  reload();
}
loadCameras();

camSelect.addEventListener("change", () => {
  currentCam = camSelect.value;
  reload();
});

// ---- date controls ---------------------------------------------------

dateInput.addEventListener("change", () => {
  currentDate = dateInput.value;
  reload();
});

document.getElementById("sum-btn-prev").addEventListener("click", () => shiftDate(-1));
document.getElementById("sum-btn-next").addEventListener("click", () => shiftDate(1));
document.getElementById("sum-btn-today").addEventListener("click", () => {
  currentDate = new Date().toISOString().slice(0, 10);
  dateInput.value = currentDate;
  reload();
});

function shiftDate(deltaDays) {
  const d = new Date(currentDate + "T12:00:00");
  d.setDate(d.getDate() + deltaDays);
  currentDate = d.toISOString().slice(0, 10);
  dateInput.value = currentDate;
  reload();
}

// ---- main reload -----------------------------------------------------

async function reload() {
  if (!currentCam) return;
  summary.textContent = "Loading…";
  let data = null;
  try {
    const url = `/api/history/${currentCam}?date=${encodeURIComponent(currentDate)}`;
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    data = await r.json();
  } catch (e) {
    summary.textContent = "Failed to load: " + e.message;
    return;
  }
  render(data);
}

// ---- rendering -------------------------------------------------------

function render(data) {
  _snapshots = (data?.snapshots) || [];

  // Bed segments are now derived from SNAPSHOTS, not the raw activity
  // track. Each snapshot is a sample point that votes "in" or "out" of
  // bed based on:
  //   * label === "correct"   → trust the system's detection on this
  //                              frame (persons > 0 → in-bed)
  //   * label === "incorrect" → flip the system's detection (it was a
  //                              false positive or false negative)
  //   * label === null/absent → fall back to the system's detection
  // Adjacent samples with the same ground-truth value collapse into
  // one segment so the timeline reads as continuous blocks.
  const bedSegments = segmentsFromSnapshots(_snapshots);

  // Counters: how many snapshots were operator-touched and how many of
  // those flipped the system's decision. "correct" labels confirm the
  // system was right on that frame; "incorrect" labels flip the
  // outcome and count as a system mistake for accuracy purposes.
  let labeledTotal = 0, correctCount = 0, incorrectCount = 0;
  for (const s of _snapshots) {
    if (s.label === "correct") { labeledTotal += 1; correctCount += 1; }
    else if (s.label === "incorrect") { labeledTotal += 1; incorrectCount += 1; }
  }
  const totalSnaps = _snapshots.length;
  const accuracy = labeledTotal > 0 ? correctCount / labeledTotal : null;

  // Day bounds — use the data's date as midnight-to-midnight.
  let dayStart, dayEnd;
  if (bedSegments.length > 0) {
    const allStart = Math.min(...bedSegments.map((s) => s.start_ts));
    const dayMidnight = new Date(currentDate + "T00:00:00").getTime() / 1000;
    dayStart = Math.min(dayMidnight, allStart);
    dayEnd = dayStart + 86400;
  } else {
    dayStart = new Date(currentDate + "T00:00:00").getTime() / 1000;
    dayEnd = dayStart + 86400;
  }

  // Totals
  let totalIn = 0, totalOut = 0, longestIn = 0, outCount = 0;
  for (const s of bedSegments) {
    const dur = (s.end_ts - s.start_ts);
    if (s.value === SUM_BED_IN) {
      totalIn += dur;
      if (dur > longestIn) longestIn = dur;
    } else {
      totalOut += dur;
      outCount += 1;
    }
  }

  const labelHint = labeledTotal > 0
    ? ` · ${labeledTotal} snapshot${labeledTotal === 1 ? "" : "s"} labeled (${incorrectCount} flipped)`
    : "";
  summary.textContent =
    `${bedSegments.length} segment${bedSegments.length === 1 ? "" : "s"} for ${data.date || currentDate}${labelHint}`;
  elTotalIn.textContent = fmtDur(totalIn);
  elTotalOut.textContent = fmtDur(totalOut);
  elOutCount.textContent = outCount;
  elLongestIn.textContent = longestIn > 0 ? fmtDur(longestIn) : "—";

  // Scoring stats
  elTotalSnaps.textContent = totalSnaps;
  elTotalScored.textContent = labeledTotal;
  if (totalSnaps > 0) {
    const pct = Math.round((labeledTotal / totalSnaps) * 100);
    elScoredPct.textContent = `${pct}% of snapshots`;
  } else {
    elScoredPct.textContent = "—";
  }
  if (accuracy != null) {
    const pct = Math.round(accuracy * 1000) / 10;  // one decimal
    elAccuracy.textContent = `${pct}%`;
    elAccuracyDetail.textContent =
      `${correctCount} correct · ${incorrectCount} incorrect`;
    // Colour the value by score so the operator can read it at a glance.
    elAccuracy.className = "sum-stat-value " + (
      pct >= 95 ? "acc-excellent" :
      pct >= 80 ? "acc-good" :
      pct >= 60 ? "acc-fair" : "acc-poor"
    );
  } else {
    elAccuracy.textContent = "—";
    elAccuracyDetail.textContent = "Score snapshots in History to compute";
    elAccuracy.className = "sum-stat-value";
  }

  renderBar(bedSegments, dayStart, dayEnd);
  renderHourTicks(dayStart, dayEnd);
  renderPeriods(bedSegments);
}

// Build 2-state bed segments from the snapshot stream, honouring
// operator labels as ground-truth overrides.
function segmentsFromSnapshots(snaps) {
  if (!snaps || snaps.length === 0) return [];
  // Sort defensively — API already returns ascending captured_at but
  // re-sort in case a future change relaxes that.
  const sorted = [...snaps].sort((a, b) => a.captured_at - b.captured_at);

  // Each snapshot represents the interval running from the midpoint
  // between it and the previous snapshot to the midpoint with the
  // next. End-snapshots fall back to a half-capture-interval cap so
  // first / last samples still contribute reasonable spans.
  const FALLBACK_HALF_INTERVAL_S = 30;
  const points = sorted.map((s, i) => {
    const prev = sorted[i - 1];
    const next = sorted[i + 1];
    const start = prev
      ? (prev.captured_at + s.captured_at) / 2
      : s.captured_at - FALLBACK_HALF_INTERVAL_S;
    const end = next
      ? (s.captured_at + next.captured_at) / 2
      : s.captured_at + FALLBACK_HALF_INTERVAL_S;
    return { start_ts: start, end_ts: end, value: _bedFromSnapshot(s) };
  });
  return compressByValue(points);
}

// Resolve the 2-state ground truth for one snapshot.
//   - persons > 0 → system thinks baby is in frame (in-bed)
//   - label "correct"   → trust system
//   - label "incorrect" → flip
//   - no label          → trust system
function _bedFromSnapshot(s) {
  const persons = (s.state && s.state.person_count) || 0;
  let systemSaysIn = persons > 0;
  if (s.label === "incorrect") systemSaysIn = !systemSaysIn;
  // "correct" and unlabeled both fall through with systemSaysIn as-is.
  return systemSaysIn ? SUM_BED_IN : SUM_BED_OUT;
}

// Collapse adjacent segments that map to the same simplified value, e.g.
// asleep → lying → sitting (all "in") become one continuous "in" block.
function compressByValue(segs) {
  if (segs.length === 0) return [];
  const out = [];
  let cur = { ...segs[0] };
  for (let i = 1; i < segs.length; i++) {
    const s = segs[i];
    // Treat segments as adjacent if they touch within 0.5 s (DB stores
    // discrete records that may have tiny gaps from poll cadence).
    if (s.value === cur.value && s.start_ts - cur.end_ts < 0.5) {
      cur.end_ts = s.end_ts;
    } else {
      out.push(cur);
      cur = { ...s };
    }
  }
  out.push(cur);
  return out;
}

function renderBar(segs, dayStart, dayEnd) {
  bar.innerHTML = '<div class="hist-tooltip"></div>';
  const span = dayEnd - dayStart;
  for (const s of segs) {
    const left = ((s.start_ts - dayStart) / span) * 100;
    const width = ((s.end_ts - s.start_ts) / span) * 100;
    if (width <= 0) continue;
    const el = document.createElement("div");
    el.className = `seg v-bed-${s.value}`;
    el.style.left = left + "%";
    el.style.width = width + "%";
    el.dataset.value = s.value;
    el.dataset.start = s.start_ts;
    el.dataset.end = s.end_ts;
    el.title =
      (s.value === SUM_BED_IN ? "In bed" : "Out of bed") +
      "\n" + fmtClock(s.start_ts) + " → " + fmtClock(s.end_ts) +
      " (" + fmtDur(s.end_ts - s.start_ts) + ")";
    bar.appendChild(el);
  }
}

function renderHourTicks(dayStart, dayEnd) {
  hoursDiv.innerHTML = "";
  const span = dayEnd - dayStart;
  // Hourly ticks 0..24
  for (let h = 0; h <= 24; h += 3) {
    const ts = dayStart + h * 3600;
    const pct = ((ts - dayStart) / span) * 100;
    const tick = document.createElement("span");
    tick.className = "tick";
    tick.style.left = pct + "%";
    tick.textContent = String(h).padStart(2, "0") + ":00";
    hoursDiv.appendChild(tick);
  }
}

// Build one row per period (in-bed or out-of-bed). Header shows time
// range + label + duration. Click to expand: a thumbnail strip is
// rendered lazily from the already-loaded snapshots filtered to the
// period's time window.
function renderPeriods(segs) {
  periodsCount.textContent = segs.length ? `· ${segs.length}` : "";
  if (segs.length === 0) {
    periodsDiv.innerHTML =
      '<div class="sum-empty">No state segments recorded for this day.</div>';
    return;
  }
  // Newest first — most people scan recent activity.
  const sorted = [...segs].sort((a, b) => b.start_ts - a.start_ts);
  periodsDiv.innerHTML = sorted.map((s, idx) => {
    const label = s.value === SUM_BED_IN ? "In bed" : "Out of bed";
    const cls = `sum-period sum-period-${s.value}`;
    return `
      <div class="${cls}" data-idx="${idx}">
        <button class="sum-period-header" type="button" aria-expanded="false">
          <span class="sum-period-chevron">▸</span>
          <span class="sum-period-dot"></span>
          <span class="sum-period-label">${label}</span>
          <span class="sum-period-range">${fmtClock(s.start_ts)} → ${fmtClock(s.end_ts)}</span>
          <span class="sum-period-dur">${fmtDur(s.end_ts - s.start_ts)}</span>
        </button>
        <div class="sum-period-body" hidden></div>
      </div>
    `;
  }).join("");

  // Wire toggle handlers + lazy thumbnail render.
  periodsDiv.querySelectorAll(".sum-period").forEach((el) => {
    const seg = sorted[Number(el.dataset.idx)];
    const header = el.querySelector(".sum-period-header");
    const body = el.querySelector(".sum-period-body");
    const chevron = el.querySelector(".sum-period-chevron");
    header.addEventListener("click", () => {
      const isOpen = !body.hidden;
      if (isOpen) {
        body.hidden = true;
        header.setAttribute("aria-expanded", "false");
        chevron.textContent = "▸";
        return;
      }
      // Lazy render thumbnails on first open.
      if (body.dataset.rendered !== "1") {
        body.innerHTML = renderPeriodThumbs(seg);
        body.dataset.rendered = "1";
      }
      body.hidden = false;
      header.setAttribute("aria-expanded", "true");
      chevron.textContent = "▾";
    });
  });
}

function renderPeriodThumbs(period) {
  const start = period.start_ts;
  const end = period.end_ts;
  const inWin = _snapshots.filter(
    (s) => s.captured_at >= start - 0.5 && s.captured_at <= end + 0.5
  );
  if (inWin.length === 0) {
    return '<div class="sum-empty">No snapshots captured during this window.</div>';
  }
  // Hard cap so a very long in-bed stretch doesn't render thousands of
  // <img> tags. Evenly sample the window when over the cap.
  const MAX_THUMBS = 60;
  let picks = inWin;
  if (inWin.length > MAX_THUMBS) {
    const step = (inWin.length - 1) / (MAX_THUMBS - 1);
    picks = [];
    for (let i = 0; i < MAX_THUMBS; i++) {
      picks.push(inWin[Math.round(i * step)]);
    }
  }
  const html = picks.map((s) => {
    const url = `/api/snapshots/${s.file_rel}`;
    const cls = "sum-thumb" + (s.label ? ` sum-thumb-${s.label}` : "");
    const labelDot = s.label === "correct"
      ? '<span class="sum-thumb-label sum-thumb-label-correct" title="Marked correct">✓</span>'
      : s.label === "incorrect"
        ? '<span class="sum-thumb-label sum-thumb-label-incorrect" title="Marked incorrect — bed-state flipped">✗</span>'
        : "";
    return `
      <a class="${cls}" href="${url}" target="_blank" rel="noopener"
         title="${fmtClock(s.captured_at)}${s.label ? " (labeled " + s.label + ")" : ""}">
        <img loading="lazy" src="${url}" alt="${fmtClock(s.captured_at)}">
        ${labelDot}
        <span class="sum-thumb-time">${fmtClock(s.captured_at)}</span>
      </a>`;
  }).join("");
  const truncated = inWin.length > MAX_THUMBS
    ? `<div class="sum-thumbs-note">Showing ${MAX_THUMBS} of ${inWin.length} snapshots (evenly spaced).</div>`
    : "";
  return `<div class="sum-thumbs">${html}</div>${truncated}`;
}

// ---- helpers ---------------------------------------------------------

function fmtClock(unix) {
  const d = new Date(unix * 1000);
  return d.toTimeString().slice(0, 8);
}

function fmtDur(seconds) {
  seconds = Math.max(0, Math.round(seconds || 0));
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}
