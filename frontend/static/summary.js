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

// Activity / motion bucketing for the overview rows. The user picked
// these specific buckets — everything else collapses into "other".
const ACTIVITY_BUCKETS = ["asleep", "lying", "sitting", "moving_a_lot"];
const ACTIVITY_LABELS = {
  asleep: "Asleep",
  lying: "Lying",
  sitting: "Sitting",
  moving_a_lot: "Moving a lot",
  other: "Other",
};
const ACTIVITY_VCLASS = {
  asleep: "v-activity-asleep",
  lying: "v-activity-lying",
  sitting: "v-activity-sitting",
  moving_a_lot: "v-activity-moving_a_lot",
  other: "v-activity-uncertain",
};

const MOTION_BUCKETS = ["still", "moving", "active"];
const MOTION_LABELS = {
  still: "Still",
  moving: "Moving",
  active: "Active",
  unknown: "Unknown",
};
const MOTION_VCLASS = {
  still: "v-motion-still",
  moving: "v-motion-moving",
  active: "v-motion-active",
  unknown: "v-motion-unknown",
};

const camSelect = document.getElementById("sum-cam-select");
const camSelect2 = document.getElementById("sum-cam-select-2");
const dateInput = document.getElementById("sum-date-input");
const overviewSection = document.getElementById("sum-overview");
const detailSection = document.getElementById("sum-detail");
const toolbarOverview = document.getElementById("sum-toolbar-overview");
const toolbarDetail = document.getElementById("sum-toolbar-detail");
const overviewSummary = document.getElementById("sum-overview-summary");
const dayRowsDiv = document.getElementById("sum-day-rows");
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

// Format a Date as YYYY-MM-DD in the BROWSER's local timezone.
// toISOString() always returns UTC and silently rolls the date back in
// +03 / +0X timezones between local midnight and 00:00 UTC.
function ymdLocal(d = new Date()) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

let currentCam = null;
let currentDate = ymdLocal();
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
    // Mirror into the detail-mode camera select so both stay in sync.
    if (camSelect2) {
      const opt2 = opt.cloneNode(true);
      camSelect2.appendChild(opt2);
    }
  });
  currentCam = cams[0].id;
  loadOverview();
}
loadCameras();

camSelect.addEventListener("change", () => {
  currentCam = camSelect.value;
  if (camSelect2) camSelect2.value = currentCam;
  _overviewCache.clear();
  loadOverview();
});
if (camSelect2) {
  camSelect2.addEventListener("change", () => {
    currentCam = camSelect2.value;
    camSelect.value = currentCam;
    _overviewCache.clear();
    reload();
  });
}

// ---- date controls ---------------------------------------------------

dateInput.addEventListener("change", () => {
  currentDate = dateInput.value;
  reload();
});

document.getElementById("sum-btn-prev").addEventListener("click", () => shiftDate(-1));
document.getElementById("sum-btn-next").addEventListener("click", () => shiftDate(1));
document.getElementById("sum-btn-today").addEventListener("click", () => {
  currentDate = ymdLocal();
  dateInput.value = currentDate;
  reload();
});

function shiftDate(deltaDays) {
  const d = new Date(currentDate + "T12:00:00");
  d.setDate(d.getDate() + deltaDays);
  currentDate = ymdLocal(d);
  dateInput.value = currentDate;
  reload();
}

// ---- view mode switching --------------------------------------------

function showOverview() {
  overviewSection.hidden = false;
  detailSection.hidden = true;
  toolbarOverview.hidden = false;
  toolbarDetail.hidden = true;
  // Refresh the system-wide stats so labels applied while in detail
  // mode are reflected in the overview totals.
  loadOverallStats();
}

function showDetail(date) {
  currentDate = date;
  dateInput.value = date;
  overviewSection.hidden = true;
  detailSection.hidden = false;
  toolbarOverview.hidden = true;
  toolbarDetail.hidden = false;
  // If we already have this day cached from the overview load, render
  // straight from the cache. Otherwise fetch.
  const cached = _overviewCache.get(date);
  if (cached) {
    render(cached);
  } else {
    reload();
  }
}

document.getElementById("sum-back-to-overview").addEventListener("click", () => {
  showOverview();
});

// ---- overview: list of recent days ----------------------------------

// Cache by date string so drill-down doesn't refetch.
const _overviewCache = new Map();
const OVERVIEW_DAYS = 4;  // today + 3 prior

async function loadOverview() {
  if (!currentCam) return;
  overviewSummary.textContent = "Loading recent days…";
  dayRowsDiv.innerHTML = "";

  // Kick off the system-wide aggregate fetch in parallel with the
  // per-day fetches — it doesn't block the day rows.
  loadOverallStats();

  // Build the list of dates: today first, then 3 days back.
  const today = new Date();
  const dates = [];
  for (let i = 0; i < OVERVIEW_DAYS; i++) {
    const d = new Date(today);
    d.setDate(d.getDate() - i);
    dates.push(ymdLocal(d));
  }

  // Fetch in parallel; tolerate per-day failures (e.g. older days
  // that have no data yet).
  const results = await Promise.all(dates.map(async (d) => {
    try {
      const r = await fetch(`/api/history/${currentCam}?date=${d}`, { cache: "no-store" });
      if (!r.ok) return null;
      return await r.json();
    } catch (e) {
      return null;
    }
  }));

  // Cache and convert to per-day stats.
  const rows = results.map((data, i) => {
    const date = dates[i];
    if (data) _overviewCache.set(date, data);
    return { date, isToday: i === 0, stats: data ? computeDayStats(data) : null };
  });

  // Today is always shown; earlier days only if they have data.
  const visible = rows.filter((r, i) => i === 0 || hasData(r.stats));

  if (visible.every((r) => !hasData(r.stats))) {
    overviewSummary.textContent =
      "No history recorded yet. The baby camera will start populating once it's been running for a few minutes.";
  } else {
    overviewSummary.textContent =
      `${visible.length} day${visible.length === 1 ? "" : "s"} of history. Click a row for details.`;
  }
  renderDayRows(visible);
}

function hasData(stats) {
  return !!stats && (stats.totalInBed > 0 || stats.totalOutBed > 0);
}

// System-wide aggregates for the selected camera. Three COUNT(*)
// queries on the hub — cheap, so we just refetch on every overview
// load rather than caching.
const elOverallTotal = document.getElementById("sum-overall-total");
const elOverallScored = document.getElementById("sum-overall-scored");
const elOverallUnscored = document.getElementById("sum-overall-unscored");
const elOverallScoredSub = document.getElementById("sum-overall-scored-sub");
const elOverallUnscoredSub = document.getElementById("sum-overall-unscored-sub");

async function loadOverallStats() {
  if (!currentCam) return;
  elOverallTotal.textContent = "…";
  elOverallScored.textContent = "…";
  elOverallUnscored.textContent = "…";
  elOverallScoredSub.textContent = "";
  elOverallUnscoredSub.textContent = "";
  try {
    const r = await fetch(`/api/snapshots/stats?cam=${encodeURIComponent(currentCam)}`, { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const s = await r.json();
    const total = s.total || 0;
    const scored = s.scored || 0;
    const unscored = s.unscored || 0;
    elOverallTotal.textContent = total.toLocaleString();
    elOverallScored.textContent = scored.toLocaleString();
    elOverallUnscored.textContent = unscored.toLocaleString();
    if (total > 0) {
      const scoredPct = Math.round((scored / total) * 100);
      const unscoredPct = 100 - scoredPct;
      elOverallScoredSub.textContent =
        `${scoredPct}% of total · ${s.correct || 0} correct · ${s.incorrect || 0} incorrect`;
      elOverallUnscoredSub.textContent = `${unscoredPct}% of total — still to label`;
    } else {
      elOverallScoredSub.textContent = "—";
      elOverallUnscoredSub.textContent = "—";
    }
  } catch (e) {
    elOverallTotal.textContent = "—";
    elOverallScored.textContent = "—";
    elOverallUnscored.textContent = "—";
    elOverallScoredSub.textContent = "Failed: " + e.message;
    elOverallUnscoredSub.textContent = "";
  }
}

// Per-day aggregates derived FROM SNAPSHOTS (not the raw activity
// track) so operator labels propagate to the overview row totals.
// Each snapshot covers the midpoint-to-midpoint interval between its
// neighbours; bed state is the label-corrected value (same logic the
// detail-view periods use). Activity attribution only counts in-bed
// time AND only when the system can be trusted on that frame:
//   correct        — trust system's activity classification
//   unlabelled     — trust system's activity classification
//   incorrect+FP   — already counts as out-of-bed (no activity)
//   incorrect+FN   — system missed; activity is unknown → "other"
function computeDayStats(data) {
  const snaps = data?.snapshots || [];
  if (snaps.length === 0) return null;
  const sorted = [...snaps].sort((a, b) => a.captured_at - b.captured_at);

  const FALLBACK_HALF_INTERVAL_S = 30;
  let totalInBed = 0;
  let totalOutBed = 0;
  const byActivity = {};
  const byMotion = {};

  for (let i = 0; i < sorted.length; i++) {
    const s = sorted[i];
    const prev = sorted[i - 1];
    const next = sorted[i + 1];
    const start = prev
      ? (prev.captured_at + s.captured_at) / 2
      : s.captured_at - FALLBACK_HALF_INTERVAL_S;
    const end = next
      ? (s.captured_at + next.captured_at) / 2
      : s.captured_at + FALLBACK_HALF_INTERVAL_S;
    const dur = Math.max(0, end - start);

    const bed = _bedFromSnapshot(s);
    if (bed === SUM_BED_IN) {
      totalInBed += dur;
      // Activity attribution: only trust the system's activity tag
      // when the system itself agreed there was a baby (persons>0)
      // AND you didn't say it was wrong. For label=incorrect+persons=0
      // (false negative — system missed a real baby), bucket time as
      // "other" since we know the baby was in bed but the system had
      // no activity classification.
      const persons = (s.state && s.state.person_count) || 0;
      if (persons > 0 && s.label !== "incorrect") {
        const rawActivity = (s.state && s.state.activity) || "other";
        const act = rawActivity !== "out_of_frame" ? rawActivity : "other";
        byActivity[act] = (byActivity[act] || 0) + dur;
      } else {
        byActivity.other = (byActivity.other || 0) + dur;
      }
    } else {
      totalOutBed += dur;
    }

    // Motion track is independent of label (it measures the actual
    // image-space motion regardless of what the system thinks).
    const mot = (s.state && s.state.motion) || "unknown";
    byMotion[mot] = (byMotion[mot] || 0) + dur;
  }

  // Activity %s computed over IN-BED time only.
  const activityPct = {};
  if (totalInBed > 0) {
    let listed = 0;
    for (const b of ACTIVITY_BUCKETS) {
      const sec = byActivity[b] || 0;
      activityPct[b] = (sec / totalInBed) * 100;
      listed += sec;
    }
    const other = totalInBed - listed;
    if (other > 0.5) activityPct.other = (other / totalInBed) * 100;
  }

  // Motion %s over the WHOLE day.
  const motionPct = {};
  let motionTotal = 0;
  for (const sec of Object.values(byMotion)) motionTotal += sec;
  if (motionTotal > 0) {
    for (const b of MOTION_BUCKETS) {
      const sec = byMotion[b] || 0;
      motionPct[b] = (sec / motionTotal) * 100;
    }
    const known = MOTION_BUCKETS.reduce((acc, b) => acc + (byMotion[b] || 0), 0);
    const unknown = motionTotal - known;
    if (unknown > 0.5) motionPct.unknown = (unknown / motionTotal) * 100;
  }

  return { totalInBed, totalOutBed, activityPct, motionPct, byActivity, byMotion };
}

function renderDayRows(rows) {
  if (rows.length === 0) {
    dayRowsDiv.innerHTML = '<div class="sum-empty">No history.</div>';
    return;
  }
  dayRowsDiv.innerHTML = rows.map((r) => renderOneDayRow(r)).join("");
  dayRowsDiv.querySelectorAll(".sum-day-row").forEach((el) => {
    el.addEventListener("click", () => showDetail(el.dataset.date));
  });
}

function renderOneDayRow({ date, isToday, stats }) {
  const dateObj = new Date(date + "T12:00:00");
  const weekday = dateObj.toLocaleDateString(undefined, { weekday: "short" });
  const human = dateObj.toLocaleDateString(undefined, { month: "short", day: "numeric" });

  if (!stats || (!hasData(stats) && isToday)) {
    return `
      <div class="sum-day-row sum-day-empty" data-date="${date}">
        <div class="sum-day-label">
          <span class="sum-day-name">${isToday ? "Today" : weekday}</span>
          <span class="sum-day-date">${human}</span>
        </div>
        <div class="sum-day-empty-msg">No recorded data yet for this day.</div>
      </div>`;
  }

  const totalDay = stats.totalInBed + stats.totalOutBed;
  const inBedPct = totalDay > 0
    ? Math.round((stats.totalInBed / totalDay) * 100)
    : 0;

  return `
    <div class="sum-day-row" data-date="${date}">
      <div class="sum-day-label">
        <span class="sum-day-name">${isToday ? "Today" : weekday}</span>
        <span class="sum-day-date">${human}</span>
      </div>

      <div class="sum-day-totals">
        <div class="sum-day-totline">
          <span class="sum-day-totkey">In bed</span>
          <span class="sum-day-totval">${fmtDur(stats.totalInBed)}</span>
        </div>
        <div class="sum-day-totline">
          <span class="sum-day-totkey">Out</span>
          <span class="sum-day-totval">${fmtDur(stats.totalOutBed)}</span>
        </div>
        <div class="sum-day-totline">
          <span class="sum-day-totkey">In-bed %</span>
          <span class="sum-day-totval">${inBedPct}%</span>
        </div>
      </div>

      <div class="sum-day-charts">
        <div class="sum-chart-block">
          <div class="sum-chart-label">Activity (of in-bed)</div>
          ${renderStackedBar(
            ACTIVITY_BUCKETS.concat(["other"]),
            stats.activityPct,
            ACTIVITY_LABELS,
            ACTIVITY_VCLASS,
          )}
        </div>
        <div class="sum-chart-block">
          <div class="sum-chart-label">Motion (of day)</div>
          ${renderStackedBar(
            MOTION_BUCKETS.concat(["unknown"]),
            stats.motionPct,
            MOTION_LABELS,
            MOTION_VCLASS,
          )}
        </div>
      </div>

      <div class="sum-day-chev" aria-hidden="true">›</div>
    </div>`;
}

// Build a horizontal stacked bar from a value→% map. Order is the
// bucket list passed in; absent / zero buckets render as zero-width.
function renderStackedBar(buckets, pct, labels, vclassMap) {
  const segs = buckets.map((b) => {
    const p = pct[b] || 0;
    if (p < 0.5) return "";
    const tip = `${labels[b] || b}: ${p.toFixed(1)}%`;
    return `<span class="sum-bar-seg ${vclassMap[b] || ""}" style="width:${p}%" title="${tip}"></span>`;
  }).join("");
  // Legend underneath with the same colour swatches — so users can read
  // the bar at a glance without hovering.
  const legend = buckets.map((b) => {
    const p = pct[b] || 0;
    if (p < 0.5) return "";
    return `
      <span class="sum-bar-legend-item">
        <span class="sum-bar-swatch ${vclassMap[b] || ""}"></span>
        <span class="sum-bar-legend-label">${labels[b] || b}</span>
        <span class="sum-bar-legend-pct">${Math.round(p)}%</span>
      </span>`;
  }).join("");
  if (!segs) {
    return '<div class="sum-bar-empty">No data</div>';
  }
  return `
    <div class="sum-bar-stack">${segs}</div>
    <div class="sum-bar-legend">${legend}</div>`;
}

// ---- main reload (detail mode) --------------------------------------

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
//
// We key on PERSON_COUNT, not activity. Activity is a downstream tag
// from BabyTracker which can lag behind raw detection — during the
// ACQUIRE / cooldown windows you see persons=1 but activity=out_of_
// frame on the same frame. That contradiction was previously bucketing
// "system saw a baby + I confirmed it" cells into the Out-of-bed
// period because activity was leading. person_count is the raw YES/NO
// of "did YOLO see something in this frame" — which is what you label
// correct/incorrect against.
//
// Logic:
//   persons > 0  → system says baby in frame
//   label "correct"   → trust system
//   label "incorrect" → flip
//   no label          → trust system
//
// Activity is still used elsewhere (border colour, activity tag on
// the thumbnail, the activity-mix bar on the overview row) — just not
// for the bed bucket itself.
function _bedFromSnapshot(s) {
  const persons = (s.state && s.state.person_count) || 0;
  let systemSaysIn = persons > 0;
  if (s.label === "incorrect") systemSaysIn = !systemSaysIn;
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
    // Border colour follows the system's activity classification —
    // identical to History (red = out_of_frame / no baby, green =
    // sitting, purple = asleep, etc). The operator's label moves to
    // the corner badge ONLY so the two pieces of information never
    // conflict visually: border = what the system saw, badge = what
    // you said about it.
    const act = (s.state && s.state.activity) || "";
    const actCls = act ? ` snap-act-${act}` : "";
    const labelCls = s.label ? ` sum-thumb-labeled-${s.label}` : "";
    const labelDot = s.label === "correct"
      ? '<span class="sum-thumb-label sum-thumb-label-correct" title="You marked correct">✓</span>'
      : s.label === "incorrect"
        ? '<span class="sum-thumb-label sum-thumb-label-incorrect" title="You marked incorrect (flips the bed bucket)">✗</span>'
        : "";
    const hasDrawn = Array.isArray(s.correction_boxes) && s.correction_boxes.length > 0;
    const drawnDot = hasDrawn
      ? '<span class="sum-thumb-drawn" title="Operator-drawn box on this snapshot (training signal)">▢</span>'
      : "";
    const tipDrawn = hasDrawn ? " · operator-drawn box" : "";
    return `
      <div class="sum-thumb${actCls}${labelCls}${hasDrawn ? " sum-thumb-has-drawn" : ""}" data-snap-id="${s.id}" role="button" tabindex="0"
         title="${fmtClock(s.captured_at)}${act ? " · " + act : ""}${s.label ? " · labeled " + s.label : ""}${tipDrawn}">
        <img loading="lazy" src="${url}" alt="${fmtClock(s.captured_at)}">
        ${labelDot}
        ${drawnDot}
        <span class="sum-thumb-time">${fmtClock(s.captured_at)}</span>
      </div>`;
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

function escapeHtmlSum(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// ====================================================================
// Lightbox: same UX as History — click a thumbnail to inspect the
// snapshot and mark it Correct/Incorrect/Clear. Uses the shared
// _lightbox.html partial so the markup and styling are identical.
// ====================================================================

const lightbox = document.getElementById("lightbox");
let _openSnap = null;

function setLightboxTab(name) {
  document.querySelectorAll("#lightbox .lb-tab").forEach((el) =>
    el.classList.toggle("active", el.dataset.tab === name),
  );
  document.querySelectorAll("#lightbox .lb-pane").forEach((el) =>
    el.classList.toggle("active", el.dataset.pane === name),
  );
  if (name !== "video") {
    const clipEl = document.getElementById("lb-clip");
    try { if (clipEl) clipEl.pause(); } catch (e) {}
  }
}

function renderParamsSum(state) {
  if (!state || typeof state !== "object") {
    return '<div class="muted">No parameters captured for this snapshot.</div>';
  }
  const rows = [];
  const push = (label, value) => {
    if (value === null || value === undefined || value === "") return;
    rows.push(`
      <div class="param-row">
        <span class="param-k">${label}</span>
        <span class="param-v">${escapeHtmlSum(String(value))}</span>
      </div>`);
  };
  push("Activity",  state.activity);
  if (state.crib_motion && state.crib_motion !== "undetected") {
    push("Crib motion", state.crib_motion === "still" ? "Still" : "Moving");
  }
  push("Posture",   state.posture);
  push("Motion",    state.motion);
  push("Persons",   state.person_count);
  push("Still for", state.still_seconds != null ? fmtDur(state.still_seconds) : null);
  push("Motion score", state.motion_score != null ? Number(state.motion_score).toFixed(3) : null);
  push("Body angle",  state.posture_angle_deg != null ? Math.round(state.posture_angle_deg) + "°" : null);
  push("FPS",         state.fps != null ? Number(state.fps).toFixed(1) : null);
  return rows.length ? rows.join("") : '<div class="muted">No parameters captured for this snapshot.</div>';
}

function openLightbox(snap) {
  const annotated = `/api/snapshots/${snap.file_rel}`;
  const raw = annotated.replace(/\.jpg$/, "_raw.jpg");
  document.getElementById("lb-caption").textContent = fmtClock(snap.captured_at);
  const rawCol = document.getElementById("lb-img-raw").parentElement;
  rawCol.classList.remove("no-raw");
  document.getElementById("lb-img-raw").style.display = "";
  document.getElementById("lb-img-annotated").src = annotated;
  document.getElementById("lb-img-raw").src = raw;

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
    tabVideo.disabled = false;
    tabVideo.classList.add("disabled");
  }

  document.getElementById("lb-params").innerHTML = renderParamsSum(snap.state);
  _openSnap = snap;
  syncLabelRowSum(snap.label || null);
  syncCorrectionVisibilitySum();
  setLightboxTab("images");
  lightbox.classList.remove("hidden");
}

// Cross-tab refresh: the standalone correction page writes to this
// localStorage key on save, picked up here so we update the affected
// thumb's badge live.
window.addEventListener("storage", (e) => {
  if (e.key !== "primeanalyze.snapshotCorrectionSaved" || !e.newValue) return;
  let payload;
  try { payload = JSON.parse(e.newValue); } catch { return; }
  if (!payload || payload.snap_id == null) return;
  const id = Number(payload.snap_id);
  const inStore = _snapshots.find((s) => s.id === id);
  if (inStore) inStore.correction_boxes = payload.box_count ? [[0, 0, 0, 0]] : [];
  const nodes = document.querySelectorAll(`.sum-thumb[data-snap-id="${id}"]`);
  nodes.forEach((n) => {
    const has = payload.box_count > 0;
    n.classList.toggle("sum-thumb-has-drawn", has);
    let dot = n.querySelector(".sum-thumb-drawn");
    if (has && !dot) {
      dot = document.createElement("span");
      dot.className = "sum-thumb-drawn";
      dot.title = "Operator-drawn box on this snapshot (training signal)";
      dot.textContent = "▢";
      n.insertBefore(dot, n.querySelector(".sum-thumb-time"));
    } else if (!has && dot) {
      dot.remove();
    }
  });
  if (_openSnap && _openSnap.id === id) {
    _openSnap.correction_boxes = payload.box_count ? [[0, 0, 0, 0]] : [];
    syncCorrectionVisibilitySum();
  }
});

function syncCorrectionVisibilitySum() {
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

document.querySelectorAll("#lightbox .lb-tab").forEach((el) =>
  el.addEventListener("click", () => setLightboxTab(el.dataset.tab)),
);
lightbox.querySelectorAll("[data-close]").forEach((el) =>
  el.addEventListener("click", () => {
    lightbox.classList.add("hidden");
    try { document.getElementById("lb-clip").pause(); } catch (e) {}
    _openSnap = null;
  }),
);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !lightbox.classList.contains("hidden")) {
    lightbox.classList.add("hidden");
    try { document.getElementById("lb-clip").pause(); } catch (e) {}
    _openSnap = null;
  }
});

function syncLabelRowSum(label) {
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
    status.textContent = "Marked incorrect (flips the bed bucket)";
    status.className = "lb-label-status status-incorrect";
  } else {
    status.textContent = "Unlabeled — Summary uses the system's own classification";
    status.className = "lb-label-status status-none";
  }
}

async function labelSnapshotSum(value) {
  if (!_openSnap || _openSnap.id == null) return;
  const newLabel = value || null;
  const prev = _openSnap.label || null;
  if (newLabel === prev) return;
  // Optimistic update — keep _snapshots in sync so re-renders match.
  _openSnap.label = newLabel;
  syncLabelRowSum(newLabel);
  syncCorrectionVisibilitySum();
  const inStore = _snapshots.find((s) => s.id === _openSnap.id);
  if (inStore) inStore.label = newLabel;
  refreshSumThumb(_openSnap);
  try {
    const r = await fetch(`/api/snapshots/${_openSnap.id}/label`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: newLabel }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
  } catch (e) {
    _openSnap.label = prev;
    if (inStore) inStore.label = prev;
    syncLabelRowSum(prev);
    refreshSumThumb(_openSnap);
    const status = document.getElementById("lb-label-status");
    status.textContent = "Save failed: " + e.message;
    status.className = "lb-label-status status-error";
  }
}

function refreshSumThumb(snap) {
  // Every thumbnail node for this snap (the same snap can appear inside
  // an expanded period and again in another period if the user opened
  // multiple). Cheap to update both — there are very few of them.
  const nodes = document.querySelectorAll(`.sum-thumb[data-snap-id="${snap.id}"]`);
  nodes.forEach((n) => {
    n.classList.remove("sum-thumb-labeled-correct", "sum-thumb-labeled-incorrect");
    if (snap.label) n.classList.add(`sum-thumb-labeled-${snap.label}`);
    let dot = n.querySelector(".sum-thumb-label");
    if (snap.label) {
      if (!dot) {
        dot = document.createElement("span");
        // Insert before the time span so the dot sits in its usual corner.
        n.insertBefore(dot, n.querySelector(".sum-thumb-time"));
      }
      dot.className = `sum-thumb-label sum-thumb-label-${snap.label}`;
      dot.textContent = snap.label === "correct" ? "✓" : "✗";
      dot.title = snap.label === "correct"
        ? "You marked correct"
        : "You marked incorrect (flips the bed bucket)";
    } else if (dot) {
      dot.remove();
    }
  });
}

document.querySelectorAll("#lb-label-row .lb-label-btn").forEach((b) =>
  b.addEventListener("click", () => labelSnapshotSum(b.dataset.label)),
);

// Event delegation on the document — thumbnails are rendered into
// expandable period rows dynamically, so we can't bind on mount.
document.addEventListener("click", (e) => {
  const thumb = e.target.closest(".sum-thumb[data-snap-id]");
  if (!thumb) return;
  // Don't fire when the click came from a control inside the thumb.
  if (e.target.closest("a, button")) return;
  e.preventDefault();
  const id = Number(thumb.dataset.snapId);
  const snap = _snapshots.find((s) => s.id === id);
  if (snap) openLightbox(snap);
});
// Keyboard activation for the role="button" thumbs.
document.addEventListener("keydown", (e) => {
  if (e.key !== "Enter" && e.key !== " ") return;
  const thumb = e.target.closest && e.target.closest(".sum-thumb[data-snap-id]");
  if (!thumb || e.target !== thumb) return;
  e.preventDefault();
  const id = Number(thumb.dataset.snapId);
  const snap = _snapshots.find((s) => s.id === id);
  if (snap) openLightbox(snap);
});
