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
  const activity = (data?.tracks?.activity) || [];
  _snapshots = (data?.snapshots) || [];

  // Reduce the multi-state activity timeline to 2-state.
  const bedSegments = compressByValue(
    activity.map((s) => ({
      value: _asBed(s.value),
      start_ts: s.start_ts,
      end_ts: s.end_ts,
    })),
  );

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

  summary.textContent =
    `${bedSegments.length} segment${bedSegments.length === 1 ? "" : "s"} for ${data.date || currentDate}`;
  elTotalIn.textContent = fmtDur(totalIn);
  elTotalOut.textContent = fmtDur(totalOut);
  elOutCount.textContent = outCount;
  elLongestIn.textContent = longestIn > 0 ? fmtDur(longestIn) : "—";

  renderBar(bedSegments, dayStart, dayEnd);
  renderHourTicks(dayStart, dayEnd);
  renderPeriods(bedSegments);
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
        body.innerHTML = renderPeriodThumbs(seg.start_ts, seg.end_ts);
        body.dataset.rendered = "1";
      }
      body.hidden = false;
      header.setAttribute("aria-expanded", "true");
      chevron.textContent = "▾";
    });
  });
}

function renderPeriodThumbs(start, end) {
  const inWin = _snapshots.filter(
    (s) => s.captured_at >= start - 0.5 && s.captured_at <= end + 0.5
  );
  if (inWin.length === 0) {
    return '<div class="sum-empty">No snapshots captured during this window.</div>';
  }
  // Hard cap so a very long in-bed stretch doesn't render thousands of
  // <img> tags. Show first + last + evenly-spaced samples up to MAX.
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
    return `
      <a class="sum-thumb" href="${url}" target="_blank" rel="noopener"
         title="${fmtClock(s.captured_at)}">
        <img loading="lazy" src="${url}" alt="${fmtClock(s.captured_at)}">
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
