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
const episodesDiv = document.getElementById("sum-episodes");
const episodesCount = document.getElementById("sum-episodes-count");

const elTotalIn = document.getElementById("sum-total-in");
const elTotalOut = document.getElementById("sum-total-out");
const elOutCount = document.getElementById("sum-out-count");
const elLongestIn = document.getElementById("sum-longest-in");

let currentCam = null;
let currentDate = new Date().toISOString().slice(0, 10);
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
  let totalIn = 0, totalOut = 0, longestIn = 0;
  const outEpisodes = [];
  for (const s of bedSegments) {
    const dur = (s.end_ts - s.start_ts);
    if (s.value === SUM_BED_IN) {
      totalIn += dur;
      if (dur > longestIn) longestIn = dur;
    } else {
      totalOut += dur;
      outEpisodes.push({ start: s.start_ts, end: s.end_ts, duration: dur });
    }
  }

  summary.textContent =
    `${bedSegments.length} segment${bedSegments.length === 1 ? "" : "s"} for ${data.date || currentDate}`;
  elTotalIn.textContent = fmtDur(totalIn);
  elTotalOut.textContent = fmtDur(totalOut);
  elOutCount.textContent = outEpisodes.length;
  elLongestIn.textContent = longestIn > 0 ? fmtDur(longestIn) : "—";

  renderBar(bedSegments, dayStart, dayEnd);
  renderHourTicks(dayStart, dayEnd);
  renderEpisodes(outEpisodes);
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

function renderEpisodes(eps) {
  episodesCount.textContent = eps.length ? `· ${eps.length}` : "";
  if (eps.length === 0) {
    episodesDiv.innerHTML =
      '<div class="sum-episodes-empty">No out-of-bed episodes recorded for this day.</div>';
    return;
  }
  episodesDiv.innerHTML = eps.map((e) => `
    <div class="sum-episode">
      <span class="sum-ep-time">${fmtClock(e.start)}</span>
      <span class="sum-ep-range">→ ${fmtClock(e.end)}</span>
      <span class="sum-ep-dur">${fmtDur(e.duration)}</span>
    </div>
  `).join("");
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
