// Event-camera detail page. Polls the worker's /state every 2s and the
// recent-captures list every 10s. No live MJPEG view — recent
// snapshots are the closest substitute and update continuously.

const CAM_ID = window.__CAM_ID;
const elStatus       = document.getElementById("ev-status");
const elStatusSub    = document.getElementById("ev-status-sub");
const elSnapsToday   = document.getElementById("ev-snaps-today");
const elLastCapture  = document.getElementById("ev-last-capture");
const elLastSub      = document.getElementById("ev-last-capture-sub");
const elPeople       = document.getElementById("ev-people");
const elStrip        = document.getElementById("ev-strip");
const modal          = document.getElementById("ev-modal");
const modalImg       = document.getElementById("ev-modal-img");
const modalMeta      = document.getElementById("ev-modal-meta");

function fmtClock(ts) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleTimeString();
}
function fmtRel(ts) {
  if (!ts) return "no captures yet";
  const dt = Date.now() / 1000 - ts;
  if (dt < 60) return `${Math.round(dt)} s ago`;
  if (dt < 3600) return `${Math.round(dt / 60)} m ago`;
  return `${Math.round(dt / 3600)} h ago`;
}

async function loadStatus() {
  try {
    const r = await fetch(`/api/cameras/${CAM_ID}/state`, { cache: "no-store" });
    if (!r.ok) {
      elStatus.textContent = "offline";
      elStatusSub.textContent = "worker unreachable";
      return;
    }
    const s = await r.json();
    if (!s.alive) {
      elStatus.textContent = "offline";
      elStatusSub.textContent = "worker not running";
      return;
    }
    const act = s.activity || "idle";
    elStatus.innerHTML = act === "person_detected"
      ? '<span class="ev-pill ev-pill-hot">person detected</span>'
      : '<span class="ev-pill ev-pill-idle">idle</span>';
    elStatusSub.textContent = `worker alive · ${(s.fps || 0).toFixed(1)} fps target`;
    elSnapsToday.textContent = s.snapshots_today ?? 0;
    elLastCapture.textContent = fmtClock(s.last_capture_ts);
    elLastSub.textContent = s.last_capture_ts
      ? fmtRel(s.last_capture_ts) + " · debounce 30 s"
      : "no captures yet";
    elPeople.textContent = s.last_person_count ?? 0;
  } catch (e) {
    elStatus.textContent = "error";
    elStatusSub.textContent = e.message;
  }
}

async function loadRecent() {
  try {
    // Reuse the event-snapshots endpoint with an unimported_cid of '_'
    // (which doesn't match any class id) so nothing is excluded —
    // returns the camera's full recent capture list.
    const r = await fetch(
      `/api/event-snapshots?camera_id=${encodeURIComponent(CAM_ID)}`
      + `&cid=__none__&hours=72&limit=24`,
      { cache: "no-store" });
    if (!r.ok) return;
    const data = await r.json();
    const items = data.available || [];
    if (!items.length) {
      elStrip.innerHTML = '<div class="muted">No captures yet — walk in front of the camera to test.</div>';
      return;
    }
    elStrip.innerHTML = items.map((s) => `
      <a class="ev-strip-thumb" data-cap="${s.captured_at}" data-pc="${s.person_count}"
         data-src="/api/snapshots/${s.file_rel}" href="#">
        <img src="/api/snapshots/${s.file_rel}" loading="lazy" alt="">
        <span class="ev-strip-time">${new Date(s.captured_at * 1000).toLocaleTimeString()}</span>
        ${s.person_count ? `<span class="ev-strip-badge">${s.person_count}</span>` : ""}
      </a>
    `).join("");
    elStrip.querySelectorAll(".ev-strip-thumb").forEach((el) =>
      el.addEventListener("click", (e) => {
        e.preventDefault();
        const src = el.dataset.src;
        modalImg.src = src;
        modalMeta.textContent = `${new Date((+el.dataset.cap) * 1000).toLocaleString()} · `
                                + `${el.dataset.pc} person${el.dataset.pc === "1" ? "" : "s"}`;
        modal.hidden = false;
      }),
    );
  } catch (e) {}
}

modal.querySelectorAll("[data-close]").forEach((el) =>
  el.addEventListener("click", () => { modal.hidden = true; modalImg.src = ""; }),
);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !modal.hidden) {
    modal.hidden = true;
    modalImg.src = "";
  }
});

loadStatus();
loadRecent();
const _s = setInterval(loadStatus, 2000);
const _r = setInterval(loadRecent, 10000);
window.addEventListener("pagehide", () => {
  clearInterval(_s); clearInterval(_r);
});
