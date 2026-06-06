// Scheduled training (auto-retrain agent) — list + details + manual run.

const list      = document.getElementById("st-iter-list");
const empty     = document.getElementById("st-iter-empty");
const elNow     = document.getElementById("st-status-now");
const elLast    = document.getElementById("st-status-last");
const elCount   = document.getElementById("st-status-count");
const btnRun    = document.getElementById("st-run-now");
const cbForce   = document.getElementById("st-force");
const cbNoProm  = document.getElementById("st-no-promote");
const modal     = document.getElementById("st-modal");
const modalContent = document.getElementById("st-modal-content");
const modalTitle = document.getElementById("st-modal-title");

let _pollHandle = null;

function fmtTs(t) {
  if (!t) return "—";
  const d = new Date(t * 1000);
  return d.toLocaleString();
}
function fmtDur(secs) {
  if (secs == null) return "—";
  secs = Math.round(secs);
  if (secs < 60) return `${secs}s`;
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return `${m}m ${s}s`;
}
function fmtPct(v) {
  if (v == null) return "—";
  return `${(v * 100).toFixed(1)}%`;
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function renderStatusBadge(it) {
  const s = it.status || "unknown";
  const cls = s === "completed" ? "ok" : s === "running" ? "running" : s === "failed" ? "fail" : "neutral";
  return `<span class="st-pill st-pill-${cls}">${s}</span>`;
}

function renderPromoteBadge(it) {
  if (it.status !== "completed") return "";
  return it.promoted
    ? '<span class="st-pill st-pill-promoted">promoted ↑</span>'
    : '<span class="st-pill st-pill-held">held</span>';
}

function renderIter(it) {
  const elapsed = (it.completed_at && it.started_at)
    ? fmtDur(it.completed_at - it.started_at) : "—";
  const baseF1 = it.baseline_metrics?.f1;
  const candF1 = it.candidate_metrics?.f1;
  const delta = it.delta_f1;
  const deltaCls = delta == null ? "" : delta > 0 ? " st-delta-up" : delta < 0 ? " st-delta-down" : "";
  return `
    <div class="st-iter" data-id="${it.id}">
      <div class="st-iter-top">
        <div class="st-iter-id">#${it.id}</div>
        <div class="st-iter-status">${renderStatusBadge(it)}${renderPromoteBadge(it)}</div>
        <div class="st-iter-time">${fmtTs(it.started_at)}</div>
      </div>
      <div class="st-iter-stats">
        <div><span class="k">New labels</span><span class="v">${it.new_labels_since_last ?? "—"}</span></div>
        <div><span class="k">Train epochs</span><span class="v">${it.epochs ?? "—"}</span></div>
        <div><span class="k">Wall-clock</span><span class="v">${elapsed}</span></div>
        <div><span class="k">Current F1</span><span class="v">${fmtPct(baseF1)}</span></div>
        <div><span class="k">Candidate F1</span><span class="v">${fmtPct(candF1)}</span></div>
        <div class="${deltaCls}"><span class="k">ΔF1</span><span class="v">${delta == null ? "—" : (delta * 100).toFixed(2) + " pp"}</span></div>
      </div>
      ${it.error ? `<div class="st-iter-err">${escapeHtml(it.error)}</div>` : ""}
    </div>`;
}

async function loadIterations() {
  let data;
  try {
    const r = await fetch("/api/auto-retrain/iterations", { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    data = await r.json();
  } catch (e) {
    list.innerHTML = `<p class="muted">Failed to load: ${escapeHtml(e.message)}</p>`;
    return;
  }
  const its = data.iterations || [];
  empty.hidden = its.length > 0;
  list.innerHTML = its.map(renderIter).join("");
  list.querySelectorAll(".st-iter").forEach((el) =>
    el.addEventListener("click", () => openDetail(Number(el.dataset.id))));
}

async function loadStatus() {
  let s;
  try {
    const r = await fetch("/api/auto-retrain/status", { cache: "no-store" });
    if (!r.ok) return;
    s = await r.json();
  } catch (e) { return; }

  elCount.textContent = s.iteration_count ?? 0;
  if (s.running) {
    elNow.innerHTML = `<span class="st-pill st-pill-running">running</span> PID ${s.pid}`;
    btnRun.disabled = true;
    btnRun.textContent = "Running…";
  } else {
    elNow.innerHTML = `<span class="st-pill st-pill-neutral">idle</span>`;
    btnRun.disabled = false;
    btnRun.textContent = "Run now";
  }
  if (s.last_iteration) {
    const it = s.last_iteration;
    elLast.innerHTML = `#${it.id} ${renderStatusBadge(it)} ${renderPromoteBadge(it)} · ${fmtTs(it.started_at)}`;
  } else {
    elLast.textContent = "—";
  }
}

async function openDetail(id) {
  modal.hidden = false;
  modalTitle.textContent = `Iteration #${id}`;
  modalContent.textContent = "Loading…";
  let it;
  try {
    const r = await fetch(`/api/auto-retrain/iterations/${id}`, { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    it = await r.json();
  } catch (e) {
    modalContent.innerHTML = `<p class="muted">Failed: ${escapeHtml(e.message)}</p>`;
    return;
  }
  const cm = it.candidate_metrics || {};
  const bm = it.baseline_metrics || {};
  modalContent.innerHTML = `
    <table class="st-metrics">
      <tr><th></th><th>Current</th><th>Candidate</th></tr>
      <tr><td>F1</td><td>${fmtPct(bm.f1)}</td><td>${fmtPct(cm.f1)}</td></tr>
      <tr><td>Precision</td><td>${fmtPct(bm.precision)}</td><td>${fmtPct(cm.precision)}</td></tr>
      <tr><td>Recall</td><td>${fmtPct(bm.recall)}</td><td>${fmtPct(cm.recall)}</td></tr>
      <tr><td>Accuracy</td><td>${fmtPct(bm.accuracy)}</td><td>${fmtPct(cm.accuracy)}</td></tr>
      <tr><td>TP / FP / FN / TN</td>
          <td>${bm.tp ?? "—"} / ${bm.fp ?? "—"} / ${bm.fn ?? "—"} / ${bm.tn ?? "—"}</td>
          <td>${cm.tp ?? "—"} / ${cm.fp ?? "—"} / ${cm.fn ?? "—"} / ${cm.tn ?? "—"}</td></tr>
      <tr><td>Holdout size</td><td colspan="2">${cm.n ?? "—"}</td></tr>
    </table>
    <div class="st-meta">
      <div><b>Started:</b> ${fmtTs(it.started_at)}</div>
      <div><b>Completed:</b> ${fmtTs(it.completed_at)}</div>
      <div><b>Status:</b> ${escapeHtml(it.status || "—")} ${it.promoted ? "(promoted)" : ""}</div>
      <div><b>Candidate weights:</b> <code>${escapeHtml(it.candidate_path || "—")}</code></div>
      ${it.error ? `<div class="st-iter-err">${escapeHtml(it.error)}</div>` : ""}
    </div>
    <details class="st-log-wrap" open>
      <summary>Log tail</summary>
      <pre class="st-log">${escapeHtml((it.log_tail || []).join("\n")) || "(no log)"}</pre>
    </details>
  `;
}

modal.querySelectorAll("[data-close]").forEach((el) =>
  el.addEventListener("click", () => { modal.hidden = true; }));
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !modal.hidden) modal.hidden = true;
});

btnRun.addEventListener("click", async () => {
  btnRun.disabled = true;
  btnRun.textContent = "Starting…";
  try {
    const r = await fetch("/api/auto-retrain/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        camera: "cam_1",
        force: cbForce.checked,
        no_promote: cbNoProm.checked,
      }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      alert("Failed to start: " + (err.error || r.status));
    }
  } finally {
    setTimeout(() => { loadStatus(); loadIterations(); }, 800);
  }
});

function pollOnce() {
  loadStatus();
  loadIterations();
}
pollOnce();
_pollHandle = setInterval(pollOnce, 5000);
window.addEventListener("pagehide", () => {
  if (_pollHandle) clearInterval(_pollHandle);
});
