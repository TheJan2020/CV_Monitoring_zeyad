// Training page — polls /api/training every 5 s and renders live progress.

const POLL_MS = 5000;

const els = {
  status: document.getElementById("trn-status"),
  runName: document.getElementById("trn-run-name"),
  epochsDone: document.getElementById("trn-epochs-done"),
  epochsTotal: document.getElementById("trn-epochs-total"),
  fill: document.getElementById("trn-progress-fill"),
  etaText: document.getElementById("trn-eta-text"),
  elapsed: document.getElementById("trn-elapsed"),
  eta: document.getElementById("trn-eta"),
  epochN: document.getElementById("trn-epoch-n"),
  precision: document.getElementById("trn-precision"),
  recall: document.getElementById("trn-recall"),
  map50: document.getElementById("trn-map50"),
  map95: document.getElementById("trn-map95"),
  boxLoss: document.getElementById("trn-box-loss"),
  clsLoss: document.getElementById("trn-cls-loss"),
  history: document.getElementById("trn-history-body"),
  log: document.getElementById("trn-log"),
};

function fmtDur(seconds) {
  if (seconds == null || isNaN(seconds)) return "—";
  seconds = Math.max(0, Math.round(seconds));
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function fmtPct(v, digits = 1) {
  if (v == null || isNaN(v)) return "—";
  return (v * 100).toFixed(digits) + "%";
}

function fmtNum(v, digits = 3) {
  if (v == null || isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}

// Ultralytics column names are inconsistent across versions (some use
// "metrics/precision(B)", some "metrics/precision"). Pick the first
// matching key from a list of candidates.
function pick(row, ...candidates) {
  if (!row) return null;
  for (const k of candidates) {
    if (k in row && row[k] != null) return row[k];
  }
  return null;
}

function metricsFromRow(row) {
  return {
    epoch: pick(row, "epoch"),
    precision: pick(row, "metrics/precision(B)", "metrics/precision"),
    recall: pick(row, "metrics/recall(B)", "metrics/recall"),
    map50: pick(row, "metrics/mAP50(B)", "metrics/mAP50"),
    map95: pick(row, "metrics/mAP50-95(B)", "metrics/mAP50-95"),
    boxLoss: pick(row, "train/box_loss", "val/box_loss"),
    clsLoss: pick(row, "train/cls_loss", "val/cls_loss"),
  };
}

function renderStatus(info) {
  let label, cls;
  if (info.active) {
    label = "● Training";
    cls = "trn-status-running";
  } else if (info.epochs_total && info.epochs_done >= info.epochs_total) {
    label = "✓ Completed";
    cls = "trn-status-done";
  } else if (info.epochs_done > 0) {
    label = "■ Paused";
    cls = "trn-status-paused";
  } else {
    label = "○ Idle";
    cls = "trn-status-idle";
  }
  els.status.textContent = label;
  els.status.className = "trn-status-badge " + cls;
  els.runName.textContent = info.run_name || "no run yet";
}

function renderProgress(info) {
  const done = info.epochs_done || 0;
  const total = info.epochs_total || 0;
  els.epochsDone.textContent = done;
  els.epochsTotal.textContent = total > 0 ? total : "?";
  const pct = total > 0 ? Math.min(100, (done / total) * 100) : 0;
  els.fill.style.width = pct.toFixed(1) + "%";
  els.elapsed.textContent = fmtDur(info.elapsed_seconds);
  els.eta.textContent = fmtDur(info.eta_seconds);
  if (info.eta_seconds != null && info.active) {
    els.etaText.textContent = `~${fmtDur(info.eta_seconds)} remaining`;
  } else if (info.epochs_total && done >= info.epochs_total) {
    els.etaText.textContent = "completed";
  } else {
    els.etaText.textContent = "";
  }
}

function renderMetrics(info) {
  const last = info.metrics ? metricsFromRow(info.metrics) : null;
  if (!last) {
    els.epochN.textContent = "";
    els.precision.textContent = "—";
    els.recall.textContent = "—";
    els.map50.textContent = "—";
    els.map95.textContent = "—";
    els.boxLoss.textContent = "—";
    els.clsLoss.textContent = "—";
    return;
  }
  els.epochN.textContent = last.epoch != null ? `· epoch ${Math.round(last.epoch)}` : "";
  els.precision.textContent = fmtPct(last.precision);
  els.recall.textContent = fmtPct(last.recall);
  els.map50.textContent = fmtPct(last.map50);
  els.map95.textContent = fmtPct(last.map95);
  els.boxLoss.textContent = fmtNum(last.boxLoss, 3);
  els.clsLoss.textContent = fmtNum(last.clsLoss, 2);
}

function renderHistory(info) {
  const rows = info.history || [];
  if (rows.length === 0) {
    els.history.innerHTML =
      '<tr><td colspan="7" class="trn-empty">No epochs completed yet.</td></tr>';
    return;
  }
  // Newest first
  const newestFirst = [...rows].reverse();
  els.history.innerHTML = newestFirst.map((r) => {
    const m = metricsFromRow(r);
    return `<tr>
      <td>${m.epoch != null ? Math.round(m.epoch) : "—"}</td>
      <td>${fmtPct(m.precision)}</td>
      <td>${fmtPct(m.recall)}</td>
      <td>${fmtPct(m.map50)}</td>
      <td>${fmtPct(m.map95)}</td>
      <td>${fmtNum(m.boxLoss, 3)}</td>
      <td>${fmtNum(m.clsLoss, 2)}</td>
    </tr>`;
  }).join("");
}

function renderLog(info) {
  const lines = info.recent_log || [];
  if (lines.length === 0) {
    els.log.textContent = "(no log yet — training may be still warming up)";
    return;
  }
  els.log.textContent = lines.join("\n");
  els.log.scrollTop = els.log.scrollHeight;
}

async function poll() {
  try {
    const r = await fetch("/api/training", { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const info = await r.json();
    renderStatus(info);
    renderProgress(info);
    renderMetrics(info);
    renderHistory(info);
    renderLog(info);
  } catch (e) {
    els.status.textContent = "✗ Failed to fetch";
    els.status.className = "trn-status-badge trn-status-error";
    els.runName.textContent = e.message;
  }
}

poll();
setInterval(poll, POLL_MS);
