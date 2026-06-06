// Dashboard — system-wide stats, polled every 15s.

function n(x) { return (typeof x === "number" ? x.toLocaleString() : "—"); }
function fmtPct(num, den) {
  if (!den) return "—";
  return `${(100 * num / den).toFixed(1)}%`;
}
function fmtTs(t) {
  if (!t) return "—";
  return new Date(t * 1000).toLocaleString();
}
function fmtPp(v) {
  if (v == null) return "—";
  const sign = v > 0 ? "+" : "";
  return `${sign}${(v * 100).toFixed(2)} pp`;
}

async function loadCameras() {
  try {
    const r = await fetch("/api/cameras", { cache: "no-store" });
    if (!r.ok) return;
    const d = await r.json();
    if (Array.isArray(d)) document.getElementById("dash-cam-count").textContent = d.length;
  } catch (e) {}
}

async function loadSnapStats() {
  try {
    const r = await fetch("/api/snapshots/stats", { cache: "no-store" });
    if (!r.ok) return;
    const s = await r.json();
    const total = s.total || 0;
    const scored = s.scored || 0;
    const correct = s.correct || 0;
    const incorrect = s.incorrect || 0;
    const unscored = s.unscored || 0;
    const corr = s.corrections_drawn || 0;

    document.getElementById("dash-snap-total").textContent = n(total);
    document.getElementById("dash-snap-scored").textContent = n(scored);
    document.getElementById("dash-snap-scored-pct").textContent =
      total ? `${fmtPct(scored, total)} of total` : "no snapshots yet";
    document.getElementById("dash-snap-unscored").textContent = n(unscored);
    document.getElementById("dash-snap-correct").textContent = n(correct);
    document.getElementById("dash-snap-correct-pct").textContent =
      scored ? `${fmtPct(correct, scored)} of labeled` : "—";
    document.getElementById("dash-snap-incorrect").textContent = n(incorrect);
    document.getElementById("dash-snap-incorrect-pct").textContent =
      scored ? `${fmtPct(incorrect, scored)} of labeled` : "—";
    document.getElementById("dash-corrections").textContent = n(corr);

    const acc = scored ? correct / scored : null;
    document.getElementById("dash-accuracy").textContent =
      acc == null ? "—" : `${(acc * 100).toFixed(1)}%`;
    document.getElementById("dash-accuracy-sub").textContent =
      scored ? `${n(correct)} correct ÷ ${n(scored)} labeled` : "label snapshots to compute";

    // Progress bar
    if (total > 0) {
      document.getElementById("dash-bar-correct").style.width = `${(correct / total) * 100}%`;
      document.getElementById("dash-bar-incorrect").style.width = `${(incorrect / total) * 100}%`;
      document.getElementById("dash-bar-unscored").style.width = `${(unscored / total) * 100}%`;
    }
  } catch (e) {}
}

async function loadAutoRetrain() {
  try {
    const r = await fetch("/api/auto-retrain/status", { cache: "no-store" });
    if (!r.ok) return;
    const s = await r.json();
    const stateEl = document.getElementById("dash-auto-state");
    if (s.running) {
      stateEl.innerHTML = `<span class="dash-pill running">running</span>`;
      document.getElementById("dash-auto-next").textContent = `PID ${s.pid} — view live progress`;
    } else {
      stateEl.innerHTML = `<span class="dash-pill idle">idle</span>`;
      document.getElementById("dash-auto-next").textContent = "next nightly run at 03:00 Riyadh";
    }

    const it = s.last_iteration;
    if (it) {
      const status = it.status || "—";
      const promotedTag = it.promoted ? ' <span class="dash-pill ok">promoted</span>' : '';
      document.getElementById("dash-auto-last").innerHTML =
        `#${it.id} <span class="dash-pill ${status === "completed" ? "ok" : status === "failed" ? "fail" : "neutral"}">${status}</span>${promotedTag}`;
      document.getElementById("dash-auto-last-sub").innerHTML =
        `${fmtTs(it.started_at)} · <a href="/scheduled-training">details ›</a>`;
      document.getElementById("dash-auto-delta").textContent = fmtPp(it.delta_f1);
      document.getElementById("dash-auto-delta-sub").textContent =
        it.delta_f1 == null ? "no F1 lift recorded" :
        it.delta_f1 > 0 ? "candidate beat the prior model" :
        it.delta_f1 < 0 ? "candidate underperformed — held" : "tie — held";
    } else {
      document.getElementById("dash-auto-last").textContent = "never";
      document.getElementById("dash-auto-last-sub").innerHTML =
        `<a href="/scheduled-training">Open scheduled training ›</a>`;
    }
  } catch (e) {}
}

async function loadCustoms() {
  try {
    const r = await fetch("/api/custom-classes", { cache: "no-store" });
    if (!r.ok) return;
    const classes = await r.json();
    const ready = classes.filter((c) => c.model_ready).length;
    const total = classes.length;
    document.getElementById("dash-customs").textContent =
      total ? `${ready} / ${total}` : "0";
  } catch (e) {}
}

function refresh() {
  loadCameras();
  loadSnapStats();
  loadAutoRetrain();
  loadCustoms();
}
refresh();
const _handle = setInterval(refresh, 15000);
window.addEventListener("pagehide", () => clearInterval(_handle));
