// Cameras page — list + add modal + connection testing + status polling.

const CATEGORY_LABEL = {
  baby:   "Baby Camera",
  worker: "Worker Camera",
};

const maskRtsp = (url) =>
  (url || "").replace(/:([^/@]+)(@)/, ":•••$2");

const grid = document.getElementById("cam-grid");
const summary = document.getElementById("cam-summary");
const empty = document.getElementById("cam-empty");
const modal = document.getElementById("add-modal");

document.getElementById("open-add").addEventListener("click", () => openModal());

modal.querySelectorAll("[data-close]").forEach((el) =>
  el.addEventListener("click", () => closeModal()),
);

function openModal() {
  document.getElementById("add-form").reset();
  document.getElementById("add-status").innerHTML = "";
  modal.classList.remove("hidden");
}

function closeModal() {
  modal.classList.add("hidden");
}

// ---- Render ----------------------------------------------------------

async function loadCameras() {
  let cams = [];
  try {
    const r = await fetch("/api/cameras", { cache: "no-store" });
    if (!r.ok) throw new Error("HTTP " + r.status);
    cams = await r.json();
  } catch (e) {
    summary.textContent = "Unable to load cameras: " + e.message;
    return;
  }

  summary.textContent = `${cams.length} camera${cams.length === 1 ? "" : "s"}.`;
  empty.classList.toggle("hidden", cams.length > 0);

  grid.innerHTML = cams.map(cardHtml).join("");
  cams.forEach(pollState);
  attachDeleteHandlers();
}

function cardHtml(c) {
  const catLabel = CATEGORY_LABEL[c.category] || "Worker Camera";
  return `
    <div class="cam-card" data-id="${c.id}">
      <div class="cam-card-head">
        <div>
          <div class="cam-name">${escapeHtml(c.name)}</div>
          <div class="cam-meta">
            <span class="badge badge-${c.category}">${catLabel}</span>
            <span class="muted">port ${c.port}</span>
          </div>
        </div>
        <div class="cam-status" id="status-${c.id}" title="checking…">
          <span class="status-dot status-unknown"></span>
          <span class="status-text">checking…</span>
        </div>
      </div>
      <div class="cam-url" title="${escapeAttr(c.rtsp_url)}">${escapeHtml(maskRtsp(c.rtsp_url))}</div>
      <div class="cam-actions">
        <a class="link" href="/cameras/${c.id}">Open live ›</a>
        <button class="btn btn-ghost btn-danger" data-delete="${c.id}" data-name="${escapeAttr(c.name)}">Remove</button>
      </div>
    </div>`;
}

function attachDeleteHandlers() {
  grid.querySelectorAll("[data-delete]").forEach((b) => {
    b.addEventListener("click", async () => {
      const id = b.dataset.delete;
      const name = b.dataset.name;
      if (!confirm(`Remove camera "${name}"?`)) return;
      b.disabled = true;
      try {
        const r = await fetch("/api/cameras/" + id, { method: "DELETE" });
        if (!r.ok) {
          const d = await r.json().catch(() => ({}));
          alert("Remove failed: " + (d.error || `HTTP ${r.status}`));
          b.disabled = false;
          return;
        }
        loadCameras();
      } catch (e) {
        alert("Remove failed: " + e.message);
        b.disabled = false;
      }
    });
  });
}

async function pollState(cam) {
  const el = document.getElementById("status-" + cam.id);
  if (!el) return;
  try {
    const r = await fetch("/api/cameras/" + cam.id + "/state", { cache: "no-store" });
    if (r.ok) {
      const s = await r.json();
      if (s && s.activity) {
        el.innerHTML = `<span class="status-dot status-ok"></span><span class="status-text">connected · ${s.activity.replace(/_/g, " ")}</span>`;
        el.title = `Active. FPS ${(s.fps || 0).toFixed(1)}.`;
        return;
      }
    }
    el.innerHTML = `<span class="status-dot status-loading"></span><span class="status-text">connecting…</span>`;
    el.title = "Worker starting up (models loading)";
  } catch (e) {
    el.innerHTML = `<span class="status-dot status-bad"></span><span class="status-text">unreachable</span>`;
    el.title = e.message;
  }
}

// ---- Test connection -------------------------------------------------

const testBtn = document.getElementById("test-btn");
testBtn.addEventListener("click", async () => {
  const fd = new FormData(document.getElementById("add-form"));
  const url = (fd.get("rtsp_url") || "").trim();
  const status = document.getElementById("add-status");
  if (!url) {
    status.className = "form-status err";
    status.textContent = "Enter an RTSP URL first.";
    return;
  }
  status.className = "form-status info";
  status.textContent = "Testing connection…";
  testBtn.disabled = true;
  try {
    const r = await fetch("/api/cameras/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rtsp_url: url }),
    });
    const d = await r.json();
    if (r.ok && d.ok) {
      const dims = (d.frame_shape || []).slice(0, 2).reverse().join(" × ");
      status.className = "form-status ok";
      status.innerHTML = `✓ Connected. ${dims ? "Frame: " + dims + " px." : ""}`;
    } else {
      status.className = "form-status err";
      status.textContent = "✗ " + (d.error || "Failed to connect.");
    }
  } catch (e) {
    status.className = "form-status err";
    status.textContent = "Test failed: " + e.message;
  } finally {
    testBtn.disabled = false;
  }
});

// ---- Submit ----------------------------------------------------------

document.getElementById("add-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const status = document.getElementById("add-status");
  const saveBtn = document.getElementById("save-btn");
  status.className = "form-status info";
  status.textContent = "Verifying connection…";
  saveBtn.disabled = true;

  const body = {
    name:     (fd.get("name") || "").trim(),
    category: fd.get("category"),
    rtsp_url: (fd.get("rtsp_url") || "").trim(),
  };
  if (!body.name || !body.rtsp_url) {
    status.className = "form-status err";
    status.textContent = "Name and RTSP URL are required.";
    saveBtn.disabled = false;
    return;
  }

  // Block-on-test: don't add if the URL doesn't accept a connection.
  try {
    const tr = await fetch("/api/cameras/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rtsp_url: body.rtsp_url }),
    });
    const td = await tr.json();
    if (!tr.ok || !td.ok) {
      status.className = "form-status err";
      status.textContent = "✗ " + (td.error || "Camera connection test failed. Check the RTSP URL.");
      saveBtn.disabled = false;
      return;
    }
  } catch (e) {
    status.className = "form-status err";
    status.textContent = "Test failed: " + e.message;
    saveBtn.disabled = false;
    return;
  }

  status.textContent = "Connection OK. Creating camera…";
  try {
    const r = await fetch("/api/cameras", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const d = await r.json();
    if (!r.ok) {
      status.className = "form-status err";
      status.textContent = "✗ " + (d.error || `HTTP ${r.status}`);
      saveBtn.disabled = false;
      return;
    }
    status.className = "form-status ok";
    status.textContent = "✓ Camera added. Worker is starting…";
    setTimeout(() => {
      closeModal();
      loadCameras();
    }, 600);
  } catch (e) {
    status.className = "form-status err";
    status.textContent = "Add failed: " + e.message;
  } finally {
    saveBtn.disabled = false;
  }
});

// ---- Helpers ---------------------------------------------------------

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]),
  );
}
function escapeAttr(s) { return escapeHtml(s); }

loadCameras();
setInterval(loadCameras, 8000);  // refresh status every 8 s
