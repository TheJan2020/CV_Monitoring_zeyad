// Classes list page — create / list / delete custom object classes.

const grid       = document.getElementById("classes-grid");
const empty      = document.getElementById("classes-empty");
const btnNew     = document.getElementById("cls-new");
const modal      = document.getElementById("cls-modal");
const modalErr   = document.getElementById("cls-modal-err");
const nameInput  = document.getElementById("cls-name-input");
const btnCreate  = document.getElementById("cls-create");
const btnCancel  = document.getElementById("cls-cancel");

const STATUS_LABELS = {
  collecting: "Collecting images",
  labeling:   "Labeling",
  training:   "Training",
  ready:      "Ready",
  failed:     "Training failed",
};

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

async function load() {
  const r = await fetch("/api/custom-classes", { cache: "no-store" });
  const classes = r.ok ? await r.json() : [];
  grid.innerHTML = "";
  empty.hidden = classes.length > 0;
  for (const c of classes) {
    const card = document.createElement("div");
    card.className = "cls-card";
    const statusText = STATUS_LABELS[c.status] || c.status;
    card.innerHTML = `
      <div class="cls-card-top">
        <h3>${escapeHtml(c.name)}</h3>
        <span class="cls-status cls-status-${c.status}">${statusText}</span>
      </div>
      <div class="cls-card-stats">
        <div><span class="k">Images</span><span class="v">${c.image_count}</span></div>
        <div><span class="k">Labeled</span><span class="v">${c.labeled_count}</span></div>
        <div><span class="k">Model</span><span class="v">${c.model_ready ? "✓" : "—"}</span></div>
      </div>
      <div class="cls-card-actions">
        <a class="btn btn-secondary" href="/classes/${encodeURIComponent(c.id)}">Open</a>
        <button class="btn btn-danger" data-action="delete" data-id="${c.id}">Delete</button>
      </div>
    `;
    grid.appendChild(card);
  }
  grid.querySelectorAll('button[data-action="delete"]').forEach((b) => {
    b.addEventListener("click", () => deleteClass(b.dataset.id));
  });
}

async function deleteClass(cid) {
  if (!confirm(`Delete class "${cid}" and all its images?\nThis cannot be undone.`)) return;
  const r = await fetch(`/api/custom-classes/${encodeURIComponent(cid)}`, { method: "DELETE" });
  if (r.ok) load();
  else alert("Delete failed");
}

function showModal() {
  modalErr.hidden = true;
  nameInput.value = "";
  modal.hidden = false;
  setTimeout(() => nameInput.focus(), 10);
}
function hideModal() { modal.hidden = true; }

async function createClass() {
  const name = nameInput.value.trim();
  if (!name) {
    modalErr.textContent = "Name is required";
    modalErr.hidden = false;
    return;
  }
  btnCreate.disabled = true;
  try {
    const r = await fetch("/api/custom-classes", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      modalErr.textContent = err.error || `HTTP ${r.status}`;
      modalErr.hidden = false;
      return;
    }
    const rec = await r.json();
    hideModal();
    location.href = `/classes/${encodeURIComponent(rec.id)}`;
  } finally {
    btnCreate.disabled = false;
  }
}

btnNew.addEventListener("click", showModal);
btnCancel.addEventListener("click", hideModal);
btnCreate.addEventListener("click", createClass);
nameInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") createClass();
  if (e.key === "Escape") hideModal();
});
modal.addEventListener("click", (e) => { if (e.target === modal) hideModal(); });

load();
