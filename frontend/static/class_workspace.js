// Class workspace — upload images, list grid, delete.

const CID = window.__CLASS_ID;
const grid = document.getElementById("cw-grid");
const empty = document.getElementById("cw-empty");
const stats = document.getElementById("cw-stats");
const titleEl = document.getElementById("cw-title");
const drop = document.getElementById("cw-drop");
const fileInput = document.getElementById("cw-file");
const btnBrowse = document.getElementById("cw-browse");
const btnDelete = document.getElementById("cw-delete");
const uploadStatus = document.getElementById("cw-upload-status");
const gridCount = document.getElementById("cw-grid-count");

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

async function loadMeta() {
  const r = await fetch("/api/custom-classes", { cache: "no-store" });
  const all = r.ok ? await r.json() : [];
  const me = all.find((c) => c.id === CID);
  if (!me) {
    document.querySelector(".hist-section").innerHTML =
      `<p class="muted">Class not found. <a href="/classes">← Back</a></p>`;
    return null;
  }
  titleEl.textContent = me.name;
  document.title = `${me.name} — PrimeAnalyze`;
  stats.textContent = `${me.image_count} image${me.image_count === 1 ? "" : "s"} · ${me.labeled_count} labeled`;
  return me;
}

async function loadImages() {
  const r = await fetch(`/api/custom-classes/${encodeURIComponent(CID)}/images`, { cache: "no-store" });
  const imgs = r.ok ? await r.json() : [];
  gridCount.textContent = imgs.length ? `(${imgs.length})` : "";
  empty.hidden = imgs.length > 0;
  grid.innerHTML = imgs.map((im) => `
    <div class="cw-thumb" data-id="${im.id}">
      <img src="/api/custom-classes/${encodeURIComponent(CID)}/images/${im.id}"
           alt="" loading="lazy">
      <div class="cw-thumb-meta">
        <span class="cw-thumb-badge cw-badge-${im.labeled ? "labeled" : "unlabeled"}">
          ${im.labeled ? `${im.box_count} box${im.box_count === 1 ? "" : "es"}` : "unlabeled"}
        </span>
        <button class="cw-thumb-del" data-id="${im.id}" title="Delete">✕</button>
      </div>
    </div>
  `).join("");
  grid.querySelectorAll(".cw-thumb-del").forEach((b) =>
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteImage(b.dataset.id);
    })
  );
}

async function deleteImage(imgId) {
  await fetch(`/api/custom-classes/${encodeURIComponent(CID)}/images/${imgId}`, { method: "DELETE" });
  refresh();
}

async function refresh() {
  await loadMeta();
  await loadImages();
}

async function upload(files) {
  if (!files || !files.length) return;
  uploadStatus.textContent = `Uploading ${files.length}…`;
  const fd = new FormData();
  for (const f of files) fd.append("file", f, f.name);
  try {
    const r = await fetch(`/api/custom-classes/${encodeURIComponent(CID)}/images`, {
      method: "POST",
      body: fd,
    });
    const data = await r.json().catch(() => ({}));
    if (r.ok) {
      const okN = (data.saved || []).length;
      const errN = (data.errors || []).length;
      uploadStatus.textContent = errN
        ? `Saved ${okN}, skipped ${errN}: ${data.errors.map((e) => e.error).join(", ")}`
        : `Saved ${okN}.`;
    } else {
      uploadStatus.textContent = `Upload failed: ${data.error || r.status}`;
    }
  } catch (e) {
    uploadStatus.textContent = "Upload failed: " + e.message;
  }
  refresh();
}

btnBrowse.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => upload(fileInput.files));

["dragenter", "dragover"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("dragging"); })
);
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, () => drop.classList.remove("dragging"))
);
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  const files = [...(e.dataTransfer?.files || [])].filter((f) => f.type.startsWith("image/"));
  upload(files);
});

btnDelete.addEventListener("click", async () => {
  if (!confirm("Delete this entire class and every image?\nThis cannot be undone.")) return;
  const r = await fetch(`/api/custom-classes/${encodeURIComponent(CID)}`, { method: "DELETE" });
  if (r.ok) location.href = "/classes";
  else alert("Delete failed");
});

refresh();
