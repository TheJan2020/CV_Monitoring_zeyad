// === Sidebar collapse / expand =======================================
(() => {
  const sidebar = document.getElementById("sidebar");
  const toggle = document.getElementById("toggle-sidebar");
  if (!sidebar || !toggle) return;

  const KEY = "primeanalyze.sidebar.collapsed";
  if (localStorage.getItem(KEY) === "1") {
    sidebar.classList.add("collapsed");
  }
  toggle.addEventListener("click", () => {
    sidebar.classList.toggle("collapsed");
    localStorage.setItem(KEY, sidebar.classList.contains("collapsed") ? "1" : "0");
  });
})();

// === Theme switching =================================================
(() => {
  const KEY = "primeanalyze.theme";
  const toggle = document.getElementById("theme-toggle");
  if (!toggle) return;

  toggle.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme") || "light";
    const next = cur === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem(KEY, next);
  });
})();
