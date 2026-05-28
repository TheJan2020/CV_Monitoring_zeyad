// Sidebar collapse/expand with localStorage persistence.
(() => {
  const sidebar = document.getElementById("sidebar");
  const toggle = document.getElementById("toggle-sidebar");
  if (!sidebar || !toggle) return;

  const KEY = "cvm.sidebar.collapsed";
  if (localStorage.getItem(KEY) === "1") {
    sidebar.classList.add("collapsed");
  }

  toggle.addEventListener("click", () => {
    sidebar.classList.toggle("collapsed");
    localStorage.setItem(KEY, sidebar.classList.contains("collapsed") ? "1" : "0");
  });
})();
