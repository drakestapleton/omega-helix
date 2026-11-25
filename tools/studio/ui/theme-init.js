/* Prevent flash of un-themed content under script-src 'self' CSP */
(() => {
  try {
    const saved = localStorage.getItem("omega-studio-color-scheme");
    if (saved === "light" || saved === "dark") {
      const m = document.querySelector('meta[name="color-scheme"]');
      if (m) m.setAttribute("content", saved);
    }
  } catch (_) {}
})();
