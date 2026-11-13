// Keep same-origin marimo exports tall enough to read without a nested scrollbar.
document.querySelectorAll("iframe.notebook-frame").forEach((frame) => {
  let resizeObserver;
  let mutationObserver;
  const observe = () => {
    const doc = frame.contentDocument;
    if (!doc || !doc.body) return;
    if (resizeObserver) resizeObserver.disconnect();
    if (mutationObserver) mutationObserver.disconnect();
    const fit = () => {
      const app = doc.getElementById("App");
      const height = app ? app.scrollHeight : doc.documentElement.scrollHeight;
      frame.style.height = `${Math.max(700, height)}px`;
    };
    fit();
    // The body follows the viewport, so its size alone misses React adding cells.
    resizeObserver = new ResizeObserver(fit);
    resizeObserver.observe(doc.body);
    mutationObserver = new MutationObserver(fit);
    mutationObserver.observe(doc.body, { childList: true, subtree: true });
  };
  frame.addEventListener("load", observe);
  // The export may already have loaded by the time this script runs.
  observe();
});
