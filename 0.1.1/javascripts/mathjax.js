// Arithmatex marks documentation math; MathJax typesets only that.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
  // Material's page event owns typesetting, including the initial page.
  startup: { typeset: false },
};

// Material's page observable also covers navigation if instant loading is enabled.
document$.subscribe(() => {
  MathJax.startup.promise = MathJax.startup.promise.then(() => {
    MathJax.startup.output.clearCache();
    MathJax.typesetClear();
    MathJax.texReset();
    return MathJax.typesetPromise();
  });
});
