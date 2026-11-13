// Interactive figures for the guide pages, drawn with Observable Plot. A page places a
// figure with an empty element whose id names it, e.g. <div id="local-planes"></div>.

// Seeded random numbers, so a figure looks the same on every load.
function rng(seed) {
  let a = seed >>> 0;
  const u = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const gauss = (mean, sd) => mean + sd * Math.sqrt(-2 * Math.log(1 - u())) * Math.cos(2 * Math.PI * u());
  return { gauss, uniform: (lo, hi) => lo + (hi - lo) * u() };
}

// Centre, principal axes (first entry non-negative) and variances of 2-D points.
function pca2(points) {
  const [mx, my] = [0, 1].map((j) => d3.mean(points, (p) => p[j]));
  let a = 0, b = 0, c = 0;
  for (const [x, y] of points) {
    a += (x - mx) ** 2;
    b += (x - mx) * (y - my);
    c += (y - my) ** 2;
  }
  [a, b, c] = [a, b, c].map((s) => s / points.length);
  const t = 0.5 * Math.atan2(2 * b, a - c), r = Math.hypot((a - c) / 2, b);
  const sign = (v) => (v[0] < 0 ? v.map((x) => -x) : v);
  return {
    centre: [mx, my],
    v1: sign([Math.cos(t), Math.sin(t)]),
    v2: sign([-Math.sin(t), Math.cos(t)]),
    variances: [(a + c) / 2 + r, (a + c) / 2 - r],
  };
}

const colours = (el) => ["--eon-blue", "--eon-orange", "--eon-guide"].map((n) => getComputedStyle(el).getPropertyValue(n).trim());
const base = { style: { background: "transparent", fontSize: "12px" } };
const square = { ...base, width: 420, height: 420, marginLeft: 48, marginRight: 12, marginTop: 28, marginBottom: 44 };
const frame = () => Plot.frame({ stroke: "currentColor", strokeOpacity: 0.15 });

function slider(parent, text, min, max, value) {
  const label = parent.appendChild(document.createElement("label"));
  const shown = document.createElement("b");
  const input = document.createElement("input");
  Object.assign(input, { type: "range", min, max, value });
  label.append(`${text}: `, shown, input);
  shown.textContent = value;
  input.addEventListener("input", () => (shown.textContent = input.value));
  return input;
}

// guide/introduction.md: 100 features, and only features 19 and 72 separate the classes.
// Same geometry as make_worms in examples/01_first_network.py.
function hiddenDimensions(el) {
  const signal = rng(7), noise = rng(8), rows = [];
  for (const [centre, label] of [[0.25, "Class 1"], [0.5, "Class 2"], [0.75, "Class 1"]]) {
    for (let i = 0; i < 55; i++) {
      const along = signal.gauss(0, 0.16) / Math.SQRT2, across = signal.gauss(0, 0.0365) / Math.SQRT2;
      const f = Array.from({ length: 100 }, () => noise.uniform(0.1, 0.9));
      f[19] = (centre + along + across + 0.3) / 1.6;
      f[72] = (centre - along + across + 0.3) / 1.6;
      rows.push({ f, label });
    }
  }
  const controls = el.appendChild(document.createElement("div"));
  const h = slider(controls, "Horizontal feature", 0, 99, 0);
  const v = slider(controls, "Vertical feature", 0, 99, 1);
  const plot = el.appendChild(document.createElement("div"));
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const draw = () => {
    const [blue, orange] = colours(el), x = +h.value, y = +v.value;
    plot.replaceChildren(Plot.plot({
      ...square, grid: true,
      x: { domain: [0, 1], ticks, label: `Feature ${x}` },
      y: { domain: [0, 1], ticks, label: `Feature ${y}` },
      color: { domain: ["Class 1", "Class 2"], range: [blue, orange], legend: true },
      marks: [
        frame(),
        Plot.dot(rows, { x: (d) => d.f[x], y: (d) => d.f[y], fill: "label", r: 3.5, fillOpacity: 0.85, tip: { format: { x: ".3f", y: ".3f" } } }),
      ],
    }));
  };
  for (const input of [h, v]) input.addEventListener("input", draw);
  return draw;
}

// guide/introduction.md: the classes differ only across the short axis of a narrow cloud,
// the direction that one-component PCA discards.
function pcaLabelLoss(el) {
  const random = rng(7), theta = (35 * Math.PI) / 180, points = [];
  for (const label of ["Class 1", "Class 2"]) {
    for (let i = 0; i < 120; i++) {
      const a = random.gauss(0, 1);
      const w = (0.09 + Math.abs(random.gauss(0, 0.025))) * (label === "Class 1" ? -1 : 1);
      points.push({ x: a * Math.cos(theta) - w * Math.sin(theta), y: a * Math.sin(theta) + w * Math.cos(theta), label });
    }
  }
  const { centre, v1, v2, variances } = pca2(points.map((d) => [d.x, d.y]));
  for (const d of points) {
    const dx = d.x - centre[0], dy = d.y - centre[1];
    d.pc1 = dx * v1[0] + dy * v1[1];
    d.pc2 = dx * v2[0] + dy * v2[1];
  }
  const share = variances.map((s) => d3.format(".0%")(s / d3.sum(variances)));
  const small = { ...base, width: 300, height: 300, marginLeft: 44, marginRight: 10, marginTop: 20, marginBottom: 42 };
  // The data plot spans both histograms; the half-ranges keep one unit the same length on both axes.
  const wide = { ...small, width: 610, height: 340 };
  const ratio = (wide.width - wide.marginLeft - wide.marginRight) / (wide.height - wide.marginTop - wide.marginBottom);
  const ey = Math.max(d3.max(points, (d) => Math.abs(d.y)), d3.max(points, (d) => Math.abs(d.x)) / ratio) + 0.1;
  const ex = ey * ratio;
  const pc1 = [-1, 1].map((s) => [centre[0] + s * ex * v1[0], centre[1] + s * ex * v1[1]]);
  const draw = () => {
    const [blue, orange, guide] = colours(el);
    const color = { domain: ["Class 1", "Class 2", "PC1 direction"], range: [blue, orange, guide] };
    // Each bin holds the two classes side by side, so both show where they coincide.
    const histogram = (key, title, label) => {
      const bins = d3.bin().value((d) => d[key]).thresholds(18)(points), bars = [];
      for (const bin of bins) {
        const mid = (bin.x0 + bin.x1) / 2;
        bars.push({ x1: bin.x0, x2: mid, label: "Class 1" }, { x1: mid, x2: bin.x1, label: "Class 2" });
        for (const bar of bars.slice(-2)) bar.count = bin.filter((d) => d.label === bar.label).length;
      }
      return Plot.plot({
        ...small, title, color, x: { label }, y: { label: "Observations", grid: true },
        marks: [Plot.rectY(bars, { x1: "x1", x2: "x2", y: "count", fill: "label", insetLeft: 0.5, insetRight: 0.5 }), Plot.ruleY([0])],
      });
    };
    el.replaceChildren(
      Plot.plot({
        ...wide, title: "Data",
        color: { ...color, legend: true },
        x: { domain: [-ex, ex], label: "Feature 1" },
        y: { domain: [-ey, ey], label: "Feature 2" },
        marks: [
          frame(),
          Plot.dot(points, { x: "x", y: "y", fill: "label", r: 2, fillOpacity: 0.8 }),
          Plot.line(pc1, { stroke: () => "PC1 direction", strokeDasharray: "2,3" }),
        ],
      }),
      histogram("pc1", `PC1 (~${share[0]} of variance)`, "First principal component"),
      histogram("pc2", `PC2, discarded (~${share[1]} of variance)`, "Second principal component"),
    );
  };
  return draw;
}

// guide/manifold.md: a noisy curve, approximated by one principal line per group of neighbours.
function localPlanes(el) {
  const random = rng(7);
  const curve = d3.range(150).map((i) => {
    const x = 0.05 + (0.9 * i) / 149;
    return [x, 0.5 + 0.28 * Math.sin(2 * Math.PI * x) + random.gauss(0, 0.012)];
  });
  const count = slider(el.appendChild(document.createElement("div")), "Number of lines", 2, 8, 3);
  const plot = el.appendChild(document.createElement("div"));
  const draw = () => {
    const [blue, orange] = colours(el), k = +count.value, lines = [];
    const size = Math.floor(curve.length / k), extra = curve.length % k;
    for (let g = 0, start = 0; g < k; g++) {
      // The same split as numpy's array_split: the first groups take one point more.
      const group = curve.slice(start, (start += size + (g < extra ? 1 : 0)));
      const { centre: [cx, cy], v1: [ux, uy] } = pca2(group);
      const [lo, hi] = d3.extent(group, ([x, y]) => (x - cx) * ux + (y - cy) * uy);
      lines.push({ cx, cy, x1: cx + lo * ux, y1: cy + lo * uy, x2: cx + hi * ux, y2: cy + hi * uy });
    }
    plot.replaceChildren(Plot.plot({
      ...square, grid: true,
      x: { domain: [0, 1], label: "Feature 1" },
      y: { domain: [0, 1], label: "Feature 2" },
      color: { domain: ["Observations", "Local lines"], range: [blue, orange], legend: true },
      marks: [
        frame(),
        Plot.dot(curve, { x: "0", y: "1", fill: () => "Observations", r: 2.5, fillOpacity: 0.7 }),
        Plot.link(lines, { x1: "x1", y1: "y1", x2: "x2", y2: "y2", stroke: () => "Local lines", strokeWidth: 3, strokeLinecap: "round" }),
        Plot.dot(lines, { x: "cx", y: "cy", fill: () => "Local lines", stroke: "var(--md-default-bg-color)", strokeWidth: 1.5, r: 5 }),
      ],
    }));
  };
  count.addEventListener("input", draw);
  return draw;
}

const figures = { "hidden-dimensions": hiddenDimensions, "pca-label-loss": pcaLabelLoss, "local-planes": localPlanes };
let redraws = [];

// Material's page observable also covers navigation if instant loading is enabled.
document$.subscribe(() => {
  redraws = [];
  for (const [id, build] of Object.entries(figures)) {
    const el = document.getElementById(id);
    if (!el) continue;
    el.replaceChildren();
    const draw = build(el);
    draw();
    redraws.push(draw);
  }
});

// Redraw in the new colours when the reader switches between light and dark.
new MutationObserver(() => redraws.forEach((draw) => draw())).observe(document.body, {
  attributes: true,
  attributeFilter: ["data-md-color-scheme"],
});
