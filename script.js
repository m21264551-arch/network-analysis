const views = {
  overview: {
    mode: "network",
    kicker: "Interactive Graph Explorer",
    title: "Karate Club Network",
    insightTitle: "Hover nodes to inspect influence and community membership",
    stats: [
      ["Nodes", "34"],
      ["Edges", "78"],
      ["Communities", "4"],
      ["Modularity", "0.4266"],
    ],
    copy:
      "Graph data: JSON export with hover, filtering, and centrality-based node sizing.",
    note: "Source: data/network-karate.json",
    download: "data/network-karate.json",
  },
  communities: {
    mode: "chart",
    chart: "community",
    kicker: "Community Detection",
    title: "Algorithm Comparison",
    insightTitle: "Modularity and NMI tell different stories",
    stats: [
      ["Best modularity", "0.4266"],
      ["Highest NMI", "0.7324"],
      ["Algorithms", "4"],
      ["Best method", "Louvain"],
    ],
    copy:
      "Compare modularity, NMI, and community count across detection algorithms.",
    note: "Output: detector comparison",
    download: "data/dashboard-metrics.json",
  },
  influence: {
    mode: "chart",
    chart: "influence",
    kicker: "Influence Analysis",
    title: "Seed Strategy Reach",
    insightTitle: "Centrality choices change simulated reach",
    stats: [
      ["Seed size", "5"],
      ["Simulations", "100"],
      ["Propagation p", "0.10"],
      ["Output", "Mean reach"],
    ],
    copy:
      "Compare seed strategies by mean activated reach.",
    note: "Output: influence spread",
    download: "data/dashboard-metrics.json",
  },
  links: {
    mode: "chart",
    chart: "heuristics",
    kicker: "Link Prediction",
    title: "Heuristic Evaluation",
    insightTitle: "Preferential attachment is the strongest heuristic",
    stats: [
      ["Best heuristic AUC", "0.7882"],
      ["Best avg precision", "0.8243"],
      ["Held-out positives", "12"],
      ["Held-out negatives", "12"],
    ],
    copy:
      "Compare AUC and average precision on held-out edges.",
    note: "Best heuristic: Preferential Attachment, AUC 0.7882",
    download: "data/dashboard-metrics.json",
  },
  methods: {
    mode: "chart",
    chart: "centrality",
    kicker: "Centrality Methods",
    title: "Top Centrality Nodes",
    insightTitle: "Reusable modules make the work reviewable",
    stats: [
      ["Package", "src/*"],
      ["CLI", "main.py"],
      ["Notebook", "analysis.ipynb"],
      ["Tests", "pytest"],
    ],
    copy:
      "Source modules, CLI, notebook, generated figures, and tests.",
    note: "Run: python main.py",
    download: "data/dashboard-metrics.json",
  },
};

const colors = ["#2f7d6d", "#2f66b3", "#d97834", "#8b6bd6", "#6b7280"];
const tabButtons = document.querySelectorAll("[data-tab]");
const mainKicker = document.querySelector("#main-kicker");
const mainTitle = document.querySelector("#main-title");
const mainImage = document.querySelector("#main-image");
const mainChart = document.querySelector("#main-chart");
const networkExplorer = document.querySelector("#network-explorer");
const insightTitle = document.querySelector("#insight-title");
const statList = document.querySelector("#stat-list");
const insightCopy = document.querySelector("#insight-copy");
const runNote = document.querySelector("#run-note");
const downloadButton = document.querySelector("#download-current");
const zoomButton = document.querySelector("#zoom-toggle");
const primaryPanel = document.querySelector(".primary-panel");
const svg = document.querySelector("#network-svg");
const tooltip = document.querySelector("#node-tooltip");
const communityFilter = document.querySelector("#community-filter");
const sizeButtons = document.querySelectorAll("[data-size-metric]");

let currentView = "overview";
let graphData = null;
let dashboardData = null;
let sizeMetric = "pagerank";
let activeCommunity = "all";

function renderView(key) {
  const view = views[key] || views.overview;
  currentView = key;

  tabButtons.forEach((button) => {
    const active = button.dataset.tab === key;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-selected", String(active));
  });

  mainKicker.textContent = view.kicker;
  mainTitle.textContent = view.title;
  insightTitle.textContent = view.insightTitle;
  insightCopy.textContent = view.copy;
  runNote.textContent = view.note;

  networkExplorer.classList.toggle("is-hidden", view.mode !== "network");
  mainImage.classList.toggle("is-hidden", view.mode !== "image");
  mainChart.classList.toggle("is-hidden", view.mode !== "chart");

  if (view.mode === "image") {
    mainImage.src = view.image;
    mainImage.alt = view.alt;
  } else if (view.mode === "network" && graphData) {
    renderNetwork();
  }
  if (view.mode === "chart" && dashboardData) {
    renderChart(mainChart, view.chart, true);
  }

  statList.innerHTML = view.stats
    .map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`)
    .join("");
}

function renderAllCharts() {
  if (!dashboardData) return;
  document.querySelectorAll("[data-chart]").forEach((container) => {
    renderChart(container, container.dataset.chart, false);
  });
  if (views[currentView]?.mode === "chart") {
    renderChart(mainChart, views[currentView].chart, true);
  }
}

function renderChart(container, key, large = false) {
  const title = {
    community: "Community Detection Metrics",
    influence: "Influence Reach by Seed Strategy",
    heuristics: "Link Prediction Heuristic Evaluation",
    features: "Supervised Model Feature Importance",
    centrality: "Top Nodes by PageRank",
  }[key];
  container.innerHTML = `<div class="chart-title">${title}</div>`;

  if (key === "community") {
    container.append(renderGroupedBars(
      dashboardData.communityDetection,
      "method",
      [
        ["modularity", "Modularity", "#2f7d6d"],
        ["nmi", "NMI", "#2f66b3"],
      ],
      large,
    ));
  }
  if (key === "influence") {
    container.append(renderSingleBars(
      dashboardData.influence,
      "strategy",
      "meanReach",
      { color: "#2f7d6d", suffix: "%", scale: 100, large },
    ));
  }
  if (key === "heuristics") {
    container.append(renderGroupedBars(
      dashboardData.heuristicEvaluation,
      "method",
      [
        ["auc", "AUC", "#2f7d6d"],
        ["averagePrecision", "Avg precision", "#d97834"],
      ],
      large,
    ));
    const best = [...dashboardData.heuristicEvaluation].sort((a, b) => b.auc - a.auc)[0];
    container.append(renderCaption(`Best by AUC: ${best.method} (${best.auc.toFixed(4)}).`));
  }
  if (key === "features") {
    container.append(renderHorizontalBars(
      dashboardData.supervisedModel.featureImportances,
      "feature",
      "importance",
      { color: "#2f7d6d", large },
    ));
  }
  if (key === "centrality") {
    const pagerank = dashboardData.centralityTopNodes.find((item) => item.metric === "pagerank");
    container.append(renderHorizontalBars(
      pagerank.nodes.map((node) => ({ label: `Node ${node.node}`, value: node.value })),
      "label",
      "value",
      { color: "#2f66b3", large },
    ));
  }
}

function chartSvg({ width, height, large = false } = {}) {
  const chartWidth = width ?? (large ? 680 : 520);
  const chartHeight = height ?? (large ? 520 : 260);
  const svgEl = createSvg("svg", {
    class: `chart-svg${large ? " is-large" : ""}`,
    viewBox: `0 0 ${chartWidth} ${chartHeight}`,
    role: "img",
  });
  return { svgEl, width: chartWidth, height: chartHeight };
}

function renderGroupedBars(rows, labelKey, series, large = false) {
  const { svgEl, width, height } = chartSvg({ large });
  const margin = { top: 18, right: 18, bottom: large ? 74 : 58, left: 42 };
  const chartW = width - margin.left - margin.right;
  const chartH = height - margin.top - margin.bottom;
  const max = Math.max(1, ...rows.flatMap((row) => series.map(([key]) => row[key])));
  const groupW = chartW / rows.length;
  const barW = Math.min(28, (groupW - 18) / series.length);

  svgEl.append(...gridLines(width, margin, chartH, max));
  rows.forEach((row, rowIndex) => {
    const groupX = margin.left + rowIndex * groupW + groupW / 2;
    series.forEach(([key, label, color], seriesIndex) => {
      const value = row[key];
      const h = (value / max) * chartH;
      const x = groupX - ((series.length * barW) / 2) + seriesIndex * barW;
      const y = margin.top + chartH - h;
      svgEl.append(barRect({ x, y, width: barW - 3, height: h, color, label: `${row[labelKey]} ${label}: ${value.toFixed(4)}` }));
    });
    svgEl.append(axisLabel(groupX, margin.top + chartH + 18, compactLabel(row[labelKey]), large));
  });
  svgEl.append(renderLegend(series.map(([, label, color]) => ({ label, color })), width - margin.right, 12));
  return svgEl;
}

function renderSingleBars(rows, labelKey, valueKey, options) {
  const { color, suffix = "", scale = 1, large = false } = options;
  const { svgEl, width, height } = chartSvg({ large });
  const margin = { top: 18, right: 18, bottom: large ? 74 : 56, left: 42 };
  const chartW = width - margin.left - margin.right;
  const chartH = height - margin.top - margin.bottom;
  const max = Math.max(...rows.map((row) => row[valueKey]));
  const groupW = chartW / rows.length;
  const barW = Math.min(46, groupW * 0.48);

  svgEl.append(...gridLines(width, margin, chartH, max));
  rows.forEach((row, index) => {
    const value = row[valueKey];
    const h = (value / max) * chartH;
    const x = margin.left + index * groupW + (groupW - barW) / 2;
    const y = margin.top + chartH - h;
    svgEl.append(barRect({ x, y, width: barW, height: h, color, label: `${row[labelKey]}: ${(value * scale).toFixed(1)}${suffix}` }));
    svgEl.append(axisLabel(x + barW / 2, margin.top + chartH + 18, compactLabel(row[labelKey]), large));
    svgEl.append(valueLabel(x + barW / 2, y - 7, `${(value * scale).toFixed(1)}${suffix}`));
  });
  return svgEl;
}

function renderHorizontalBars(rows, labelKey, valueKey, options) {
  const { color, large = false } = options;
  const { svgEl, width, height } = chartSvg({ large });
  const margin = { top: 18, right: 24, bottom: 18, left: large ? 140 : 116 };
  const chartW = width - margin.left - margin.right;
  const rowH = (height - margin.top - margin.bottom) / rows.length;
  const max = Math.max(...rows.map((row) => row[valueKey]));

  rows.forEach((row, index) => {
    const y = margin.top + index * rowH + rowH * 0.18;
    const w = (row[valueKey] / max) * chartW;
    svgEl.append(axisText(margin.left - 10, y + rowH * 0.35, compactLabel(row[labelKey]), "end"));
    svgEl.append(barRect({ x: margin.left, y, width: w, height: rowH * 0.5, color, label: `${row[labelKey]}: ${row[valueKey].toFixed(4)}` }));
    svgEl.append(valueLabel(margin.left + w + 8, y + rowH * 0.35 + 4, row[valueKey].toFixed(3), "start"));
  });
  return svgEl;
}

function gridLines(width, margin, chartH, max) {
  return [0, 0.5, 1].map((tick) => {
    const y = margin.top + chartH - tick * chartH;
    const line = createSvg("line", {
      class: "chart-grid-line",
      x1: margin.left,
      x2: width - margin.right,
      y1: y,
      y2: y,
    });
    const label = axisText(margin.left - 8, y + 4, (tick * max).toFixed(max <= 1 ? 1 : 0), "end");
    const group = createSvg("g");
    group.append(line, label);
    return group;
  });
}

function barRect({ x, y, width, height, color, label }) {
  const rect = createSvg("rect", {
    class: "chart-bar",
    x,
    y,
    width: Math.max(1, width),
    height: Math.max(1, height),
    rx: 4,
    fill: color,
    tabindex: 0,
  });
  rect.append(createSvg("title", {}, label));
  return rect;
}

function renderLegend(items, rightX, y) {
  const group = createSvg("g", { class: "chart-legend" });
  const swatchWidth = 9;
  const labelGap = 14;
  const itemGap = 24;
  const charWidth = 9.6;
  const itemWidths = items.map((item) => swatchWidth + labelGap + item.label.length * charWidth);
  const totalWidth = itemWidths.reduce((sum, width) => sum + width, 0) + itemGap * (items.length - 1);
  let cursor = Math.max(46, rightX - totalWidth);

  items.forEach((item, index) => {
    group.append(createSvg("rect", { x: cursor, y, width: swatchWidth, height: 9, rx: 2, fill: item.color }));
    group.append(axisText(cursor + labelGap, y + 9, item.label, "start"));
    cursor += itemWidths[index] + itemGap;
  });
  return group;
}

function axisLabel(x, y, text, large) {
  const label = axisText(x, y, text, "middle");
  if (large) label.setAttribute("font-size", "13");
  return label;
}

function axisText(x, y, text, anchor = "middle") {
  return createSvg("text", {
    class: "chart-axis-text",
    x,
    y,
    "text-anchor": anchor,
  }, text);
}

function valueLabel(x, y, text, anchor = "middle") {
  return createSvg("text", {
    class: "chart-value-text",
    x,
    y,
    "text-anchor": anchor,
  }, text);
}

function renderCaption(text) {
  const caption = document.createElement("p");
  caption.className = "chart-caption";
  caption.textContent = text;
  return caption;
}

function compactLabel(value) {
  const map = {
    "Label Propagation": "Label Prop.",
    "Spectral Clustering": "Spectral",
    "Jaccard Coefficient": "Jaccard",
    "Preferential Attachment": "Pref. Attach.",
    "Common Neighbors": "Common N.",
  };
  return map[value] || value;
}

function metricValue(node) {
  if (sizeMetric === "degree") return node.degree;
  if (sizeMetric === "betweenness") return node.betweenness;
  return node.pagerank;
}

function radiusScale(nodes) {
  const values = nodes.map(metricValue);
  const min = Math.min(...values);
  const max = Math.max(...values);
  return (node) => {
    if (max === min) return 10;
    return 5 + ((metricValue(node) - min) / (max - min)) * 18;
  };
}

function transformNodes(nodes) {
  const width = 1000;
  const height = 520;
  const margin = 58;
  const xs = nodes.map((node) => node.x);
  const ys = nodes.map((node) => node.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  return nodes.map((node) => ({
    ...node,
    sx: margin + ((node.x - minX) / (maxX - minX || 1)) * (width - margin * 2),
    sy: margin + ((node.y - minY) / (maxY - minY || 1)) * (height - margin * 2),
  }));
}

function renderCommunityControls() {
  const communities = [...new Set(graphData.nodes.map((node) => node.community))].sort(
    (a, b) => a - b,
  );
  communityFilter.innerHTML = [
    `<button class="${activeCommunity === "all" ? "is-active" : ""}" type="button" data-community="all">All</button>`,
    ...communities.map(
      (community) =>
        `<button class="${String(activeCommunity) === String(community) ? "is-active" : ""}" type="button" data-community="${community}">C${community}</button>`,
    ),
  ].join("");
}

function renderNetwork() {
  if (!graphData) return;

  const nodes = transformNodes(graphData.nodes);
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const radius = radiusScale(nodes);
  const activeNodes = new Set(
    nodes
      .filter((node) => activeCommunity === "all" || String(node.community) === String(activeCommunity))
      .map((node) => node.id),
  );
  const topNodes = new Set(
    [...nodes].sort((a, b) => b.pagerank - a.pagerank).slice(0, 6).map((node) => node.id),
  );

  svg.setAttribute("viewBox", "0 0 1000 520");
  svg.innerHTML = "";

  const edgeGroup = createSvg("g", { class: "edge-layer" });
  graphData.edges.forEach((edge) => {
    const source = nodeMap.get(edge.source);
    const target = nodeMap.get(edge.target);
    const hidden = !activeNodes.has(source.id) || !activeNodes.has(target.id);
    edgeGroup.append(
      createSvg("line", {
        class: `network-edge${hidden ? " is-dim" : ""}`,
        x1: source.sx,
        y1: source.sy,
        x2: target.sx,
        y2: target.sy,
        "data-source": source.id,
        "data-target": target.id,
      }),
    );
  });

  const nodeGroup = createSvg("g", { class: "node-layer" });
  nodes.forEach((node) => {
    const hidden = !activeNodes.has(node.id);
    const circle = createSvg("circle", {
      class: `network-node${hidden ? " is-dim" : ""}`,
      cx: node.sx,
      cy: node.sy,
      r: radius(node).toFixed(2),
      fill: colors[node.community % colors.length],
      tabindex: 0,
      role: "button",
      "aria-label": `Node ${node.id}, community ${node.community}, PageRank ${node.pagerank}`,
      "data-node": node.id,
    });
    circle.addEventListener("pointerenter", (event) => showNode(node, event));
    circle.addEventListener("pointermove", (event) => positionTooltip(event));
    circle.addEventListener("pointerleave", clearNode);
    circle.addEventListener("click", (event) => showNode(node, event));
    circle.addEventListener("focus", (event) => showNode(node, event));
    circle.addEventListener("blur", clearNode);
    circle.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        showNode(node, event);
      }
    });
    nodeGroup.append(circle);

    if (topNodes.has(node.id)) {
      nodeGroup.append(
        createSvg("text", {
          class: "network-label",
          x: node.sx + radius(node) + 5,
          y: node.sy + 4,
        }, String(node.id)),
      );
    }
  });

  svg.append(edgeGroup, nodeGroup);
  renderCommunityControls();
}

function createSvg(tag, attrs = {}, text = "") {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
  if (text) el.textContent = text;
  return el;
}

function connectedNodeIds(nodeId) {
  const ids = new Set([nodeId]);
  graphData.edges.forEach((edge) => {
    if (edge.source === nodeId) ids.add(edge.target);
    if (edge.target === nodeId) ids.add(edge.source);
  });
  return ids;
}

function showNode(node, event) {
  const connected = connectedNodeIds(node.id);
  svg.querySelectorAll(".network-node").forEach((circle) => {
    const id = Number(circle.dataset.node);
    circle.classList.toggle("is-active", id === node.id);
    circle.classList.toggle("is-dim", !connected.has(id));
  });
  svg.querySelectorAll(".network-edge").forEach((line) => {
    const source = Number(line.dataset.source);
    const target = Number(line.dataset.target);
    const active = source === node.id || target === node.id;
    line.classList.toggle("is-dim", !active);
  });

  tooltip.hidden = false;
  tooltip.innerHTML = `
    <strong>Node ${node.id}</strong>
    <span><em>Community</em><b>${node.community}</b></span>
    <span><em>Degree</em><b>${node.degree}</b></span>
    <span><em>PageRank</em><b>${node.pagerank.toFixed(4)}</b></span>
    <span><em>Betweenness</em><b>${node.betweenness.toFixed(4)}</b></span>
  `;
  positionTooltip(event);
}

function positionTooltip(event) {
  const bounds = document.querySelector(".network-stage").getBoundingClientRect();
  const clientX = Number.isFinite(event.clientX) ? event.clientX : bounds.left + bounds.width / 2;
  const clientY = Number.isFinite(event.clientY) ? event.clientY : bounds.top + bounds.height / 2;
  const x = Math.min(clientX - bounds.left + 16, bounds.width - 230);
  const y = Math.max(clientY - bounds.top - 20, 12);
  tooltip.style.left = `${Math.max(12, x)}px`;
  tooltip.style.top = `${y}px`;
}

function clearNode() {
  tooltip.hidden = true;
  svg.querySelectorAll(".network-node, .network-edge").forEach((el) => {
    el.classList.remove("is-active", "is-dim");
  });
  if (activeCommunity !== "all") renderNetwork();
}

tabButtons.forEach((button) => {
  button.addEventListener("click", () => renderView(button.dataset.tab));
});

sizeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    sizeMetric = button.dataset.sizeMetric;
    sizeButtons.forEach((item) => item.classList.toggle("is-active", item === button));
    renderNetwork();
  });
});

communityFilter.addEventListener("click", (event) => {
  const button = event.target.closest("[data-community]");
  if (!button) return;
  activeCommunity = button.dataset.community;
  renderNetwork();
});

downloadButton.addEventListener("click", () => {
  const asset = views[currentView].download;
  const link = document.createElement("a");
  link.href = asset;
  link.download = asset.split("/").pop();
  link.click();
});

zoomButton.addEventListener("click", () => {
  const expanded = primaryPanel.classList.toggle("is-expanded");
  zoomButton.title = expanded ? "Collapse panel" : "Expand panel";
  if (graphData && views[currentView].mode === "network") renderNetwork();
});

Promise.all([
  fetch("data/network-karate.json").then((response) => response.json()),
  fetch("data/dashboard-metrics.json").then((response) => response.json()),
])
  .then(([networkPayload, metricsPayload]) => {
    graphData = networkPayload;
    dashboardData = metricsPayload;
    renderAllCharts();
    renderView("overview");
  })
  .catch(() => {
    networkExplorer.classList.add("is-hidden");
    mainImage.classList.remove("is-hidden");
    renderView("communities");
  });
