const $ = (selector) => document.querySelector(selector);
const colors = [
  "#2e6749",
  "#ad5b2a",
  "#506aad",
  "#8d477b",
  "#807227",
  "#476e76",
  "#746554",
  "#7245a1",
];
const symbols = ["●", "■", "▲", "◆", "✚", "○", "□", "△"];
const fmt = (n, d = 3) => (Number.isFinite(n) ? n.toFixed(d) : "Unavailable");
const pct = (n) => `${fmt(n * 100, 1)}%`;
let bundle,
  data,
  view = "groups",
  method = "Louvain",
  size = "pagerank",
  selected = 0,
  probability = 0.1,
  count = 3,
  strategy = "PageRank",
  frame = 0;
function validate(value) {
  if (
    value.schemaVersion !== 1 ||
    !Array.isArray(value.experiments) ||
    value.experiments.length !== 3
  )
    throw Error("Unsupported or incomplete experiment bundle.");
  for (const e of value.experiments)
    if (
      !e.nodes?.length ||
      !e.edges?.length ||
      e.communities?.length !== 4 ||
      e.linkRuns?.length !== 20 ||
      !e.influence?.length
    )
      throw Error("The experiment bundle is incomplete.");
  return value;
}
const table = (heads, rows) =>
  `<div class="table-wrap" tabindex="0" role="region" aria-label="${heads[0]} results"><table><thead><tr>${heads.map((x) => `<th scope="col">${x}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((x) => `<td>${x}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
const options = (items, value) =>
  items
    .map(
      (x) =>
        `<option ${String(x) === String(value) ? "selected" : ""}>${x}</option>`,
    )
    .join("");
function graph(partition, active = null, seeds = []) {
  const pos = (n) => ({ x: 55 + (n.x + 1) * 285, y: 45 + (n.y + 1) * 230 });
  const nodeMap = new Map(data.nodes.map((n) => [n.id, n]));
  const max = Math.max(...data.nodes.map((n) => n[size]));
  const edges = data.edges
    .map((e) => {
      const a = pos(nodeMap.get(e.source)),
        b = pos(nodeMap.get(e.target));
      return `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="#cbd6c9" stroke-width="1.3"/>`;
    })
    .join("");
  const nodes = data.nodes
    .map((n) => {
      const p = pos(n),
        group = partition[n.id] ?? n.reference,
        r = 8 + Math.sqrt(n[size] / max) * 9;
      const on = active === null || active.has(n.id);
      return `<g class="node" role="button" tabindex="0" data-node="${n.id}" aria-label="Inspect node ${n.id}, group ${group + 1}, ${n.degree_count} neighbours" aria-pressed="${n.id === selected}"><circle cx="${p.x}" cy="${p.y}" r="${r}" fill="${on ? colors[group % colors.length] : "#fff"}" stroke="${n.id === selected ? "#152a1c" : colors[group % colors.length]}" stroke-width="${n.id === selected ? 4 : 1.5}"/><text x="${p.x}" y="${p.y + 4}" text-anchor="middle" font-size="10" font-weight="700" fill="${on ? "#fff" : "#24362d"}">${n.id}</text>${seeds.includes(n.id) ? `<circle cx="${p.x}" cy="${p.y}" r="${r + 5}" fill="none" stroke="#1d3927" stroke-dasharray="3 2"/>` : ""}</g>`;
    })
    .join("");
  return `<figure class="plot"><svg viewBox="0 0 680 540" aria-label="${data.name} graph, select a numbered node to inspect its position" role="group">${edges}${nodes}</svg><figcaption>Edges are unweighted. Node numbers identify individuals or synthetic nodes. ${active ? "Filled nodes are active; dashed rings identify seeds." : "Colour indicates group; numeric group labels appear in node details."} Positions are fixed across comparisons.</figcaption></figure>`;
}
function details(partition) {
  const n = data.nodes.find((n) => n.id === selected) ?? data.nodes[0];
  return `<h3>Node ${n.id}</h3><p>Reference group ${n.reference + 1} · ${view === "groups" ? `displayed group ${(partition[n.id] ?? n.reference) + 1}` : "inspect an activated node or a seed"}</p><dl><dt>Neighbours</dt><dd>${n.degree_count}</dd><dt>PageRank</dt><dd>${fmt(n.pagerank, 4)}</dd><dt>Betweenness</dt><dd>${fmt(n.betweenness, 4)}</dd><dt>Eigenvector</dt><dd>${fmt(n.eigenvector, 4)}</dd></dl><p class="muted">Degree counts immediate neighbours. PageRank rewards links from well-connected nodes. Betweenness measures how often a node lies on shortest paths. These answer different questions.</p>`;
}
function render() {
  if (!data) return;
  $("#counts").textContent =
    `${data.nodes.length} nodes · ${data.edges.length} edges`;
  $("#description").textContent = data.description;
  document
    .querySelectorAll("[data-view]")
    .forEach((b) => b.setAttribute("aria-pressed", b.dataset.view === view));
  const partition =
    method === "Reference groups"
      ? Object.fromEntries(data.nodes.map((n) => [n.id, n.reference]))
      : data.communities.find((c) => c.method === method).partition;
  let content = "";
  if (view === "groups") {
    const best = [...data.communities].sort((a, b) => b.nmi - a.nmi)[0];
    content = `<div class="experiment"><section><div class="controls"><label>Display groups<select id="method">${options(["Reference groups", ...data.communities.map((c) => c.method)], method)}</select></label><label>Size nodes by<select id="size">${options(["pagerank", "degree", "betweenness"], size)}</select></label></div>${graph(partition)}</section><aside><h2>Do the groups agree?</h2><p>Compare a partition with the observed split or planted labels. A visually convincing cluster is not necessarily the reference group.</p><div class="note">${best.method} has the highest label agreement here: NMI ${fmt(best.nmi)}. This is one fixed graph, not a universal ranking.</div>${details(partition)}</aside></div><h3>Partition comparison</h3>${table(
      ["Method", "Groups", "Modularity", "NMI ↑", "Adjusted Rand ↑"],
      data.communities.map((c) => [
        c.method,
        c.num_communities,
        fmt(c.modularity),
        fmt(c.nmi),
        fmt(c.ari),
      ]),
    )}<p class="muted">Modularity measures dense internal connections relative to a null model; NMI and adjusted Rand compare against reference labels. Spectral clustering and Girvan–Newman are given the known number of groups. Louvain and label propagation choose their own partition; this is not an equal-information contest.</p>`;
  } else if (view === "influence") {
    const rows = data.influence.filter(
        (r) => r.seedCount === count && r.probability === probability,
      ),
      chosen = rows.find((r) => r.strategy === strategy);
    frame = Math.min(frame, chosen.replay.length - 1);
    const active = new Set(chosen.replay.slice(0, frame + 1).flat()),
      same = rows
        .filter(
          (r) =>
            r.strategy !== strategy &&
            [...r.seeds].sort().join() === [...chosen.seeds].sort().join(),
        )
        .map((r) => r.strategy);
    content = `<div class="experiment"><section><div class="controls"><label>Seed strategy<select id="strategy">${options(
      rows.map((r) => r.strategy),
      strategy,
    )}</select></label><label>Seed count<select id="count">${options([1, 3, 5], count)}</select></label><label>Propagation probability<select id="probability">${options([0.05, 0.1, 0.2], probability)}</select></label></div>${graph(Object.fromEntries(data.nodes.map((n) => [n.id, n.reference])), active, chosen.seeds)}<div class="controls"><button id="restart">Restart replay</button><button id="step" ${frame === chosen.replay.length - 1 ? "disabled" : ""}>Next wave →</button><span>Wave ${frame} · ${active.size} active nodes</span></div></section><aside><h2>Where should spread begin?</h2><p>Each active node has one independent chance to activate each neighbour. This is an assumption about diffusion, not a measurement of human behaviour.</p><p>Seeds: <strong>${chosen.seeds.join(", ")}</strong></p><div class="note">${same.length ? `${same.join(" and ")} selects the same seed set. Paired trials must therefore give the same reach.` : `${strategy} reaches ${pct(chosen.mean)} on average across ${chosen.reaches.length} paired trials. The replay shows trial 0, not the average.`}</div>${details(partition)}</aside></div><h3>Reach across the same random worlds</h3><div class="bars">${rows.map((r) => `<div class="bar-row"><span>${r.strategy}</span><div class="track"><div class="fill" style="width:${r.mean * 100}%"></div><div class="spread" style="left:${Math.max(0, r.mean - r.std) * 100}%;width:${(Math.min(1, r.mean + r.std) - Math.max(0, r.mean - r.std)) * 100}%"></div></div><span>${pct(r.mean)}</span></div>`).join("")}</div>${table(
      ["Strategy", "Mean reach", "SD of reach", "Seeds"],
      rows.map((r) => [
        r.strategy,
        pct(r.mean),
        pct(r.std),
        r.seeds.join(", "),
      ]),
    )}<p class="muted">Dark bars show ±1 standard deviation, not confidence intervals. Random chooses one fixed seed set; trials vary propagation only. Shared live-edge worlds make strategy comparisons paired.</p>`;
  } else if (view === "links") {
    const best = [...data.linkComparison].sort(
        (a, b) => b.aucMean - a.aucMean,
      )[0],
      ml = data.linkComparison.find((r) => r.method === "Gradient Boosting");
    content = `<h2>Does a learned model recover more missing links?</h2><p>Remove edges, rebuild every graph feature, and score balanced held-out edges and true non-edges. Repeat across ${data.linkRuns.length} seeded splits on this graph.</p><div class="note">${best.method} leads at mean ROC-AUC ${fmt(best.aucMean)}. Gradient Boosting scores ${fmt(ml.aucMean)}${ml.aucMean < best.aucMean ? " — additional model complexity did not improve the mean result." : "."} Variation between splits matters more than any single run.</div>${table(
      [
        "Method",
        "Mean ROC-AUC ↑",
        "SD",
        "Average precision ↑",
        "ROC-AUC across splits",
      ],
      data.linkComparison.map((r) => [
        r.method,
        fmt(r.aucMean),
        fmt(r.aucStd),
        fmt(r.apMean),
        `<svg class="score-range" viewBox="0 0 210 35" role="img" aria-label="AUC ranges from ${fmt(Math.min(...r.aucValues))} to ${fmt(Math.max(...r.aucValues))}"><line x1="5" y1="18" x2="205" y2="18" stroke="#bdcab8"/><line x1="105" y1="6" x2="105" y2="30" stroke="#687b65" stroke-dasharray="2 2"/>${r.aucValues.map((x, i) => `<circle cx="${5 + x * 200}" cy="${10 + (i % 3) * 7}" r="3" fill="#315b41"/>`).join("")}</svg>`,
      ]),
    )}<p class="muted">Dot plots run from 0 to 1; dashed line marks AUC 0.5. Each dot is an outer split, not an independent dataset. Splits preserve connectivity, so bridge edges are never held out. Balanced negatives simplify the task; precision does not represent the natural rate of missing links.</p><details><summary>Inspect the exact split sizes</summary>${table(
      [
        "Split seed",
        "Training edges",
        "Positive test edges",
        "Negative test pairs",
      ],
      data.linkRuns.map((r) => [
        r.seed,
        r.trainEdges,
        r.positiveEdges.length,
        r.negativeEdges.length,
      ]),
    )}</details>`;
  } else {
    content = `<section class="method"><h2>A saved experiment you can reproduce</h2><p>This page is a static explorer. Python generates the graph, partitions, centrality scores, propagation trials and link-recovery metrics together. Changing controls selects a saved experiment; it does not train a model in your browser.</p><h3>Graphs and reference labels</h3><p>Karate Club uses Zachary's observed network and subsequent two-group split. The stochastic block models plant four groups of 20 nodes. Strong separation uses p(within)=0.30 and p(between)=0.015; weak separation uses 0.18 and 0.08. All graphs are unweighted, undirected and generated/layout-seeded with 42.</p><h3>Community detection</h3><p>Louvain optimizes modularity. Label propagation follows neighbours' labels. Girvan–Newman removes high-betweenness edges and stops at the known group count. Spectral clustering uses Laplacian eigenvectors with the known group count. Agreement with labels and internal density are distinct criteria.</p><h3>Link recovery</h3><p>Twenty outer splits hold out about 15% of edges while preserving connectivity. Test negatives are true non-edges, disjoint from training negatives. All topology features are rebuilt on the remaining graph. When extracting a positive training pair, that pair's own edge is temporarily removed. Endpoint features use min/max to respect an undirected graph. Fixed Gradient Boosting settings: 200 estimators, depth 3, learning rate 0.05. No row-wise cross-validation over cached graph features is claimed.</p><h3>Influence</h3><p>Independent Cascade uses common directed live-edge worlds across strategies. One fixed set of randomly selected seeds is included as a baseline. One hundred seeded propagation trials per configuration estimate variability, not real social contagion.</p><h3>Reproduce</h3><code>python main.py --export-dashboard data/experiments.json<br>python -m pytest<br>python -m http.server 8080</code><p>Download the bundle for graph coordinates, each partition, exact held-out pairs, per-split scores, every trial's reach and a deterministic replay.</p><h3>Limitations</h3><p>These are three small graphs. Repeated splits share the same underlying topology and are not independent evidence across populations. Connectivity-preserving removal biases the missing-link task. Synthetic groups are planted by construction. A method's performance here cannot establish a general winner.</p><p><a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html">Karate Club source and citation</a> · <a href="https://github.com/m21264551-arch/network-analysis">Full source and setup</a></p></section>`;
  }
  $("#workspace").innerHTML = content;
  document.querySelectorAll("[data-node]").forEach((el) => {
    const choose = () => {
      selected = Number(el.dataset.node);
      render();
      document.querySelector(`[data-node="${selected}"]`)?.focus();
    };
    el.onclick = choose;
    el.onkeydown = (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        choose();
      }
    };
  });
  for (const [id, update] of Object.entries({
    method: (v) => (method = v),
    size: (v) => (size = v),
    count: (v) => {
      count = +v;
      frame = 0;
    },
    probability: (v) => {
      probability = +v;
      frame = 0;
    },
    strategy: (v) => {
      strategy = v;
      frame = 0;
    },
  })) {
    const el = $(`#${id}`);
    if (el)
      el.onchange = () => {
        update(el.value);
        render();
        $(`#${id}`).focus();
      };
  }
  if ($("#step"))
    $("#step").onclick = () => {
      frame++;
      render();
      $("#step").focus();
    };
  if ($("#restart"))
    $("#restart").onclick = () => {
      frame = 0;
      render();
      $("#restart").focus();
    };
}
async function load() {
  try {
    const response = await fetch("data/experiments.json");
    if (!response.ok) throw Error(`HTTP ${response.status}`);
    bundle = validate(await response.json());
    data = bundle.experiments[0];
    $("#dataset").innerHTML = bundle.experiments
      .map((e) => `<option value="${e.id}">${e.name}</option>`)
      .join("");
    document.querySelectorAll("[data-view]").forEach((button) => { button.disabled = false; });
    $("#dataset").disabled = false;
    $("#download").disabled = false;
    $("#status").textContent = "";
    $("#workspace").hidden = false;
    $("#provenance").textContent =
      `Saved experiment · generated ${bundle.generatedAt.slice(0, 10)} · code ${bundle.codeRevision.slice(0, 8)} · source SHA256 ${bundle.sourceChecksum}`;
    render();
  } catch (error) {
    $("#status").innerHTML =
      `<p>Experiment unavailable. The data could not be loaded or validated.</p><button id="retry">Try again</button> <a href="https://github.com/m21264551-arch/network-analysis">Open source and reproduction instructions</a>`;
    $("#retry").onclick = load;
    console.error(error);
  }
}
$("#dataset").onchange = () => {
  data = bundle.experiments.find((e) => e.id === $("#dataset").value);
  selected = 0;
  frame = 0;
  render();
};
document.querySelectorAll("[data-view]").forEach(
  (b) =>
    (b.onclick = () => {
      view = b.dataset.view;
      render();
    }),
);
$("#download").onclick = () => {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify({ ...bundle, experiments: [data] }, null, 2)], {
      type: "application/json",
    }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = `network-${data.id}-v1.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
load();
