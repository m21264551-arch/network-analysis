import { alignPartitionColours } from "./graph-view.js";
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

const fmt = (n, d = 3) => (Number.isFinite(n) ? n.toFixed(d) : "Unavailable");
const pct = (n) => `${fmt(n * 100, 1)}%`;
let bundle,
  data,
  view = "groups",
  method = "Louvain",
  size = "pagerank",
  selected = 0,
  reference = false,
  highlight = false,
  splitIndex = 0,
  revealLinks = false,
  probability = 0.1,
  count = 3,
  strategy = "PageRank",
  frame = 0;
let viewOptions = false;
let compare = false, comparisonMethod = "Reference groups", viewport = { zoom: 1, x: 0, y: 0 };
let suppressClick = false, showDifferences = false, disagreements = new Set();
const rank = (node, metric) => 1 + data.nodes.filter(other => other[metric] > node[metric]).length;
function nodeReading() {
  const node = data.nodes.find(n => n.id === selected) || data.nodes[0];
  return `Node ${node.id}: ${node.degree_count} connections · betweenness rank ${rank(node,"betweenness")} of ${data.nodes.length}.`;
}
function pairExample(split) {
  const removed = new Set(split.positiveEdges.map(pair => [...pair].sort((a,b)=>a-b).join(":")));
  const neighbours = new Map(data.nodes.map(node => [node.id,new Set()]));
  data.edges.filter(edge => !removed.has([edge.source,edge.target].sort((a,b)=>a-b).join(":"))).forEach(edge => {
    neighbours.get(edge.source).add(edge.target); neighbours.get(edge.target).add(edge.source);
  });
  return `<details class="pair-explanation"><summary>Why might a method choose this pair?</summary><p>Two pairs from this split, using only the remaining training graph. Common neighbours counts shared friends. Preferential attachment multiplies the two connection counts. These scores have different scales.</p>${table(["Pair", "Shared neighbours", "Connections × connections", "Recorded link"], [split.positiveEdges[0],split.negativeEdges[0]].map(([a,b],i)=>{
    const first=neighbours.get(a),second=neighbours.get(b);
    return [`${a}–${b}`,[...first].filter(id=>second.has(id)).length,`${first.size} × ${second.size} = ${first.size*second.size}`,revealLinks ? i===0 ? "Withheld link" : "No recorded link" : "Reveal withheld links to check"];
  }))}<p>Illustrative heuristic scores for two saved test pairs. The aggregate results below use every test pair across twenty splits.</p></details>`;
}
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
function graph(partition, active = null, seeds = [], split = null, colourMap = null, pane = "primary") {
  const comparison = compare && view === "groups";
  const small = matchMedia("(max-width: 999px)").matches;
  const available = $("#workspace").clientWidth;
  const width = Math.max(280, comparison && !small ? (available - 16) / 2 : available - (small ? 0 : 272));
  const height = small ? 350 : Math.max(280, innerHeight - (comparison ? 440 : view === "influence" ? 380 : view === "links" ? 300 : 350)), padding = 32;
  const xs = data.nodes.map(n => n.x), ys = data.nodes.map(n => n.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const scale = Math.min((width-2*padding)/Math.max(.01,maxX-minX),(height-2*padding)/Math.max(.01,maxY-minY));
  const pos = n => ({x:width/2+(n.x-(minX+maxX)/2)*scale,y:height/2+(n.y-(minY+maxY)/2)*scale});
  const nodeMap = new Map(data.nodes.map((n) => [n.id, n]));
  const pairKey = (a, b) => [a,b].sort((x,y) => x-y).join(":");
  const withheld = new Set(split?.positiveEdges.map(([a,b]) => pairKey(a,b)) || []);
  const observed = data.edges.filter((e) => !withheld.has(pairKey(e.source,e.target)));
  const neighbours = new Set(observed.flatMap((e) => e.source === selected ? [e.target] : e.target === selected ? [e.source] : []));
  const emphasize = !active && !split && highlight;
  const line = (source, target, kind) => {
    const a = pos(nodeMap.get(source)), b = pos(nodeMap.get(target));
    const incident = source === selected || target === selected;
    const color = kind === "withheld" ? "#ad5b2a" : kind === "negative" ? "#506aad" : incident && emphasize ? "#245940" : "#bcc9c0";
    return `<line data-edge="${pairKey(source,target)}" data-kind="${kind}" data-incident="${incident}" x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="${color}" stroke-width="${kind !== "observed" || incident && emphasize ? 2.5 : 1.2}" opacity="${emphasize && !incident ? .3 : 1}" ${kind !== "observed" ? `stroke-dasharray="${kind === "withheld" ? "7 4" : "2 5"}"` : ""}/>`;
  };
  const edges = observed.map((e) => line(e.source,e.target,"observed")).join("") +
    (split && revealLinks ? split.positiveEdges.map(([a,b]) => line(a,b,"withheld")).join("") : "") +
    (split?.negativeEdges.length ? line(...split.negativeEdges[0], "negative") : "");
  const max = Math.max(...data.nodes.map((n) => n[size]));
  const labels = new Map(), occupied = [];
  for (const n of [...data.nodes].sort((a,b) => (a.id===selected ? -2 : neighbours.has(a.id) ? -1 : 0) - (b.id===selected ? -2 : neighbours.has(b.id) ? -1 : 0))) {
    if (viewport.zoom < 1.4 && n.id !== selected && !neighbours.has(n.id) && !seeds.includes(n.id)) { labels.set(n.id,null); continue; }
    const p = pos(n);
    const offsets = [[0,0],[0,-28],[0,28],[-28,0],[28,0],[0,-44],[0,44]];
    const offset = offsets.find(([dx,dy]) => {
      const x=p.x+dx,y=p.y+dy;
      return x>12 && x<width-12 && y>12 && y<height-12 &&
        !occupied.some(([a,b]) => Math.abs(x-a)<23 && Math.abs(y-b)<18) &&
        !data.nodes.some(other => other.id!==n.id && Math.hypot(x-pos(other).x,y-pos(other).y)<23);
    });
    const chosen = offset || (n.id === selected ? [0,-28] : null);
    labels.set(n.id,chosen); if(chosen) occupied.push([p.x+chosen[0],p.y+chosen[1]]);
  }
  const coloursFor = group => colors[(colourMap?.[group] ?? group) % colors.length];
  const nodes = data.nodes.map((n) => {
    const p = pos(n), group = partition[n.id] ?? n.reference, r = 5 + Math.sqrt(n[size] / max) * 6;
    const label = labels.get(n.id), [dx,dy] = label || [0,0];
    const colour = coloursFor(group);
    const moved = dx !== 0 || dy !== 0;
    const on = active === null || active.has(n.id), related = n.id === selected || neighbours.has(n.id);
    return `<g class="node" role="button" tabindex="${n.id === selected ? 0 : -1}" data-node="${n.id}" data-pane="${pane}" data-related="${related}" data-disagreement="${comparison && disagreements.has(n.id)}" aria-label="Inspect node ${n.id}, group ${group + 1}, ${n.degree_count} neighbours" aria-pressed="${n.id === selected}" opacity="${comparison && showDifferences && n.id !== selected && !disagreements.has(n.id) ? .25 : emphasize && !related ? .35 : 1}"><circle class="node-body" cx="${p.x}" cy="${p.y}" r="${r}" fill="${on ? colour : "#fff"}" stroke="${n.id === selected ? "#152a1c" : colour}" stroke-width="${n.id === selected ? 4 : 1.5}"/>${comparison && showDifferences && disagreements.has(n.id) ? `<circle cx="${p.x}" cy="${p.y}" r="${r+4}" fill="none" stroke="#172d20" stroke-width="2" stroke-dasharray="3 2" pointer-events="none"/>` : ""}<circle class="node-hit" cx="${p.x}" cy="${p.y}" r="${Math.max(12,r)}" fill="transparent" stroke="none"/>${label && moved ? `<line x1="${p.x}" y1="${p.y}" x2="${p.x+dx}" y2="${p.y+dy}" stroke="#586d60"/><rect x="${p.x+dx-11}" y="${p.y+dy-9}" width="22" height="18" fill="#fbfcfb"/>` : ""}${label ? `<text x="${p.x+dx}" y="${p.y+dy + 4}" text-anchor="middle" font-size="12" font-weight="700" fill="${on && !moved ? "#fff" : "#24362d"}">${n.id}</text>` : ""}${seeds.includes(n.id) ? `<circle cx="${p.x}" cy="${p.y}" r="${r + 5}" fill="none" stroke="#1d3927" stroke-dasharray="3 2"/>` : ""}</g>`;
  }).join("");
  const caption = split ? `${revealLinks ? "Dashed orange: withheld links." : "Withheld links are hidden."} Dotted blue: a pair with no recorded link.` : active ? "Filled nodes are active. Dashed rings identify seeds." : highlight ? `Node ${selected} and its neighbours are highlighted. Colour: community.` : "Select a node to highlight its connections. Colour: community.";
  const transform = `translate(${width/2+viewport.x*width} ${height/2+viewport.y*height}) scale(${viewport.zoom}) translate(${-width/2} ${-height/2})`;
  return `<figure class="plot">${view !== "links" ? `<p class="node-reading" aria-live="polite">${nodeReading()} <a href="#node-details">Details ↓</a></p>` : ""}<svg class="graph-canvas" style="height:${height}px" viewBox="0 0 ${width} ${height}" data-pane="${pane}" data-viewport="${JSON.stringify(viewport).replaceAll('"','&quot;')}" tabindex="-1" aria-label="${data.name} graph. Arrow keys move between nodes. Shift and arrow keys pan. Plus and minus zoom. Home fits the graph." role="group"><g class="graph-geometry" transform="${transform}">${edges}${nodes}</g></svg><figcaption>${caption}</figcaption>${view === "groups" ? groupLegend(partition,colourMap) : ""}</figure>`;

}
function groupLegend(partition, colourMap) {
  const groups=[...new Set(Object.values(partition))].sort((a,b)=>a-b);
  return `<div class="group-legend" aria-label="Community colour key">${groups.map(group=>`<span><i style="background:${colors[(colourMap?.[group] ?? group)%colors.length]}"></i>Group ${group+1} · ${Object.values(partition).filter(value=>value===group).length} nodes</span>`).join("")}</div>`;
}
function details(partition) {
  const n = data.nodes.find((n) => n.id === selected) ?? data.nodes[0];
  return `<h2 id="node-details" tabindex="-1">Node ${n.id}</h2><p>Reference group ${n.reference + 1} · ${view === "groups" ? `detected group ${(partition[n.id] ?? n.reference) + 1}` : view === "links" ? "metrics below describe the original graph" : "inspect an activated node or a seed"}</p><p class="node-meaning">${rank(n,"betweenness") === 1 ? "This node lies on the largest share of shortest paths in this graph." : `Betweenness rank ${rank(n,"betweenness")} of ${data.nodes.length}: a rank closer to 1 means more shortest paths pass through this node.`} PageRank rank ${rank(n,"pagerank")} of ${data.nodes.length}. These are structural measures, not observed influence.</p><dl><dt>Neighbours</dt><dd>${n.degree_count}</dd><dt>PageRank</dt><dd>${fmt(n.pagerank, 3)}</dd><dt>Betweenness</dt><dd>${fmt(n.betweenness, 3)}</dd><dt>Eigenvector</dt><dd>${fmt(n.eigenvector, 3)}</dd></dl><details><summary>Metric definitions</summary><p class="muted">Degree counts immediate neighbours. PageRank rewards links from well-connected nodes. Betweenness measures how often a node lies on shortest paths. Eigenvector centrality rewards connections to nodes that are themselves well connected. Ranks use competition ranking, so ties share a rank. </p></details>`;
}
function render() {
  if (!data) return;
  $("#counts").textContent =
    `${data.nodes.length} nodes · ${data.edges.length} edges`;
  $("#description").textContent =
    data.id === "karate"
      ? "34 club members and their friendships. Reference groups show the club’s observed split."
      : data.description;
  document
    .querySelectorAll("[data-view]")
    .forEach((b) => b.setAttribute("aria-pressed", b.dataset.view === view));
  const partition =
    reference
      ? Object.fromEntries(data.nodes.map((n) => [n.id, n.reference]))
      : data.communities.find((c) => c.method === method).partition;
  let content = "";
  if (view === "groups") {
    const best = [...data.communities].sort((a, b) => b.nmi - a.nmi)[0];
    const other = comparisonMethod === "Reference groups" ? Object.fromEntries(data.nodes.map(n=>[n.id,n.reference])) : data.communities.find(c=>c.method===comparisonMethod).partition;
    const colourMap = alignPartitionColours(partition, other);
    disagreements = new Set(data.nodes.filter(n => partition[n.id] !== colourMap[other[n.id]]).map(n=>n.id));
    const controls = `<div class="controls"><label>Community method<select id="method" ${reference ? "disabled" : ""}>${options(data.communities.map((c) => c.method), method)}</select></label><button id="reference" aria-pressed="${reference}">Reference groups</button></div>`;
    content = `${controls}${compare ? `<div class="comparison-reading"><span>${new Set(Object.values(partition)).size} groups vs ${new Set(Object.values(other)).size} · ${disagreements.size} nodes change groups. Corresponding colours mark groups with the most shared members.</span><button id="differences" aria-pressed="${showDifferences}">Highlight differences</button></div><div class="disagreement-nodes" aria-label="Nodes with different assignments">${[...disagreements].map(id=>`<button data-difference="${id}">Node ${id}</button>`).join("")}</div>` : ""}${compare ? `<div class="partition-panes"><section><h2>${reference ? "Reference groups" : method}</h2>${graph(partition)}</section><section><label>Compare with<select id="compare-method">${options(["Reference groups",...data.communities.map(c=>c.method)],comparisonMethod)}</select></label>${graph(other,null,[],null,colourMap,"comparison")}</section></div><aside class="shared-inspector">${details(partition)}<p>Colours are aligned by shared members. Group labels and scores are unchanged.</p></aside>` : `<div class="experiment"><section>${graph(partition)}${highlight ? `<button class="clear-selection" id="show-all">Show all connections</button>` : ""}</section><aside>${details(partition)}</aside></div>`}<h3>Partition comparison</h3><p class="observation">${best.method} has the highest agreement with reference groups: NMI ${fmt(best.nmi)}. Agreement and modularity measure different things.</p>${table(
      ["Method", "Groups", "Modularity", "NMI ↑", "Adjusted Rand ↑"],
      data.communities.map((c) => [
        `<span ${c.method === method && !reference ? 'class="chosen-method"' : ""}>${c.method}${c.method === method && !reference ? " · selected" : ""}</span>`,
        c.num_communities,
        fmt(c.modularity),
        fmt(c.nmi),
        fmt(c.ari),
      ]),
    )}<details><summary>Reading these results</summary><p class="muted">Modularity measures dense internal connections relative to a null model. NMI and adjusted Rand compare against reference labels. Spectral clustering and Girvan–Newman are given the known number of groups. Louvain and label propagation choose their own partition, so the methods receive different information.</p></details>`;
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
    )}</select></label><label>Seed count<select id="count">${options([1, 3, 5], count)}</select></label><label>Propagation probability<select id="probability">${options([0.05, 0.1, 0.2], probability)}</select></label></div>${graph(Object.fromEntries(data.nodes.map((n) => [n.id, n.reference])), active, chosen.seeds)}<div class="controls"><button id="restart">Restart replay</button><button id="step" ${frame === chosen.replay.length - 1 ? "disabled" : ""}>Next wave →</button><strong>Wave ${frame} · ${active.size} active nodes</strong><span>Seeds: ${chosen.seeds.join(", ")} · one trial of ${chosen.reaches.length}</span></div></section><aside><h2>Simulated reach</h2><p>Each active node gets one chance to activate each neighbour.</p><p>Seeds: <strong>${chosen.seeds.join(", ")}</strong></p><div class="note">${same.length ? `${[strategy, ...same].join(", ")} choose the same ${chosen.seeds.length === 1 ? "seed" : "seeds"} here and have the same simulated reach.` : `${strategy} reaches ${pct(chosen.mean)} on average across ${chosen.reaches.length} paired trials. This replay is one trial. The bars below summarise all trials.`}</div>${details(partition)}</aside></div><h3>Reach across paired trials</h3><div class="bars">${rows.map((r) => `<div class="bar-row"><span>${r.strategy}</span><div class="track"><div class="fill" style="width:${r.mean * 100}%"></div><div class="spread" style="left:${Math.max(0, r.mean - r.std) * 100}%;width:${(Math.min(1, r.mean + r.std) - Math.max(0, r.mean - r.std)) * 100}%"></div></div><span>${pct(r.mean)}</span></div>`).join("")}</div>${table(
      ["Strategy", "Mean reach", "SD of reach", "Seeds"],
      rows.map((r) => [
        r.strategy,
        pct(r.mean),
        pct(r.std),
        r.seeds.join(", "),
      ]),
    )}<p class="muted">Dark bars show ±1 standard deviation, not confidence intervals. Random chooses one fixed seed set. Trials vary propagation only. Every strategy is tested with the same propagation draws.</p>`;
  } else if (view === "links") {
    const best = [...data.linkComparison].sort(
        (a, b) => b.aucMean - a.aucMean,
      )[0],
      ml = data.linkComparison.find((r) => r.method === "Gradient Boosting");
    const split = data.linkRuns[splitIndex];
    const negative = split.negativeEdges[0];
    content = `<div class="experiment link-example"><section>${graph(partition, null, [], split)}<div class="controls"><label>Split<select id="split">${data.linkRuns.map((r,i) => `<option value="${i}" ${i === splitIndex ? "selected" : ""}>${i+1} · seed ${r.seed}</option>`).join("")}</select></label><button id="reveal-links" aria-pressed="${revealLinks}">${revealLinks ? "Hide withheld links" : "Show withheld links"}</button></div></section><aside><h2>Which links are missing?</h2><p>${split.positiveEdges.length} recorded connections were removed. The remaining ${split.trainEdges} connections are the input to each method.</p><p>The dotted pair, <strong>${negative.join("–")}</strong>, has no recorded connection. It is one negative test example.</p><p>This picture explains one split. The results below summarize all ${data.linkRuns.length} splits.</p>${details(partition)}</aside></div>${pairExample(split)}<h2>Link recovery results</h2><p>Remove edges, rebuild every graph feature, and score balanced held-out edges and true non-edges. Repeat across ${data.linkRuns.length} seeded splits on this graph.</p><div class="note">${best.method} leads at mean ROC-AUC ${fmt(best.aucMean)}. Gradient Boosting scores ${fmt(ml.aucMean)}${ml.aucMean < best.aucMean ? " . The more complex model had a lower mean score." : "."}</div>${table(
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
    )}<p class="muted">Dot plots run from 0 to 1. The dashed line marks AUC 0.5. Each dot is an outer split, not an independent dataset. Splits preserve connectivity, so bridge edges are never held out. Balanced negatives simplify the task. Precision does not represent the natural rate of missing links.</p><details><summary>Inspect the exact split sizes</summary>${table(
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
  }
  $("#workspace").innerHTML = `<div class="viewport-controls"><label class="node-navigation"><span>Node</span><select id="node-picker" aria-label="Select node">${options(data.nodes.map(n=>n.id),selected)}</select></label><button id="inspect-selected">Node details ↓</button>${view === "groups" ? `<button id="compare" aria-pressed="${compare}">Compare partitions</button>` : ""}<details class="view-options" ${viewOptions ? "open" : ""}><summary>View</summary><div class="view-options-body"><div><button id="zoom-out" aria-label="Zoom out">−</button><button id="zoom-in" aria-label="Zoom in">+</button><button id="fit-graph">Fit graph</button></div><label>Size nodes by<select id="size">${[["pagerank","PageRank"],["degree","Connections"],["betweenness","Betweenness"]].map(([v,label]) => `<option value="${v}" ${size === v ? "selected" : ""}>${label}</option>`).join("")}</select></label></div></details></div>` + content;
  document.querySelectorAll("[data-difference]").forEach(button => { button.onclick=()=>{selected=+button.dataset.difference;highlight=true;render();document.querySelector(`[data-difference="${selected}"]`)?.focus({preventScroll:true});}; });
  $("#inspect-selected").onclick=()=>{ $("#node-details").focus(); $("#node-details").scrollIntoView({block:"nearest"}); };
  if($("#differences")) $("#differences").onclick=()=>{showDifferences=!showDifferences;render();$("#differences").focus({preventScroll:true});};
  document.querySelectorAll("[data-node]").forEach((el) => {
    const choose = () => {
      if(suppressClick) return;
      const pane = el.dataset.pane;
      selected = Number(el.dataset.node);
      highlight = true;
      render();
      document.querySelector(`[data-node="${selected}"][data-pane="${pane}"]`)?.focus({preventScroll:true});
    };
    el.onclick = choose;
    el.onkeydown = (e) => {
      if (!e.shiftKey && ["ArrowLeft","ArrowRight","ArrowUp","ArrowDown"].includes(e.key)) {
        e.preventDefault(); e.stopPropagation();
        const origin=data.nodes.find(n=>n.id===Number(el.dataset.node));
        const horizontal=["ArrowLeft","ArrowRight"].includes(e.key), sign=["ArrowLeft","ArrowUp"].includes(e.key)?-1:1;
        const candidates=data.nodes.filter(n=>n.id!==origin.id).map(n=>{
          const dx=n.x-origin.x,dy=n.y-origin.y,forward=(horizontal?dx:dy)*sign,cross=horizontal?dy:dx;
          return {n,forward,distance:Math.hypot(dx,dy)+Math.abs(cross)*2};
        }).filter(n=>n.forward>0).sort((a,b)=>a.distance-b.distance);
        if(candidates.length) {
          const pane=el.dataset.pane;
          selected=candidates[0].n.id;highlight=true;render();
          const target=document.querySelector(`[data-node="${selected}"][data-pane="${pane}"]`);
          keepNodeVisible(target);
          target?.focus({preventScroll:true});
        }
        return;
      }
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        choose();
      }
    };
  });
  $(".view-options").ontoggle = e => { viewOptions=e.currentTarget.open; };
  wireViewport();
  if($("#compare")) $("#compare").onclick = () => { compare=!compare; render(); $("#compare").focus({preventScroll:true}); };
  for (const [id, update] of Object.entries({
    "compare-method": (v) => { comparisonMethod=v; },
    split: (v) => { splitIndex = +v; revealLinks = false; },
    "node-picker": (v) => { selected = +v; highlight = true; },
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
        if (id === "node-picker") keepNodeVisible(document.querySelector(`.graph-canvas[data-pane="primary"] [data-node="${selected}"]`));
        $(`#${id}`).focus();
      };
  }
  if ($("#show-all")) $("#show-all").onclick = () => { highlight = false; render(); $("#node-picker").focus(); };
  if ($("#reference")) $("#reference").onclick = () => { reference = !reference; render(); $("#reference").focus(); };
  if ($("#reveal-links")) $("#reveal-links").onclick = () => { revealLinks = !revealLinks; render(); $("#reveal-links").focus(); };
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
async function load(retrying = false) {
  try {
    const response = await fetch("data/experiments.json");
    if (!response.ok) throw Error(`HTTP ${response.status}`);
    bundle = validate(await response.json());
    data = bundle.experiments[0];
    $("#dataset").innerHTML = bundle.experiments
      .map((e) => `<option value="${e.id}">${e.name}</option>`)
      .join("");
    document.querySelectorAll("[data-view]").forEach((button) => {
      button.disabled = false;
    });
    $("#dataset").disabled = false;
    $("#download").disabled = false;
    $("#status").textContent = "";
    $("#workspace").hidden = false;
    $("#provenance").textContent =
      `Saved experiments · ${bundle.generatedAt.slice(0, 10)}`;
    $("#method-content").innerHTML = `<section class="method"><h2>Data and method</h2><p>This page is a static explorer. Python generates the graph, partitions, centrality scores, propagation trials and link-recovery metrics together. Controls select saved results. The browser does not train models.</p><p class="identity">Generated ${bundle.generatedAt.slice(0, 10)} · code ${bundle.codeRevision.slice(0, 8)}<br>SHA256 ${bundle.sourceChecksum}</p><h3>Graphs and reference labels</h3><p>Karate Club uses Zachary's observed network and subsequent two-group split. The stochastic block models plant four groups of 20 nodes. Strong separation uses p(within)=0.30 and p(between)=0.015. Weak separation uses 0.18 and 0.08. All graphs are unweighted, undirected and generated/layout-seeded with 42.</p><h3>Community detection</h3><p>Louvain optimizes modularity. Label propagation follows neighbours' labels. Girvan–Newman removes high-betweenness edges and stops at the known group count. Spectral clustering uses Laplacian eigenvectors with the known group count. Agreement with labels and internal density are distinct criteria.</p><h3>Link recovery</h3><p>Twenty outer splits hold out about 15% of edges while preserving connectivity. Test negatives are true non-edges, disjoint from training negatives. All topology features are rebuilt on the remaining graph. When extracting a positive training pair, that pair's own edge is temporarily removed. Endpoint features use min/max to respect an undirected graph. Fixed Gradient Boosting settings: 200 estimators, depth 3, learning rate 0.05.</p><h3>Influence</h3><p>Independent Cascade uses common directed live-edge worlds across strategies. One fixed set of randomly selected seeds is included as a baseline. Each configuration uses 100 seeded propagation trials. These describe simulated spread, not observed social behaviour.</p><h3>Reproduce</h3><code>python main.py --export-dashboard data/experiments.json<br>python -m pytest<br>python -m http.server 8080</code><p>Download the bundle for graph coordinates, each partition, exact held-out pairs, per-split scores, every trial's reach and a deterministic replay.</p><h3>Limitations</h3><p>These are three small graphs. Repeated splits share the same underlying topology and are not independent evidence across populations. Connectivity-preserving removal biases the missing-link task. Synthetic groups are planted by construction. A method's performance here cannot establish a general winner.</p><p><a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html">Karate Club source and citation</a> · <a href="https://github.com/m21264551-arch/network-analysis">Full source and setup</a></p></section>`;
    $("#method-link").disabled = false;
    render();
    if (retrying) requestAnimationFrame(() => $("#node-picker")?.focus());
  } catch (error) {
    $("#status").innerHTML =
      `<p>Experiment unavailable. The data could not be loaded or validated.</p><button id="retry">Try again</button> <a href="https://github.com/m21264551-arch/network-analysis">Open source and reproduction instructions</a>`;
    $("#retry").onclick = () => load(true);
    console.error(error);
  }
}
$("#dataset").onchange = () => {
  data = bundle.experiments.find((e) => e.id === $("#dataset").value);
  selected = 0;
  viewport = {zoom:1,x:0,y:0};
  highlight = false;
  frame = 0;
  splitIndex = 0; revealLinks = false;
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
function renderPreservingFocus() {
  const active=document.activeElement;
  const focusId=active?.id, node=active?.dataset.node;
  const pane=active?.closest?.('.graph-canvas')?.dataset.pane;
  render();
  if(focusId) document.getElementById(focusId)?.focus({preventScroll:true});
  else if(pane) document.querySelector(`.graph-canvas[data-pane="${pane}"]${node ? ` [data-node="${node}"]` : ''}`)?.focus({preventScroll:true});
}
let previousWidth = 0;
new ResizeObserver(([entry]) => {
  const width=Math.round(entry.contentRect.width);
  if(width===previousWidth) return;
  previousWidth=width;
  renderPreservingFocus();
}).observe($("#workspace"));
window.addEventListener("resize", () => { if(data) renderPreservingFocus(); });
load();

function applyViewport() {
  document.querySelectorAll(".graph-canvas").forEach(svg => {
    const {width,height}=svg.viewBox.baseVal;
    svg.dataset.viewport=JSON.stringify(viewport);
    svg.querySelector(".graph-geometry").setAttribute("transform",`translate(${width/2+viewport.x*width} ${height/2+viewport.y*height}) scale(${viewport.zoom}) translate(${-width/2} ${-height/2})`);
  });
}
function keepNodeVisible(node) {
  if (!node) return;
  const canvas = node.closest(".graph-canvas").getBoundingClientRect();
  const marker = node.querySelector(".node-body").getBoundingClientRect();
  const shift = (start, end, low, high) => start < low + 20 ? low + 20 - start : end > high - 20 ? high - 20 - end : 0;
  viewport.x += shift(marker.left, marker.right, canvas.left, canvas.right) / canvas.width;
  viewport.y += shift(marker.top, marker.bottom, canvas.top, canvas.bottom) / canvas.height;
  applyViewport();
}
function zoom(factor) { viewport.zoom=Math.max(.5,Math.min(4,viewport.zoom*factor)); applyViewport(); }
function wireViewport() {
  $("#zoom-in").onclick=()=>zoom(1.25);
  $("#zoom-out").onclick=()=>zoom(.8);
  $("#fit-graph").onclick=()=>{viewport={zoom:1,x:0,y:0};applyViewport();};
  document.querySelectorAll(".graph-canvas").forEach(svg=>{
    let drag=null;
    svg.onkeydown=e=>{
      if(e.target!==svg && !e.shiftKey && !["Home","+","=","-"].includes(e.key)) return;
      if(!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown","+","=","-","Home"].includes(e.key)) return;
      e.preventDefault();
      if(e.key==="Home") viewport={zoom:1,x:0,y:0};
      else if(["+","="].includes(e.key)) zoom(1.25);
      else if(e.key==="-") zoom(.8);
      else { viewport.x+=e.key==="ArrowLeft"?.06:e.key==="ArrowRight"?-.06:0; viewport.y+=e.key==="ArrowUp"?.06:e.key==="ArrowDown"?-.06:0; }
      applyViewport();
    };
    svg.onpointerdown=e=>{
      if(e.button!==0 || e.target.closest(".node")) return;
      drag={id:e.pointerId,x:e.clientX,y:e.clientY,panX:viewport.x,panY:viewport.y,touch:e.pointerType==="touch",active:false};
      suppressClick=false;
    };
    svg.onpointermove=e=>{
      if(!drag || drag.id!==e.pointerId) return;
      const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
      if(!drag.active && Math.hypot(dx,dy)>8) {
        if(drag.touch && Math.abs(dy)>Math.abs(dx)) {drag=null;return;}
        drag.active=true;svg.setPointerCapture(e.pointerId);
      }
      if(!drag.active) return;
      const rect=svg.getBoundingClientRect();
      viewport.x=drag.panX+dx/rect.width;viewport.y=drag.panY+dy/rect.height;
      suppressClick=true;applyViewport();
    };
    const end=()=>{drag=null;setTimeout(()=>{suppressClick=false;},0);};
    svg.onpointerup=end;svg.onpointercancel=end;
  });
}
$("#method-link").onclick=()=>{ $("#method-details").open=true; $("#method-details").scrollIntoView({block:"start"}); $("#method-details summary").focus({preventScroll:true}); };
