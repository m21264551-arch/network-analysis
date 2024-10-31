import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import fs from "node:fs";
const bundle = JSON.parse(fs.readFileSync("data/experiments.json", "utf8"));

test("visible counts, comparisons and download use the exported experiment", async ({
  page,
}) => {
  await page.goto("/");
  for (const experiment of bundle.experiments) {
    await page
      .getByLabel("Dataset", { exact: true })
      .selectOption(experiment.id);
    await expect(page.locator("#counts")).toHaveText(
      `${experiment.nodes.length} nodes · ${experiment.edges.length} edges`,
    );
    await page.getByRole("button", { name: "Link recovery" }).click();
    for (const row of experiment.linkComparison) {
      await expect(
        page.getByRole("row").filter({
          has: page.getByRole("cell", { name: row.method, exact: true }),
        }),
      ).toContainText(row.aucMean.toFixed(3));
    }
  }
  const waiting = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download data" }).click();
  const d = await waiting;
  const exported = JSON.parse(fs.readFileSync(await d.path(), "utf8"));
  expect(exported.experiments[0]).toEqual(bundle.experiments[2]);
});
test("node keyboard inspection and contextual influence controls", async ({
  page,
}) => {
  await page.goto("/");
  const node = page.getByRole("button", {
    name: "Inspect node 0,",
    exact: false,
  });
  await node.focus();
  await node.press("Enter");
  await expect(
    page.getByRole("heading", { name: "Node 0", exact: true }),
  ).toBeVisible();
  await page.getByRole("combobox", { name: "Select node", exact: true }).selectOption("33");
  await expect(
    page.getByRole("heading", { name: "Node 33", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Influence" }).click();
  await expect(page.getByLabel("Community method")).toHaveCount(0);
  await page.getByLabel("Propagation probability").selectOption("0.2");
  await page.getByRole("button", { name: "Next wave" }).click();
  await expect(page.getByText(/Wave 1/)).toBeVisible();
  await page.getByRole("button", { name: "Restart replay" }).click();
  await expect(page.getByText(/Wave 0/)).toBeVisible();
});
test("missing artifact offers recovery without invented results", async ({
  page,
}) => {
  await page.route("**/data/experiments.json", (r) => r.abort());
  await page.goto("/");
  await expect(page.getByRole("button", { name: "Try again" })).toBeVisible();
  await expect(page.locator("#workspace")).toBeHidden();
  await page.getByRole("link", {name:"How to use",exact:true}).click();
  await expect(page.getByRole("heading", {name:"How to use Network Explorer",exact:true})).toBeVisible();
});
for (const width of [1440, 390])
  test(`accessible main journey at ${width}px`, async ({ page }) => {
    await page.setViewportSize({ width, height: 900 });
    await page.goto("/");
    await page.getByRole("heading", { name: "Node 0", exact: true }).waitFor();
    expect((await new AxeBuilder({ page }).analyze()).violations).toEqual([]);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth,
      ),
    ).toBe(true);
  });

test("experiment controls wait for the artifact before accepting interaction", async ({
  page,
}) => {
  let release;
  const gate = new Promise((resolve) => {
    release = resolve;
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.route("**/data/experiments.json", async (route) => {
    await gate;
    await route.continue();
  });
  await page.goto("/");
  const links = page.getByRole("button", { name: "Link recovery" });
  await expect(links).toBeDisabled();
  await links.evaluate((button) => button.click());
  release();
  await expect(links).toBeEnabled();
  await links.click();
  await expect(
    page.getByRole("cell", { name: "0.768", exact: true }),
  ).toBeVisible();
  expect(errors).toEqual([]);
});

test("guide navigation keeps the current selection", async ({page})=>{
  await page.goto("/?source=portfolio");
  await page.getByRole("combobox",{name:"Select node",exact:true}).selectOption("33");
  await page.getByRole("link",{name:"How to use",exact:true}).click();
  await expect(page.getByRole("heading",{name:"How to use Network Explorer",exact:true})).toBeVisible();
  await page.getByRole("link",{name:"Back to tool ↑"}).click();
  expect(new URL(page.url()).search).toBe("?source=portfolio");
  await expect(page.getByRole("combobox",{name:"Select node",exact:true})).toHaveValue("33");
});

test("node selection highlights neighbours without moving the graph", async ({page}) => {
  await page.goto("/");
  await page.getByRole("heading", {name:"Node 0",exact:true}).waitFor();
  const positions = () => page.locator(".node circle:first-child").evaluateAll((nodes)=>nodes.map((n)=>[n.getAttribute("cx"),n.getAttribute("cy")]));
  const before=await positions();
  await page.getByRole("combobox",{name:"Select node",exact:true}).selectOption("33");
  expect(await positions()).toEqual(before);
  const adjacent = bundle.experiments[0].edges.filter((e)=>e.source===33||e.target===33);
  await expect(page.locator('[data-incident="true"]')).toHaveCount(adjacent.length);
  await expect(page.locator('.node[data-related="true"]')).toHaveCount(adjacent.length+1);
  await page.getByLabel("Community method").selectOption("Girvan-Newman");
  await page.getByRole("button", {name:"Reference groups",exact:true}).click();
  await expect(page.getByLabel("Community method")).toBeDisabled();
  expect(await positions()).toEqual(before);
  await page.getByRole("button", {name:"Reference groups",exact:true}).click();
  await expect(page.getByLabel("Community method")).toHaveValue("Girvan-Newman");
});

test("link illustration removes exactly the saved held-out edges", async ({page}) => {
  await page.goto("/"); await page.getByRole("button", {name:"Link recovery",exact:true}).click();
  const exp=bundle.experiments[0]; const split=exp.linkRuns[0];
  await expect(page.locator('[data-kind="observed"]')).toHaveCount(split.trainEdges);
  for(const [a,b] of split.positiveEdges) await expect(page.locator(`[data-kind="observed"][data-edge="${[a,b].sort((a,b)=>a-b).join(":")}"]`)).toHaveCount(0);
  await page.getByRole("button",{name:"Show withheld links"}).click();
  await expect(page.locator('[data-kind="withheld"]')).toHaveCount(split.positiveEdges.length);
  await expect(page.locator('[data-kind="negative"]')).toHaveCount(1);
  await page.getByRole("combobox",{name:"Split",exact:true}).selectOption("1");
  await expect(page.locator('[data-kind="withheld"]')).toHaveCount(0);
  await expect(page.locator('[data-kind="observed"]')).toHaveCount(exp.linkRuns[1].trainEdges);
});

for(const [width,height] of [[1280,720],[800,900],[390,844]]) test(`graph fits at ${width}px`,async({page})=>{
 await page.setViewportSize({width,height}); await page.goto("/");
 await page.getByRole("heading",{name:"Node 0",exact:true}).waitFor();
 expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBe(width);
 const box=await page.locator('.plot svg').boundingBox(); expect(box.y).toBeLessThan(height-200);
 await page.screenshot({path:`../../outputs/distinct-tested-network-${width}.png`});
});

test('partition comparison synchronizes coordinates, pan, selection and keyboard focus', async ({page}) => {
  await page.setViewportSize({width:1280,height:720});
  await page.goto('/?source=portfolio');
  await page.getByRole('button',{name:'Compare partitions',exact:true}).click();
  const primary=page.locator('.graph-canvas[data-pane="primary"]');
  const other=page.locator('.graph-canvas[data-pane="comparison"]');
  const positions=svg=>svg.locator('.node circle:first-child').evaluateAll(ns=>ns.map(n=>[n.getAttribute('cx'),n.getAttribute('cy')]));
  expect(await positions(primary)).toEqual(await positions(other));
  const before=await positions(primary);
  await page.getByRole('button',{name:'Zoom in',exact:true}).click();
  await primary.focus();await primary.press('ArrowRight');
  const viewport=await primary.getAttribute('data-viewport');
  await expect(other).toHaveAttribute('data-viewport',viewport);
  await other.locator('[data-node="33"]').focus();await page.keyboard.press('Enter');
  await expect(other.locator('[data-node="33"]')).toBeFocused();
  await expect(page.getByRole('combobox',{name:'Select node',exact:true})).toHaveValue('33');
  await page.getByRole('combobox',{name:'Compare with',exact:true}).selectOption('Girvan-Newman');
  expect(await positions(primary)).toEqual(before);
  expect(await positions(other)).toEqual(before);
  await expect(primary).toHaveAttribute('data-viewport',viewport);
  await other.focus();await page.setViewportSize({width:390,height:844});
  await expect(other).toBeFocused();
  await expect(other).toHaveAttribute('data-viewport',viewport);
  // Read both panes in one browser turn: resize observers may render between
  // separate locator evaluations, even though each render updates both panes.
  const panes=await page.locator('.graph-canvas').evaluateAll(svgs=>svgs.map(svg=>[...svg.querySelectorAll('.node circle:first-child')].map(n=>[n.getAttribute('cx'),n.getAttribute('cy')])));
  expect(panes[0]).toEqual(panes[1]);
  await other.press('Home');
  await expect(primary).toHaveAttribute('data-viewport',JSON.stringify({zoom:1,x:0,y:0}));
  const waiting=page.waitForEvent('download');await page.getByRole('button',{name:'Download data'}).click();
  const d=await waiting;expect(JSON.parse(fs.readFileSync(await d.path(),'utf8')).experiments[0]).toEqual(bundle.experiments[0]);
});


test('one graph Tab stop supports directional inspection and explicit partition differences',async({page})=>{
 await page.goto('/');await page.locator('.node[tabindex="0"]').waitFor();
 await expect(page.locator('.node[tabindex="0"]')).toHaveCount(1);
 await page.locator('.node[tabindex="0"]').focus();await page.keyboard.press('ArrowRight');
 await expect(page.getByRole('combobox',{name:'Select node',exact:true})).not.toHaveValue('0');
 await expect(page.locator('.node[tabindex="0"]')).toBeFocused();
 const selected=await page.getByRole('combobox',{name:'Select node',exact:true}).inputValue();
 await page.getByRole('button',{name:'Compare partitions',exact:true}).click();
 await expect(page.locator('.node[tabindex="0"]')).toHaveCount(2);
 await page.getByRole('button',{name:'Highlight differences',exact:true}).click();
 const primary=page.locator('.graph-canvas[data-pane="primary"]');
 const expected=await primary.locator('[data-disagreement="true"]').count();
 await expect(page.locator('.comparison-reading')).toContainText(`${expected} nodes differ`);
 await expect(page.getByRole('combobox',{name:'Select node',exact:true})).toHaveValue(selected);
 await page.getByRole('button',{name:'Node details'}).click();
 await expect(page.getByRole('heading',{name:`Node ${selected}`,exact:true})).toBeFocused();
 await expect(page.locator('.group-legend').first()).toContainText('Group 1');
});

test('desktop influence action stays beside its visible graph',async({page})=>{
 await page.setViewportSize({width:1280,height:720});await page.goto('/');
 await page.getByRole('button',{name:'Influence',exact:true}).click();
 await expect(page.getByRole('button',{name:'Next wave'})).toBeInViewport();
});


test('keyboard selection pans only enough to keep a zoomed node visible',async({page})=>{
 await page.goto('/');const canvas=page.locator('.graph-canvas');await canvas.waitFor();
 for(let i=0;i<4;i++) await page.getByRole('button',{name:'Zoom in',exact:true}).click();
 await page.getByRole('combobox',{name:'Select node',exact:true}).selectOption('14');
 const box=await canvas.boundingBox(),marker=await page.locator('[data-node="14"] .node-body').boundingBox();
 expect(marker.x).toBeGreaterThanOrEqual(box.x);expect(marker.x+marker.width).toBeLessThanOrEqual(box.x+box.width);
 expect(marker.y).toBeGreaterThanOrEqual(box.y);expect(marker.y+marker.height).toBeLessThanOrEqual(box.y+box.height);
});
