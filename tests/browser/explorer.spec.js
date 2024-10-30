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
  await expect(page.getByLabel("Display groups")).toHaveCount(0);
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
