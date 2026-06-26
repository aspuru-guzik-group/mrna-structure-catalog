#!/usr/bin/env node
/*
 * Option A — forna-exact pre-render.
 *
 * Reads JSON [{seq, struct, coords:[[x,y],...]}] from argv[2] (coords = ViennaRNA NAView,
 * computed in Python, identical to forna's /struct_positions server output), runs forna's
 * EXACT client pipeline (RNAGraph -> addPositions -> reinforce -> addRNAJSON -> force) using
 * forna's OWN jquery+d3+fornac.js+fornac.css, and writes {css, svgs} to argv[3].
 *
 * One molecule at a time (full force relaxation per structure, like the website).
 */
const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer");

// forna's own jquery/d3/fornac.js/fornac.css, vendored locally so rendering is offline
// and reproducible (no dependency on the live rna.tbi.univie.ac.at site at generation time).
const VENDOR = path.join(__dirname, "forna_vendor");
const SETTLE_MS = Number(process.env.FORNA_SETTLE_MS || 3000);
const W = 760, H = 640;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function resolveChrome() {
  const p = process.env.PUPPETEER_EXECUTABLE_PATH || process.env.CHROME_PATH;
  if (p) return p;
  for (const c of ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                   "/Applications/Chromium.app/Contents/MacOS/Chromium"]) {
    if (fs.existsSync(c)) return c;
  }
  return undefined;
}

async function main() {
  const [, , inPath, outPath] = process.argv;
  const entries = JSON.parse(fs.readFileSync(inPath, "utf8"));
  const browser = await puppeteer.launch({ headless: true, executablePath: resolveChrome(), args: ["--no-sandbox"] });
  try {
    const page = await browser.newPage();
    await page.setContent(`<!DOCTYPE html><html><head></head><body><div id="s" style="width:${W}px;height:${H}px"></div></body></html>`);
    await page.addStyleTag({ path: path.join(VENDOR, "fornac.css") });
    await page.addScriptTag({ path: path.join(VENDOR, "jquery.js") });
    await page.addScriptTag({ path: path.join(VENDOR, "d3.js") });
    await page.addScriptTag({ path: path.join(VENDOR, "fornac.js") });
    await page.waitForFunction('typeof FornaContainer !== "undefined" && typeof RNAGraph !== "undefined"');

    const svgs = new Array(entries.length).fill(null);
    let css = "";
    for (let i = 0; i < entries.length; i++) {
      await page.evaluate((seq, db, coords, w, h) => {
        document.getElementById("s").innerHTML = "";
        const r = new RNAGraph(seq, db, "m");
        r.circularizeExternal = true;
        r.elementsToJson().addPositions("nucleotide", coords).addLabels(1)
          .reinforceStems().reinforceLoops().connectFakeNodes();
        window._c = new FornaContainer("#s", { initialSize: [w, h] });
        window._c.addRNAJSON(r, true);
        if (window._c.startAnimation) window._c.startAnimation();
      }, entries[i].seq, entries[i].struct, entries[i].coords, W, H);

      await sleep(SETTLE_MS);

      const r = await page.evaluate(() => {
        if (window._c.stopAnimation) window._c.stopAnimation();
        if (window._c.setSize) window._c.setSize();
        const el = document.querySelector("#s svg");
        return el ? el.outerHTML : null;
      });
      svgs[i] = r;
      if (!css) css = fs.existsSync("__never__") ? "" : await page.evaluate(() => {
        const st = document.querySelector("style"); return st ? st.textContent : "";
      });
      process.stderr.write(`[forna] ${i + 1}/${entries.length}\n`);
    }
    // include forna's own fornac.css so the captured SVGs render identically standalone
    const fornacCss = fs.readFileSync(path.join(VENDOR, "fornac.css"), "utf8");
    fs.writeFileSync(outPath, JSON.stringify({ css: fornacCss + "\n" + css, svgs }));
  } finally {
    await browser.close();
  }
}

main().catch((e) => { console.error(e); process.exit(1); });
