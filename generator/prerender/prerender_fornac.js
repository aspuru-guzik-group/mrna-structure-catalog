#!/usr/bin/env node
/*
 * Headless fornac pre-render (issue #57).
 *
 * Reads a JSON array of {structure, sequence} from argv[2], renders each with fornac
 * (force + pan/zoom OFF) in headless Chromium, and writes a JSON array of the resulting
 * static <svg> strings to argv[3]. The catalog generator embeds these so the published
 * HTML has fixed positions and needs no client-side layout.
 *
 * Rendered in batches on one page (built, briefly settled, captured, cleared) so memory
 * stays bounded for thousands of entries.
 */
const fs = require("fs");
const puppeteer = require("puppeteer");

const FORNAC_URL = process.env.FORNAC_URL || "https://unpkg.com/fornac@1.1.8/dist/scripts/fornac.js";
// With force on we run the simulation to settle then bake the relaxed coords (good for
// long, cramped structures): smaller batches + a longer settle so the force can cool.
const FORCE = process.env.PRERENDER_FORCE === "1";
const BATCH = 100; // static path only
// Per-restart settle (force) / fit-apply wait (static).
const SETTLE_MS = FORCE ? Number(process.env.PRERENDER_SETTLE_MS || 3000) : 250;
// Adaptive multi-restart: try up to this many randomized layouts per structure, keeping
// the least-crossing one and stopping early at 0 crossings.
const MAX_RESTARTS = FORCE ? Number(process.env.PRERENDER_RESTARTS || 10) : 1;
// Force layout is relaxed on a LARGE canvas so d3's center-gravity doesn't crowd nodes
// (forna uses a big canvas); the SVG is then scaled into the 320x280 card via viewBox.
// 800x700 keeps the card's 8:7 aspect ratio. Static (force-off) renders at card size.
const [SVG_W, SVG_H] = FORCE
  ? (process.env.PRERENDER_FORCE_SIZE || "800x700").split("x").map(Number)
  : [320, 280];
// fornac hardcodes node charge to -30; a mild boost spreads helices apart (combined with
// multi-restart this is robust). Override/disable via env (0 = fornac's native -30).
const CHARGE = Number(process.env.PRERENDER_CHARGE != null ? process.env.PRERENDER_CHARGE : -50);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Prefer an explicit Chrome (env), then a system Chrome (the puppeteer-bundled
// "Chrome for Testing" build fails to spawn on some macOS setups), else let
// puppeteer use its own download.
function resolveExecutablePath() {
  const fromEnv = process.env.PUPPETEER_EXECUTABLE_PATH || process.env.CHROME_PATH;
  if (fromEnv) return fromEnv;
  const candidates = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
  ];
  return candidates.find((p) => fs.existsSync(p)) || undefined;
}

async function main() {
  const [, , inPath, outPath] = process.argv;
  if (!inPath || !outPath) {
    console.error("usage: prerender_fornac.js <in.json> <out.json>");
    process.exit(2);
  }
  const entries = JSON.parse(fs.readFileSync(inPath, "utf8"));

  const browser = await puppeteer.launch({
    headless: true,
    executablePath: resolveExecutablePath(),
    args: ["--no-sandbox"],
  });
  try {
    const page = await browser.newPage();
    await page.setContent('<!DOCTYPE html><html><head></head><body><div id="stage"></div></body></html>');
    await page.addScriptTag({ url: FORNAC_URL });
    await page.waitForFunction('typeof fornac !== "undefined"', { timeout: 30000 });

    // Browser-side helpers for the force/multi-restart path.
    await page.evaluate(() => {
      // Build one container; on restarts (>0) jitter initial node positions so the force
      // sim explores a different local minimum.
      window._buildOne = function (id, rna, w, h, charge, jitter) {
        const div = document.createElement("div");
        div.id = id;
        div.style.width = w + "px";
        div.style.height = h + "px";
        document.getElementById("stage").appendChild(div);
        const c = new fornac.FornaContainer("#" + id, {
          applyForce: true, allowPanningAndZooming: false, initialSize: [w, h], transitionDuration: 0,
        });
        c.addRNA(rna.structure, { sequence: rna.sequence });
        if (charge) c.force.charge(charge);
        if (jitter) {
          c.graph.nodes.forEach((n) => {
            n.x += (Math.random() - 0.5) * w;
            n.y += (Math.random() - 0.5) * h;
            n.px = n.x; n.py = n.y;
          });
        }
        c.force.start();
        window._c = c; window._cid = id;
      };
      // Freeze + re-fit, count strand crossings (backbone+basepair links between nucleotide
      // nodes, excluding shared endpoints), return {svg, cross}, and clear the stage.
      window._captureOne = function () {
        const c = window._c;
        if (c.stopAnimation) c.stopAnimation();
        if (c.setSize) c.setSize();
        // Only the DRAWN strands — fornac also has invisible "fake"/"label" layout links.
        const links = c.graph.links.filter(
          (l) => l.linkType === "backbone" || l.linkType === "basepair");
        const ccw = (p, q, r) => (r.y - p.y) * (q.x - p.x) > (q.y - p.y) * (r.x - p.x);
        const inter = (a, b, d, e) =>
          ccw(a, d, e) !== ccw(b, d, e) && ccw(a, b, d) !== ccw(a, b, e);
        let cross = 0;
        for (let i = 0; i < links.length; i++) {
          for (let j = i + 1; j < links.length; j++) {
            const p = links[i], q = links[j];
            if (p.source === q.source || p.source === q.target ||
                p.target === q.source || p.target === q.target) continue;
            if (inter(p.source, p.target, q.source, q.target)) cross++;
          }
        }
        const el = document.querySelector("#" + window._cid + " svg");
        const svg = el ? el.outerHTML : null;
        document.getElementById("stage").innerHTML = "";
        window._c = null;
        return { svg, cross };
      };
    });

    const svgs = new Array(entries.length).fill(null);
    let css = "";

    if (FORCE) {
      // Adaptive multi-restart: keep the least-crossing layout per structure; stop early at 0.
      for (let i = 0; i < entries.length; i++) {
        let best = null, bestCross = Infinity, used = 0;
        for (let r = 0; r < MAX_RESTARTS; r++) {
          used = r + 1;
          await page.evaluate(
            (id, rna, w, h, charge, jitter) => window._buildOne(id, rna, w, h, charge, jitter),
            "pr_" + i, entries[i], SVG_W, SVG_H, CHARGE, r > 0);
          await sleep(SETTLE_MS);
          const { svg, cross } = await page.evaluate(() => window._captureOne());
          if (cross < bestCross) { bestCross = cross; best = svg; }
          if (!css && svg) {
            css = await page.evaluate(() => {
              const el = document.querySelector("head style, body > style, style");
              return el ? el.textContent : "";
            });
          }
          if (cross === 0) break; // clean — done
        }
        svgs[i] = best;
        process.stderr.write(`[prerender] ${i + 1}/${entries.length} crossings=${bestCross} (tries=${used})\n`);
      }
    } else {
      // Fast static path: batched, no force, single pass.
      for (let start = 0; start < entries.length; start += BATCH) {
        const batch = entries.slice(start, start + BATCH);
        await page.evaluate((batch, start, w, h) => {
          const stage = document.getElementById("stage");
          batch.forEach((rna, k) => {
            const div = document.createElement("div");
            div.id = "pr_" + (start + k);
            div.style.width = w + "px";
            div.style.height = h + "px";
            stage.appendChild(div);
            const c = new fornac.FornaContainer("#" + div.id, {
              applyForce: false, allowPanningAndZooming: false, initialSize: [w, h], transitionDuration: 0,
            });
            c.addRNA(rna.structure, { sequence: rna.sequence });
            c.setSize();
          });
        }, batch, start, SVG_W, SVG_H);

        await sleep(SETTLE_MS);

        const captured = await page.evaluate((batch, start) => {
          const out = [];
          for (let k = 0; k < batch.length; k++) {
            const svg = document.querySelector("#pr_" + (start + k) + " svg");
            out.push(svg ? svg.outerHTML : null);
          }
          document.getElementById("stage").innerHTML = "";
          return out;
        }, batch, start);

        captured.forEach((s, k) => (svgs[start + k] = s));
        if (!css) {
          css = await page.evaluate(() => {
            const el = document.querySelector("head style, body > style, style");
            return el ? el.textContent : "";
          });
        }
        process.stderr.write(`[prerender] ${Math.min(start + BATCH, entries.length)}/${entries.length}\n`);
      }
    }

    fs.writeFileSync(outPath, JSON.stringify({ css, svgs }));
  } finally {
    await browser.close();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
