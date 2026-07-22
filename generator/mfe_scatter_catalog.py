#!/usr/bin/env python3
"""MFE scatter-plot catalog (issue #57 follow-up).

A card-per-population page of Knotergy-MFE scatter plots comparing the three folds
(Knotty / ViennaRNA / DeRNA) pairwise. Each card switches between the three pairings
(Knotty–DeRNA, Knotty–ViennaRNA, ViennaRNA–DeRNA) and a Compare view (all three); a page
toolbar toggles pk-mixed (PK + non-PK, colour-coded) vs pk-only.

All MFEs are Knotergy (Dirks-Pierce09) energies — apple-to-apple:
  Knotty = e_pk, ViennaRNA = e_nested, DeRNA = compute_derna_knotergy.py output.

Populations mirror the structure catalogs' granularity:
  AA=20 (1000), AA=20 (100 subset), AA=60 (99), all pseudoknotted (AA 4–60).

Example
-------
    python3 scripts/eval/mfe_scatter_catalog.py \
        --parquet data/pk_profiling/aa60_n100_20260620/pk_profile.parquet \
        --derna-mfe-json all_derna_knotergy.json \
        --out docs/mfe_scatter.html
"""
from __future__ import annotations

import argparse
import html
import json
import os

# CVD-validated categorical pair (dataviz skill validator, light surface):
#   non-PK #2b6cb0 / PK #e8833a  — adjacent CVD ΔE 24.4 (protan). Legend supplies the
#   secondary (text) encoding the contrast WARN requires.
COLOR_NONPK = "#2b6cb0"
COLOR_PK = "#e8833a"


def _load(parquet: str, derna_mfe_json: str):
    import pandas as pd

    df = pd.read_parquet(parquet)
    df = df[df["ok"]] if "ok" in df.columns else df
    dm = json.load(open(derna_mfe_json))
    dm = dm.get("energies", dm)

    def derna_of(r):
        return dm.get(f"{int(r.aa_len)}_{int(r.idx)}", dm.get(str(int(r.idx))))

    def rows_for(sub):
        out = []
        for r in sub.itertuples():
            d = derna_of(r)
            if d is None:
                continue
            # [knotty, vienna, derna, pk, idx, aa_len] — compact; energies rounded to 2dp
            out.append([round(float(r.e_pk), 2), round(float(r.e_nested), 2),
                        round(float(d), 2), 1 if bool(r.has_pk) else 0,
                        int(r.idx), int(r.aa_len)])
        return out

    aa20 = df[df["aa_len"] == 20].sort_values("idx")
    aa60 = df[df["aa_len"] == 60].sort_values("idx")
    allpk = df[df["has_pk"]].sort_values(["aa_len", "idx"])
    # aa20_100 = the first 100 aa20 indices (matches the aa20_100 structure catalog: --indices 0:100)
    aa20_100 = aa20[aa20["idx"] < 100]

    pops = [
        {"id": "aa20_1000", "name": "AA = 20 — all 1000", "pts": rows_for(aa20)},
        {"id": "aa20_100", "name": "AA = 20 — 100 (subset)", "pts": rows_for(aa20_100)},
        {"id": "aa60_99", "name": "AA = 60 — 99", "pts": rows_for(aa60)},
        {"id": "all_pk", "name": "All pseudoknotted (AA 4–60)", "pts": rows_for(allpk)},
    ]
    for p in pops:
        n = len(p["pts"])
        npk = sum(1 for r in p["pts"] if r[3])
        p["n"], p["n_pk"], p["n_nonpk"] = n, npk, n - npk
        p["sub"] = f"{npk} PK / {n - npk} non-PK" if npk != n else f"{n} PK"
    return pops


_CSS = """\
:root { color-scheme: light; }
body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 24px; background: #fafafa; color: #222; }
h1 { font-size: 20px; font-weight: 600; margin: 0 0 4px; }
.note { color: #888; font-size: 12px; margin: 0 0 14px; max-width: 860px; line-height: 1.5; }
a.back { font-size: 13px; color: #0b5cab; text-decoration: none; }
.toolbar { margin: 0 0 16px; font-size: 13px; color: #444; display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
.toolbar .seg { display: inline-flex; gap: 4px; }
.tbtn { padding: 4px 12px; border: 1px solid #ccc; background: #fff; border-radius: 6px; cursor: pointer; font-size: 12px; }
.tbtn:hover { background: #f3f3f3; }
.tbtn.active { background: #d8ebff; border-color: #9cc8f0; color: #0b5cab; }
.legend { margin-left: auto; display: flex; gap: 14px; align-items: center; font-size: 12px; color: #555; }
.legend .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 16px; }
.card { position: relative; margin: 0; background: #fff; border: 1px solid #e3e3e3; border-radius: 8px;
        overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
.topcap { display: flex; gap: 4px; padding: 6px 8px; border-bottom: 1px solid #f0f0f0; background: #fafafa; flex-wrap: wrap; }
.mbtn { padding: 2px 8px; border: 1px solid #d3d3d3; background: #fff; border-radius: 6px; cursor: pointer;
        font-size: 11px; font-weight: 600; color: #555; }
.mbtn:hover { background: #f0f0f0; }
.mbtn.active { background: #d8ebff; border-color: #9cc8f0; color: #0b5cab; }
.mbtn.cmp { margin-left: auto; color: #5b3ea8; }
.plot { padding: 10px 12px 4px; }
.plot svg { display: block; width: 100%; height: auto; }
figcaption { padding: 6px 12px 12px; }
.title { display: block; font-weight: 600; font-size: 14px; }
.sub { display: block; color: #666; font-size: 12px; margin-top: 2px; }
.stat { display: block; color: #444; font-size: 12px; margin-top: 4px; font-family: ui-monospace, Menlo, monospace; }
/* Compare modal */
.cmp-modal { position: fixed; inset: 0; background: rgba(0,0,0,0.55); z-index: 100; display: none;
             align-items: center; justify-content: center; padding: 24px; }
.cmp-modal.open { display: flex; }
.cmp-box { background: #fff; border-radius: 10px; max-width: 1100px; width: 100%; max-height: 92vh; overflow: auto; padding: 18px 20px 22px; }
.cmp-box h2 { font-size: 16px; margin: 0 0 2px; }
.cmp-box .csub { color: #777; font-size: 12px; margin: 0 0 14px; }
.cmp-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.cmp-cell { border: 1px solid #eee; border-radius: 8px; padding: 8px; }
.cmp-cell h3 { font-size: 12px; margin: 0 0 4px; color: #444; }
.cmp-cell .stat { font-size: 11px; }
.cmp-close { float: right; border: 1px solid #ccc; background: #fff; border-radius: 6px; cursor: pointer; font-size: 13px; padding: 3px 10px; }
.tip { position: fixed; pointer-events: none; background: #222; color: #fff; font-size: 11px; padding: 4px 7px;
       border-radius: 4px; opacity: 0; transition: opacity .08s; z-index: 200; font-family: ui-monospace, Menlo, monospace; }
@media (max-width: 720px) { .cmp-grid { grid-template-columns: 1fr; } }
"""


# Client JS: draws every scatter as inline SVG, handles the per-card pair switch, the
# pk-mixed/pk-only toolbar, the Compare modal, hover tooltips, and a y=x reference line.
_JS = """\
const POPS = __POPS__;
const COL = {nonpk: "__CNONPK__", pk: "__CPK__"};
// comparison pairings: indices into the point tuple [knotty, vienna, derna, pk, idx, aa_len]
const CMP = {
  kd: {x: 0, y: 2, xl: "Knotty", yl: "DeRNA"},
  kv: {x: 0, y: 1, xl: "Knotty", yl: "ViennaRNA"},
  vd: {x: 1, y: 2, xl: "ViennaRNA", yl: "DeRNA"},
};
let pkOnly = false;
const tip = document.getElementById("tip");

function pearson(pts, xi, yi) {
  const n = pts.length; if (n < 2) return NaN;
  let sx=0, sy=0, sxx=0, syy=0, sxy=0;
  for (const p of pts) { const x=p[xi], y=p[yi]; sx+=x; sy+=y; sxx+=x*x; syy+=y*y; sxy+=x*y; }
  const cov = sxy - sx*sy/n, vx = sxx - sx*sx/n, vy = syy - sy*sy/n;
  return (vx>0 && vy>0) ? cov/Math.sqrt(vx*vy) : NaN;
}

// Build one scatter SVG (string). W,H in px; shared x/y scale so y=x is a true 45° line.
function scatterSVG(pop, cmpKey, w, h, small) {
  const c = CMP[cmpKey];
  let pts = pop.pts;
  if (pkOnly) pts = pts.filter(p => p[3] === 1);
  // shared range over BOTH axes' values (so the diagonal is meaningful)
  let lo = Infinity, hi = -Infinity;
  for (const p of pts) { lo = Math.min(lo, p[c.x], p[c.y]); hi = Math.max(hi, p[c.x], p[c.y]); }
  if (!isFinite(lo)) { lo = -1; hi = 0; }
  const pad = (hi - lo) * 0.06 || 1; lo -= pad; hi += pad;
  const m = small ? {l: 34, r: 8, t: 8, b: 26} : {l: 40, r: 10, t: 10, b: 30};
  const pw = w - m.l - m.r, ph = h - m.t - m.b;
  const sx = v => m.l + (v - lo) / (hi - lo) * pw;
  const sy = v => m.t + (1 - (v - lo) / (hi - lo)) * ph;
  const parts = [`<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg" font-family="-apple-system,Helvetica,Arial,sans-serif">`];
  // plot frame
  parts.push(`<rect x="${m.l}" y="${m.t}" width="${pw}" height="${ph}" fill="#fff" stroke="#e6e6e6"/>`);
  // y=x reference line (recessive dashed)
  const d0 = Math.max(lo, lo), d1 = Math.min(hi, hi);
  parts.push(`<line x1="${sx(lo).toFixed(1)}" y1="${sy(lo).toFixed(1)}" x2="${sx(hi).toFixed(1)}" y2="${sy(hi).toFixed(1)}" stroke="#bbb" stroke-width="1" stroke-dasharray="4 3"/>`);
  // ticks (lo / mid / hi)
  const fmt = v => v.toFixed(0);
  const mid = (lo + hi) / 2;
  const fs = small ? 9 : 10;
  for (const v of [lo + pad, mid, hi - pad]) {
    parts.push(`<text x="${sx(v).toFixed(1)}" y="${(m.t+ph+ (small?16:18)).toFixed(1)}" font-size="${fs}" fill="#999" text-anchor="middle">${fmt(v)}</text>`);
    parts.push(`<text x="${(m.l-6).toFixed(1)}" y="${(sy(v)+3).toFixed(1)}" font-size="${fs}" fill="#999" text-anchor="end">${fmt(v)}</text>`);
  }
  // axis labels
  parts.push(`<text x="${(m.l+pw/2).toFixed(1)}" y="${h-2}" font-size="${fs+1}" fill="#555" text-anchor="middle">${c.xl} MFE</text>`);
  parts.push(`<text x="10" y="${(m.t+ph/2).toFixed(1)}" font-size="${fs+1}" fill="#555" text-anchor="middle" transform="rotate(-90 10 ${(m.t+ph/2).toFixed(1)})">${c.yl} MFE</text>`);
  // points (non-PK first so PK draws on top); radius/opacity scale with density
  const r = small ? 2.2 : (pts.length > 500 ? 2.4 : 3.2);
  const op = pts.length > 500 ? 0.45 : 0.6;
  const order = pkOnly ? pts : pts.slice().sort((a,b)=>a[3]-b[3]);
  for (const p of order) {
    const col = p[3] ? COL.pk : COL.nonpk;
    parts.push(`<circle cx="${sx(p[c.x]).toFixed(1)}" cy="${sy(p[c.y]).toFixed(1)}" r="${r}" fill="${col}" fill-opacity="${op}" stroke="${col}" stroke-width="0.5" data-x="${p[c.x]}" data-y="${p[c.y]}" data-idx="${p[4]}" data-aa="${p[5]}" data-xl="${c.xl}" data-yl="${c.yl}"/>`);
  }
  parts.push(`</svg>`);
  return parts.join("");
}

function statLine(pop, cmpKey) {
  const c = CMP[cmpKey];
  let pts = pop.pts; if (pkOnly) pts = pts.filter(p => p[3] === 1);
  const r = pearson(pts, c.x, c.y);
  // mean signed gap (y - x): how much less stable fold-y is vs fold-x
  let g = 0; for (const p of pts) g += p[c.y] - p[c.x]; g = pts.length ? g/pts.length : 0;
  return `n=${pts.length} · r=${isNaN(r)?"—":r.toFixed(3)} · mean(${c.yl}−${c.xl})=${g.toFixed(2)}`;
}

function drawCard(card) {
  const pop = POPS.find(p => p.id === card.dataset.pop);
  const key = card.dataset.cmp;
  card.querySelector(".plot").innerHTML = scatterSVG(pop, key, 360, 300, false);
  card.querySelector(".stat").textContent = statLine(pop, key);
}

function drawAll() { document.querySelectorAll(".card").forEach(drawCard); }

// hover tooltip (delegated)
document.addEventListener("mousemove", (e) => {
  const t = e.target;
  if (t && t.tagName === "circle" && t.dataset.idx !== undefined) {
    tip.textContent = `AA${t.dataset.aa} idx${t.dataset.idx} · ${t.dataset.xl} ${t.dataset.x} / ${t.dataset.yl} ${t.dataset.y}`;
    tip.style.left = (e.clientX + 12) + "px"; tip.style.top = (e.clientY + 12) + "px"; tip.style.opacity = 1;
  } else { tip.style.opacity = 0; }
});

// per-card pair switch + Compare
document.querySelector(".grid").addEventListener("click", (e) => {
  const btn = e.target.closest(".topcap .mbtn"); if (!btn) return;
  const card = btn.closest(".card");
  if (btn.dataset.cmp === "compare") { openCompare(card.dataset.pop); return; }
  card.dataset.cmp = btn.dataset.cmp;
  card.querySelectorAll(".topcap .mbtn").forEach(b => b.classList.toggle("active", b === btn));
  drawCard(card);
});

// toolbar: pk-mixed / pk-only
document.querySelectorAll(".toolbar .tbtn[data-pk]").forEach(b => b.addEventListener("click", () => {
  pkOnly = b.dataset.pk === "only";
  document.querySelectorAll(".toolbar .tbtn[data-pk]").forEach(x => x.classList.toggle("active", x === b));
  drawAll();
}));

// Compare modal: all three pairings for one population, side by side
const modal = document.getElementById("cmp-modal");
function openCompare(popId) {
  const pop = POPS.find(p => p.id === popId);
  modal.querySelector("h2").textContent = pop.name;
  modal.querySelector(".csub").textContent = pop.sub + " · " + (pkOnly ? "PK only" : "PK-mixed") + " · all MFEs Knotergy (kcal/mol)";
  const cell = (k, lbl) => `<div class="cmp-cell"><h3>${lbl}</h3>${scatterSVG(pop, k, 340, 300, true)}<span class="stat">${statLine(pop, k)}</span></div>`;
  modal.querySelector(".cmp-grid").innerHTML =
    cell("kd", "Knotty vs DeRNA") + cell("kv", "Knotty vs ViennaRNA") + cell("vd", "ViennaRNA vs DeRNA");
  modal.classList.add("open");
}
modal.addEventListener("click", (e) => { if (e.target === modal || e.target.classList.contains("cmp-close")) modal.classList.remove("open"); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") modal.classList.remove("open"); });

drawAll();
"""


def build_html(pops, title: str) -> str:
    cards = []
    for p in pops:
        cards.append(
            f'<figure class="card" data-pop="{p["id"]}" data-cmp="kd">\n'
            '  <figcaption class="topcap">\n'
            '    <button class="mbtn active" data-cmp="kd">Knotty–DeRNA</button>\n'
            '    <button class="mbtn" data-cmp="kv">Knotty–ViennaRNA</button>\n'
            '    <button class="mbtn" data-cmp="vd">ViennaRNA–DeRNA</button>\n'
            '    <button class="mbtn cmp" data-cmp="compare">Compare</button>\n'
            '  </figcaption>\n'
            '  <div class="plot"></div>\n'
            '  <figcaption>\n'
            f'    <span class="title">{html.escape(p["name"])}</span>\n'
            f'    <span class="sub">{html.escape(p["sub"])}</span>\n'
            '    <span class="stat"></span>\n'
            '  </figcaption>\n'
            '</figure>'
        )
    payload = json.dumps([{k: p[k] for k in ("id", "name", "sub", "pts", "n", "n_pk", "n_nonpk")} for p in pops],
                         separators=(",", ":"))
    js = (_JS.replace("__POPS__", payload).replace("__CNONPK__", COLOR_NONPK).replace("__CPK__", COLOR_PK))
    note = ("Pairwise Knotergy-MFE scatter plots of the three folds. Each card switches "
            "between the pairings and Compare (all three); the toolbar toggles PK-mixed vs "
            "PK-only. All energies are Knotergy (Dirks-Pierce09, kcal/mol): "
            "Knotty=e_pk, ViennaRNA=e_nested, DeRNA computed. Dashed line = y=x (equal energy).")
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="color-scheme" content="light">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<title>{html.escape(title)}</title>\n<style>\n{_CSS}</style>\n</head>\n<body>\n'
        f'<h1>{html.escape(title)}</h1>\n'
        '<p class="note"><a class="back" href="index.html">← catalog home</a><br>' + html.escape(note) + '</p>\n'
        '<div class="toolbar">Population:\n'
        '  <span class="seg"><button class="tbtn active" data-pk="mixed">PK-mixed</button>'
        '<button class="tbtn" data-pk="only">PK-only</button></span>\n'
        f'  <span class="legend"><span><span class="dot" style="background:{COLOR_NONPK}"></span>non-PK</span>'
        f'<span><span class="dot" style="background:{COLOR_PK}"></span>PK</span></span>\n'
        '</div>\n'
        f'<div class="grid">\n{chr(10).join(cards)}\n</div>\n'
        '<div id="cmp-modal" class="cmp-modal"><div class="cmp-box"><button class="cmp-close">Close ✕</button>'
        '<h2></h2><p class="csub"></p><div class="cmp-grid"></div></div></div>\n'
        '<div id="tip" class="tip"></div>\n'
        f'<script>\n{js}\n</script>\n</body>\n</html>\n'
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--derna-mfe-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="mRNA MFE scatter catalog — Knotty / ViennaRNA / DeRNA")
    args = ap.parse_args()

    pops = _load(args.parquet, args.derna_mfe_json)
    for p in pops:
        print(f"  {p['id']}: {p['n']} points ({p['n_pk']} PK / {p['n_nonpk']} non-PK)")
    out_html = build_html(pops, args.title)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(out_html)
    print(f"Wrote {args.out} ({sum(p['n'] for p in pops)} points across {len(pops)} populations)")


if __name__ == "__main__":
    main()
