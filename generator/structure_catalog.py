#!/usr/bin/env python3
"""Secondary-structure catalog generator (issue #57).

Render a grid of RNA secondary structures from a BPSeq-tokenised ``.npy`` dataset,
where each cell is drawn by `forna <http://rna.tbi.univie.ac.at/forna/>`_ — the same
force-directed layout as the web tool, via the self-contained ``fornac.js`` bundle.

The output is a single self-contained HTML file: open it in a browser and each grid
cell renders the structure interactively (pan/zoom, pseudoknots included).

Dependency-light by design (numpy + stdlib only). The token vocabulary and the
dot-bracket conversion mirror ``src/rna_converter.py`` (``RNAConverter`` with
``max_partner_idx = protein_len * 3``) but are reproduced here so the catalog can be
generated in environments without torch.

Example
-------
    python3 scripts/eval/structure_catalog.py \
        --data path/to/derna20_val_bptok_pk.npy \
        --indices 995 \
        --out artifacts/structure_catalog/val995.html
"""
from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
from typing import Dict, List, Tuple

import numpy as np

BASES = ["A", "C", "G", "U"]
# Bracket symbols per pseudoknot level, matching RNAConverter.bpseq_to_dot_bracket.
LEVEL_BRACKETS = [
    ("(", ")"), ("[", "]"), ("{", "}"), ("<", ">"),
    ("A", "a"), ("B", "b"), ("C", "c"), ("D", "d"),
]


def build_id_to_token(max_partner_idx: int) -> List[str]:
    """RNA token list indexed by token id, mirroring RNAConverter._build_rna_tokens."""
    tokens = ["STOP"]
    tokens.extend(
        f"{base}{partner_idx}"
        for partner_idx in range(max_partner_idx + 1)
        for base in BASES
    )
    return tokens


def decode_row(row: np.ndarray, id_to_token: List[str]) -> Tuple[str, str]:
    """Convert one row of BPSeq token ids to (sequence, dot-bracket structure).

    Mirrors RNAConverter.bptok_to_bpseq + bpseq_to_dot_bracket: partner indices are
    1-based positions; crossing pairs are greedily assigned to ascending bracket
    levels so pseudoknots survive the round-trip.
    """
    sequence: List[str] = []
    pairs: List[Tuple[int, int]] = []
    for idx, tok_id in enumerate(int(t) for t in row):
        tok = id_to_token[tok_id]
        if tok == "STOP":
            break
        base, suffix = tok[0], tok[1:]
        sequence.append(base)
        partner_1based = int(suffix) if suffix else 0
        if partner_1based != 0:
            partner_0based = partner_1based - 1
            if idx < partner_0based:  # record each pair once
                pairs.append((idx, partner_0based))

    n = len(sequence)
    levels: List[List[Tuple[int, int]]] = []
    for u, v in pairs:
        placed = False
        for lvl_pairs in levels:
            if not any(x < u < y < v for x, y in lvl_pairs):
                lvl_pairs.append((u, v))
                placed = True
                break
        if not placed:
            levels.append([(u, v)])

    structure = ["."] * n
    for lvl_idx, lvl_pairs in enumerate(levels):
        open_char, close_char = LEVEL_BRACKETS[lvl_idx] if lvl_idx < len(LEVEL_BRACKETS) else ("!", "!")
        for u, v in lvl_pairs:
            structure[u] = open_char
            structure[v] = close_char

    return "".join(sequence), "".join(structure)


def structure_stats(structure: str) -> dict:
    """Summary used in each card's caption."""
    pk_levels = sum(1 for o, _ in LEVEL_BRACKETS if o in structure)
    n_pairs = sum(structure.count(o) for o, _ in LEVEL_BRACKETS)
    # pseudoknotted pairs = openers in any level beyond the primary ``()`` level.
    pk_pairs = sum(structure.count(o) for o, _ in LEVEL_BRACKETS[1:])
    return {
        "length": len(structure),
        "pairs": n_pairs,
        "pk_pairs": pk_pairs,
        "pk_levels": pk_levels,
        "is_pseudoknotted": pk_levels > 1,
    }


CARD_TEMPLATE = """    <figure class="card">
      <div class="fornac" id="rna_{i}" data-i="{i}"></div>
      <figcaption>
        <span class="title">{title}</span>
        <span class="meta">{meta}</span>
      </figcaption>
      <span class="badge {pk_class}">{pk_label}</span>
    </figure>"""


def build_live_html(entries: List[dict], title: str, fornac_url: str) -> str:
    """Assemble the live, lazy-rendered catalog HTML (fornac runs in the browser)."""
    cards = "\n".join(
        CARD_TEMPLATE.format(
            i=i,
            title=html.escape(e["title"]),
            meta=html.escape(e["meta"]),
            pk_class="pk" if e["pk"] else "nonpk",
            pk_label="PK" if e["pk"] else "Non-PK",
        )
        for i, e in enumerate(entries)
    )
    rna_payload = json.dumps(
        [{"structure": e["structure"], "sequence": e["sequence"]} for e in entries]
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
  body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 24px; background: #fafafa; }}
  h1 {{ font-size: 20px; font-weight: 600; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 16px;
  }}
  .card {{
    position: relative;
    margin: 0; background: #fff; border: 1px solid #e3e3e3; border-radius: 8px;
    overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }}
  .fornac {{ width: 100%; height: 280px; overflow: hidden; }}
  figcaption {{ padding: 8px 12px 12px; border-top: 1px solid #f0f0f0; }}
  .title {{ display: block; font-weight: 600; font-size: 14px; }}
  .meta {{ display: block; color: #666; font-size: 12px; margin-top: 2px; }}
  .badge {{
    position: absolute; bottom: 8px; right: 8px;
    padding: 2px 8px; border-radius: 10px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.3px;
  }}
  .badge.pk {{ background: #fce4d6; color: #b5360b; border: 1px solid #f3b08a; }}
  .badge.nonpk {{ background: #e6f0e6; color: #2f6f3a; border: 1px solid #b7d8bb; }}
  .controls {{ margin: 0 0 14px; }}
  .controls .note {{ color: #888; font-size: 12px; }}
</style>
<script src="{fornac_url}"></script>
</head>
<body>
<h1>{html.escape(title)}</h1>
<div class="controls">
  <span id="build-stat" class="note">loading…</span>
</div>
<div class="grid">
{cards}
</div>
<script>
// Static, lazy/virtualized catalog: force layout and pan/zoom are both disabled for
// speed, and each structure is built only while its card is near the viewport (then
// torn down) so the page scales to thousands of entries without a load freeze.
// NOTE: `applyForce` is the fornac@1.1.8 option name (sets internal `animation`);
// newer fornac renamed it to `animation`. If you bump the version, update this flag.
const RNAS = {rna_payload};
const built = new Set();   // indices already rendered (built once, never torn down)
const queue = [];          // indices waiting to render
let scheduled = false;
const PER_FRAME = 4;       // cap builds per animation frame so scrolling never blocks

function buildCard(i) {{
  if (built.has(i)) return;
  built.add(i);
  const c = new fornac.FornaContainer("#rna_" + i,
    {{ applyForce: false, allowPanningAndZooming: false, initialSize: [320, 280] }});
  c.addRNA(RNAS[i].structure, {{ sequence: RNAS[i].sequence }});
}}

function drain() {{
  scheduled = false;
  let n = 0;
  while (queue.length && n < PER_FRAME) {{
    buildCard(queue.shift());
    n++;
  }}
  document.getElementById("build-stat").textContent =
    `${{RNAS.length}} entries · ${{built.size}} rendered (force/zoom off)`;
  if (queue.length) schedule();
}}

function schedule() {{
  if (scheduled) return;
  scheduled = true;
  requestAnimationFrame(drain);
}}

// Build each card once, the first time it nears the viewport — then stop observing it.
// Heights are fixed (CSS), so building never shifts layout; no teardown means no
// rebuild churn or scroll jumps when scrolling back.
const io = new IntersectionObserver((entries) => {{
  for (const e of entries) {{
    if (!e.isIntersecting) continue;
    io.unobserve(e.target);
    const i = +e.target.dataset.i;
    if (!built.has(i)) {{ queue.push(i); schedule(); }}
  }}
}}, {{ rootMargin: "400px 0px" }});

document.querySelectorAll(".fornac").forEach((el) => io.observe(el));
drain();
</script>
</body>
</html>
"""


STATIC_CARD_TEMPLATE = """    <figure class="card" data-idx="{idx}" data-aalen="{aalen}" data-bp="{bp}" data-pkbp="{pkbp}" data-mfe="{mfe}">
      <div class="fornac">{inner}</div>
      <figcaption>
        <span class="title">{title}</span>
        <span class="meta">{meta}</span>
        <span class="aa">{aa}</span>
      </figcaption>
      <span class="badge {pk_class}">{pk_label}</span>
    </figure>"""

# Shared page CSS for the pre-rendered (static) catalogs. Plain string (no interpolation).
_STATIC_PAGE_CSS = """\
  :root { color-scheme: light; }  /* opt out of browser auto-dark (it inverts SVG bg to black) */
  body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 24px; background: #fafafa; }
  h1 { font-size: 20px; font-weight: 600; }
  .note { color: #888; font-size: 12px; margin: 0 0 14px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }
  .card { position: relative; margin: 0; background: #fff; border: 1px solid #e3e3e3;
          border-radius: 8px; overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
  .fornac { width: 100%; height: 280px; overflow: hidden; }
  .fornac svg, .fornac img { display: block; margin: 0 auto; }
  /* override fornac's injected `svg { width:100% }` so inline SVGs stay fixed + centered */
  .fornac svg { width: 320px; min-width: 0; height: 280px; }
  .fornac img { width: 320px; height: 280px; }
  figcaption { padding: 8px 12px 12px; border-top: 1px solid #f0f0f0; }
  .title { display: block; font-weight: 600; font-size: 14px; }
  .meta { display: block; color: #666; font-size: 12px; margin-top: 2px; }
  .aa { display: block; margin-top: 4px; font-family: ui-monospace, Menlo, Consolas, monospace;
        font-size: 10.5px; color: #777; word-break: break-all; line-height: 1.35; }
  body.hide-aa .aa { display: none; }
  .badge { position: absolute; bottom: 8px; right: 8px; padding: 2px 8px; border-radius: 10px;
           font-size: 11px; font-weight: 700; letter-spacing: 0.3px; }
  .badge.pk { background: #fce4d6; color: #b5360b; border: 1px solid #f3b08a; }
  .badge.nonpk { background: #e6f0e6; color: #2f6f3a; border: 1px solid #b7d8bb; }
  .sortbar { margin: 0 0 14px; font-size: 13px; color: #444; }
  .sortbar button { margin-left: 6px; padding: 3px 10px; border: 1px solid #ccc; background: #fff;
                    border-radius: 6px; cursor: pointer; font-size: 12px; }
  .sortbar button:hover { background: #f3f3f3; }
  .sortbar button.active { background: #d8ebff; border-color: #9cc8f0; color: #0b5cab; }
  .force-toggle { margin: 0 0 14px; padding: 4px 12px; border: 1px solid #ccc; background: #fff;
                  border-radius: 6px; cursor: pointer; font-size: 13px; }
  .force-toggle:hover { background: #f3f3f3; }
  .force-toggle.active { background: #d8ebff; border-color: #9cc8f0; color: #0b5cab; }
  /* A/B compare: show the force-off layout by default, force-on when body.force-on */
  .fornac .lay-on { display: none; }
  body.force-on .fornac .lay-off { display: none; }
  body.force-on .fornac .lay-on { display: block; }
"""

# Client-side sort (DOM reorder only — no re-render). Plain string; no interpolation.
_SORT_JS = """\
(function () {
  const grid = document.querySelector(".grid");
  const LABELS = { aalen: "AA length", idx: "Index", bp: "Base pairs", mfe: "MFE", pkbp: "PK pairs" };
  let cur = "idx", asc = true;
  const num = (c, k) => {
    const v = c.dataset[k];
    if (v === "" || v == null) return k === "mfe" ? Infinity : -Infinity; // missing sinks last
    return parseFloat(v);
  };
  function relabel() {
    for (const k in LABELS) {
      const b = document.querySelector('.sortbar button[data-key="' + k + '"]');
      b.textContent = LABELS[k] + (k === cur ? (asc ? " \\u2191" : " \\u2193") : "");
      b.classList.toggle("active", k === cur);
    }
  }
  function sortBy(k) {
    if (k === cur) asc = !asc; else { cur = k; asc = (k === "idx" || k === "mfe" || k === "aalen"); }
    [...grid.children]
      .sort((a, b) => (asc ? num(a, k) - num(b, k) : num(b, k) - num(a, k)))
      .forEach((c) => grid.appendChild(c));
    relabel();
  }
  document.querySelectorAll(".sortbar button[data-key]")
    .forEach((b) => b.addEventListener("click", () => sortBy(b.dataset.key)));
  relabel();
  const aaBtn = document.getElementById("aa-toggle");
  if (aaBtn) aaBtn.addEventListener("click", () => {
    const hidden = document.body.classList.toggle("hide-aa");
    aaBtn.textContent = hidden ? "Show AA seq" : "Hide AA seq";
    aaBtn.classList.toggle("active", hidden);
  });
})();
"""

# Toggle between the two baked layouts (force-off vs force-relaxed). DOM/CSS only.
_COMPARE_JS = """\
(function () {
  const btn = document.getElementById("force-toggle");
  if (!btn) return;
  btn.addEventListener("click", () => {
    const on = document.body.classList.toggle("force-on");
    btn.textContent = "Force layout: " + (on ? "ON (relaxed)" : "OFF (static)");
    btn.classList.toggle("active", on);
  });
})();
"""


_CLOSER_TO_OPENER = {c: o for o, c in LEVEL_BRACKETS}
_OPENERS = {o for o, _ in LEVEL_BRACKETS}
# forna's "structure" colour scheme: domain s,m,i,e,t,h -> these fills (x = transparent).
_ELEM_COLOR = {"s": "lightgreen", "m": "#ff9896", "i": "#dbdb8d",
               "e": "lightsalmon", "t": "lightcyan", "h": "lightblue", "p": "lightgreen"}


_CODON = {
    "UUU": "F", "UUC": "F", "UUA": "L", "UUG": "L", "CUU": "L", "CUC": "L", "CUA": "L", "CUG": "L",
    "AUU": "I", "AUC": "I", "AUA": "I", "AUG": "M", "GUU": "V", "GUC": "V", "GUA": "V", "GUG": "V",
    "UCU": "S", "UCC": "S", "UCA": "S", "UCG": "S", "AGU": "S", "AGC": "S", "CCU": "P", "CCC": "P",
    "CCA": "P", "CCG": "P", "ACU": "T", "ACC": "T", "ACA": "T", "ACG": "T", "GCU": "A", "GCC": "A",
    "GCA": "A", "GCG": "A", "UAU": "Y", "UAC": "Y", "CAU": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAU": "N", "AAC": "N", "AAA": "K", "AAG": "K", "GAU": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "UGU": "C", "UGC": "C", "UGG": "W", "CGU": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R",
    "AGG": "R", "GGU": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    "UAA": "*", "UAG": "*", "UGA": "*",
}


def translate(rna: str) -> str:
    """Translate an mRNA sequence to its protein (stop at the first stop codon)."""
    out = []
    for i in range(0, len(rna) - 2, 3):
        aa = _CODON.get(rna[i:i + 3], "?")
        if aa == "*":
            break
        out.append(aa)
    return "".join(out)


def parse_pairs(structure: str):
    """All base pairs as (i, j, is_pk); is_pk = pair from a non-primary (crossing) level."""
    stacks = {o: [] for o, _ in LEVEL_BRACKETS}
    pairs = []
    for i, ch in enumerate(structure):
        if ch in _OPENERS:
            stacks[ch].append(i)
        elif ch in _CLOSER_TO_OPENER:
            opener = _CLOSER_TO_OPENER[ch]
            if stacks[opener]:
                j = stacks[opener].pop()
                pairs.append((j, i, opener != "("))
    return pairs


def element_string(structure: str):
    """Per-position structural element (forna scheme): s stem, h hairpin, i interior/bulge,
    m multiloop, e exterior; PK-paired positions -> p. Pure-Python loop decomposition over
    the nested ``()`` skeleton (matches forgi's f/s/h/t/i/m classes for drawing)."""
    n = len(structure)
    pt = [-1] * n
    st = []
    for i, ch in enumerate(structure):
        if ch == "(":
            st.append(i)
        elif ch == ")" and st:
            j = st.pop()
            pt[i] = j
            pt[j] = i
    pk = {i for i, ch in enumerate(structure) if ch in (_OPENERS | set(_CLOSER_TO_OPENER)) and ch not in "()"}

    enclosing = [-1] * n
    stack = []
    for i in range(n):
        if pt[i] > i:          # opener
            enclosing[i] = stack[-1] if stack else -1
            stack.append(i)
        elif pt[i] != -1:      # closer
            if stack and stack[-1] == pt[i]:
                stack.pop()
            enclosing[i] = stack[-1] if stack else -1
        else:                  # unpaired
            enclosing[i] = stack[-1] if stack else -1

    children = {}
    for i in range(n):
        if pt[i] > i:
            children.setdefault(enclosing[i], 0)
            children[enclosing[i]] += 1

    es = []
    for i in range(n):
        if i in pk:
            es.append("p")
        elif pt[i] != -1:
            es.append("s")
        elif enclosing[i] == -1:
            es.append("e")
        else:
            c = children.get(enclosing[i], 0)
            es.append("h" if c == 0 else "i" if c == 1 else "m")
    return es


def relax_loops(px, py, es, bond, iters=140):
    """Open up NAView's collapsed loops/bulges: pin all PAIRED nodes (stems + PK) and let
    only UNPAIRED nodes move under repulsion + backbone springs. Pinning stems means no
    global distortion and no PK tangle — bulges/loops simply bow outward."""
    n = len(px)
    movable = [c not in ("s", "p") for c in es]
    if not any(movable):
        return px, py
    rep_r = bond * 1.5
    for _ in range(iters):
        fx = [0.0] * n
        fy = [0.0] * n
        for a in range(n):
            xa, ya = px[a], py[a]
            for b in range(a + 1, n):
                dx, dy = xa - px[b], ya - py[b]
                d2 = dx * dx + dy * dy
                if 1e-6 < d2 < rep_r * rep_r:
                    d = d2 ** 0.5
                    f = (rep_r - d) / d * 0.3
                    fx[a] += dx * f; fy[a] += dy * f
                    fx[b] -= dx * f; fy[b] -= dy * f
        for i in range(n - 1):
            dx, dy = px[i + 1] - px[i], py[i + 1] - py[i]
            d = (dx * dx + dy * dy) ** 0.5 or 1.0
            f = (d - bond) / d * 0.5
            fx[i] += dx * f; fy[i] += dy * f
            fx[i + 1] -= dx * f; fy[i + 1] -= dy * f
        # Laplacian smoothing on movable nodes -> rounds loops into smooth arcs (no spikes)
        for i in range(1, n - 1):
            if movable[i]:
                fx[i] += ((px[i - 1] + px[i + 1]) / 2 - px[i]) * 0.3
                fy[i] += ((py[i - 1] + py[i + 1]) / 2 - py[i]) * 0.3
        for i in range(n):
            if movable[i]:
                px[i] += max(-bond, min(bond, fx[i])) * 0.12
                py[i] += max(-bond, min(bond, fy[i])) * 0.12
    return px, py


def naview_svg(sequence: str, structure: str) -> str:
    """Render a crossing-free static SVG via ViennaRNA's NAView layout, styled like forna
    (structure colour scheme, letters, position labels every 10, #999 backbone). Loops are
    opened by a stems-pinned relaxation; pseudoknot pairs drawn as red links (they cross)."""
    import RNA

    n = len(structure)
    nested = "".join(ch if ch in "()" else "." for ch in structure)
    co = RNA.naview_xy_coordinates(nested)
    px = [co[i].X for i in range(n)]
    py = [-co[i].Y for i in range(n)]  # flip Y (SVG y grows downward); natural units
    es = element_string(structure)
    pairs = parse_pairs(structure)
    partner = [-1] * n
    for i, j, _ in pairs:
        partner[i] = j
        partner[j] = i

    # node geometry scales with the layout's bond length -> constant visual density
    bond = sum(((px[i + 1] - px[i]) ** 2 + (py[i + 1] - py[i]) ** 2) ** 0.5
               for i in range(n - 1)) / max(n - 1, 1) or 15.0
    px, py = relax_loops(px, py, es, bond)  # bow out collapsed bulges/loops (stems pinned)

    radius = bond * 0.30
    lw = bond * 0.11
    fs = bond * 0.52
    margin = bond * 2.2

    def line(i, j, stroke, width, opacity=1.0):
        return (f'<line x1="{px[i]:.1f}" y1="{py[i]:.1f}" x2="{px[j]:.1f}" y2="{py[j]:.1f}" '
                f'stroke="{stroke}" stroke-width="{width:.2f}" stroke-opacity="{opacity}"/>')

    minx, maxx = min(px) - margin, max(px) + margin
    miny, maxy = min(py) - margin, max(py) + margin
    vb_w, vb_h = maxx - minx, maxy - miny
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{minx:.1f} {miny:.1f} {vb_w:.1f} {vb_h:.1f}" '
             f'width="{vb_w:.0f}" height="{vb_h:.0f}">']
    # base pairs (under nodes): nested gray, pseudoknot red
    for i, j, is_pk in pairs:
        parts.append(line(i, j, "#d9534f" if is_pk else "#999", lw, 0.6 if is_pk else 0.8))
    # backbone (forna: #999, opacity .8)
    for i in range(n - 1):
        parts.append(line(i, i + 1, "#999", lw, 0.8))
    # position labels every 10: perpendicular to the backbone, on the side away from the partner
    for i in range(n):
        if (i + 1) % 10 == 0 or i == 0:
            a, b = max(i - 1, 0), min(i + 1, n - 1)
            tx, ty = px[b] - px[a], py[b] - py[a]
            tl = (tx * tx + ty * ty) ** 0.5 or 1.0
            nx, ny = -ty / tl, tx / tl  # unit perpendicular
            if partner[i] != -1 and (nx * (px[partner[i]] - px[i]) + ny * (py[partner[i]] - py[i])) > 0:
                nx, ny = -nx, -ny       # flip to the outside of the helix
            lx, ly = px[i] + nx * (radius + bond * 0.9), py[i] + ny * (radius + bond * 0.9)
            parts.append(f'<text x="{lx:.1f}" y="{ly + fs * 0.35:.1f}" text-anchor="middle" font-size="{fs:.1f}" '
                         f'font-family="Tahoma,Geneva,sans-serif" fill="#999">{i + 1}</text>')
    # nucleotide nodes + letters
    for i in range(n):
        base = sequence[i] if i < len(sequence) else "N"
        parts.append(
            f'<circle cx="{px[i]:.1f}" cy="{py[i]:.1f}" r="{radius:.2f}" fill="{_ELEM_COLOR.get(es[i], "#fff")}" stroke="#ccc" stroke-width="{lw:.2f}"/>'
            f'<text x="{px[i]:.1f}" y="{py[i] + radius * 0.5:.1f}" text-anchor="middle" font-size="{radius * 1.25:.1f}" '
            f'font-family="Tahoma,Geneva,sans-serif" font-weight="bold" fill="rgb(100,100,100)">{base}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def minify_svg(svg: str) -> str:
    """Shrink a captured fornac SVG: drop hover tooltips + interaction-only attrs, round
    coordinates to 2 decimals, strip the (unreferenced) root id, collapse whitespace.
    Keeps text-anchor / class / inline style (all visually significant)."""
    svg = re.sub(r"<title>.*?</title>", "", svg)
    # drop forna's invisible "outline_node" halo circles: they're default-black and only
    # hidden via CSS (fragile — fails to apply in some embed contexts, e.g. the HF Space
    # iframe, turning cards black). Useless in a static catalog, so remove them outright.
    svg = re.sub(r'<circle\b[^>]*\bclass="outline_node"[^>]*></circle>', "", svg)
    # drop forna's opaque white background rect: a white <rect> inside the SVG gets inverted
    # to black by browser auto-dark (turning cards black, e.g. in the HF Space). Removing it
    # makes the SVG transparent so the card's own (color-scheme-protected) background shows.
    svg = re.sub(r'<rect\b[^>]*\bid="zrect"[^>]*></rect>', "", svg)
    svg = re.sub(r'\s(?:pointer-events|link_type|label_type)="[^"]*"', "", svg)
    svg = re.sub(r'\sid="plotting-area"', "", svg)
    svg = re.sub(r"-?\d+\.\d{3,}", lambda m: f"{float(m.group(0)):.2f}", svg)
    svg = re.sub(r">\s+<", "><", svg)
    svg = re.sub(r"\s{2,}", " ", svg).strip()
    # Add a viewBox (from the render width/height) so a large force-relaxed canvas scales
    # crisply into the fixed 320x280 card instead of being cropped.
    if "viewBox" not in svg[:300]:
        m = re.search(r'<svg[^>]*\bwidth="(\d+)"[^>]*\bheight="(\d+)"', svg)
        if m:
            svg = svg.replace("<svg ", f'<svg viewBox="0 0 {m.group(1)} {m.group(2)}" ', 1)
    return svg


def run_prerender(entries: List[Dict], force: bool = False):
    """Render every entry's structure headlessly via fornac and return (css, [svg, ...]).

    With ``force=True`` the force simulation runs and settles before capture, so the
    relaxed (less cramped) coordinates are baked into the static SVG — useful for long
    structures. The output page stays static either way.
    """
    import subprocess
    import tempfile

    payload = [{"structure": e["structure"], "sequence": e["sequence"]} for e in entries]
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prerender", "prerender_fornac.js")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(payload, fh)
        in_path = fh.name
    out_path = in_path + ".out.json"
    env = {**os.environ, "PRERENDER_FORCE": "1"} if force else None
    try:
        subprocess.run(["node", script, in_path, out_path], check=True, env=env)
        data = json.loads(open(out_path).read())
    finally:
        for p in (in_path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass
    return data.get("css", ""), data.get("svgs", [])


def run_forna_prerender(entries: List[Dict], jobs: int = 1):
    """Option A: render via forna's exact pipeline (NAView coords computed locally + forna's
    own fornac.js force). Returns (css, [svg, ...]).

    With jobs>1 the work is sharded across that many concurrent headless-Chrome processes
    (each ~1 core) — the practical way to parallelize the force-settle bottleneck locally."""
    import subprocess
    import tempfile
    import RNA

    payload = []
    for e in entries:
        nested = "".join(c if c in "()" else "." for c in e["structure"])
        co = RNA.naview_xy_coordinates(nested)
        coords = [[co[i].X, co[i].Y] for i in range(len(e["structure"]))]
        payload.append({"seq": e["sequence"], "struct": e["structure"], "coords": coords})
    if not payload:
        return "", []

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prerender", "prerender_forna.js")
    jobs = max(1, min(jobs, len(payload)))
    chunk = (len(payload) + jobs - 1) // jobs
    shards = [list(range(i, min(i + chunk, len(payload)))) for i in range(0, len(payload), chunk)]

    procs = []
    for sh in shards:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump([payload[i] for i in sh], fh)
            in_path = fh.name
        out_path = in_path + ".out.json"
        p = subprocess.Popen(["node", script, in_path, out_path])
        procs.append((p, sh, in_path, out_path))

    svgs = [None] * len(payload)
    css = ""
    for p, sh, in_path, out_path in procs:
        p.wait()
        try:
            data = json.loads(open(out_path).read())
            for local_i, gi in enumerate(sh):
                svgs[gi] = data["svgs"][local_i] if local_i < len(data["svgs"]) else None
            css = css or data.get("css", "")
        finally:
            for q in (in_path, out_path):
                try:
                    os.unlink(q)
                except OSError:
                    pass
    return css, svgs


def _static_cards(entries, inners):
    return "\n".join(
        STATIC_CARD_TEMPLATE.format(
            inner=inner,
            idx=e["idx"],
            aalen=e.get("aa_len", 0),
            bp=e["bp"],
            pkbp=e["pk_pairs"],
            mfe="" if e.get("mfe") is None else f"{e['mfe']:.4f}",
            title=html.escape(e["title"]),
            meta=html.escape(e["meta"]),
            aa="AA " + html.escape(e.get("aa", "")),
            pk_class="pk" if e["pk"] else "nonpk",
            pk_label="PK" if e["pk"] else "Non-PK",
        )
        for e, inner in zip(entries, inners)
    )


_SORTBAR_HTML = (
    '<div class="sortbar">Sort:'
    '<button data-key="aalen">AA length</button>'
    '<button data-key="idx">Index</button>'
    '<button data-key="bp">Base pairs</button>'
    '<button data-key="mfe">MFE</button>'
    '<button data-key="pkbp">PK pairs</button>'
    '<button id="aa-toggle" style="margin-left:16px">Hide AA seq</button>'
    "</div>"
)


def _static_page(title: str, note: str, head_extra: str, cards: str,
                 controls_html: str = "", extra_js: str = "") -> str:
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="color-scheme" content="light">\n'
        f"<title>{html.escape(title)}</title>\n<style>\n{_STATIC_PAGE_CSS}</style>\n"
        f"{head_extra}</head>\n<body>\n<h1>{html.escape(title)}</h1>\n"
        f'<p class="note">{html.escape(note)}</p>\n{controls_html}{_SORTBAR_HTML}\n'
        f'<div class="grid">\n{cards}\n</div>\n'
        f"<script>\n{_SORT_JS}{extra_js}</script>\n</body>\n</html>\n"
    )


def build_inline_svg_html(entries, title, fornac_css, svgs) -> str:
    """Single self-contained file with every structure embedded as inline <svg>."""
    cards = _static_cards(entries, svgs)
    head_extra = f"<style>\n{fornac_css}\n</style>\n"
    note = f"{len(entries)} entries · pre-rendered static SVG · fixed positions"
    return _static_page(title, note, head_extra, cards)


_SVG_XMLNS = "http://www.w3.org/2000/svg"


def _assets_dir_for(out_path: str):
    out_abs = os.path.abspath(out_path)
    stem = os.path.splitext(os.path.basename(out_abs))[0]
    dirname = f"{stem}_assets"
    path = os.path.join(os.path.dirname(out_abs), dirname)
    os.makedirs(path, exist_ok=True)
    return dirname, path


def _write_standalone_svg(assets_dir: str, fname: str, svg: str, style_tag: str):
    """fornac's class-based styles must live INSIDE each file (image SVGs ignore page CSS)."""
    if "xmlns=" not in svg[:200]:
        svg = svg.replace("<svg ", f'<svg xmlns="{_SVG_XMLNS}" ', 1)
    svg = re.sub(r"(<svg[^>]*>)", r"\1" + style_tag, svg, count=1)
    with open(os.path.join(assets_dir, fname), "w") as fh:
        fh.write(svg)


def build_img_svg_html(entries, title, fornac_css, svgs, out_path: str) -> str:
    """Write one standalone .svg per structure into a sibling folder; reference them with
    native lazy <img> so the HTML stays tiny and the browser decodes only visible cards."""
    assets_dirname, assets_dir = _assets_dir_for(out_path)
    style_tag = f"<style>{fornac_css}</style>"
    inners = []
    for e, svg in zip(entries, svgs):
        fname = f"aa{e.get('aa_len', 0)}_idx{e['idx']}.svg"  # unique across AA lengths
        _write_standalone_svg(assets_dir, fname, svg, style_tag)
        src = f"{assets_dirname}/{fname}"
        inners.append(f'<img loading="lazy" width="320" height="280" src="{src}" alt="idx {e["idx"]}">')

    note = f"{len(entries)} entries · pre-rendered SVG files (lazy <img>) · fixed positions · ./{assets_dirname}/"
    return _static_page(title, note, "", _static_cards(entries, inners))


def build_img_compare_html(entries, title, fornac_css, svgs_off, svgs_on, out_path: str) -> str:
    """A/B layout compare: bake BOTH force-off and force-relaxed SVGs; a button flips every
    card between them (DOM/CSS only — instant, no live force)."""
    assets_dirname, assets_dir = _assets_dir_for(out_path)
    style_tag = f"<style>{fornac_css}</style>"
    inners = []
    for e, svg_off, svg_on in zip(entries, svgs_off, svgs_on):
        f_off, f_on = f"idx_{e['idx']}.svg", f"idx_{e['idx']}_force.svg"
        _write_standalone_svg(assets_dir, f_off, svg_off, style_tag)
        _write_standalone_svg(assets_dir, f_on, svg_on, style_tag)
        inners.append(
            f'<img class="lay-off" loading="lazy" width="320" height="280" src="{assets_dirname}/{f_off}" alt="idx {e["idx"]} static">'
            f'<img class="lay-on" loading="lazy" width="320" height="280" src="{assets_dirname}/{f_on}" alt="idx {e["idx"]} force">'
        )
    controls = '<button id="force-toggle" class="force-toggle">Force layout: OFF (static)</button>\n'
    note = f"{len(entries)} entries · A/B: toggle force-off vs force-relaxed baked layouts · ./{assets_dirname}/"
    return _static_page(title, note, "", _static_cards(entries, inners),
                        controls_html=controls, extra_js=_COMPARE_JS)


def load_npy_rows(path: str, protein_len: int) -> List[Dict]:
    """Load a BPSeq-tokenised .npy dataset into normalised rows (decode each token row)."""
    data = np.load(path)
    n_cols = data.shape[1]
    expected_cols = protein_len * 3
    if n_cols != expected_cols:
        raise SystemExit(
            f"Token length {n_cols} != protein_len*3 ({expected_cols}); pass --protein-len {n_cols // 3}"
        )
    id_to_token = build_id_to_token(expected_cols)
    rows = []
    for idx in range(data.shape[0]):
        sequence, structure = decode_row(data[idx], id_to_token)
        rows.append({"idx": idx, "sequence": sequence, "structure": structure,
                     "pk": structure_stats(structure)["is_pseudoknotted"],
                     "aa": translate(sequence), "aa_len": protein_len})
    return rows


def load_parquet_rows(path: str, aa_len, seq_col: str, struct_col: str, pk_col: str) -> List[Dict]:
    """Load a per-sample profiling parquet (e.g. issue #47 pk_profile.parquet) into rows.

    Uses the dataset's own PK flag (``pk_col``, e.g. ``has_pk``) as the authoritative
    label rather than re-deriving it from the bracket string.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    if aa_len is not None and "aa_len" in df.columns:
        df = df[df["aa_len"] == aa_len]
    if "ok" in df.columns:
        df = df[df["ok"]]
    idx_col = "idx" if "idx" in df.columns else None
    has_mfe = "e_pk" in df.columns  # Knotergy energy of the displayed (PK-allowed) fold
    has_aa = "aa_seq" in df.columns
    has_len = "aa_len" in df.columns
    rows = []
    for _, r in df.iterrows():
        seq = str(r[seq_col])
        rows.append({
            "idx": int(r[idx_col]) if idx_col else len(rows),
            "sequence": seq,
            "structure": str(r[struct_col]),
            "pk": bool(r[pk_col]),
            "mfe": float(r["e_pk"]) if has_mfe else None,
            "aa": str(r["aa_seq"]) if has_aa else translate(seq),
            "aa_len": int(r["aa_len"]) if has_len else len(seq) // 3,
        })
    return rows


def select_rows(rows: List[Dict], *, filt: str, random_n, seed: int, indices: str) -> List[Dict]:
    """Apply the PK filter, then either an explicit index list / slice or a random sample."""
    if filt == "pk":
        rows = [r for r in rows if r["pk"]]
    elif filt == "nonpk":
        rows = [r for r in rows if not r["pk"]]

    if random_n is not None:
        if random_n < len(rows):
            rows = random.Random(seed).sample(rows, random_n)
            rows.sort(key=lambda r: r["idx"])
        return rows

    if indices and indices != "all":
        by_idx = {r["idx"]: r for r in rows}
        if ":" in indices:
            lo, hi = (int(x) if x else None for x in indices.split(":", 1))
            want = range(*slice(lo, hi).indices(max(by_idx) + 1 if by_idx else 0))
            return [by_idx[i] for i in want if i in by_idx]
        return [by_idx[int(x)] for x in indices.split(",") if int(x) in by_idx]
    return rows


def finalise_entries(rows: List[Dict]) -> List[Dict]:
    """Attach display title/meta + numeric sort keys (bp, pk-bp, mfe) to each row."""
    entries = []
    for r in rows:
        stats = structure_stats(r["structure"])
        mfe = r.get("mfe")
        aa_len = r.get("aa_len")
        parts = []
        if aa_len is not None:
            parts.append(f"AA{aa_len}")
        parts += [f"{stats['length']} nt", f"{stats['pairs']} bp"]
        if r["pk"]:
            parts.append(f"{stats['pk_pairs']} PK-bp")
        if mfe is not None:
            parts.append(f"MFE {mfe:.1f}")
        entries.append({
            "aa_len": aa_len if aa_len is not None else 0,
            "idx": r["idx"],
            "sequence": r["sequence"],
            "structure": r["structure"],
            "title": f"idx {r['idx']}",
            "meta": " · ".join(parts),
            "pk": r["pk"],
            "bp": stats["pairs"],
            "pk_pairs": stats["pk_pairs"],
            "mfe": mfe,
            "aa": r.get("aa", ""),
        })
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="Path to BPSeq-tokenised .npy dataset")
    src.add_argument("--parquet", help="Path to a per-sample parquet (seq + structure + PK columns)")
    parser.add_argument("--protein-len", type=int, default=20, help="Protein length (npy: token length / 3)")
    parser.add_argument("--aa-len", type=int, default=None, help="parquet: filter to this aa_len")
    parser.add_argument("--seq-col", default="mrna_seq", help="parquet sequence column")
    parser.add_argument("--struct-col", default="knotty_canon_db", help="parquet dot-bracket column")
    parser.add_argument("--pk-col", default="has_pk", help="parquet boolean PK-flag column")
    parser.add_argument("--indices", default="all", help="Row indices: comma-list, 'a:b' slice, or 'all'")
    parser.add_argument("--filter", choices=["none", "pk", "nonpk"], default="none", help="Keep only PK / non-PK rows")
    parser.add_argument("--random", type=int, default=None, metavar="N", help="Randomly sample N rows (after --filter)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for --random")
    parser.add_argument("--out", required=True, help="Output HTML path")
    parser.add_argument("--title", default=None, help="Catalog title")
    parser.add_argument(
        "--prerender", choices=["none", "inline", "img"], default="none",
        help="none: live fornac (browser layout); inline: bake static SVG into one file; "
             "img: bake one .svg per structure + lazy <img> (best for large catalogs)",
    )
    parser.add_argument(
        "--prerender-force", action="store_true",
        help="run + settle the force layout before baking (less cramped for long structures); page stays static",
    )
    parser.add_argument(
        "--prerender-compare-force", action="store_true",
        help="bake BOTH force-off and force-relaxed layouts (lazy <img>) with a button to flip between them",
    )
    parser.add_argument(
        "--prerender-forna", action="store_true",
        help="Option A: render via forna's exact pipeline (NAView coords + forna's own fornac.js force, headless)",
    )
    parser.add_argument(
        "--prerender-hybrid", action="store_true",
        help="Hybrid: forna force (A) for non-PK structures, pure-Python NAView (B) for PK structures",
    )
    parser.add_argument(
        "--forna-jobs", type=int, default=4,
        help="parallel headless-Chrome processes for forna-force rendering (non-PK)",
    )
    parser.add_argument(
        "--fornac-url",
        default="https://unpkg.com/fornac@1.1.8/dist/scripts/fornac.js",
        help="URL (or local path) to fornac.js (live mode only)",
    )
    args = parser.parse_args()

    if args.data:
        rows = load_npy_rows(args.data, args.protein_len)
        source_name = os.path.basename(args.data)
    else:
        rows = load_parquet_rows(args.parquet, args.aa_len, args.seq_col, args.struct_col, args.pk_col)
        source_name = os.path.basename(args.parquet)
        if args.aa_len is not None:
            source_name += f" aa{args.aa_len}"

    total = len(rows)
    rows = select_rows(rows, filt=args.filter, random_n=args.random, seed=args.seed, indices=args.indices)
    entries = finalise_entries(rows)

    n_pk = sum(1 for e in entries if e["pk"])
    title = args.title or f"Secondary Structure Catalog — {source_name} ({len(entries)} entries)"

    if args.prerender_hybrid:
        # Hybrid: forna force (A) for non-PK, NAView (B) for PK.
        svgs = [None] * len(entries)
        nonpk = [k for k, e in enumerate(entries) if not e["pk"]]
        css = ""
        if nonpk:
            css, fsvgs = run_forna_prerender([entries[k] for k in nonpk], jobs=args.forna_jobs)
            for k, s in zip(nonpk, fsvgs):
                svgs[k] = minify_svg(s) if s else ""
        for k, e in enumerate(entries):
            if e["pk"]:
                svgs[k] = naview_svg(e["sequence"], e["structure"])
        n_pk_r = sum(1 for e in entries if e["pk"])
        print(f"  hybrid: {len(nonpk)} non-PK via forna-force, {n_pk_r} PK via NAView")
        if args.prerender == "img":
            out_html = build_img_svg_html(entries, title, css, svgs, args.out)
        else:
            out_html = build_inline_svg_html(entries, title, css, svgs)
    elif args.prerender_forna:
        # Option A: forna-exact (headless forna pipeline)
        css, svgs = run_forna_prerender(entries, jobs=args.forna_jobs)
        svgs = [minify_svg(s) if s else "" for s in svgs]
        if args.prerender == "img":
            out_html = build_img_svg_html(entries, title, css, svgs, args.out)
        else:
            out_html = build_inline_svg_html(entries, title, css, svgs)
    elif args.prerender_compare_force:
        # legacy A/B fornac force comparison (headless puppeteer)
        css, svgs_off = run_prerender(entries, force=False)
        _, svgs_on = run_prerender(entries, force=True)
        svgs_off = [minify_svg(s) if s else "" for s in svgs_off]
        svgs_on = [minify_svg(s) if s else "" for s in svgs_on]
        out_html = build_img_compare_html(entries, title, css, svgs_off, svgs_on, args.out)
    elif args.prerender == "none":
        out_html = build_live_html(entries, title, args.fornac_url)
    else:
        # NAView layout (ViennaRNA) — deterministic, crossing-free for nested structures;
        # styling is inline in the SVG, so no external CSS is needed.
        svgs = [naview_svg(e["sequence"], e["structure"]) for e in entries]
        if args.prerender == "inline":
            out_html = build_inline_svg_html(entries, title, "", svgs)
        else:
            out_html = build_img_svg_html(entries, title, "", svgs, args.out)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(out_html)

    mode = "compare-force" if args.prerender_compare_force else args.prerender
    print(f"Wrote {args.out}: {len(entries)}/{total} entries selected ({n_pk} PK, {len(entries) - n_pk} non-PK)"
          f" · mode={mode}")


if __name__ == "__main__":
    main()
