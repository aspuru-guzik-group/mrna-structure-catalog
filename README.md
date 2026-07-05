# mRNA Secondary Structure Catalog

Browsable, [forna](http://rna.tbi.univie.ac.at/forna/)-style catalogs of DeRNA+Knotty
mRNA secondary structures (DeRNA-generated sequences, folded with Knotty).

**🔎 Live site:** https://aspuru-guzik-group.github.io/mrna-structure-catalog/

| Catalog | Contents |
|---|---|
| AA=20 (1000) | all 1000 AA=20 structures — 149 PK / 851 non-PK |
| AA=60 (99) | 99 AA=60 structures — 6 PK / 93 non-PK |
| All pseudoknotted | all 823 pseudoknotted structures, AA 4–60 |

Each card shows the structure, a PK / Non-PK badge, `AA-length · nt · bp · PK-bp · MFE`,
and the protein (AA) sequence (toggle-able). Sort by AA length, index, base pairs, MFE, or
PK pairs.

## Rendering
- **Non-pseudoknotted** structures use [forna](http://rna.tbi.univie.ac.at/forna/)'s
  force-directed layout (compact, readable), rendered headlessly into static SVG.
- **Pseudoknotted** structures use ViennaRNA **NAView**, with crossing (pseudoknot) base
  pairs drawn as red links (force layouts tangle pseudoknots).

Pages are served as static HTML; each catalog references a sibling `_assets/` folder of
per-structure SVGs (lazy-loaded via `<img>`).

## Regenerate
Code under [`generator/`](generator/). Requires `ViennaRNA` (NAView) and, for the
forna-force path, Node + a system Chrome (forna's JS/CSS are vendored under
`generator/prerender/forna_vendor/`, so generation is offline). Input is the per-sample
parquet from the DeRNA+Knotty profiling run (issue #47).

```bash
cd generator && (cd prerender && npm install)   # once
PQ=path/to/pk_profile.parquet
python3 structure_catalog.py --parquet $PQ --aa-len 20 \
  --prerender-hybrid --forna-jobs 6 --prerender img --out ../docs/aa20_1000.html
```

Data: `mRNA-project/mrna-transformer-data` (issue #47 profiling run). Model and analysis
code live in the (private) `aspuru-guzik-group/mrna-transformer` repository.
