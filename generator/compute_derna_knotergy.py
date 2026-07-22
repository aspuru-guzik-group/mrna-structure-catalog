#!/usr/bin/env python3
"""Compute Knotergy (Dirks-Pierce09) energy of DeRNA's initial fold for each structure.

Portable: no repo imports. Mirrors src/rewards.py::RewardsCalculator.knotergy /
tests/test_knotergy_self_consistency.py::knotergy — runs the `Knotergy` binary with
`"{seq} {db} -p {params}"` and parses the energy from stdout.

Usage:
  python3 compute_derna_knotergy.py --in aa20_derna_input.json --out aa20_derna_knotergy.json \
      --params /project/yuma/mRNA/mrna_transformer/params/common/rna_DirksPierce09.par
"""
import argparse
import json
import re
import subprocess
import sys
import time

# Match the production parser (src/rewards.py): the labelled ENERGY line.
_KG_RE = re.compile(r"ENERGY:\s*(-?\d+\.\d+)\s*kcal/mol")


def knotergy(seq: str, db: str, params: str) -> float:
    """Energy of dot-bracket `db` on `seq` under Knotergy 0.1.1 (CLI: -s/-r/-p)."""
    if len(seq) != len(db):
        raise ValueError(f"len(seq)={len(seq)} != len(db)={len(db)}")
    proc = subprocess.run(
        ["Knotergy", "-s", seq, "-r", db, "-p", params],
        capture_output=True, text=True, check=True, encoding="utf-8",
    )
    m = _KG_RE.search(proc.stdout)
    if m is None:
        raise RuntimeError(f"could not parse Knotergy output:\n{proc.stdout}\n{proc.stderr}")
    return float(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--params", required=True)
    args = ap.parse_args()

    rows = json.load(open(args.inp))
    out = {}
    errors = {}
    t0 = time.time()
    for i, r in enumerate(rows):
        try:
            out[str(r["idx"])] = knotergy(r["seq"], r["db"], args.params)
        except Exception as exc:  # keep going; record failures
            errors[str(r["idx"])] = f"{type(exc).__name__}: {exc}"[:200]
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(rows)}] {time.time()-t0:.1f}s", flush=True)
    json.dump({"energies": out, "errors": errors,
               "n": len(rows), "n_ok": len(out), "n_err": len(errors),
               "wall_s": time.time() - t0, "params": args.params},
              open(args.out, "w"), indent=0)
    print(f"[done] {len(out)}/{len(rows)} ok, {len(errors)} errors, "
          f"{time.time()-t0:.1f}s -> {args.out}", flush=True)
    if errors:
        print("[warn] first errors:", dict(list(errors.items())[:3]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
