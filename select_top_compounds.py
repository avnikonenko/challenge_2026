#!/usr/bin/env python3
"""
Select top-k ranked compounds from a ranking CSV and write their IDs (one per line)
to a headerless text file named final_output_<suffix>.txt.

Defaults:
- ranking file: outputs/consensus/ecr_consensus.csv
- top_k: 100
- suffix: timestamp-free "latest"
"""

import argparse
from pathlib import Path
import pandas as pd
from chem_utils import bemis_murcko_scaffold


def main():
    parser = argparse.ArgumentParser(description="Export top-ranked compound IDs to plain text.")
    parser.add_argument(
        "--rank_file",
        default="outputs/consensus/ecr_consensus.csv",
        help="Path to ranked CSV (must include name/mol_id column).",
    )
    parser.add_argument("--top_k", type=int, default=100, help="Number of compounds to export.")
    parser.add_argument(
        "--scaffold_cap",
        type=int,
        default=0,
        help="Optional max compounds per Bemis–Murcko scaffold (0 disables).",
    )
    parser.add_argument("--suffix", default="latest", help="Suffix for output filename.")
    parser.add_argument(
        "--output_dir",
        default="outputs",
        help="Directory to place the final output file (created if missing).",
    )
    args = parser.parse_args()

    rank_path = Path(args.rank_file)
    if not rank_path.exists():
        raise FileNotFoundError(f"Ranking file not found: {rank_path}")

    df = pd.read_csv(rank_path)

    # Determine ID column preference.
    id_col = None
    for col in ["mol_id", "name", "compound_id", "id"]:
        if col in df.columns:
            id_col = col
            break
    if id_col is None:
        raise ValueError(f"No ID column found in {rank_path}. Expected one of mol_id/name/compound_id/id.")

    top_k = max(1, min(args.top_k, len(df)))
    df = df.head(top_k).copy()
    if args.scaffold_cap and args.scaffold_cap > 0:
        df["scaffold"] = df["smiles"].apply(bemis_murcko_scaffold) if "smiles" in df.columns else ""
        kept = []
        counts = {}
        for _, row in df.iterrows():
            scaf = row.get("scaffold", "")
            counts.setdefault(scaf, 0)
            if counts[scaf] < args.scaffold_cap:
                kept.append(row[id_col])
                counts[scaf] += 1
        ids = pd.Series(kept, dtype=str)
    else:
        ids = df[id_col].astype(str)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"final_output_{args.suffix}.txt"
    out_path.write_text("\n".join(ids) + "\n", encoding="utf-8")

    print(f"[select_top_compounds] Wrote {top_k} IDs -> {out_path}")


if __name__ == "__main__":
    main()
