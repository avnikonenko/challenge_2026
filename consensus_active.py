#!/usr/bin/env python3
"""
Consensus based on high-ranking molecules across prediction files.

Collects top-N predictions from all model prediction CSVs in outputs/predictions/,
votes them, and outputs a consensus list ranked by vote count, mean rank, and score.
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def load_pred_file(path: Path, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    col_rank = "rank" if "rank" in df.columns else None
    col_score = "score" if "score" in df.columns else None
    col_smiles = "smiles" if "smiles" in df.columns else df.columns[1]
    col_name = "name" if "name" in df.columns else df.columns[0]

    df = df.copy()
    if col_rank is None:
        df["rank"] = np.arange(1, len(df) + 1)
        col_rank = "rank"
    if col_score is None:
        df["score"] = 1.0 / df[col_rank]  # pseudo-score if missing
        col_score = "score"

    df = df.sort_values(col_rank).head(top_n)
    df["model"] = path.stem
    return df[[col_name, col_smiles, col_rank, col_score, "model"]].rename(
        columns={col_name: "name", col_smiles: "smiles", col_rank: "rank", col_score: "score"}
    )


def consensus(preds: pd.DataFrame) -> pd.DataFrame:
    grouped = preds.groupby(["name", "smiles"])
    votes = grouped["model"].nunique()
    mean_rank = grouped["rank"].mean()
    mean_score = grouped["score"].mean()
    max_score = grouped["score"].max()

    cons = pd.DataFrame(
        {
            "votes": votes,
            "mean_rank": mean_rank,
            "mean_score": mean_score,
            "max_score": max_score,
        }
    ).reset_index()

    cons = cons.sort_values(
        by=["votes", "mean_score", "max_score", "mean_rank"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    cons.insert(0, "consensus_rank", cons.index + 1)
    return cons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", default="outputs/predictions")
    parser.add_argument("--out_csv", default="outputs/consensus/consensus_active.csv")
    parser.add_argument("--top_n", type=int, default=200, help="Top-N per model to consider for voting.")
    parser.add_argument("--top_k", type=int, default=100, help="Limit final consensus to top K rows.")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    files = sorted(pred_dir.glob("*_blind_ranked.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction files found in {pred_dir}")

    frames: List[pd.DataFrame] = [load_pred_file(f, args.top_n) for f in files]
    all_preds = pd.concat(frames, ignore_index=True)
    cons = consensus(all_preds).head(args.top_k)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cons.to_csv(out_path, index=False)
    print(f"[consensus_active] used {len(files)} models; wrote {len(cons)} rows -> {out_path}")


if __name__ == "__main__":
    main()
