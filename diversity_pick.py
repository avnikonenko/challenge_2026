#!/usr/bin/env python3
"""
Stage 3: Diversity-aware top-100 selection from Stage 2 predictions.

Workflow:
1) Read prediction CSV (default: outputs/consensus/ecr_consensus.csv).
2) Determine diversity cap per cluster based on train→blind similarity from split_mode.json.
3) Take a candidate pool (top N by score), cluster by Morgan/ECFP (Butina), and
   pick up to `cap_per_cluster` per cluster until 100 molecules are selected.
4) Write a final diversified ranking CSV.
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina


def load_split_mode(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def regime_from_shift(report: dict) -> str:
    t2b = report.get("train_to_blind", {})
    q50 = t2b.get("q50", 0)
    p07 = t2b.get("p_ge_0.7", 0)
    if q50 >= 0.6 or p07 >= 0.2:
        return "analog"
    if q50 <= 0.4 and p07 <= 0.1:
        return "novel"
    return "balanced"


def compute_cap(regime: str, cap_default: int, cap_high: int, cap_low: int) -> int:
    if regime == "analog":
        return cap_high
    if regime == "novel":
        return cap_low
    return cap_default


def build_fps(smiles: List[str], radius: int = 2, n_bits: int = 2048):
    fps = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps.append(fp)
    return fps


def butina_clusters(fps: List[DataStructs.ExplicitBitVect], cutoff: float):
    # Prepare distance matrix (1 - tanimoto)
    nfps = len(fps)
    dists = []
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])
    clusters = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    cluster_ids = np.empty(nfps, dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            cluster_ids[m] = cid
    return cluster_ids


def ensure_capacity(cluster_sizes, cap, target):
    while sum(min(cap, sz) for sz in cluster_sizes.values()) < target:
        cap += 5
        if cap > max(cluster_sizes.values()) + 5:
            break
    return cap


def select_diverse(df: pd.DataFrame, cluster_ids: np.ndarray, cap: int, target: int):
    df = df.copy()
    df["cluster_id"] = cluster_ids
    cluster_sizes = df["cluster_id"].value_counts().to_dict()
    cap = ensure_capacity(cluster_sizes, cap, target)

    counts = {cid: 0 for cid in cluster_sizes}
    picked_rows = []
    for _, row in df.sort_values("score", ascending=False).iterrows():
        cid = row["cluster_id"]
        if counts[cid] >= cap:
            continue
        picked_rows.append(row)
        counts[cid] += 1
        if len(picked_rows) >= target:
            break

    # If still short, relax cap fully
    if len(picked_rows) < target:
        remaining = df[~df.index.isin([r.name for r in picked_rows])].sort_values("score", ascending=False)
        for _, row in remaining.iterrows():
            picked_rows.append(row)
            if len(picked_rows) >= target:
                break

    out_df = pd.DataFrame(picked_rows).reset_index(drop=True)
    out_df.insert(0, "rank_diverse", out_df.index + 1)
    out_df["cap_used"] = cap
    out_df["cluster_size"] = out_df["cluster_id"].map(cluster_sizes)
    return out_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", default="outputs/consensus/ecr_consensus.csv")
    parser.add_argument("--split_mode_json", default="split_mode.json")
    parser.add_argument("--out_csv", default="outputs/consensus/diverse_top100.csv")
    parser.add_argument("--pool_size", type=int, default=500)
    parser.add_argument("--target", type=int, default=100)
    parser.add_argument("--butina_cutoff", type=float, default=0.3, help="Distance cutoff => 1-Tanimoto; 0.3 ~ T=0.7")
    parser.add_argument("--cap_default", type=int, default=15)
    parser.add_argument("--cap_high", type=int, default=25)
    parser.add_argument("--cap_low", type=int, default=10)
    args = parser.parse_args()

    pred_df = pd.read_csv(args.pred_csv)
    smiles_col = "smiles" if "smiles" in pred_df.columns else pred_df.columns[1]
    score_col = "score" if "score" in pred_df.columns else pred_df.columns[-1]

    pool = pred_df.sort_values(score_col, ascending=False).head(args.pool_size).reset_index(drop=True)
    fps = build_fps(pool[smiles_col].tolist())
    good_idx = [i for i, fp in enumerate(fps) if fp is not None]
    if len(good_idx) < args.target:
        raise RuntimeError("Not enough valid fingerprints to select targets.")

    # Filter to valid fps only
    pool_valid = pool.iloc[good_idx].reset_index(drop=True)
    fps_valid = [fps[i] for i in good_idx]
    cluster_ids = butina_clusters(fps_valid, args.butina_cutoff)

    split_mode = load_split_mode(Path(args.split_mode_json))
    regime = regime_from_shift(split_mode.get("report", {}))
    cap = compute_cap(regime, args.cap_default, args.cap_high, args.cap_low)

    selected = select_diverse(pool_valid.assign(score=pool_valid[score_col]), cluster_ids, cap, args.target)
    selected["regime"] = regime
    selected["strategy"] = split_mode.get("chosen_strategy", "unknown")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_path, index=False)

    print(f"[diversity_pick] regime={regime}, cap={cap}, pool={len(pool_valid)}, clusters={len(set(cluster_ids))}")
    print(f"[diversity_pick] wrote {len(selected)} rows -> {out_path}")


if __name__ == "__main__":
    main()
