"""
Similarity-based prioritization.

Computes Tanimoto similarity between each blind molecule and all known actives,
reports max similarity and average of top-k (k from config), and outputs a ranked file.

Usage:
    python similarity_rank.py --config config.json
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from chem_utils import (
    FeaturizationConfig,
    ensure_dir,
    load_config,
    mol_from_smiles,
    morgan_fingerprint,
    seed_everything,
    tanimoto_similarity,
)


def read_known_actives(path: str, smiles_col: str, status_col: str) -> pd.DataFrame:
    sep = "\t" if path.endswith(".tsv") else ","
    df = pd.read_csv(path, sep=sep)
    df = df.dropna(subset=[smiles_col, status_col]).copy()
    df[status_col] = df[status_col].str.lower().str.strip()
    actives = df[df[status_col] == "active"].reset_index(drop=True)
    return actives


def read_blind(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=["smiles", "name"])
    df = df.dropna(subset=["smiles"]).copy()
    if "name" not in df.columns or df["name"].isna().all():
        df["name"] = df.index.astype(str)
    else:
        df["name"] = df["name"].fillna(df.index.astype(str))
    return df


def prepare_active_fps(actives: pd.DataFrame, cfg: FeaturizationConfig, smiles_col: str):
    active_rdkit_fps = []
    for smi in actives[smiles_col]:
        mol = mol_from_smiles(smi)
        _, fp = morgan_fingerprint(mol, cfg)
        if fp is not None:
            active_rdkit_fps.append(fp)
    return active_rdkit_fps


def compute_similarity_metrics(
    blind_df: pd.DataFrame,
    active_fps: List,
    cfg: FeaturizationConfig,
    top_k: List[int],
) -> pd.DataFrame:
    records = []
    for smi, name in zip(blind_df["smiles"], blind_df["name"]):
        mol = mol_from_smiles(smi)
        _, fp = morgan_fingerprint(mol, cfg)
        sims = tanimoto_similarity(fp, active_fps) if active_fps else []
        sims_sorted = sorted(sims, reverse=True)
        max_sim = sims_sorted[0] if sims_sorted else 0.0
        rec: Dict[str, float] = {
            "name": name,
            "smiles": smi,
            "max_tanimoto": max_sim,
        }
        for k in top_k:
            k_eff = min(k, len(sims_sorted))
            key = f"avg_top{k}"
            rec[key] = float(np.mean(sims_sorted[:k_eff])) if k_eff > 0 else 0.0
        # Ranking score that weights max similarity and the mid-range (top10) average if available.
        rec["rank_score"] = rec["max_tanimoto"] * 0.7 + rec.get("avg_top10", rec["max_tanimoto"]) * 0.3
        records.append(rec)
    df = pd.DataFrame(records)
    df = df.sort_values(by=["rank_score", "max_tanimoto"], ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    return df


def main(config_path: str) -> None:
    cfg_dict = load_config(config_path)
    seed_everything(cfg_dict.get("random_seed", 42))
    feat_cfg = FeaturizationConfig(**cfg_dict.get("fingerprint", {}))
    top_k = cfg_dict.get("top_k", [5, 10, 20])

    known_path = cfg_dict["known_path"]
    blind_path = cfg_dict["blind_path"]
    smiles_col = cfg_dict.get("smiles_col", "smiles")
    status_col = cfg_dict.get("status_col", "status")
    out_dir = cfg_dict.get("output_dir", "outputs")
    sim_dir = os.path.join(out_dir, "similarity")
    ensure_dir(sim_dir)

    actives_df = read_known_actives(known_path, smiles_col, status_col)
    blind_df = read_blind(blind_path)
    active_fps = prepare_active_fps(actives_df, feat_cfg, smiles_col)

    print(f"Computing similarity for {len(blind_df)} blind molecules against {len(active_fps)} actives.")
    sim_df = compute_similarity_metrics(blind_df, active_fps, feat_cfg, top_k)

    out_path = os.path.join(sim_dir, "blind_similarity_ranked.csv")
    sim_df.to_csv(out_path, index=False)
    print(f"Wrote ranked similarity file -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()
    main(args.config)
