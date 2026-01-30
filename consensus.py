#!/usr/bin/env python3
"""
Aggregate per-model rankings and compute Exponential Consensus Ranking (ECR).

Inputs:
- All per-model ranked CSVs in outputs/predictions/ (e.g., rf_morgan_blind_ranked.csv, chemprop_blind_ranked.csv)
- Similarity ranking in outputs/similarity/blind_similarity_ranked.csv (if present)

Outputs:
- outputs/consensus/all_model_rankings.csv : stacked per-model ranks with model + params + scaffold
- outputs/consensus/ecr_consensus.csv : consensus score/rank over all available models
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from chem_utils import bemis_murcko_scaffold, ensure_dir, load_config


def load_prediction_file(path: Path, model_name: str, params: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expect at least rank, smiles, and score columns; tolerate name column variations.
    col_smiles = "smiles"
    if col_smiles not in df.columns:
        raise ValueError(f"{path} missing smiles column.")
    col_name = "name" if "name" in df.columns else None
    col_score = "score" if "score" in df.columns else None
    col_rank = "rank" if "rank" in df.columns else None

    df_out = pd.DataFrame()
    if col_rank in df.columns:
        df_out["rank"] = df[col_rank].astype(int)
    else:
        df_out["rank"] = np.arange(1, len(df) + 1)
    df_out["smiles"] = df[col_smiles]
    df_out["name"] = df[col_name] if col_name else np.arange(len(df))
    df_out["score"] = df[col_score] if col_score else np.nan
    df_out["model"] = model_name
    df_out["params"] = params
    return df_out


def add_scaffolds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["scaffold"] = df["smiles"].apply(bemis_murcko_scaffold)
    return df


def compute_ecr(df: pd.DataFrame, tau: float, weights: Dict[str, float], focus_k: int) -> pd.DataFrame:
    # Exponential consensus: weighted by per-model validation strength; optional focus on top-k ranks only.
    grouped = df.groupby("smiles")
    # For stable naming/scaffolds, pick first occurrence.
    names = grouped["name"].first()
    scaffolds = grouped["scaffold"].first()
    ranks = grouped["rank"].apply(list)

    scores = []
    models = grouped["model"].apply(list)

    for smi, rank_list in ranks.items():
        arr = np.array(rank_list, dtype=float)
        mlist = models[smi]
        # Ignore contributions beyond focus_k
        mask = arr <= focus_k
        if not mask.any():
            scores.append((smi, 0.0))
            continue
        arr = arr[mask]
        mlist = np.array(mlist)[mask]
        w = np.array([weights.get(m, 1.0) for m in mlist], dtype=float)
        num = np.sum(w * np.exp(-(arr - 1) / tau))
        denom = np.sum(w) if np.sum(w) > 0 else len(arr)
        ecr_score = num / denom
        scores.append((smi, ecr_score))

    score_df = pd.DataFrame(scores, columns=["smiles", "ecr_score"])
    score_df["name"] = score_df["smiles"].map(names)
    score_df["scaffold"] = score_df["smiles"].map(scaffolds)
    score_df = score_df.sort_values(by="ecr_score", ascending=False).reset_index(drop=True)
    score_df.insert(0, "rank", score_df.index + 1)
    return score_df


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    output_dir = Path(cfg.get("output_dir", "outputs"))
    preds_dir = output_dir / "predictions"
    sim_file = output_dir / "similarity" / "blind_similarity_ranked.csv"
    cons_dir = output_dir / "consensus"
    ensure_dir(cons_dir)
    tau = float(cfg.get("consensus_tau", 50.0))
    focus_k = int(cfg.get("consensus_focus_k", 100))
    metric_key = cfg.get("consensus_metric", "precision@100")
    weight_fallback = float(cfg.get("consensus_weight_fallback", 1.0))
    # Optional robustness weight using secondary split metrics if present
    robustness_suffix = cfg.get("secondary_split_suffix", "_cluster_t0.60")
    # Preference for calibrated models in OOD (low similarity) regimes
    split_mode_path = cfg.get("split_mode_json", None)
    ood_pref_factor = 1.0
    if split_mode_path and Path(split_mode_path).exists():
        try:
            split_info = load_config(split_mode_path)
            t2b = split_info.get("report", {}).get("train_to_blind", {})
            q50 = float(t2b.get("q50", 1.0))
            # If median train→blind similarity is low, treat as OOD and favor calibrated models.
            if q50 < 0.4:
                ood_pref_factor = 1.15
            elif q50 < 0.55:
                ood_pref_factor = 1.05
        except Exception:
            ood_pref_factor = 1.0
    calibrated_models = {"rf", "xgb", "lgbm", "hgb"}

    frames: List[pd.DataFrame] = []

    model_weights: Dict[str, float] = {}

    # Load model prediction files
    for path_str in glob.glob(str(preds_dir / "*_blind_ranked.csv")):
        path = Path(path_str)
        model_name = path.stem.replace("_blind_ranked", "")
        params = model_name
        frames.append(load_prediction_file(path, model_name=model_name, params=params))
        # Load validation weight
        metrics_path = output_dir / "metrics" / f"{model_name}_cv.json"
        weight = weight_fallback
        base_metric = None
        if metrics_path.exists():
            try:
                m = load_config(metrics_path)
                if "precision@20" in m and "precision@100" in m:
                    base_metric = float((m["precision@20"] + m["precision@100"]) / 2.0)
                elif "ef@100" in m:
                    base_metric = float(m["ef@100"])
                else:
                    base_metric = float(m.get(metric_key, m.get("pr_auc", weight_fallback)))
            except Exception:
                base_metric = None
        elif model_name == "chemprop":
            metrics_path = output_dir / "metrics" / "chemprop_cv.json"
            if metrics_path.exists():
                try:
                    m = load_config(metrics_path)
                    if "precision@20" in m and "precision@100" in m:
                        base_metric = float((m["precision@20"] + m["precision@100"]) / 2.0)
                    elif "ef@100" in m:
                        base_metric = float(m["ef@100"])
                    else:
                        base_metric = float(m.get(metric_key, m.get("pr_auc", weight_fallback)))
                except Exception:
                    base_metric = None
        # Secondary robustness metric (e.g., cluster_t0.60 CV)
        secondary_path = output_dir / "metrics" / f"{model_name}{robustness_suffix}_cv.json"
        robustness_metric = None
        if secondary_path.exists():
            try:
                m2 = load_config(secondary_path)
                if "precision@20" in m2 and "precision@100" in m2:
                    robustness_metric = float((m2["precision@20"] + m2["precision@100"]) / 2.0)
                elif "ef@100" in m2:
                    robustness_metric = float(m2["ef@100"])
                else:
                    robustness_metric = float(m2.get(metric_key, m2.get("pr_auc", weight_fallback)))
            except Exception:
                robustness_metric = None

        if base_metric is not None and robustness_metric is not None:
            weight = 0.6 * base_metric + 0.4 * robustness_metric
        elif base_metric is not None:
            weight = base_metric
        elif robustness_metric is not None:
            weight = robustness_metric
        # Calibrated preference in OOD: boost calibrated models, slightly damp others.
        model_prefix = model_name.split("_")[0]
        if ood_pref_factor != 1.0:
            if model_prefix in calibrated_models:
                weight *= ood_pref_factor
            else:
                weight *= max(0.85, 2 - ood_pref_factor)  # light penalty
        model_weights[model_name] = max(weight, 1e-6)

    # Similarity ranking intentionally excluded from consensus per request

    if not frames:
        raise RuntimeError("No prediction files found to build consensus.")

    all_preds = pd.concat(frames, ignore_index=True)
    used_models = sorted(model_weights.keys())
    all_preds = add_scaffolds(all_preds)

    all_path = cons_dir / "all_model_rankings.csv"
    all_preds.to_csv(all_path, index=False)
    print(f"[consensus] Wrote all model rankings -> {all_path}")

    ecr_df = compute_ecr(all_preds, tau=tau, weights=model_weights, focus_k=focus_k)
    ecr_path = cons_dir / "ecr_consensus.csv"
    ecr_df.to_csv(ecr_path, index=False)
    print(f"[consensus] Wrote ECR consensus -> {ecr_path}")
    print(f"[consensus] Models included: {', '.join(used_models)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()
    main(args.config)
