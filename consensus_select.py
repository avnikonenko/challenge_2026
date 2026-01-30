#!/usr/bin/env python3
"""
Recompute consensus (ECR) over a user-chosen subset of ranked prediction files.

Usage:
    python consensus_select.py --rank_files preds/a.csv preds/b.csv \
        --metrics_dir outputs/metrics --output_dir outputs/consensus_custom

Weights:
    - Uses mean(precision@20, precision@100) if present, else ef@100,
      else `metric_key`, else `weight_fallback`.
    - Optional secondary metrics (e.g., cluster split) via --secondary_suffix.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from consensus import load_prediction_file, add_scaffolds, compute_ecr
from chem_utils import load_config


def load_metrics(model: str, metrics_dir: Path, metric_key: str, fallback: float, secondary_suffix: str) -> float:
    def _extract(path: Path) -> float:
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        if "precision@20" in m and "precision@100" in m:
            return float((m["precision@20"] + m["precision@100"]) / 2.0)
        if "ef@100" in m:
            return float(m["ef@100"])
        return float(m.get(metric_key, m.get("pr_auc", fallback)))

    base_metric = None
    sec_metric = None
    base_path = metrics_dir / f"{model}_cv.json"
    if base_path.exists():
        try:
            base_metric = _extract(base_path)
        except Exception:
            base_metric = None

    if secondary_suffix:
        sec_path = metrics_dir / f"{model}{secondary_suffix}_cv.json"
        if sec_path.exists():
            try:
                sec_metric = _extract(sec_path)
            except Exception:
                sec_metric = None

    if base_metric is not None and sec_metric is not None:
        return 0.6 * base_metric + 0.4 * sec_metric
    if base_metric is not None:
        return base_metric
    if sec_metric is not None:
        return sec_metric
    return fallback


def main():
    parser = argparse.ArgumentParser(description="Recompute consensus for selected prediction files.")
    parser.add_argument("--rank_files", nargs="+", required=True, help="List of ranked prediction CSVs to include.")
    parser.add_argument("--metrics_dir", default="outputs/metrics", help="Directory containing *_cv.json metric files.")
    parser.add_argument("--output_dir", default="outputs/consensus_custom", help="Directory to write consensus outputs.")
    parser.add_argument("--consensus_tau", type=float, default=35.0)
    parser.add_argument("--consensus_focus_k", type=int, default=150)
    parser.add_argument("--metric_key", default="precision@100")
    parser.add_argument("--weight_fallback", type=float, default=1.0)
    parser.add_argument("--secondary_suffix", default="_cluster_t0.60", help="Suffix for secondary split metrics (optional).")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    model_weights: Dict[str, float] = {}

    for path_str in args.rank_files:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Rank file not found: {path}")
        model_name = path.stem.replace("_blind_ranked", "")
        params = model_name
        frames.append(load_prediction_file(path, model_name=model_name, params=params))
        weight = load_metrics(model_name, metrics_dir, args.metric_key, args.weight_fallback, args.secondary_suffix)
        model_weights[model_name] = max(weight, 1e-6)

    if not frames:
        raise RuntimeError("No rank files provided for consensus.")

    all_preds = pd.concat(frames, ignore_index=True)
    all_preds = add_scaffolds(all_preds)

    all_path = out_dir / "all_model_rankings.csv"
    all_preds.to_csv(all_path, index=False)
    print(f"[consensus_select] Wrote all model rankings -> {all_path}")

    ecr_df = compute_ecr(all_preds, tau=args.consensus_tau, weights=model_weights, focus_k=args.consensus_focus_k)
    ecr_path = out_dir / "ecr_consensus.csv"
    ecr_df.to_csv(ecr_path, index=False)
    print(f"[consensus_select] Wrote ECR consensus -> {ecr_path}")
    print(f"[consensus_select] Models included: {', '.join(sorted(model_weights.keys()))}")


if __name__ == "__main__":
    main()
