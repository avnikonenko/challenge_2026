"""
Model training and scoring script.

Trains calibrated classifiers (RandomForest, Gradient Boosting variants) on Morgan
fingerprints and 2D descriptors using scaffold split cross-validation. Reports early
enrichment metrics and produces ranked predictions for the blind set.

Usage:
    python train_models.py --config config.json
"""

import argparse
import os
import json
from copy import deepcopy
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from chem_utils import (
    FeaturizationConfig,
    bedroc_score,
    ef_at_k,
    ef_at_percent,
    ensure_dir,
    load_config,
    precision_at_k,
    seed_everything,
    normalize_activity,
    scaffold_split_indices,
)
from split_choice import load_split_mode, make_split_indices


def load_feature_set(
    feat_dir: str,
    kind: str,
    known_df: pd.DataFrame,
    smiles_col: str,
    name_col: str,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str], np.ndarray, pd.DataFrame]:
    def _read(base: str) -> pd.DataFrame:
        p_parquet = os.path.join(feat_dir, f"{base}_{kind}.parquet")
        p_pkl = os.path.join(feat_dir, f"{base}_{kind}.pkl")
        if os.path.exists(p_parquet):
            return pd.read_parquet(p_parquet)
        if os.path.exists(p_pkl):
            return pd.read_pickle(p_pkl)
        raise FileNotFoundError(f"Missing feature file for {base}_{kind} (.parquet or .pkl)")

    act = _read("actives")
    inact = _read("inactives")
    blind = _read("unknown")

    act = act.copy()
    inact = inact.copy()
    act["label"] = 1
    inact["label"] = 0

    merged = pd.concat([act, inact], ignore_index=True)
    # align to known_df order using smiles + name
    known_key = known_df[[name_col, smiles_col]].copy()
    known_key["__order"] = np.arange(len(known_key))
    merged = merged.merge(
        known_key,
        left_on=["name", "smiles"],
        right_on=[name_col, smiles_col],
        how="left",
    )
    if merged["__order"].isna().any() or len(merged) != len(known_df):
        missing = merged[merged["__order"].isna()][["name", "smiles"]]
        raise ValueError(
            f"Feature alignment failed: {missing.shape[0]} rows could not be matched by name+smiles."
        )
    merged = merged.sort_values("__order").reset_index(drop=True)

    feat_cols = [c for c in merged.columns if c not in {"label", "smiles", "name", name_col, smiles_col, "__order"}]
    X = merged[feat_cols].to_numpy(dtype=np.float32)
    y = merged["label"].to_numpy(dtype=np.int8)
    meta = merged[["name", "smiles", "label"]].copy()

    blind_X = blind[feat_cols].to_numpy(dtype=np.float32)
    blind_meta = blind[["name", "smiles"]].copy()

    return X, y, meta, feat_cols, blind_X, blind_meta


def build_models(seed: int) -> List[Tuple[str, object]]:
    models: List[Tuple[str, object]] = []
    # Random Forest
    models.append(
        (
            "rf",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                n_jobs=-1,
                class_weight="balanced",
                random_state=seed,
            ),
        )
    )

    # Gradient boosting via xgboost / lightgbm if available.
    try:
        import xgboost as xgb  # type: ignore

        models.append(
            (
                "xgb",
                xgb.XGBClassifier(
                    n_estimators=600,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=seed,
                    n_jobs=-1,
                ),
            )
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier  # type: ignore

        models.append(
            (
                "lgbm",
                LGBMClassifier(
                    n_estimators=800,
                    learning_rate=0.05,
                    max_depth=-1,
                    num_leaves=63,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary",
                    n_jobs=-1,
                    random_state=seed,
                ),
            )
        )
    except Exception:
        pass

    # Scikit-learn hist gradient boosting as fallback boosting option.
    models.append(
        (
            "hgb",
            HistGradientBoostingClassifier(
                max_depth=8,
                learning_rate=0.05,
                max_iter=500,
                random_state=seed,
                class_weight={0: 1.0, 1: 2.0},
            ),
        )
    )

    return models


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, eval_topk: List[int], bedroc_alpha: float) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        metrics["pr_auc"] = float("nan")

    # Precision@k
    for k in eval_topk:
        metrics[f"precision@{k}"] = precision_at_k(y_true, y_score, k)
        metrics[f"ef@{k}"] = ef_at_k(y_true, y_score, k)
    # Always record EF@100 for comparability
    metrics["ef@100"] = ef_at_k(y_true, y_score, 100)

    # Early enrichment
    metrics["ef1%"] = ef_at_percent(y_true, y_score, 0.01)
    metrics["ef2%"] = ef_at_percent(y_true, y_score, 0.02)
    metrics["ef5%"] = ef_at_percent(y_true, y_score, 0.05)
    metrics["bedroc"] = bedroc_score(y_true, y_score, alpha=bedroc_alpha)
    return metrics


def train_with_split(
    base_model,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    eval_topk: List[int],
    bedroc_alpha: float,
    blind_X: np.ndarray,
):
    calibrate_method = "isotonic" if len(train_idx) >= 200 else "sigmoid"
    estimator = clone(base_model)
    calibrated = CalibratedClassifierCV(estimator, method=calibrate_method, cv=3, n_jobs=-1)
    calibrated.fit(X[train_idx], y[train_idx])

    # Validation metrics
    if len(val_idx) > 0:
        val_probs = calibrated.predict_proba(X[val_idx])[:, 1]
        metrics = compute_metrics(y[val_idx], val_probs, eval_topk, bedroc_alpha)
    else:
        metrics = {k: float("nan") for k in ["roc_auc", "pr_auc"] + [f"precision@{k}" for k in eval_topk]}

    # Predict blind set
    blind_probs = calibrated.predict_proba(blind_X)[:, 1]

    return metrics, blind_probs, calibrated


def fit_full_model(base_model, X: np.ndarray, y: np.ndarray, blind_X: np.ndarray):
    """
    Retrain on all labeled data to score the blind set.
    """
    calibrate_method = "isotonic" if len(y) >= 200 else "sigmoid"
    estimator = clone(base_model)
    calibrated = CalibratedClassifierCV(estimator, method=calibrate_method, cv=3, n_jobs=-1)
    calibrated.fit(X, y)
    blind_probs = calibrated.predict_proba(blind_X)[:, 1]
    return blind_probs, calibrated


def aggregate_cv_metrics(per_fold_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    metrics = {}
    for key in per_fold_metrics[0].keys():
        vals = [m.get(key, float("nan")) for m in per_fold_metrics]
        metrics[key] = float(np.nanmean(vals))
    return metrics


def save_metrics(path: str, metrics: Dict[str, float]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    seed = cfg.get("random_seed", 42)
    seed_everything(seed)
    feat_cfg = FeaturizationConfig(**cfg.get("fingerprint", {}))
    split_mode_json = cfg.get("split_mode_json", None)
    if split_mode_json and not os.path.exists(split_mode_json):
        raise FileNotFoundError(f"split_mode_json specified but not found: {split_mode_json}")
    split_mode = load_split_mode(split_mode_json) if split_mode_json else None

    output_dir = cfg.get("output_dir", "outputs")
    feat_dir = os.path.join(output_dir, "features")
    pred_dir = os.path.join(output_dir, "predictions")
    metrics_dir = os.path.join(output_dir, "metrics")
    model_dir = os.path.join(output_dir, "models")
    for d in [pred_dir, metrics_dir, model_dir]:
        ensure_dir(d)

    eval_topk = cfg.get("eval_topk", [20, 100])
    bedroc_alpha = cfg.get("bedroc_alpha", 20.0)

    feature_sets = ["morgan", "descriptors"]
    # load original known data for ordering
    known_path = cfg["known_path"]
    smiles_col = cfg.get("smiles_col", "smiles")
    status_col = cfg.get("status_col", "act")
    name_col = cfg.get("name_col", "name")
    if known_path.endswith(".smi"):
        known_df = pd.read_csv(known_path, sep="\t", header=None, names=[smiles_col, name_col, status_col])
    else:
        sep = "\t" if known_path.endswith(".tsv") else ","
        known_df = pd.read_csv(known_path, sep=sep)
    known_df[status_col] = normalize_activity(known_df[status_col])
    known_df[name_col] = known_df[name_col] if name_col in known_df.columns else known_df.index.astype(str)
    known_df["act"] = (known_df[status_col] == "active").astype(int)

    n_folds = int(cfg.get("cv_folds", 5))
    if n_folds < 2:
        n_folds = 1

    for feat_kind in feature_sets:
        X, y, meta, feat_cols, blind_X, blind_meta = load_feature_set(
            feat_dir, feat_kind, known_df, smiles_col, name_col
        )
        smiles = meta["smiles"].tolist()

        # Build CV splits: use chosen split_mode if provided, otherwise scaffold CV.
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        if split_mode:
            cfg_used = split_mode.get("config_used", {})
            strategy = split_mode["chosen_strategy"]
            for r in range(n_folds):
                splits.append(
                    make_split_indices(
                        known_df,
                        strategy,
                        cfg_used,
                        seed + r,
                    )
                )
        else:
            splits = scaffold_split_indices(smiles, n_folds=n_folds, seed=seed)
            if not splits:
                splits = [(np.arange(len(y)), np.array([], dtype=int))]

        for model_tag, base_model in build_models(seed):
            tag = f"{model_tag}_{feat_kind}"
            split_label = split_mode["chosen_strategy"] if split_mode else "scaffold_cv"
            print(f"Training {tag} with {split_label} ({len(splits)} folds)")

            per_fold_metrics: List[Dict[str, float]] = []
            for fold_id, (train_idx, val_idx) in enumerate(splits):
                metrics_fold, _, fitted_model = train_with_split(
                    base_model,
                    X,
                    y,
                    train_idx,
                    val_idx,
                    eval_topk,
                    bedroc_alpha,
                    blind_X,
                )
                metrics_fold["fold"] = fold_id
                per_fold_metrics.append(metrics_fold)

            metrics = aggregate_cv_metrics(per_fold_metrics)
            metrics["split_strategy"] = split_label
            metrics["cv_folds"] = len(splits)

            # Save metrics
            metrics_path = os.path.join(metrics_dir, f"{tag}_cv.json")
            save_metrics(metrics_path, metrics)

            # Save ranked predictions
            pred_df = blind_meta.copy()
            # Retrain on full data for final scoring
            full_blind_probs, full_model = fit_full_model(base_model, X, y, blind_X)
            pred_df["score"] = full_blind_probs
            pred_df = pred_df.sort_values(by="score", ascending=False).reset_index(drop=True)
            pred_df.insert(0, "rank", pred_df.index + 1)
            pred_path = os.path.join(pred_dir, f"{tag}_blind_ranked.csv")
            pred_df.to_csv(pred_path, index=False)

            # Persist calibrated model
            model_path = os.path.join(model_dir, f"{tag}.pkl")
            joblib.dump(
                {
                    "model": full_model,
                    "features": feat_cols,
                    "fingerprint_config": feat_cfg.__dict__,
                    "split_trained_model": fitted_model,
                },
                model_path,
            )

            print(f"Finished {tag}: metrics -> {metrics_path}, predictions -> {pred_path}, model -> {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()
    main(args.config)
