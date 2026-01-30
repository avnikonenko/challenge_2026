#!/usr/bin/env python3
"""
Chemprop D-MPNN ensemble trainer and predictor.

Trains 5 seeded models with scaffold splits, reports precision@k and PR-AUC on CV,
and averages ensemble probabilities to rank the blind set.

Requires `chemprop` Python package and CLI (`chemprop_train`, `chemprop_predict`).
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from chem_utils import (
    bedroc_score,
    ef_at_k,
    ef_at_percent,
    load_config,
    precision_at_k,
    seed_everything,
    normalize_activity,
)
from split_choice import load_split_mode, make_split_indices


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print(f"[chemprop] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd))


def prepare_csv(df: pd.DataFrame, smiles_col: str, status_col: str, path: Path) -> None:
    out = df[[smiles_col, status_col]].copy()
    out.columns = ["smiles", "target"]
    out.to_csv(path, index=False)


def train_fold(
    train_csv: Path,
    val_csv: Path,
    save_dir: Path,
    seed: int,
    use_rdkit_desc: bool,
    morgan_radius: int,
    morgan_bits: int,
) -> Path:
    cmd = [
        "chemprop_train",
        "--data_path",
        str(train_csv),
        "--separate_val_path",
        str(val_csv),
        "--dataset_type",
        "classification",
        "--save_dir",
        str(save_dir),
        "--metric",
        "prc-auc",
        "--extra_metrics",
        "roc-auc",
        "--split_type",
        "scaffold",
        "--epochs",
        "40",
        "--batch_size",
        "64",
        "--ensemble_size",
        "1",
        "--seed",
        str(seed),
        "--quiet",
        "--features_generator",
        "morgan",
        "--morgan_radius",
        str(morgan_radius),
        "--morgan_num_bits",
        str(morgan_bits),
    ]
    if use_rdkit_desc:
        # Append RDKit 2D features in addition to Morgan bits
        cmd.extend(["--features_generator", "rdkit_2d_normalized"])
    run_cmd(cmd, save_dir)
    # Return checkpoint path
    # Chemprop versions differ in checkpoint layout; search recursively.
    ckpt = next(save_dir.rglob("*.pt"), None)
    if ckpt is None:
        raise RuntimeError(f"No checkpoint found in {save_dir}")
    return ckpt


def predict(
    checkpoint: Path,
    data_csv: Path,
    out_csv: Path,
    use_rdkit_desc: bool,
    morgan_radius: int,
    morgan_bits: int,
) -> None:
    cmd = [
        "chemprop_predict",
        "--checkpoint_path",
        str(checkpoint),
        "--test_path",
        str(data_csv),
        "--preds_path",
        str(out_csv),
        "--quiet",
        "--features_generator",
        "morgan",
        "--morgan_radius",
        str(morgan_radius),
        "--morgan_num_bits",
        str(morgan_bits),
    ]
    if use_rdkit_desc:
        cmd.extend(["--features_generator", "rdkit_2d_normalized"])
    run_cmd(cmd, out_csv.parent)


def evaluate_fold(preds: pd.DataFrame, labels: pd.Series, eval_topk: List[int], bedroc_alpha: float) -> Dict[str, float]:
    y_true = labels.to_numpy()
    y_score = preds["prediction"].to_numpy()
    metrics: Dict[str, float] = {}
    # PR-AUC
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        metrics["pr_auc"] = float("nan")
        metrics["roc_auc"] = float("nan")

    for k in eval_topk:
        metrics[f"precision@{k}"] = precision_at_k(y_true, y_score, k)
        metrics[f"ef@{k}"] = ef_at_k(y_true, y_score, k)
    metrics["ef@100"] = ef_at_k(y_true, y_score, 100)
    metrics["ef1%"] = ef_at_percent(y_true, y_score, 0.01)
    metrics["ef2%"] = ef_at_percent(y_true, y_score, 0.02)
    metrics["ef5%"] = ef_at_percent(y_true, y_score, 0.05)
    metrics["bedroc"] = bedroc_score(y_true, y_score, alpha=bedroc_alpha)
    return metrics


def ensemble_predict(
    checkpoints: List[Path],
    blind_csv: Path,
    use_rdkit_desc: bool,
    morgan_radius: int,
    morgan_bits: int,
) -> pd.DataFrame:
    all_preds = []
    for ckpt in checkpoints:
        tmp_path = blind_csv.parent / f"pred_{ckpt.stem}.csv"
        predict(ckpt, blind_csv, tmp_path, use_rdkit_desc, morgan_radius, morgan_bits)
        df = pd.read_csv(tmp_path)
        all_preds.append(df["prediction"].to_numpy())
    mean_probs = np.mean(np.vstack(all_preds), axis=0)
    result = pd.read_csv(blind_csv).copy()
    result["score"] = mean_probs
    return result


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    seed = cfg.get("random_seed", 42)
    seed_everything(seed)

    smiles_col = cfg.get("smiles_col", "smiles")
    status_col = cfg.get("status_col", "act")
    name_col = cfg.get("name_col", "name")
    # Ensure consistency with split_mode (uses config_used["y_col"])
    split_y_col = None
    eval_topk = cfg.get("eval_topk", [20, 100])
    bedroc_alpha = cfg.get("bedroc_alpha", 20.0)
    out_dir = Path(cfg.get("output_dir", "outputs"))
    chemprop_dir = out_dir / "chemprop"
    preds_dir = out_dir / "predictions"
    metrics_dir = out_dir / "metrics"
    for d in [chemprop_dir, preds_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    use_rdkit_desc = cfg.get("chemprop_use_rdkit_desc", False)
    ensemble_size = cfg.get("chemprop_ensemble", 5)
    morgan_radius = cfg.get("chemprop_morgan_radius", 2)
    morgan_bits = cfg.get("chemprop_morgan_bits", 2048)
    split_mode_json = cfg.get("split_mode_json", None)
    if split_mode_json and not os.path.exists(split_mode_json):
        raise FileNotFoundError(f"split_mode_json specified but not found: {split_mode_json}")
    split_mode = load_split_mode(split_mode_json) if split_mode_json else None

    known_path = cfg["known_path"]
    blind_path = cfg["blind_path"]

    if known_path.endswith(".smi"):
        known_df = pd.read_csv(known_path, sep="\t", header=None, names=[smiles_col, name_col, status_col])
    else:
        sep = "\t" if known_path.endswith(".tsv") else ","
        known_df = pd.read_csv(known_path, sep=sep)
    known_df = known_df.dropna(subset=[smiles_col, status_col]).copy()
    known_df[status_col] = normalize_activity(known_df[status_col])
    known_df[name_col] = known_df[name_col] if name_col in known_df.columns else known_df.index.astype(str)
    known_df["target"] = (known_df[status_col] == "active").astype(int)

    if blind_path.endswith(".smi"):
        blind_df = pd.read_csv(blind_path, sep="\t", header=None, names=["smiles", name_col])
    else:
        sep_blind = "\t" if blind_path.endswith(".tsv") else ","
        blind_df = pd.read_csv(blind_path, sep=sep_blind)
    if blind_df[name_col].isna().all():
        blind_df[name_col] = blind_df.index.astype(str)

    if split_mode:
        print(f"[chemprop] Using split strategy: {split_mode['chosen_strategy']}")
        split_cfg = split_mode.get("config_used", {})
        split_y_col = split_cfg.get("y_col", status_col)
        # If split config used a different y_col, derive it on the fly
        if split_y_col not in known_df.columns and split_y_col != status_col:
            known_df[split_y_col] = known_df[status_col]
        train_idx, val_idx = make_split_indices(known_df, split_mode["chosen_strategy"], split_cfg, seed)
    else:
        from sklearn.model_selection import train_test_split
        print("[chemprop] split_mode_json not provided; using stratified random split.")
        train_idx, val_idx = train_test_split(
            np.arange(len(known_df)), test_size=0.2, random_state=seed, stratify=known_df["target"]
        )

    fold_metrics = []
    split_checkpoints: List[Path] = []

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        # Prepare blind CSV for prediction
        blind_csv = tmpdir / "blind.csv"
        blind_df[["smiles"]].to_csv(blind_csv, index=False)

        fold_id = 0
        fold_dir = chemprop_dir / "split_eval" / f"seed_{seed}"
        if fold_dir.exists():
            # Avoid stale checkpoints from prior runs
            for old in fold_dir.rglob("*.pt"):
                old.unlink()
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_csv = tmpdir / "train_split.csv"
        val_csv = tmpdir / "val_split.csv"
        # Always train Chemprop on numeric target column
        prepare_csv(known_df.iloc[train_idx], smiles_col, "target", train_csv)
        prepare_csv(known_df.iloc[val_idx], smiles_col, "target", val_csv)

        ckpt = train_fold(
            train_csv,
            val_csv,
            fold_dir,
            seed + fold_id,
            use_rdkit_desc,
            morgan_radius,
            morgan_bits,
        )
        split_checkpoints.append(ckpt)

        pred_csv = tmpdir / "valpred_split.csv"
        predict(ckpt, val_csv, pred_csv, use_rdkit_desc, morgan_radius, morgan_bits)
        preds = pd.read_csv(pred_csv)
        metrics = evaluate_fold(preds, known_df.iloc[val_idx]["target"], eval_topk, bedroc_alpha)
        metrics["split_strategy"] = split_mode["chosen_strategy"] if split_mode else "random_stratified"
        fold_metrics.append(metrics)

        metrics_df = pd.DataFrame(fold_metrics)
        mean_metrics = metrics_df.mean(numeric_only=True).to_dict()
        metrics_path = metrics_dir / "chemprop_cv.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(mean_metrics, f, indent=2)
        print(f"[chemprop] Split metrics -> {metrics_path}")

        # Retrain ensemble on full labeled data for final scoring
        full_dir = chemprop_dir / "full_train"
        if full_dir.exists():
            for old in full_dir.rglob("*.pt"):
                old.unlink()
        full_dir.mkdir(parents=True, exist_ok=True)
        full_train_csv = tmpdir / "train_full.csv"
        prepare_csv(known_df, smiles_col, "target", full_train_csv)

        full_ckpts: List[Path] = []
        for i in range(ensemble_size):
            seed_i = seed + 100 + i  # offset seeds for diversity
            cmd = [
                "chemprop_train",
                "--data_path",
                str(full_train_csv),
                "--dataset_type",
                "classification",
                "--save_dir",
                str(full_dir / f"ensemble_{i}"),
                "--metric",
                "prc-auc",
                "--extra_metrics",
                "roc-auc",
                "--split_type",
                "scaffold",
                "--split_sizes",
                "1.0",
                "0.0",
                "0.0",
                "--epochs",
                "40",
                "--batch_size",
                "64",
                "--ensemble_size",
                "1",
                "--seed",
                str(seed_i),
                "--quiet",
                "--features_generator",
                "morgan",
                "--morgan_radius",
                str(morgan_radius),
                "--morgan_num_bits",
                str(morgan_bits),
            ]
            if use_rdkit_desc:
                cmd.extend(["--features_generator", "rdkit_2d_normalized"])
            run_cmd(cmd, full_dir)
            ckpt_path = next((full_dir / f"ensemble_{i}").rglob("*.pt"), None)
            if ckpt_path:
                full_ckpts.append(ckpt_path)

        if not full_ckpts:
            # Fallback to original checkpoint if full retrain failed
            full_ckpts = split_checkpoints[:ensemble_size]

        pred_df = ensemble_predict(full_ckpts, blind_csv, use_rdkit_desc, morgan_radius, morgan_bits)
        pred_df[name_col] = blind_df[name_col].values
        pred_df = pred_df.sort_values(by="score", ascending=False).reset_index(drop=True)
        pred_df.insert(0, "rank", pred_df.index + 1)
        out_pred_path = preds_dir / "chemprop_blind_ranked.csv"
        pred_df.to_csv(out_pred_path, index=False)
        print(f"[chemprop] Blind rankings -> {out_pred_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()
    main(args.config)
