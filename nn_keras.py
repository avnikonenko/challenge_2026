#!/usr/bin/env python3
"""
Feed-forward Keras baseline on Morgan fingerprints with enforced split mode.
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC

from chem_utils import (
    load_config,
    ensure_dir,
    precision_at_k,
    enrichment_factor_at_k,
    normalize_activity,
)
from split_choice import load_split_mode, make_split_indices


def compute_class_weight(y):
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    w0 = (n_pos + n_neg) / (2.0 * n_neg)
    w1 = (n_pos + n_neg) / (2.0 * n_pos)
    return {0: w0, 1: w1}


class PrecisionAtKCallback(Callback):
    def __init__(self, x_val, y_val, k=100):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.k = k
        self.best = -1.0
        self.best_weights = None
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred = self.model.predict(self.x_val, verbose=0).ravel()
        idx = np.argsort(-y_pred)[: self.k]
        p_at_k = float(np.mean(self.y_val[idx] == 1))
        logs["val_p100"] = p_at_k
        if p_at_k > self.best:
            self.best = p_at_k
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch + 1
        print(f" — val_p100: {p_at_k:.4f} (best {self.best:.4f})")

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)


def build_model(input_dim, lr=8e-4, drop=0.35, l2_reg=1e-5):
    model = Sequential()
    model.add(Dense(1024, input_dim=input_dim, activation="relu", kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(drop))

    model.add(Dense(512, activation="relu", kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(drop))

    model.add(Dense(256, activation="relu", kernel_regularizer=l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(drop * 0.8))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=lr),
        metrics=[AUC(name="roc_auc"), AUC(curve="PR", name="pr_auc")],
    )
    return model


def load_features(feat_dir: str, known_df: pd.DataFrame):
    morgan_act = pd.read_parquet(os.path.join(feat_dir, "actives_morgan.parquet"))
    morgan_inact = pd.read_parquet(os.path.join(feat_dir, "inactives_morgan.parquet"))
    morgan_blind = pd.read_parquet(os.path.join(feat_dir, "unknown_morgan.parquet"))

    morgan_act["label"] = 1
    morgan_inact["label"] = 0
    merged = pd.concat([morgan_act, morgan_inact], ignore_index=True)

    known_key = known_df[["name", "smiles"]].copy()
    known_key["__order"] = np.arange(len(known_key))
    merged = merged.merge(known_key, on=["name", "smiles"], how="left")
    if merged["__order"].isna().any() or len(merged) != len(known_df):
        missing = merged[merged["__order"].isna()][["name", "smiles"]]
        raise ValueError(
            f"Feature alignment failed: {missing.shape[0]} rows could not be matched by name+smiles."
        )
    merged = merged.sort_values("__order").reset_index(drop=True)
    feat_cols = [c for c in merged.columns if c.startswith("bit_")]

    X = merged[feat_cols].to_numpy(dtype=np.float32)
    y = merged["label"].to_numpy(dtype=np.int8)

    blind_X = morgan_blind[feat_cols].to_numpy(dtype=np.float32)
    blind_meta = morgan_blind[["name", "smiles"]].copy()
    return X, y, blind_X, blind_meta, feat_cols


def train_one_seed(X, y, train_idx, val_idx, seed=42, k=100):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    x_tr, x_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    p100_cb = PrecisionAtKCallback(x_val, y_val, k=k)
    callbacks = [
        p100_cb,
        EarlyStopping(monitor="val_p100", mode="max", patience=20, restore_best_weights=False),
        ReduceLROnPlateau(monitor="val_p100", mode="max", factor=0.5, patience=8, min_lr=1e-6),
    ]
    cw = compute_class_weight(y_tr)
    model = build_model(input_dim=X.shape[1])
    hist = model.fit(
        x_tr,
        y_tr,
        epochs=250,
        batch_size=128,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        class_weight=cw,
        verbose=2,
    )
    best_epoch = p100_cb.best_epoch if p100_cb.best_epoch > 0 else len(hist.history.get("loss", []))
    return model, best_epoch


def train_full_seed(X, y, seed=42, epochs=80):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    cw = compute_class_weight(y)
    model = build_model(input_dim=X.shape[1])
    # Fixed-length training on all data (no holdout) to satisfy full-data retrain.
    model.fit(
        X,
        y,
        epochs=epochs,
        batch_size=128,
        class_weight=cw,
        verbose=2,
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = cfg.get("random_seed", 42)

    split_mode_path = cfg.get("split_mode_json", None)
    if split_mode_path and not os.path.exists(split_mode_path):
        raise FileNotFoundError(f"split_mode_json specified but not found: {split_mode_path}")
    split_mode = load_split_mode(split_mode_path) if split_mode_path else None

    known_path = cfg["known_path"]
    blind_path = cfg["blind_path"]
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
    known_df = known_df.rename(columns={name_col: "name", smiles_col: "smiles"})

    if blind_path.endswith(".smi"):
        blind_df = pd.read_csv(blind_path, sep="\t", header=None, names=["smiles", "name"])
    else:
        sep_blind = "\t" if blind_path.endswith(".tsv") else ","
        blind_df = pd.read_csv(blind_path, sep=sep_blind)
    if blind_df["name"].isna().all():
        blind_df["name"] = blind_df.index.astype(str)

    if split_mode:
        print(f"[keras] Using split strategy: {split_mode['chosen_strategy']}")
        train_idx, val_idx = make_split_indices(known_df, split_mode["chosen_strategy"], split_mode["config_used"], seed)
    else:
        from sklearn.model_selection import train_test_split

        print("[keras] split_mode_json not provided; using stratified random split.")
        train_idx, val_idx = train_test_split(
            np.arange(len(known_df)), test_size=0.2, random_state=seed, stratify=known_df["act"]
        )

    feat_dir = os.path.join(cfg.get("output_dir", "outputs"), "features")
    X, y, blind_X, blind_meta, feat_cols = load_features(feat_dir, known_df)

    seeds = cfg.get("keras_seeds", [1, 2, 3, 4, 5])
    preds = []
    val_metrics = []
    for s in seeds:
        print(f"\n=== Training Keras seed {s} (val for metrics) ===")
        model, best_epoch = train_one_seed(X, y, train_idx, val_idx, seed=s, k=100)
        # evaluate on val split
        val_probs = model.predict(X[val_idx], verbose=0).ravel()
        p100 = precision_at_k(y[val_idx], val_probs, 100)
        ef100 = enrichment_factor_at_k(y[val_idx], val_probs, 100)
        val_metrics.append({"seed": s, "precision@100": p100, "ef@100": ef100, "best_epoch": best_epoch})

        # Retrain on full labeled data for final prediction using best_epoch
        print(f"=== Retraining Keras seed {s} on full data for {best_epoch} epochs ===")
        full_model = train_full_seed(X, y, seed=s, epochs=best_epoch)
        preds.append(full_model.predict(blind_X, verbose=0).ravel())

    mean_val = {k: float(np.mean([m[k] for m in val_metrics])) for k in ["precision@100", "ef@100"]}

    y_pred = np.mean(np.vstack(preds), axis=0)
    pred_df = blind_meta.copy()
    pred_df["score"] = y_pred
    pred_df = pred_df.sort_values(by="score", ascending=False).reset_index(drop=True)
    pred_df.insert(0, "rank", pred_df.index + 1)

    output_dir = Path(cfg.get("output_dir", "outputs"))
    preds_dir = output_dir / "predictions"
    metrics_dir = output_dir / "metrics"
    ensure_dir(preds_dir)
    ensure_dir(metrics_dir)

    pred_path = preds_dir / "keras_morgan_blind_ranked.csv"
    pred_df.to_csv(pred_path, index=False)

    metrics_path = metrics_dir / "keras_morgan_cv.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "precision@100": mean_val["precision@100"],
                "ef@100": mean_val["ef@100"],
                "split_strategy": split_mode["chosen_strategy"] if split_mode else "random_stratified",
                "seeds": seeds,
            },
            f,
            indent=2,
        )

    print(f"[keras] saved predictions -> {pred_path}")
    print(f"[keras] saved metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
