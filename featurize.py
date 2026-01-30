"""
Feature generation script.

Reads labelled known compounds (CSV/TSV with SMILES + status) and blind set (.smi with
SMILES<TAB>name), computes Morgan fingerprints and 2D descriptors, and writes split
outputs for actives, inactives, and unknowns.

Usage:
    python featurize.py --config config.json
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from chem_utils import (
    FeaturizationConfig,
    compute_descriptors,
    descriptor_names,
    ensure_dir,
    load_config,
    mol_from_smiles,
    morgan_fingerprint,
    seed_everything,
    normalize_activity,
)


def read_known(path: str, smiles_col: str, status_col: str, name_col: str) -> pd.DataFrame:
    """
    Supports headerless .smi files (tab-separated) with columns:
    smi, mol_id, class -> mapped to smiles, name, status.
    Falls back to CSV/TSV with headers.
    """
    if path.endswith(".smi"):
        df = pd.read_csv(path, sep="\t", header=None, names=[smiles_col, name_col, status_col])
    else:
        sep = "\t" if path.endswith(".tsv") else ","
        df = pd.read_csv(path, sep=sep)
    for col in [smiles_col, status_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in known dataset.")
    if name_col not in df.columns:
        df[name_col] = df.index.astype(str)
    # Normalize to a standard name column for downstream scripts.
    if name_col != "name":
        df["name"] = df[name_col]
    df = df.dropna(subset=[smiles_col, status_col]).copy()
    df[status_col] = normalize_activity(df[status_col])
    return df


def read_blind(path: str, smiles_col: str = "smiles", name_col: str = "name") -> pd.DataFrame:
    """
    Supports:
    - headerless .smi (tab): smi, mol_id
    - CSV/TSV with headers containing smiles_col and optional name_col
    """
    if path.endswith(".smi"):
        df = pd.read_csv(path, sep="\t", header=None, names=[smiles_col, name_col])
    else:
        from chem_utils import infer_sep

        sep = infer_sep(path)
        df = pd.read_csv(path, sep=sep)
        if smiles_col not in df.columns:
            raise ValueError(f"Missing required column '{smiles_col}' in blind set.")
        if name_col not in df.columns:
            df[name_col] = df.index.astype(str)
    df = df.dropna(subset=[smiles_col]).copy()
    if name_col not in df.columns or df[name_col].isna().all():
        df[name_col] = df.index.to_series().astype(str)
    else:
        df[name_col] = df[name_col].fillna(df.index.to_series().astype(str))
    # Normalize to standard downstream names
    if name_col != "name":
        df["name"] = df[name_col]
    return df


def fingerprints_df(df: pd.DataFrame, cfg: FeaturizationConfig, smiles_col: str) -> Tuple[pd.DataFrame, List]:
    fps = []
    rdkit_fps = []
    for smi in df[smiles_col]:
        mol = mol_from_smiles(smi)
        arr, fp = morgan_fingerprint(mol, cfg)
        fps.append(arr)
        rdkit_fps.append(fp)
    fp_mat = np.vstack([fp if fp.size else np.zeros((cfg.n_bits,), dtype=np.uint8) for fp in fps])
    fp_cols = [f"bit_{i}" for i in range(cfg.n_bits)]
    fp_df = pd.DataFrame(fp_mat, columns=fp_cols)
    fp_df.insert(0, "smiles", df[smiles_col].values)
    fp_df.insert(1, "name", df["name"].values if "name" in df.columns else df.index.astype(str))
    return fp_df, rdkit_fps


def descriptors_df(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    names = descriptor_names()
    rows = []
    for smi in df[smiles_col]:
        mol = mol_from_smiles(smi)
        rows.append(compute_descriptors(mol, names))
    desc_df = pd.DataFrame(rows)
    desc_df.insert(0, "smiles", df[smiles_col].values)
    desc_df.insert(1, "name", df["name"].values if "name" in df.columns else df.index.astype(str))
    desc_df = desc_df.replace([np.inf, -np.inf], np.nan)
    desc_df = desc_df.fillna(desc_df.median(numeric_only=True))
    return desc_df


def write_split(
    df: pd.DataFrame,
    status: str,
    base_out: str,
    feature_kind: str,
) -> str:
    path_parquet = os.path.join(base_out, f"{status}_{feature_kind}.parquet")
    try:
        df.to_parquet(path_parquet, index=False)
        return path_parquet
    except Exception:
        # Fallback when pyarrow/fastparquet is unavailable.
        path_pkl = os.path.join(base_out, f"{status}_{feature_kind}.pkl")
        df.to_pickle(path_pkl)
        return path_pkl


def _all_feature_files_exist(base_out: str) -> bool:
    """Check if required feature artifacts already exist.

    We only require the Morgan feature files (actives/inactives/unknown) and the schema.
    Descriptor files are optional; if missing we still reuse to avoid needless recompute.
    """
    morgan_bases = [
        "actives_morgan",
        "inactives_morgan",
        "unknown_morgan",
    ]
    for b in morgan_bases:
        parquet = os.path.join(base_out, f"{b}.parquet")
        pkl = os.path.join(base_out, f"{b}.pkl")
        if not (os.path.exists(parquet) or os.path.exists(pkl)):
            return False
    schema = os.path.join(base_out, "feature_schema.json")
    return os.path.exists(schema)


def main(config_path: str) -> None:
    cfg_dict: Dict = load_config(config_path)
    feat_cfg = FeaturizationConfig(**cfg_dict.get("fingerprint", {}))
    seed_everything(cfg_dict.get("random_seed", 42))

    known_path = cfg_dict["known_path"]
    blind_path = cfg_dict["blind_path"]
    smiles_col = cfg_dict.get("smiles_col", "smiles")
    status_col = cfg_dict.get("status_col", "act")
    name_col = cfg_dict.get("name_col", "name")
    out_dir = cfg_dict.get("output_dir", "outputs")
    feat_dir = os.path.join(out_dir, "features")
    ensure_dir(feat_dir)

    # Reuse precomputed features when all files are present.
    if _all_feature_files_exist(feat_dir):
        print(f"[featurize] Existing feature files detected in {feat_dir}; reusing them.")
        return

    known_df = read_known(known_path, smiles_col, status_col, name_col)
    blind_df = read_blind(blind_path, smiles_col=smiles_col, name_col=name_col)

    actives = known_df[known_df[status_col] == "active"].reset_index(drop=True)
    inactives = known_df[known_df[status_col] == "inactive"].reset_index(drop=True)

    print(f"Loaded {len(actives)} actives, {len(inactives)} inactives, {len(blind_df)} unknowns.")

    for subset_name, subset_df in [("actives", actives), ("inactives", inactives), ("unknown", blind_df)]:
        fp_df, _ = fingerprints_df(subset_df, feat_cfg, smiles_col)
        desc_df = descriptors_df(subset_df, smiles_col)
        fp_path = write_split(fp_df, subset_name, feat_dir, "morgan")
        desc_path = write_split(desc_df, subset_name, feat_dir, "descriptors")
        print(f"Wrote {subset_name} morgan -> {fp_path}")
        print(f"Wrote {subset_name} descriptors -> {desc_path}")

    # Persist the featurization schema for downstream scripts.
    schema_path = os.path.join(feat_dir, "feature_schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "morgan_bits": feat_cfg.n_bits,
                "descriptor_names": descriptor_names(),
                "fingerprint_config": feat_cfg.__dict__,
                "smiles_col": smiles_col,
                "name_col": "name",
            },
            f,
            indent=2,
        )
    print(f"Saved schema -> {schema_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()
    main(args.config)
