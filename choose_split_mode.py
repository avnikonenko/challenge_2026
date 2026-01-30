#!/usr/bin/env python3
"""
Stage 1 runner: choose split mode based on train/blind similarity shift.
"""

import argparse
import numpy as np
import pandas as pd

from split_choice import choose_split_strategy, save_split_mode
from split_choice import build_fps_from_smiles, max_similarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--blind_csv", required=True)
    parser.add_argument("--out_json", default="split_mode.json")
    parser.add_argument("--smiles_col", default="smiles")
    parser.add_argument("--y_col", default="act")
    # Defaults tuned for more stable diagnostics
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--random_state_base", type=int, default=2024)
    args = parser.parse_args()

    if args.train_csv.endswith(".smi"):
        # Headerless .smi expected as: smi, mol_id, class
        train_df = pd.read_csv(args.train_csv, sep="\t", header=None, names=[args.smiles_col, "mol_id", args.y_col])
    else:
        train_df = pd.read_csv(args.train_csv)

    if args.blind_csv.endswith(".smi"):
        blind_df = pd.read_csv(args.blind_csv, sep="\t", header=None, names=[args.smiles_col, "mol_id"])
    else:
        blind_df = pd.read_csv(args.blind_csv)

    config = {
        "smiles_col": args.smiles_col,
        "y_col": args.y_col,
        "test_size": args.test_size,
        "repeats": args.repeats,
        "random_state_base": args.random_state_base,
        "cluster_thresholds": [0.70, 0.50],
        "radius": 2,
        "n_bits": 2048,
    }

    chosen, report = choose_split_strategy(train_df, blind_df, config)

    # Secondary evaluation with cluster_group_t0.60 for robustness (kept alongside primary)
    secondary_strategy = "cluster_group_t0.50"
    report["secondary_strategy"] = secondary_strategy

    # Print diagnostics
    t2b = report["train_to_blind"]
    print("Train→Blind similarity summary:")
    print(f" q10={t2b['q10']:.3f} q50={t2b['q50']:.3f} q90={t2b['q90']:.3f} "
          f"p>=0.7={t2b['p_ge_0.7']:.2f} p>=0.6={t2b['p_ge_0.6']:.2f} p>=0.5={t2b['p_ge_0.5']:.2f}")
    print("\nStrategies:")
    print(" name\tmean_ks\tstd_ks\tvalid_repeats\tavg_val_actives")
    for name, stats in report["per_strategy"].items():
        print(f" {name}\t{stats['mean_ks']:.3f}\t{stats['std_ks']:.3f}\t{stats['valid_repeats']}\t{stats['avg_val_actives']:.1f}")
    print(f"\nChosen: {chosen}")
    print(f"Rationale: {report['rationale']}")

    # Additional diagnostics: blind cluster structure and leakage stats on chosen split
    try:
        # Blind cluster stats using same radius/bits
        blind_fps = build_fps_from_smiles(blind_df[args.smiles_col].tolist(), radius=config["radius"], n_bits=config["n_bits"])
        # Count valid and unique fingerprints
        valid_fps = [fp for fp in blind_fps if fp is not None]
        report["blind_valid_fps"] = int(len(valid_fps))
        try:
            bitstrings = [fp.ToBitString() for fp in valid_fps]
            report["blind_unique_fps"] = int(len(set(bitstrings)))
        except Exception:
            report["blind_unique_fps"] = None
    except Exception:
        report["blind_valid_fps"] = None
        report["blind_unique_fps"] = None

    # Split leakage stats: similarity of val to train for chosen split
    from split_choice import make_split_indices

    train_idx, val_idx = make_split_indices(train_df, chosen, config, args.random_state_base)
    train_fps = build_fps_from_smiles(train_df[args.smiles_col].tolist(), radius=config["radius"], n_bits=config["n_bits"])
    val_fps = [train_fps[i] for i in val_idx]
    train_core = [train_fps[i] for i in train_idx]
    s_val = max_similarity(val_fps, train_core)
    report["leakage_val_to_train"] = {
        "q10": float(np.quantile(s_val, 0.10)) if len(s_val) else 0.0,
        "q50": float(np.quantile(s_val, 0.50)) if len(s_val) else 0.0,
        "q90": float(np.quantile(s_val, 0.90)) if len(s_val) else 0.0,
    }

    save_split_mode(args.out_json, chosen, report, config)
    print(f"Saved split mode -> {args.out_json}")


if __name__ == "__main__":
    main()
