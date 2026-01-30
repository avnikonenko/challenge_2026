#!/usr/bin/env python3
"""
Stage 1 runner: choose split mode based on train/blind similarity shift.
"""

import argparse
import pandas as pd

from split_choice import choose_split_strategy, save_split_mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--blind_csv", required=True)
    parser.add_argument("--out_json", default="split_mode.json")
    parser.add_argument("--smiles_col", default="smiles")
    parser.add_argument("--y_col", default="act")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--random_state_base", type=int, default=42)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    blind_df = pd.read_csv(args.blind_csv)

    config = {
        "smiles_col": args.smiles_col,
        "y_col": args.y_col,
        "test_size": args.test_size,
        "repeats": args.repeats,
        "random_state_base": args.random_state_base,
        "cluster_thresholds": [0.70, 0.60],
        "radius": 2,
        "n_bits": 2048,
    }

    chosen, report = choose_split_strategy(train_df, blind_df, config)

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

    save_split_mode(args.out_json, chosen, report, config)
    print(f"Saved split mode -> {args.out_json}")


if __name__ == "__main__":
    main()
