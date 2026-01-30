#!/usr/bin/env python3
"""
Stage 2 runner: enforce chosen split across the pipeline.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def run_cmd(cmd, cwd):
    print(f"[run_pipeline] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--blind_csv", required=True)
    parser.add_argument("--split_mode_json", required=True)
    parser.add_argument("--config_template", default="config.json")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chemprop", action="store_true", help="run chemprop models")
    parser.add_argument("--keras", action="store_true", help="run keras dense model")
    parser.add_argument("--skip-rf", action="store_true", help="Skip RandomForest model (train_models.py).")
    parser.add_argument("--skip-xgb", action="store_true", help="Skip XGBoost model (train_models.py).")
    parser.add_argument("--skip-lgbm", action="store_true", help="Skip LightGBM model (train_models.py).")
    parser.add_argument("--skip-hgb", action="store_true", help="Skip HistGradientBoosting model (train_models.py).")
    args = parser.parse_args()

    cwd = Path(__file__).resolve().parent

    # Load template config and override paths
    with open(args.config_template, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["known_path"] = args.train_csv
    cfg["blind_path"] = args.blind_csv
    cfg["output_dir"] = args.output_dir
    cfg["random_seed"] = args.seed
    cfg["split_mode_json"] = args.split_mode_json

    tmp_config = Path(tempfile.mktemp(suffix=".json", prefix="run_cfg_"))
    with open(tmp_config, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Run pipeline steps with enforced config
    run_cmd([ "python", "featurize.py", "--config", str(tmp_config)], cwd)
    run_cmd([ "python", "similarity_rank.py", "--config", str(tmp_config)], cwd)

    train_cmd = ["python", "train_models.py", "--config", str(tmp_config)]
    if args.skip_rf:
        train_cmd.append("--skip-rf")
    if args.skip_xgb:
        train_cmd.append("--skip-xgb")
    if args.skip_lgbm:
        train_cmd.append("--skip-lgbm")
    if args.skip_hgb:
        train_cmd.append("--skip-hgb")
    run_cmd(train_cmd, cwd)
    if args.chemprop:
        run_cmd([ "python", "chemprop_runner.py", "--config", str(tmp_config)], cwd)
    if args.keras:
        run_cmd([ "python", "nn_keras.py", "--config", str(tmp_config)], cwd)
    run_cmd([ "python", "metrics_summary.py", "--config", str(tmp_config)], cwd)
    run_cmd([ "python", "consensus.py", "--config", str(tmp_config)], cwd)

    # keep temp config for reproducibility
    final_cfg = Path(args.output_dir) / "config_used.json"
    final_cfg.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(tmp_config, final_cfg)
    print(f"[run_pipeline] Saved config snapshot -> {final_cfg}")


if __name__ == "__main__":
    main()
