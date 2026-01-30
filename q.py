#!/usr/bin/env python3
"""
One-shot runner for the virtual screening pipeline.

Runs featurization -> similarity ranking -> model training/prediction in sequence.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(script: str, config: str, cwd: Path) -> None:
    cmd = [sys.executable, str(cwd / script), "--config", config]
    print(f"[q] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(cwd))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline runner")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config JSON")
    parser.add_argument("--skip-featurize", action="store_true", help="Skip feature generation")
    parser.add_argument("--skip-similarity", action="store_true", help="Skip similarity ranking")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training/prediction")
    parser.add_argument("--chemprop", action="store_true", help="Also run Chemprop D-MPNN ensemble (requires chemprop).")
    parser.add_argument("--skip-consensus", action="store_true", help="Skip consensus/ECR aggregation")
    parser.add_argument("--skip-metrics-summary", action="store_true", help="Skip metrics summary aggregation")
    parser.add_argument("--keras", action="store_true", help="Run Keras dense baseline on Morgan bits")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    if not args.skip_featurize:
        run_step("featurize.py", args.config, root)
    if not args.skip_similarity:
        run_step("similarity_rank.py", args.config, root)
    if not args.skip_train:
        run_step("train_models.py", args.config, root)
    if args.chemprop:
        run_step("chemprop_runner.py", args.config, root)
    if args.keras:
        run_step("nn_keras.py", args.config, root)
    if not args.skip_metrics_summary:
        run_step("metrics_summary.py", args.config, root)
    if not args.skip_consensus:
        run_step("consensus.py", args.config, root)

    print("[q] Done.")


if __name__ == "__main__":
    main()
