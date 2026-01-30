#!/usr/bin/env python3
"""
Aggregate per-model CV metrics into a single CSV for quick comparison.
Honors output_dir from the provided config.
"""

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from chem_utils import load_config


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main(config: str) -> None:
    cfg = load_config(config)
    output_dir = Path(cfg.get("output_dir", "outputs"))
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for path_str in glob.glob(str(metrics_dir / "*.json")):
        path = Path(path_str)
        data = load_json(path)
        model = path.stem.replace("_cv", "")
        row = {"model": model}
        row.update(data)
        rows.append(row)
    if not rows:
        print(f"No metric JSON files found in {metrics_dir}.")
        return

    df = pd.DataFrame(rows)
    out_path = metrics_dir / "all_models.csv"
    df.to_csv(out_path, index=False)
    print(f"[metrics_summary] Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json")
    args = parser.parse_args()
    main(args.config)
