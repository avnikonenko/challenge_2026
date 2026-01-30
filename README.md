# Virtual Screening Pipeline

Pipeline to pick the top ~100 actives from a blind set (~3000 compounds) using similarity and ML models on Morgan fingerprints and 2D descriptors. Includes scaffold-split CV and early-enrichment metrics.

## Requirements
- Python 3.8+
- RDKit (with `Chem`/fingerprints/descriptors)
- pandas, numpy, scikit-learn, joblib
- Optional: xgboost, lightgbm (if absent, fallback models still run)

## Inputs
- `known_path` (CSV/TSV): columns `smiles`, `status` (`active`/`inactive`), optional `name`
- `blind_path` (.smi): `smiles<TAB>name`
- Configure paths and settings in `config.json`.

## One-Command Run
```bash
python q.py --config config.json
```
Options: `--skip-featurize`, `--skip-similarity`, `--skip-train`.
To also run Chemprop D-MPNN ensemble (5 seeds): add `--chemprop` (requires `chemprop` installed).
To run the Keras dense Morgan baseline: add `--keras` (requires tensorflow/keras).
Consensus/ECR aggregation runs by default; skip with `--skip-consensus`.
Metrics summary aggregation runs by default; skip with `--skip-metrics-summary`.

## Individual Steps
- Featurize (Morgan + 2D): `python featurize.py --config config.json`
- Similarity rank vs actives: `python similarity_rank.py --config config.json`
- Train + predict (scaffold CV, calibrated RF/XGB/LGBM/HGB): `python train_models.py --config config.json`
- Chemprop D-MPNN ensemble (with Morgan bits): `python chemprop_runner.py --config config.json`
- Keras dense baseline on Morgan bits: `python nn_keras.py --config config.json`
- Consensus/ECR: `python consensus.py --config config.json`
- Metrics summary: `python metrics_summary.py --config config.json`
- Split choice (Stage 1): `python choose_split_mode.py --train_csv TRAIN.csv --blind_csv BLIND.csv --out_json split_mode.json --smiles_col smiles`
- Stage 2 runner enforcing the chosen split:  
  `python run_pipeline.py --train_csv TRAIN.csv --blind_csv BLIND.csv --split_mode_json split_mode.json [--chemprop] [--keras]`

## Outputs (under `outputs/`)
- `features/actives|inactives|unknown_{morgan,descriptors}.parquet`
- `similarity/blind_similarity_ranked.csv`
- `predictions/{model}_{feat}_blind_ranked.csv` (ranked scores; take top 100)
- `metrics/{model}_{feat}_cv.json` (precision@k, EF, BEDROC, PR-AUC, ROC-AUC)
- `metrics/all_models.csv` aggregated CV metrics for quick comparison
- `models/{model}_{feat}.pkl` (CalibratedClassifierCV + feature list)
- `predictions/chemprop_blind_ranked.csv`, `metrics/chemprop_cv.json` when Chemprop run
- `predictions/keras_morgan_blind_ranked.csv`, `metrics/keras_morgan_cv.json` when Keras run
- `consensus/all_model_rankings.csv` stacked ranks with model/params/scaffold; `consensus/ecr_consensus.csv` exponential consensus ranking across all methods
- `split_mode.json` chosen validation split strategy + diagnostics (from Stage 1)

## Installation (conda/mamba recommended)
Create env:
```bash
mamba create -n vscreen python=3.10 -y
conda activate vscreen
```

Core deps (CPU):
```bash
mamba install -y rdkit -c conda-forge
pip install pandas numpy scikit-learn scipy joblib
```

ML models:
```bash
pip install xgboost lightgbm
```

Chemprop (for D-MPNN):
```bash
pip install chemprop
```

Keras/TensorFlow (for `nn_keras.py`):
```bash
pip install tensorflow
```

If you prefer pure pip:
```bash
pip install rdkit-pypi pandas numpy scikit-learn scipy joblib xgboost lightgbm chemprop tensorflow
```

After install, run Stage 1 → Stage 2 as described above.

## Recommended “best-performance” run
1) Choose the split (uses train+blind chemistry):
```bash
python choose_split_mode.py --train_csv TRAIN.csv --blind_csv BLIND.csv --out_json split_mode.json --smiles_col smiles --y_col act --repeats 10 --test_size 0.2
```
2) Run full pipeline with all strong models (tree ensembles + Chemprop + Keras) and consensus:
```bash
python run_pipeline.py --train_csv TRAIN.csv --blind_csv BLIND.csv --split_mode_json split_mode.json --chemprop --keras
```
Outputs to review:
- `outputs/predictions/*_blind_ranked.csv` (per-model ranks)
- `outputs/consensus/ecr_consensus.csv` (weighted consensus ranking; use this as primary pick list)

## Notes
- Uses Bemis–Murcko scaffold split for honest validation.
- Probabilities are calibrated (isotonic or Platt) before ranking.
- Early-enrichment focus: precision@k, EF1%, BEDROC.
- Adjust fingerprint radius/bits, top-k settings, and seeds in `config.json`. Chemprop uses binary Morgan (`chemprop_morgan_radius`, `chemprop_morgan_bits`) and can append RDKit 2D descriptors when `chemprop_use_rdkit_desc` is true. Consensus knobs: `consensus_tau` (decay), `consensus_focus_k` (only ranks up to this count), `consensus_metric` (weighting metric, e.g., precision@100 or pr_auc), `consensus_weight_fallback` (default weight if metric missing).
