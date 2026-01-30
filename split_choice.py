"""
Split selection and deterministic split generation for QSAR/VS pipelines.
"""

import json
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from scipy.stats import ks_2samp
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# ------------------------------
# Fingerprint helpers
# ------------------------------


def get_feature_mode(train_df: pd.DataFrame, config: Dict) -> Tuple[str, str]:
    smiles_col = config.get("smiles_col", "smiles")
    if smiles_col in train_df.columns:
        return "smiles", smiles_col
    raise ValueError("SMILES column not found; only SMILES mode is supported.")


def build_fps_from_smiles(smiles_list: List[str], radius: int = 2, n_bits: int = 2048) -> List[DataStructs.ExplicitBitVect]:
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        fps.append(fp)
    return fps


def max_similarity(query_fps: List[Optional[DataStructs.ExplicitBitVect]], train_fps: List[Optional[DataStructs.ExplicitBitVect]]) -> np.ndarray:
    sims = []
    clean_train = [fp for fp in train_fps if fp is not None]
    if len(clean_train) == 0:
        return np.zeros(len(query_fps))
    for fp in query_fps:
        if fp is None:
            sims.append(0.0)
            continue
        vals = DataStructs.BulkTanimotoSimilarity(fp, clean_train)
        sims.append(max(vals) if len(vals) else 0.0)
    return np.asarray(sims, dtype=float)


# ------------------------------
# Strategy implementations
# ------------------------------


def _random_stratified_split(train_df: pd.DataFrame, y: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    train_idx, val_idx = train_test_split(
        np.arange(len(train_df)),
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return train_idx, val_idx


def _murcko_groups(train_df: pd.DataFrame, smiles_col: str) -> np.ndarray:
    scaffolds = []
    for smi in train_df[smiles_col]:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol:
            try:
                scaff = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            except Exception:
                scaff = "INVALID_SCAFFOLD"
        else:
            scaff = "INVALID_SCAFFOLD"
        scaffolds.append(scaff)
    return np.asarray(scaffolds, dtype=object)


def _cluster_groups(train_fps: List[Optional[DataStructs.ExplicitBitVect]], cutoff: float) -> Optional[np.ndarray]:
    # Assign invalid fingerprints as singleton clusters instead of failing.
    valid_indices = [i for i, fp in enumerate(train_fps) if fp is not None]
    invalid_indices = [i for i, fp in enumerate(train_fps) if fp is None]
    fps = [train_fps[i] for i in valid_indices]
    if len(fps) == 0:
        return None
    n = len(fps)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])
    clusters = Butina.ClusterData(dists, nPts=n, distThresh=cutoff, isDistData=True)
    cluster_ids = np.empty(len(train_fps), dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            cluster_ids[valid_indices[m]] = cid
    # Assign unique cluster IDs to invalids
    next_cid = len(clusters)
    for inv_i in invalid_indices:
        cluster_ids[inv_i] = next_cid
        next_cid += 1
    return cluster_ids


def _group_split(groups: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx = np.arange(len(groups))
    train_idx, val_idx = next(gss.split(idx, groups=groups))
    return train_idx, val_idx


def _group_stratified_split(groups: np.ndarray, y: np.ndarray, test_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate stratified split that keeps groups intact while matching class ratio.
    Greedy assignment of groups to train/val to minimize deviation from overall positive rate and target size.
    """
    rng = np.random.RandomState(seed)
    unique_groups, inv = np.unique(groups, return_inverse=True)
    group_indices = {g: np.where(inv == i)[0] for i, g in enumerate(unique_groups)}
    group_stats = []
    for g, idxs in group_indices.items():
        n = len(idxs)
        pos = int(np.sum(y[idxs] == 1))
        group_stats.append((g, n, pos))
    rng.shuffle(group_stats)

    total = len(y)
    target_val = int(round(test_size * total))
    target_train = total - target_val
    overall_pos_rate = float(np.mean(y == 1)) if total > 0 else 0.0

    train_set: List[int] = []
    val_set: List[int] = []
    n_train = pos_train = 0
    n_val = pos_val = 0
    lambda_size = 0.1

    for g, n, pos in group_stats:
        # Force to train if val would exceed target excessively
        if n_val + n > target_val and n_train < target_train:
            dest = "train"
        elif n_train + n > target_train and n_val < target_val:
            dest = "val"
        else:
            new_pos_val = pos_val + pos
            new_n_val = n_val + n
            new_pos_train = pos_train + pos
            new_n_train = n_train + n

            val_rate = (new_pos_val / new_n_val) if new_n_val > 0 else overall_pos_rate
            train_rate = (new_pos_train / new_n_train) if new_n_train > 0 else overall_pos_rate

            val_cost = (val_rate - overall_pos_rate) ** 2 + lambda_size * ((new_n_val - target_val) / total) ** 2
            train_cost = (train_rate - overall_pos_rate) ** 2 + lambda_size * ((new_n_train - target_train) / total) ** 2
            dest = "val" if val_cost < train_cost else "train"

        if dest == "val":
            val_set.extend(group_indices[g])
            n_val += n
            pos_val += pos
        else:
            train_set.extend(group_indices[g])
            n_train += n
            pos_train += pos

    return np.asarray(train_set, dtype=int), np.asarray(val_set, dtype=int)


# ------------------------------
# Main selection logic
# ------------------------------


def _summarize_shift(sim_vals: np.ndarray) -> Dict:
    if len(sim_vals) == 0:
        return {k: 0.0 for k in ["q10", "q50", "q90", "p_ge_0.7", "p_ge_0.6", "p_ge_0.5"]}
    return {
        "q10": float(np.quantile(sim_vals, 0.10)),
        "q50": float(np.quantile(sim_vals, 0.50)),
        "q90": float(np.quantile(sim_vals, 0.90)),
        "p_ge_0.7": float(np.mean(sim_vals >= 0.7)),
        "p_ge_0.6": float(np.mean(sim_vals >= 0.6)),
        "p_ge_0.5": float(np.mean(sim_vals >= 0.5)),
    }


def choose_split_strategy(train_df: pd.DataFrame, blind_df: pd.DataFrame, config: Dict) -> Tuple[str, Dict]:
    repeats = int(config.get("repeats", 10))
    random_state_base = int(config.get("random_state_base", 42))
    test_size = float(config.get("test_size", 0.2))
    y_col = config.get("y_col", "act")
    smiles_col = config.get("smiles_col", "smiles")
    radius = int(config.get("radius", 2))
    n_bits = int(config.get("n_bits", 2048))
    cluster_thresholds = config.get("cluster_thresholds", [0.70, 0.60])

    mode, smiles_col = get_feature_mode(train_df, config)
    if mode != "smiles":
        raise ValueError("Only SMILES mode currently supported.")

    y_series = train_df[y_col]
    if y_series.dtype == object:
        mapped = y_series.astype(str).str.lower().str.strip().map({"active": 1, "inactive": 0})
        if mapped.isna().any():
            unknown = y_series[mapped.isna()].unique()
            raise ValueError(f"Unknown labels in y_col '{y_col}': {unknown}")
        y = mapped.to_numpy()
    else:
        y = y_series.to_numpy()
    n_actives_total = int(np.sum(y == 1))
    a_min = int(config.get("A_min", max(20, int(0.05 * n_actives_total))))

    train_fps = build_fps_from_smiles(train_df[smiles_col].tolist(), radius=radius, n_bits=n_bits)
    blind_fps = build_fps_from_smiles(blind_df[smiles_col].tolist(), radius=radius, n_bits=n_bits)
    s_blind = max_similarity(blind_fps, train_fps)

    report: Dict = {"train_to_blind": _summarize_shift(s_blind), "per_strategy": {}}

    candidate_strats: List[Tuple[str, Dict]] = [("random_stratified", {})]
    candidate_strats.append(("murcko_group", {}))
    for thr in cluster_thresholds:
        candidate_strats.append((f"cluster_group_t{thr:.2f}", {"thr": thr}))

    valid_results = []
    rationale_lines = []

    implementable: Dict[str, bool] = {}

    for strat_name, strat_cfg in candidate_strats:
        ks_values = []
        val_active_counts = []
        invalid = 0
        implementable[strat_name] = True

        # Precompute groups / clusters when possible
        group_labels = None
        cluster_failed = False
        if strat_name == "murcko_group":
            group_labels = _murcko_groups(train_df, smiles_col)
        elif strat_name.startswith("cluster_group"):
            cutoff = 1.0 - strat_cfg["thr"]
            cluster_labels = _cluster_groups(train_fps, cutoff)
            if cluster_labels is None:
                cluster_failed = True
                implementable[strat_name] = False
            else:
                group_labels = cluster_labels

        for r in range(repeats):
            seed = random_state_base + r
            try:
                if strat_name == "random_stratified":
                    train_idx, val_idx = _random_stratified_split(train_df, y, test_size, seed)
                elif strat_name == "murcko_group":
                    train_idx, val_idx = _group_stratified_split(group_labels, y, test_size, seed)
                elif strat_name.startswith("cluster_group") and not cluster_failed:
                    train_idx, val_idx = _group_stratified_split(group_labels, y, test_size, seed)
                else:
                    invalid += 1
                    continue
            except Exception:
                invalid += 1
                continue

            val_actives = int(np.sum(y[val_idx] == 1))
            train_actives = int(np.sum(y[train_idx] == 1))
            if val_actives < a_min or train_actives < a_min:
                invalid += 1
                continue

            # Similarity shift val vs blind
            s_val = max_similarity([train_fps[i] for i in val_idx], [train_fps[i] for i in train_idx])
            ks = ks_2samp(s_val, s_blind).statistic if len(s_val) > 0 and len(s_blind) > 0 else 1.0
            ks_values.append(ks)
            val_active_counts.append(val_actives)

        valid_repeats = repeats - invalid
        mean_ks = float(np.mean(ks_values)) if ks_values else 1.0
        std_ks = float(np.std(ks_values)) if ks_values else 0.0
        avg_val_actives = float(np.mean(val_active_counts)) if val_active_counts else 0.0
        report["per_strategy"][strat_name] = {
            "mean_ks": mean_ks,
            "std_ks": std_ks,
            "valid_repeats": valid_repeats,
            "avg_val_actives": avg_val_actives,
            "implementable": implementable[strat_name],
        }

        if valid_repeats > repeats / 2:
            valid_results.append((mean_ks, strat_name))
        else:
            rationale_lines.append(f"{strat_name} invalid: only {valid_repeats}/{repeats} valid repeats.")

    chosen_strategy = None
    if valid_results:
        valid_results.sort(key=lambda x: x[0])
        chosen_strategy = valid_results[0][1]
        rationale_lines.append(f"Chose {chosen_strategy} with lowest mean KS={valid_results[0][0]:.3f}.")
    else:
        # fallback
        for fb in ["cluster_group_t0.70", "random_stratified"]:
            if fb in report["per_strategy"] and report["per_strategy"][fb].get("implementable", True):
                chosen_strategy = fb
                rationale_lines.append(f"Fallback to {fb} (no valid strategies).")
                break
        if chosen_strategy is None:
            chosen_strategy = "random_stratified"
            rationale_lines.append("Fallback to random_stratified as default.")

    report["chosen_strategy"] = chosen_strategy
    report["rationale"] = " ".join(rationale_lines)
    return chosen_strategy, report


def make_split_indices(train_df: pd.DataFrame, chosen_strategy: str, config: Dict, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    test_size = float(config.get("test_size", 0.2))
    y_col = config.get("y_col", "act")
    smiles_col = config.get("smiles_col", "smiles")
    radius = int(config.get("radius", 2))
    n_bits = int(config.get("n_bits", 2048))
    y_series = train_df[y_col]
    if y_series.dtype == object:
        mapped = y_series.astype(str).str.lower().str.strip().map({"active": 1, "inactive": 0})
        if mapped.isna().any():
            unknown = y_series[mapped.isna()].unique()
            raise ValueError(f"Unknown labels in y_col '{y_col}': {unknown}")
        y_series = mapped
    y = y_series.to_numpy()

    if chosen_strategy == "random_stratified":
        return _random_stratified_split(train_df, y, test_size, seed)

    # For group-based splits we need groups
    if smiles_col not in train_df.columns:
        raise ValueError("SMILES column required for group-based splits.")
    train_fps = build_fps_from_smiles(train_df[smiles_col].tolist(), radius=radius, n_bits=n_bits)

    if chosen_strategy == "murcko_group":
        groups = _murcko_groups(train_df, smiles_col)
        return _group_stratified_split(groups, y, test_size, seed)

    if chosen_strategy.startswith("cluster_group"):
        try:
            thr = float(chosen_strategy.split("_t")[-1])
        except Exception:
            thr = 0.7
        cutoff = 1.0 - thr
        groups = _cluster_groups(train_fps, cutoff)
        if groups is None:
            raise ValueError("Cluster grouping failed.")
        return _group_stratified_split(groups, y, test_size, seed)

    raise ValueError(f"Unknown strategy {chosen_strategy}")


def save_split_mode(path: str, chosen_strategy: str, report: Dict, config: Dict) -> None:
    payload = {
        "chosen_strategy": chosen_strategy,
        "report": report,
        "config_used": config,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_split_mode(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
