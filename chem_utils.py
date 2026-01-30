import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


# Light-weight config loader that stays JSON-only to avoid extra deps.
def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class FeaturizationConfig:
    radius: int = 2
    n_bits: int = 2048
    use_chirality: bool = True


def mol_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    if not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.SanitizeMol(mol, catchErrors=True)
    return mol


def morgan_fingerprint(
    mol: Chem.Mol, cfg: FeaturizationConfig
) -> Tuple[np.ndarray, Optional[DataStructs.ExplicitBitVect]]:
    if mol is None:
        return np.array([]), None
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        cfg.radius,
        nBits=cfg.n_bits,
        useChirality=cfg.use_chirality,
    )
    arr = np.zeros((cfg.n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr, fp


def descriptor_names() -> List[str]:
    return [name for name, _ in Descriptors._descList]


def compute_descriptors(mol: Chem.Mol, names: Sequence[str]) -> Dict[str, float]:
    if mol is None:
        return {n: np.nan for n in names}
    values = {}
    for name, fn in Descriptors._descList:
        if name not in names:
            continue
        try:
            values[name] = float(fn(mol))
        except Exception:
            values[name] = np.nan
    return values


def bemis_murcko_scaffold(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return "INVALID_SCAFFOLD"
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return "INVALID_SCAFFOLD"


def tanimoto_similarity(fp: DataStructs.ExplicitBitVect, fp_list: List[DataStructs.ExplicitBitVect]) -> List[float]:
    if fp is None:
        return [0.0] * len(fp_list)
    return DataStructs.BulkTanimotoSimilarity(fp, fp_list)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    k = max(1, min(k, len(y_true)))
    order = np.argsort(-y_score)
    topk = y_true[order][:k]
    return float(np.sum(topk)) / float(k)


def ef_at_percent(y_true: np.ndarray, y_score: np.ndarray, percent: float) -> float:
    assert 0 < percent <= 1.0
    n = len(y_true)
    top_n = max(1, int(math.ceil(percent * n)))
    order = np.argsort(-y_score)
    hits_top = int(np.sum(y_true[order][:top_n]))
    total_actives = int(np.sum(y_true))
    if total_actives == 0:
        return 0.0
    expected = (total_actives / n) * top_n
    if expected == 0:
        return 0.0
    return (hits_top / expected)


def ef_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if k <= 0 or len(y_true) == 0:
        return 0.0
    k_eff = min(k, len(y_true))
    order = np.argsort(-y_score)
    hits_top = int(np.sum(y_true[order][:k_eff]))
    total_actives = int(np.sum(y_true))
    if total_actives == 0:
        return 0.0
    expected = (total_actives / len(y_true)) * k_eff
    if expected == 0:
        return 0.0
    return hits_top / expected


def enrichment_factor_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int = 100) -> float:
    return ef_at_k(y_true, y_score, k)


def bedroc_score(y_true: np.ndarray, y_score: np.ndarray, alpha: float = 20.0) -> float:
    """
    Implementation based on Truchon & Bayly, J. Chem. Inf. Model. 2007.
    """
    n = len(y_true)
    n_actives = int(np.sum(y_true))
    if n_actives == 0 or n == 0:
        return 0.0

    order = np.argsort(-y_score)
    ranks = np.arange(1, n + 1)
    active_positions = ranks[order][y_true[order] == 1]
    x_i = active_positions / n

    const = alpha / (1.0 - math.exp(-alpha))
    rie = (const / n_actives) * np.sum(np.exp(-alpha * x_i))

    # Extremes for normalization
    best_positions = np.arange(1, n_actives + 1) / n
    worst_positions = np.arange(n - n_actives + 1, n + 1) / n
    rie_max = (const / n_actives) * np.sum(np.exp(-alpha * best_positions))
    rie_min = (const / n_actives) * np.sum(np.exp(-alpha * worst_positions))

    if rie_max == rie_min:
        return 0.0
    return float((rie - rie_min) / (rie_max - rie_min))


def scaffold_split_indices(smiles: Sequence[str], n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Deterministic scaffold split that keeps entire Bemis–Murcko scaffolds
    in the same fold. Balances fold sizes greedily.
    """
    rng = random.Random(seed)
    scaffolds: Dict[str, List[int]] = {}
    for idx, smi in enumerate(smiles):
        scaff = bemis_murcko_scaffold(smi)
        scaffolds.setdefault(scaff, []).append(idx)

    scaffold_groups = list(scaffolds.items())
    rng.shuffle(scaffold_groups)
    scaffold_groups.sort(key=lambda kv: len(kv[1]), reverse=True)

    folds: List[List[int]] = [[] for _ in range(n_folds)]
    fold_sizes = [0] * n_folds

    for _, idxs in scaffold_groups:
        target = fold_sizes.index(min(fold_sizes))
        folds[target].extend(idxs)
        fold_sizes[target] += len(idxs)

    splits = []
    all_indices = set(range(len(smiles)))
    for fold_indices in folds:
        val_idx = np.array(sorted(fold_indices), dtype=int)
        train_idx = np.array(sorted(all_indices - set(fold_indices)), dtype=int)
        splits.append((train_idx, val_idx))
    return splits
