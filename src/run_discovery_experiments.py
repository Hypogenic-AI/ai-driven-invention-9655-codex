"""Offline active-discovery benchmark for atomic materials.

This script tests whether a structured, surrogate-guided discovery loop finds
stable inorganic materials faster than random trial-and-error when the oracle is
precomputed energy-above-hull data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"

STABLE_THRESHOLD = 0.05
ON_HULL_THRESHOLD = 0.0

SYMBOLS = [
    "",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]
SYMBOL_TO_Z = {sym: z for z, sym in enumerate(SYMBOLS) if sym}
FORMULA_RE = re.compile(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    device: str = "cuda:0"
    train_batch_size: int = 128
    predict_batch_size: int = 8192
    mlp_epochs: int = 14
    eval_epochs: int = 22
    mc_passes: int = 5
    candidate_pool_size: int = 25000
    initial_size: int = 400
    acquisition_batch_size: int = 250
    rounds: int = 8
    active_seeds: tuple[int, ...] = (13, 29, 41, 53, 67)
    policies: tuple[str, ...] = (
        "random",
        "uncertainty",
        "greedy",
        "ucb",
        "diverse_ucb",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_workspace() -> None:
    cwd = Path.cwd().resolve()
    if cwd != ROOT:
        raise RuntimeError(f"Run from workspace root {ROOT}, got {cwd}")
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)


def z_to_period(z_values: np.ndarray) -> np.ndarray:
    breaks = np.array([2, 10, 18, 36, 54, 86, 118])
    return np.searchsorted(breaks, z_values, side="left") + 1


def z_to_block_code(z_values: np.ndarray) -> np.ndarray:
    z = np.asarray(z_values)
    block = np.zeros_like(z, dtype=float)
    # 1=s, 2=p, 3=d, 4=f, based on periodic-table regions.
    s_block = np.isin(z, [1, 2, 3, 4, 11, 12, 19, 20, 37, 38, 55, 56, 87, 88])
    f_block = ((57 <= z) & (z <= 71)) | ((89 <= z) & (z <= 103))
    d_block = (
        ((21 <= z) & (z <= 30))
        | ((39 <= z) & (z <= 48))
        | ((72 <= z) & (z <= 80))
        | ((104 <= z) & (z <= 112))
    )
    p_block = ~(s_block | d_block | f_block)
    block[s_block] = 1
    block[p_block] = 2
    block[d_block] = 3
    block[f_block] = 4
    return block


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mean = float(np.sum(values * weights))
    var = float(np.sum(weights * (values - mean) ** 2))
    return mean, math.sqrt(max(var, 0.0))


def composition_feature_from_counts(counts_by_z: np.ndarray) -> np.ndarray:
    counts = counts_by_z.astype(np.float32)
    total = float(counts.sum())
    if total <= 0:
        return np.zeros(118 + 13, dtype=np.float32)

    fractions = counts / total
    nonzero = np.flatnonzero(counts) + 1
    nz_frac = fractions[nonzero - 1]
    z_vals = nonzero.astype(np.float32)
    periods = z_to_period(z_vals).astype(np.float32)
    blocks = z_to_block_code(z_vals).astype(np.float32)
    mean_z, std_z = weighted_mean_std(z_vals, nz_frac)
    mean_period, std_period = weighted_mean_std(periods, nz_frac)
    mean_block, std_block = weighted_mean_std(blocks, nz_frac)
    entropy = float(-(nz_frac * np.log(nz_frac + 1e-12)).sum())
    summaries = np.array(
        [
            total,
            len(nonzero),
            entropy,
            float(nz_frac.max()),
            float(nz_frac.min()),
            mean_z,
            std_z,
            float(z_vals.min()),
            float(z_vals.max()),
            float(z_vals.max() - z_vals.min()),
            mean_period,
            std_period,
            mean_block + 0.1 * std_block,
        ],
        dtype=np.float32,
    )
    return np.concatenate([fractions.astype(np.float32), summaries])


def comp_feature_names() -> list[str]:
    summary_names = [
        "atom_count_from_formula",
        "n_unique_elements",
        "composition_entropy",
        "max_element_fraction",
        "min_element_fraction",
        "weighted_mean_z",
        "weighted_std_z",
        "min_z",
        "max_z",
        "z_range",
        "weighted_mean_period",
        "weighted_std_period",
        "weighted_block_code",
    ]
    return [f"frac_{sym}" for sym in SYMBOLS[1:]] + summary_names


def parse_formula_counts(formula: str) -> np.ndarray:
    counts = np.zeros(118, dtype=np.float32)
    if not isinstance(formula, str):
        return counts
    for sym, raw_count in FORMULA_RE.findall(formula):
        z = SYMBOL_TO_Z.get(sym)
        if not z:
            continue
        count = float(raw_count) if raw_count else 1.0
        counts[z - 1] += count
    return counts


def cell_descriptors(cell: np.ndarray, num_atoms: float) -> np.ndarray:
    mat = np.asarray(cell, dtype=np.float64)
    lengths = np.linalg.norm(mat, axis=1)
    volume = abs(float(np.linalg.det(mat)))
    safe_lengths = np.where(lengths <= 1e-12, 1e-12, lengths)
    cos_alpha = np.clip(np.dot(mat[1], mat[2]) / (safe_lengths[1] * safe_lengths[2]), -1, 1)
    cos_beta = np.clip(np.dot(mat[0], mat[2]) / (safe_lengths[0] * safe_lengths[2]), -1, 1)
    cos_gamma = np.clip(np.dot(mat[0], mat[1]) / (safe_lengths[0] * safe_lengths[1]), -1, 1)
    angles = np.degrees(np.arccos([cos_alpha, cos_beta, cos_gamma]))
    n = max(float(num_atoms), 1.0)
    return np.array(
        [
            volume,
            math.log1p(volume),
            volume / n,
            math.log1p(volume / n),
            lengths[0],
            lengths[1],
            lengths[2],
            lengths.max() / max(lengths.min(), 1e-12),
            angles[0],
            angles[1],
            angles[2],
        ],
        dtype=np.float32,
    )


def load_or_build_mp_features(force: bool = False) -> dict[str, np.ndarray | list[str]]:
    cache = RESULTS / "mp_features.npz"
    if cache.exists() and not force:
        loaded = np.load(cache, allow_pickle=True)
        if "raw_row_count" not in loaded or "excluded_missing_energy_above_hull" not in loaded:
            return load_or_build_mp_features(force=True)
        return {
            "X_comp": loaded["X_comp"],
            "X_struct": loaded["X_struct"],
            "y": loaded["y"],
            "energy_above_hull": loaded["energy_above_hull"],
            "material_id": loaded["material_id"],
            "comp_names": list(loaded["comp_names"]),
            "struct_names": list(loaded["struct_names"]),
            "raw_row_count": int(loaded["raw_row_count"]),
            "excluded_missing_energy_above_hull": int(loaded["excluded_missing_energy_above_hull"]),
        }

    h5_path = DATASETS / "materials_project_hf" / "extracted" / "data.hdf5"
    with h5py.File(h5_path, "r") as h5:
        z_flat = h5["data/z"][:]
        offsets = h5["indexing/num_atoms"][:]
        num_atoms = h5["data/num_atoms"][:]
        cells = h5["data/cell"][:]
        e_hull = h5["data/energy_above_hull"][:]
        material_id = h5["data/material_id"][:]

    n = len(e_hull)
    comp_features = np.zeros((n, 118 + 13), dtype=np.float32)
    struct_extra = np.zeros((n, 11), dtype=np.float32)
    for i in range(n):
        zs = z_flat[offsets[i] : offsets[i + 1]]
        counts = np.bincount(zs, minlength=119)[1:119]
        comp_features[i] = composition_feature_from_counts(counts)
        struct_extra[i] = cell_descriptors(cells[i], num_atoms[i])

    X_comp_all = np.nan_to_num(comp_features, nan=0.0, posinf=0.0, neginf=0.0)
    X_struct_all = np.nan_to_num(np.hstack([X_comp_all, struct_extra]), nan=0.0, posinf=0.0, neginf=0.0)
    valid = np.isfinite(e_hull)
    raw_row_count = int(len(e_hull))
    excluded_missing = int((~valid).sum())
    X_comp = X_comp_all[valid]
    X_struct = X_struct_all[valid]
    e_hull = e_hull[valid]
    material_id = material_id[valid]
    y = (e_hull <= STABLE_THRESHOLD).astype(np.int8)
    comp_names = comp_feature_names()
    struct_names = comp_names + [
        "cell_volume",
        "log_cell_volume",
        "volume_per_atom",
        "log_volume_per_atom",
        "cell_a",
        "cell_b",
        "cell_c",
        "cell_aspect_ratio",
        "cell_alpha",
        "cell_beta",
        "cell_gamma",
    ]
    np.savez_compressed(
        cache,
        X_comp=X_comp,
        X_struct=X_struct,
        y=y,
        energy_above_hull=e_hull,
        material_id=material_id,
        comp_names=np.array(comp_names),
        struct_names=np.array(struct_names),
        raw_row_count=np.array(raw_row_count),
        excluded_missing_energy_above_hull=np.array(excluded_missing),
    )
    return {
        "X_comp": X_comp,
        "X_struct": X_struct,
        "y": y,
        "energy_above_hull": e_hull,
        "material_id": material_id,
        "comp_names": comp_names,
        "struct_names": struct_names,
        "raw_row_count": raw_row_count,
        "excluded_missing_energy_above_hull": excluded_missing,
    }


def audit_data(mp: dict[str, np.ndarray | list[str]]) -> dict[str, object]:
    y = np.asarray(mp["y"])
    e_hull = np.asarray(mp["energy_above_hull"])
    audit = {
        "materials_project_raw_rows": int(mp.get("raw_row_count", len(y))),
        "materials_project_labeled_rows_used": int(len(y)),
        "excluded_missing_energy_above_hull": int(mp.get("excluded_missing_energy_above_hull", 0)),
        "stable_threshold_ev_per_atom": STABLE_THRESHOLD,
        "stable_count": int(y.sum()),
        "stable_prevalence": float(y.mean()),
        "energy_above_hull": {
            "mean": float(np.nanmean(e_hull)),
            "std": float(np.nanstd(e_hull)),
            "min": float(np.nanmin(e_hull)),
            "median": float(np.nanmedian(e_hull)),
            "max": float(np.nanmax(e_hull)),
            "missing": int(np.isnan(e_hull).sum()),
        },
        "feature_counts": {
            "composition_features": int(np.asarray(mp["X_comp"]).shape[1]),
            "structure_features": int(np.asarray(mp["X_struct"]).shape[1]),
        },
    }

    gnome_path = DATASETS / "gnome_summaries" / "stable_materials_r2scan.csv"
    gnome = pd.read_csv(
        gnome_path,
        usecols=["Composition", "Decomposition Energy Per Atom", "NSites"],
    )
    decomp = gnome["Decomposition Energy Per Atom"].astype(float)
    audit["gnome_r2scan_rows"] = int(len(gnome))
    audit["gnome_on_hull_prevalence"] = float((decomp <= ON_HULL_THRESHOLD).mean())
    audit["gnome_near_hull_prevalence"] = float((decomp <= STABLE_THRESHOLD).mean())
    audit["gnome_decomposition_energy"] = {
        "mean": float(decomp.mean()),
        "std": float(decomp.std()),
        "min": float(decomp.min()),
        "median": float(decomp.median()),
        "max": float(decomp.max()),
        "missing": int(decomp.isna().sum()),
    }
    return audit


class MLP(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def make_scaler(X: np.ndarray, fit_idx: np.ndarray | None = None) -> tuple[StandardScaler, np.ndarray]:
    scaler = StandardScaler()
    if fit_idx is None:
        scaler.fit(X)
    else:
        scaler.fit(X[fit_idx])
    X_scaled = scaler.transform(X).astype(np.float32)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    return scaler, X_scaled


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    config: ExperimentConfig,
    epochs: int | None = None,
) -> MLP:
    set_seed(seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model = MLP(X.shape[1]).to(device)
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y.astype(np.float32), dtype=torch.float32, device=device)
    pos = float(y.sum())
    neg = float(len(y) - pos)
    pos_weight = torch.tensor([max(neg / max(pos, 1.0), 1.0)], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    n = len(y)
    num_epochs = epochs or config.mlp_epochs
    use_amp = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    for epoch in range(num_epochs):
        g = torch.Generator(device=device)
        g.manual_seed(seed + epoch)
        order = torch.randperm(n, generator=g, device=device)
        for start in range(0, n, config.train_batch_size):
            idx = order[start : start + config.train_batch_size]
            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    logits = model(X_t[idx])
                    loss = loss_fn(logits, y_t[idx])
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                logits = model(X_t[idx])
                loss = loss_fn(logits, y_t[idx])
                loss.backward()
                opt.step()
    return model


@torch.no_grad()
def predict_mlp(
    model: MLP,
    X: np.ndarray,
    *,
    config: ExperimentConfig,
    mc_passes: int = 1,
    dropout: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    preds = []
    if dropout:
        model.train()
    else:
        model.eval()
    for _ in range(mc_passes):
        chunks = []
        for start in range(0, len(X), config.predict_batch_size):
            logits = model(X_t[start : start + config.predict_batch_size])
            chunks.append(torch.sigmoid(logits).detach().cpu().numpy())
        preds.append(np.concatenate(chunks))
    arr = np.vstack(preds)
    return arr.mean(axis=0), arr.std(axis=0)


def best_f1_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    if len(thresholds) == 0:
        return 0.5
    best = int(np.nanargmax(f1[:-1]))
    return float(thresholds[best])


def classifier_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "average_precision": float(average_precision_score(y_true, probs)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def run_predictive_signal(
    X_comp: np.ndarray,
    X_struct: np.ndarray,
    y: np.ndarray,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=20000,
        train_size=60000,
        random_state=config.seed,
        stratify=y,
    )
    rows = []
    artifacts: dict[str, object] = {"train_idx": train_idx, "test_idx": test_idx}
    for name, X in [("composition_only", X_comp), ("composition_plus_cell", X_struct)]:
        scaler, X_scaled = make_scaler(X, train_idx)
        model = train_mlp(
            X_scaled[train_idx],
            y[train_idx],
            seed=config.seed + (0 if name == "composition_only" else 101),
            config=config,
            epochs=config.eval_epochs,
        )
        train_probs, _ = predict_mlp(model, X_scaled[train_idx], config=config)
        threshold = best_f1_threshold(y[train_idx], train_probs)
        test_probs, _ = predict_mlp(model, X_scaled[test_idx], config=config)
        metrics = classifier_metrics(y[test_idx], test_probs, threshold)
        metrics.update({"feature_set": name, "train_rows": len(train_idx), "test_rows": len(test_idx)})
        rows.append(metrics)
        artifacts[f"{name}_model"] = model
        artifacts[f"{name}_scaler"] = scaler
        artifacts[f"{name}_test_probs"] = test_probs
    return pd.DataFrame(rows), artifacts


def policy_offset(policy: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(policy))


def select_diverse(
    candidate_indices: np.ndarray,
    scores: np.ndarray,
    X_diversity: np.ndarray,
    batch_size: int,
    diversity_weight: float = 0.18,
) -> np.ndarray:
    if len(candidate_indices) <= batch_size:
        return candidate_indices
    local_scores = scores.copy()
    score_range = local_scores.max() - local_scores.min()
    if score_range > 0:
        local_scores = (local_scores - local_scores.min()) / score_range
    selected_local = [int(np.argmax(local_scores))]
    selected_mask = np.zeros(len(candidate_indices), dtype=bool)
    selected_mask[selected_local[0]] = True
    features = X_diversity[candidate_indices].astype(np.float32)
    min_dist = np.linalg.norm(features - features[selected_local[0]], axis=1)
    for _ in range(1, min(batch_size, len(candidate_indices))):
        dist_range = min_dist.max() - min_dist.min()
        dist_score = (min_dist - min_dist.min()) / dist_range if dist_range > 0 else min_dist
        combined = local_scores + diversity_weight * dist_score
        combined[selected_mask] = -np.inf
        next_local = int(np.argmax(combined))
        selected_mask[next_local] = True
        selected_local.append(next_local)
        new_dist = np.linalg.norm(features - features[next_local], axis=1)
        min_dist = np.minimum(min_dist, new_dist)
    return candidate_indices[np.array(selected_local, dtype=int)]


def run_single_policy(
    X_scaled_pool: np.ndarray,
    X_diversity_pool: np.ndarray,
    y_pool: np.ndarray,
    initial_idx: np.ndarray,
    policy: str,
    seed: int,
    config: ExperimentConfig,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed + policy_offset(policy))
    observed = np.zeros(len(y_pool), dtype=bool)
    observed[initial_idx] = True
    pool_stable = int(y_pool.sum())
    pool_prevalence = float(y_pool.mean())
    rows = []

    def append_trace(round_id: int) -> None:
        stable_found = int(y_pool[observed].sum())
        observed_count = int(observed.sum())
        precision = stable_found / observed_count
        recall = stable_found / max(pool_stable, 1)
        rows.append(
            {
                "seed": seed,
                "policy": policy,
                "round": round_id,
                "observed": observed_count,
                "stable_found": stable_found,
                "precision": precision,
                "pool_stable_prevalence": pool_prevalence,
                "discovery_acceleration_factor": precision / max(pool_prevalence, 1e-12),
                "stable_recall": recall,
            }
        )

    append_trace(0)
    for round_id in range(1, config.rounds + 1):
        unobserved = np.flatnonzero(~observed)
        batch = min(config.acquisition_batch_size, len(unobserved))
        if policy == "random":
            selected = rng.choice(unobserved, size=batch, replace=False)
        else:
            model = train_mlp(
                X_scaled_pool[observed],
                y_pool[observed],
                seed=seed + 1000 * round_id + policy_offset(policy),
                config=config,
                epochs=config.mlp_epochs,
            )
            p_mean, p_std = predict_mlp(
                model,
                X_scaled_pool[unobserved],
                config=config,
                mc_passes=config.mc_passes,
                dropout=True,
            )
            if policy == "greedy":
                score = p_mean
                selected = unobserved[np.argsort(score)[-batch:]]
            elif policy == "uncertainty":
                score = 1.0 - 2.0 * np.abs(p_mean - 0.5)
                selected = unobserved[np.argsort(score)[-batch:]]
            elif policy == "ucb":
                score = p_mean + 0.5 * p_std
                selected = unobserved[np.argsort(score)[-batch:]]
            elif policy == "diverse_ucb":
                score = p_mean + 0.5 * p_std
                shortlist_size = min(len(unobserved), batch * 8)
                shortlist_local = np.argsort(score)[-shortlist_size:]
                shortlist = unobserved[shortlist_local]
                shortlist_scores = score[shortlist_local]
                selected = select_diverse(
                    shortlist,
                    shortlist_scores,
                    X_diversity_pool,
                    batch,
                )
            else:
                raise ValueError(f"Unknown policy: {policy}")
        observed[selected] = True
        append_trace(round_id)
    return rows


def run_active_discovery(
    X_struct: np.ndarray,
    X_comp: np.ndarray,
    y: np.ndarray,
    config: ExperimentConfig,
) -> pd.DataFrame:
    all_rows = []
    for seed in config.active_seeds:
        rng = np.random.default_rng(seed)
        pool_idx = rng.choice(len(y), size=min(config.candidate_pool_size, len(y)), replace=False)
        pool_prevalence = y[pool_idx].mean()
        if pool_prevalence == 0:
            raise RuntimeError("Sampled a pool with no stable examples; increase pool size.")
        initial_idx = rng.choice(len(pool_idx), size=config.initial_size, replace=False)
        scaler, X_pool_scaled = make_scaler(X_struct[pool_idx])
        _ = scaler
        X_diversity_pool = X_comp[pool_idx, :118]
        y_pool = y[pool_idx]
        for policy in config.policies:
            rows = run_single_policy(
                X_pool_scaled,
                X_diversity_pool,
                y_pool,
                initial_idx,
                policy,
                seed,
                config,
            )
            all_rows.extend(rows)
    return pd.DataFrame(all_rows)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    p = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = (m - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def summarize_active_results(trace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    final_round = trace["round"].max()
    final = trace[trace["round"] == final_round].copy()
    summary = (
        final.groupby("policy")
        .agg(
            mean_stable_found=("stable_found", "mean"),
            std_stable_found=("stable_found", "std"),
            mean_precision=("precision", "mean"),
            std_precision=("precision", "std"),
            mean_daf=("discovery_acceleration_factor", "mean"),
            std_daf=("discovery_acceleration_factor", "std"),
            mean_recall=("stable_recall", "mean"),
            std_recall=("stable_recall", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    pivot = final.pivot(index="seed", columns="policy", values="stable_found")
    tests = []
    baseline = pivot["random"].values
    raw_p = []
    policies = [p for p in pivot.columns if p != "random"]
    for policy in policies:
        vals = pivot[policy].values
        diff = vals - baseline
        mean_diff = float(diff.mean())
        sd_diff = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
        cohens_d = mean_diff / sd_diff if sd_diff > 0 else float("inf")
        shapiro_p = float(stats.shapiro(diff).pvalue) if len(diff) >= 3 else np.nan
        t_p = float(stats.ttest_rel(vals, baseline).pvalue)
        try:
            w_p = float(stats.wilcoxon(vals, baseline, zero_method="wilcox").pvalue)
        except ValueError:
            w_p = 1.0
        chosen_test = "paired_t" if (not np.isnan(shapiro_p) and shapiro_p >= 0.05) else "wilcoxon"
        raw = t_p if chosen_test == "paired_t" else w_p
        raw_p.append(raw)
        ci_low, ci_high = mean_confidence_interval(diff)
        tests.append(
            {
                "comparison": f"{policy} vs random",
                "policy": policy,
                "mean_difference_stable_found": mean_diff,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "cohens_d_paired": cohens_d,
                "shapiro_p_for_differences": shapiro_p,
                "paired_t_p": t_p,
                "wilcoxon_p": w_p,
                "chosen_test": chosen_test,
                "raw_p": raw,
            }
        )
    adjusted = holm_adjust(raw_p)
    for row, p_adj in zip(tests, adjusted):
        row["holm_adjusted_p"] = p_adj
    return summary, pd.DataFrame(tests)


def mean_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        val = float(values.mean()) if len(values) else 0.0
        return val, val
    mean = float(values.mean())
    sem = stats.sem(values)
    h = float(sem * stats.t.ppf((1 + confidence) / 2, len(values) - 1))
    return mean - h, mean + h


def build_gnome_comp_features() -> tuple[pd.DataFrame, np.ndarray]:
    gnome_path = DATASETS / "gnome_summaries" / "stable_materials_r2scan.csv"
    cols = [
        "Composition",
        "MaterialId",
        "Reduced Formula",
        "NSites",
        "Decomposition Energy Per Atom",
        "Formation Energy Per Atom",
        "Bandgap",
        "Crystal System",
    ]
    gnome = pd.read_csv(gnome_path, usecols=cols)
    X = np.zeros((len(gnome), 118 + 13), dtype=np.float32)
    for i, formula in enumerate(gnome["Composition"].values):
        X[i] = composition_feature_from_counts(parse_formula_counts(formula))
    return gnome, X


def run_gnome_transfer(
    mp_comp_model: MLP,
    mp_comp_scaler: StandardScaler,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gnome, X_gnome = build_gnome_comp_features()
    X_gnome_scaled = mp_comp_scaler.transform(X_gnome).astype(np.float32)
    probs, _ = predict_mlp(mp_comp_model, X_gnome_scaled, config=config)
    gnome = gnome.copy()
    gnome["mp_trained_stability_score"] = probs
    gnome["on_hull"] = gnome["Decomposition Energy Per Atom"] <= ON_HULL_THRESHOLD
    gnome["near_hull"] = gnome["Decomposition Energy Per Atom"] <= STABLE_THRESHOLD
    gnome.sort_values("mp_trained_stability_score", ascending=False).to_csv(
        RESULTS / "gnome_ranked_candidates.csv",
        index=False,
    )

    base_on_hull = float(gnome["on_hull"].mean())
    base_near = float(gnome["near_hull"].mean())
    rng = np.random.default_rng(config.seed)
    rows = []
    for k in [100, 500, 1000, 5000, 10000]:
        top = gnome.nlargest(k, "mp_trained_stability_score")
        random_rates = []
        random_near = []
        random_mean_decomp = []
        for _ in range(200):
            sample = gnome.iloc[rng.choice(len(gnome), size=k, replace=False)]
            random_rates.append(float(sample["on_hull"].mean()))
            random_near.append(float(sample["near_hull"].mean()))
            random_mean_decomp.append(float(sample["Decomposition Energy Per Atom"].mean()))
        rows.append(
            {
                "top_k": k,
                "top_on_hull_rate": float(top["on_hull"].mean()),
                "random_on_hull_rate_mean": float(np.mean(random_rates)),
                "random_on_hull_rate_std": float(np.std(random_rates, ddof=1)),
                "on_hull_daf": float(top["on_hull"].mean() / max(base_on_hull, 1e-12)),
                "top_near_hull_rate": float(top["near_hull"].mean()),
                "random_near_hull_rate_mean": float(np.mean(random_near)),
                "near_hull_daf": float(top["near_hull"].mean() / max(base_near, 1e-12)),
                "top_mean_decomposition_energy": float(top["Decomposition Energy Per Atom"].mean()),
                "random_mean_decomposition_energy": float(np.mean(random_mean_decomp)),
            }
        )
    summary = pd.DataFrame(rows)
    return gnome, summary


def make_plots(
    predictive: pd.DataFrame,
    artifacts: dict[str, object],
    y: np.ndarray,
    trace: pd.DataFrame,
    active_summary: pd.DataFrame,
    gnome_summary: pd.DataFrame,
) -> None:
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.lineplot(
        data=trace,
        x="observed",
        y="stable_found",
        hue="policy",
        estimator="mean",
        errorbar=("ci", 95),
        ax=axes[0],
    )
    axes[0].set_title("Cumulative stable discoveries")
    axes[0].set_xlabel("Validated candidates")
    axes[0].set_ylabel("Stable / near-hull discoveries")
    sns.lineplot(
        data=trace,
        x="observed",
        y="discovery_acceleration_factor",
        hue="policy",
        estimator="mean",
        errorbar=("ci", 95),
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Discovery acceleration over random prevalence")
    axes[1].set_xlabel("Validated candidates")
    axes[1].set_ylabel("Discovery acceleration factor")
    fig.tight_layout()
    fig.savefig(FIGURES / "active_learning_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    order = active_summary.sort_values("mean_stable_found", ascending=False)["policy"]
    sns.barplot(data=active_summary, x="policy", y="mean_stable_found", order=order, ax=ax)
    ax.errorbar(
        x=np.arange(len(order)),
        y=active_summary.set_index("policy").loc[order, "mean_stable_found"],
        yerr=active_summary.set_index("policy").loc[order, "std_stable_found"],
        fmt="none",
        c="black",
        capsize=4,
    )
    ax.set_title("Final discoveries by policy")
    ax.set_xlabel("Policy")
    ax.set_ylabel("Mean stable discoveries at final budget")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIGURES / "active_final_discoveries.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(gnome_summary))
    width = 0.35
    ax.bar(x - width / 2, gnome_summary["top_on_hull_rate"], width, label="Top-ranked")
    ax.bar(x + width / 2, gnome_summary["random_on_hull_rate_mean"], width, label="Random")
    ax.set_xticks(x)
    ax.set_xticklabels(gnome_summary["top_k"].astype(str))
    ax.set_xlabel("Top K GNoME candidates")
    ax.set_ylabel("On-hull rate")
    ax.set_title("GNoME transfer ranking")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "gnome_transfer_topk.png", dpi=180)
    plt.close(fig)

    test_idx = artifacts["test_idx"]
    probs = artifacts["composition_plus_cell_test_probs"]
    calibration = pd.DataFrame({"prob": probs, "stable": y[test_idx]})
    calibration["decile"] = pd.qcut(calibration["prob"], 10, duplicates="drop")
    cal = calibration.groupby("decile", observed=True).agg(
        mean_prob=("prob", "mean"),
        observed_rate=("stable", "mean"),
        count=("stable", "size"),
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cal["mean_prob"], cal["observed_rate"], marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("Surrogate calibration by score decile")
    ax.set_xlabel("Mean predicted stability probability")
    ax.set_ylabel("Observed stable rate")
    fig.tight_layout()
    fig.savefig(FIGURES / "surrogate_calibration.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=predictive, x="feature_set", y="average_precision", ax=ax)
    ax.set_title("Predictive signal for stability")
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Average precision")
    fig.tight_layout()
    fig.savefig(FIGURES / "predictive_signal.png", dpi=180)
    plt.close(fig)


def reproducibility_check(
    X_struct: np.ndarray,
    X_comp: np.ndarray,
    y: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, object]:
    seed = config.active_seeds[0]
    rng = np.random.default_rng(seed)
    pool_idx = rng.choice(len(y), size=min(6000, len(y)), replace=False)
    initial_idx = rng.choice(len(pool_idx), size=200, replace=False)
    _, X_pool_scaled = make_scaler(X_struct[pool_idx])
    X_div = X_comp[pool_idx, :118]
    y_pool = y[pool_idx]
    mini_config = ExperimentConfig(
        seed=config.seed,
        device=config.device,
        train_batch_size=config.train_batch_size,
        predict_batch_size=config.predict_batch_size,
        mlp_epochs=6,
        eval_epochs=config.eval_epochs,
        mc_passes=3,
        candidate_pool_size=6000,
        initial_size=200,
        acquisition_batch_size=100,
        rounds=2,
        active_seeds=(seed,),
        policies=("greedy",),
    )
    first = pd.DataFrame(
        run_single_policy(X_pool_scaled, X_div, y_pool, initial_idx, "greedy", seed, mini_config)
    )
    second = pd.DataFrame(
        run_single_policy(X_pool_scaled, X_div, y_pool, initial_idx, "greedy", seed, mini_config)
    )
    same_trace = first[["round", "observed", "stable_found"]].equals(
        second[["round", "observed", "stable_found"]]
    )
    return {
        "policy": "greedy",
        "seed": seed,
        "same_trace": bool(same_trace),
        "first_final_stable_found": int(first.iloc[-1]["stable_found"]),
        "second_final_stable_found": int(second.iloc[-1]["stable_found"]),
    }


def write_environment(config: ExperimentConfig, elapsed_seconds: float | None = None) -> None:
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_memory_gib": round(props.total_memory / 1024**3, 2),
                }
            )
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_info": gpu_info,
        "sklearn_device_note": "PyTorch MLP surrogate used cuda:0 when available; tabular preprocessing/statistics used CPU.",
        "config": asdict(config),
        "elapsed_seconds": elapsed_seconds,
    }
    with open(RESULTS / "environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-feature-cache", action="store_true")
    args = parser.parse_args()

    ensure_workspace()
    config = ExperimentConfig()
    if not torch.cuda.is_available():
        config = ExperimentConfig(device="cpu")
    set_seed(config.seed)
    start = time.time()
    write_environment(config)

    mp = load_or_build_mp_features(force=args.force_feature_cache)
    X_comp = np.asarray(mp["X_comp"], dtype=np.float32)
    X_struct = np.asarray(mp["X_struct"], dtype=np.float32)
    y = np.asarray(mp["y"], dtype=np.int8)

    audit = audit_data(mp)
    with open(RESULTS / "data_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    predictive, artifacts = run_predictive_signal(X_comp, X_struct, y, config)
    predictive.to_csv(RESULTS / "predictive_metrics.csv", index=False)

    trace = run_active_discovery(X_struct, X_comp, y, config)
    trace.to_csv(RESULTS / "active_learning_trace.csv", index=False)
    active_summary, stat_tests = summarize_active_results(trace)
    active_summary.to_csv(RESULTS / "active_learning_summary.csv", index=False)
    stat_tests.to_csv(RESULTS / "stat_tests.csv", index=False)

    _, gnome_summary = run_gnome_transfer(
        artifacts["composition_only_model"],
        artifacts["composition_only_scaler"],
        config,
    )
    gnome_summary.to_csv(RESULTS / "gnome_transfer_summary.csv", index=False)

    repro = reproducibility_check(X_struct, X_comp, y, config)
    with open(RESULTS / "reproducibility_check.json", "w", encoding="utf-8") as f:
        json.dump(repro, f, indent=2)

    make_plots(predictive, artifacts, y, trace, active_summary, gnome_summary)

    elapsed = time.time() - start
    write_environment(config, elapsed_seconds=elapsed)
    print(json.dumps({"status": "complete", "elapsed_seconds": elapsed, "reproducible": repro}, indent=2))


if __name__ == "__main__":
    main()
