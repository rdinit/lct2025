from __future__ import annotations
import json, os
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from catboost import CatBoostRegressor

@dataclass
class TSConfig:
    time_col: str = "timestamp"
    group_col: str = "group_id"
    seq_col: str = "sequence_id"
    target_cols: Tuple[str, str] = ("bpm", "uterus")
    lags: List[int] = (1, 2, 3, 6, 12, 24)
    roll_windows: List[int] = (3, 6, 12)
    val_size: float = 0.2
    random_state: int = 42
    iterations: int = 3000
    learning_rate: float = 0.03
    depth: int = 6
    task_type: str = "CPU"
    verbose: int = 200

def _to_seconds_timestamp(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    dt = pd.to_datetime(s, errors="coerce")
    return dt.astype("int64") / 1e9

def _build_features_per_sequence(sdf: pd.DataFrame, targets: Tuple[str, str],
                                 lags: List[int], roll_windows: List[int]) -> pd.DataFrame:
    out = sdf.copy()
    t1, t2 = targets
    for L in lags:
        out[f"{t1}_lag_{L}"] = out[t1].shift(L)
        out[f"{t2}_lag_{L}"] = out[t2].shift(L)
    for W in roll_windows:
        out[f"{t1}_rollmean_{W}"] = out[t1].rolling(W, min_periods=1).mean().shift(1)
        out[f"{t2}_rollmean_{W}"] = out[t2].rolling(W, min_periods=1).mean().shift(1)
        out[f"{t1}_rollstd_{W}"]  = out[t1].rolling(W, min_periods=1).std(ddof=0).shift(1)
        out[f"{t2}_rollstd_{W}"]  = out[t2].rolling(W, min_periods=1).std(ddof=0).shift(1)
    return out

def make_supervised(df: pd.DataFrame, cfg: TSConfig) -> Tuple[pd.DataFrame, List[str]]:
    data = df.copy()
    data[cfg.time_col] = _to_seconds_timestamp(data[cfg.time_col])
    data = data.sort_values([cfg.seq_col, cfg.time_col]).reset_index(drop=True)
    feats_parts = []
    for seq, sdf in data.groupby(cfg.seq_col, sort=False):
        feats_parts.append(_build_features_per_sequence(sdf, cfg.target_cols, list(cfg.lags), list(cfg.roll_windows)))
    feats = pd.concat(feats_parts, axis=0)
    t1, t2 = cfg.target_cols
    feature_cols = []
    for L in cfg.lags:
        feature_cols += [f"{t1}_lag_{L}", f"{t2}_lag_{L}"]
    for W in cfg.roll_windows:
        feature_cols += [f"{t1}_rollmean_{W}", f"{t2}_rollmean_{W}", f"{t1}_rollstd_{W}", f"{t2}_rollstd_{W}"]
    feats = feats.dropna(subset=feature_cols + list(cfg.target_cols)).reset_index(drop=True)
    return feats, feature_cols

def split_train_valid_indices(X: pd.DataFrame, cfg: TSConfig) -> Tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.val_size, random_state=cfg.random_state)
    tr_idx, va_idx = next(gss.split(X, groups=X[cfg.group_col]))
    return tr_idx, va_idx

def save_artifacts(model: CatBoostRegressor, meta: Dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model.save_model(os.path.join(out_dir, "model.cbm"))
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_artifacts(model_dir: str) -> Tuple[CatBoostRegressor, Dict]:
    model = CatBoostRegressor()
    model.load_model(os.path.join(model_dir, "model.cbm"))
    with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

def train_multivariate_forecaster(df: pd.DataFrame, cfg: TSConfig, out_dir: str | None = None) -> Tuple[CatBoostRegressor, Dict]:
    feats, feature_cols = make_supervised(df, cfg)
    X = feats[feature_cols]
    y = feats[list(cfg.target_cols)].values
    tr_idx, va_idx = split_train_valid_indices(feats, cfg)
    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_va, y_va = X.iloc[va_idx], y[va_idx]
    model = CatBoostRegressor(
        loss_function="MultiRMSE",
        eval_metric="MultiRMSE",
        iterations=cfg.iterations,
        learning_rate=cfg.learning_rate,
        depth=cfg.depth,
        task_type=cfg.task_type,
        random_seed=cfg.random_state,
        od_type="Iter",
        od_wait=400,
        verbose=cfg.verbose,
    )
    model.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    meta = {"config": asdict(cfg), "feature_cols": feature_cols}
    if out_dir is not None:
        save_artifacts(model, meta, out_dir)
    return model, meta
