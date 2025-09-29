from __future__ import annotations
import os, json, pickle
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import IsolationForest

@dataclass
class ADConfig:
    time_col: str = "timestamp"
    group_col: str = "group_id"
    seq_col: str = "sequence_id"
    target_cols: Tuple[str, str] = ("bpm", "uterus")
    lags: List[int] = (1, 2, 3, 6, 12, 24)
    roll_windows: List[int] = (3, 6, 12)
    val_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    max_samples: str | int | float = "auto"
    contamination: str | float = "auto"
    max_features: float = 1.0
    n_jobs: int | None = None
    verbose: int = 0

def _to_seconds_timestamp(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    dt = pd.to_datetime(s, errors="coerce")
    return dt.astype("int64") / 1e9

def _build_features_per_sequence(
    sdf: pd.DataFrame, targets: Tuple[str, str], lags: List[int], roll_windows: List[int]
) -> pd.DataFrame:
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

def make_supervised(df: pd.DataFrame, cfg: ADConfig) -> Tuple[pd.DataFrame, List[str]]:
    data = df.copy()
    data[cfg.time_col] = _to_seconds_timestamp(data[cfg.time_col])
    data = data.sort_values([cfg.seq_col, cfg.time_col]).reset_index(drop=True)
    parts = []
    for _, sdf in data.groupby(cfg.seq_col, sort=False):
        parts.append(_build_features_per_sequence(sdf, cfg.target_cols, list(cfg.lags), list(cfg.roll_windows)))
    feats = pd.concat(parts, axis=0)
    t1, t2 = cfg.target_cols
    feature_cols: List[str] = []
    for L in cfg.lags:
        feature_cols += [f"{t1}_lag_{L}", f"{t2}_lag_{L}"]
    for W in cfg.roll_windows:
        feature_cols += [f"{t1}_rollmean_{W}", f"{t2}_rollmean_{W}",
                         f"{t1}_rollstd_{W}",  f"{t2}_rollstd_{W}"]
    feats = feats.dropna(subset=feature_cols + list(cfg.target_cols)).reset_index(drop=True)
    return feats, feature_cols

def split_by_group(X: pd.DataFrame, cfg: ADConfig) -> Tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.val_size, random_state=cfg.random_state)
    tr_idx, va_idx = next(gss.split(X, groups=X[cfg.group_col]))
    return tr_idx, va_idx

def train_isolation_forest(
    df: pd.DataFrame, cfg: ADConfig, out_dir: str | None = None
) -> Tuple[IsolationForest, Dict]:
    feats, feature_cols = make_supervised(df, cfg)
    X = feats[feature_cols].astype(np.float32)
    tr_idx, _ = split_by_group(feats, cfg)
    X_tr = X.iloc[tr_idx]
    model = IsolationForest(
        n_estimators=cfg.n_estimators,
        max_samples=cfg.max_samples,
        contamination=cfg.contamination,
        max_features=cfg.max_features,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        verbose=cfg.verbose,
    )
    model.fit(X_tr)
    meta = {"config": asdict(cfg), "feature_cols": feature_cols}
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "model_if.pkl"), "wb") as f:
            pickle.dump(model, f)
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return model, meta

def load_if_artifacts(model_dir: str) -> Tuple[IsolationForest, Dict]:
    with open(os.path.join(model_dir, "model_if.pkl"), "rb") as f:
        model: IsolationForest = pickle.load(f)
    with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta