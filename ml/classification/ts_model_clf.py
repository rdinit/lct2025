from __future__ import annotations
import json, os, pickle
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
import antropy as ant
from scipy.signal import find_peaks
import time
import warnings
from scipy.signal import savgol_filter
warnings.filterwarnings("ignore")


EPS = 1e-12

@dataclass
class CLFConfig:
    time_col: str = "timestamp"
    group_col: str = "group_id"
    seq_col: str = "sequence_id"
    target_cols: Tuple[str, str] = ("bpm", "uterus")
    target_col: str = "target"
    roll_windows: List[int] = (5, 15, 30)
    val_size: float = 0.2
    random_state: int = 42
    iterations: int = 1000
    learning_rate: float = 0.03
    depth: int = 6
    min_seq_length: int = 1000
    task_type: str = "CPU"
    verbose: int = 200
    n_splits: int = 5

def _to_seconds_timestamp(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    dt = pd.to_datetime(s, errors="coerce")
    return dt.astype("int64") / 1e9

def _svd_entropy(arr):
    try:
        arr = np.asarray(arr, dtype=float)
        if len(arr) < 2:
            return np.nan
        return ant.svd_entropy(arr, normalize=True)
    except Exception:
        return np.nan

def _petrosian_fd(arr):
    try:
        arr = np.asarray(arr, dtype=float)
        if len(arr) < 3:
            return np.nan
        return ant.petrosian_fd(arr)
    except Exception:
        return np.nan

def _peak_count_scipy(arr):
    try:
        arr = np.asarray(arr, dtype=float)
        if len(arr) < 3 or np.any(np.isnan(arr)):
            return 0.0
        peaks, _ = find_peaks(arr, height=0)
        return float(len(peaks))
    except Exception:
        return 0.0

def _quantile_func(q: float):
    def _q(s: pd.Series) -> float:
        return s.quantile(q)
    _q.__name__ = f"q{int(q*100)}"
    return _q

SEQ_AGGS = ["mean", "std", "min", "max", _quantile_func(0.10), _quantile_func(0.50), _quantile_func(0.90)]

def build_rolling_features_for_group(g: pd.DataFrame, ts_cols: List[str], windows: List[int]) -> pd.DataFrame:
    g = g.sort_values("timestamp") if "timestamp" in g.columns else g.copy()
    out = pd.DataFrame(index=g.index)

    for col in ts_cols:
        if col not in g.columns:
            continue
        x = g[col].astype(float)

        out[f"{col}__diff1"]   = x.diff(1)
        out[f"{col}__cumsum"]  = x.cumsum()

        for w in windows:
            roll = x.rolling(window=w, min_periods=1)
            mean = roll.mean(); std = roll.std(); rmin = roll.min(); rmax = roll.max()
            q25  = roll.quantile(0.25); q75 = roll.quantile(0.75)
            rsum = roll.sum()

            out[f"{col}__w{w}__mean"] = mean
            out[f"{col}__w{w}__std"]  = std
            out[f"{col}__w{w}__min"]  = rmin
            out[f"{col}__w{w}__max"]  = rmax
            out[f"{col}__w{w}__q25"]  = q25
            out[f"{col}__w{w}__q75"]  = q75
            out[f"{col}__w{w}__sum"]  = rsum

            out[f"{col}__w{w}__dev_mean"] = x - mean
            out[f"{col}__w{w}__dev_q25"]  = x - q25
            out[f"{col}__w{w}__dev_q75"]  = x - q75

            out[f"{col}__w{w}__svd_entropy"] = roll.apply(_svd_entropy, raw=True)
            out[f"{col}__w{w}__petrosian_fd"] = roll.apply(_petrosian_fd, raw=True)           
            out[f"{col}__w{w}__num_peaks"] = roll.apply(_peak_count_scipy, raw=True)
    return out

def kalman_1d(z: np.ndarray,
              Q: float = 1e-5,
              R: float = 1e-2,
              x0: float | None = None,
              P0: float = 1.0) -> np.ndarray:
    n = len(z)
    if n == 0:
        return np.array([])
    x_hat = np.empty(n)
    x_est = z[0] if x0 is None else x0
    P = P0
    for k in range(n):
        x_pred = x_est
        P = P + Q
        K = P / (P + R)
        x_est = x_pred + K * (z[k] - x_pred)
        P = (1.0 - K) * P
        x_hat[k] = x_est
    return x_hat

def apply_kalman_to_series(series: pd.Series, Q: float = 1e-5, R: float = 1e-2) -> pd.Series:
    arr = np.asarray(series, dtype=float)
    if len(arr) == 0:
        return series
    filtered = kalman_1d(arr, Q=Q, R=R)
    return pd.Series(filtered, index=series.index)

def apply_savgol_to_series(series: pd.Series, window_length: int = 11, polyorder: int = 3) -> pd.Series:
    arr = np.asarray(series, dtype=float)
    if len(arr) < window_length:
        return series  
    try:
        filtered = savgol_filter(arr, window_length=window_length, polyorder=polyorder)
        return pd.Series(filtered, index=series.index)
    except Exception:
        return series 

def process_single_sequence(group: pd.DataFrame, ts_cols: List[str], windows: List[int]) -> pd.Series:
    if len(group) < CLFConfig.min_seq_length:
        return None
    group = group.head(CLFConfig.min_seq_length).copy()
    
    for col in ts_cols:
            if col in group.columns:
                # Option 1: Kalman filter
                # group[col] = apply_kalman_to_series(group[col], Q=1e-5, R=1e-2)
                # Option 2: Savitzky-Golay (uncomment if preferred or use both)
                group[col] = apply_savgol_to_series(group[col], window_length=11, polyorder=3)
    
    step_feats = build_rolling_features_for_group(group, ts_cols=ts_cols, windows=windows)
    
    step_feats["sequence_id"] = group["sequence_id"].iloc[0]
    step_feats["group_id"] = group["group_id"].iloc[0]
    tgt_mode = group["target"].mode(dropna=False)
    step_feats["target"] = tgt_mode.iloc[0] if len(tgt_mode) else group["target"].iloc[0]
    
    feat_cols = [c for c in step_feats.columns if c not in ["sequence_id", "group_id", "target"]]
    agg_dict = {c: SEQ_AGGS for c in feat_cols}
    seq_agg = step_feats.agg(agg_dict)
    
    result = seq_agg.to_dict()
    result["sequence_id"] = group["sequence_id"].iloc[0]
    result["group_id"] = group["group_id"].iloc[0]
    result["target"] = step_feats["target"].iloc[0]
    
    return pd.Series(result)

def flatten_feature_series(feature_series: pd.Series) -> pd.DataFrame:
    expanded_dfs = []
    
    for col_name, value in feature_series.items():
        if col_name in ["sequence_id", "group_id", "target"]:
            continue
            
        if isinstance(value, dict):
            for stat_name, stat_value in value.items():
                new_col_name = f"{col_name}__{stat_name}"
                expanded_dfs.append(pd.DataFrame({new_col_name: [stat_value]}))
        else:
            expanded_dfs.append(pd.DataFrame({col_name: [value]}))
    
    flattened_df = pd.concat(expanded_dfs, axis=1) if expanded_dfs else pd.DataFrame()
    return flattened_df

def make_supervised(df: pd.DataFrame, cfg: CLFConfig) -> Tuple[pd.DataFrame, List[str]]:
    data = df.copy()
    data[cfg.time_col] = _to_seconds_timestamp(data[cfg.time_col])
    data = data.sort_values([cfg.seq_col, cfg.time_col]).reset_index(drop=True)
    
    seq_features_list = []
    for seq_id, group in data.groupby(cfg.seq_col, sort=False):
        seq_feat = process_single_sequence(group, list(cfg.target_cols), list(cfg.roll_windows))
        if seq_feat is not None:
            seq_features_list.append(seq_feat)
    
    seq_features = pd.DataFrame(seq_features_list).reset_index(drop=True)
    
    meta_cols = ["sequence_id", "group_id", "target"]
    final_features = [seq_features[meta_cols]]
    
    for col in seq_features.columns:
        if col not in meta_cols:
            if seq_features[col].apply(lambda x: isinstance(x, dict)).any():
                expanded = pd.json_normalize(seq_features[col]).add_prefix(f"{col}__")
                final_features.append(expanded)
            else:
                final_features.append(seq_features[[col]])
    
    feats = pd.concat(final_features, axis=1)
    
    feature_cols = [c for c in feats.columns if c not in meta_cols]
    
    feats = feats.dropna(subset=[cfg.target_col]).reset_index(drop=True)
    
    return feats, feature_cols

def split_train_valid_indices(X: pd.DataFrame, y: pd.Series, groups: pd.Series, cfg: CLFConfig) -> Tuple[np.ndarray, np.ndarray]:
    skf = StratifiedGroupKFold(n_splits=int(1/cfg.val_size), shuffle=True, random_state=cfg.random_state)
    train_idx, valid_idx = next(skf.split(X, y, groups=groups))
    return train_idx, valid_idx

def save_artifacts(model: CatBoostClassifier, label_encoder: LabelEncoder, meta: Dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model.save_model(os.path.join(out_dir, "model.cbm"))
    
    with open(os.path.join(out_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def load_artifacts(model_dir: str) -> Tuple[CatBoostClassifier, LabelEncoder, Dict]:
    model = CatBoostClassifier()
    model.load_model(os.path.join(model_dir, "model.cbm"))
    
    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    
    with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    return model, label_encoder, meta

def train_classifier(df: pd.DataFrame, cfg: CLFConfig, out_dir: str | None = None) -> Tuple[CatBoostClassifier, LabelEncoder, Dict]:
    feats, feature_cols = make_supervised(df, cfg)
    
    X = feats[feature_cols]
    y = feats[cfg.target_col].astype(str)
    groups = feats[cfg.group_col].astype(str)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    is_binary = len(np.unique(y_enc)) == 2
    
    train_idx, valid_idx = split_train_valid_indices(X, y_enc, groups, cfg)
    X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
    y_tr, y_val = y_enc[train_idx], y_enc[valid_idx]
    
    params = dict(
        random_state=cfg.random_state,
        verbose=cfg.verbose,
        loss_function="Logloss" if is_binary else "MultiClass",
        eval_metric="F1" if is_binary else "TotalF1",
        iterations=cfg.iterations,
        learning_rate=cfg.learning_rate,
        depth=cfg.depth,
        task_type=cfg.task_type,
        od_type="Iter",
        od_wait=100,
    )
    
    model = CatBoostClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=100
    )
    
    meta = {
        "config": asdict(cfg),
        "feature_cols": feature_cols,
        "classes": list(le.classes_),
        "is_binary": is_binary
    }
    
    if out_dir is not None:
        save_artifacts(model, le, meta, out_dir)
    
    return model, le, meta

def train_classifier_cv(df: pd.DataFrame, cfg: CLFConfig) -> Dict:
    feats, feature_cols = make_supervised(df, cfg)
    
    X = feats[feature_cols]
    y = feats[cfg.target_col].astype(str)
    groups = feats[cfg.group_col].astype(str)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    is_binary = len(np.unique(y_enc)) == 2
    
    results = {
        "fold_f1": [],
        "oof_pred": None,
        "label_encoder": le,
        "models": [],
        "feature_importances_": None,
        "classes_": list(le.classes_),
    }
    
    oof = np.full(shape=(len(y_enc),), fill_value=np.nan)
    skf = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    importances = []
    
    params = dict(
        random_state=cfg.random_state,
        verbose=False,
        loss_function="Logloss" if is_binary else "MultiClass",
        eval_metric="F1" if is_binary else "TotalF1",
        iterations=cfg.iterations,
        learning_rate=cfg.learning_rate,
        depth=cfg.depth,
        task_type=cfg.task_type,
    )
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y_enc, groups=groups), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
        y_tr, y_val = y_enc[train_idx], y_enc[valid_idx]
        
        model = CatBoostClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=100
        )
        results["models"].append(model)
        
        if is_binary:
            proba = model.predict_proba(X_val)[:, 1]
            pred = (proba >= 0.5).astype(int)
        else:
            pred = model.predict(X_val)
            if pred.ndim > 1:
                pred = pred.argmax(axis=1)
            pred = pred.astype(int)
        
        f1 = f1_score(y_val, pred, average="macro")
        results["fold_f1"].append(f1)
        print(f"Fold {fold}: F1(macro) = {f1:.4f}")
        
        oof[valid_idx] = pred
        importances.append(model.get_feature_importance())
    
    mean_f1 = np.nanmean(results["fold_f1"]) if results["fold_f1"] else np.nan
    print(f"\nMean CV F1(macro): {mean_f1:.4f}")
    
    results["oof_pred"] = oof
    if importances:
        results["feature_importances_"] = np.mean(np.vstack(importances), axis=0)
    
    return results