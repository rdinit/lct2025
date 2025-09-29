from __future__ import annotations
import os, json, pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

@dataclass
class _MetaCfg:
    time_col: str
    group_col: str
    seq_col: str
    target_cols: Tuple[str, str]
    lags: List[int]
    roll_windows: List[int]
    feature_cols: List[str]

class StreamingAnomalyDetector:
    def __init__(self, model_dir: str, external_forecast_fn: Optional[Callable[[List[float], List[float], int], Tuple[List[float], List[float]]]] = None):
        with open(os.path.join(model_dir, "model_if.pkl"), "rb") as f:
            self.model: IsolationForest = pickle.load(f)
        with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        cfg = meta["config"]
        self.meta = _MetaCfg(
            time_col=cfg["time_col"], group_col=cfg["group_col"], seq_col=cfg["seq_col"],
            target_cols=tuple(cfg["target_cols"]), lags=list(cfg["lags"]),
            roll_windows=list(cfg["roll_windows"]), feature_cols=list(meta["feature_cols"])
        )
        self.max_lag = max(self.meta.lags) if self.meta.lags else 1
        self.buffers: Dict[str, List[float]] = {self.meta.target_cols[0]: [], self.meta.target_cols[1]: []}
        self.last_ts_sec: Optional[float] = None
        self.step_sec: float = 1.0
        self.group_id = None
        self.sequence_id = None
        self.external_forecast_fn = external_forecast_fn

    def _ready(self) -> bool:
        return (self.last_ts_sec is not None) and all(len(self.buffers[t]) >= self.max_lag for t in self.buffers)

    @staticmethod
    def _ts_to_sec(ts_val) -> float:
        if isinstance(ts_val, (int, float, np.floating, np.integer)):
            return float(ts_val)
        return pd.to_datetime(ts_val).value / 1e9

    def warm_start(self, history_df: pd.DataFrame):
        tcol = self.meta.time_col
        df = history_df.sort_values(tcol).reset_index(drop=True).copy()
        if not pd.api.types.is_numeric_dtype(df[tcol]):
            df[tcol] = pd.to_datetime(df[tcol]).astype("int64") / 1e9
        if len(df) >= 2:
            diffs = np.diff(df[tcol].astype(float).to_numpy())
            step = np.median(diffs)
            self.step_sec = float(step) if np.isfinite(step) and step > 0 else 1.0
        else:
            self.step_sec = 1.0
        self.last_ts_sec = float(df[tcol].iloc[-1]) if len(df) else None
        self.group_id = df[self.meta.group_col].iloc[-1] if self.meta.group_col in df else None
        self.sequence_id = df[self.meta.seq_col].iloc[-1] if self.meta.seq_col in df else None
        t1, t2 = self.meta.target_cols
        h1 = df[t1].astype(float).tolist()
        h2 = df[t2].astype(float).tolist()
        if len(h1) < self.max_lag or len(h2) < self.max_lag:
            raise ValueError(f"Need at least {self.max_lag} historical points per target for warm start")
        self.buffers[t1] = h1[-self.max_lag:]
        self.buffers[t2] = h2[-self.max_lag:]

    def update_one(self, ts, bpm, uterus, group_id=None, sequence_id=None):
        ts_sec = self._ts_to_sec(ts)
        if self.last_ts_sec is not None:
            dt = float(ts_sec) - float(self.last_ts_sec)
            if np.isfinite(dt) and dt > 0:
                self.step_sec = float(dt)
        self.last_ts_sec = float(ts_sec)
        t1, t2 = self.meta.target_cols
        self.buffers[t1].append(float(bpm))
        self.buffers[t2].append(float(uterus))
        keep = max(self.max_lag, max(self.meta.roll_windows) if self.meta.roll_windows else 1)
        for k in self.buffers:
            if len(self.buffers[k]) > keep * 2:
                self.buffers[k] = self.buffers[k][-keep:]
        if group_id is not None: self.group_id = group_id
        if sequence_id is not None: self.sequence_id = sequence_id

    def _feature_row(self, b1: List[float], b2: List[float]) -> Dict[str, float]:
        t1, t2 = self.meta.target_cols
        row = {}
        for L in self.meta.lags:
            row[f"{t1}_lag_{L}"] = b1[-L]
            row[f"{t2}_lag_{L}"] = b2[-L]
        for W in self.meta.roll_windows:
            c1 = b1[-W:] if W <= len(b1) else b1
            c2 = b2[-W:] if W <= len(b2) else b2
            row[f"{t1}_rollmean_{W}"] = float(np.mean(c1)) if len(c1) else np.nan
            row[f"{t2}_rollmean_{W}"] = float(np.mean(c2)) if len(c2) else np.nan
            row[f"{t1}_rollstd_{W}"]  = float(np.std(c1, ddof=0)) if len(c1) else np.nan
            row[f"{t2}_rollstd_{W}"]  = float(np.std(c2, ddof=0)) if len(c2) else np.nan
        return row

    def _roll_forward_values(self, b1: List[float], b2: List[float], horizon: int) -> Tuple[List[float], List[float]]:
        if len(b1) == 0 or len(b2) == 0:
            return b1, b2
        v1, v2 = b1[-1], b2[-1]
        ext1 = b1 + [v1] * horizon
        ext2 = b2 + [v2] * horizon
        return ext1, ext2

    def forecast(self, horizon: int) -> pd.DataFrame:
        if not self._ready():
            raise RuntimeError("Detector not ready; warm_start or accumulate more points first")
        src_b1 = list(self.buffers[self.meta.target_cols[0]])
        src_b2 = list(self.buffers[self.meta.target_cols[1]])

        if self.external_forecast_fn is not None:
            fb1, fb2 = self.external_forecast_fn(src_b1, src_b2, horizon)
            tb1 = list(fb1)
            tb2 = list(fb2)
        else:
            tb1, tb2 = self._roll_forward_values(src_b1, src_b2, horizon)

        rows = []
        for h in range(1, horizon + 1):
            b1h = tb1[:len(src_b1) + h]
            b2h = tb2[:len(src_b2) + h]
            feat = self._feature_row(b1h, b2h)
            rows.append(feat)

        X = pd.DataFrame(rows, columns=self.meta.feature_cols).astype(np.float32)
        y_pred_pm1 = self.model.predict(X)
        flags = (y_pred_pm1 == -1).astype(int).tolist()
        scores = self.model.decision_function(X).tolist()

        preds = []
        for h in range(1, horizon + 1):
            preds.append({
                self.meta.time_col: self.last_ts_sec + h * self.step_sec,
                self.meta.group_col: self.group_id,
                self.meta.seq_col: self.sequence_id,
                "anomaly_flag": int(flags[h-1]),
                "anomaly_score": float(scores[h-1]),
                "h": h
            })
        return pd.DataFrame(preds)

class AnomalyService:
    def __init__(self, model_dir: str, external_forecast_fn: Optional[Callable[[List[float], List[float], int], Tuple[List[float], List[float]]]] = None):
        self.template = StreamingAnomalyDetector(model_dir, external_forecast_fn)
        self.max_lag = self.template.max_lag
        self.model_dir = model_dir
        self.external_forecast_fn = external_forecast_fn
        self._states: Dict[Tuple[str, str], Dict] = {}

    def _get_or_create(self, gid: str, sid: str) -> Dict:
        key = (gid, sid)
        st = self._states.get(key)
        if st is None:
            st = {"det": StreamingAnomalyDetector(self.model_dir, self.external_forecast_fn), "hist": []}
            self._states[key] = st
        return st

    def process_message(self, msg: Dict, horizon: int) -> Dict:
        ts = StreamingAnomalyDetector._ts_to_sec(msg["timestamp"])
        bpm = float(msg["bpm"]); ut = float(msg["uterus"])
        gid = str(msg["group_id"]); sid = str(msg["sequence_id"])
        st = self._get_or_create(gid, sid)
        det: StreamingAnomalyDetector = st["det"]

        if not det._ready():
            st["hist"].append({det.meta.time_col: ts, det.meta.group_col: gid, det.meta.seq_col: sid,
                               det.meta.target_cols[0]: bpm, det.meta.target_cols[1]: ut})
            if len(st["hist"]) >= self.max_lag:
                det.warm_start(pd.DataFrame(st["hist"]))
            return {"ready": det._ready(), "forecast": [], "needed": max(0, self.max_lag - len(st["hist"]))}

        det.update_one(ts, bpm, ut, gid, sid)
        pred = det.forecast(horizon=horizon)
        return {"ready": True, "forecast": pred.to_dict(orient="records"), "needed": 0}