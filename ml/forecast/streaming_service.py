from __future__ import annotations
import json, os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

@dataclass
class _MetaCfg:
    time_col: str
    group_col: str
    seq_col: str
    target_cols: Tuple[str, str]
    lags: List[int]
    roll_windows: List[int]
    feature_cols: List[str]

class StreamingForecaster:
    def __init__(self, model_dir: str):
        self.model = CatBoostRegressor()
        self.model.load_model(os.path.join(model_dir, "model.cbm"))
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

    def _ready(self) -> bool:
        return (self.last_ts_sec is not None) and all(len(self.buffers[t]) >= self.max_lag for t in self.buffers)

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

    def update_one(self, ts_sec: float, bpm: float, uterus: float, group_id=None, sequence_id=None):
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

    def forecast(self, horizon: int) -> pd.DataFrame:
        if not self._ready():
            raise RuntimeError("Forecaster not ready; warm_start or accumulate more points first")
        tb1 = list(self.buffers[self.meta.target_cols[0]])
        tb2 = list(self.buffers[self.meta.target_cols[1]])
        preds = []
        for h in range(1, horizon + 1):
            row = self._feature_row(tb1, tb2)
            X = pd.DataFrame([row], columns=self.meta.feature_cols)
            y_hat = self.model.predict(X)
            y1, y2 = float(y_hat[0][0]), float(y_hat[0][1])
            tb1.append(y1); tb2.append(y2)
            preds.append({
                self.meta.time_col: self.last_ts_sec + h * self.step_sec,
                self.meta.group_col: self.group_id,
                self.meta.seq_col: self.sequence_id,
                self.meta.target_cols[0]: y1,
                self.meta.target_cols[1]: y2,
                "h": h
            })
        return pd.DataFrame(preds)

class ForecastService:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        tmp = StreamingForecaster(model_dir)
        self.max_lag = tmp.max_lag
        self.template_meta = tmp.meta
        self._states: Dict[Tuple[str, str], Dict] = {}

    def _get_or_create_state(self, group_id, sequence_id) -> Dict:
        key = (str(group_id), str(sequence_id))
        st = self._states.get(key)
        if st is None:
            st = {
                "sf": StreamingForecaster(self.model_dir),
                "hist": []
            }
            self._states[key] = st
        return st

    @staticmethod
    def _ts_to_sec(ts_val) -> float:
        return float(ts_val)

    def _forecast_to_string(self, forecast_list: List[Dict]) -> str:
        # каждый прогноз: timestamp,group_id,sequence_id,bpm,uterus,h
        # прогнозы разделяются символом ';'
        forecast_string = ""
        for pred in forecast_list:
            print(pred)
            forecast_string += f"{pred['timestamp']},{pred['bpm']},{pred['uterus']}\n"
        return forecast_string

    def process_message(self, msg: str, horizon: int) -> str:
        # timestamp,bpm,uterus,group_id,sequence_id

        current_gid = "0"
        current_sid = "0"

        data = []

        cols = msg.strip().split('\n')

        for col in cols:
            parts = col.split(',')
            data.append({
                "ts": self._ts_to_sec(parts[0]),
                "bpm": float(parts[1]),
                "ut": float(parts[2]),
                "gid": current_gid,
                "sid": current_sid
            })       


        st = self._get_or_create_state(current_gid, current_sid)
        sf: StreamingForecaster = st["sf"]

        if not sf._ready():
            ready_i = -1
            for i in range(len(data)):
                col = data[i]
                st["hist"].append({
                    sf.meta.time_col: col["ts"], 
                    "bpm": col["bpm"], 
                    "uterus": col["ut"],
                    sf.meta.group_col: current_gid, 
                    sf.meta.seq_col: current_sid
                    }
                )
                if len(st["hist"]) >= self.max_lag:
                    hist_df = pd.DataFrame(st["hist"])
                    sf.warm_start(hist_df)
                    ready_i = i
                    break
            if ready_i != -1:
                if ready_i < len(data) - 1:
                    data = data[ready_i + 1:]
                else:
                    data = []
            else:
                needed = max(0, self.max_lag - len(st["hist"]))
                return {
                    "ready": False,
                    "needed": needed,
                    "forecast": []
                }
        
        
        for col in data:
            sf.update_one(ts_sec=col["ts"], bpm=col["bpm"], uterus=col["ut"], group_id=current_gid, sequence_id=current_sid)


        pred_df = sf.forecast(horizon=horizon)
        forecast_list = pred_df.to_dict(orient="records")


        print(forecast_list)
        forecast_str = self._forecast_to_string(forecast_list)

        return {
            "ready": True,
            "needed": 0,
            "forecast": forecast_str
        }

# svc = ForecastService(model_dir="artifacts")
# out = svc.process_message(msg, horizon=K)
# # Парсинг ответа: "ready:true,forecast:ts1,gid,sid,bpm1,ut1,1;ts2,gid,sid,bpm2,ut2,2,needed:0"