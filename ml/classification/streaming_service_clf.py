from __future__ import annotations
import json, os, pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from ts_model_clf import process_single_sequence, flatten_feature_series

@dataclass
class _MetaCfg:
    time_col: str
    group_col: str
    seq_col: str
    target_cols: Tuple[str, str]
    target_col: str
    roll_windows: List[int]
    feature_cols: List[str]
    classes: List[str]
    is_binary: bool

class StreamingTimeSeriesClassifier:
    def __init__(self, model_dir: str):
        self.model = CatBoostClassifier()
        self.model.load_model(os.path.join(model_dir, "model.cbm"))
        
        with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
            self.label_encoder: LabelEncoder = pickle.load(f)
        
        with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        cfg = meta["config"]
        self.meta = _MetaCfg(
            time_col=cfg["time_col"],
            group_col=cfg["group_col"], 
            seq_col=cfg["seq_col"],
            target_cols=tuple(cfg["target_cols"]),
            target_col=cfg["target_col"],
            roll_windows=list(cfg["roll_windows"]),
            feature_cols=list(meta["feature_cols"]),
            classes=list(meta["classes"]),
            is_binary=meta["is_binary"]
        )
        
        self.max_window = max(self.meta.roll_windows) if self.meta.roll_windows else 30
        self.buffers: Dict[str, List[float]] = {
            self.meta.target_cols[0]: [],  # bpm
            self.meta.target_cols[1]: []   # uterus
        }
        self.timestamps: List[float] = []
        self.group_id = None
        self.sequence_id = None
        
    def _ready(self) -> bool:
        return (len(self.timestamps) >= self.max_window and 
                all(len(self.buffers[t]) >= self.max_window for t in self.buffers))
    
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
        
        t1, t2 = self.meta.target_cols
        self.timestamps = df[tcol].astype(float).tolist()[-self.max_window:]
        self.buffers[t1] = df[t1].astype(float).tolist()[-self.max_window:]
        self.buffers[t2] = df[t2].astype(float).tolist()[-self.max_window:]
        
        self.group_id = df[self.meta.group_col].iloc[-1] if self.meta.group_col in df else None
        self.sequence_id = df[self.meta.seq_col].iloc[-1] if self.meta.seq_col in df else None
        
        if len(self.timestamps) < self.max_window:
            raise ValueError(f"Need at least {self.max_window} historical points for warm start")
    
    def update_one(self, ts, bpm, uterus, group_id=None, sequence_id=None):
        ts_sec = self._ts_to_sec(ts)
        
        self.timestamps.append(float(ts_sec))
        t1, t2 = self.meta.target_cols
        self.buffers[t1].append(float(bpm))
        self.buffers[t2].append(float(uterus))
        
        keep_size = self.max_window * 2
        if len(self.timestamps) > keep_size:
            self.timestamps = self.timestamps[-keep_size:]
            for k in self.buffers:
                self.buffers[k] = self.buffers[k][-keep_size:]
        
        if group_id is not None: 
            self.group_id = group_id
        if sequence_id is not None: 
            self.sequence_id = sequence_id
    
    def _extract_features_from_timeseries(self) -> pd.DataFrame:
        if not self._ready():
            raise RuntimeError("Classifier not ready; need more data points")
        
        t1, t2 = self.meta.target_cols
        ts_data = pd.DataFrame({
            self.meta.time_col: self.timestamps,
            t1: self.buffers[t1],
            t2: self.buffers[t2],
            self.meta.seq_col: self.sequence_id,
            self.meta.group_col: self.group_id,
            self.meta.target_col: "unknown"
        })
                
        features_series = process_single_sequence(ts_data, list(self.meta.target_cols), self.meta.roll_windows)
        features_df = flatten_feature_series(features_series)
        
        missing_cols = set(self.meta.feature_cols) - set(features_df.columns)
        for col in missing_cols:
            features_df[col] = np.nan
        
        return features_df[self.meta.feature_cols]
    
    def classify_timeseries(self) -> Dict[str, any]:
        if not self._ready():
            raise RuntimeError("Classifier not ready; warm_start or accumulate more points first")
        
        X = self._extract_features_from_timeseries()
        
        pred_proba = self.model.predict_proba(X.iloc[[0]])[0]
        pred_class_idx = np.argmax(pred_proba)
        pred_class = self.label_encoder.classes_[pred_class_idx]
        
        result = {
            self.meta.time_col: self.timestamps[-1],
            self.meta.group_col: self.group_id,
            self.meta.seq_col: self.sequence_id,
            "bpm_current": self.buffers[self.meta.target_cols[0]][-1],
            "uterus_current": self.buffers[self.meta.target_cols[1]][-1],
            "predicted_class": pred_class,
            "class_probabilities": {cls: float(prob) for cls, prob in zip(self.label_encoder.classes_, pred_proba)},
            "confidence": float(np.max(pred_proba))
        }
        
        return result

class ClassificationService:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        template = StreamingTimeSeriesClassifier(model_dir)
        self.max_window = template.max_window
        self.template_meta = template.meta
        self._states: Dict[Tuple[str, str], Dict] = {}
    
    def _get_or_create_state(self, group_id: str, sequence_id: str) -> Dict:
        key = (str(group_id), str(sequence_id))
        state = self._states.get(key)
        if state is None:
            state = {
                "classifier": StreamingTimeSeriesClassifier(self.model_dir),
                "history": []
            }
            self._states[key] = state
        return state
    
    @staticmethod
    def _ts_to_sec(ts_val) -> float:
        return float(ts_val)
    
    def _classification_to_string(self, classification: Dict) -> str:
        prob_values = [f"{prob:.4f}" for prob in classification["class_probabilities"].values()]
        return (f"{classification[self.template_meta.time_col]},"
                f"{classification[self.template_meta.group_col]},"
                f"{classification[self.template_meta.seq_col]},"
                f"{classification['bpm_current']},"
                f"{classification['uterus_current']},"
                f"{classification['predicted_class']},"
                f"{classification['confidence']:.4f},"
                f"{','.join(prob_values)}")
    
    def process_message(self, msg: str) -> str:
        
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

        state = self._get_or_create_state(current_gid, current_sid)
        classifier: StreamingTimeSeriesClassifier = state["classifier"]
        

        if not classifier._ready():
            ready_i = -1
            for i in range(len(data)):
                col = data[i]
                state["history"].append({
                    classifier.meta.time_col: col["ts"],
                    classifier.meta.group_col: current_gid,
                    classifier.meta.seq_col: current_sid,
                    classifier.meta.target_cols[0]: col["bpm"],
                    classifier.meta.target_cols[1]: col["ut"]
                })

                if len(state["history"]) >= self.max_lag:
                    hist_df = pd.DataFrame(state["history"])
                    classifier.warm_start(hist_df)
                    ready_i = i
                    break
            if ready_i != -1:
                if ready_i < len(data) - 1:
                    data = data[ready_i + 1:]
                else:
                    data = []
            else:
                needed = max(0, self.max_lag - len(state["history"]))
                return {
                    "ready": False,
                    "needed": needed,
                    "classification": []
                }
            
        for col in data:
            classifier.update_one(col['ts'], col['bpm'], col['ut'], current_gid, current_sid)

        classification = classifier.classify_timeseries()
        classification_str = self._classification_to_string(classification)
        
        return {
            "ready": True,
            "classification": classification_str,
            "needed": 0,
        }

# service = ClassificationService(model_dir="artifacts")
# response = service.process_message("1234567890,120,0.5,group1,seq1")
# print(response)