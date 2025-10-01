from __future__ import annotations
import os, json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import deque

@dataclass
class _MetaCfg:
    time_col: str
    group_col: str
    seq_col: str
    target_cols: Tuple[str, str]
    seq_len: int
    threshold: float
    scaler_stats: Dict[str, Dict[str, float]]

class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.conv(x)

class UNet1D(nn.Module):
    def __init__(self, in_channels=2, base_ch=16, depth=3, dropout=0.1, out_channels=2):
        super().__init__()
        chs = [base_ch * (2**i) for i in range(depth)]
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for c in chs:
            self.enc_blocks.append(ConvBlock1D(prev, c, dropout=dropout))
            self.pools.append(nn.MaxPool1d(kernel_size=2))
            prev = c

        self.bottleneck = ConvBlock1D(prev, prev*2, dropout=dropout)

        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        dec_chs = list(reversed(chs))
        prev_dec = prev*2
        for c in dec_chs:
            self.upconvs.append(nn.ConvTranspose1d(prev_dec, c, kernel_size=2, stride=2))
            self.dec_blocks.append(ConvBlock1D(prev_dec, c, dropout=dropout))
            prev_dec = c

        self.out_conv = nn.Conv1d(prev_dec, out_channels, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        skips = []
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)
        x = self.bottleneck(x)
        for up, dec, skip in zip(self.upconvs, self.dec_blocks, reversed(skips)):
            x = up(x)
            if x.size(-1) != skip.size(-1):
                diff = skip.size(-1) - x.size(-1)
                if diff > 0:
                    x = nn.functional.pad(x, (0, diff))
                else:
                    x = x[..., :skip.size(-1)]
            x = torch.cat([skip, x], dim=1)
            x = dec(x)
        x = self.out_conv(x)
        x = x.permute(0, 2, 1)
        return x

class StreamingUNetAnomalyDetector:
    def __init__(self, model_dir: str):
        with open(os.path.join(model_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        cfg = meta["config"]
        self.meta = _MetaCfg(
            time_col=cfg["time_col"],
            group_col=cfg["group_col"], 
            seq_col=cfg["seq_col"],
            target_cols=tuple(cfg["target_cols"]),
            seq_len=cfg["seq_len"],
            threshold=meta["train_stats"]["threshold"],
            scaler_stats=meta["scaler_stats"]
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNet1D(
            in_channels=2,
            base_ch=cfg.get("base_ch", 16),
            depth=cfg.get("depth", 3),
            dropout=cfg.get("dropout", 0.1),
            out_channels=2
        )
        
        state_dict = torch.load(os.path.join(model_dir, "ae_unet1d_final_unet_all_data.pt"), map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        self.buffers: Dict[str, deque] = {
            self.meta.target_cols[0]: deque(maxlen=self.meta.seq_len),
            self.meta.target_cols[1]: deque(maxlen=self.meta.seq_len)
        }
        self.last_ts_sec: Optional[float] = None
        self.group_id = None
        self.sequence_id = None

    def _ready(self) -> bool:
        return (self.last_ts_sec is not None) and \
               all(len(self.buffers[t]) >= self.meta.seq_len for t in self.buffers)

    @staticmethod
    def _ts_to_sec(ts_val) -> float:
        return float(ts_val)

    def _preprocess_sequence(self, bpm_seq: np.ndarray, uterus_seq: np.ndarray) -> np.ndarray:
        bpm_stats = self.meta.scaler_stats['bpm']
        uterus_stats = self.meta.scaler_stats['uterus']
        
        bpm_normalized = (bpm_seq - bpm_stats['mean']) / bpm_stats['std']
        uterus_normalized = (uterus_seq - uterus_stats['mean']) / uterus_stats['std']
        
        x = np.stack([bpm_normalized, uterus_normalized], axis=1)
        
        if len(x) > self.meta.seq_len:
            x = x[-self.meta.seq_len:]
        elif len(x) < self.meta.seq_len:
            pad_len = self.meta.seq_len - len(x)
            x = np.concatenate([np.zeros((pad_len, 2), dtype=np.float32), x], axis=0)
        
        return x

    def warm_start(self, history_df: pd.DataFrame):
        """Initialize detector with historical data"""
        tcol = self.meta.time_col
        df = history_df.sort_values(tcol).reset_index(drop=True).copy()
        
        if not pd.api.types.is_numeric_dtype(df[tcol]):
            df[tcol] = pd.to_datetime(df[tcol]).astype("int64") / 1e9
        
        self.last_ts_sec = float(df[tcol].iloc[-1]) if len(df) else None
        self.group_id = df[self.meta.group_col].iloc[-1] if self.meta.group_col in df else None
        self.sequence_id = df[self.meta.seq_col].iloc[-1] if self.meta.seq_col in df else None
        
        t1, t2 = self.meta.target_cols
        h1 = df[t1].astype(float).tolist()
        h2 = df[t2].astype(float).tolist()
        
        if len(h1) < self.meta.seq_len or len(h2) < self.meta.seq_len:
            raise ValueError(f"Need at least {self.meta.seq_len} historical points per target for warm start")
        
        for val in h1[-self.meta.seq_len:]:
            self.buffers[t1].append(float(val))
        for val in h2[-self.meta.seq_len:]:
            self.buffers[t2].append(float(val))

    def update_one(self, ts, bpm, uterus, group_id=None, sequence_id=None):
        ts_sec = self._ts_to_sec(ts)
        self.last_ts_sec = float(ts_sec)
        
        t1, t2 = self.meta.target_cols
        self.buffers[t1].append(float(bpm))
        self.buffers[t2].append(float(uterus))
        
        if group_id is not None: 
            self.group_id = group_id
        if sequence_id is not None: 
            self.sequence_id = sequence_id

    def _get_pointwise_reconstruction_error(self, x_input: np.ndarray) -> float:
        with torch.no_grad():
            x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(self.device)
            reconstruction = self.model(x_tensor)
            
            input_last = x_tensor[0, -1, :]  # Shape: (2,)
            recon_last = reconstruction[0, -1, :]  # Shape: (2,)
            
            mae = torch.mean(torch.abs(input_last - recon_last)).item()
            return mae

    def detect_anomaly(self) -> Dict[str, any]:
        if not self._ready():
            raise RuntimeError("Detector not ready; warm_start or accumulate more points first")
        
        t1, t2 = self.meta.target_cols
        bpm_seq = np.array(list(self.buffers[t1]))
        uterus_seq = np.array(list(self.buffers[t2]))
        
        x_input = self._preprocess_sequence(bpm_seq, uterus_seq)
        recon_error = self._get_pointwise_reconstruction_error(x_input)
        anomaly_flag = 1 if recon_error > self.meta.threshold else 0
        
        return {
            self.meta.time_col: self.last_ts_sec,
            self.meta.group_col: self.group_id,
            self.meta.seq_col: self.sequence_id,
            "bpm": bpm_seq[-1],
            "uterus": uterus_seq[-1],
            "bpm_anomaly": anomaly_flag,
            "uterus_anomaly": anomaly_flag,
            "anomaly_score": float(recon_error)
        }

class AnomalyService:
    def __init__(self, model_dir: str):
        self.template = StreamingUNetAnomalyDetector(model_dir)
        self.min_required = self.template.meta.seq_len
        self.model_dir = model_dir
        self._states: Dict[Tuple[str, str], Dict] = {}

    def _get_or_create(self, gid: str, sid: str) -> Dict:
        key = (gid, sid)
        st = self._states.get(key)
        if st is None:
            st = {"det": StreamingUNetAnomalyDetector(self.model_dir), "hist": []}
            self._states[key] = st
        return st

    def _detection_to_string(self, detection: Dict) -> str:
        # timestamp,group_id,sequence_id,bpm,uterus,bpm_anomaly,uterus_anomaly,anomaly_score
        return f"{detection[self.template.meta.time_col]},{detection[self.template.meta.group_col]},{detection[self.template.meta.seq_col]},{detection['bpm']},{detection['uterus']},{detection['bpm_anomaly']},{detection['uterus_anomaly']},{detection['anomaly_score']}"

    def process_message(self, msg: str) -> Dict[str, any]:
        # timestamp,bpm,uterus,group_id,sequence_id
        parts = msg.strip().split(',')

        current_sid = "0"
        current_gid = "0"
        try:
            ts = StreamingUNetAnomalyDetector._ts_to_sec(parts[0])
            bpm = float(parts[1])
            ut = float(parts[2])
            gid = current_gid = "0"
            sid = current_sid
        except (ValueError, IndexError) as e:
            return {
                "ready": False,
                "detection": "",
                "needed": 0,
                "error": f"Failed to parse values - {str(e)}"
            }
        
        st = self._get_or_create(gid, sid)
        det: StreamingUNetAnomalyDetector = st["det"]

        if not det._ready():
            st["hist"].append({
                det.meta.time_col: ts, 
                det.meta.group_col: gid, 
                det.meta.seq_col: sid,
                det.meta.target_cols[0]: bpm, 
                det.meta.target_cols[1]: ut
            })
            
            if len(st["hist"]) >= self.min_required:
                det.warm_start(pd.DataFrame(st["hist"]))
            
            needed = max(0, self.min_required - len(st["hist"]))
            return {
                "ready": False,
                "detection": "",
                "needed": needed
            }

        det.update_one(ts, bpm, ut, gid, sid)
        detection = det.detect_anomaly()
        detection_str = self._detection_to_string(detection)
        
        return {
            "ready": True,
            "detection": detection_str,
            "needed": 0
        }