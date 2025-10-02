from __future__ import annotations
import numpy as np
from typing import List, Union


class MADAnomalyDetector:
    def __init__(self, threshold: float = 3.5):
        self.threshold = threshold

    @staticmethod
    def detect_anomalies(x: Union[List[float], np.ndarray], threshold: float = 3.5) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return np.array([], dtype=int)
        
        median = np.median(x)
        mad = np.median(np.abs(x - median))
        
        if mad == 0.0:
            return np.array([], dtype=int)
        
        robust_z = 0.6745 * (x - median) / mad
        return np.where(np.abs(robust_z) > threshold)[0]

    def __call__(self, x: Union[List[float], np.ndarray]) -> List[int]:
        anomaly_indices = self.detect_anomalies(x, threshold=self.threshold)
        return anomaly_indices.tolist()