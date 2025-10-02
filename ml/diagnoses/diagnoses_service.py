from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np



def _mask_to_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Преобразует булеву маску в список (start, end) с включающими индексами."""
    if not np.any(mask):
        return []
    extended = np.concatenate([[False], mask, [False]])
    diff = np.diff(extended.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return [(s, e) for s, e in zip(starts, ends) if s <= e and e < len(mask)]

@dataclass(frozen=True)
class AnomalyDetectionConfig:
    # Общие параметры
    sampling_dt_fallback: float = 1.0  # сек, если t не позволяет вычислить dt

    # Тахикардия
    tachycardia_threshold: float = 160.0      # уд/мин
    tachycardia_min_duration_sec: float = 600.0  # 10 минут
    tachycardia_smoothing_window_sec: float = 60.0  # окно сглаживания

    # Брадикардия
    bradycardia_threshold: float = 110.0      # уд/мин
    bradycardia_min_duration_sec: float = 300.0  # 5 минут
    bradycardia_smoothing_window_sec: float = 60.0

    # Децелерации
    deceleration_depth_threshold: float = 15.0   # снижение от базовой линии
    deceleration_min_duration_sec: float = 15.0
    deceleration_max_duration_sec: float = 120.0
    deceleration_baseline_window_sec: float = 600.0  # ~10 минут

    # Вариабельность
    variability_std_threshold: float = 5.0       # уд/мин
    variability_window_sec: float = 60.0
    variability_min_duration_sec: float = 300.0  # 5 минут

def detect_tachycardia(
    x: np.ndarray,
    t: np.ndarray,
    config: AnomalyDetectionConfig
) -> List[Tuple[int, int]]:
    if len(x) == 0:
        return []
    
    dt = np.median(np.diff(t)) if len(t) > 1 else config.sampling_dt_fallback
    win_size = max(1, int(config.tachycardia_smoothing_window_sec / dt))
    rolling_mean = np.convolve(x, np.ones(win_size) / win_size, mode='same')
    
    mask = rolling_mean > config.tachycardia_threshold
    segments = _mask_to_segments(mask)
    min_points = int(config.tachycardia_min_duration_sec / dt)
    return [(s, e) for s, e in segments if (e - s) >= min_points]


def detect_bradycardia(
    x: np.ndarray,
    t: np.ndarray,
    config: AnomalyDetectionConfig
) -> List[Tuple[int, int]]:
    if len(x) == 0:
        return []
    
    dt = np.median(np.diff(t)) if len(t) > 1 else config.sampling_dt_fallback
    win_size = max(1, int(config.bradycardia_smoothing_window_sec / dt))
    rolling_mean = np.convolve(x, np.ones(win_size) / win_size, mode='same')
    
    mask = rolling_mean < config.bradycardia_threshold
    segments = _mask_to_segments(mask)
    min_points = int(config.bradycardia_min_duration_sec / dt)
    return [(s, e) for s, e in segments if (e - s) >= min_points]


def detect_decelerations(
    x: np.ndarray,
    t: np.ndarray,
    config: AnomalyDetectionConfig
) -> List[Tuple[int, int]]:
    if len(x) < 10:
        return []
    
    dt = np.median(np.diff(t)) if len(t) > 1 else config.sampling_dt_fallback
    min_points = max(1, int(config.deceleration_min_duration_sec / dt))
    max_points = max(2, int(config.deceleration_max_duration_sec / dt))
    
    baseline_win = max(1, int(config.deceleration_baseline_window_sec / dt))
    baseline_win = min(baseline_win, len(x))
    baseline = np.convolve(x, np.ones(baseline_win) / baseline_win, mode='same')
    
    diff = x - baseline
    decel_mask = diff < -config.deceleration_depth_threshold
    segments = _mask_to_segments(decel_mask)
    
    return [
        (s, e) for s, e in segments
        if min_points <= (e - s) <= max_points
    ]


def detect_reduced_variability(
    x: np.ndarray,
    t: np.ndarray,
    config: AnomalyDetectionConfig
) -> List[Tuple[int, int]]:
    if len(x) < 10:
        return []
    
    dt = np.median(np.diff(t)) if len(t) > 1 else config.sampling_dt_fallback
    win_size = max(1, int(config.variability_window_sec / dt))
    min_points = max(1, int(config.variability_min_duration_sec / dt))
    
    # Скользящее STD (центральное окно)
    half_win = win_size // 2
    rolling_std = np.empty(len(x))
    for i in range(len(x)):
        start = max(0, i - half_win)
        end = min(len(x), i + half_win + 1)
        rolling_std[i] = np.std(x[start:end], ddof=0)
    
    mask = rolling_std < config.variability_std_threshold
    segments = _mask_to_segments(mask)
    return [(s, e) for s, e in segments if (e - s) >= min_points]