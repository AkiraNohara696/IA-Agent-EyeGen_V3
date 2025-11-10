from __future__ import annotations
from collections import defaultdict

ema_area = defaultdict(lambda: None)
ema_center = defaultdict(lambda: None)


def smooth_area(key, new_area, alpha=0.4):
    prev = ema_area[key]
    ema_area[key] = new_area if prev is None else (alpha * new_area + (1 - alpha) * prev)
    return ema_area[key]


def smooth_center(key, new_c, alpha=0.5):
    prev = ema_center[key]
    ema_center[key] = new_c if prev is None else (alpha * new_c + (1 - alpha) * prev)
    return ema_center[key]


def estimate_distance_bins_from_rel_area(rel_area):
    if rel_area > 0.40:  return 0.5
    if rel_area > 0.25:  return 1.0
    if rel_area > 0.12:  return 2.0
    if rel_area > 0.06:  return 3.0
    return 5.0