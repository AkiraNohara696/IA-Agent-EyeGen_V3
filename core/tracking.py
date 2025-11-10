from __future__ import annotations
import time, math
from collections import defaultdict, deque
from typing import Deque, Dict, Any, Tuple
from core import config

# memória por rótulo -> lista de tracks
obj_memory: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"tracks": []})


def now_ts() -> float:
    return time.time()


def prune_old_tracks():
    t = now_ts()
    for label in list(obj_memory.keys()):
        tracks = obj_memory[label]["tracks"]
        obj_memory[label]["tracks"] = [tr for tr in tracks if (t - tr["ts"]) <= config.MEM_TRACK_TTL]
        if not obj_memory[label]["tracks"]:
            del obj_memory[label]


def match_track(label: str, cx: float, cy: float):
    best, best_d = None, 1e9
    for tr in obj_memory[label]["tracks"]:
        d = math.hypot(tr["cx"] - cx, tr["cy"] - cy)
        if d < best_d and d <= config.CENTER_MATCH_PX:
            best, best_d = tr, d
    return best


def upsert_track(label: str, cx: float, cy: float, dist_bin: float, rel_area: float):
    prune_old_tracks()
    t = now_ts()
    tr = match_track(label, cx, cy)
    if tr is None:
        tr = {"cx": cx, "cy": cy, "dist_bin": dist_bin, "ts": t, "rel_area_hist": deque(maxlen=6)}
        tr["rel_area_hist"].append(rel_area)
        obj_memory[label]["tracks"].append(tr)
        if len(obj_memory[label]["tracks"]) > config.MAX_TRACKS_PER_LABEL:
            obj_memory[label]["tracks"].pop(0)
        return True, False, tr
    dist_changed = (tr["dist_bin"] != dist_bin)
    tr["cx"], tr["cy"], tr["dist_bin"], tr["ts"] = cx, cy, dist_bin, t
    tr["rel_area_hist"].append(rel_area)
    return False, dist_changed, tr


def is_approaching(rel_hist: Deque[float], min_len: int = 3, eps: float = 1e-4) -> bool:
    if len(rel_hist) < min_len:
        return False
    inc = 0
    for i in range(1, len(rel_hist)):
        if (rel_hist[i] - rel_hist[i-1]) > eps:
            inc += 1
    return inc >= (min_len - 1)