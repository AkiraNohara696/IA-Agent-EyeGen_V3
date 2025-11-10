from __future__ import annotations

def center_offset(xyxy, frame_w):
    x1, y1, x2, y2 = xyxy
    box_center = (x1 + x2) / 2.0
    delta = (box_center - frame_w / 2.0) / (frame_w / 2.0)  # -1 esq, +1 dir
    return max(-1.0, min(1.0, float(delta)))


def choose_direction(centralidade, proximidade_m):
    if proximidade_m <= 0.5:
        return "pare"
    if centralidade < -0.2:
        return "desvie para a esquerda"
    if centralidade > 0.2:
        return "desvie para a direita"
    if proximidade_m >= 2.0:
        return "siga em frente"
    return "pare"


def bin_distance_label(d):
    mapping = {0.5: "0,5 m", 1.0: "1 m", 2.0: "2 m", 3.0: "3 m", 5.0: "5 m"}
    return mapping.get(d, "1 m")