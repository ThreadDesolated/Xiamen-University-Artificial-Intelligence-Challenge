"""评测指标实现（numpy，无 sklearn 依赖）。"""
from __future__ import annotations

import numpy as np

HIGHER_IS_BETTER = {
    "accuracy": True,
    "f1_macro": True,
    "mae": False,
    "rmse": False,
    "r2": True,
}


def accuracy(y_true, y_pred) -> float:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    if yt.size == 0:
        return 0.0
    return float(np.mean(yt == yp))


def f1_macro(y_true, y_pred) -> float:
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for cls in np.unique(yt):
        tp = float(np.sum((yp == cls) & (yt == cls)))
        fp = float(np.sum((yp == cls) & (yt != cls)))
        fn = float(np.sum((yp != cls) & (yt == cls)))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def mae(y_true, y_pred) -> float:
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2(y_true, y_pred) -> float:
    yt, yp = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


METRIC_FUNCS = {
    "accuracy": accuracy,
    "f1_macro": f1_macro,
    "mae": mae,
    "rmse": rmse,
    "r2": r2,
}


def compute(metric: str, y_true, y_pred) -> float:
    if metric not in METRIC_FUNCS:
        raise ValueError(f"不支持的指标: {metric}，可用: {list(METRIC_FUNCS)}")
    return METRIC_FUNCS[metric](y_true, y_pred)


def higher_is_better(metric: str) -> bool:
    return HIGHER_IS_BETTER.get(metric, True)
