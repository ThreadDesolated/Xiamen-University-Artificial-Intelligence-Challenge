"""固定种子 K 折交叉验证协议（可复现评测的唯一入口）。

约定：
- 同一份数据 + 同一个 seed，fold 划分完全可复现；
- 分类用分层 K 折，回归用普通 K 折；
- 所有统计量（均值/标准化统计量/编码字典）只在训练折上拟合，避免泄漏；
- 同时记录训练折指标，用于判断过拟合/欠拟合。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..schemas import Metrics, ValidationError
from ..sandbox.operators import Pipeline, build_model
from .metrics import compute


def feature_columns(df: pd.DataFrame, target: str, features=None) -> list[str]:
    if features:
        cols = [c for c in features if c in df.columns]
        if not cols:
            raise ValidationError("任务配置里的 features 都不在数据集中")
        return cols
    return [str(c) for c in df.columns if c != target]


def stratified_folds(y, n_splits: int, seed: int) -> list[np.ndarray]:
    y_arr = np.asarray(y)
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for cls in np.unique(y_arr):
        idx = np.where(y_arr == cls)[0]
        rng.shuffle(idx)
        for k, i in enumerate(idx):
            folds[k % n_splits].append(int(i))
    return [np.array(sorted(f), dtype=int) for f in folds]


def kfold_indices(n: int, n_splits: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [np.array(part, dtype=int) for part in np.array_split(idx, n_splits)]


def _prepare(df: pd.DataFrame, target: str, features=None):
    if target not in df.columns:
        raise ValidationError(f"数据里没有目标列 {target!r}")
    df = df.dropna(subset=[target]).reset_index(drop=True)
    cols = feature_columns(df, target, features)
    return df[cols], df[target].to_numpy()


def run_cv(
    plan,
    df: pd.DataFrame,
    task_type: str,
    target: str,
    metric: str,
    n_splits: int = 5,
    seed: int = 42,
    features=None,
) -> dict:
    """执行一次完整交叉验证，返回可直接 JSON 化的结果 dict。"""
    started = time.perf_counter()
    X, y_raw = _prepare(df, target, features)
    is_classification = task_type in ("binary_classification", "multiclass_classification")

    if is_classification:
        y = pd.factorize(pd.Series(y_raw), sort=True)[0].astype(np.int64)
        folds = stratified_folds(y, n_splits, seed)
    else:
        y = np.asarray(y_raw, dtype=float)
        folds = kfold_indices(len(y), n_splits, seed)

    op_names = [str(o.get("op", "")) for o in plan.ops]
    drop_rows = "drop_missing_rows" in op_names

    metric_values: list[float] = []
    train_values: list[float] = []
    fold_sizes: list[dict] = []
    for k, test_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != k])
        X_tr, y_tr = X.iloc[train_idx].copy(), y[train_idx]
        X_te, y_te = X.iloc[test_idx].copy(), y[test_idx]

        if drop_rows:
            mask = X_tr.notna().all(axis=1)
            X_tr, y_tr = X_tr[mask].copy(), y_tr[np.asarray(mask)]

        if len(X_tr) < 10:
            raise ValidationError(f"第 {k + 1} 折训练样本不足（{len(X_tr)} 行）")

        pipe = Pipeline(plan.ops).fit(X_tr)
        X_tr_t = pipe.transform(X_tr)
        X_te_t = pipe.transform(X_te)

        model = build_model(plan.model).fit(X_tr_t, y_tr)
        pred = model.predict(X_te_t)
        metric_values.append(compute(metric, y_te, pred))
        train_values.append(compute(metric, y_tr, model.predict(X_tr_t)))
        fold_sizes.append({"fold": k + 1, "train": int(len(X_tr)), "test": int(len(X_te))})

    values = np.asarray(metric_values, dtype=float)
    result = {
        "metrics": {
            metric: Metrics(
                name=metric,
                mean=float(values.mean()),
                std=float(values.std(ddof=0)),
                values=[round(float(v), 6) for v in values],
                n=len(values),
            ).to_dict()
        },
        "train_metric": float(np.mean(train_values)),
        "seconds": round(time.perf_counter() - started, 3),
        "n_folds": len(folds),
        "n_samples": int(len(y)),
        "fold_sizes": fold_sizes,
    }
    return result
