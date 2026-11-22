"""固定种子 K 折交叉验证协议（可复现评测的唯一入口）。

约定：
- 同一份数据 + 同一个 seed，fold 划分完全可复现；
- 分类用分层 K 折，回归用普通 K 折；
- 所有统计量（均值/标准化统计量/编码字典）只在训练折上拟合，避免泄漏；
- 同时记录训练折指标，用于判断过拟合/欠拟合；
- 可选留出集：holdout_ratio > 0 时先切出独立数据（不参与折划分与选模），
  只在交叉验证结束后用最终方案评估一次，报告"对外泛化"成绩（防验证集过拟合）。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..schemas import Metrics, ValidationError
from ..sandbox.operators import Pipeline, build_model
from .metrics import compute
from .model_store import save_model


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


def holdout_indices(n: int, ratio: float, seed: int) -> np.ndarray:
    """回归/无标签场景：按比例随机切留出集（固定种子，可复现）。"""
    n_hold = max(1, int(round(n * ratio)))
    n_hold = min(n_hold, n - 1)
    rng = np.random.default_rng(seed + 10007)  # 与 K 折划分的随机流错开
    return np.sort(rng.permutation(n)[:n_hold])


def stratified_holdout_indices(y, ratio: float, seed: int) -> np.ndarray:
    """分类场景：按类别比例分层切留出集（最大余数法分配名额）。"""
    y_arr = np.asarray(y)
    n = len(y_arr)
    n_hold = min(max(1, int(round(n * ratio))), n - 1)
    classes, counts = np.unique(y_arr, return_counts=True)
    raw = counts * (n_hold / float(n))
    alloc = np.floor(raw).astype(int)
    remainder = n_hold - int(alloc.sum())
    order = np.argsort(-(raw - alloc))
    for i in order[: max(0, remainder)]:
        alloc[i] += 1

    rng = np.random.default_rng(seed + 10007)
    picked: list[np.ndarray] = []
    for k, cls in enumerate(classes):
        idx = np.where(y_arr == cls)[0]
        rng.shuffle(idx)
        take = int(alloc[k])
        if take == 0 and len(idx) >= 2:
            take = 1  # 小类别也尽量留一点进留出集
        take = min(take, len(idx) - 1)  # 每类至少保留 1 行给训练
        if take > 0:
            picked.append(idx[:take])
    if not picked:
        return np.array([], dtype=int)
    return np.sort(np.concatenate(picked))


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
    holdout_ratio: float = 0.0,
    model_path=None,
    task_id: str | None = None,
) -> dict:
    """执行一次完整交叉验证，返回可直接 JSON 化的结果 dict。

    holdout_ratio > 0 时：先按比例切出留出集（分类分层），K 折只在剩余训练池上做；
    交叉验证结束后，用最终方案在训练池上重新拟合，并在留出集上评估一次。

    model_path 不为空时：在训练池上拟合「预处理 Pipeline + 模型」，导出为单文件
    （reports/<run_id>/model.joblib），并在结果 artifacts 里记录模型元信息。
    """
    started = time.perf_counter()
    X, y_raw = _prepare(df, target, features)
    is_classification = task_type in ("binary_classification", "multiclass_classification")
    ratio = float(holdout_ratio or 0.0)
    if not (0.0 <= ratio < 0.5):
        raise ValidationError("holdout_ratio 需在 [0, 0.5) 之间（0 表示不启用留出集）")

    if is_classification:
        y_codes, uniques = pd.factorize(pd.Series(y_raw), sort=True)
        y = y_codes.astype(np.int64)
        classes = uniques.tolist() if hasattr(uniques, "tolist") else list(uniques)
        holdout_idx = stratified_holdout_indices(y, ratio, seed) if ratio > 0 else np.array([], dtype=int)
    else:
        y = np.asarray(y_raw, dtype=float)
        classes = None
        holdout_idx = holdout_indices(len(y), ratio, seed) if ratio > 0 else np.array([], dtype=int)

    if holdout_idx.size:
        train_idx = np.setdiff1d(np.arange(len(y)), holdout_idx)
        if len(train_idx) < n_splits * 10:
            raise ValidationError(
                f"留出集 {holdout_idx.size} 行后训练样本不足（{len(train_idx)} 行），请调小 holdout_ratio"
            )
        X_cv = X.iloc[train_idx].reset_index(drop=True)
        y_cv = y[train_idx]
        X_holdout = X.iloc[holdout_idx].reset_index(drop=True)
        y_holdout = y[holdout_idx]
    else:
        X_cv, y_cv = X, y
        X_holdout = y_holdout = None

    if is_classification:
        folds = stratified_folds(y_cv, n_splits, seed)
    else:
        folds = kfold_indices(len(y_cv), n_splits, seed)

    op_names = [str(o.get("op", "")) for o in plan.ops]
    drop_rows = "drop_missing_rows" in op_names

    def _fit_final(X_train, y_train):
        """在给定训练池上拟合"预处理 + 模型"（与每一折内部流程一致）。"""
        if drop_rows:
            mask = X_train.notna().all(axis=1)
            X_train, y_train = X_train[mask].copy(), y_train[np.asarray(mask)]
        if len(X_train) < 10:
            raise ValidationError(f"训练样本不足（{len(X_train)} 行）")
        pipe = Pipeline(plan.ops).fit(X_train)
        model = build_model(plan.model).fit(pipe.transform(X_train), y_train)
        return pipe, model, X_train, y_train

    metric_values: list[float] = []
    train_values: list[float] = []
    fold_sizes: list[dict] = []
    for k, test_idx in enumerate(folds):
        train_idx = np.concatenate([f for j, f in enumerate(folds) if j != k])
        X_tr, y_tr = X_cv.iloc[train_idx].copy(), y_cv[train_idx]
        X_te, y_te = X_cv.iloc[test_idx].copy(), y_cv[test_idx]

        pipe, model, X_tr_fit, y_tr_fit = _fit_final(X_tr, y_tr)
        pred = model.predict(pipe.transform(X_te))
        metric_values.append(compute(metric, y_te, pred))
        train_values.append(compute(metric, y_tr_fit, model.predict(pipe.transform(X_tr_fit))))
        fold_sizes.append({"fold": k + 1, "train": int(len(X_tr_fit)), "test": int(len(X_te))})

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
        "n_cv_samples": int(len(y_cv)),
        "fold_sizes": fold_sizes,
        "holdout": None,
    }

    final_pipe = None
    final_model = None
    if X_holdout is not None and y_holdout is not None:
        final_pipe, final_model, _, _ = _fit_final(X_cv, y_cv)
        pred = final_model.predict(final_pipe.transform(X_holdout))
        result["holdout"] = {
            "metric": metric,
            "value": round(float(compute(metric, y_holdout, pred)), 6),
            "n_samples": int(len(y_holdout)),
            "n_train_samples": int(len(X_cv)),
            "ratio": round(float(len(y_holdout) / len(y)), 4),
            "split_seed": int(seed),
        }
        result["seconds"] = round(time.perf_counter() - started, 3)

    if model_path:
        if final_pipe is None:
            final_pipe, final_model, _, _ = _fit_final(X_cv, y_cv)
        result["artifacts"] = {
            "model": save_model(
                model_path,
                final_pipe,
                final_model,
                plan=plan,
                task_id=task_id,
                task_type=task_type,
                target=target,
                metric=metric,
                classes=classes,
                cv_mean=float(values.mean()),
            )
        }
        result["seconds"] = round(time.perf_counter() - started, 3)
    return result
