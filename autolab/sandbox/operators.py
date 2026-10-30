"""白名单算子与 numpy 基线模型（最小运行依赖：numpy + pandas）。

设计原则：
- 算子全部"只在训练折上拟合"，再 transform 测试折，杜绝泄漏；
- 执行后若仍有非数值列 / 缺失值，直接抛 ValidationError —— 这本身就是可归因的失败信号；
- W2 起在 whitelist_ops.json 里追加 sklearn/flaml 模型即可，执行框架不变。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..schemas import ValidationError

# 由评测协议在切折后处理（需要同时裁剪 X 与 y），Pipeline 中跳过
SKIP_OPS = {"drop_missing_rows"}


def _numeric_cols(frame: pd.DataFrame) -> list[str]:
    return [str(c) for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]


def _non_numeric_cols(frame: pd.DataFrame) -> list[str]:
    return [str(c) for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]


class Pipeline:
    """把方案里的算子列表变成可 fit/transform 的预处理管线。"""

    def __init__(self, ops: list | None):
        self.ops = list(ops or [])
        self.state: dict = {}
        self._fitted_cols: list[str] = []

    # ------------------------------------------------------------------ #
    def fit(self, X) -> "Pipeline":
        frame = pd.DataFrame(X).copy()
        for idx, spec in enumerate(self.ops):
            op = str(spec.get("op", ""))
            if op in SKIP_OPS:
                continue
            frame = self._apply(idx, op, spec.get("params") or {}, frame, fit=True)
        self._fitted_cols = [str(c) for c in frame.columns]
        return self

    def transform(self, X) -> np.ndarray:
        frame = pd.DataFrame(X).copy()
        for idx, spec in enumerate(self.ops):
            op = str(spec.get("op", ""))
            if op in SKIP_OPS:
                continue
            frame = self._apply(idx, op, spec.get("params") or {}, frame, fit=False)
        frame = frame.reindex(columns=self._fitted_cols, fill_value=0.0)
        if frame.isna().any().any():
            bad = [str(c) for c in frame.columns if frame[c].isna().any()][:5]
            raise ValidationError(
                f"算子执行后仍有缺失值（列: {bad}），方案需要补 impute_* 算子"
            )
        return frame.to_numpy(dtype=np.float64)

    # ------------------------------------------------------------------ #
    def _apply(self, idx: int, op: str, params: dict, frame: pd.DataFrame, fit: bool) -> pd.DataFrame:
        key = f"{idx}:{op}"

        if op == "impute_mean":
            cols = _numeric_cols(frame)
            if fit:
                self.state[key] = frame[cols].mean()
            frame[cols] = frame[cols].fillna(self.state[key])
            return frame

        if op == "impute_median":
            cols = _numeric_cols(frame)
            if fit:
                self.state[key] = frame[cols].median()
            frame[cols] = frame[cols].fillna(self.state[key])
            return frame

        if op == "impute_zero":
            frame[_numeric_cols(frame)] = frame[_numeric_cols(frame)].fillna(0.0)
            return frame

        if op == "scale_standard":
            cols = _numeric_cols(frame)
            if fit:
                mean = frame[cols].mean()
                std = frame[cols].std(ddof=0).replace(0.0, 1.0)
                self.state[key] = (mean, std)
            mean, std = self.state[key]
            frame[cols] = (frame[cols] - mean) / std
            return frame

        if op == "onehot":
            max_categories = int(params.get("max_categories", 20))
            if fit:
                mapping: dict[str, list[str]] = {}
                for col in _non_numeric_cols(frame):
                    values = frame[col].astype("object").where(frame[col].notna(), "__nan__").astype(str)
                    mapping[col] = list(values.value_counts().index[:max_categories])
                self.state[key] = mapping
            for col, keep in self.state.get(key, {}).items():
                if col not in frame.columns:
                    continue
                values = frame[col].astype("object").where(frame[col].notna(), "__nan__").astype(str)
                for cat in keep:
                    frame[f"{col}={cat}"] = (values == cat).astype(float)
                frame = frame.drop(columns=[col])
            return frame

        if op == "variance_filter":
            cols = _numeric_cols(frame)
            if fit:
                variances = frame[cols].var(ddof=0)
                threshold = float(params.get("threshold", 0.0))
                self.state[key] = [c for c in cols if float(variances[c]) > threshold]
            frame = frame[self.state[key]]
            return frame

        if op == "log1p":
            for col in _numeric_cols(frame):
                frame[col] = np.log1p(frame[col].clip(lower=0.0))
            return frame

        raise ValidationError(f"未知算子: {op!r}")


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class NearestCentroid:
    """最近质心分类器（刻意设计的弱 baseline）。"""

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self.classes_: np.ndarray = np.array([])
        self.centroids_: np.ndarray = np.empty((0, 0))

    def fit(self, X, y) -> "NearestCentroid":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        dist = ((X[:, None, :] - self.centroids_[None, :, :]) ** 2).sum(axis=2)
        return self.classes_[np.argmin(dist, axis=1)]


class LogisticGD:
    """numpy 实现的 softmax 逻辑回归（梯度下降 + L2 + balanced 类权重）。"""

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.lr = float(p.get("lr", 0.1))
        self.epochs = int(p.get("epochs", 300))
        self.l2 = float(p.get("l2", 0.001))
        self.class_weight = str(p.get("class_weight", "none"))
        self.classes_: np.ndarray = np.array([])
        self.W_: np.ndarray = np.empty((0, 0))

    def fit(self, X, y) -> "LogisticGD":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n = len(X)
        self.classes_ = np.unique(y)
        Xb = np.hstack([X, np.ones((n, 1))])
        self.W_ = np.zeros((Xb.shape[1], len(self.classes_)))

        sample_weight = np.ones(n)
        if self.class_weight == "balanced":
            for cls in self.classes_:
                mask = y == cls
                sample_weight[mask] = n / (len(self.classes_) * max(1, int(mask.sum())))

        onehot = (y[:, None] == self.classes_[None, :]).astype(np.float64)
        for _ in range(self.epochs):
            probs = _softmax(Xb @ self.W_)
            grad = Xb.T @ ((probs - onehot) * sample_weight[:, None]) / n
            grad[:-1] += self.l2 * self.W_[:-1]
            self.W_ -= self.lr * grad
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Xb = np.hstack([X, np.ones((len(X), 1))])
        return self.classes_[np.argmax(Xb @ self.W_, axis=1)]


class LinearGD:
    """numpy 实现的线性回归（梯度下降 + L2）。"""

    def __init__(self, params: dict | None = None):
        p = params or {}
        self.lr = float(p.get("lr", 0.05))
        self.epochs = int(p.get("epochs", 300))
        self.l2 = float(p.get("l2", 0.001))
        self.w_: np.ndarray = np.empty(0)

    def fit(self, X, y) -> "LinearGD":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = len(X)
        Xb = np.hstack([X, np.ones((n, 1))])
        self.w_ = np.zeros(Xb.shape[1])
        for _ in range(self.epochs):
            grad = Xb.T @ (Xb @ self.w_ - y) / n
            grad[:-1] += self.l2 * self.w_[:-1]
            self.w_ -= self.lr * grad
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        Xb = np.hstack([X, np.ones((len(X), 1))])
        return Xb @ self.w_


MODEL_REGISTRY = {
    "nearest_centroid": NearestCentroid,
    "logistic_gd": LogisticGD,
    "linear_gd": LinearGD,
}


def build_model(spec: dict):
    """按方案里的 model 字段实例化模型。"""
    kind = str((spec or {}).get("kind", ""))
    factory = MODEL_REGISTRY.get(kind)
    if factory is None:
        raise ValidationError(f"未知模型: {kind!r}，可用: {list(MODEL_REGISTRY)}")
    return factory((spec or {}).get("params") or {})
