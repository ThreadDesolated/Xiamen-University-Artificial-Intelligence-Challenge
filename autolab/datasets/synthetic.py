"""内置合成数据集（零依赖、离线可跑）。

`make_t1` 生成带缺失值、类别失衡与量纲差异的合成表格数据；
用于 W1 验收与断网演示：第一轮弱方案不达标 → 归因 → 第二轮修复后达标。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 各特征的中心与尺度：x0–x3 承载真实信号，但量纲差异大；x4–x11 为噪声列
_LOC = np.array([2.0, -1.0, 0.5, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
_SCALE = np.array([0.15, 0.8, 0.04, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)


def make_t1(
    n_rows: int = 1000,
    n_features: int = 8,
    positive_ratio: float = 0.25,
    missing_ratio: float = 0.05,
    noise: float = 0.6,
    seed: int = 42,
) -> pd.DataFrame:
    """生成表格二分类数据：x0..x{n-1} 为特征，label 为 0/1 标签。"""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n_rows, n_features))
    loc = _LOC[:n_features]
    scale = _SCALE[:n_features]
    X = latent * scale + loc
    # 真实信号集中在 x0/x1/x2（正相关）与 x3（负相关），按潜变量标准化后叠加
    score = 1.15 * latent[:, 0] + 0.75 * latent[:, 1] + 0.55 * latent[:, 2] - 0.45 * latent[:, 3]
    score += rng.normal(scale=noise, size=n_rows)
    threshold = np.quantile(score, 1.0 - positive_ratio)
    y = (score > threshold).astype(int)

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])
    df["label"] = y

    for col in ("x0", "x2", "x4"):
        mask = rng.random(n_rows) < missing_ratio
        df.loc[mask, col] = np.nan
    return df


def make_t1_shift(
    n_rows: int = 1400,
    n_features: int = 12,
    positive_ratio: float = 0.2,
    missing_ratio: float = 0.04,
    noise: float = 0.55,
    seed: int = 2026,
) -> pd.DataFrame:
    """与 `make_t1` 同任务类型、画像相近但指纹不同的数据集。

    用于演示**跨任务配方迁移**：先跑 t1_demo 沉淀配方，再跑 t1_shift，
    画像相似度达到门槛时直接复用历史方案，首轮达标。
    """
    return make_t1(
        n_rows=n_rows,
        n_features=n_features,
        positive_ratio=positive_ratio,
        missing_ratio=missing_ratio,
        noise=noise,
        seed=seed,
    )
