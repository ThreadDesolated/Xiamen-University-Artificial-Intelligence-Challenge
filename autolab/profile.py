"""数据画像：在规划之前先"看懂"数据集。

画像结果会写进 trace，也是配方库检索的索引键。
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd

from .schemas import ValidationError


def infer_feature_types(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    """划分数值特征列与类别特征列。"""
    numeric, categorical = [], []
    for col in df.columns:
        if col == target:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            # 取值很少的整数列（如编码后的类别）仍按数值处理，由算子自行决定
            numeric.append(str(col))
        else:
            categorical.append(str(col))
    return numeric, categorical


def profile_dataframe(
    df: pd.DataFrame,
    target: str,
    task_type: str = "binary_classification",
) -> dict:
    """生成数据画像 dict（字段即接口契约）。"""
    if target not in df.columns:
        raise ValidationError(f"数据里没有目标列 {target!r}")

    feature_cols = [c for c in df.columns if c != target]
    n_rows, n_cols = df.shape
    n_cells = max(1, n_rows * len(feature_cols))
    missing_ratio = float(df[feature_cols].isna().sum().sum()) / n_cells
    columns_with_missing = {
        str(c): round(float(df[c].isna().mean()), 6) for c in feature_cols if df[c].isna().any()
    }
    numeric, categorical = infer_feature_types(df, target)

    target_info: dict[str, Any] = {"name": target, "type": task_type}
    imbalance = None
    if task_type.startswith("binary") or task_type.startswith("multiclass"):
        counts = df[target].value_counts(dropna=True)
        if len(counts) >= 2:
            imbalance = float(counts.max()) / max(1.0, float(counts.min()))
        target_info.update(
            {
                "n_classes": int(len(counts)),
                "class_counts": {str(k): int(v) for k, v in counts.items()},
                "imbalance_ratio": round(imbalance, 4) if imbalance else None,
                "has_nan": bool(df[target].isna().any()),
            }
        )
    else:
        series = pd.to_numeric(df[target], errors="coerce")
        target_info.update(
            {
                "mean": round(float(series.mean()), 6),
                "std": round(float(series.std()), 6),
                "min": round(float(series.min()), 6),
                "max": round(float(series.max()), 6),
                "has_nan": bool(series.isna().any()),
            }
        )

    constant_cols = [
        str(c) for c in feature_cols if df[c].nunique(dropna=True) <= 1
    ]
    high_card = [
        str(c) for c in categorical if df[c].nunique(dropna=True) > 50
    ]

    warnings: list[str] = []
    if missing_ratio > 0.02:
        warnings.append("high_missing_ratio")
    if imbalance is not None and imbalance > 2.0:
        warnings.append("class_imbalance")
    if constant_cols:
        warnings.append("constant_columns")
    if high_card:
        warnings.append("high_cardinality")
    if n_rows < 200:
        warnings.append("small_dataset")

    numeric_summary = {
        str(c): {
            "mean": round(float(pd.to_numeric(df[c], errors="coerce").mean()), 6),
            "std": round(float(pd.to_numeric(df[c], errors="coerce").std()), 6),
        }
        for c in numeric[:30]
    }

    return {
        "schema_version": "0.1.0",
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "n_features": len(feature_cols),
        "feature_types": {"numeric": numeric, "categorical": categorical},
        "missing_ratio": round(missing_ratio, 6),
        "max_col_missing_ratio": round(max(columns_with_missing.values()), 6)
        if columns_with_missing
        else 0.0,
        "columns_with_missing": columns_with_missing,
        "constant_columns": constant_cols,
        "high_cardinality_columns": high_card,
        "numeric_summary": numeric_summary,
        "target": target_info,
        "warnings": warnings,
    }


def profile_hash(profile: dict) -> str:
    """粗粒度画像指纹，用作配方库检索键（避免过拟合到具体数据行数）。"""

    def bucket(value: float, edges: tuple[float, ...]) -> int:
        for i, edge in enumerate(edges):
            if value <= edge:
                return i
        return len(edges)

    target = profile.get("target", {})
    payload = {
        "task_type": target.get("type"),
        "rows": bucket(float(profile.get("n_rows", 0)), (200, 500, 2000, 10000)),
        "missing": bucket(float(profile.get("missing_ratio", 0.0)), (0.001, 0.02, 0.1, 0.3)),
        "imbalance": bucket(float(target.get("imbalance_ratio") or 1.0), (1.2, 2.0, 5.0)),
        "n_numeric": len(profile.get("feature_types", {}).get("numeric", [])),
        "n_categorical": len(profile.get("feature_types", {}).get("categorical", [])),
        "warnings": sorted(profile.get("warnings", [])),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
