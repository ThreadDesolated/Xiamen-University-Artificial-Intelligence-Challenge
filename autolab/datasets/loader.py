"""数据集加载：支持 `synthetic:<name>`、CSV、NPZ（相对仓库根目录解析）。"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from .. import config
from ..schemas import ValidationError
from .synthetic import make_t1

SYNTHETIC_FACTORIES = {
    "t1": make_t1,
}


def _resolve(path: str) -> pathlib.Path:
    p = pathlib.Path(path)
    return p if p.is_absolute() else config.ROOT / p


def load_dataframe(data_path: str, target: str) -> pd.DataFrame:
    """加载数据集为 DataFrame；target 必须存在。"""
    if data_path.startswith("synthetic:"):
        name = data_path.split(":", 1)[1]
        factory = SYNTHETIC_FACTORIES.get(name)
        if factory is None:
            raise ValidationError(f"未知合成数据集: {data_path}（可用: {list(SYNTHETIC_FACTORIES)}）")
        df = factory()
    else:
        p = _resolve(data_path)
        if not p.exists():
            raise FileNotFoundError(
                f"数据集不存在: {p}\n先运行 `python -m autolab prepare --task <task_id>` 准备真实数据集，"
                f"或改用 synthetic:t1 离线演示。"
            )
        if p.suffix == ".csv":
            df = pd.read_csv(p)
        elif p.suffix == ".npz":
            data = np.load(p, allow_pickle=True)
            df = pd.DataFrame({k: data[k] for k in data.files})
        else:
            raise ValidationError(f"不支持的数据格式: {p.suffix}（支持 .csv/.npz）")
    if target not in df.columns:
        raise ValidationError(f"数据集缺少目标列 {target!r}；现有列: {list(df.columns)[:20]}")
    return df


def describe_dataset(task) -> dict:
    """给面板/报告用的数据集描述。"""
    try:
        df = load_dataframe(task.data_path, task.target)
    except Exception as exc:  # noqa: BLE001 - 描述失败不应让主流程崩溃
        return {"available": False, "error": str(exc), "data_path": task.data_path}
    return {
        "available": True,
        "data_path": task.data_path,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": [str(c) for c in df.columns[:50]],
    }
