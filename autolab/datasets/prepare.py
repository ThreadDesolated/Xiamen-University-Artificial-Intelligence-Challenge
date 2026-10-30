"""真实公开数据集准备脚本（W2/W3 用；需要联网，只需跑一次）。

用法：
    python -m autolab prepare --task t1_adult
    python -m autolab prepare --task t2_digits
    python -m autolab prepare --task t3_timeseries

下载结果缓存到 data/raw/，处理后的固定子集写到 data/processed/。
所有处理都固定随机种子，保证可复现。
"""
from __future__ import annotations

import pathlib
import urllib.request

import numpy as np
import pandas as pd

from .. import config

TIMEOUT_SECONDS = 60


def _download(url: str, dest: pathlib.Path) -> pathlib.Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    print(f"[prepare] 下载 {url}")
    with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as resp:  # noqa: S310
        dest.write_bytes(resp.read())
    return dest


def _save(df: pd.DataFrame, name: str) -> pathlib.Path:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PROCESSED_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"[prepare] 写出 {path}（{len(df)} 行 x {df.shape[1]} 列）")
    return path


# --------------------------------------------------------------------------- #
# T1：UCI Adult（收入二分类，固定 1000 行分层子集）
# --------------------------------------------------------------------------- #
ADULT_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]


def prepare_t1_adult(seed: int = 42, n_rows: int = 1000) -> pathlib.Path:
    raw = _download(ADULT_URL, config.RAW_DIR / "adult.data")
    df = pd.read_csv(raw, names=ADULT_COLUMNS, skipinitialspace=True, na_values="?")
    df["income"] = df["income"].str.strip().map({"<=50K": 0, ">50K": 1})
    df = df.dropna(subset=["income"])
    for col in ("workclass", "occupation", "native_country"):
        df[col] = df[col].fillna("__missing__")
    # 分层抽样：保持 25% 左右正类比例
    rng = np.random.default_rng(seed)
    positives = df[df["income"] == 1].sample(frac=1.0, random_state=seed)
    negatives = df[df["income"] == 0].sample(frac=1.0, random_state=seed)
    n_pos = int(n_rows * 0.25)
    sample = pd.concat([positives.head(n_pos), negatives.head(n_rows - n_pos)])
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    assert len(sample) == n_rows
    return _save(sample, "t1_adult")


# --------------------------------------------------------------------------- #
# T2：UCI 手写数字（optdigits，2000 张 8x8 展平）
# --------------------------------------------------------------------------- #
OPTDIGITS_URLS = {
    "train": "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",
    "test": "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes",
}


def prepare_t2_digits(seed: int = 42, n_rows: int = 2000) -> pathlib.Path:
    frames = []
    for split, url in OPTDIGITS_URLS.items():
        raw = _download(url, config.RAW_DIR / f"optdigits_{split}.txt")
        frame = pd.read_csv(raw, header=None)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).head(n_rows).reset_index(drop=True)
    df.columns = [f"p{i}" for i in range(df.shape[1] - 1)] + ["label"]
    return _save(df, "t2_digits")


# --------------------------------------------------------------------------- #
# T3：UCI 空气质量时序（滑窗特征回归，1-2k 点）
# --------------------------------------------------------------------------- #
AIRQUALITY_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"


def prepare_t3_timeseries(seed: int = 42, n_rows: int = 1200) -> pathlib.Path:
    import io
    import zipfile

    raw = _download(AIRQUALITY_URL, config.RAW_DIR / "AirQualityUCI.zip")
    with zipfile.ZipFile(raw) as zf:
        name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
        df = pd.read_csv(io.BytesIO(zf.read(name)), sep=";", decimal=",")
    df = df[["CO(GT)", "PT08.S1(CO)", "NOx(GT)", "PT08.S2(NMHC)", "C6H6(GT)", "T", "RH"]]
    df = df.replace(-200.0, np.nan).dropna().reset_index(drop=True)
    series = df["CO(GT)"].astype(float)
    frame = pd.DataFrame({
        "lag1": series.shift(1),
        "lag2": series.shift(2),
        "lag3": series.shift(3),
        "roll3": series.shift(1).rolling(3).mean(),
        "temp": df["T"].astype(float),
        "humidity": df["RH"].astype(float),
        "target": series,
    }).dropna().reset_index(drop=True)
    frame = frame.head(n_rows)
    # 目标标准化，便于使用统一的阈值口径
    frame["target"] = (frame["target"] - frame["target"].mean()) / frame["target"].std()
    return _save(frame, "t3_timeseries")


PREPARERS = {
    "t1_adult": prepare_t1_adult,
    "t2_digits": prepare_t2_digits,
    "t3_timeseries": prepare_t3_timeseries,
}


def prepare_task(task_id: str) -> pathlib.Path:
    if task_id not in PREPARERS:
        raise SystemExit(f"未知任务 {task_id!r}；可用: {list(PREPARERS)}")
    return PREPARERS[task_id]()
