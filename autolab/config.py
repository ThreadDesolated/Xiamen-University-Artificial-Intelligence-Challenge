"""运行配置：路径、环境变量、白名单加载。

所有路径都相对仓库根目录解析；可用环境变量覆盖（见 .env.example）。
"""
from __future__ import annotations

import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_dotenv(path: pathlib.Path) -> None:
    """极简 .env 解析（避免强依赖 python-dotenv）。已存在的环境变量优先。"""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_dotenv(ROOT / ".env")


def _path(name: str, default: str) -> pathlib.Path:
    value = os.environ.get(name, default)
    p = pathlib.Path(value)
    return p if p.is_absolute() else ROOT / p


TRACE_DIR = _path("AUTOLAB_TRACE_DIR", "traces")
REPORT_DIR = _path("AUTOLAB_REPORT_DIR", "reports")
DB_PATH = _path("AUTOLAB_DB_PATH", "autolab.sqlite3")
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TASK_DIR = ROOT / "configs" / "tasks"
WHITELIST_PATH = ROOT / "configs" / "whitelist_ops.json"

SEED = int(os.environ.get("AUTOLAB_SEED", "42"))
MODEL = os.environ.get("AUTOLAB_MODEL", "deepseek-chat")
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

SUPPORTED_TASK_TYPES = ("binary_classification", "multiclass_classification", "regression")
SUPPORTED_METRICS = {
    "binary_classification": ("accuracy", "f1_macro"),
    "multiclass_classification": ("accuracy", "f1_macro"),
    "regression": ("mae", "rmse", "r2"),
}


def load_whitelist() -> dict:
    """加载动作空间白名单（configs/whitelist_ops.json）。"""
    with open(WHITELIST_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_task_config(task: str) -> dict:
    """按 任务 id（t1_adult）或配置文件路径 加载任务配置。"""
    p = pathlib.Path(task)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        # 当成任务 id 处理
        p = TASK_DIR / f"{task}.json"
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs() -> None:
    for d in (TRACE_DIR, REPORT_DIR, RAW_DIR, PROCESSED_DIR):
        d.mkdir(parents=True, exist_ok=True)
