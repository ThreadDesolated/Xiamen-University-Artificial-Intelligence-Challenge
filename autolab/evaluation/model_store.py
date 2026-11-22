"""训练后模型导出与加载（只导出最终模型）。

一次 Agent 运行会产生多个候选方案，但真正对外交付的只有"最终模型"：

- 内容：预处理 Pipeline + 模型本体 + 可追溯元信息（任务、方案、CV 均值、类别映射等）；
- 文件：reports/<run_id>/model.joblib（随报告、复现包 zip、面板下载一起交付）；
- 依赖：优先用 joblib（scikit-learn 自带）序列化；未安装 joblib 时自动
  退回标准库 pickle，保证零依赖的 numpy demo（t1_demo）也能导出模型。
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

MODEL_FORMAT = "autolab.model.v1"
MODEL_FILENAME = "model.joblib"


def _joblib():
    try:
        import joblib

        return joblib
    except ImportError:  # pragma: no cover - 零依赖环境才会走到
        return None


def _serializer() -> str:
    return "joblib" if _joblib() is not None else "pickle"


def _dump(payload: dict, path: Path, serializer: str) -> None:
    joblib = _joblib()
    if serializer == "joblib" and joblib is not None:
        # 模型文件要随报告/面板下载交付，压缩一下体积（joblib.load 会自动解压）
        joblib.dump(payload, path, compress=("zlib", 3))
        return
    with path.open("wb") as f:
        pickle.dump(payload, f)


def _load(path: Path) -> dict:
    joblib = _joblib()
    if joblib is not None:
        return joblib.load(path)  # joblib.load 同时能读 pickle 文件
    with path.open("rb") as f:
        return pickle.load(f)


def _sklearn_version() -> str | None:
    try:
        import sklearn

        return str(sklearn.__version__)
    except Exception:  # noqa: BLE001
        return None


def save_model(
    path,
    pipe,
    model,
    *,
    plan,
    task_type: str | None = None,
    target: str | None = None,
    metric: str | None = None,
    task_id: str | None = None,
    classes=None,
    cv_mean: float | None = None,
) -> dict:
    """把「预处理 Pipeline + 模型」导出为单文件，返回模型元信息（可直接 JSON 化）。"""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    serializer = _serializer()
    meta = {
        "format": MODEL_FORMAT,
        "serializer": serializer,
        "created_at": time.time(),
        "task_id": task_id,
        "task_type": task_type,
        "target": target,
        "metric": metric,
        "model_kind": (plan.model or {}).get("kind"),
        "model_params": (plan.model or {}).get("params") or {},
        "classes": list(classes) if classes is not None else None,
        "cv_mean": None if cv_mean is None else float(cv_mean),
        "sklearn_version": _sklearn_version(),
        "plan": plan.to_dict(),
    }
    _dump({"meta": meta, "pipeline": pipe, "model": model}, target_path, serializer)
    return {
        "model_path": str(target_path),
        "format": MODEL_FORMAT,
        "serializer": serializer,
        "model_kind": meta["model_kind"],
        "model_params": meta["model_params"],
        "cv_mean": meta["cv_mean"],
        "bytes": int(target_path.stat().st_size),
    }


def load_model(path) -> dict:
    """加载 AutoLab 导出的模型 payload（dict：meta / pipeline / model）。"""
    payload = _load(Path(path))
    if not isinstance(payload, dict) or (payload.get("meta") or {}).get("format") != MODEL_FORMAT:
        raise ValueError("不是 AutoLab 导出的模型文件")
    return payload


def predict(payload: dict, X):
    """用导出的模型预测；分类场景自动映射回原始类别标签。"""
    pipe = payload["pipeline"]
    model = payload["model"]
    raw = model.predict(pipe.transform(X))
    classes = (payload.get("meta") or {}).get("classes")
    if classes is not None:
        return [classes[int(i)] for i in raw]
    return list(raw)
