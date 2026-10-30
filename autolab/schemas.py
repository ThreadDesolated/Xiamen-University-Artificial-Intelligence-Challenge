"""AutoLab 核心数据结构（与 docs/接口契约.md 一一对应）。

约定：所有结构均可与 JSON 互转，字段名即契约字段名；不要随意改名。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = "0.1.0"


class ValidationError(ValueError):
    """任务配置 / 实验方案不合法。"""


def new_id(prefix: str) -> str:
    """生成短 id：<prefix>_<时间>_<随机>。"""
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


class _JsonMixin:
    """dataclass <-> dict/JSON 互转。"""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, data: dict):
        names = {f.name for f in dataclasses.fields(cls)}  # type: ignore[arg-type]
        return cls(**{k: v for k, v in data.items() if k in names})  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# 1. 任务配置
# --------------------------------------------------------------------------- #
@dataclass
class TaskSpec(_JsonMixin):
    task_id: str
    name: str
    task_type: str
    data_path: str
    target: str
    metric: str = "accuracy"
    threshold: float = 0.85
    higher_is_better: bool = True
    n_splits: int = 5
    seed: int = 42
    budget_seconds: int = 120
    max_rounds: int = 3
    features: Optional[list[str]] = None
    notes: str = ""

    def validate(self) -> "TaskSpec":
        from .config import SUPPORTED_METRICS, SUPPORTED_TASK_TYPES

        if self.task_type not in SUPPORTED_TASK_TYPES:
            raise ValidationError(f"不支持的 task_type: {self.task_type}")
        allowed = SUPPORTED_METRICS[self.task_type]
        if self.metric not in allowed:
            raise ValidationError(f"{self.task_type} 只支持指标 {allowed}，收到 {self.metric}")
        if self.n_splits < 2:
            raise ValidationError("n_splits 至少为 2")
        if self.budget_seconds <= 0 or self.max_rounds <= 0:
            raise ValidationError("budget_seconds / max_rounds 必须为正数")
        return self

    @classmethod
    def load(cls, path) -> "TaskSpec":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f)).validate()  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 2. 实验方案（受限动作空间）
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentPlan(_JsonMixin):
    """Agent 产出的实验方案：算子序列 + 模型 + 评估协议。"""

    plan_id: str = ""
    ops: list = field(default_factory=list)  # [{"op": "impute_mean", "params": {}}, ...]
    model: dict = field(default_factory=dict)  # {"kind": "logistic_gd", "params": {...}}
    eval: dict = field(default_factory=dict)  # {"n_splits": 5, "seed": 42}
    source: str = "template"  # template | llm | attribution | recipe
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.plan_id:
            self.plan_id = new_id("plan")

    def op_names(self) -> list[str]:
        return [str(o.get("op", "")) for o in self.ops]

    def fingerprint(self) -> str:
        payload = json.dumps(
            {"ops": self.ops, "model": self.model, "eval": self.eval},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def copy_with_id(self, source: str) -> "ExperimentPlan":
        return ExperimentPlan(
            plan_id=new_id("plan"),
            ops=[dict(o) for o in self.ops],
            model=dict(self.model),
            eval=dict(self.eval),
            source=source,
            notes=self.notes,
        )


# --------------------------------------------------------------------------- #
# 3. 指标 / 实验结果
# --------------------------------------------------------------------------- #
@dataclass
class Metrics(_JsonMixin):
    name: str
    mean: float
    std: float = 0.0
    values: list = field(default_factory=list)
    n: int = 0

    def __str__(self) -> str:
        return f"{self.name}={self.mean:.4f}±{self.std:.4f}"


@dataclass
class ExperimentResult(_JsonMixin):
    status: str = "ok"  # ok | error | timeout | invalid
    metrics: dict = field(default_factory=dict)  # {metric_name: Metrics.to_dict()}
    train_metric: Optional[float] = None
    seconds: float = 0.0
    error: str = ""
    stdout_tail: str = ""
    warnings: list = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)

    def metric_mean(self, name: str) -> Optional[float]:
        item = self.metrics.get(name)
        if not item:
            return None
        return float(item.get("mean"))  # type: ignore[union-attr]

    @property
    def ok(self) -> bool:
        return self.status == "ok"


# --------------------------------------------------------------------------- #
# 4. 归因 / trace
# --------------------------------------------------------------------------- #
@dataclass
class Attribution(_JsonMixin):
    cause: str
    confidence: float
    evidence: str = ""
    fix: dict = field(default_factory=dict)  # 建议的方案修改
    round_idx: int = 0


@dataclass
class TraceEvent(_JsonMixin):
    run_id: str
    seq: int
    ts: float
    node: str
    tool: str = ""
    input: Any = None
    output: Any = None
    duration_ms: Optional[float] = None


# --------------------------------------------------------------------------- #
# 5. 小工具
# --------------------------------------------------------------------------- #
def dump_json(path, data) -> None:
    """原子写入 JSON（先写临时文件再替换）。"""
    import pathlib

    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(p)


def load_json(path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
