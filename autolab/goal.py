"""自然语言目标解析：把"一句话目标"翻译成可执行的评测约束。

支持三类约束，缺省时沿用任务配置：
- 指标阈值："准确率做到 0.86 以上" / "阈值 0.8"
- 时间预算："预算 90 秒" / "2 分钟内"
- 最大轮次："最多 2 轮" / "3 轮以内"

不调用大模型：确定性正则解析，离线可用、可测试，现场不会因为限流失效。
解析结果写进 trace 的 `goal` 事件，报告头部同步展示。
"""
from __future__ import annotations

import dataclasses
import re

MIN_BUDGET_SECONDS = 5
MAX_BUDGET_SECONDS = 3600
MAX_ROUNDS_LIMIT = 10

_BUDGET_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(分钟|分|min(?:ute)?s?|m|秒钟|秒|s)", re.I)
_ROUNDS_RE = re.compile(r"(?:最多|不超过|至多)?\s*(\d+)\s*(?:轮次|轮|rounds?)", re.I)
_DECIMAL_RE = re.compile(r"(?<![\d.])(0?\.\d+)(?![\d])")

#: 只有带这些上下文的小数才算"指标阈值"，避免把"留出集 0.2"当成阈值
_THRESHOLD_CUES = (
    "阈值", "指标", "准确率", "精确率", "召回率", "分数", "达标",
    "以上", "以下", "不低于", "不高于", "至少", "达到", "做到", "提升",
    "高于", "低于", "accuracy", "acc", "f1", "auc", "mae", "rmse", "r2",
)


def _parse_threshold(text: str) -> float | None:
    for match in _DECIMAL_RE.finditer(text):
        value = float(match.group(1))
        if not 0.0 < value < 1.0:
            continue
        window = text[max(0, match.start() - 16) : match.end() + 8].lower()
        if any(cue in window for cue in _THRESHOLD_CUES):
            return round(value, 4)
    return None


def _parse_budget(text: str) -> int | None:
    match = _BUDGET_RE.search(text)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("分") or unit.startswith("min") or unit == "m":
        seconds = amount * 60
    else:
        seconds = amount
    seconds_int = int(round(seconds))
    if not MIN_BUDGET_SECONDS <= seconds_int <= MAX_BUDGET_SECONDS:
        return None
    return seconds_int


def _parse_rounds(text: str) -> int | None:
    match = _ROUNDS_RE.search(text)
    if not match:
        return None
    rounds = int(match.group(1))
    if not 1 <= rounds <= MAX_ROUNDS_LIMIT:
        return None
    return rounds


def parse_goal(text: str) -> dict:
    """解析一句话目标，返回契约字段：text / threshold / budget_seconds / max_rounds / applied。"""
    text = (text or "").strip()
    threshold = _parse_threshold(text) if text else None
    budget_seconds = _parse_budget(text) if text else None
    max_rounds = _parse_rounds(text) if text else None

    applied: dict = {}
    if threshold is not None:
        applied["threshold"] = threshold
    if budget_seconds is not None:
        applied["budget_seconds"] = budget_seconds
    if max_rounds is not None:
        applied["max_rounds"] = max_rounds

    return {
        "text": text,
        "threshold": threshold,
        "budget_seconds": budget_seconds,
        "max_rounds": max_rounds,
        "applied": applied,
    }


def apply_goal(task, goal: dict | None):
    """把解析出的约束合并进 TaskSpec（不改原对象），返回新任务。"""
    if not goal:
        return task
    overrides: dict = {}
    if goal.get("threshold") is not None:
        overrides["threshold"] = float(goal["threshold"])
    if goal.get("budget_seconds") is not None:
        overrides["budget_seconds"] = int(goal["budget_seconds"])
    if goal.get("max_rounds") is not None:
        overrides["max_rounds"] = int(goal["max_rounds"])
    if not overrides:
        return task
    return dataclasses.replace(task, **overrides).validate()
