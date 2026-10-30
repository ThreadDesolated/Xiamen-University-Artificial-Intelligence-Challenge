"""提示词：让大模型只做"规划/解释/归因"，产物永远是受约束的 JSON。"""
from __future__ import annotations

import json


def planner_system_prompt(whitelist: dict) -> str:
    compact = {
        "ops": {name: sorted((defn.get("params") or {}).keys()) for name, defn in whitelist.get("ops", {}).items()},
        "models": {
            name: {
                "tasks": defn.get("tasks", []),
                "params": sorted((defn.get("params") or {}).keys()),
            }
            for name, defn in whitelist.get("models", {}).items()
        },
    }
    return (
        "你是 AutoLab 的实验方案规划器。你的任务是根据数据画像和实验目标，产出一份【受约束的实验方案 JSON】。\n"
        "硬性规则：\n"
        "1. 只能使用白名单里的算子与模型，参数必须在允许范围内；\n"
        "2. 方案必须是 JSON（不要 markdown 代码块、不要解释文字），格式：\n"
        '   {"ops": [{"op": "...", "params": {}}], "model": {"kind": "...", "params": {}}, '
        '"eval": {"n_splits": 5, "seed": 42}, "notes": "一句话说明设计理由"}\n'
        "3. 特征未标准化时不要用线性/距离类模型；有缺失值时必须有 impute_* 算子；类别失衡时考虑 class_weight=balanced；\n"
        "4. 只做一轮方案规划，不要写代码、不要输出训练过程。\n\n"
        f"白名单（算子名 → 参数名；模型名 → 支持任务与参数名）：\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
    )


def planner_user_prompt(profile: dict, task) -> str:
    task_view = {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "target": task.target,
        "metric": task.metric,
        "threshold": task.threshold,
        "higher_is_better": task.higher_is_better,
        "n_splits": task.n_splits,
        "budget_seconds": task.budget_seconds,
    }
    profile_view = {
        "n_rows": profile.get("n_rows"),
        "n_features": profile.get("n_features"),
        "feature_types": profile.get("feature_types"),
        "missing_ratio": profile.get("missing_ratio"),
        "target": profile.get("target"),
        "warnings": profile.get("warnings"),
    }
    return (
        f"任务：{json.dumps(task_view, ensure_ascii=False)}\n"
        f"数据画像：{json.dumps(profile_view, ensure_ascii=False)}\n"
        "请输出实验方案 JSON。"
    )


def attribution_system_prompt() -> str:
    return (
        "你是 AutoLab 的失败归因助手。给定一次未达标的实验（方案、指标、训练/验证差距、数据画像），"
        "请判断最可能的失败原因，并给出可在白名单内执行的修复动作。\n"
        "输出 JSON：{\"causes\": [{\"cause\": \"...\", \"confidence\": 0.0-1.0, "
        "\"evidence\": \"...\", \"fix\": {\"add_ops\": [...], \"remove_ops\": [...], "
        "\"set_model\": {...}, \"set_model_params\": {...}}}], \"summary\": \"...\"}\n"
        "只输出 JSON。"
    )
