"""提示词：让大模型只做"规划/解释/归因"，产物永远是受约束的 JSON。"""
from __future__ import annotations

import json


def _compact_whitelist(whitelist: dict) -> dict:
    """把白名单压成"名字 → 参数规格"，让模型看到 enum 取值与数值范围，少编参数。"""
    return {
        "ops": {name: defn.get("params", {}) for name, defn in whitelist.get("ops", {}).items()},
        "models": {
            name: {"tasks": defn.get("tasks", []), "params": defn.get("params", {})}
            for name, defn in whitelist.get("models", {}).items()
        },
        "limits": whitelist.get("limits", {}),
    }


def planner_system_prompt(whitelist: dict) -> str:
    compact = _compact_whitelist(whitelist)
    return (
        "你是 AutoLab 的实验方案规划器。你的任务是根据数据画像和实验目标，产出一份【受约束的实验方案 JSON】。\n"
        "硬性规则：\n"
        "1. 只能使用白名单里的算子与模型；参数名、类型、范围和 enum 取值都必须符合规格；\n"
        "2. 方案必须是 JSON（不要 markdown 代码块、不要解释文字），格式：\n"
        '   {"ops": [{"op": "...", "params": {}}], "model": {"kind": "...", "params": {}}, '
        '"eval": {"n_splits": 5, "seed": 42}, "notes": "一句话说明设计理由"}\n'
        "3. 特征未标准化时不要用线性/距离类模型；有缺失值时必须有 impute_* 算子；类别失衡时考虑 class_weight=balanced；\n"
        "4. 只做一轮方案规划，不要写代码、不要输出训练过程。\n\n"
        f"白名单（参数规格：type/min/max/values/default）：\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
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


def attribution_system_prompt(causes: tuple[str, ...], whitelist: dict) -> str:
    compact = _compact_whitelist(whitelist)
    return (
        "你是 AutoLab 的失败归因助手。给定一次未达标的实验（数据画像、方案、指标、训练/验证差距），"
        "请判断最可能的失败原因，并给出可在白名单内执行的修复动作。\n"
        "硬性规则：\n"
        f"1. cause 只能取：{', '.join(causes)}；\n"
        "2. 每条归因必须给出 evidence（引用具体指标/差距）和可在下一轮直接执行的 fix；\n"
        "3. fix 只允许 add_ops / remove_ops / set_model / set_model_params / reduce_splits，"
        "算子与模型必须来自白名单，参数名、类型、范围和 enum 取值都必须符合规格；\n"
        "4. 指标口径：accuracy 不要建议 class_weight=balanced；若当前方案已用 balanced，"
        "应建议改成 none；只有 f1_macro 这类对少数类敏感的指标才考虑 balanced；\n"
        "5. 最多 3 条，按 confidence 从高到低；不要写代码。\n"
        "输出 JSON：{\"causes\": [{\"cause\": \"...\", \"confidence\": 0.0-1.0, "
        "\"evidence\": \"...\", \"fix\": {...}}], \"summary\": \"一句话归因结论\"}\n"
        "只输出 JSON。\n\n"
        f"白名单：\n{json.dumps(compact, ensure_ascii=False, indent=2)}"
    )


def attribution_user_prompt(profile: dict, plan, result, task, rule_hints: list[dict]) -> str:
    result_view = {
        "status": result.status,
        "error": str(result.error or "")[:300],
        "metric": task.metric,
        "mean": result.metric_mean(task.metric),
        "train_metric": result.train_metric,
        "threshold": task.threshold,
        "higher_is_better": task.higher_is_better,
        "seconds": result.seconds,
        "holdout": result.holdout,
    }
    profile_view = {
        "n_rows": profile.get("n_rows"),
        "n_features": profile.get("n_features"),
        "feature_types": profile.get("feature_types"),
        "missing_ratio": profile.get("missing_ratio"),
        "target": profile.get("target"),
        "warnings": profile.get("warnings"),
    }
    plan_view = {
        "ops": plan.ops,
        "model": plan.model,
        "eval": plan.eval,
        "source": plan.source,
    }
    return (
        f"任务类型：{task.task_type}；指标方向：{'越高越好' if task.higher_is_better else '越低越好'}\n"
        f"数据画像：{json.dumps(profile_view, ensure_ascii=False)}\n"
        f"当前方案：{json.dumps(plan_view, ensure_ascii=False)}\n"
        f"执行结果：{json.dumps(result_view, ensure_ascii=False)}\n"
        f"规则库参考（可采纳也可推翻）：{json.dumps(rule_hints, ensure_ascii=False)}\n"
        "请输出归因 JSON。"
    )
