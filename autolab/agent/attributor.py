"""失败归因与重规划：把"没达标"翻译成结构化的原因 + 可执行修复动作。

归因优先级（先规则、后大模型；规则可解释、可离线、可现场演示）：
  1. 执行失败（超时 / 非法方案 / 运行错误）
  2. 缺失值处理不到位（只零填充或没填）
  3. 类别失衡（未加 balanced 类权重）
  4. 特征未标准化（线性/距离类模型）
  5. 过拟合（训练折远好于验证折）
  6. 欠拟合（训练折也不达标）
"""
from __future__ import annotations

import copy

from ..schemas import Attribution, ExperimentPlan


def evaluate_target(score, threshold: float, higher_is_better: bool) -> bool:
    """判断指标是否达标。"""
    if score is None:
        return False
    return score >= threshold if higher_is_better else score <= threshold


def _op_names(plan: ExperimentPlan) -> list[str]:
    return plan.op_names()


def attribute(profile: dict, plan: ExperimentPlan, result, task) -> list[Attribution]:
    """返回按置信度排序的归因列表（空表示无法归因）。"""
    causes: list[Attribution] = []
    metric = task.metric
    score = result.metric_mean(metric)
    target = profile.get("target", {}) or {}
    task_type = getattr(task, "task_type", "")

    op_names = _op_names(plan)
    model_params = (plan.model or {}).get("params") or {}
    model_kind = str((plan.model or {}).get("kind", ""))
    # 最近质心这类弱 baseline 不支持 class_weight；归因时直接升级为可加权的线性模型
    weak_model = model_kind == "nearest_centroid"
    strong_model = {
        "kind": "logistic_gd",
        "params": {"lr": 0.1, "epochs": 400, "l2": 0.001, "class_weight": "balanced"},
    }

    if result.status == "timeout":
        return [
            Attribution(
                cause="budget_exceeded",
                confidence=0.9,
                evidence=result.error,
                fix={"reduce_splits": True, "set_model_params": {"epochs": 150}},
            )
        ]
    if result.status in ("error", "invalid"):
        text = str(result.error or "")
        error_fix: dict = {}
        if "缺失值" in text or "impute" in text:
            # 沙箱自愈：方案漏了填充算子时，下一轮直接补中位数填充
            error_fix = {
                "add_ops": [{"op": "impute_median", "params": {}}],
                "remove_ops": ["impute_zero"],
            }
            if weak_model:
                error_fix["set_model"] = copy.deepcopy(strong_model)
        return [
            Attribution(
                cause="execution_failed",
                confidence=0.9,
                evidence=text[:400],
                fix=error_fix,
            )
        ]
    if score is None:
        return []

    # 1) 缺失值处理不到位
    has_missing = bool(profile.get("columns_with_missing") or {}) or float(
        profile.get("missing_ratio", 0.0)
    ) > 0.02
    if has_missing:
        if "impute_mean" not in op_names and "impute_median" not in op_names:
            fix: dict = {"add_ops": [{"op": "impute_median", "params": {}}]}
            if "impute_zero" in op_names:
                fix["remove_ops"] = ["impute_zero"]
            if weak_model:
                fix["set_model"] = copy.deepcopy(strong_model)
            causes.append(
                Attribution(
                    cause="missing_values_poor_imputation",
                    confidence=0.75,
                    evidence=(
                        f"缺失比例 {profile.get('missing_ratio'):.3f}，当前算子 {op_names}："
                        "零填充会偏移特征分布，改用中位数填充更稳"
                    ),
                    fix=fix,
                )
            )

    # 2) 类别失衡
    imbalance = target.get("imbalance_ratio")
    if (
        task_type != "regression"
        and imbalance
        and float(imbalance) > 1.5
        and str(model_params.get("class_weight", "none")) != "balanced"
    ):
        imbalance_fix = (
            {"set_model": copy.deepcopy(strong_model)}
            if weak_model
            else {"set_model_params": {"class_weight": "balanced"}}
        )
        causes.append(
            Attribution(
                cause="class_imbalance",
                confidence=0.8,
                evidence=f"类别失衡比 {imbalance}，模型未启用 class_weight=balanced",
                fix=imbalance_fix,
            )
        )

    # 3) 特征未标准化
    if "scale_standard" not in op_names and task_type != "regression":
        causes.append(
            Attribution(
                cause="feature_scale",
                confidence=0.6,
                evidence="未标准化特征：距离/梯度类模型对量纲敏感",
                fix={"add_ops": [{"op": "scale_standard", "params": {}}]},
            )
        )

    # 4) 过拟合 / 欠拟合
    train_score = result.train_metric
    if train_score is not None:
        gap = float(train_score) - float(score) if task.higher_is_better else float(score) - float(train_score)
        if gap > 0.08:
            causes.append(
                Attribution(
                    cause="overfit",
                    confidence=0.55,
                    evidence=f"训练折指标 {train_score:.4f} 明显好于验证折 {score:.4f}（gap={gap:.3f}）",
                    fix={"set_model_params": {"l2": 0.01, "epochs": 200}},
                )
            )
        elif float(train_score) < (task.threshold - 0.04) and task.higher_is_better:
            causes.append(
                Attribution(
                    cause="underfit",
                    confidence=0.6,
                    evidence=f"训练折指标也只有 {train_score:.4f}，模型容量/参数明显不足",
                    fix={
                        "set_model": {
                            "kind": "logistic_gd",
                            "params": {"lr": 0.1, "epochs": 400, "l2": 0.001, "class_weight": "balanced"},
                        }
                    },
                )
            )

    if not causes:
        causes.append(
            Attribution(
                cause="unknown_shortfall",
                confidence=0.3,
                evidence=f"指标 {score:.4f} 未达阈值 {task.threshold}，规则库未命中明确原因",
                fix={
                    "set_model": {
                        "kind": "logistic_gd",
                        "params": {"lr": 0.1, "epochs": 400, "l2": 0.001, "class_weight": "balanced"},
                    },
                    "add_ops": [{"op": "scale_standard", "params": {}}],
                },
            )
        )
    causes.sort(key=lambda c: c.confidence, reverse=True)
    return causes


def replan(plan: ExperimentPlan, causes: list[Attribution], task) -> ExperimentPlan:
    """把归因建议合并成下一轮方案（去重、排序、可复现）。"""
    new_plan = plan.copy_with_id("attribution")
    set_model: dict | None = None
    params_patch: dict = {}
    added_ops: list[dict] = []

    for cause in causes:
        fix = cause.fix or {}
        for op in fix.get("add_ops", []):
            if str(op.get("op")) not in {str(o.get("op")) for o in new_plan.ops} | {
                str(o.get("op")) for o in added_ops
            }:
                added_ops.append(dict(op))
        for name in fix.get("remove_ops", []):
            new_plan.ops = [o for o in new_plan.ops if o.get("op") != name]
        if "set_model" in fix and set_model is None:
            set_model = copy.deepcopy(fix["set_model"])
        params_patch.update(fix.get("set_model_params") or {})
        if fix.get("reduce_splits"):
            new_plan.eval["n_splits"] = max(2, int(new_plan.eval.get("n_splits", 5)) - 1)

    new_plan.ops.extend(added_ops)
    if set_model is not None:
        new_plan.model = set_model
    if params_patch:
        new_plan.model.setdefault("params", {}).update(params_patch)

    # 固定算子顺序：填充 → 变换/编码 → 过滤 → 标准化
    order = {
        "drop_missing_rows": 0,
        "impute_mean": 1,
        "impute_median": 1,
        "impute_zero": 1,
        "log1p": 2,
        "onehot": 2,
        "variance_filter": 3,
        "scale_standard": 4,
    }
    new_plan.ops.sort(key=lambda o: order.get(str(o.get("op")), 9))
    new_plan.source = "attribution"
    new_plan.notes = "；".join(
        f"{c.cause}({c.confidence:.2f})" for c in causes[:3]
    )
    return new_plan
