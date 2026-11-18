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
import importlib.util

from ..schemas import Attribution, ExperimentPlan


def evaluate_target(score, threshold: float, higher_is_better: bool) -> bool:
    """判断指标是否达标。"""
    if score is None:
        return False
    return score >= threshold if higher_is_better else score <= threshold


def _op_names(plan: ExperimentPlan) -> list[str]:
    return plan.op_names()


# 模型容量等级：重规划时若有多个 set_model 建议，取容量最高的一条（不能被弱模型建议覆盖）
_CAPACITY_RANK = {
    "nearest_centroid": 0,
    "logistic_gd": 1,
    "linear_gd": 1,
    "random_forest": 2,
}
_LINEAR_KINDS = ("logistic_gd", "linear_gd")
_STRONG_LINEAR_MODEL = {
    "kind": "logistic_gd",
    "params": {"lr": 0.1, "epochs": 400, "l2": 0.001, "class_weight": "balanced"},
}
_TREE_MODEL = {
    "kind": "random_forest",
    "params": {"n_estimators": 300, "max_depth": 8, "min_samples_leaf": 1},
}


def _sklearn_available() -> bool:
    """随机森林依赖 scikit-learn；最小依赖环境要能自动退回线性模型，保证 demo 可跑。"""
    try:
        return importlib.util.find_spec("sklearn") is not None
    except (ImportError, ValueError):  # pragma: no cover - 异常环境兜底
        return False


def _capacity_fix(model_kind: str, model_params: dict) -> dict:
    """欠拟合 / 未知缺口时升级模型容量。

    - 线性模型（含最近质心）→ sklearn 随机森林（scikit-learn 可用时）；
    - 已是随机森林 → 加深 / 加树；
    - 最小依赖环境（无 scikit-learn）→ 退回 logistic_gd + balanced，保持零依赖 demo 可跑。
    """
    if model_kind == "random_forest":
        depth = int(model_params.get("max_depth", 8) or 8)
        n_estimators = int(model_params.get("n_estimators", 300) or 300)
        return {
            "set_model_params": {
                "max_depth": min(20, depth + 2),
                "n_estimators": min(500, max(300, n_estimators + 100)),
            }
        }
    if _sklearn_available():
        fix = copy.deepcopy(_TREE_MODEL)
        if str(model_params.get("class_weight", "none")) == "balanced":
            fix["params"]["class_weight"] = "balanced"
        return {"set_model": fix}
    return {"set_model": copy.deepcopy(_STRONG_LINEAR_MODEL)}


def _regularize_fix(model_kind: str, model_params: dict) -> dict:
    """过拟合时按模型类型收缩容量（参数必须在白名单范围内）。"""
    if model_kind in _LINEAR_KINDS:
        return {"set_model_params": {"l2": 0.01, "epochs": 200}}
    if model_kind == "random_forest":
        depth = int(model_params.get("max_depth", 8) or 8)
        return {"set_model_params": {"max_depth": max(2, depth - 2)}}
    return {"set_model": copy.deepcopy(_STRONG_LINEAR_MODEL)}


def _timeout_fix(model_kind: str, model_params: dict) -> dict:
    """预算超时时减少折数，并按模型类型降低单次训练成本。"""
    fix: dict = {"reduce_splits": True}
    if model_kind in _LINEAR_KINDS:
        fix["set_model_params"] = {"epochs": 150}
    elif model_kind == "random_forest":
        n_estimators = int(model_params.get("n_estimators", 300) or 300)
        fix["set_model_params"] = {"n_estimators": max(50, n_estimators // 2)}
    return fix


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
    # 最近质心这类弱 baseline 不支持 class_weight；归因时先升级到可加权的线性模型
    weak_model = model_kind == "nearest_centroid"

    if result.status == "timeout":
        return [
            Attribution(
                cause="budget_exceeded",
                confidence=0.9,
                evidence=result.error,
                fix=_timeout_fix(model_kind, model_params),
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
                error_fix["set_model"] = copy.deepcopy(_STRONG_LINEAR_MODEL)
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
                fix["set_model"] = copy.deepcopy(_STRONG_LINEAR_MODEL)
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
            {"set_model": copy.deepcopy(_STRONG_LINEAR_MODEL)}
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
                    fix=_regularize_fix(model_kind, model_params),
                )
            )
        elif float(train_score) < (task.threshold - 0.04) and task.higher_is_better:
            causes.append(
                Attribution(
                    cause="underfit",
                    confidence=0.6,
                    evidence=f"训练折指标也只有 {train_score:.4f}，模型容量/参数明显不足",
                    fix=_capacity_fix(model_kind, model_params),
                )
            )

    if not causes:
        causes.append(
            Attribution(
                cause="unknown_shortfall",
                confidence=0.3,
                evidence=f"指标 {score:.4f} 未达阈值 {task.threshold}，规则库未命中明确原因",
                fix={
                    **_capacity_fix(model_kind, model_params),
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
    set_model_rank = -1
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
        candidate = fix.get("set_model")
        if candidate:
            rank = _CAPACITY_RANK.get(str(candidate.get("kind", "")), 0)
            # 多条建议同时升级模型时取容量最高者，避免弱模型建议覆盖"升级树模型"
            if rank > set_model_rank:
                set_model = copy.deepcopy(candidate)
                set_model_rank = rank
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
