"""失败归因与重规划：把"没达标"翻译成结构化的原因 + 可执行修复动作。

归因优先级：配置了大模型时先让 `llm_attribute` 产出归因与修复；模型不可用、超时或
输出不合规（原因不在枚举、修复过不了白名单）时，整批回退到下面的规则库。
规则库本身覆盖：
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

from .. import config
from ..llm import chat_json
from ..schemas import Attribution, ExperimentPlan, ValidationError
from ..sandbox.validate import validate_plan

#: 归因类型枚举（提示词、前端标签、报告共用，模型只能从中选）
CAUSE_IDS = (
    "missing_values_poor_imputation",
    "class_imbalance",
    "metric_weight_mismatch",
    "feature_scale",
    "overfit",
    "underfit",
    "budget_exceeded",
    "execution_failed",
    "unknown_shortfall",
)
_FIX_KEYS = {"add_ops", "remove_ops", "set_model", "set_model_params", "reduce_splits"}


def _clean_fix(raw_fix) -> dict:
    """把模型给的 fix 收紧成 replan 能安全消费的结构；形状不对直接判不合规。"""
    if not isinstance(raw_fix, dict):
        return {}
    fix: dict = {}
    if "add_ops" in raw_fix:
        ops = raw_fix["add_ops"]
        if isinstance(ops, str):
            ops = [{"op": ops, "params": {}}]
        if not isinstance(ops, list):
            raise ValidationError("fix.add_ops 必须是数组")
        cleaned = []
        for op in ops[:8]:
            if isinstance(op, str):
                op = {"op": op, "params": {}}
            if not isinstance(op, dict) or not isinstance(op.get("op"), str):
                raise ValidationError("fix.add_ops 的每一项都必须是 {op, params} 对象")
            params = op.get("params", {})
            if params is None:
                params = {}
            if not isinstance(params, dict):
                raise ValidationError(f"算子 {op['op']} 的 params 必须是对象")
            cleaned.append({"op": op["op"], "params": params})
        fix["add_ops"] = cleaned
    if "remove_ops" in raw_fix:
        names = raw_fix["remove_ops"]
        if names is None:
            names = []
        if isinstance(names, str):
            names = [names]
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise ValidationError("fix.remove_ops 必须是字符串数组")
        fix["remove_ops"] = names[:8]
    if "set_model" in raw_fix:
        model = raw_fix["set_model"]
        if isinstance(model, str):
            model = {"kind": model, "params": {}}
        if not isinstance(model, dict) or not isinstance(model.get("kind"), str):
            raise ValidationError("fix.set_model 必须是 {kind, params} 对象")
        if not isinstance(model.get("params", {}), dict):
            raise ValidationError("fix.set_model.params 必须是对象")
        fix["set_model"] = {"kind": model["kind"], "params": model.get("params") or {}}
    if "set_model_params" in raw_fix:
        params = raw_fix["set_model_params"]
        if not isinstance(params, dict):
            raise ValidationError("fix.set_model_params 必须是对象")
        fix["set_model_params"] = params
    if "reduce_splits" in raw_fix:
        fix["reduce_splits"] = bool(raw_fix["reduce_splits"])
    return {key: value for key, value in fix.items() if key in _FIX_KEYS}


def _prune_model_params(fix: dict, plan: ExperimentPlan, whitelist: dict) -> dict:
    """丢掉模型不认识的参数（例如把线性模型的 l2 带到随机森林上）。

    模型偶尔会把参数包成 {"model": {...}, "params": {...}}；这里只保留目标模型
    白名单声明的键，剩余部分仍由 validate_plan 兜底校验。
    """
    params = fix.get("set_model_params")
    if not isinstance(params, dict) or not params:
        return fix
    model = fix.get("set_model")
    kind = str((model or {}).get("kind", "")) if isinstance(model, dict) else ""
    if not kind:
        kind = str((plan.model or {}).get("kind", ""))
    allowed = (((whitelist.get("models") or {}).get(kind) or {}).get("params") or {})
    pruned = {key: value for key, value in params.items() if key in allowed}
    if pruned:
        fix["set_model_params"] = pruned
    else:
        fix.pop("set_model_params", None)
    return fix


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
    "params": {"lr": 0.1, "epochs": 400, "l2": 0.001},
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


def _strong_linear(metric: str) -> dict:
    """强线性兜底模型；只有对少数类敏感的指标才默认带 balanced。"""
    model = copy.deepcopy(_STRONG_LINEAR_MODEL)
    if metric == "f1_macro":
        model["params"]["class_weight"] = "balanced"
    return model


def _capacity_fix(
    model_kind: str,
    model_params: dict,
    task_type: str = "binary_classification",
    metric: str = "",
) -> dict:
    """欠拟合 / 未知缺口时升级模型容量。

    - 回归任务只有 linear_gd 在白名单里，不能升级到分类树模型；
      改为加标准化 + 增加训练轮数，避免产出必然非法的方案；
    - 线性模型（含最近质心）→ sklearn 随机森林（scikit-learn 可用时）；
    - 已是随机森林 → 加深 / 加树；
    - 最小依赖环境（无 scikit-learn）→ 退回强化线性模型，保持零依赖 demo 可跑；
    - accuracy 这类指标不默认加 balanced，只有 f1_macro 才沿用/启用类权重。
    """
    if task_type == "regression":
        epochs = int(model_params.get("epochs", 300) or 300)
        return {
            "set_model_params": {"epochs": min(2000, max(300, epochs) + 300)},
            "add_ops": [{"op": "scale_standard", "params": {}}],
        }
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
        if metric == "f1_macro" and str(model_params.get("class_weight", "none")) == "balanced":
            fix["params"]["class_weight"] = "balanced"
        return {"set_model": fix}
    return {"set_model": _strong_linear(metric)}


def _regularize_fix(model_kind: str, model_params: dict, metric: str = "") -> dict:
    """过拟合时按模型类型收缩容量（参数必须在白名单范围内）。"""
    if model_kind in _LINEAR_KINDS:
        return {"set_model_params": {"l2": 0.01, "epochs": 200}}
    if model_kind == "random_forest":
        depth = int(model_params.get("max_depth", 8) or 8)
        return {"set_model_params": {"max_depth": max(2, depth - 2)}}
    return {"set_model": _strong_linear(metric)}


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
                error_fix["set_model"] = _strong_linear(metric)
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

    # 2) 类别失衡：只有 f1_macro 这类对少数类敏感的指标才值得加 balanced。
    #    accuracy 追求整体正确率，balanced 会牺牲多数类，通常反而掉分。
    imbalance = target.get("imbalance_ratio")
    if (
        task_type != "regression"
        and metric == "f1_macro"
        and imbalance
        and float(imbalance) > 1.5
        and str(model_params.get("class_weight", "none")) != "balanced"
    ):
        imbalance_fix = (
            {"set_model": _strong_linear(metric)}
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

    # 2b) 指标方向与权重口径冲突：accuracy 下沿用 balanced 会压低整体准确率
    if task_type != "regression" and metric == "accuracy" and str(
        model_params.get("class_weight", "none")
    ) == "balanced":
        causes.append(
            Attribution(
                cause="metric_weight_mismatch",
                confidence=0.85,
                evidence="当前指标是 accuracy，class_weight=balanced 会为少数类牺牲整体准确率",
                fix={"set_model_params": {"class_weight": "none"}},
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
                    fix=_regularize_fix(model_kind, model_params, metric),
                )
            )
        elif float(train_score) < (task.threshold - 0.04) and task.higher_is_better:
            causes.append(
                Attribution(
                    cause="underfit",
                    confidence=0.6,
                    evidence=f"训练折指标也只有 {train_score:.4f}，模型容量/参数明显不足",
                    fix=_capacity_fix(model_kind, model_params, task_type, metric),
                )
            )

    if not causes:
        causes.append(
            Attribution(
                cause="unknown_shortfall",
                confidence=0.3,
                evidence=f"指标 {score:.4f} 未达阈值 {task.threshold}，规则库未命中明确原因",
                fix={
                    **_capacity_fix(model_kind, model_params, task_type, metric),
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


def _strip_balanced(fix: dict) -> dict:
    """accuracy 口径下去掉 balanced 权重；返回裁剪后的新 fix。"""
    fix = copy.deepcopy(fix)
    params = fix.get("set_model_params")
    if isinstance(params, dict) and str(params.get("class_weight", "")).lower() == "balanced":
        params.pop("class_weight", None)
        if params:
            fix["set_model_params"] = params
        else:
            fix.pop("set_model_params", None)
    model = fix.get("set_model")
    if isinstance(model, dict):
        model_params = model.get("params") or {}
        if str(model_params.get("class_weight", "")).lower() == "balanced":
            model_params.pop("class_weight", None)
            model["params"] = model_params
    return fix


def _apply_metric_policy(
    causes: list[Attribution], plan: ExperimentPlan, task
) -> list[Attribution]:
    """按指标方向收紧归因：accuracy 不默认加 balanced，已带 balanced 则建议移除。"""
    if task.metric != "accuracy":
        return causes
    plan_params = (plan.model or {}).get("params") or {}
    cleaned: list[Attribution] = []
    for cause in causes:
        if cause.cause == "class_imbalance":
            continue
        fix = _strip_balanced(cause.fix or {})
        if fix:
            cleaned.append(
                Attribution(
                    cause=cause.cause,
                    confidence=cause.confidence,
                    evidence=cause.evidence,
                    fix=fix,
                )
            )
    if str(plan_params.get("class_weight", "none")) == "balanced" and not any(
        cause.cause == "metric_weight_mismatch" for cause in cleaned
    ):
        cleaned.append(
            Attribution(
                cause="metric_weight_mismatch",
                confidence=0.85,
                evidence="当前指标是 accuracy，class_weight=balanced 会为少数类牺牲整体准确率",
                fix={"set_model_params": {"class_weight": "none"}},
            )
        )
    return cleaned


def llm_attribute(
    profile: dict,
    plan: ExperimentPlan,
    result,
    task,
    timeout: int = 30,
) -> tuple[list[Attribution], str]:
    """让大模型生成归因结论与修复建议。

    返回 (归因列表, 一句话结论)。任何一项不合规都抛 `ValidationError`：
    原因不在枚举、没有可执行修复、或修复套回方案后过不了白名单；
    调用方据此整批回退到规则库，绝不把模型的非法动作送进执行器。
    """
    if not config.API_KEY:
        raise ValidationError("未配置 DEEPSEEK_API_KEY，无法使用大模型归因")

    from .prompts import attribution_system_prompt, attribution_user_prompt

    hints = [
        {"cause": cause.cause, "evidence": cause.evidence[:200]}
        for cause in attribute(profile, plan, result, task)[:3]
    ]
    whitelist = config.load_whitelist()
    data = chat_json(
        attribution_system_prompt(CAUSE_IDS, whitelist),
        attribution_user_prompt(profile, plan, result, task, hints),
        timeout=timeout,
        max_tokens=900,
    )

    raw_causes = data.get("causes")
    if not isinstance(raw_causes, list) or not raw_causes:
        raise ValidationError("模型没有给出任何归因")

    causes: list[Attribution] = []
    for raw in raw_causes[:3]:
        if not isinstance(raw, dict):
            raise ValidationError("归因条目必须是 JSON 对象")
        cause_id = str(raw.get("cause", "")).strip()
        if cause_id not in CAUSE_IDS:
            raise ValidationError(f"模型给出未知归因类型: {cause_id!r}")
        try:
            confidence = float(raw.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        evidence = str(raw.get("evidence", "") or "").strip()[:400]
        fix = _prune_model_params(_clean_fix(raw.get("fix")), plan, whitelist)
        causes.append(
            Attribution(
                cause=cause_id,
                confidence=max(0.0, min(1.0, confidence)),
                evidence=evidence or f"模型归因：{cause_id}",
                fix=fix,
            )
        )

    if not any(cause.fix for cause in causes):
        raise ValidationError("模型归因没有给出可执行修复")

    causes = _apply_metric_policy(causes, plan, task)
    if not causes:
        raise ValidationError("模型归因在指标口径收紧后没有可执行修复")

    # 多条 fix 合并后必须仍然合法：例如 A 条把模型换成随机森林、B 条却带线性模型的 l2，
    # 单独看都过校验，拼在一起就非法。逐条累加校验，冲突的那条丢弃；全丢则回退规则库。
    kept: list[Attribution] = []
    rejected: list[str] = []
    for cause in sorted(causes, key=lambda item: -item.confidence):
        candidate = replan(plan, kept + [cause], task)
        errors = validate_plan(candidate, task, whitelist)
        if errors:
            rejected.append(f"{cause.cause}: {errors[0]}")
            continue
        kept.append(cause)
    if not kept:
        raise ValidationError(f"模型修复建议合并后不合规: {'；'.join(rejected[:3])}")
    causes = kept

    summary = str(data.get("summary", "") or "").strip()[:500]
    return causes, summary
