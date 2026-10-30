"""Schema + 白名单双重校验：非法方案绝不进执行器。"""
from __future__ import annotations

from .. import config
from ..schemas import ExperimentPlan, ValidationError

_ALLOWED_EVAL_KEYS = {"n_splits", "seed"}


def _check_params(defn: dict, params: dict, where: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(params, dict):
        return [f"{where}: params 必须是对象"]
    for key, value in params.items():
        spec = defn.get(key)
        if spec is None:
            errors.append(f"{where}: 未声明的参数 {key!r}")
            continue
        kind = spec.get("type")
        try:
            if kind == "int":
                if not isinstance(value, int) or isinstance(value, bool):
                    errors.append(f"{where}.{key}: 需要 int，收到 {type(value).__name__}")
                elif not (spec.get("min", value) <= value <= spec.get("max", value)):
                    errors.append(f"{where}.{key}: 超出范围 [{spec.get('min')}, {spec.get('max')}]")
            elif kind == "float":
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    errors.append(f"{where}.{key}: 需要 float，收到 {type(value).__name__}")
                elif not (spec.get("min", value) <= float(value) <= spec.get("max", value)):
                    errors.append(f"{where}.{key}: 超出范围 [{spec.get('min')}, {spec.get('max')}]")
            elif kind == "enum":
                if value not in spec.get("values", []):
                    errors.append(f"{where}.{key}: 只允许 {spec.get('values')}，收到 {value!r}")
        except TypeError:
            errors.append(f"{where}.{key}: 取值类型非法（{value!r}）")
    return errors


def validate_plan(
    plan: ExperimentPlan,
    task,
    whitelist: dict | None = None,
) -> list[str]:
    """返回错误列表；空列表表示方案合法。"""
    wl = whitelist or config.load_whitelist()
    errors: list[str] = []
    limits = wl.get("limits", {})
    ops_def = wl.get("ops", {})
    models_def = wl.get("models", {})

    if not isinstance(plan, ExperimentPlan):
        return ["plan 必须是 ExperimentPlan"]

    if len(plan.ops) > int(limits.get("max_ops", 8)):
        errors.append(f"算子数量 {len(plan.ops)} 超过上限 {limits.get('max_ops')}")

    seen_ops: set[str] = set()
    for i, spec in enumerate(plan.ops):
        if not isinstance(spec, dict):
            errors.append(f"ops[{i}] 必须是对象")
            continue
        op = spec.get("op")
        if not isinstance(op, str) or op not in ops_def:
            errors.append(f"ops[{i}]: 算子不在白名单: {op!r}")
            continue
        if op in seen_ops:
            errors.append(f"ops[{i}]: 重复算子 {op!r}")
        seen_ops.add(op)
        errors.extend(_check_params(ops_def[op].get("params", {}), spec.get("params") or {}, f"ops[{i}]({op})"))

    model = plan.model or {}
    kind = model.get("kind")
    if not isinstance(kind, str) or kind not in models_def:
        errors.append(f"model.kind 不在白名单: {kind!r}")
    else:
        allowed_tasks = models_def[kind].get("tasks", [])
        if task.task_type not in allowed_tasks:
            errors.append(f"model {kind!r} 不支持任务类型 {task.task_type!r}（支持: {allowed_tasks}）")
        errors.extend(
            _check_params(models_def[kind].get("params", {}), model.get("params") or {}, f"model({kind})")
        )

    eval_spec = plan.eval or {}
    unknown = set(eval_spec) - _ALLOWED_EVAL_KEYS
    if unknown:
        errors.append(f"eval 不认识的键: {sorted(unknown)}")
    for key in ("n_splits", "seed"):
        if key in eval_spec and (not isinstance(eval_spec[key], int) or isinstance(eval_spec[key], bool)):
            errors.append(f"eval.{key} 必须是整数")

    try:
        size = len(plan.to_json())
    except Exception:  # noqa: BLE001
        size = 0
    if size > int(limits.get("max_plan_bytes", 20000)):
        errors.append(f"方案 JSON 过大（{size} 字节）")

    return errors


def validate_or_fallback(
    plan: ExperimentPlan,
    task,
    profile: dict,
    whitelist: dict | None = None,
) -> tuple[ExperimentPlan, dict]:
    """校验方案，非法则回退到模板方案（保证演示不中断）。"""
    errors = validate_plan(plan, task, whitelist)
    if not errors:
        return plan, {"valid": True, "errors": []}

    from ..agent.planner import template_plan

    fallback = template_plan(profile, task)
    fallback_errors = validate_plan(fallback, task, whitelist)
    if fallback_errors:
        raise ValidationError(
            f"方案非法且模板兜底方案也非法: plan_errors={errors}, fallback_errors={fallback_errors}"
        )
    return fallback, {"valid": False, "errors": errors, "fallback_plan_id": fallback.plan_id}
