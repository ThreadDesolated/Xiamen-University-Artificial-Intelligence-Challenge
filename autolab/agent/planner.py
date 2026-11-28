"""实验方案规划器。

两个入口，互为兜底：
- `template_plan`：规则规划器，**永远可用、零依赖、离线**（而且刻意偏弱，给归因留出空间）；
- `llm_plan`：调用 OpenAI 兼容网关（默认 DeepSeek），产物必须过白名单校验，失败自动回退模板。
"""
from __future__ import annotations

import json

from .. import config
from ..llm import chat_json
from ..schemas import ExperimentPlan, ValidationError


def template_plan(profile: dict, task) -> ExperimentPlan:
    """规则规划器：从画像直接产出一份保守方案（W1 基线）。"""
    ops: list[dict] = []
    missing_cols = profile.get("columns_with_missing") or {}
    if missing_cols or float(profile.get("missing_ratio", 0.0)) > 0.02:
        # 故意先用零填充：比中位数/均值更粗，给归因阶段留出"改进空间"
        ops.append({"op": "impute_zero", "params": {}})

    if profile.get("feature_types", {}).get("categorical"):
        ops.append({"op": "onehot", "params": {"max_categories": 20}})

    if task.task_type == "regression":
        model = {"kind": "linear_gd", "params": {"lr": 0.05, "epochs": 300, "l2": 0.001}}
    else:
        # 弱 baseline：最近质心，且不做标准化
        model = {"kind": "nearest_centroid", "params": {}}

    return ExperimentPlan(
        ops=ops,
        model=model,
        eval={"n_splits": int(task.n_splits), "seed": int(task.seed)},
        source="template",
        notes="规则基线：缺失值零填充 + 最近质心（未标准化），预期首轮不达标",
    )


def plan_from_recipe(recipe: dict, task, source: str = "recipe") -> ExperimentPlan:
    """把配方库里检索到的历史方案转换成当前任务的方案。

    `source="recipe"` 表示同画像精确命中，`source="transfer"` 表示跨任务迁移。
    """
    plan = ExperimentPlan.from_dict(recipe.get("plan", {}))
    plan.plan_id = ""
    plan.__post_init__()
    plan.eval = {"n_splits": int(task.n_splits), "seed": int(task.seed)}
    plan.source = source
    return plan


#: 算子/模型的中文说明，供面板与 trace 展示"为什么这么规划"
_OP_RATIONALE = {
    "impute_zero": ("缺失值", "零填充", "先做最粗的处理，给归因阶段留出改进空间"),
    "impute_mean": ("缺失值", "均值填充", "用训练集均值补齐数值列，保留全部样本"),
    "impute_median": ("缺失值", "中位数填充", "比均值更抗异常值，分类与回归都适用"),
    "drop_missing_rows": ("缺失值", "删除缺失行", "缺失集中在少数行时最直接"),
    "onehot": ("类别特征", "独热编码", "类别列转数值，限制稀有类别避免维度爆炸"),
    "variance_filter": ("特征筛选", "低方差过滤", "去掉常量列与近常量列，减少噪声"),
    "scale_standard": ("特征缩放", "标准化", "把量纲拉齐，距离类与梯度类模型对尺度敏感"),
    "log1p": ("特征变换", "log1p", "压缩长尾分布，降低极端值影响"),
}

_MODEL_RATIONALE = {
    "nearest_centroid": ("模型", "最近质心", "弱 baseline：先暴露问题，再靠归因升级"),
    "logistic_gd": ("模型", "逻辑回归", "线性模型，训练快、可解释，支持类别权重"),
    "linear_gd": ("模型", "线性回归", "回归基线，训练快，便于观察欠拟合/过拟合"),
    "random_forest": ("模型", "随机森林", "树模型容量更高，适合非线性与特征交互"),
}


def plan_rationale(plan: ExperimentPlan, profile: dict, task) -> list[dict]:
    """把方案翻译成"每一步做什么、为什么"的结构化说明（面板方案卡使用）。"""
    items: list[dict] = []
    missing_ratio = float(profile.get("missing_ratio") or 0.0)
    n_categorical = len((profile.get("feature_types") or {}).get("categorical") or [])
    for op in plan.ops:
        name = str(op.get("op", ""))
        stage, action, reason = _OP_RATIONALE.get(name, ("预处理", name, "白名单算子"))
        if name.startswith("impute"):
            reason = f"{reason}；本次画像缺失比例 {missing_ratio:.3f}"
        elif name == "onehot" and n_categorical:
            reason = f"{reason}；本次有 {n_categorical} 个类别特征"
        items.append({"stage": stage, "action": action, "op": name, "reason": reason})

    model = plan.model or {}
    kind = str(model.get("kind", ""))
    stage, action, reason = _MODEL_RATIONALE.get(kind, ("模型", kind, "白名单模型"))
    params = model.get("params") or {}
    if params:
        reason = f"{reason}；参数 {json.dumps(params, ensure_ascii=False)}"
    items.append({"stage": stage, "action": action, "op": kind, "reason": reason})

    n_splits = int((plan.eval or {}).get("n_splits", task.n_splits))
    seed = int((plan.eval or {}).get("seed", task.seed))
    items.append(
        {
            "stage": "评估",
            "action": f"{n_splits} 折交叉验证",
            "op": "cv",
            "reason": f"固定种子 {seed}，用均值±标准差作为奖励信号",
        }
    )
    return items


def llm_plan(profile: dict, task, whitelist: dict | None = None, timeout: int = 30) -> ExperimentPlan:
    """调用大模型产出方案；未配置 Key、调用失败或输出不合规时抛异常，由调用方回退模板。"""
    if not config.API_KEY:
        raise ValidationError("未配置 DEEPSEEK_API_KEY，无法使用 LLM 规划器")

    from .prompts import planner_system_prompt, planner_user_prompt

    whitelist = whitelist or config.load_whitelist()
    raw = chat_json(
        planner_system_prompt(whitelist),
        planner_user_prompt(profile, task),
        timeout=timeout,
        max_tokens=800,
    )
    if not isinstance(raw.get("ops", []), list) or not isinstance(raw.get("model", {}), dict):
        raise ValidationError("模型方案结构不合规：ops 必须是数组、model 必须是对象")
    plan = ExperimentPlan.from_dict(
        {
            "ops": raw.get("ops", []),
            "model": raw.get("model", {}),
            "eval": {"n_splits": int(task.n_splits), "seed": int(task.seed)},
            "source": "llm",
            "notes": str(raw.get("notes", ""))[:500],
        }
    )
    return plan
