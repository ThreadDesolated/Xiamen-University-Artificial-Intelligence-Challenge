"""实验方案规划器。

两个入口，互为兜底：
- `template_plan`：规则规划器，**永远可用、零依赖、离线**（而且刻意偏弱，给归因留出空间）；
- `llm_plan`：调用 OpenAI 兼容网关（默认 DeepSeek），产物必须过白名单校验，失败自动回退模板。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from .. import config
from ..schemas import ExperimentPlan, ValidationError


def template_plan(profile: dict, task) -> ExperimentPlan:
    """规则规划器：从画像直接产出一份保守方案（W1 基线）。"""
    ops: list[dict] = []
    missing_cols = profile.get("columns_with_missing") or {}
    if missing_cols or float(profile.get("missing_ratio", 0.0)) > 0.02:
        # 故意先用零填充：比中位数/均值更粗，给归因阶段留出"改进空间"
        ops.append({"op": "impute_zero", "params": {}})

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


def plan_from_recipe(recipe: dict, task) -> ExperimentPlan:
    """把配方库里检索到的历史方案转换成当前任务的方案。"""
    plan = ExperimentPlan.from_dict(recipe.get("plan", {}))
    plan.plan_id = ""
    plan.__post_init__()
    plan.eval = {"n_splits": int(task.n_splits), "seed": int(task.seed)}
    plan.source = "recipe"
    return plan


def _extract_json(text: str) -> dict:
    """从模型输出里截取第一个完整的 JSON 对象。"""
    start = text.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : idx + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValidationError("模型输出里没有可解析的 JSON")


def llm_plan(profile: dict, task, whitelist: dict | None = None, timeout: int = 30) -> ExperimentPlan:
    """调用大模型产出方案；未配置 Key 或调用失败时抛异常，由调用方回退模板。"""
    if not config.API_KEY:
        raise ValidationError("未配置 DEEPSEEK_API_KEY，无法使用 LLM 规划器")

    from .prompts import planner_system_prompt, planner_user_prompt

    whitelist = whitelist or config.load_whitelist()
    payload = {
        "model": config.MODEL,
        "temperature": 0.2,
        "max_tokens": 800,
        "messages": [
            {"role": "system", "content": planner_system_prompt(whitelist)},
            {"role": "user", "content": planner_user_prompt(profile, task)},
        ],
    }
    req = urllib.request.Request(
        f"{config.BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.API_KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ValidationError(f"LLM 规划调用失败: {exc}") from exc

    content = body["choices"][0]["message"]["content"]
    raw = _extract_json(content)
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
