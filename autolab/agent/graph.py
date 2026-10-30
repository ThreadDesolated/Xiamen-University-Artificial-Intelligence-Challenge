"""编排主链路：AutoLab 的 Agent 闭环。

一次 run 的步骤（每一步都写 trace，可回放、可剖析）：
    load_data → profile → plan（配方/LLM/模板）→ validate
    → execute（沙箱子进程）→ evaluate
        ├─ 达标 → save_recipe → export（报告 + 复现包）→ done
        └─ 未达标 → attribute（归因）→ replan（重规划）→ 回到 execute（最多 max_rounds 轮）
"""
from __future__ import annotations

import time

from .. import config
from ..datasets.loader import load_dataframe
from ..profile import profile_dataframe, profile_hash
from ..report import env_info, export_bundle
from ..sandbox.executor import execute_plan
from ..sandbox.validate import validate_or_fallback
from ..schemas import ExperimentPlan, ExperimentResult, TaskSpec, new_id
from ..store import db
from ..tracing import TraceWriter
from .attributor import attribute, evaluate_target, replan
from .planner import llm_plan, plan_from_recipe, template_plan

DEFAULT_MAX_ROUNDS = 3


def _as_task(task) -> TaskSpec:
    if isinstance(task, TaskSpec):
        return task.validate()
    if isinstance(task, dict):
        return TaskSpec.from_dict(task).validate()
    raise TypeError(f"task 必须是 TaskSpec 或 dict，收到 {type(task)!r}")


def _better(score, best, higher_is_better: bool):
    if best is None:
        return True
    return score >= best if higher_is_better else score <= best


def run_agent(
    task,
    *,
    use_llm: bool = True,
    max_rounds: int | None = None,
    recipe_lookup: bool = True,
    run_id: str | None = None,
    fixed_plan=None,
) -> dict:
    """跑一次完整闭环，返回 summary dict（同时落库、落 trace、导出报告）。"""
    config.ensure_dirs()
    db.init_db()
    task = _as_task(task)
    max_rounds = int(max_rounds or task.max_rounds or DEFAULT_MAX_ROUNDS)
    run_id = run_id or new_id("run")
    trace = TraceWriter(config.TRACE_DIR / run_id / "trace.jsonl", run_id)
    started = time.time()

    summary: dict = {
        "run_id": run_id,
        "task_id": task.task_id,
        "task_name": task.name,
        "task_type": task.task_type,
        "metric_name": task.metric,
        "threshold": task.threshold,
        "higher_is_better": task.higher_is_better,
        "max_rounds": max_rounds,
        "started_at": started,
        "plan": {},
        "profile": {},
        "profile_hash": "",
        "rounds": [],
        "attributions": [],
        "rounds_used": 0,
        "final_metric": None,
        "best_metric": None,
        "passed": False,
        "status": "error",
        "error": "",
        "planner_source": "",
        "recipe_hit": False,
        "trace_path": str(trace.path),
    }

    try:
        # 1) 数据加载 ------------------------------------------------------
        df = load_dataframe(task.data_path, task.target)
        trace.emit(
            "load_data",
            tool="datasets.load_dataframe",
            input={"data_path": task.data_path, "target": task.target},
            output={"n_rows": int(len(df)), "n_cols": int(df.shape[1])},
        )

        # 2) 数据画像（决定后续规划与归因的上下文） -------------------------
        profile = profile_dataframe(df, task.target, task.task_type)
        phash = profile_hash(profile)
        summary["profile"] = profile
        summary["profile_hash"] = phash
        trace.emit(
            "profile",
            tool="profile.profile_dataframe",
            output={
                "n_rows": profile.get("n_rows"),
                "n_cols": profile.get("n_cols"),
                "n_features": profile.get("n_features"),
                "missing_ratio": profile.get("missing_ratio"),
                "target": profile.get("target"),
                "warnings": profile.get("warnings"),
            },
        )

        whitelist = config.load_whitelist()
        plan: ExperimentPlan | None = None
        rounds_allowed = max_rounds

        # 3) 首轮方案：固定方案（复现）→ 配方 → LLM → 模板 -----------------
        if fixed_plan is not None:
            plan = fixed_plan if isinstance(fixed_plan, ExperimentPlan) else ExperimentPlan.from_dict(dict(fixed_plan))
            plan.source = "reproduce"
            rounds_allowed = 1
            summary["planner_source"] = "reproduce"
            trace.emit("plan", tool="planner.reproduce", input={"source": "saved_plan"}, output=plan.to_dict())
        else:
            if recipe_lookup:
                recipe = db.find_recipe(task.task_type, phash)
                if recipe:
                    plan = plan_from_recipe(recipe, task)
                    summary["recipe_hit"] = True
                    summary["planner_source"] = "recipe"
                    trace.emit(
                        "recipe_hit",
                        tool="store.find_recipe",
                        input={"task_type": task.task_type, "profile_hash": phash},
                        output={
                            "recipe_id": recipe["recipe_id"],
                            "metric_name": recipe["metric_name"],
                            "metric_value": recipe["metric_value"],
                        },
                    )
            if plan is None and use_llm:
                try:
                    plan = llm_plan(profile, task, whitelist)
                    summary["planner_source"] = "llm"
                except Exception as exc:  # noqa: BLE001 - 规划失败必须能降级
                    trace.emit(
                        "plan_fallback",
                        tool="planner.llm_plan",
                        output={"error": f"{type(exc).__name__}: {exc}", "fallback": "template"},
                    )
            if plan is None:
                plan = template_plan(profile, task)
                summary["planner_source"] = summary["planner_source"] or "template"
            trace.emit(
                "plan",
                tool=f"planner.{plan.source}",
                input={"task": task.task_id, "profile_hash": phash},
                output=plan.to_dict(),
            )

        # 4) 执行 → 评估 → 归因 → 重规划 循环 ------------------------------
        best_metric = None
        best_plan = None
        final_plan = plan
        for round_idx in range(1, rounds_allowed + 1):
            final_plan, info = validate_or_fallback(final_plan, task, profile, whitelist)
            if not info.get("valid", True):
                trace.emit(
                    "plan_invalid",
                    tool="sandbox.validate_plan",
                    input={"round": round_idx},
                    output=info,
                )

            exec_started = time.perf_counter()
            raw = execute_plan(final_plan, task, run_id, round_idx)
            duration_ms = (time.perf_counter() - exec_started) * 1000.0
            result = ExperimentResult.from_dict(raw)
            score = result.metric_mean(task.metric)
            passed = bool(result.ok) and evaluate_target(score, task.threshold, task.higher_is_better)
            trace.emit(
                "execute",
                tool="sandbox.execute_plan",
                input={
                    "round": round_idx,
                    "plan_id": final_plan.plan_id,
                    "source": final_plan.source,
                    "model": final_plan.model,
                },
                output={
                    "status": result.status,
                    "metric": task.metric,
                    "mean": score,
                    "train_metric": result.train_metric,
                    "seconds": result.seconds,
                    "passed": passed,
                    "error": result.error,
                },
                duration_ms=duration_ms,
            )
            summary["rounds"].append(
                {
                    "round": round_idx,
                    "plan": final_plan.to_dict(),
                    "result": result.to_dict(),
                    "score": score,
                    "passed": passed,
                }
            )
            summary["rounds_used"] = round_idx
            if score is not None and _better(score, best_metric, task.higher_is_better):
                best_metric = score
                best_plan = final_plan

            if passed:
                break

            # 未达标：归因 → 入库 → 重规划
            causes = attribute(profile, final_plan, result, task)
            for cause in causes:
                cause.round_idx = round_idx
                db.save_attribution(run_id, round_idx, cause)
            summary["attributions"].extend(cause.to_dict() for cause in causes)
            trace.emit(
                "attribute",
                tool="attributor.attribute",
                input={"round": round_idx, "status": result.status, "score": score},
                output={
                    "causes": [
                        {
                            "cause": cause.cause,
                            "confidence": cause.confidence,
                            "evidence": cause.evidence,
                            "fix": cause.fix,
                        }
                        for cause in causes
                    ]
                },
            )

            if round_idx >= rounds_allowed:
                break

            final_plan = replan(final_plan, causes, task)
            trace.emit(
                "replan",
                tool="attributor.replan",
                input={"round": round_idx, "causes": [c.cause for c in causes]},
                output=final_plan.to_dict(),
            )

        # 5) 汇总、落库、配方沉淀 -----------------------------------------
        summary["plan"] = (final_plan if best_plan is None else final_plan).to_dict()
        summary["final_metric"] = summary["rounds"][-1]["score"] if summary["rounds"] else None
        summary["best_metric"] = best_metric
        summary["passed"] = bool(summary["rounds"][-1]["passed"]) if summary["rounds"] else False
        summary["status"] = "passed" if summary["passed"] else "failed"

        if summary["passed"]:
            db.save_recipe(
                task.task_type,
                phash,
                summary["plan"],
                task.metric,
                summary["final_metric"] or 0.0,
                notes=f"{task.task_id} / {summary['planner_source']}",
            )
        db.save_run(
            {
                "run_id": run_id,
                "task_id": task.task_id,
                "task_type": task.task_type,
                "started_at": started,
                "status": summary["status"],
                "rounds": summary["rounds_used"],
                "best_metric": summary["final_metric"],
                "plan": summary["plan"],
                "trace_path": summary["trace_path"],
            }
        )

        # 6) 导出报告与可复现包 ------------------------------------------
        summary["env"] = env_info()
        report_files = export_bundle(summary)
        summary.update(report_files)
        trace.emit(
            "export",
            tool="report.export_bundle",
            output={"report_md": report_files.get("report_md"), "bundle_path": report_files.get("bundle_path")},
        )
        return summary
    except Exception as exc:  # noqa: BLE001 - 单次 run 失败不能拖垮 CLI/服务
        summary["status"] = "error"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        trace.emit("error", output={"error": summary["error"]})
        try:
            db.save_run(
                {
                    "run_id": run_id,
                    "task_id": task.task_id,
                    "task_type": task.task_type,
                    "started_at": started,
                    "status": "error",
                    "rounds": summary["rounds_used"],
                    "best_metric": summary["final_metric"],
                    "plan": summary["plan"],
                    "trace_path": summary["trace_path"],
                }
            )
        except Exception:  # noqa: BLE001
            pass
        return summary
    finally:
        trace.emit(
            "done",
            output={
                "status": summary["status"],
                "passed": summary["passed"],
                "rounds": summary["rounds_used"],
                "error": summary["error"],
            },
        )
        trace.close()
