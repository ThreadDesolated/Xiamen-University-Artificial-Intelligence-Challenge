"""AutoLab 命令行入口。

命令一览：
  python -m autolab demo                      # 离线演示（无需 API Key）
  python -m autolab run --task t1_demo        # 跑任务；加 --llm 启用大模型规划
  python -m autolab run --task t1_demo --plan reports/<run>/plan.json   # 按方案原样复现
  python -m autolab prepare --task t1_adult   # 下载并准备真实数据集
  python -m autolab replay --run-id <run_id>  # 回放一次运行的调用链
  python -m autolab serve                     # 启动 trace 面板后端（需 Web 依赖）
  python -m autolab init-db                   # 初始化 SQLite
  python -m autolab list                      # 查看历史运行
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from . import config
from .schemas import ExperimentPlan, TaskSpec


def _load_task(task_id: str) -> TaskSpec:
    return TaskSpec.from_dict(config.load_task_config(task_id)).validate()


def _fmt_metric(summary: dict) -> str:
    value = summary.get("final_metric")
    metric = summary.get("metric_name", "metric")
    digits = 4
    try:
        return f"{metric}={float(value):.{digits}f}"
    except (TypeError, ValueError):
        return f"{metric}={value}"


def print_summary(summary: dict) -> None:
    """把 run summary 打印成人能直接读的演示输出。"""
    print("")
    print("=" * 72)
    print(f"AutoLab run: {summary.get('run_id')}  task={summary.get('task_id')}  "
          f"planner={summary.get('planner_source')}")
    print("=" * 72)
    if summary.get("error"):
        print(f"[x] 运行出错: {summary['error']}")
        print(f"    trace: {summary.get('trace_path')}")
        return
    for item in summary.get("rounds") or []:
        result = item.get("result") or {}
        metrics = result.get("metrics") or {}
        detail = metrics.get(summary.get("metric_name"), {})
        mean = detail.get("mean")
        source = (item.get("plan") or {}).get("source")
        flag = "OK " if item.get("passed") else ".. "
        value = "-" if mean is None else f"{float(mean):.4f}"
        note = ""
        if not item.get("passed"):
            causes = [a for a in summary.get("attributions", []) if a.get("round_idx") == item.get("round")]
            note = "  <- " + ",".join(str(c.get("cause")) for c in causes[:2])
        print(f"  {flag}[round {item.get('round')}] source={source:<10} "
              f"{summary.get('metric_name')}={value} ({result.get('seconds')}s){note}")
    verdict = "达标 ✅" if summary.get("passed") else "未达标 ❌"
    print("-" * 72)
    print(f"  结论: {verdict}  {_fmt_metric(summary)}  轮次: {summary.get('rounds_used')} / {summary.get('max_rounds')}")
    print(f"  配方命中: {summary.get('recipe_hit')}   trace: {summary.get('trace_path')}")
    if summary.get("report_md"):
        print(f"  报告: {summary.get('report_md')}")
    if summary.get("bundle_path"):
        print(f"  复现包: {summary.get('bundle_path')}")
    print("")


def _cmd_demo(args) -> int:
    from .agent.graph import run_agent

    task = _load_task("t1_demo")
    summary = run_agent(
        task,
        use_llm=False,
        recipe_lookup=not args.no_recipe,
        max_rounds=args.max_rounds,
        run_id=args.run_id,
    )
    print_summary(summary)
    if summary.get("passed") and summary.get("recipe_hit"):
        print("提示：本次命中了历史配方（首轮即达标）。想再看一遍「失败→归因→重规划」，"
              "加 --no-recipe 参数。")
    return 0 if summary.get("status") in ("passed", "failed") else 1


def _cmd_run(args) -> int:
    from .agent.graph import run_agent

    task = _load_task(args.task)
    fixed_plan = None
    if args.plan:
        path = pathlib.Path(args.plan)
        if not path.is_absolute():
            path = config.ROOT / path
        fixed_plan = ExperimentPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))
        print(f"按已保存方案复现: {path}")
    summary = run_agent(
        task,
        use_llm=bool(args.llm),
        recipe_lookup=not args.no_recipe and fixed_plan is None,
        max_rounds=args.max_rounds,
        run_id=args.run_id,
        fixed_plan=fixed_plan,
    )
    print_summary(summary)
    return 0 if summary.get("status") in ("passed", "failed") else 1


def _cmd_prepare(args) -> int:
    from .datasets.prepare import prepare_task

    path = prepare_task(args.task)
    print(f"数据集已就绪: {path}")
    return 0


def _cmd_replay(args) -> int:
    from .tracing import replay

    if args.run_id:
        trace_path = config.TRACE_DIR / args.run_id / "trace.jsonl"
    elif args.trace:
        trace_path = pathlib.Path(args.trace)
    else:
        print("需要 --run-id 或 --trace 参数", file=sys.stderr)
        return 2
    events = list(replay(trace_path, delay=args.delay))
    if not events:
        print(f"没有找到 trace 事件: {trace_path}", file=sys.stderr)
        return 1
    return 0


def _cmd_serve(args) -> int:
    try:
        import uvicorn
    except ImportError:
        print("需要安装 Web 依赖：pip install -r requirements.txt", file=sys.stderr)
        return 1
    from .serving.server import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0


def _cmd_init_db(_args) -> int:
    from .store import db

    path = db.init_db()
    print(f"SQLite 已初始化: {path}")
    return 0


def _cmd_list(args) -> int:
    from .store import db

    runs = db.list_runs(limit=args.limit)
    if not runs:
        print("暂无历史运行记录")
        return 0
    print(f"{'run_id':<28} {'task':<12} {'status':<8} {'rounds':<7} {'metric':<10} started")
    for run in runs:
        print(
            f"{run['run_id']:<28} {run['task_id']:<12} {run['status']:<8} "
            f"{run['rounds']:<7} {run['best_metric'] if run['best_metric'] is not None else '-':<10} "
            f"{run['started_at']}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autolab", description="AutoLab：科研实验自动化智能体")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="离线演示（synthetic 数据，无需 API Key）")
    demo.add_argument("--max-rounds", type=int, default=3)
    demo.add_argument("--no-recipe", action="store_true", help="不使用配方库（必看失败→归因→重规划）")
    demo.add_argument("--run-id", default=None)
    demo.set_defaults(func=_cmd_demo)

    run = sub.add_parser("run", help="按任务配置运行一次 Agent 闭环")
    run.add_argument("--task", required=True, help="任务 id（configs/tasks/*.json）或配置文件路径")
    run.add_argument("--max-rounds", type=int, default=None)
    run.add_argument("--llm", action="store_true", help="启用大模型规划（需 DEEPSEEK_API_KEY）")
    run.add_argument("--no-recipe", action="store_true", help="跳过配方库检索")
    run.add_argument("--plan", default=None, help="按保存的方案 JSON 原样复现")
    run.add_argument("--run-id", default=None)
    run.set_defaults(func=_cmd_run)

    prepare = sub.add_parser("prepare", help="下载并准备真实数据集")
    prepare.add_argument("--task", required=True, help="t1_adult / t2_digits / t3_timeseries")
    prepare.set_defaults(func=_cmd_prepare)

    replay_p = sub.add_parser("replay", help="回放一次运行的调用链")
    replay_p.add_argument("--run-id", default=None)
    replay_p.add_argument("--trace", default=None)
    replay_p.add_argument("--delay", type=float, default=0.0)
    replay_p.set_defaults(func=_cmd_replay)

    serve = sub.add_parser("serve", help="启动 trace 面板后端（FastAPI + SSE）")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_cmd_serve)

    init_db = sub.add_parser("init-db", help="初始化 SQLite")
    init_db.set_defaults(func=_cmd_init_db)

    list_runs = sub.add_parser("list", help="查看历史运行")
    list_runs.add_argument("--limit", type=int, default=20)
    list_runs.set_defaults(func=_cmd_list)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)
