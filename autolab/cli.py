"""AutoLab 命令行入口。

命令一览：
  python -m autolab demo                      # 离线演示（无需 API Key）
  python -m autolab run --task t1_demo        # 跑任务；加 --llm 启用大模型规划
  python -m autolab run --task t1_demo --plan reports/<run>/plan.json   # 按方案原样复现
  python -m autolab run --plan latest                     # 用最近一次实验保存的方案复现
  python -m autolab prepare --task t1_adult   # 下载并准备真实数据集
  python -m autolab replay --run-id <run_id>  # 回放一次运行的调用链
  python -m autolab predict --model reports/<run_id>/model.joblib --csv data.csv  # 用导出的最终模型预测
  python -m autolab model-info --model reports/<run_id>/model.joblib  # 查看模型全貌（超参/特征/重要性）
  python -m autolab model-info --run-id <run_id>                      # 同上，直接按 run_id 查（推荐）
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
    holdout = summary.get("holdout") or {}
    if holdout.get("value") is not None:
        print(f"  留出集复验: {holdout.get('metric')}={holdout.get('value'):.4f}"
              f"（{holdout.get('n_samples')} 行，不参与达标判定）")
    print(f"  配方命中: {summary.get('recipe_hit')}   trace: {summary.get('trace_path')}")
    if summary.get("report_md"):
        print(f"  报告: {summary.get('report_md')}")
    if summary.get("bundle_path"):
        print(f"  复现包: {summary.get('bundle_path')}")
    if summary.get("model_path"):
        print(f"  最终模型: {summary.get('model_path')}")
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


def _latest_plan() -> tuple[str | None, pathlib.Path | None]:
    """找最近一次实验保存的方案，返回 (task_id, plan 路径)；没有则 (None, None)。

    用于 `run --plan latest`：仓库里不一定带 reports/acceptance_1（那是交付包证据），
    复现验收脚本可以用最近一次真实运行的方案复现，而不是直接报错。
    """
    reports = config.REPORT_DIR
    if not reports.is_dir():
        return None, None
    candidates = sorted(
        reports.glob("*/plan.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not candidates:
        return None, None
    plan_path = candidates[0]
    task_id: str | None = None
    manifest = plan_path.parent / "manifest.json"
    if manifest.is_file():
        try:
            task_id = json.loads(manifest.read_text(encoding="utf-8")).get("task_id")
        except (OSError, json.JSONDecodeError):
            task_id = None
    return task_id, plan_path


def _cmd_run(args) -> int:
    from .agent.graph import run_agent

    fixed_plan = None
    plan_task_id: str | None = None
    if args.plan:
        spec = str(args.plan)
        if spec.lower() in ("latest", "auto"):
            plan_task_id, path = _latest_plan()
            if path is None:
                print("没有找到任何已保存的实验方案，无法自动复现。", file=sys.stderr)
                print("可以先运行内置演示: python -m autolab demo", file=sys.stderr)
                return 2
        else:
            path = pathlib.Path(spec)
            if not path.is_absolute():
                path = config.ROOT / path
        if not path.is_file():
            print(f"找不到实验方案文件: {path}", file=sys.stderr)
            print(
                "提示：reports/acceptance_1/plan.json 这类验收证据只在交付包里；"
                "在仓库里可用 python -m autolab run --plan latest 取最近一次实验方案。",
                file=sys.stderr,
            )
            return 2
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            fixed_plan = ExperimentPlan.from_dict(payload)
        except (OSError, ValueError) as exc:
            print(f"实验方案文件无法解析: {path}（{exc}）", file=sys.stderr)
            return 2
        print(f"按已保存方案复现: {path}")

    if args.task:
        task = _load_task(args.task)
    elif plan_task_id:
        task = _load_task(plan_task_id)
    else:
        print("需要 --task 或 --plan latest 参数", file=sys.stderr)
        return 2
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


def _cmd_predict(args) -> int:
    """用导出的最终模型对新数据做预测（不依赖运行历史，可离线使用）。"""
    import pandas as pd

    from .evaluation.model_store import load_model, predict

    model_path = pathlib.Path(args.model)
    if not model_path.is_file():
        print(f"模型文件不存在: {model_path}", file=sys.stderr)
        return 2
    try:
        payload = load_model(model_path)
    except Exception as exc:  # noqa: BLE001 - 统一转成可读错误
        print(f"无法加载模型（不是 AutoLab 导出的文件？）: {exc}", file=sys.stderr)
        return 2

    meta = payload.get("meta") or {}
    csv_path = pathlib.Path(args.csv)
    if not csv_path.is_file():
        print(f"数据文件不存在: {csv_path}", file=sys.stderr)
        return 2
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        print(f"无法读取数据文件: {exc}", file=sys.stderr)
        return 2

    target = args.target or meta.get("target")
    if args.target and target not in df.columns:
        print(f"数据里没有目标列 {target!r}（可用 --target 指定其他列名）", file=sys.stderr)
        return 2
    features = df.drop(columns=[target]) if target and target in df.columns else df

    try:
        predictions = predict(payload, features)
    except Exception as exc:  # noqa: BLE001
        print(f"预测失败: {exc}", file=sys.stderr)
        return 2

    probabilities = None
    if args.proba:
        classes = meta.get("classes")
        if not classes:
            print("该模型是回归模型，忽略 --proba", file=sys.stderr)
        else:
            try:
                import numpy as np

                proba = payload["model"].predict_proba(payload["pipeline"].transform(features))
                probabilities = [float(row[-1]) for row in np.asarray(proba)]
            except Exception as exc:  # noqa: BLE001 - 概率不是必需能力
                print(f"该模型不支持 predict_proba，忽略 --proba: {exc}", file=sys.stderr)

    print("=" * 68)
    print(f"AutoLab predict: {model_path}")
    print(f"  任务: {meta.get('task_id')}  模型: {meta.get('model_kind')}  "
          f"CV: {meta.get('cv_mean')}")
    print(f"  数据: {csv_path}  行数: {len(df)}  特征列: {len(features.columns)}")
    print("=" * 68)
    print(f"  预测前 10 行: {predictions[:10]}")
    if probabilities is not None:
        print(f"  正类概率前 10 行: {[round(p, 4) for p in probabilities[:10]]}")

    if args.out:
        out_df = pd.DataFrame({"row": range(1, len(predictions) + 1), "prediction": predictions})
        if probabilities is not None:
            out_df["probability"] = probabilities
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  已写出: {out_path}")
    else:
        classes = meta.get("classes")
        if classes:
            from collections import Counter

            counts = Counter(str(v) for v in predictions)
            print(f"  预测分布: {dict(sorted(counts.items()))}")
        else:
            import numpy as np

            values = np.asarray(predictions, dtype=float)
            print(f"  预测统计: min={values.min():.4f}  mean={values.mean():.4f}  max={values.max():.4f}")
    return 0


def _brief(values, limit: int = 6) -> str:
    """把一组数值压成一行摘要。"""
    items = list(values)
    shown = ", ".join(f"{float(v):.4g}" for v in items[:limit])
    if len(items) > limit:
        shown += f", …（共 {len(items)}）"
    return shown


def _cmd_model_info(args) -> int:
    """打印导出的模型文件全貌：元信息、实验方案、预处理参数、特征名单、训练结果。

    模型文件可用 `--model` 直接指定，也可以用 `--run-id` 自动定位
    `reports/<run_id>/model.joblib`。
    """
    import re

    import numpy as np

    from .evaluation.model_store import load_model

    model_arg = getattr(args, "model", None)
    run_id = getattr(args, "run_id", None)
    if model_arg:
        model_path = pathlib.Path(model_arg)
    elif run_id:
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$", str(run_id)):
            print(f"非法的 run_id: {run_id!r}", file=sys.stderr)
            return 2
        model_path = config.REPORT_DIR / str(run_id) / "model.joblib"
    else:
        print("需要 --model 或 --run-id 之一", file=sys.stderr)
        return 2
    if not model_path.is_file():
        if run_id and not model_arg:
            print(f"该运行没有导出模型: {model_path}", file=sys.stderr)
        else:
            print(f"模型文件不存在: {model_path}", file=sys.stderr)
        return 2
    try:
        payload = load_model(model_path)
    except Exception as exc:  # noqa: BLE001 - 统一转成可读错误
        print(f"无法加载模型（不是 AutoLab 导出的文件？）: {exc}", file=sys.stderr)
        return 2

    meta = payload.get("meta") or {}
    plan = meta.get("plan") or {}
    pipeline = payload.get("pipeline")
    model_obj = payload.get("model")
    cols = [str(c) for c in (getattr(pipeline, "_fitted_cols", None) or [])]
    state = dict(getattr(pipeline, "state", None) or {})

    # ---- 预处理管线（按 plan.ops 顺序，附拟合参数）----
    steps = []
    for idx, op_spec in enumerate(plan.get("ops") or []):
        op = str(op_spec.get("op"))
        params = op_spec.get("params") or {}
        value = state.get(f"{idx}:{op}")
        detail = {"index": idx, "op": op, "params": params}
        if op == "onehot" and isinstance(value, dict):
            detail["columns"] = {
                str(col): {"n": len(cats), "categories": [str(c) for c in cats]}
                for col, cats in value.items()
            }
        elif op in ("impute_mean", "impute_median") and value is not None:
            detail["values"] = {str(k): float(v) for k, v in dict(value).items()}
        elif op == "scale_standard" and isinstance(value, tuple) and len(value) == 2:
            means, stds = value
            detail["mean"] = {str(k): float(v) for k, v in dict(means).items()}
            detail["std"] = {str(k): float(v) for k, v in dict(stds).items()}
        elif op == "variance_filter" and value:
            detail["kept_columns"] = [str(c) for c in value]
        elif value is not None:
            detail["fitted"] = type(value).__name__
        steps.append(detail)

    # ---- 训练结果摘要 ----
    kind = meta.get("model_kind") or (plan.get("model") or {}).get("kind") or type(model_obj).__name__
    model_params = meta.get("model_params") or (plan.get("model") or {}).get("params") or {}
    trained: dict = {"kind": kind}
    if kind == "random_forest":
        inner = getattr(model_obj, "model_", None)
        raw_trees = getattr(inner, "estimators_", None)
        trees = list(raw_trees) if raw_trees is not None else []
        node_counts = [int(t.tree_.node_count) for t in trees]
        depths = [int(t.tree_.max_depth) for t in trees]
        raw_importances = getattr(inner, "feature_importances_", None)
        importances = list(np.asarray(raw_importances, dtype=float)) if raw_importances is not None else []
        pairs = []
        for i, value in enumerate(importances):
            name = cols[i] if i < len(cols) else f"feature_{i}"
            pairs.append([name, float(value)])
        pairs.sort(key=lambda item: -item[1])
        trained.update(
            {
                "inner": type(inner).__name__,
                "n_estimators": len(trees),
                "n_nodes": sum(node_counts),
                "avg_nodes": round(sum(node_counts) / len(node_counts), 1) if node_counts else None,
                "max_depth": max(depths) if depths else None,
                "n_features": int(getattr(inner, "n_features_in_", len(importances)) or len(importances)),
                "n_classes": int(getattr(inner, "n_classes_", 0) or 0),
                "top_importances": pairs[: max(1, args.top)],
            }
        )
    elif kind == "logistic_gd":
        weights = np.asarray(getattr(model_obj, "W_", np.empty((0, 0))), dtype=float)
        real = weights[:-1] if weights.ndim == 2 and weights.shape[0] else weights
        if real.ndim == 2:
            strength = np.abs(real).sum(axis=1)
        else:
            strength = np.abs(real)
        pairs = sorted(
            ([[cols[i] if i < len(cols) else f"feature_{i}", float(v)] for i, v in enumerate(strength)]),
            key=lambda item: -item[1],
        )
        trained.update(
            {
                "n_weights": int(strength.size),
                "n_classes": int(weights.shape[1]) if weights.ndim == 2 else None,
                "top_weights": pairs[: max(1, args.top)],
            }
        )
    elif kind == "linear_gd":
        weights = np.asarray(getattr(model_obj, "w_", np.empty(0)), dtype=float)
        real = weights[:-1] if weights.size else weights
        pairs = sorted(
            ([[cols[i] if i < len(cols) else f"feature_{i}", float(v)] for i, v in enumerate(np.abs(real))]),
            key=lambda item: -item[1],
        )
        trained.update({"n_weights": int(real.size), "top_weights": pairs[: max(1, args.top)]})
    elif kind == "nearest_centroid":
        centroids = np.asarray(getattr(model_obj, "centroids_", np.empty((0, 0))), dtype=float)
        trained.update(
            {
                "n_centroids": int(centroids.shape[0]) if centroids.ndim == 2 else 0,
                "n_features": int(centroids.shape[1]) if centroids.ndim == 2 else 0,
                "classes": [str(c) for c in np.asarray(getattr(model_obj, "classes_", []))],
            }
        )
    else:
        trained["attributes"] = sorted(vars(model_obj).keys())

    summary = {
        "file": str(model_path),
        "size_bytes": model_path.stat().st_size,
        "meta": meta,
        "plan": plan,
        "pipeline": {"steps": steps, "n_features": len(cols), "features": cols},
        "model": {"kind": kind, "params": model_params, "trained": trained},
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return 0

    print("=" * 72)
    print(f"AutoLab model-info: {model_path}")
    print("=" * 72)
    print(f"  文件大小: {model_path.stat().st_size / 1024:.1f} KB"
          f"  格式: {meta.get('format')}  序列化: {meta.get('serializer')}")
    created_at = meta.get("created_at")
    if created_at:
        try:
            from datetime import datetime

            print(f"  导出时间: {datetime.fromtimestamp(float(created_at)).strftime('%Y-%m-%d %H:%M:%S')}")
        except (TypeError, ValueError, OSError):
            pass
    print(f"  任务: {meta.get('task_id')}（{meta.get('task_type')}）"
          f"  目标列: {meta.get('target')}  指标: {meta.get('metric')}")
    print(f"  模型: {kind}  超参数: {json.dumps(model_params, ensure_ascii=False)}")
    print(f"  CV 均值: {meta.get('cv_mean')}  类别: {meta.get('classes')}"
          f"  sklearn: {meta.get('sklearn_version')}")

    print("-" * 72)
    print("实验方案")
    print(json.dumps(plan, ensure_ascii=False, indent=2))

    print("-" * 72)
    print(f"预处理管线（特征列 {len(cols)} 个）")
    for step in steps:
        op = step["op"]
        params = json.dumps(step["params"], ensure_ascii=False) if step["params"] else ""
        print(f"  {step['index'] + 1}) {op} {params}".rstrip())
        if "columns" in step:
            for col, info in step["columns"].items():
                preview = ", ".join(info["categories"][:6])
                if info["n"] > 6:
                    preview += f", …（共 {info['n']} 类）"
                print(f"       {col}: {preview}")
        elif "values" in step:
            shown = 0
            for col, value in step["values"].items():
                if not args.features and shown >= args.top:
                    print(f"       …（共 {len(step['values'])} 列，--features 显示全部）")
                    break
                print(f"       {col}: {value:.4g}")
                shown += 1
        elif "mean" in step:
            shown = 0
            for col, value in step["mean"].items():
                if not args.features and shown >= args.top:
                    print(f"       …（共 {len(step['mean'])} 列，--features 显示全部）")
                    break
                std = step["std"].get(col)
                suffix = f"（std={std:.4g}）" if std is not None else ""
                print(f"       {col}: mean={value:.4g}{suffix}")
                shown += 1
        elif "kept_columns" in step:
            print(f"       保留 {len(step['kept_columns'])} 列: {', '.join(step['kept_columns'][:8])}")
    if cols:
        if args.features:
            print("  特征名单:")
            for name in cols:
                print(f"       {name}")
        else:
            print(f"  特征名单（前 10 个，--features 查看全部）: {', '.join(cols[:10])} …")

    print("-" * 72)
    print("训练结果")
    if kind == "random_forest":
        print(f"  内部实现: sklearn {trained['inner']}"
              f"  实际树数: {trained['n_estimators']}  总节点数: {trained['n_nodes']}"
              f"  平均节点/树: {trained['avg_nodes']}  最大树深: {trained['max_depth']}")
        print(f"  特征数: {trained['n_features']}  类别数: {trained['n_classes']}")
        print(f"  特征重要性 Top {len(trained['top_importances'])}:")
        for name, value in trained["top_importances"]:
            print(f"       {name}: {value:.4f}")
    elif kind in ("logistic_gd", "linear_gd"):
        label = "权重" if kind == "logistic_gd" else "回归系数"
        print(f"  {label} 维度: {trained['n_weights']}")
        print(f"  |{label}| Top {len(trained['top_weights'])}:")
        for name, value in trained["top_weights"]:
            print(f"       {name}: {value:.4f}")
    elif kind == "nearest_centroid":
        print(f"  质心数: {trained['n_centroids']}  特征数: {trained['n_features']}"
              f"  类别: {trained['classes']}")
    else:
        print(f"  属性: {', '.join(trained['attributes'])}")
    print("")
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
    run.add_argument("--task", default=None,
                     help="任务 id（configs/tasks/*.json）或配置文件路径；--plan latest 时可省略")
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

    predict_p = sub.add_parser("predict", help="用导出的最终模型对新数据预测")
    predict_p.add_argument("--model", required=True, help="模型文件（reports/<run_id>/model.joblib）")
    predict_p.add_argument("--csv", required=True, help="待预测数据 CSV（特征列与训练数据一致）")
    predict_p.add_argument("--target", default=None,
                           help="目标列名（默认取模型元信息里的 target；存在则从特征中剔除）")
    predict_p.add_argument("--out", default=None, help="预测结果输出 CSV 路径")
    predict_p.add_argument("--proba", action="store_true", help="分类任务同时输出正类概率")
    predict_p.set_defaults(func=_cmd_predict)

    model_info = sub.add_parser("model-info", help="查看导出模型：超参数、预处理参数、特征名单、重要性/权重")
    model_info.add_argument("--model", default=None, help="模型文件路径（reports/<run_id>/model.joblib 或面板下载的）")
    model_info.add_argument("--run-id", default=None, help="按运行 id 查模型（自动定位 reports/<run_id>/model.joblib）")
    model_info.add_argument("--top", type=int, default=15, help="重要性/权重显示前 N 项（默认 15）")
    model_info.add_argument("--features", action="store_true", help="打印全部特征名")
    model_info.add_argument("--json", action="store_true", help="以 JSON 输出完整信息（便于脚本处理）")
    model_info.set_defaults(func=_cmd_model_info)

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
