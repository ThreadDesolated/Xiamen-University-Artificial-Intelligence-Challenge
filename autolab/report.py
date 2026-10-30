"""报告与复现包：把一次 Agent 运行变成"可交付、可验收"的东西。

输出三件套：
1. reports/<run_id>/report.md  —— 人读的实验报告（结论、轮次、归因、复现方式）
2. reports/<run_id>/*.json     —— 机读的方案/结果/画像/环境快照
3. reports/<run_id>.zip        —— 一键复现包（含 trace、复现脚本）
"""
from __future__ import annotations

import json
import pathlib
import platform
import subprocess
import sys
import time
import zipfile

from . import config
from .schemas import dump_json


def _pkg_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "not-installed"


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(config.ROOT),
            capture_output=True,
            text=True,
            timeout=3,
        )
        return (out.stdout or "").strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def env_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": _pkg_version("numpy"),
        "pandas": _pkg_version("pandas"),
        "seed": config.SEED,
        "model": config.MODEL,
        "llm_enabled": bool(config.API_KEY),
        "git_commit": _git_commit(),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_report(summary: dict) -> str:
    """生成中文 Markdown 实验报告。"""
    task = summary.get("task", {}) or {}
    metric = summary.get("metric_name", "metric")
    gained = "达标" if summary.get("passed") else "未达标"
    lines: list[str] = []
    lines.append(f"# AutoLab 实验报告：{summary.get('task_name') or summary.get('task_id', '')}")
    lines.append("")
    lines.append(f"- 运行 ID：`{summary.get('run_id')}`")
    lines.append(f"- 任务类型：{summary.get('task_type')}")
    lines.append(
        f"- 结论：**{gained}**（{metric} = {_fmt(summary.get('final_metric'))}，"
        f"阈值 {summary.get('threshold')}，{'越高越好' if summary.get('higher_is_better') else '越低越好'}）"
    )
    lines.append(
        f"- 轮次：{summary.get('rounds_used')} / {summary.get('max_rounds')}，"
        f"规划来源：{summary.get('planner_source')}，配方命中：{summary.get('recipe_hit')}"
    )
    lines.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("## 一、迭代过程")
    lines.append("")
    lines.append("| 轮次 | 方案来源 | 指标（均值±标准差） | 是否达标 | 耗时 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for item in summary.get("rounds") or []:
        result = item.get("result") or {}
        metrics = result.get("metrics") or {}
        detail = metrics.get(metric) or {}
        mean, std = detail.get("mean"), detail.get("std")
        lines.append(
            "| {round} | {source} | {value} | {passed} | {seconds}s |".format(
                round=item.get("round"),
                source=(item.get("plan") or {}).get("source", "-"),
                value=f"{_fmt(mean)} ± {_fmt(std)}",
                passed="是" if item.get("passed") else "否",
                seconds=_fmt(result.get("seconds"), 2),
            )
        )
    lines.append("")

    lines.append("## 二、失败归因（如有）")
    lines.append("")
    attributions = summary.get("attributions") or []
    if attributions:
        lines.append("| 轮次 | 原因 | 置信度 | 证据 | 修复动作 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for item in attributions:
            fix = item.get("fix") or {}
            fix_brief = []
            if fix.get("add_ops"):
                fix_brief.append("add=" + ",".join(str(op.get("op")) for op in fix["add_ops"]))
            if fix.get("remove_ops"):
                fix_brief.append("remove=" + ",".join(fix["remove_ops"]))
            if fix.get("set_model"):
                fix_brief.append("set_model=" + str((fix["set_model"] or {}).get("kind")))
            if fix.get("set_model_params"):
                fix_brief.append("params=" + ",".join(fix["set_model_params"].keys()))
            lines.append(
                "| {round} | {cause} | {conf} | {evidence} | {fix} |".format(
                    round=item.get("round_idx"),
                    cause=item.get("cause"),
                    conf=_fmt(item.get("confidence"), 2),
                    evidence=str(item.get("evidence", "")).replace("|", "/")[:80],
                    fix="；".join(fix_brief) or "-",
                )
            )
    else:
        lines.append("（首轮或本轮即达标，无失败归因记录）")
    lines.append("")

    lines.append("## 三、最终实验方案")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary.get("plan") or {}, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")

    lines.append("## 四、数据画像摘要")
    lines.append("")
    profile = summary.get("profile") or {}
    lines.append(
        f"- 样本数 {profile.get('n_rows')}，特征数 {profile.get('n_features')}，"
        f"缺失比例 {_fmt(profile.get('missing_ratio'))}"
    )
    lines.append(f"- 特征类型：{json.dumps(profile.get('feature_types'), ensure_ascii=False)}")
    lines.append(f"- 目标分布：{json.dumps(profile.get('target'), ensure_ascii=False)}")
    lines.append(f"- 画像指纹：{summary.get('profile_hash')}")
    lines.append("")

    lines.append("## 五、复现方式")
    lines.append("")
    lines.append("```bash")
    lines.append("# 方式一：按保存的方案原样复跑（推荐）")
    lines.append(f"python -m autolab run --task {summary.get('task_id')} --plan plan.json")
    lines.append("# 方式二：回放本轮的完整调用链")
    lines.append(f"python -m autolab replay --run-id {summary.get('run_id')}")
    lines.append("```")
    lines.append("")
    lines.append("## 六、环境信息")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(summary.get("env") or {}, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def export_bundle(summary: dict) -> dict:
    """把报告、方案、结果、画像、环境、trace 打包成可复现 zip。"""
    run_id = str(summary.get("run_id") or time.strftime("run_%Y%m%d_%H%M%S"))
    run_dir = config.REPORT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    report_text = markdown_report(summary)
    (run_dir / "report.md").write_text(report_text, encoding="utf-8")

    dump_json(run_dir / "plan.json", summary.get("plan") or {})
    dump_json(run_dir / "result.json", (summary.get("rounds") or [{}])[-1].get("result") or {})
    dump_json(run_dir / "profile.json", summary.get("profile") or {})
    dump_json(run_dir / "env.json", summary.get("env") or {})
    dump_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "task_id": summary.get("task_id"),
            "status": summary.get("status"),
            "passed": summary.get("passed"),
            "metric_name": summary.get("metric_name"),
            "final_metric": summary.get("final_metric"),
            "rounds_used": summary.get("rounds_used"),
            "profile_hash": summary.get("profile_hash"),
            "created_at": time.time(),
        },
    )

    trace_src = pathlib.Path(str(summary.get("trace_path") or ""))
    if trace_src.exists():
        (run_dir / "trace.jsonl").write_text(trace_src.read_text(encoding="utf-8"), encoding="utf-8")

    (run_dir / "reproduce.ps1").write_text(
        "# AutoLab 一键复现（Windows PowerShell）\n"
        '$env:PYTHONUTF8="1"\n'
        f"python -m autolab run --task {summary.get('task_id')} --plan reports/{run_id}/plan.json\n"
        f"python -m autolab replay --run-id {run_id}\n",
        encoding="utf-8",
    )
    (run_dir / "reproduce.sh").write_text(
        "#!/usr/bin/env bash\nset -e\n"
        f"python -m autolab run --task {summary.get('task_id')} --plan reports/{run_id}/plan.json\n"
        f"python -m autolab replay --run-id {run_id}\n",
        encoding="utf-8",
    )

    bundle = config.REPORT_DIR / f"{run_id}.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(run_dir.iterdir()):
            if path.is_file():
                zf.write(path, arcname=f"{run_id}/{path.name}")

    return {
        "report_md": str(run_dir / "report.md"),
        "report_dir": str(run_dir),
        "bundle_path": str(bundle),
        "env": summary.get("env") or {},
    }
