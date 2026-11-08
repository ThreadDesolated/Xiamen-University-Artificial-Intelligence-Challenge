"""沙箱执行器：子进程隔离 + 超时即杀 + 禁网 + 结果结构化。

为什么不直接在本进程跑？——现场演示最怕"模型幻觉方案把进程跑挂/联网卡死"，
隔离到子进程后，执行器永远能拿到一个结构化结果，归因与重规划都能继续。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from .. import config
from ..schemas import ExperimentPlan, TaskSpec


def _timeout_seconds(task) -> int:
    """子进程超时时间：默认 max(30, budget+20)，可用 AUTOLAB_SANDBOX_TIMEOUT 覆盖。

    覆盖开关的用途：慢机器/断网彩排时临时放宽；测试里把它调小学生到秒级，
    才能在 CI 里验证"超时必杀"这条兜底路径。
    """
    override = os.environ.get("AUTOLAB_SANDBOX_TIMEOUT", "").strip()
    if override:
        try:
            return max(1, int(float(override)))
        except ValueError:
            pass
    return max(30, int(task.budget_seconds) + 20)


def execute_plan(
    plan: ExperimentPlan,
    task: TaskSpec,
    run_id: str = "run",
    round_idx: int = 1,
) -> dict:
    """在沙箱子进程中执行一次实验，返回结果 dict（ExperimentResult 字段）。"""
    workdir = config.TRACE_DIR / run_id / "sandbox"
    workdir.mkdir(parents=True, exist_ok=True)
    spec_path = workdir / f"spec_r{round_idx}.json"
    out_path = workdir / f"result_r{round_idx}.json"
    if out_path.exists():
        out_path.unlink()

    spec_path.write_text(
        json.dumps({"task": task.to_dict(), "plan": plan.to_dict()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    env = dict(os.environ)
    env.update(
        {
            "AUTOLAB_NO_NET": "1",
            "AUTOLAB_IN_SANDBOX": "1",
            "PYTHONPATH": str(config.ROOT)
            + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    )
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env.pop(key, None)

    cmd = [sys.executable, "-m", "autolab.sandbox.runner", "--spec", str(spec_path), "--out", str(out_path)]
    timeout = _timeout_seconds(task)
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(config.ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        elapsed = round(time.perf_counter() - started, 3)
        if out_path.exists():
            result = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            result = {"status": "error", "error": "沙箱未产出结果文件", "metrics": {}, "seconds": elapsed}
        result.setdefault("stdout_tail", (proc.stdout or "")[-3000:])
        result["stderr_tail"] = (proc.stderr or "")[-3000:]
        if result.get("status") == "ok":
            result["seconds"] = float(result.get("seconds") or elapsed)
        if proc.returncode != 0 and result.get("status") == "ok":
            result["status"] = "error"
            result["error"] = result.get("error") or f"沙箱退出码 {proc.returncode}"
        return result
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "error": f"沙箱执行超时（>{timeout}s），已强制终止",
            "metrics": {},
            "seconds": round(time.perf_counter() - started, 3),
        }
