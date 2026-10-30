"""服务层：FastAPI + SSE（W3 的 trace 面板后端）。

设计：主流程完全在 run_agent 里，服务层只做"触发 + 读 trace + 推事件"，
因此没有 Web 依赖时 CLI 也能跑（import 本模块不会失败，只有 create_app 会报缺依赖）。
"""
from __future__ import annotations

import json
import threading
import time
import uuid

from .. import config
from ..store import db
from ..tracing import read_trace

MAX_LIMIT = 200


def list_tasks() -> list[dict]:
    """列出 configs/tasks 下的任务配置。"""
    tasks = []
    for path in sorted(config.TASK_DIR.glob("*.json")):
        try:
            tasks.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return tasks


def create_app():
    """创建 FastAPI 应用；未安装 fastapi 时给出明确提示。"""
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
        from pydantic import BaseModel
    except ImportError as exc:
        raise ImportError(
            "启动 trace 面板需要 Web 依赖：pip install -r requirements.txt（含 fastapi/uvicorn）"
        ) from exc

    from ..agent.graph import run_agent
    from ..schemas import TaskSpec

    app = FastAPI(title="AutoLab", version="0.1.0")
    runs: dict[str, dict] = {}
    lock = threading.Lock()

    class RunRequest(BaseModel):
        task_id: str
        use_llm: bool = False
        max_rounds: int | None = None
        recipe_lookup: bool = True

    def _run_async(run_id: str, task: TaskSpec, use_llm: bool, max_rounds, recipe_lookup: bool) -> None:
        summary = run_agent(
            task,
            use_llm=use_llm,
            max_rounds=max_rounds,
            recipe_lookup=recipe_lookup,
            run_id=run_id,
        )
        with lock:
            runs[run_id] = summary

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "time": time.time(), "llm_enabled": bool(config.API_KEY)}

    @app.get("/api/tasks")
    def tasks() -> list[dict]:
        return list_tasks()

    @app.get("/api/stats")
    def stats() -> dict:
        return db.stats()

    @app.get("/api/runs")
    def list_runs(limit: int = Query(default=20, ge=1, le=MAX_LIMIT)) -> list[dict]:
        return db.list_runs(limit=limit)

    @app.post("/api/runs")
    def start_run(req: RunRequest) -> dict:
        try:
            task = TaskSpec.from_dict(config.load_task_config(req.task_id)).validate()
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"任务不存在: {req.task_id}")
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        with lock:
            runs[run_id] = {"run_id": run_id, "status": "running", "task_id": req.task_id}
        thread = threading.Thread(
            target=_run_async,
            args=(run_id, task, req.use_llm, req.max_rounds, req.recipe_lookup),
            daemon=True,
        )
        thread.start()
        return {"run_id": run_id, "status": "running", "trace": f"/api/runs/{run_id}/trace"}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        with lock:
            summary = runs.get(run_id)
        if summary is not None:
            return summary
        raise HTTPException(status_code=404, detail="运行不存在或已过期（内存态）")

    @app.get("/api/runs/{run_id}/trace")
    def get_trace(run_id: str, after: int = 0) -> list[dict]:
        events = read_trace(config.TRACE_DIR / run_id / "trace.jsonl")
        return events[max(0, int(after)) :]

    @app.get("/api/runs/{run_id}/report")
    def get_report(run_id: str):
        report = config.REPORT_DIR / run_id / "report.md"
        if not report.exists():
            raise HTTPException(status_code=404, detail="报告尚未生成")
        return PlainTextResponse(report.read_text(encoding="utf-8"), media_type="text/markdown")

    @app.get("/api/runs/{run_id}/stream")
    def stream_trace(run_id: str):
        """SSE：持续推送 trace 新事件，收到 done 事件后自动结束。"""
        path = config.TRACE_DIR / run_id / "trace.jsonl"
        if not path.exists():
            raise HTTPException(status_code=404, detail="trace 尚未生成")

        def event_stream():
            cursor = 0
            idle = 0.0
            while True:
                events = read_trace(path)
                while cursor < len(events):
                    event = events[cursor]
                    cursor += 1
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    if event.get("node") == "done":
                        return
                    idle = 0.0
                idle += 0.25
                if idle > 600:
                    return
                time.sleep(0.25)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app
