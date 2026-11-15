"""服务层：FastAPI + SSE（W3 的 trace 面板后端）。

设计：主流程完全在 run_agent 里，服务层只做"触发 + 读 trace + 推事件"，
因此没有 Web 依赖时 CLI 也能跑（import 本模块不会失败，只有 create_app 会报缺依赖）。
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from .. import config
from ..store import db
from ..tracing import read_trace

try:  # Web 依赖可选：未安装时本模块仍可导入，只有 create_app 会给出明确提示
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - 缺 Web 依赖时 create_app 会先报错
    BaseModel = object  # type: ignore[assignment,misc]

# 注意：`Request` 必须导入到**模块级**。本文件启用了 from __future__ import annotations，
# endpoint 的注解是字符串，FastAPI 只在 endpoint.__globals__（即本模块命名空间）里解析它；
# 写成 create_app() 内部的局部导入，`Request` 会被当成普通参数 → 422。
try:
    from starlette.requests import Request
except ImportError:  # pragma: no cover
    Request = object  # type: ignore[assignment,misc]

MAX_LIMIT = 200
#: 批量删除一次最多接受的 run 数量（防止一个请求误删整个历史库）。
MAX_BATCH = 100

#: run_id 只允许字母/数字/下划线/点/连字符（首字符必须为字母或数字），
#: 用于 DELETE 时拼路径，防止 ../ 之类的路径穿越。
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _run_artifacts(run_id: str) -> tuple[Path, Path, Path]:
    """返回一次运行对应的（轨迹目录, 报告目录, 复现包 zip）路径。"""
    return (
        config.TRACE_DIR / run_id,
        config.REPORT_DIR / run_id,
        config.REPORT_DIR / f"{run_id}.zip",
    )


def _remove_run_data(run_id: str) -> dict:
    """删除一次运行的 DB 记录与轨迹/报告/复现包文件，返回删除明细。

    调用方负责校验 run_id 格式与运行状态（运行中不允许删）；配方库不动，
    原因见 `db.delete_run` 的注释。
    """
    trace_dir, report_dir, bundle_zip = _run_artifacts(run_id)
    had_trace = trace_dir.is_dir()
    had_report = report_dir.is_dir()
    had_bundle = bundle_zip.is_file()
    counts = db.delete_run(run_id)
    if had_trace:
        shutil.rmtree(trace_dir, ignore_errors=True)
    if had_report:
        shutil.rmtree(report_dir, ignore_errors=True)
    if had_bundle:
        bundle_zip.unlink(missing_ok=True)
    return {
        "db_runs": counts["runs"],
        "attributions": counts["attributions"],
        "trace": had_trace,
        "report": had_report,
        "bundle": had_bundle,
    }

#: SSE 参数：轮询间隔、心跳间隔、连接最长空闲时间（秒）
SSE_POLL_SECONDS = 0.2
SSE_HEARTBEAT_SECONDS = 15.0
SSE_MAX_IDLE_SECONDS = 600.0


def _tail_events(path, offset: int) -> tuple[list[dict], int]:
    """从 offset 起读取 trace 新增的**完整行**，返回 (事件列表, 新 offset)。

    - 增量读：不再每次把整个 trace 文件从头解析一遍，事件攒到几百条也不会变慢；
    - 只消费到最后一个换行符，写端正在写半行时不会解析出坏 JSON；
    - 坏行跳过，和 `read_trace` 的容错口径一致。
    """
    try:
        with open(path, "rb") as fh:
            fh.seek(offset)
            chunk = fh.read()
    except OSError:
        return [], offset
    if not chunk:
        return [], offset
    cut = chunk.rfind(b"\n")
    if cut < 0:
        return [], offset  # 只有半行，等写端补齐
    usable = chunk[: cut + 1]
    new_offset = offset + len(usable)
    events: list[dict] = []
    for raw in usable.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            events.append(json.loads(raw.decode("utf-8", "replace")))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return events, new_offset


# 注意：请求模型必须定义在模块级。本模块启用了 from __future__ import annotations，
# endpoint 注解是字符串，FastAPI 只在 endpoint 的 __globals__（即本模块命名空间）里
# 解析它；定义在 create_app() 内部的局部类会解析失败，导致 /openapi.json 与 /docs
# 返回 500（PydanticUserError: `RunRequest` is not fully defined）。
class RunRequest(BaseModel):
    """POST /api/runs 的请求体。"""

    task_id: str
    use_llm: bool = False
    max_rounds: int | None = None
    recipe_lookup: bool = True


class BatchDeleteRequest(BaseModel):
    """DELETE /api/runs 的请求体（面板「批量删除」用）。

    同样必须定义在模块级：本模块启用了 postponed annotations，FastAPI 只在
    模块命名空间里解析请求模型的注解，局部类会让 /openapi.json 构建失败。
    """

    run_ids: list[str]


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
    except ImportError as exc:
        raise ImportError(
            "启动 trace 面板需要 Web 依赖：pip install -r requirements.txt（含 fastapi/uvicorn）"
        ) from exc

    from ..agent.graph import run_agent
    from ..schemas import TaskSpec

    app = FastAPI(title="AutoLab", version="0.1.0")
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    web_dir = Path(__file__).with_name("web")
    app.mount("/panel", StaticFiles(directory=web_dir), name="panel")

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(web_dir / "index.html")

    runs: dict[str, dict] = {}
    lock = threading.Lock()

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

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str) -> dict:
        """删除一条历史记录：runs 行 + 该次运行的全部归因 + 轨迹/报告/复现包文件。

        - 运行中的实验返回 409（等它结束后再删，避免把 trace 删到一半）；
        - 记录与文件都不存在时返回 404；
        - 配方库不动：已沉淀的成功配方属于跨任务经验，不随历史记录一起删除。
        """
        if not _RUN_ID_RE.match(run_id):
            raise HTTPException(status_code=400, detail="非法的 run_id")
        with lock:
            summary = runs.get(run_id)
            if summary is not None and summary.get("status") == "running":
                raise HTTPException(status_code=409, detail="实验正在运行，结束后才能删除")
            deleted = _remove_run_data(run_id)
            runs.pop(run_id, None)
        if not deleted["db_runs"] and not (
            deleted["trace"] or deleted["report"] or deleted["bundle"]
        ):
            raise HTTPException(status_code=404, detail="运行不存在")
        return {"ok": True, "run_id": run_id, "deleted": deleted}

    @app.delete("/api/runs")
    def delete_runs(req: BatchDeleteRequest) -> dict:
        """批量删除历史记录（面板「全选 / 批量删除」）。

        语义（前端按这个接）：
        - 请求体 `{"run_ids": ["...", ...]}`，自动去重，最多 `MAX_BATCH` 条，空列表 400；
        - 逐条尽量删：非法的 id、运行中的、不存在的记入 `failed`，其余照常删除，
          整体仍返回 200（部分成功），前端据此展示"删了几条、哪几条没删"；
        - 删除内容与单条删除一致：runs 行 + 归因 + 轨迹/报告/复现包，配方库保留；
        - 响应 `deleted` 是明细数组，`counts` 汇总 requested/deleted/failed。
        """
        if not req.run_ids:
            raise HTTPException(status_code=400, detail="run_ids 不能为空")
        unique: list[str] = []
        for run_id in req.run_ids:
            if run_id not in unique:
                unique.append(run_id)
        if len(unique) > MAX_BATCH:
            raise HTTPException(status_code=400, detail=f"一次最多删除 {MAX_BATCH} 条记录")
        deleted: list[dict] = []
        failed: list[dict] = []
        for run_id in unique:
            if not _RUN_ID_RE.match(run_id):
                failed.append({"run_id": run_id, "reason": "非法的 run_id"})
                continue
            with lock:
                summary = runs.get(run_id)
                running = summary is not None and summary.get("status") == "running"
            if running:
                failed.append({"run_id": run_id, "reason": "实验正在运行，结束后才能删除"})
                continue
            info = _remove_run_data(run_id)
            with lock:
                runs.pop(run_id, None)
            if not info["db_runs"] and not (info["trace"] or info["report"] or info["bundle"]):
                failed.append({"run_id": run_id, "reason": "运行不存在"})
                continue
            deleted.append({"run_id": run_id, **info})
        return {
            "ok": True,
            "deleted": deleted,
            "failed": failed,
            "counts": {
                "requested": len(unique),
                "deleted": len(deleted),
                "failed": len(failed),
            },
        }

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
    def stream_trace(run_id: str, request: Request, after: int = 0):
        """SSE：持续推送 trace 新事件，收到 done 事件后自动结束。

        稳定性约定（前端按这个接）：
        - 每帧格式 `id: <seq>\\ndata: <json>\\n\\n`；断线重连时浏览器会自动带
          `Last-Event-ID`，服务端据此续推，不会重复也不会漏；
        - 空闲时每 15 秒发一个 `: ping` 注释帧，防止中间代理/网关掐断；
        - 响应头带 no-cache / X-Accel-Buffering: no，避免被缓冲成"卡住不动"；
        - trace 文件还没生成时返回 404，前端等 200ms 重试即可。
        """
        path = config.TRACE_DIR / run_id / "trace.jsonl"
        if not path.exists():
            raise HTTPException(status_code=404, detail="trace 尚未生成")

        last_event_id = (request.headers.get("last-event-id") or "").strip()
        try:
            resume_from = max(int(after), int(last_event_id))
        except ValueError:
            resume_from = int(after)

        def event_stream():
            offset = 0
            idle_since = time.monotonic()
            while True:
                events, offset = _tail_events(path, offset)
                fresh = 0
                for event in events:
                    seq = int(event.get("seq") or 0)
                    if seq > resume_from:
                        fresh += 1
                        yield f"id: {seq}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
                    if event.get("node") == "done":
                        return  # 跑完了（含"重连时 done 已 push 过"的情况）
                now = time.monotonic()
                if fresh:
                    idle_since = now
                elif now - idle_since >= SSE_HEARTBEAT_SECONDS:
                    yield ": ping\n\n"  # 心跳注释帧：EventSource 会忽略，代理不会掐线
                    idle_since = now
                if now - idle_since > SSE_MAX_IDLE_SECONDS:
                    return
                time.sleep(SSE_POLL_SECONDS)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app
