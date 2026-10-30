"""JSONL trace 写入器。

设计要点（也是"调用链路剖析"的底座）：
- 每步一条 JSON 行：谁（node）、什么工具（tool）、入参/出参、耗时；
- 边写边 flush，面板可以实时 tail；
- 支持订阅回调，SSE 直接复用，不需要另建消息队列。
"""
from __future__ import annotations

import json
import pathlib
import threading
import time
from typing import Any, Callable

from ..schemas import TraceEvent


def _safe(value: Any, limit: int = 4000) -> Any:
    """保证可 JSON 化；超长内容截断，避免 trace 文件爆掉。"""
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)[:limit]
    if len(text) <= limit:
        return value
    return {"_truncated": True, "_preview": text[:limit]}


class TraceWriter:
    """一个 run 一个 writer，对应 traces/<run_id>/trace.jsonl。"""

    def __init__(self, path, run_id: str):
        self.path = pathlib.Path(path)
        self.run_id = run_id
        self.seq = 0
        self._lock = threading.Lock()
        self._subscribers: list[Callable[[dict], None]] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8")

    # -- 写入 ---------------------------------------------------------------
    def emit(
        self,
        node: str,
        tool: str = "",
        *,
        input: Any = None,
        output: Any = None,
        duration_ms: float | None = None,
    ) -> dict:
        with self._lock:
            self.seq += 1
            event = TraceEvent(
                run_id=self.run_id,
                seq=self.seq,
                ts=round(time.time(), 4),
                node=node,
                tool=tool,
                input=_safe(input),
                output=_safe(output),
                duration_ms=None if duration_ms is None else round(float(duration_ms), 2),
            ).to_dict()
            self._fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._fh.flush()
        for callback in list(self._subscribers):
            try:
                callback(event)
            except Exception:  # noqa: BLE001 - 面板坏了不能影响主流程
                pass
        return event

    def subscribe(self, callback: Callable[[dict], None]) -> None:
        self._subscribers.append(callback)

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def read_trace(path) -> list[dict]:
    """读取 trace JSONL；坏行跳过（现场文件被截断时也能回放）。"""
    p = pathlib.Path(path)
    if p.is_dir():
        p = p / "trace.jsonl"
    if not p.exists():
        return []
    events: list[dict] = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events
