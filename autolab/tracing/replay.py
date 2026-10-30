"""trace 回放：把一次 run 按时间线打印/逐步吐出（现场断网兜底用）。"""
from __future__ import annotations

import time
from typing import Iterator

from .logger import read_trace

_NODE_LABELS = {
    "load_data": "加载数据",
    "profile": "数据画像",
    "plan": "方案规划",
    "plan_invalid": "方案校验/回退",
    "execute": "沙箱执行",
    "attribute": "失败归因",
    "replan": "重规划",
    "export": "导出复现包",
    "recipe_hit": "配方命中",
    "plan_fallback": "规划兜底",
    "done": "完成",
}


def _line(event: dict) -> str:
    label = _NODE_LABELS.get(event.get("node", ""), event.get("node", ""))
    tool = event.get("tool") or "-"
    output = event.get("output")
    brief = ""
    if isinstance(output, dict):
        for key in ("status", "mean", "accuracy", "passed", "n_rows", "final_metric"):
            if key in output and not isinstance(output[key], (dict, list)):
                brief = f"{key}={output[key]}"
                break
        if not brief and "causes" in output:
            causes = output.get("causes") or []
            brief = "causes=" + ",".join(str(c.get("cause")) for c in causes[:3] if isinstance(c, dict))
    elif output is not None:
        brief = str(output)[:60]
    return f"[{event.get('seq', 0):02d}] {label:<10} {tool:<28} {brief}"


def replay(path, *, limit: int | None = None, delay: float = 0.0, printer=print) -> Iterator[dict]:
    """回放 trace：yield 每个事件，同时打印一行摘要。"""
    events = read_trace(path)
    if limit:
        events = events[: int(limit)]
    for event in events:
        printer(_line(event))
        if delay:
            time.sleep(float(delay))
        yield event
