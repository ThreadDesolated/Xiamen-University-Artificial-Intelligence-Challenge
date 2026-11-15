"""SQLite 持久层：runs（运行记录）、recipes（配方库）、attributions（归因库）。

选 SQLite 的理由：零部署、单文件、可随仓库/离线包分发，现场演示不会因为
"数据库服务没起"翻车。三条表的定位分别对应 Agent 闭环里的：
"跑过什么"、"什么方案有效"、"为什么失败"。
"""
from __future__ import annotations

import json
import pathlib
import sqlite3
import time
from contextlib import contextmanager

from .. import config
from ..schemas import ExperimentPlan, new_id

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    task_id     TEXT NOT NULL,
    task_type   TEXT NOT NULL,
    started_at  REAL NOT NULL,
    status      TEXT NOT NULL,
    rounds      INTEGER NOT NULL DEFAULT 0,
    best_metric REAL,
    plan_json   TEXT,
    trace_path  TEXT
);

CREATE TABLE IF NOT EXISTS recipes (
    recipe_id    TEXT PRIMARY KEY,
    task_type    TEXT NOT NULL,
    profile_hash TEXT NOT NULL,
    plan_json    TEXT NOT NULL,
    metric_name  TEXT NOT NULL,
    metric_value REAL NOT NULL,
    created_at   REAL NOT NULL,
    notes        TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS attributions (
    attribution_id TEXT PRIMARY KEY,
    run_id         TEXT NOT NULL,
    round_idx      INTEGER NOT NULL,
    cause          TEXT NOT NULL,
    confidence     REAL NOT NULL,
    evidence       TEXT DEFAULT '',
    fix_json       TEXT,
    created_at     REAL NOT NULL
);
"""


def connect(db_path=None) -> sqlite3.Connection:
    """每次调用开新连接（脚本/服务/测试都安全），自动建目录。"""
    path = pathlib.Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def open_conn(db_path=None):
    """用完必关的连接上下文。

    ⚠️ 不能用 `with sqlite3.connect(...) as conn`：Connection 的上下文管理器**只管
    事务提交/回滚，不会关闭连接**，脚本/服务反复调用会漏文件句柄，最后报
    `ResourceWarning: unclosed database` 或 `database is locked`。
    """
    conn = connect(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path=None) -> pathlib.Path:
    path = pathlib.Path(db_path or config.DB_PATH)
    with open_conn(path) as conn:
        conn.executescript(_SCHEMA)
    return path


def save_run(run: dict, db_path=None) -> str:
    """保存一次运行记录。run 需含 run_id/task_id/task_type/status/rounds/best_metric 等。"""
    init_db(db_path)
    run_id = str(run.get("run_id") or new_id("run"))
    with open_conn(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs"
            "(run_id, task_id, task_type, started_at, status, rounds, best_metric, plan_json, trace_path)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                str(run.get("task_id", "")),
                str(run.get("task_type", "")),
                float(run.get("started_at") or time.time()),
                str(run.get("status", "unknown")),
                int(run.get("rounds") or 0),
                run.get("best_metric"),
                json.dumps(run.get("plan") or {}, ensure_ascii=False),
                str(run.get("trace_path", "")),
            ),
        )
    return run_id


def delete_run(run_id: str, db_path=None) -> dict:
    """删除一次运行记录及其失败归因（历史记录里的"删除"）。

    只删"记录"本身：runs 行 + 该次运行的 attributions 行；**配方库保留**——
    已经沉淀的成功方案属于跨任务经验，不应因为删除一条历史记录而悄悄消失。
    返回 {"runs": 删除的运行行数, "attributions": 删除的归因行数}。
    """
    init_db(db_path)
    with open_conn(db_path) as conn:
        attributions = conn.execute(
            "DELETE FROM attributions WHERE run_id=?", (str(run_id),)
        ).rowcount
        runs = conn.execute("DELETE FROM runs WHERE run_id=?", (str(run_id),)).rowcount
    return {"runs": int(runs), "attributions": int(attributions)}


def delete_runs(run_ids: list[str], db_path=None) -> dict:
    """批量删除运行记录及其失败归因（面板「批量删除」）。

    与 `delete_run` 语义一致：只删"记录"（runs 行 + 对应 attributions 行），
    配方库保留；用一条 SQL 的 `IN (...)` 处理，避免逐条开关连接。
    返回 {"runs": 删除的运行行数, "attributions": 删除的归因行数}。
    """
    ids = [str(run_id) for run_id in run_ids]
    if not ids:
        return {"runs": 0, "attributions": 0}
    init_db(db_path)
    placeholders = ",".join("?" * len(ids))
    with open_conn(db_path) as conn:
        attributions = conn.execute(
            f"DELETE FROM attributions WHERE run_id IN ({placeholders})", ids
        ).rowcount
        runs = conn.execute(
            f"DELETE FROM runs WHERE run_id IN ({placeholders})", ids
        ).rowcount
    return {"runs": int(runs), "attributions": int(attributions)}


def save_recipe(
    task_type: str,
    profile_hash: str,
    plan,
    metric_name: str,
    metric_value: float,
    notes: str = "",
    db_path=None,
) -> str:
    """把一个达标方案写进配方库（同画像同任务只保留指标最高的一条）。"""
    init_db(db_path)
    plan_dict = plan.to_dict() if isinstance(plan, ExperimentPlan) else dict(plan)
    recipe_id = new_id("recipe")
    with open_conn(db_path) as conn:
        old = conn.execute(
            "SELECT recipe_id, metric_value FROM recipes WHERE task_type=? AND profile_hash=?",
            (task_type, profile_hash),
        ).fetchone()
        if old is not None and float(old["metric_value"]) >= float(metric_value):
            return str(old["recipe_id"])
        if old is not None:
            conn.execute("DELETE FROM recipes WHERE recipe_id=?", (old["recipe_id"],))
        conn.execute(
            "INSERT INTO recipes"
            "(recipe_id, task_type, profile_hash, plan_json, metric_name, metric_value, created_at, notes)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                recipe_id,
                task_type,
                profile_hash,
                json.dumps(plan_dict, ensure_ascii=False),
                metric_name,
                float(metric_value),
                time.time(),
                notes,
            ),
        )
    return recipe_id


def find_recipe(task_type: str, profile_hash: str, db_path=None) -> dict | None:
    """按 (任务类型, 画像指纹) 检索历史成功配方。"""
    path = pathlib.Path(db_path or config.DB_PATH)
    if not path.exists():
        return None
    with open_conn(db_path) as conn:
        row = conn.execute(
            "SELECT recipe_id, plan_json, metric_name, metric_value, created_at"
            " FROM recipes WHERE task_type=? AND profile_hash=? ORDER BY metric_value DESC LIMIT 1",
            (task_type, profile_hash),
        ).fetchone()
    if row is None:
        return None
    return {
        "recipe_id": row["recipe_id"],
        "plan": json.loads(row["plan_json"]),
        "metric_name": row["metric_name"],
        "metric_value": row["metric_value"],
        "created_at": row["created_at"],
    }


def save_attribution(run_id: str, round_idx: int, attribution, db_path=None) -> str:
    """记录一条失败归因（形成可复用的"失败→原因→修复"知识）。"""
    init_db(db_path)
    data = attribution.to_dict() if hasattr(attribution, "to_dict") else dict(attribution)
    attribution_id = new_id("attr")
    with open_conn(db_path) as conn:
        conn.execute(
            "INSERT INTO attributions"
            "(attribution_id, run_id, round_idx, cause, confidence, evidence, fix_json, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                attribution_id,
                run_id,
                int(round_idx),
                str(data.get("cause", "")),
                float(data.get("confidence") or 0.0),
                str(data.get("evidence", "")),
                json.dumps(data.get("fix") or {}, ensure_ascii=False),
                time.time(),
            ),
        )
    return attribution_id


def list_runs(limit: int = 20, db_path=None) -> list[dict]:
    path = pathlib.Path(db_path or config.DB_PATH)
    if not path.exists():
        return []
    with open_conn(db_path) as conn:
        rows = conn.execute(
            "SELECT run_id, task_id, task_type, status, rounds, best_metric, started_at"
            " FROM runs ORDER BY started_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
    return [dict(row) for row in rows]


def stats(db_path=None) -> dict:
    """配方库/归因库规模（演示"经验在增长"时直接展示）。"""
    path = pathlib.Path(db_path or config.DB_PATH)
    if not path.exists():
        return {"runs": 0, "recipes": 0, "attributions": 0}
    with open_conn(db_path) as conn:
        counts = {}
        for table in ("runs", "recipes", "attributions"):
            counts[table] = int(conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"])
    return counts
