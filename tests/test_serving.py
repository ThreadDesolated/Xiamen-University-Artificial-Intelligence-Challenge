"""服务层（W3 trace 面板后端）回归测试。

历史 bug：RunRequest 曾定义在 create_app() 内部，而本模块启用了
from __future__ import annotations，endpoint 注解是字符串；FastAPI 只在模块命名空间
（endpoint.__globals__）里解析它，局部类解析失败，导致 GET /openapi.json 与 GET /docs
返回 500（PydanticUserError: `RunRequest` is not fully defined）。
"""
from __future__ import annotations

import json
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

from autolab import config
from autolab.store import db

#: docs/接口契约.md 约定的 HTTP 接口
EXPECTED_PATHS = {
    "/api/health",
    "/api/tasks",
    "/api/stats",
    "/api/runs",
    "/api/runs/{run_id}",
    "/api/runs/{run_id}/trace",
    "/api/runs/{run_id}/report",
    "/api/runs/{run_id}/stream",
}


class _ServingTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            from autolab.serving.server import create_app
        except ImportError as exc:  # 没装 Web 依赖时跳过，不算失败
            raise unittest.SkipTest(str(exc))
        cls.app = create_app()


class OpenAPITest(_ServingTestBase):
    def test_openapi_json_generates(self) -> None:
        """这个调用就是原 bug 的触发点：生成 openapi.json 时字段 schema 构建失败即 500。"""
        spec = self.app.openapi()
        self.assertIn("RunRequest", spec["components"]["schemas"])

    def test_post_api_runs_binds_run_request(self) -> None:
        spec = self.app.openapi()
        body = spec["paths"]["/api/runs"]["post"]["requestBody"]
        content = body["content"]["application/json"]
        self.assertEqual(content["schema"]["$ref"], "#/components/schemas/RunRequest")

    def test_documented_paths_present(self) -> None:
        self.assertTrue(EXPECTED_PATHS <= set(self.app.openapi()["paths"]))


class RunRequestTest(_ServingTestBase):
    def test_defaults(self) -> None:
        from autolab.serving.server import RunRequest

        req = RunRequest(task_id="t1_demo")
        self.assertFalse(req.use_llm)
        self.assertIsNone(req.max_rounds)
        self.assertTrue(req.recipe_lookup)

    def test_task_id_required(self) -> None:
        from pydantic import ValidationError

        from autolab.serving.server import RunRequest

        with self.assertRaises(ValidationError):
            RunRequest()


class RequestValidationTest(_ServingTestBase):
    """有 httpx 时顺带验证真实请求路径（依赖注入阶段不再抛 500）。"""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        try:
            from fastapi.testclient import TestClient
        except ImportError as exc:  # 未安装 httpx：跳过请求级用例
            raise unittest.SkipTest(str(exc))
        cls.client = TestClient(cls.app)

    def test_missing_task_id_returns_422(self) -> None:
        resp = self.client.post("/api/runs", json={})
        self.assertEqual(resp.status_code, 422, resp.text[:500])

    def test_unknown_task_returns_404(self) -> None:
        resp = self.client.post("/api/runs", json={"task_id": "no_such_task"})
        self.assertEqual(resp.status_code, 404, resp.text[:500])


class StreamEndpointTest(_ServingTestBase):
    """SSE 回归：`request: Request` 若只写成 create_app() 里的局部导入，注解解析不到，
    FastAPI 会把 request 当成必填查询参数，前端一连就是 422。
    """

    run_id = "serving_test_stream"

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        try:
            from fastapi.testclient import TestClient
        except ImportError as exc:  # 未安装 httpx：跳过请求级用例
            raise unittest.SkipTest(str(exc))
        cls.client = TestClient(cls.app)
        cls.trace_dir = config.TRACE_DIR / cls.run_id
        cls.trace_dir.mkdir(parents=True, exist_ok=True)
        events = [
            {"run_id": cls.run_id, "seq": 1, "ts": 0.0, "node": "load_data", "tool": "datasets.load_dataframe",
             "input": None, "output": {}, "duration_ms": None},
            {"run_id": cls.run_id, "seq": 2, "ts": 0.1, "node": "done", "tool": "",
             "input": None, "output": {"status": "passed"}, "duration_ms": None},
        ]
        (cls.trace_dir / "trace.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n", encoding="utf-8"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.trace_dir, ignore_errors=True)

    def test_headers_and_frames(self) -> None:
        resp = self.client.get(f"/api/runs/{self.run_id}/stream")
        self.assertEqual(resp.status_code, 200, resp.text[:300])
        self.assertIn("text/event-stream", resp.headers["content-type"])
        self.assertEqual(resp.headers["cache-control"], "no-cache, no-transform")
        self.assertEqual(resp.headers["x-accel-buffering"], "no")
        self.assertIn("id: 1", resp.text)
        self.assertIn("data: ", resp.text)
        self.assertIn('"node": "done"', resp.text)

    def test_resume_skips_delivered_events(self) -> None:
        resp = self.client.get(f"/api/runs/{self.run_id}/stream?after=1")
        self.assertEqual(resp.status_code, 200, resp.text[:300])
        self.assertNotIn("id: 1\n", resp.text)
        self.assertIn("id: 2", resp.text)

    def test_missing_trace_returns_404(self) -> None:
        resp = self.client.get("/api/runs/__no_such_run__/stream")
        self.assertEqual(resp.status_code, 404)


class DeleteRunTest(unittest.TestCase):
    """DELETE /api/runs/{run_id}：记录、归因、轨迹/报告/复现包一并删除，配方保留。"""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from autolab.serving.server import create_app
        except ImportError as exc:
            raise unittest.SkipTest(str(exc))
        try:
            from fastapi.testclient import TestClient
        except ImportError as exc:
            raise unittest.SkipTest(str(exc))
        cls.client = TestClient(create_app())

    def setUp(self) -> None:
        self.run_id = "run_delete_unit"
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="autolab_del_"))
        self.trace_root = self.tmp / "traces"
        self.report_root = self.tmp / "reports"
        self.db_path = self.tmp / "autolab.sqlite3"
        self._patches = [
            mock.patch.object(config, "TRACE_DIR", self.trace_root),
            mock.patch.object(config, "REPORT_DIR", self.report_root),
            mock.patch.object(config, "DB_PATH", self.db_path),
        ]
        for patch in self._patches:
            patch.start()
        trace_dir = self.trace_root / self.run_id
        trace_dir.mkdir(parents=True)
        (trace_dir / "trace.jsonl").write_text(
            json.dumps({"run_id": self.run_id, "seq": 1, "node": "done", "output": {}}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        report_dir = self.report_root / self.run_id
        report_dir.mkdir(parents=True)
        (report_dir / "report.md").write_text("# 报告\n", encoding="utf-8")
        (self.report_root / f"{self.run_id}.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)
        db.save_run(
            {
                "run_id": self.run_id,
                "task_id": "t1_demo",
                "task_type": "binary_classification",
                "status": "passed",
                "rounds": 2,
                "best_metric": 0.873,
            }
        )
        db.save_attribution(
            self.run_id,
            1,
            {"cause": "class_imbalance", "confidence": 0.8, "evidence": "test", "fix": {}},
        )

    def tearDown(self) -> None:
        for patch in self._patches:
            patch.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_delete_removes_record_attributions_and_artifacts(self) -> None:
        resp = self.client.delete(f"/api/runs/{self.run_id}")
        self.assertEqual(resp.status_code, 200, resp.text[:300])
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual(
            body["deleted"],
            {"db_runs": 1, "attributions": 1, "trace": True, "report": True, "bundle": True},
        )
        self.assertFalse((self.trace_root / self.run_id).exists())
        self.assertFalse((self.report_root / self.run_id).exists())
        self.assertFalse((self.report_root / f"{self.run_id}.zip").exists())
        self.assertEqual(db.stats()["runs"], 0)
        self.assertEqual(db.stats()["attributions"], 0)
        self.assertEqual(self.client.get("/api/runs").json(), [])

    def test_delete_unknown_returns_404(self) -> None:
        resp = self.client.delete("/api/runs/run_not_exists")
        self.assertEqual(resp.status_code, 404, resp.text[:300])

    def test_delete_illegal_id_returns_400(self) -> None:
        resp = self.client.delete("/api/runs/bad%20id")
        self.assertEqual(resp.status_code, 400, resp.text[:300])

    def test_recipe_survives_delete(self) -> None:
        db.save_recipe(
            "binary_classification",
            "hash123",
            {"ops": [], "model": {"kind": "logistic_gd", "params": {}}, "eval": {"n_splits": 5, "seed": 42}},
            "accuracy",
            0.9,
        )
        resp = self.client.delete(f"/api/runs/{self.run_id}")
        self.assertEqual(resp.status_code, 200, resp.text[:300])
        self.assertEqual(db.stats()["recipes"], 1)
        self.assertIsNotNone(db.find_recipe("binary_classification", "hash123"))


class BatchDeleteRunTest(unittest.TestCase):
    """DELETE /api/runs：批量删除历史记录，逐条尽量删，部分失败返回 200 + failed 明细。"""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from autolab.serving.server import create_app
        except ImportError as exc:
            raise unittest.SkipTest(str(exc))
        try:
            from fastapi.testclient import TestClient
        except ImportError as exc:
            raise unittest.SkipTest(str(exc))
        cls.client = TestClient(create_app())

    def setUp(self) -> None:
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="autolab_batchdel_"))
        self.trace_root = self.tmp / "traces"
        self.report_root = self.tmp / "reports"
        self.db_path = self.tmp / "autolab.sqlite3"
        self._patches = [
            mock.patch.object(config, "TRACE_DIR", self.trace_root),
            mock.patch.object(config, "REPORT_DIR", self.report_root),
            mock.patch.object(config, "DB_PATH", self.db_path),
        ]
        for patch in self._patches:
            patch.start()

    def tearDown(self) -> None:
        for patch in self._patches:
            patch.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _seed(self, run_id: str) -> None:
        """造一条带轨迹/报告/复现包/归因的历史记录。"""
        trace_dir = self.trace_root / run_id
        trace_dir.mkdir(parents=True)
        (trace_dir / "trace.jsonl").write_text(
            json.dumps({"run_id": run_id, "seq": 1, "node": "done", "output": {}}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        report_dir = self.report_root / run_id
        report_dir.mkdir(parents=True)
        (report_dir / "report.md").write_text("# 报告\n", encoding="utf-8")
        (self.report_root / f"{run_id}.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)
        db.save_run(
            {
                "run_id": run_id,
                "task_id": "t1_demo",
                "task_type": "binary_classification",
                "status": "passed",
                "rounds": 2,
                "best_metric": 0.873,
            }
        )
        db.save_attribution(
            run_id, 1, {"cause": "class_imbalance", "confidence": 0.8, "evidence": "test", "fix": {}}
        )

    def _delete(self, run_ids):
        return self.client.request("DELETE", "/api/runs", json={"run_ids": run_ids})

    def test_batch_delete_multiple(self) -> None:
        self._seed("run_batch_a")
        self._seed("run_batch_b")
        resp = self._delete(["run_batch_a", "run_batch_b"])
        self.assertEqual(resp.status_code, 200, resp.text[:300])
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertEqual([d["run_id"] for d in body["deleted"]], ["run_batch_a", "run_batch_b"])
        self.assertEqual(body["failed"], [])
        self.assertEqual(body["counts"], {"requested": 2, "deleted": 2, "failed": 0})
        self.assertTrue(all(d["trace"] and d["report"] and d["bundle"] for d in body["deleted"]))
        for run_id in ("run_batch_a", "run_batch_b"):
            self.assertFalse((self.trace_root / run_id).exists())
            self.assertFalse((self.report_root / run_id).exists())
            self.assertFalse((self.report_root / f"{run_id}.zip").exists())
        stats = db.stats()
        self.assertEqual((stats["runs"], stats["attributions"]), (0, 0))
        self.assertEqual(self.client.get("/api/runs").json(), [])

    def test_batch_delete_partial_success(self) -> None:
        self._seed("run_batch_exist")
        resp = self._delete(["run_batch_exist", "run_batch_missing"])
        self.assertEqual(resp.status_code, 200, resp.text[:300])
        body = resp.json()
        self.assertEqual([d["run_id"] for d in body["deleted"]], ["run_batch_exist"])
        self.assertEqual(
            body["failed"], [{"run_id": "run_batch_missing", "reason": "运行不存在"}]
        )
        self.assertEqual(body["counts"], {"requested": 2, "deleted": 1, "failed": 1})
        self.assertFalse((self.trace_root / "run_batch_exist").exists())

    def test_batch_delete_illegal_id_marked_failed(self) -> None:
        self._seed("run_batch_ok")
        resp = self._delete(["bad id", "run_batch_ok"])
        self.assertEqual(resp.status_code, 200, resp.text[:300])
        body = resp.json()
        self.assertEqual(body["failed"], [{"run_id": "bad id", "reason": "非法的 run_id"}])
        self.assertEqual([d["run_id"] for d in body["deleted"]], ["run_batch_ok"])

    def test_batch_delete_dedupes_ids(self) -> None:
        self._seed("run_batch_dup")
        resp = self._delete(["run_batch_dup", "run_batch_dup"])
        self.assertEqual(resp.status_code, 200, resp.text[:300])
        body = resp.json()
        self.assertEqual(body["counts"], {"requested": 1, "deleted": 1, "failed": 0})
        self.assertEqual(len(body["deleted"]), 1)

    def test_batch_delete_empty_returns_400(self) -> None:
        resp = self._delete([])
        self.assertEqual(resp.status_code, 400, resp.text[:300])

    def test_batch_delete_too_many_returns_400(self) -> None:
        resp = self._delete([f"run_batch_{i}" for i in range(101)])
        self.assertEqual(resp.status_code, 400, resp.text[:300])

    def test_batch_recipe_survives_delete(self) -> None:
        self._seed("run_batch_recipe")
        db.save_recipe(
            "binary_classification",
            "hash123",
            {"ops": [], "model": {"kind": "logistic_gd", "params": {}}, "eval": {"n_splits": 5, "seed": 42}},
            "accuracy",
            0.9,
        )
        resp = self._delete(["run_batch_recipe"])
        self.assertEqual(resp.status_code, 200, resp.text[:300])
        self.assertEqual(db.stats()["recipes"], 1)
        self.assertIsNotNone(db.find_recipe("binary_classification", "hash123"))


if __name__ == "__main__":
    unittest.main()
