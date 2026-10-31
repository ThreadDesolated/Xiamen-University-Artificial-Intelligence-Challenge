"""服务层（W3 trace 面板后端）回归测试。

历史 bug：RunRequest 曾定义在 create_app() 内部，而本模块启用了
from __future__ import annotations，endpoint 注解是字符串；FastAPI 只在模块命名空间
（endpoint.__globals__）里解析它，局部类解析失败，导致 GET /openapi.json 与 GET /docs
返回 500（PydanticUserError: `RunRequest` is not fully defined）。
"""
from __future__ import annotations

import unittest

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


if __name__ == "__main__":
    unittest.main()
