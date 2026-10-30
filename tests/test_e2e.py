"""端到端测试：失败 → 归因 → 重规划 → 达标 → 配方沉淀 → 复现包。

测试使用 synthetic:t1（t1_demo 任务），完全不依赖网络与 API Key，CI/现场都能跑。
"""
from __future__ import annotations

import pathlib
import tempfile
import unittest

from autolab import config
from autolab.agent.graph import run_agent
from autolab.schemas import TaskSpec
from autolab.store import db
from autolab.tracing import read_trace


class E2ETest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # 所有产物写到临时目录，不污染仓库
        cls.tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="autolab_e2e_"))
        config.TRACE_DIR = cls.tmpdir / "traces"
        config.REPORT_DIR = cls.tmpdir / "reports"
        config.DB_PATH = cls.tmpdir / "autolab_test.sqlite3"
        cls.task = TaskSpec.from_dict(config.load_task_config("t1_demo")).validate()
        cls.first = run_agent(cls.task, use_llm=False, recipe_lookup=False, max_rounds=3)

    def test_first_round_fails_then_recovers(self) -> None:
        summary = self.first
        self.assertEqual(summary["status"], "passed", summary.get("error"))
        self.assertTrue(summary["passed"])
        self.assertGreaterEqual(summary["rounds_used"], 2)
        first_round = summary["rounds"][0]
        self.assertFalse(first_round["passed"])
        self.assertTrue(summary["attributions"], "失败后应产出归因")
        causes = {item["cause"] for item in summary["attributions"]}
        self.assertTrue({"missing_values_poor_imputation", "class_imbalance"} & causes, causes)

    def test_trace_contains_full_agent_loop(self) -> None:
        events = read_trace(self.first["trace_path"])
        nodes = [event["node"] for event in events]
        for expected in ("load_data", "profile", "plan", "execute", "attribute", "replan", "export", "done"):
            self.assertIn(expected, nodes)
        self.assertTrue(any(event["tool"].startswith("sandbox.") for event in events))

    def test_report_and_bundle_exported(self) -> None:
        report_md = pathlib.Path(self.first["report_md"])
        bundle = pathlib.Path(self.first["bundle_path"])
        self.assertTrue(report_md.exists())
        self.assertTrue(bundle.exists())
        self.assertIn("AutoLab 实验报告", report_md.read_text(encoding="utf-8"))

    def test_recipe_saved_and_reused(self) -> None:
        stats = db.stats()
        self.assertGreaterEqual(stats["recipes"], 1)
        again = run_agent(self.task, use_llm=False, recipe_lookup=True, max_rounds=3)
        self.assertTrue(again["recipe_hit"])
        self.assertEqual(again["rounds_used"], 1)
        self.assertTrue(again["passed"])

    def test_fixed_plan_reproduces(self) -> None:
        plan = self.first["plan"]
        reproduced = run_agent(self.task, use_llm=False, recipe_lookup=False, fixed_plan=plan)
        self.assertTrue(reproduced["passed"])
        self.assertEqual(reproduced["rounds_used"], 1)
        self.assertAlmostEqual(
            float(reproduced["final_metric"]),
            float(self.first["final_metric"]),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
