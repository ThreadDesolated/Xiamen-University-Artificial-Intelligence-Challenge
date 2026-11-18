"""真实数据达标测试：UCI Adult 子集（T1）+ 随机森林模型升级。

背景：Adult 1k 行固定子集上，numpy 线性/质心模型上限约 0.845，达不到 0.85 阈值；
归因在"欠拟合/未知缺口"时把模型升级为 sklearn 随机森林，第二轮即可达标。

- 数据文件随仓库提交（data/processed/t1_adult.csv），不依赖网络；
- scikit-learn 是可选依赖：缺失时树模型用例整体跳过（零依赖 demo 仍走 numpy 线性模型兜底）。
"""
from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest
from unittest import mock

import pandas as pd

from autolab import config
from autolab.agent.attributor import attribute, replan
from autolab.agent.graph import run_agent
from autolab.agent.planner import template_plan
from autolab.evaluation.protocol import run_cv
from autolab.profile import profile_dataframe
from autolab.sandbox.validate import validate_plan
from autolab.schemas import ExperimentPlan, TaskSpec

ROOT = pathlib.Path(__file__).resolve().parents[1]
ADULT_DATA = ROOT / "data" / "processed" / "t1_adult.csv"
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


class _FakeResult:
    """只实现归因需要的字段，避免真跑一次沙箱。"""

    def __init__(self, mean: float, train: float | None, status: str = "ok"):
        self.status = status
        self.error = "超出预算" if status == "timeout" else None
        self.train_metric = train
        self._mean = mean

    def metric_mean(self, metric: str) -> float:
        return self._mean


def _adult_task() -> TaskSpec:
    return TaskSpec.from_dict(config.load_task_config("t1_adult")).validate()


def _adult_profile() -> dict:
    return profile_dataframe(pd.read_csv(ADULT_DATA), "income")


def _rf_plan() -> ExperimentPlan:
    return ExperimentPlan(
        ops=[
            {"op": "onehot", "params": {"max_categories": 20}},
            {"op": "scale_standard", "params": {}},
        ],
        model={"kind": "random_forest", "params": {"n_estimators": 300, "max_depth": 8}},
        eval={"n_splits": 5, "seed": 42},
    )


class RandomForestWhitelistTest(unittest.TestCase):
    def test_random_forest_plan_is_valid(self) -> None:
        self.assertEqual(validate_plan(_rf_plan(), _adult_task()), [])

    def test_random_forest_out_of_range_rejected(self) -> None:
        plan = _rf_plan()
        plan.model["params"]["max_depth"] = 99
        errors = validate_plan(plan, _adult_task())
        self.assertTrue(any("max_depth" in err for err in errors), errors)


class ZeroDependencyFallbackTest(unittest.TestCase):
    def test_underfit_falls_back_to_linear_without_sklearn(self) -> None:
        task = _adult_task()
        plan = template_plan(_adult_profile(), task)  # nearest_centroid
        with mock.patch("autolab.agent.attributor._sklearn_available", return_value=False):
            causes = attribute(_adult_profile(), plan, _FakeResult(0.52, 0.51), task)
        new_plan = replan(plan, causes, task)
        self.assertEqual(new_plan.model["kind"], "logistic_gd")
        self.assertEqual(validate_plan(new_plan, task), [])


@unittest.skipUnless(HAS_SKLEARN, "需要 scikit-learn")
@unittest.skipUnless(ADULT_DATA.exists(), "缺少 data/processed/t1_adult.csv")
class TreeModelAccuracyTest(unittest.TestCase):
    def test_random_forest_reaches_adult_threshold(self) -> None:
        task = _adult_task()
        result = run_cv(
            _rf_plan(),
            pd.read_csv(ADULT_DATA),
            task_type=task.task_type,
            target=task.target,
            metric=task.metric,
            n_splits=task.n_splits,
            seed=task.seed,
        )
        self.assertGreaterEqual(result["metrics"][task.metric]["mean"], 0.85)

    def test_attributor_escalates_underfit_to_random_forest(self) -> None:
        task = _adult_task()
        profile = _adult_profile()
        plan = template_plan(profile, task)
        causes = attribute(profile, plan, _FakeResult(0.52, 0.51), task)
        self.assertIn("underfit", [cause.cause for cause in causes])
        new_plan = replan(plan, causes, task)
        self.assertEqual(new_plan.model["kind"], "random_forest")
        self.assertEqual(validate_plan(new_plan, task), [])

    def test_timeout_fix_keeps_tree_model_params_valid(self) -> None:
        task = _adult_task()
        profile = _adult_profile()
        plan = _rf_plan()
        causes = attribute(profile, plan, _FakeResult(0.84, 0.9, status="timeout"), task)
        self.assertEqual(causes[0].cause, "budget_exceeded")
        new_plan = replan(plan, causes, task)
        self.assertIn("n_estimators", new_plan.model.get("params", {}))
        self.assertEqual(validate_plan(new_plan, task), [])


@unittest.skipUnless(HAS_SKLEARN, "需要 scikit-learn")
@unittest.skipUnless(ADULT_DATA.exists(), "缺少 data/processed/t1_adult.csv")
class AdultEndToEndTest(unittest.TestCase):
    """失败 → 归因 → 升级随机森林 → 达标（<=2 轮）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="autolab_adult_e2e_"))
        cls.backup = (config.TRACE_DIR, config.REPORT_DIR, config.DB_PATH)
        config.TRACE_DIR = cls.tmpdir / "traces"
        config.REPORT_DIR = cls.tmpdir / "reports"
        config.DB_PATH = cls.tmpdir / "autolab_test.sqlite3"
        cls.summary = run_agent(_adult_task(), use_llm=False, recipe_lookup=False, max_rounds=3)

    @classmethod
    def tearDownClass(cls) -> None:
        config.TRACE_DIR, config.REPORT_DIR, config.DB_PATH = cls.backup

    def test_passed_within_two_rounds(self) -> None:
        summary = self.summary
        self.assertEqual(summary["status"], "passed", summary.get("error"))
        self.assertTrue(summary["passed"])
        self.assertLessEqual(summary["rounds_used"], 2)
        self.assertGreaterEqual(float(summary["final_metric"]), 0.85)
        self.assertEqual(summary["plan"]["model"]["kind"], "random_forest")
        first = summary["rounds"][0]
        self.assertFalse(first["passed"])
        self.assertTrue(summary["attributions"], "首轮失败后应产出归因")


if __name__ == "__main__":
    unittest.main()
