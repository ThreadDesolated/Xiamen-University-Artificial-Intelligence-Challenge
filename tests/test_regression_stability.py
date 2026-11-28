"""回归任务的数值稳定性与 trace JSON 安全。

现场问题：t3_timeseries 的特征量纲差异大（湿度 ~70、温度 ~20），未标准化时
线性回归梯度直接发散成 NaN；NaN 写进 JSONL 后浏览器 JSON.parse 整行失败，
面板上"执行与评估"事件被静默丢弃，看起来像实验根本没有执行。
"""
from __future__ import annotations

import json
import math
import pathlib
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from autolab import config
from autolab.agent.attributor import attribute, replan
from autolab.agent.planner import template_plan
from autolab.evaluation.protocol import run_cv
from autolab.sandbox.operators import LinearGD
from autolab.schemas import ExperimentPlan, TaskSpec, ValidationError
from autolab.tracing import TraceWriter, read_trace


def _regression_frame(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x1 = rng.normal(70, 15, rows)     # 湿度量级
    x2 = rng.normal(20, 8, rows)      # 温度量级
    x3 = rng.normal(2, 0.5, rows)
    y = 0.02 * x1 - 0.05 * x2 + 0.8 * x3 + 3 + rng.normal(0, 0.05, rows)
    return pd.DataFrame({"humidity": x1, "temp": x2, "lag1": x3, "target": y})


class TraceJsonSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="autolab_trace_json_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_non_finite_values_become_null(self) -> None:
        path = self.tmp / "run_x" / "trace.jsonl"
        with TraceWriter(path, "run_x") as writer:
            writer.emit("execute", output={"mean": float("nan"), "train": math.inf, "ok": 1.0})
        text = path.read_text(encoding="utf-8")
        self.assertNotIn("NaN", text)
        self.assertNotIn("Infinity", text)
        event = json.loads(text.strip())
        self.assertIsNone(event["output"]["mean"])
        self.assertIsNone(event["output"]["train"])
        self.assertEqual(event["output"]["ok"], 1.0)

    def test_legacy_nan_trace_is_sanitized_on_read(self) -> None:
        path = self.tmp / "old" / "trace.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(
            '{"run_id":"old","seq":1,"node":"execute","output":{"mean":NaN}}\n',
            encoding="utf-8",
        )
        events = read_trace(path)
        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0]["output"]["mean"])


class LinearGdStabilityTest(unittest.TestCase):
    def test_wide_scale_features_stay_finite(self) -> None:
        frame = _regression_frame()
        X = frame[["humidity", "temp", "lag1"]].to_numpy()
        y = frame["target"].to_numpy()
        model = LinearGD({"lr": 0.05, "epochs": 300, "l2": 0.001}).fit(X, y)
        pred = model.predict(X)
        self.assertTrue(np.all(np.isfinite(pred)))
        mae = float(np.mean(np.abs(pred - y)))
        self.assertTrue(math.isfinite(mae))
        self.assertLess(mae, 1.0)

    def test_regression_cv_without_scaling_is_finite(self) -> None:
        frame = _regression_frame()
        plan = ExperimentPlan(
            ops=[],
            model={"kind": "linear_gd", "params": {"lr": 0.05, "epochs": 300, "l2": 0.001}},
            eval={"n_splits": 3, "seed": 42},
            source="template",
        )
        payload = run_cv(
            plan,
            frame,
            task_type="regression",
            target="target",
            metric="mae",
            n_splits=3,
            seed=42,
        )
        mean = payload["metrics"]["mae"]["mean"]
        self.assertTrue(math.isfinite(mean))
        self.assertTrue(math.isfinite(payload["train_metric"]))


class RegressionReplanTest(unittest.TestCase):
    def test_regression_shortfall_does_not_suggest_tree_model(self) -> None:
        task = config.load_task_config("t3_timeseries")
        from autolab.schemas import TaskSpec

        task = TaskSpec.from_dict(task).validate()
        profile = {
            "n_rows": 1200,
            "n_features": 6,
            "missing_ratio": 0.0,
            "feature_types": {"numeric": [f"x{i}" for i in range(6)], "categorical": []},
            "target": {"type": "regression", "mean": 0.0, "std": 1.0},
            "warnings": [],
        }
        plan = template_plan(profile, task)
        self.assertEqual(plan.model["kind"], "linear_gd")

        class Result:
            status = "ok"
            error = ""
            train_metric = 0.42

            def metric_mean(self, metric):
                return 0.42

        causes = attribute(profile, plan, Result(), task)
        self.assertTrue(causes)
        new_plan = replan(plan, causes, task)
        self.assertEqual(new_plan.model["kind"], "linear_gd")
        self.assertNotIn(
            "random_forest",
            json.dumps([cause.fix for cause in causes], ensure_ascii=False),
        )
        from autolab.sandbox.validate import validate_plan

        self.assertEqual(validate_plan(new_plan, task), [])


class TaskDirectionTest(unittest.TestCase):
    """阈值口径必须和指标方向一致，避免把 MAE 当成"至少 0.45"。"""

    def test_mae_with_higher_is_better_is_rejected(self) -> None:
        raw = config.load_task_config("t3_timeseries")
        raw["higher_is_better"] = True
        with self.assertRaises(ValidationError) as ctx:
            TaskSpec.from_dict(raw).validate()
        self.assertIn("越低越好", str(ctx.exception))

    def test_accuracy_with_lower_is_better_is_rejected(self) -> None:
        raw = config.load_task_config("t1_demo")
        raw["higher_is_better"] = False
        with self.assertRaises(ValidationError):
            TaskSpec.from_dict(raw).validate()

    def test_all_shipped_tasks_are_consistent(self) -> None:
        for path in sorted(config.TASK_DIR.glob("*.json")):
            with self.subTest(task=path.stem):
                TaskSpec.from_dict(config.load_task_config(path.stem)).validate()


if __name__ == "__main__":
    unittest.main()
