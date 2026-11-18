"""留出集（holdout）评测协议测试。

设计目标：
- holdout 在 K 折之前切出，K 折只在剩余训练池上做；
- holdout 不参与选模/调参，只做最终复验（防验证集过拟合）；
- 分类留出分层、种子与 K 折错开；未启用（ratio=0）时行为与旧版完全一致。
"""
from __future__ import annotations

import pathlib
import unittest

import numpy as np
import pandas as pd

from autolab.datasets import load_dataframe
from autolab.evaluation.protocol import (
    holdout_indices,
    kfold_indices,
    run_cv,
    stratified_folds,
    stratified_holdout_indices,
)
from autolab.report import markdown_report
from autolab.schemas import ExperimentPlan, TaskSpec, ValidationError


def _demo_task(**overrides) -> TaskSpec:
    base = dict(
        task_id="t1_demo",
        name="内置合成任务",
        task_type="binary_classification",
        data_path="synthetic:t1",
        target="label",
        metric="accuracy",
        threshold=0.85,
    )
    base.update(overrides)
    return TaskSpec(**base)


def _simple_plan() -> ExperimentPlan:
    return ExperimentPlan(
        plan_id="holdout_test",
        ops=[{"op": "impute_median"}, {"op": "scale_standard"}],
        model={"kind": "logistic_gd", "params": {"epochs": 100, "class_weight": "balanced"}},
        eval={"n_splits": 5, "seed": 42},
    )


class HoldoutTaskSpecTest(unittest.TestCase):
    def test_default_is_disabled(self) -> None:
        self.assertEqual(_demo_task().holdout_ratio, 0.0)

    def test_valid_ratio_accepted(self) -> None:
        self.assertEqual(_demo_task(holdout_ratio=0.2).holdout_ratio, 0.2)

    def test_out_of_range_rejected(self) -> None:
        for bad in (-0.1, 0.5, 0.8):
            with self.assertRaises(ValidationError, msg=str(bad)):
                _demo_task(holdout_ratio=bad).validate()


class HoldoutSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.df = load_dataframe("synthetic:t1", "label")

    def test_split_sizes_and_stratification(self) -> None:
        task = _demo_task(holdout_ratio=0.2)
        result = run_cv(
            _simple_plan(),
            self.df,
            task_type=task.task_type,
            target=task.target,
            metric=task.metric,
            n_splits=5,
            seed=task.seed,
            holdout_ratio=task.holdout_ratio,
        )
        holdout = result["holdout"]
        self.assertIsNotNone(holdout)
        self.assertEqual(holdout["n_samples"], 200)
        self.assertEqual(result["n_cv_samples"], 800)
        self.assertEqual(result["n_samples"], 1000)
        self.assertEqual(holdout["metric"], "accuracy")
        self.assertEqual(holdout["split_seed"], 42)
        self.assertAlmostEqual(holdout["ratio"], 0.2, places=2)
        self.assertTrue(0.0 <= holdout["value"] <= 1.0)

    def test_disabled_keeps_old_behaviour(self) -> None:
        task = _demo_task()
        result = run_cv(
            _simple_plan(),
            self.df,
            task_type=task.task_type,
            target=task.target,
            metric=task.metric,
            n_splits=5,
            seed=task.seed,
        )
        self.assertIsNone(result["holdout"])
        self.assertEqual(result["n_cv_samples"], result["n_samples"])

    def test_stratified_holdout_keeps_class_ratio(self) -> None:
        y = np.array([0] * 900 + [1] * 100)
        idx = stratified_holdout_indices(y, 0.2, seed=42)
        self.assertEqual(len(idx), 200)
        holdout_classes, counts = np.unique(y[idx], return_counts=True)
        self.assertEqual(list(holdout_classes), [0, 1])
        self.assertEqual(list(counts), [180, 20])

    def test_no_overlap_between_holdout_and_cv_folds(self) -> None:
        n = 1000
        y = np.array([0] * 750 + [1] * 250)
        holdout = stratified_holdout_indices(y, 0.2, seed=42)
        train_idx = np.setdiff1d(np.arange(n), holdout)
        folds = stratified_folds(y[train_idx], 5, 42)
        # folds 返回的是「训练池内位置」，映射回原始下标后再校验
        mapped = np.concatenate([train_idx[fold] for fold in folds])
        self.assertEqual(len(mapped), len(train_idx))
        self.assertEqual(sorted(mapped.tolist()), sorted(train_idx.tolist()))
        self.assertEqual(len(np.intersect1d(mapped, holdout)), 0)

    def test_regression_holdout_indices_reproducible(self) -> None:
        a = holdout_indices(50, 0.2, seed=42)
        b = holdout_indices(50, 0.2, seed=42)
        self.assertEqual(a.tolist(), b.tolist())
        self.assertEqual(len(a), 10)
        self.assertEqual(len(kfold_indices(50, 5, 42)), 5)

    def test_too_small_train_pool_rejected(self) -> None:
        df = self.df.head(40).copy()
        with self.assertRaises(ValidationError) as ctx:
            run_cv(
                _simple_plan(),
                df,
                task_type="binary_classification",
                target="label",
                metric="accuracy",
                n_splits=5,
                seed=42,
                holdout_ratio=0.2,
            )
        self.assertIn("训练样本不足", str(ctx.exception))


class ReportHoldoutTest(unittest.TestCase):
    def test_report_renders_holdout_section(self) -> None:
        summary = {
            "run_id": "run_test",
            "task_id": "t1_adult",
            "task_name": "Adult",
            "task_type": "binary_classification",
            "metric_name": "accuracy",
            "threshold": 0.85,
            "higher_is_better": True,
            "passed": True,
            "final_metric": 0.856,
            "rounds_used": 2,
            "max_rounds": 3,
            "planner_source": "template",
            "recipe_hit": False,
            "rounds": [
                {
                    "round": 1,
                    "plan": {"source": "template"},
                    "result": {"metrics": {"accuracy": {"mean": 0.52, "std": 0.01}}, "seconds": 0.3},
                    "holdout": {"metric": "accuracy", "value": 0.51, "n_samples": 200},
                    "passed": False,
                },
                {
                    "round": 2,
                    "plan": {"source": "attribution"},
                    "result": {"metrics": {"accuracy": {"mean": 0.856, "std": 0.01}}, "seconds": 1.5},
                    "holdout": {"metric": "accuracy", "value": 0.842, "n_samples": 200},
                    "passed": True,
                },
            ],
            "plan": {"model": {"kind": "random_forest"}},
            "profile": {"n_rows": 1000, "n_features": 14},
            "env": {},
            "holdout": {"metric": "accuracy", "value": 0.842, "n_samples": 200, "ratio": 0.2, "split_seed": 42},
        }
        text = markdown_report(summary)
        self.assertIn("留出集复验", text)
        self.assertIn("## 五、留出集复验", text)
        self.assertIn("## 七、环境信息", text)
        self.assertIn("0.842", text)


if __name__ == "__main__":
    unittest.main()
