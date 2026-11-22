"""`model-info` 命令（查看导出模型全貌）的回归测试。"""
from __future__ import annotations

import contextlib
import io
import json
import pathlib
import tempfile
import unittest

import numpy as np
import pandas as pd

from autolab.cli import main
from autolab.evaluation import model_store, protocol
from autolab.schemas import ExperimentPlan


def _df(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    label = np.where(x1 + 0.5 * x2 + 0.2 * rng.normal(size=n) > 0, "yes", "no")
    return pd.DataFrame({"x1": x1, "x2": x2, "target": label})


def _plan(kind: str = "random_forest") -> ExperimentPlan:
    model = (
        {"kind": "random_forest", "params": {"n_estimators": 20, "max_depth": 3, "min_samples_leaf": 1}}
        if kind == "random_forest"
        else {"kind": "logistic_gd", "params": {"lr": 0.1, "epochs": 80, "l2": 0.001, "class_weight": "balanced"}}
    )
    return ExperimentPlan.from_dict(
        {
            "plan_id": "plan_model_info_test",
            "ops": [{"op": "impute_median"}, {"op": "scale_standard"}],
            "model": model,
            "eval": {"n_splits": 3, "seed": 42},
            "source": "template",
        }
    )


def _run_info(*argv: str) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        code = main(["model-info", *argv])
    return code, stdout.getvalue(), stderr.getvalue()


class ModelInfoTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="autolab_model_info_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def _train(self, kind: str = "random_forest") -> pathlib.Path:
        path = self.tmp / f"{kind}.joblib"
        protocol.run_cv(
            _plan(kind),
            _df(),
            task_type="binary_classification",
            target="target",
            metric="accuracy",
            n_splits=3,
            model_path=path,
            task_id="t1_demo",
        )
        return path

    def test_random_forest_info_prints_hyperparams_features_and_importance(self) -> None:
        path = self._train("random_forest")
        code, out, err = _run_info("--model", str(path), "--top", "5")
        self.assertEqual(code, 0, err)
        for text in ("AutoLab model-info", "超参数", "n_estimators", "实验方案",
                     "预处理管线", "特征名单", "训练结果", "特征重要性", "x1"):
            self.assertIn(text, out)

    def test_json_output_is_parseable_and_complete(self) -> None:
        path = self._train("random_forest")
        code, out, err = _run_info("--model", str(path), "--json")
        self.assertEqual(code, 0, err)
        data = json.loads(out)
        self.assertEqual(data["meta"]["model_kind"], "random_forest")
        self.assertEqual(data["model"]["trained"]["n_classes"], 2)
        self.assertGreaterEqual(data["pipeline"]["n_features"], 2)
        self.assertEqual(data["model"]["kind"], "random_forest")

    def test_features_flag_prints_all_feature_names(self) -> None:
        path = self._train("random_forest")
        code, out, err = _run_info("--model", str(path), "--features")
        self.assertEqual(code, 0, err)
        self.assertIn("x2", out)
        self.assertNotIn("前 10 个", out)

    def test_logistic_model_prints_weight_summary(self) -> None:
        path = self._train("logistic_gd")
        code, out, err = _run_info("--model", str(path))
        self.assertEqual(code, 0, err)
        self.assertIn("权重", out)
        self.assertIn("维度", out)

    def test_missing_model_returns_2(self) -> None:
        code, out, err = _run_info("--model", str(self.tmp / "nope.joblib"))
        self.assertEqual(code, 2)
        self.assertIn("模型文件不存在", err)

    def test_foreign_file_returns_2(self) -> None:
        foreign = self.tmp / "foreign.joblib"
        model_store._dump({"x": 1}, foreign, model_store._serializer())
        code, out, err = _run_info("--model", str(foreign))
        self.assertEqual(code, 2)
        self.assertIn("无法加载模型", err)


if __name__ == "__main__":
    unittest.main()
