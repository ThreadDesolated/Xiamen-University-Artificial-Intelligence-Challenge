"""predict 子命令测试：用导出的最终模型对新数据预测。"""
from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from autolab import cli
from autolab.datasets.loader import load_dataframe
from autolab.evaluation.protocol import run_cv
from autolab.schemas import ExperimentPlan


class PredictCliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory(prefix="autolab_predict_")
        root = Path(cls._tmp.name)
        cls.model_path = root / "model.joblib"
        frame = load_dataframe("synthetic:t1", "label")
        plan = ExperimentPlan.from_dict(
            {
                "plan_id": "predict_test",
                "ops": [{"op": "impute_median"}, {"op": "scale_standard"}],
                "model": {"kind": "logistic_gd", "params": {"epochs": 60}},
                "eval": {"n_splits": 3, "seed": 42},
                "source": "template",
            }
        )
        run_cv(
            plan,
            frame,
            task_type="binary_classification",
            target="label",
            metric="accuracy",
            n_splits=3,
            seed=42,
            model_path=cls.model_path,
            task_id="t1_demo",
        )
        cls.frame = frame

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _run(self, argv) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli.main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_predict_writes_csv_and_drops_target(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autolab_predict_out_") as tmp:
            csv_path = Path(tmp) / "data.csv"
            out_path = Path(tmp) / "pred.csv"
            self.frame.to_csv(csv_path, index=False)
            code, stdout, stderr = self._run(
                ["predict", "--model", str(self.model_path), "--csv", str(csv_path), "--out", str(out_path)]
            )
            self.assertEqual(code, 0, stderr)
            self.assertIn("AutoLab predict", stdout)
            self.assertTrue(out_path.is_file())
            result = pd.read_csv(out_path)
            self.assertEqual(list(result.columns), ["row", "prediction"])
            self.assertEqual(len(result), len(self.frame))
            self.assertTrue(set(result["prediction"].unique()) <= {0, 1})

    def test_predict_without_target_column(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autolab_predict_out_") as tmp:
            csv_path = Path(tmp) / "features.csv"
            self.frame.drop(columns=["label"]).to_csv(csv_path, index=False)
            code, stdout, _stderr = self._run(
                ["predict", "--model", str(self.model_path), "--csv", str(csv_path)]
            )
            self.assertEqual(code, 0)
            self.assertIn("行数: 1000", stdout)
            self.assertIn("预测分布", stdout)

    def test_predict_proba(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autolab_predict_out_") as tmp:
            csv_path = Path(tmp) / "data.csv"
            out_path = Path(tmp) / "pred.csv"
            self.frame.to_csv(csv_path, index=False)
            code, stdout, stderr = self._run(
                [
                    "predict",
                    "--model",
                    str(self.model_path),
                    "--csv",
                    str(csv_path),
                    "--out",
                    str(out_path),
                    "--proba",
                ]
            )
            self.assertEqual(code, 0, stderr)
            self.assertIn("正类概率前 10 行", stdout)
            result = pd.read_csv(out_path)
            self.assertIn("probability", result.columns)
            self.assertTrue(((result["probability"] >= 0) & (result["probability"] <= 1)).all())

    def test_predict_unknown_target_returns_2(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autolab_predict_out_") as tmp:
            csv_path = Path(tmp) / "data.csv"
            self.frame.to_csv(csv_path, index=False)
            code, _stdout, stderr = self._run(
                [
                    "predict",
                    "--model",
                    str(self.model_path),
                    "--csv",
                    str(csv_path),
                    "--target",
                    "no_such_column",
                ]
            )
            self.assertEqual(code, 2)
            self.assertIn("没有目标列", stderr)

    def test_predict_missing_model_returns_2(self) -> None:
        with tempfile.TemporaryDirectory(prefix="autolab_predict_out_") as tmp:
            csv_path = Path(tmp) / "data.csv"
            self.frame.to_csv(csv_path, index=False)
            code, _stdout, stderr = self._run(
                ["predict", "--model", str(Path(tmp) / "nope.joblib"), "--csv", str(csv_path)]
            )
            self.assertEqual(code, 2)
            self.assertIn("模型文件不存在", stderr)


if __name__ == "__main__":
    unittest.main()
