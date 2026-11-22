"""训练后模型导出（model_store / protocol / report / server）的回归测试。"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

import numpy as np
import pandas as pd

from autolab import config
from autolab.evaluation import model_store, protocol
from autolab.report import export_bundle
from autolab.schemas import ExperimentPlan


def _df(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    label = np.where(x1 + 0.5 * x2 + 0.2 * rng.normal(size=n) > 0, "yes", "no")
    return pd.DataFrame({"x1": x1, "x2": x2, "target": label})


def _plan() -> ExperimentPlan:
    return ExperimentPlan.from_dict(
        {
            "plan_id": "plan_model_test",
            "ops": [{"op": "impute_median"}, {"op": "scale_standard"}],
            "model": {
                "kind": "logistic_gd",
                "params": {"lr": 0.1, "epochs": 80, "l2": 0.001, "class_weight": "balanced"},
            },
            "eval": {"n_splits": 3, "seed": 42},
            "source": "template",
        }
    )


class ModelExportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="autolab_model_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))

    def test_run_cv_exports_model_with_meta_and_label_mapping(self) -> None:
        path = self.tmp / "model.joblib"
        result = protocol.run_cv(
            _plan(),
            _df(),
            task_type="binary_classification",
            target="target",
            metric="accuracy",
            n_splits=3,
            seed=42,
            model_path=path,
            task_id="t1_demo",
        )
        self.assertTrue(path.exists())
        model_meta = result["artifacts"]["model"]
        self.assertEqual(model_meta["format"], model_store.MODEL_FORMAT)
        self.assertEqual(model_meta["model_kind"], "logistic_gd")
        self.assertEqual(model_meta["model_params"]["class_weight"], "balanced")
        self.assertIsNotNone(model_meta["cv_mean"])

        payload = model_store.load_model(path)
        self.assertEqual(payload["meta"]["task_id"], "t1_demo")
        self.assertEqual(payload["meta"]["classes"], ["no", "yes"])
        pred = model_store.predict(payload, _df().drop(columns=["target"]).head(8))
        self.assertEqual(len(pred), 8)
        self.assertTrue(set(pred) <= {"no", "yes"})

    def test_run_cv_without_model_path_has_no_artifacts(self) -> None:
        result = protocol.run_cv(
            _plan(), _df(), task_type="binary_classification", target="target", metric="accuracy", n_splits=3
        )
        self.assertFalse(result.get("artifacts"))

    def test_pickle_fallback_when_joblib_missing(self) -> None:
        path = self.tmp / "model.joblib"
        with mock.patch.dict(sys.modules, {"joblib": None}):
            result = protocol.run_cv(
                _plan(),
                _df(),
                task_type="binary_classification",
                target="target",
                metric="accuracy",
                n_splits=3,
                model_path=path,
                task_id="t1_demo",
            )
            self.assertEqual(result["artifacts"]["model"]["serializer"], "pickle")
            payload = model_store.load_model(path)
            pred = model_store.predict(payload, _df().drop(columns=["target"]).head(4))
        self.assertEqual(len(pred), 4)

    def test_export_bundle_copies_model_into_report_dir_and_zip(self) -> None:
        reports = self.tmp / "reports"
        traces = self.tmp / "traces"
        reports.mkdir()
        traces.mkdir()
        with mock.patch.object(config, "REPORT_DIR", reports), mock.patch.object(config, "TRACE_DIR", traces):
            model_path = self.tmp / "round_model.joblib"
            result = protocol.run_cv(
                _plan(),
                _df(),
                task_type="binary_classification",
                target="target",
                metric="accuracy",
                n_splits=3,
                model_path=model_path,
                task_id="t1_demo",
            )
            summary = {
                "run_id": "run_model_test",
                "task_id": "t1_demo",
                "status": "passed",
                "passed": True,
                "metric_name": "accuracy",
                "final_metric": 0.9,
                "rounds_used": 1,
                "rounds": [{"round": 1, "plan": _plan().to_dict(), "result": result, "score": 0.9, "passed": True}],
                "profile": {},
                "env": {},
                "plan": _plan().to_dict(),
                "trace_path": str(self.tmp / "no_such_trace.jsonl"),
            }
            files = export_bundle(summary)
            run_dir = pathlib.Path(files["report_dir"])
            model_file = run_dir / "model.joblib"
            self.assertTrue(model_file.exists())
            self.assertEqual(files["model_path"], str(model_file))

            report_text = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("model.joblib", report_text)
            self.assertIn("加载与预测", report_text)

            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["model"]["model_file"], "model.joblib")

            with zipfile.ZipFile(files["bundle_path"]) as zf:
                names = zf.namelist()
            self.assertIn("run_model_test/model.joblib", names)
            self.assertIn("run_model_test/manifest.json", names)

    def test_load_model_rejects_foreign_file(self) -> None:
        foreign = self.tmp / "foreign.joblib"
        model_store._dump({"x": 1}, foreign, model_store._serializer())
        with self.assertRaises(ValueError):
            model_store.load_model(foreign)


class ModelEndpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="autolab_model_api_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, ignore_errors=True))
        self.run_id = "run_20260101_000000_abcdef"
        run_dir = self.tmp / self.run_id
        run_dir.mkdir(parents=True)
        (run_dir / "model.joblib").write_bytes(b"fake-model-bytes")
        self.run_dir = run_dir

    def _client(self):
        from fastapi.testclient import TestClient

        from autolab.serving.server import create_app

        return TestClient(create_app())

    def test_download_model_returns_file(self) -> None:
        with mock.patch.object(config, "REPORT_DIR", self.tmp):
            response = self._client().get(f"/api/runs/{self.run_id}/model")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"fake-model-bytes")
        self.assertIn("model.joblib", response.headers.get("content-disposition", ""))

    def test_download_model_missing_returns_404(self) -> None:
        with mock.patch.object(config, "REPORT_DIR", self.tmp):
            response = self._client().get("/api/runs/run_missing_model/model")
        self.assertEqual(response.status_code, 404)

    def test_download_model_illegal_id_returns_400(self) -> None:
        with mock.patch.object(config, "REPORT_DIR", self.tmp):
            response = self._client().get("/api/runs/bad%20id/model")
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
