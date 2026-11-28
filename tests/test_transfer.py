"""跨任务配方迁移：画像相似度、配方库检索、端到端迁移证据。

迁移与"同任务命中"的分工：
- 同画像指纹命中（db.find_recipe）优先，保证复现稳定；
- 指纹不同但画像结构相似时，才走 transfer.find_transfer_recipe 复用历史方案。
"""
from __future__ import annotations

import pathlib
import shutil
import sqlite3
import tempfile
import unittest
from unittest import mock

from autolab import config
from autolab.datasets.loader import load_dataframe
from autolab.profile import profile_dataframe, profile_hash, profile_similarity
from autolab.store import db
from autolab.transfer import find_transfer_recipe

#: 最小可用方案（迁移只关心"能不能复用"，不执行）
PLAN = {
    "ops": [{"op": "impute_median", "params": {}}, {"op": "scale_standard", "params": {}}],
    "model": {"kind": "logistic_gd", "params": {"class_weight": "balanced"}},
    "eval": {"n_splits": 5, "seed": 42},
    "source": "attribution",
    "notes": "",
}


def _profile(rows=1000, n_numeric=8, n_categorical=0, missing=0.05, imbalance=3.0,
             task_type="binary_classification"):
    return {
        "n_rows": rows,
        "n_features": n_numeric + n_categorical,
        "missing_ratio": missing,
        "feature_types": {
            "numeric": [f"x{i}" for i in range(n_numeric)],
            "categorical": [f"c{i}" for i in range(n_categorical)],
        },
        "target": {"type": task_type, "imbalance_ratio": imbalance},
        "warnings": [],
    }


class ProfileSimilarityTest(unittest.TestCase):
    def test_identical_profile_scores_one(self) -> None:
        profile = _profile()
        self.assertEqual(profile_similarity(profile, dict(profile)), 1.0)

    def test_different_task_type_scores_zero(self) -> None:
        self.assertEqual(
            profile_similarity(_profile(), _profile(task_type="regression")), 0.0
        )

    def test_near_profile_above_threshold_far_profile_below(self) -> None:
        base = _profile()
        near = _profile(rows=1400, n_numeric=12, missing=0.04, imbalance=4.0)
        far = _profile(rows=800, n_numeric=40, n_categorical=30, missing=0.25, imbalance=1.0)
        self.assertGreaterEqual(profile_similarity(base, near), 0.6)
        self.assertLess(profile_similarity(base, far), 0.6)
        self.assertGreater(profile_similarity(base, near), profile_similarity(base, far))

    def test_synthetic_tasks_are_similar_but_not_identical(self) -> None:
        demo = profile_dataframe(load_dataframe("synthetic:t1", "label"), "label", "binary_classification")
        shift = profile_dataframe(
            load_dataframe("synthetic:t1_shift", "label"), "label", "binary_classification"
        )
        self.assertNotEqual(profile_hash(demo), profile_hash(shift), "指纹不同才会走迁移")
        self.assertGreaterEqual(profile_similarity(demo, shift), 0.6)


class RecipeTransferLookupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="autolab_transfer_"))
        self.db_path = self.tmp / "autolab.sqlite3"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_find_transfer_recipe_prefers_most_similar(self) -> None:
        db.save_recipe(
            "binary_classification", "hash_near", PLAN, "accuracy", 0.9,
            db_path=self.db_path, task_id="t_near", profile=_profile(rows=1200, n_numeric=10),
        )
        db.save_recipe(
            "binary_classification", "hash_far", PLAN, "accuracy", 0.95,
            db_path=self.db_path, task_id="t_far",
            profile=_profile(rows=300, n_numeric=50, n_categorical=40, missing=0.3),
        )
        current = _profile()
        found = find_transfer_recipe("binary_classification", current, db_path=self.db_path)
        self.assertIsNotNone(found)
        self.assertEqual(found["task_id"], "t_near")
        self.assertGreaterEqual(found["similarity"], 0.6)

    def test_exact_hash_excluded_from_transfer(self) -> None:
        profile = _profile()
        phash = profile_hash(profile)
        db.save_recipe(
            "binary_classification", phash, PLAN, "accuracy", 0.9,
            db_path=self.db_path, task_id="t_same", profile=profile,
        )
        self.assertIsNone(
            find_transfer_recipe(
                "binary_classification", profile, exclude_profile_hash=phash, db_path=self.db_path
            )
        )

    def test_threshold_filters_and_task_type_isolated(self) -> None:
        db.save_recipe(
            "binary_classification", "hash_far", PLAN, "accuracy", 0.9,
            db_path=self.db_path, task_id="t_far",
            profile=_profile(rows=300, n_numeric=50, n_categorical=40, missing=0.3),
        )
        self.assertIsNone(find_transfer_recipe("binary_classification", _profile(), db_path=self.db_path))
        # 回归任务不会复用分类配方（画像 task_type 不同，相似度为 0）
        regression = _profile(task_type="regression")
        self.assertIsNone(find_transfer_recipe("regression", regression, db_path=self.db_path))

    def test_old_recipe_without_profile_is_skipped(self) -> None:
        db.save_recipe(
            "binary_classification", "hash_old", PLAN, "accuracy", 0.9,
            db_path=self.db_path, task_id="t_old",
        )
        self.assertIsNone(find_transfer_recipe("binary_classification", _profile(), db_path=self.db_path))

    def test_old_schema_is_migrated(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        conn.executescript(
            "CREATE TABLE recipes (recipe_id TEXT PRIMARY KEY, task_type TEXT NOT NULL,"
            " profile_hash TEXT NOT NULL, plan_json TEXT NOT NULL, metric_name TEXT NOT NULL,"
            " metric_value REAL NOT NULL, created_at REAL NOT NULL, notes TEXT DEFAULT '');"
            "INSERT INTO recipes VALUES ('r_old','binary_classification','h','{}','accuracy',0.8,0,'');"
        )
        conn.commit()
        conn.close()
        db.init_db(self.db_path)
        check = sqlite3.connect(str(self.db_path))
        try:
            columns = {row[1] for row in check.execute("PRAGMA table_info(recipes)")}
            count = check.execute("SELECT COUNT(*) FROM recipes").fetchone()[0]
        finally:
            check.close()
        self.assertTrue({"task_id", "profile_json"} <= columns)
        self.assertEqual(count, 1, "迁移不能丢历史配方")


class TransferEndToEndTest(unittest.TestCase):
    """t1_demo 沉淀配方后，t1_shift（指纹不同、画像相近）首轮迁移达标。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = pathlib.Path(tempfile.mkdtemp(prefix="autolab_transfer_e2e_"))
        cls._patches = [
            mock.patch.object(config, "TRACE_DIR", cls.tmp / "traces"),
            mock.patch.object(config, "REPORT_DIR", cls.tmp / "reports"),
            mock.patch.object(config, "DB_PATH", cls.tmp / "autolab.sqlite3"),
        ]
        for patch in cls._patches:
            patch.start()

        from autolab.agent.graph import run_agent
        from autolab.schemas import TaskSpec

        cls.run_agent = staticmethod(run_agent)
        demo = TaskSpec.from_dict(config.load_task_config("t1_demo")).validate()
        shift = TaskSpec.from_dict(config.load_task_config("t1_shift")).validate()
        cls.demo_summary = run_agent(demo, use_llm=False, recipe_lookup=False, max_rounds=3)
        cls.shift_summary = run_agent(shift, use_llm=False, recipe_lookup=True, transfer_lookup=True)

    @classmethod
    def tearDownClass(cls) -> None:
        for patch in cls._patches:
            patch.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_demo_seeds_recipe(self) -> None:
        self.assertTrue(self.demo_summary["passed"], self.demo_summary.get("error"))
        self.assertGreaterEqual(db.stats()["recipes"], 1)

    def test_shift_hits_transfer_and_passes_first_round(self) -> None:
        summary = self.shift_summary
        self.assertTrue(summary["passed"], summary.get("error"))
        self.assertTrue(summary["recipe_hit"])
        self.assertTrue(summary["recipe_transfer"])
        self.assertEqual(summary["rounds_used"], 1)
        self.assertEqual(summary["planner_source"], "transfer")
        transfer = summary["transfer"] or {}
        self.assertEqual(transfer.get("source_task_id"), "t1_demo")
        self.assertGreaterEqual(transfer.get("similarity", 0), 0.6)

    def test_trace_records_transfer_evidence(self) -> None:
        from autolab.tracing import read_trace

        events = read_trace(self.shift_summary["trace_path"])
        hit = next(e for e in events if e["node"] == "recipe_hit")
        self.assertEqual(hit["output"]["match"], "similar")
        self.assertEqual(hit["output"]["source_task_id"], "t1_demo")
        plan = next(e for e in events if e["node"] == "plan")
        self.assertEqual(plan["output"]["plan"]["source"], "transfer")
        self.assertTrue(plan["output"]["rationale"])


if __name__ == "__main__":
    unittest.main()
