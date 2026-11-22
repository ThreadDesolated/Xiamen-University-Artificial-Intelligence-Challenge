"""`run --plan latest` 与「方案文件缺失」友好提示的测试。

背景：复现验收.cmd 在仓库里跑时，reports/acceptance_1/plan.json 可能不存在
（该验收证据只在交付包里），此时不应抛 FileNotFoundError 的 Traceback，
而应给出中文提示 + 非零退出码；`--plan latest` 应能回退到最近一次实验的方案。
"""
from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from autolab import cli, config


def _write_plan(root: Path, name: str, task_id: str, mtime: float) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    plan_path = run_dir / "plan.json"
    plan_path.write_text(json.dumps({"plan_id": "p1", "ops": [], "model": {"kind": "nearest_centroid", "params": {}}, "eval": {"n_splits": 5, "seed": 42}}), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": name, "task_id": task_id}), encoding="utf-8")
    import os
    os.utime(plan_path, (mtime, mtime))
    return plan_path


class LatestPlanTest(unittest.TestCase):
    def test_picks_newest_plan_and_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_plan(root, "run_old", "t1_demo", 1000.0)
            newest = _write_plan(root, "run_new", "t1_adult", 2000.0)
            with mock.patch.object(config, "REPORT_DIR", root):
                task_id, path = cli._latest_plan()
            self.assertEqual(task_id, "t1_adult")
            self.assertEqual(path, newest)

    def test_returns_none_when_no_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with mock.patch.object(config, "REPORT_DIR", root):
                self.assertEqual(cli._latest_plan(), (None, None))


class MissingPlanTest(unittest.TestCase):
    def _args(self, plan, task=None):
        return Namespace(plan=plan, task=task, llm=False, no_recipe=False,
                         max_rounds=None, run_id=None)

    def test_missing_plan_file_returns_2_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "reports" / "acceptance_1" / "plan.json"
            with mock.patch("sys.stderr") as err:
                code = cli._cmd_run(self._args(str(missing)))
            self.assertEqual(code, 2)
            text = "".join(str(c.args[0]) for c in err.write.call_args_list)
            self.assertIn("找不到实验方案文件", text)

    def test_no_task_and_no_plan_returns_2(self):
        with mock.patch("sys.stderr") as err:
            code = cli._cmd_run(self._args(None))
        self.assertEqual(code, 2)
        text = "".join(str(c.args[0]) for c in err.write.call_args_list)
        self.assertIn("需要 --task 或 --plan latest", text)


if __name__ == "__main__":
    unittest.main()
