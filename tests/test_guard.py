"""容错兜底测试：系统被"喂坏方案"时必须优雅降级，不能崩。

覆盖三道防线（对应 docs/接口契约.md 的稳定性约定）：
1. 白名单拦截：未知算子 / 参数越界 → `validate_plan` 报错，`validate_or_fallback`
   自动退回模板方案，**且退回后的方案真的能跑出结果**（不只是"校验通过"）；
2. 子进程超时强杀：合法的重方案跑不完 → 执行器杀掉子进程，返回
   `status="timeout"` 的结构化结果，主流程继续（归因→重规划）；
3. 运行时异常兜底：数据文件缺失这类错误也要变成结构化结果，而不是抛穿主流程。

超时用例靠 `AUTOLAB_SANDBOX_TIMEOUT` 把预算压到 2 秒，避免测试跑 30 秒以上。
"""
from __future__ import annotations

import dataclasses
import os
import shutil
import time
import unittest
from unittest import mock

from autolab import config
from autolab.sandbox.executor import execute_plan
from autolab.sandbox.validate import validate_or_fallback, validate_plan
from autolab.schemas import ExperimentPlan, TaskSpec

#: 画像片段：模板规划器只需要缺失比例就能产出兜底方案
PROFILE = {"missing_ratio": 0.05, "target": {"imbalance_ratio": 3.0}}

#: 跑得动的模板兜底方案（弱 baseline，但合法）
FALLBACK_MODEL = {"kind": "nearest_centroid", "params": {}}


def _task(**overrides) -> TaskSpec:
    task = TaskSpec.from_dict(config.load_task_config("t1_demo")).validate()
    return dataclasses.replace(task, **overrides) if overrides else task


def _plan(ops=None, model=None) -> ExperimentPlan:
    return ExperimentPlan(
        ops=ops if ops is not None else [],
        model=model or dict(FALLBACK_MODEL),
        eval={"n_splits": 5, "seed": 42},
    )


class _GuardTestBase(unittest.TestCase):
    """每个用例用独立 run_id，产物落在 traces/ 下（git 忽略），跑完即删。"""

    run_id = "guard_test"

    def setUp(self) -> None:
        self.whitelist = config.load_whitelist()
        self.task = _task()
        self.run_id = f"guard_test_{self._testMethodName}"
        shutil.rmtree(config.TRACE_DIR / self.run_id, ignore_errors=True)

    def tearDown(self) -> None:
        shutil.rmtree(config.TRACE_DIR / self.run_id, ignore_errors=True)


class WhitelistInterceptionTest(_GuardTestBase):
    def test_unknown_op_is_blocked_then_falls_back_and_runs(self) -> None:
        """伪造一个 LLM 幻觉出来的算子：必须先被拦，再回退，最后仍跑出指标。"""
        rogue = _plan(ops=[{"op": "drop_all_tables", "params": {}}])

        errors = validate_plan(rogue, self.task, self.whitelist)
        self.assertTrue(any("算子不在白名单" in err for err in errors), errors)

        fallback, info = validate_or_fallback(rogue, self.task, PROFILE, self.whitelist)
        self.assertFalse(info["valid"])
        self.assertEqual(validate_plan(fallback, self.task, self.whitelist), [])

        result = execute_plan(fallback, self.task, run_id=self.run_id, round_idx=1)
        self.assertEqual(result["status"], "ok", result.get("error"))
        self.assertGreater(result["metrics"]["accuracy"]["mean"], 0.5)

    def test_huge_epochs_blocked_by_param_range(self) -> None:
        """`epochs: 99999999` 这类"跑到天荒地老"的方案在白名单参数范围就被拒。"""
        rogue = _plan(model={"kind": "logistic_gd", "params": {"epochs": 99_999_999}})

        errors = validate_plan(rogue, self.task, self.whitelist)
        self.assertTrue(any("epochs" in err and "超出范围" in err for err in errors), errors)

        fallback, info = validate_or_fallback(rogue, self.task, PROFILE, self.whitelist)
        self.assertFalse(info["valid"])
        self.assertEqual(fallback.model.get("kind"), "nearest_centroid")

    def test_unknown_model_blocked(self) -> None:
        rogue = _plan(model={"kind": "gpt_finetune_black_box", "params": {}})
        errors = validate_plan(rogue, self.task, self.whitelist)
        self.assertTrue(any("model.kind 不在白名单" in err for err in errors), errors)


class SubprocessTimeoutTest(_GuardTestBase):
    def test_heavy_plan_is_killed_and_reported_as_timeout(self) -> None:
        """重方案（40 折 × 2000 轮）正常要 ~5 秒：把超时压到 2 秒，验证必杀。"""
        task = _task(n_splits=40)
        heavy = _plan(
            ops=[{"op": "impute_median", "params": {}}, {"op": "scale_standard", "params": {}}],
            model={
                "kind": "logistic_gd",
                "params": {"lr": 0.1, "epochs": 2000, "l2": 0.001, "class_weight": "balanced"},
            },
        )
        self.assertEqual(validate_plan(heavy, task, self.whitelist), [])

        with mock.patch.dict(os.environ, {"AUTOLAB_SANDBOX_TIMEOUT": "2"}):
            started = time.perf_counter()
            result = execute_plan(heavy, task, run_id=self.run_id, round_idx=1)
            elapsed = time.perf_counter() - started

        self.assertEqual(result["status"], "timeout", result.get("error"))
        self.assertIn("超时", result["error"])
        # 必须真的在 2 秒附近被掐断，而不是等它自己跑完（~5 秒）
        self.assertLess(elapsed, 4.0, f"超时未生效，实际耗时 {elapsed:.2f}s")
        self.assertIsNone(result["metrics"].get("accuracy"))

    def test_timeout_budget_override_parsing(self) -> None:
        """覆盖开关要容错：非法值退回默认预算，不能被一个坏环境变量搞崩。"""
        from autolab.sandbox.executor import _timeout_seconds

        with mock.patch.dict(os.environ, {"AUTOLAB_SANDBOX_TIMEOUT": "6"}):
            self.assertEqual(_timeout_seconds(self.task), 6)
        with mock.patch.dict(os.environ, {"AUTOLAB_SANDBOX_TIMEOUT": "abc"}):
            self.assertEqual(_timeout_seconds(self.task), max(30, self.task.budget_seconds + 20))
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AUTOLAB_SANDBOX_TIMEOUT", None)
            self.assertEqual(_timeout_seconds(self.task), max(30, self.task.budget_seconds + 20))


class RuntimeErrorIsStructuredTest(_GuardTestBase):
    def test_missing_data_file_returns_error_dict(self) -> None:
        """数据文件不存在：沙箱把异常转成结构化结果，主流程不会被打断。"""
        task = _task(data_path="data/processed/__no_such_file__.csv")
        result = execute_plan(_plan(), task, run_id=self.run_id, round_idx=1)

        self.assertIn(result["status"], ("error", "invalid"), result)
        self.assertTrue(result.get("error"), result)
        self.assertIsInstance(result["metrics"], dict)


if __name__ == "__main__":
    unittest.main()
