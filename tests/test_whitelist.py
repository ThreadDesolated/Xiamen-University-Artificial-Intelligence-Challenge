"""白名单校验测试：动作空间受限是整套系统安全边界的第一道门。"""
from __future__ import annotations

import unittest

from autolab import config
from autolab.agent.planner import template_plan
from autolab.sandbox.validate import validate_or_fallback, validate_plan
from autolab.schemas import ExperimentPlan, TaskSpec


class WhitelistTest(unittest.TestCase):
    def setUp(self) -> None:
        self.whitelist = config.load_whitelist()
        self.task = TaskSpec.from_dict(config.load_task_config("t1_demo")).validate()
        self.profile = {"missing_ratio": 0.05, "target": {"imbalance_ratio": 3.0}}

    def test_template_plan_is_valid(self) -> None:
        plan = template_plan(self.profile, self.task)
        self.assertEqual(validate_plan(plan, self.task, self.whitelist), [])

    def test_unknown_model_rejected(self) -> None:
        plan = ExperimentPlan(
            ops=[],
            model={"kind": "magic_net", "params": {}},
            eval={"n_splits": 5, "seed": 42},
        )
        errors = validate_plan(plan, self.task, self.whitelist)
        self.assertTrue(any("model.kind" in err for err in errors), errors)

    def test_unknown_op_rejected(self) -> None:
        plan = ExperimentPlan(
            ops=[{"op": "magic_feature", "params": {}}],
            model={"kind": "nearest_centroid", "params": {}},
            eval={"n_splits": 5, "seed": 42},
        )
        errors = validate_plan(plan, self.task, self.whitelist)
        self.assertTrue(any("算子不在白名单" in err for err in errors), errors)

    def test_param_out_of_range_rejected(self) -> None:
        plan = ExperimentPlan(
            ops=[],
            model={"kind": "logistic_gd", "params": {"lr": 99.0}},
            eval={"n_splits": 5, "seed": 42},
        )
        errors = validate_plan(plan, self.task, self.whitelist)
        self.assertTrue(any("lr" in err for err in errors), errors)

    def test_invalid_plan_falls_back_to_template(self) -> None:
        plan = ExperimentPlan(
            ops=[{"op": "magic_feature", "params": {}}],
            model={"kind": "nearest_centroid", "params": {}},
            eval={"n_splits": 5, "seed": 42},
        )
        fallback, info = validate_or_fallback(plan, self.task, self.profile, self.whitelist)
        self.assertFalse(info["valid"])
        self.assertEqual(validate_plan(fallback, self.task, self.whitelist), [])


if __name__ == "__main__":
    unittest.main()
