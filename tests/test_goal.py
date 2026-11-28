"""自然语言目标解析：一句话 → 阈值/预算/轮次。"""
from __future__ import annotations

import unittest

from autolab import config
from autolab.goal import apply_goal, parse_goal
from autolab.schemas import TaskSpec


class ParseGoalTest(unittest.TestCase):
    def test_full_goal(self) -> None:
        parsed = parse_goal("把验证准确率做到 0.86 以上，预算 90 秒，最多 2 轮")
        self.assertEqual(parsed["threshold"], 0.86)
        self.assertEqual(parsed["budget_seconds"], 90)
        self.assertEqual(parsed["max_rounds"], 2)
        self.assertEqual(
            parsed["applied"], {"threshold": 0.86, "budget_seconds": 90, "max_rounds": 2}
        )

    def test_threshold_needs_metric_context(self) -> None:
        self.assertEqual(parse_goal("准确率 0.85 以上")["threshold"], 0.85)
        # "留出集 0.2" 不是指标阈值：没有阈值语义的上下文就忽略
        self.assertIsNone(parse_goal("留出集 0.2，随便跑")["threshold"])

    def test_budget_units(self) -> None:
        self.assertEqual(parse_goal("预算 2 分钟")["budget_seconds"], 120)
        self.assertEqual(parse_goal("预算 45s")["budget_seconds"], 45)

    def test_rounds(self) -> None:
        self.assertEqual(parse_goal("3 轮以内")["max_rounds"], 3)
        self.assertEqual(parse_goal("最多 5 轮")["max_rounds"], 5)

    def test_empty_goal(self) -> None:
        parsed = parse_goal("   ")
        self.assertEqual(parsed["applied"], {})
        self.assertIsNone(parsed["threshold"])

    def test_out_of_range_ignored(self) -> None:
        self.assertIsNone(parse_goal("准确率 1.5")["threshold"])
        self.assertIsNone(parse_goal("预算 9999 秒")["budget_seconds"])
        self.assertIsNone(parse_goal("100 轮")["max_rounds"])


class ApplyGoalTest(unittest.TestCase):
    def test_apply_overrides_task_without_mutating_source(self) -> None:
        task = TaskSpec.from_dict(config.load_task_config("t1_demo")).validate()
        updated = apply_goal(task, parse_goal("准确率 0.9 以上，预算 120 秒，最多 1 轮"))
        self.assertEqual(updated.threshold, 0.9)
        self.assertEqual(updated.budget_seconds, 120)
        self.assertEqual(updated.max_rounds, 1)
        self.assertEqual(task.threshold, 0.85, "原始任务配置不应被修改")

    def test_empty_goal_returns_same_task(self) -> None:
        task = TaskSpec.from_dict(config.load_task_config("t1_demo")).validate()
        self.assertIs(apply_goal(task, parse_goal("")), task)
        self.assertIs(apply_goal(task, None), task)


if __name__ == "__main__":
    unittest.main()
