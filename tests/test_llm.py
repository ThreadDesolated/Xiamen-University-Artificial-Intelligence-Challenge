"""大模型规划 / 归因的接入与降级验证。

所有网络调用都被 mock 掉：这里只验证"模型输出 → 校验 → 执行"的接线与回退，
不依赖真实 API Key，CI 与离线演示都能跑。
"""
from __future__ import annotations

import json
import pathlib
import shutil
import tempfile
import unittest
from unittest import mock

from autolab import config
from autolab.agent import attributor, planner
from autolab.agent.attributor import llm_attribute
from autolab.agent.graph import run_agent
from autolab.llm import chat_json
from autolab.sandbox.validate import validate_plan
from autolab.schemas import Attribution, TaskSpec, ValidationError
from autolab.tracing import read_trace


class LlmChatClientTest(unittest.TestCase):
    def test_missing_key_raises(self) -> None:
        with mock.patch.object(config, "API_KEY", ""):
            with self.assertRaises(ValidationError):
                chat_json("system", "user")


class LlmGraphFallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="autolab_llm_"))
        cls._patches = [
            mock.patch.object(config, "TRACE_DIR", cls.tmpdir / "traces"),
            mock.patch.object(config, "REPORT_DIR", cls.tmpdir / "reports"),
            mock.patch.object(config, "DB_PATH", cls.tmpdir / "autolab_test.sqlite3"),
        ]
        for patch in cls._patches:
            patch.start()
        cls.task = TaskSpec.from_dict(config.load_task_config("t1_demo")).validate()

    @classmethod
    def tearDownClass(cls) -> None:
        for patch in cls._patches:
            patch.stop()
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def _run(self, plan_reply=None, plan_error=None, attr_reply=None, attr_error=None, rounds=1):
        plan_side = mock.Mock(side_effect=plan_error) if plan_error else mock.Mock(return_value=plan_reply)
        attr_side = mock.Mock(side_effect=attr_error) if attr_error else mock.Mock(return_value=attr_reply)
        with mock.patch.object(planner, "chat_json", plan_side), mock.patch.object(
            attributor, "chat_json", attr_side
        ):
            return run_agent(
                self.task,
                use_llm=True,
                max_rounds=rounds,
                recipe_lookup=False,
                transfer_lookup=False,
            )

    def test_non_compliant_llm_plan_falls_back_to_template(self) -> None:
        summary = self._run(plan_reply={"ops": [], "model": {"kind": "svm", "params": {}}})
        events = read_trace(summary["trace_path"])
        nodes = [event["node"] for event in events]
        self.assertIn("plan_fallback", nodes)
        self.assertEqual(summary["planner_source"], "template")
        plan_event = next(event for event in events if event["node"] == "plan")
        self.assertEqual(plan_event["output"]["plan"]["source"], "template")

    def test_llm_plan_is_used_when_valid(self) -> None:
        reply = {
            "ops": [{"op": "impute_median", "params": {}}, {"op": "scale_standard", "params": {}}],
            "model": {"kind": "logistic_gd", "params": {"lr": 0.05, "epochs": 300, "l2": 0.001}},
            "notes": "补中位数并标准化",
        }
        summary = self._run(plan_reply=reply, rounds=2)
        self.assertEqual(summary["planner_source"], "llm")
        plan_event = next(
            event for event in read_trace(summary["trace_path"]) if event["node"] == "plan"
        )
        self.assertEqual(plan_event["output"]["plan"]["source"], "llm")

    def test_non_compliant_llm_attribution_falls_back_to_rules(self) -> None:
        # 回归树模型在分类任务里合法，但把算子写成不存在的名字必然过不了白名单
        bad_fix = {"add_ops": [{"op": "not_in_whitelist", "params": {}}]}
        summary = self._run(
            plan_error=ValidationError("no llm"),
            attr_reply={
                "causes": [
                    {"cause": "feature_scale", "confidence": 0.9, "evidence": "量纲差异", "fix": bad_fix}
                ],
                "summary": "缺标准化",
            },
        )
        events = read_trace(summary["trace_path"])
        nodes = [event["node"] for event in events]
        self.assertIn("attribute_fallback", nodes)
        self.assertEqual(summary["attribution_source"], "rules")
        attribute = next(event for event in events if event["node"] == "attribute")
        self.assertEqual(attribute["output"]["source"], "rules")
        self.assertTrue(attribute["output"]["causes"])

    def test_valid_llm_attribution_is_used(self) -> None:
        summary = self._run(
            plan_error=ValidationError("no llm"),
            attr_reply={
                "causes": [
                    {
                        "cause": "feature_scale",
                        "confidence": 0.88,
                        "evidence": "特征量纲差异大，线性模型收敛慢",
                        "fix": {"add_ops": [{"op": "scale_standard", "params": {}}]},
                    }
                ],
                "summary": "缺标准化导致指标偏低",
            },
        )
        events = read_trace(summary["trace_path"])
        attribute = next(event for event in events if event["node"] == "attribute")
        self.assertEqual(summary["attribution_source"], "llm")
        self.assertEqual(attribute["tool"], "attributor.llm")
        self.assertEqual(attribute["output"]["source"], "llm")
        self.assertIn("缺标准化", attribute["output"]["summary"])
        self.assertEqual(attribute["output"]["causes"][0]["cause"], "feature_scale")


class LlmAttributionValidationTest(unittest.TestCase):
    def _fake(self):
        class Result:
            status = "ok"
            error = ""
            train_metric = 0.9
            seconds = 1.0
            holdout = None

            def metric_mean(self, name):
                return 0.6

        profile = {
            "n_rows": 120,
            "n_features": 3,
            "feature_types": {"numeric": ["a", "b", "c"], "categorical": []},
            "missing_ratio": 0.0,
            "target": {"type": "binary", "mean": 0.5, "std": 0.5},
            "warnings": [],
        }
        task = TaskSpec.from_dict(config.load_task_config("t1_demo")).validate()
        plan = planner.template_plan(profile, task)
        # class_weight 只有线性模型支持；这里换成 logistic_gd 以验证"合法修复可被采纳"
        plan.model = {"kind": "logistic_gd", "params": {"lr": 0.05, "epochs": 300, "l2": 0.001}}
        return profile, plan, Result(), task

    def _call(self, reply):
        profile, plan, result, task = self._fake()
        with mock.patch.object(attributor, "chat_json", return_value=reply):
            return llm_attribute(profile, plan, result, task)

    def test_unknown_cause_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            self._call(
                {
                    "causes": [
                        {"cause": "made_up", "confidence": 0.9, "evidence": "x", "fix": {"add_ops": []}}
                    ]
                }
            )

    def test_attribution_without_actionable_fix_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            self._call(
                {"causes": [{"cause": "underfit", "confidence": 0.7, "evidence": "训练折也低", "fix": {}}]}
            )

    def test_malformed_fix_shape_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            self._call(
                {
                    "causes": [
                        {
                            "cause": "feature_scale",
                            "confidence": 0.8,
                            "evidence": "量纲差异",
                            "fix": {"add_ops": 42},
                        }
                    ]
                }
            )

    def test_string_op_is_normalized(self) -> None:
        causes, _ = self._call(
            {
                "causes": [
                    {
                        "cause": "feature_scale",
                        "confidence": 0.7,
                        "evidence": "量纲差异",
                        "fix": {"add_ops": "scale_standard"},
                    }
                ]
            }
        )
        self.assertEqual(causes[0].fix["add_ops"][0]["op"], "scale_standard")

    def test_conflicting_fixes_are_merged_safely(self) -> None:
        causes, _ = self._call(
            {
                "causes": [
                    {
                        "cause": "underfit",
                        "confidence": 0.7,
                        "evidence": "训练折也未达标",
                        "fix": {
                            "set_model": {"kind": "random_forest", "params": {}},
                            "set_model_params": {"n_estimators": 300},
                        },
                    },
                    {
                        "cause": "overfit",
                        "confidence": 0.5,
                        "evidence": "训练验证差距大",
                        "fix": {"set_model_params": {"l2": 0.01}},
                    },
                ]
            }
        )
        self.assertEqual([cause.cause for cause in causes], ["underfit"])
        _, plan, _, task = self._fake()
        merged = attributor.replan(plan, causes, task)
        self.assertEqual(validate_plan(merged, task), [])

    def test_null_remove_ops_is_tolerated(self) -> None:
        causes, _ = self._call(
            {
                "causes": [
                    {
                        "cause": "feature_scale",
                        "confidence": 0.7,
                        "evidence": "量纲差异",
                        "fix": {"add_ops": ["scale_standard"], "remove_ops": None},
                    }
                ]
            }
        )
        self.assertEqual(causes[0].fix["add_ops"][0]["op"], "scale_standard")

    def test_unknown_model_params_are_pruned(self) -> None:
        causes, _ = self._call(
            {
                "causes": [
                    {
                        "cause": "underfit",
                        "confidence": 0.7,
                        "evidence": "训练折也低",
                        "fix": {
                            "set_model": {"kind": "random_forest", "params": {}},
                            "set_model_params": {"model": "x", "params": {}, "n_estimators": 300},
                        },
                    }
                ]
            }
        )
        params = causes[0].fix["set_model_params"]
        self.assertEqual(params, {"n_estimators": 300})

    def test_replan_does_not_mutate_input_plan(self) -> None:
        _, plan, _, task = self._fake()
        before = json.dumps(plan.to_dict(), ensure_ascii=False, sort_keys=True)
        attributor.replan(
            plan,
            [
                Attribution(
                    cause="overfit",
                    confidence=0.5,
                    evidence="训练验证差距大",
                    fix={"set_model_params": {"max_depth": 10}},
                )
            ],
            task,
        )
        after = json.dumps(plan.to_dict(), ensure_ascii=False, sort_keys=True)
        self.assertEqual(before, after, "归因补丁必须打在副本上，不能改动当前合法方案")

    def test_accuracy_policy_strips_balanced(self) -> None:
        profile, plan, result, task = self._fake()
        plan.model["params"]["class_weight"] = "balanced"
        with mock.patch.object(
            attributor,
            "chat_json",
            return_value={
                "causes": [
                    {
                        "cause": "underfit",
                        "confidence": 0.7,
                        "evidence": "训练折也低",
                        "fix": {"set_model_params": {"epochs": 600, "class_weight": "balanced"}},
                    }
                ],
                "summary": "容量不足",
            },
        ):
            causes, _ = llm_attribute(profile, plan, result, task)
        ids = [cause.cause for cause in causes]
        self.assertIn("metric_weight_mismatch", ids)
        self.assertTrue(all("balanced" not in str(cause.fix) for cause in causes))
        merged = attributor.replan(plan, causes, task)
        self.assertEqual(validate_plan(merged, task), [])

    def test_valid_attribution_is_sorted_by_confidence(self) -> None:
        causes, summary = self._call(
            {
                "causes": [
                    {
                        "cause": "underfit",
                        "confidence": 0.4,
                        "evidence": "训练折也低",
                        "fix": {"set_model_params": {"epochs": 600}},
                    },
                    {
                        "cause": "feature_scale",
                        "confidence": 0.8,
                        "evidence": "量纲差异",
                        "fix": {"add_ops": [{"op": "scale_standard", "params": {}}]},
                    },
                ],
                "summary": "先标准化再提升容量",
            }
        )
        self.assertEqual([cause.cause for cause in causes], ["feature_scale", "underfit"])
        self.assertIn("标准化", summary)


if __name__ == "__main__":
    unittest.main()
