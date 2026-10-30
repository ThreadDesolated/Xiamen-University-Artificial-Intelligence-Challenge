"""沙箱子进程入口。

    python -m autolab.sandbox.runner --spec <spec.json> --out <result.json>

spec.json:
    {"task": {TaskSpec 字段}, "plan": {ExperimentPlan 字段}}
"""
from __future__ import annotations

import argparse
import json
import traceback


def main() -> int:
    from . import netguard

    netguard.install()  # 必须在任何重活之前

    parser = argparse.ArgumentParser(description="AutoLab sandbox runner")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result: dict = {"status": "ok", "metrics": {}, "seconds": 0.0, "error": ""}
    try:
        from ..datasets.loader import load_dataframe
        from ..evaluation.protocol import run_cv
        from ..schemas import ExperimentPlan, TaskSpec, ValidationError

        with open(args.spec, encoding="utf-8") as f:
            spec = json.load(f)

        task = TaskSpec.from_dict(spec["task"]).validate()
        plan = ExperimentPlan.from_dict(spec["plan"])
        df = load_dataframe(task.data_path, task.target)
        payload = run_cv(
            plan,
            df,
            task_type=task.task_type,
            target=task.target,
            metric=task.metric,
            n_splits=task.n_splits,
            seed=task.seed,
            features=task.features,
        )
        result.update(payload)
        result["status"] = "ok"
        result["network_blocked"] = netguard.is_network_blocked()
    except Exception as exc:  # noqa: BLE001 - 沙箱必须把异常转成结构化结果
        from ..schemas import ValidationError

        result = {
            "status": "invalid" if isinstance(exc, ValidationError) else "error",
            "metrics": {},
            "seconds": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback_tail": traceback.format_exc()[-4000:],
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({"status": result["status"], "error": result.get("error", "")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
