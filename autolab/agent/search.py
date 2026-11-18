"""Bounded, deterministic candidates; evaluation protocol stays fixed."""
import copy
import importlib.util
from ..schemas import ExperimentPlan


def candidates(task):
    if task.task_type not in ("binary_classification", "multiclass_classification"):
        return []
    ops = [{"op": "onehot", "params": {"max_categories": 20}}, {"op": "scale_standard", "params": {}}]
    plans = []
    for weight, epochs, l2 in [("balanced",400,.001),("none",400,.001),("none",1000,.001),("none",1000,.01)]:
        plans.append(ExperimentPlan(ops=copy.deepcopy(ops), model={"kind":"logistic_gd", "params":{"lr":.1,"epochs":epochs,"l2":l2,"class_weight":weight}}, eval={"n_splits":task.n_splits,"seed":task.seed}, source="search"))
    if importlib.util.find_spec("sklearn"):
        for kind, depth in [("random_forest",10),("gradient_boosting",2)]:
            plans.append(ExperimentPlan(ops=copy.deepcopy(ops),model={"kind":kind,"params":{"n_estimators":100,"max_depth":depth,"min_samples_leaf":5}},eval={"n_splits":task.n_splits,"seed":task.seed},source="search"))
    return plans


def next_untried(task, seen):
    return next((p for p in candidates(task) if p.fingerprint() not in seen), None)
