import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from autolab import config
from autolab.agent.graph import run_agent
from autolab.agent.search import candidates
from autolab.schemas import TaskSpec
from autolab.sandbox.operators import build_model
from autolab.sandbox.validate import validate_plan

class OptimizationTests(unittest.TestCase):
    def test_candidates_valid_and_unique(self):
        task=TaskSpec.from_dict(config.load_task_config('t1_adult')).validate()
        plans=candidates(task)
        self.assertEqual(len(plans),len({p.fingerprint() for p in plans}))
        for p in plans:
            self.assertEqual(validate_plan(p,task),[])

    def test_tree_predictions_reproduce(self):
        X=np.random.default_rng(42).normal(size=(80,4)); y=(X[:,0]>0).astype(int)
        for kind in ('random_forest','gradient_boosting'):
            spec={'kind':kind,'params':{'n_estimators':10,'max_depth':2,'min_samples_leaf':2}}
            first=build_model(spec).fit(X,y).predict(X)
            np.testing.assert_array_equal(first,build_model(spec).fit(X,y).predict(X))

    def test_no_repeats_and_export_best_not_last(self):
        task=TaskSpec.from_dict(config.load_task_config('t1_demo')).validate()
        task.threshold=.99
        scores=iter([.80,.70,.60]); seen=[]
        def execute(plan,*args):
            seen.append(plan.fingerprint()); score=next(scores)
            return {'status':'ok','metrics':{'accuracy':{'name':'accuracy','mean':score,'std':0,'values':[score],'n':1}},'train_metric':score,'seconds':.01}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(config,'TRACE_DIR',root/'traces'), patch.object(config,'REPORT_DIR',root/'reports'), patch.object(config,'DB_PATH',root/'db.sqlite3'), patch('autolab.agent.graph.execute_plan',side_effect=execute), patch('autolab.agent.graph.replan',side_effect=lambda p,*_:copy.deepcopy(p)):
                result=run_agent(task,use_llm=False,recipe_lookup=False,max_rounds=3)
                exported=json.loads((Path(result['report_dir'])/'result.json').read_text(encoding='utf-8'))
                self.assertEqual(exported['metrics']['accuracy']['mean'],.80)
        self.assertEqual(result['status'],'failed',result.get('error'))
        self.assertEqual(len(seen),3)
        self.assertEqual(len(set(seen)),3)
        self.assertEqual(result['final_metric'],.80)
        self.assertEqual(result['last_metric'],.60)
        self.assertEqual(result['plan'],result['rounds'][0]['plan'])
