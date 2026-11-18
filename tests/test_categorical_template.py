import unittest
import numpy as np
import pandas as pd
from autolab import config
from autolab.profile import profile_dataframe
from autolab.agent.planner import template_plan
from autolab.schemas import TaskSpec
from autolab.sandbox.operators import Pipeline
from autolab.sandbox.validate import validate_plan

class CategoricalTemplateTest(unittest.TestCase):
    def test_template_encodes_strings_and_handles_unseen_category(self):
        task = TaskSpec.from_dict(config.load_task_config('t1_adult')).validate()
        frame = pd.DataFrame({'age': [20, 30, 40], 'workclass': ['Private', 'Public', 'Private'], 'income': [0, 1, 0]})
        plan = template_plan(profile_dataframe(frame, 'income'), task)
        self.assertIn('onehot', plan.op_names())
        pipe = Pipeline(plan.ops).fit(frame.drop(columns='income'))
        result = pipe.transform(pd.DataFrame({'age': [25], 'workclass': ['Unseen']}))
        self.assertEqual(result.shape, (1, 3))
        self.assertTrue(np.isfinite(result).all())
