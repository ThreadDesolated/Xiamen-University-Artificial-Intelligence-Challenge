"""Reproduce six predeclared Adult candidates, then evaluate one winner on unused rows."""
import hashlib
import json
import subprocess
import time
from pathlib import Path
import numpy as np
import pandas as pd
from autolab import config
from autolab.agent.search import candidates
from autolab.datasets.prepare import ADULT_COLUMNS
from autolab.evaluation.protocol import stratified_folds
from autolab.evaluation.metrics import compute
from autolab.sandbox.executor import execute_plan
from autolab.sandbox.operators import Pipeline, build_model
from autolab.sandbox.validate import validate_plan
from autolab.schemas import TaskSpec


def main():
    out = config.ROOT / 'experiments/adult_optimization'
    out.mkdir(parents=True, exist_ok=True)
    task = TaskSpec.from_dict(config.load_task_config('t1_adult')).validate()
    df = pd.read_csv(config.ROOT / task.data_path)
    results = []
    folds = stratified_folds(df[task.target], task.n_splits, task.seed)
    (out/'folds.json').write_text(json.dumps([f.tolist() for f in folds]),encoding='utf-8')
    started = time.monotonic()
    for label, plan in zip('ABCDEF', candidates(task)):
        assert not validate_plan(plan, task)
        if time.monotonic() - started > task.budget_seconds:
            break
        result = execute_plan(plan, task, 'adult_optimization_'+label, 1)
        row = {'label':label,'plan':plan.to_dict(),'result':result}
        results.append(row)
        print(label,plan.model,result.get('metrics'),flush=True)
    valid = [r for r in results if r['result']['status']=='ok']
    best = max(valid,key=lambda r:r['result']['metrics']['accuracy']['mean'])
    (out/'best_plan.json').write_text(json.dumps(best['plan'],indent=2),encoding='utf-8')
    # Candidate selection is finished before this holdout is constructed or scored.
    raw = pd.read_csv(config.RAW_DIR/'adult.data',names=ADULT_COLUMNS,skipinitialspace=True,na_values='?')
    raw['income']=raw['income'].str.strip().map({'<=50K':0,'>50K':1})
    raw=raw.dropna(subset=['income'])
    for c in ('workclass','occupation','native_country'):
        raw[c]=raw[c].fillna('__missing__')
    features=[c for c in df if c != task.target]
    # Exclude duplicate feature rows, even if labels differ.
    def keys(frame):
        return frame[features].astype(str).agg('|'.join,axis=1)
    excluded=set(keys(df))
    pool=raw.loc[~keys(raw).isin(excluded)].drop_duplicates(subset=features)
    holdout=pool.sample(n=min(5000,len(pool)),random_state=42).reset_index(drop=True)
    assert not set(keys(holdout)) & excluded
    pipe=Pipeline(best['plan']['ops']).fit(df[features])
    model=build_model(best['plan']['model']).fit(pipe.transform(df[features]),df[task.target].to_numpy())
    pred=model.predict(pipe.transform(holdout[features]))
    scores={name:compute(name,holdout[task.target].to_numpy(),pred) for name in ('accuracy','f1_macro')}
    holdout.to_csv(config.PROCESSED_DIR/'t1_adult_holdout.csv',index=False)
    payload={'candidates':results,'winner':best['label'],'holdout':{'n':len(holdout),'metrics':scores,'positive_ratio':float(holdout[task.target].mean()),'note':'Unused, feature-deduplicated rows from original Adult training source; not official benchmark test.'},'data_sha256':hashlib.sha256((config.ROOT/task.data_path).read_bytes()).hexdigest(),'holdout_sha256':hashlib.sha256((config.PROCESSED_DIR/'t1_adult_holdout.csv').read_bytes()).hexdigest(),'seed':42,'n_splits':5}
    (out/'results.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Adult optimization','', 'Same 1000 rows, seed 42, five stratified folds. Six candidates declared before evaluation.','', '|Candidate|Model|Accuracy mean ± std|Macro F1|Train accuracy|Seconds|','|---|---|---|---|---|---|']
    for r in valid:
        m=r['result']['metrics']; lines.append(f"|{r['label']}|{r['plan']['model']['kind']}|{m['accuracy']['mean']:.4f} ± {m['accuracy']['std']:.4f}|{m['f1_macro']['mean']:.4f}|{r['result']['train_metric']:.4f}|{r['result']['seconds']}|")
    lines += ['',f"Winner: {best['label']}. Holdout ({len(holdout)} rows): {scores}.",'Holdout evaluated only after selecting the winner. It is not the official Adult benchmark test. No holdout-based tuning.','All plans and fold scores are in results.json. Reproduce from repository root: `.venv/Scripts/python.exe -m scripts.optimize_adult`.']
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'environment-lock.txt').write_text(subprocess.check_output([__import__('sys').executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    print('WINNER',best['label'],'HOLDOUT',scores,flush=True)

if __name__=='__main__':
    main()
