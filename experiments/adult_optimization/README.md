# Adult optimization

Same 1000 rows, seed 42, five stratified folds. Six candidates declared before evaluation.

|Candidate|Model|Accuracy mean ± std|Macro F1|Train accuracy|Seconds|
|---|---|---|---|---|---|
|A|logistic_gd|0.7910 ± 0.0146|0.7523|0.8340|0.596|
|B|logistic_gd|0.8320 ± 0.0150|0.7641|0.8758|0.495|
|C|logistic_gd|0.8320 ± 0.0172|0.7649|0.8773|0.831|
|D|logistic_gd|0.8380 ± 0.0133|0.7710|0.8755|0.933|
|E|random_forest|0.8480 ± 0.0087|0.7696|0.8660|7.53|
|F|gradient_boosting|0.8570 ± 0.0129|0.7926|0.8925|1.726|

Winner: F. Holdout (5000 rows): {'accuracy': 0.8504, 'f1_macro': 0.7750344004883318}.
Holdout evaluated only after selecting the winner. It is not the official Adult benchmark test. No holdout-based tuning.
All plans and fold scores are in results.json. Reproduce from repository root: `.venv/Scripts/python.exe -m scripts.optimize_adult`.

## 手动运行

在仓库根目录 PowerShell 中执行：

```powershell
$env:PYTHONUTF8 = "1"
.\.venv\Scripts\python.exe -m pip install scikit-learn
.\.venv\Scripts\python.exe -m autolab run --task t1_adult_optimized --no-recipe
```

本次完整闭环七轮：52.3% → 79.1% → 83.2% → 83.2% → 83.8% → 84.8% → 85.7%。第三、四轮参数不同，尽管平均分相同，不是重复方案。
原任务配置仍限制三轮；优化配置允许七轮，目标仍为 85%，预算仍为 120 秒。
最佳方案可直接复现：

```powershell
.\.venv\Scripts\python.exe -m autolab run --task t1_adult --plan experiments/adult_optimization/best_plan.json
```

结果来自一次固定种子的实验。五折均值 85.7%，折间标准差约 1.29 个百分点；不能据此保证未来数据稳定超过 85%。留出集仅在六组候选选择完成后评估，未据此继续调参。
验证：41 项 unittest 全部通过。新增测试覆盖候选合法性、重复方案拦截、最佳方案及对应结果导出、树模型确定性。
