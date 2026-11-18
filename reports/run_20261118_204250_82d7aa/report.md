# AutoLab 实验报告：UCI Adult 收入二分类（固定子集 1000 行）

- 运行 ID：`run_20261118_204250_82d7aa`
- 任务类型：binary_classification
- 结论：**未达标**（accuracy = 0.7910，阈值 0.85，越高越好）
- 轮次：3 / 3，规划来源：template，配方命中：False
- 生成时间：2026-11-18 20:42:53

## 一、迭代过程

| 轮次 | 方案来源 | 指标（均值±标准差） | 是否达标 | 耗时 |
| --- | --- | --- | --- | --- |
| 1 | template | 0.5230 ± 0.0378 | 否 | 0.21s |
| 2 | attribution | 0.7910 ± 0.0146 | 否 | 0.48s |
| 3 | attribution | 0.7910 ± 0.0146 | 否 | 0.47s |

## 二、失败归因（如有）

| 轮次 | 原因 | 置信度 | 证据 | 修复动作 |
| --- | --- | --- | --- | --- |
| 1 | class_imbalance | 0.80 | 类别失衡比 3.0，模型未启用 class_weight=balanced | set_model=logistic_gd |
| 1 | feature_scale | 0.60 | 未标准化特征：距离/梯度类模型对量纲敏感 | add=scale_standard |
| 1 | underfit | 0.60 | 训练折指标也只有 0.5162，模型容量/参数明显不足 | set_model=logistic_gd |
| 2 | unknown_shortfall | 0.30 | 指标 0.7910 未达阈值 0.85，规则库未命中明确原因 | add=scale_standard；set_model=logistic_gd |
| 3 | unknown_shortfall | 0.30 | 指标 0.7910 未达阈值 0.85，规则库未命中明确原因 | add=scale_standard；set_model=logistic_gd |

## 三、最终实验方案

```json
{
  "plan_id": "plan_20261118_204252_7ddeb5",
  "ops": [
    {
      "op": "onehot",
      "params": {
        "max_categories": 20
      }
    },
    {
      "op": "scale_standard",
      "params": {}
    }
  ],
  "model": {
    "kind": "logistic_gd",
    "params": {
      "lr": 0.1,
      "epochs": 400,
      "l2": 0.001,
      "class_weight": "balanced"
    }
  },
  "eval": {
    "n_splits": 5,
    "seed": 42
  },
  "source": "attribution",
  "notes": "unknown_shortfall(0.30)"
}
```

## 四、数据画像摘要

- 样本数 1000，特征数 14，缺失比例 0.0000
- 特征类型：{"numeric": ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"], "categorical": ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]}
- 目标分布：{"name": "income", "type": "binary_classification", "n_classes": 2, "class_counts": {"0": 750, "1": 250}, "imbalance_ratio": 3.0, "has_nan": false}
- 画像指纹：cb47589ffb98

## 五、复现方式

```bash
# 方式一：按保存的方案原样复跑（推荐）
python -m autolab run --task t1_adult --plan plan.json
# 方式二：回放本轮的完整调用链
python -m autolab replay --run-id run_20261118_204250_82d7aa
```

## 六、环境信息

```json
{
  "python": "3.14.4",
  "platform": "Windows-11-10.0.26200-SP0",
  "numpy": "2.5.5",
  "pandas": "3.1.1",
  "seed": 42,
  "model": "deepseek-chat",
  "llm_enabled": false,
  "git_commit": "55f8075",
  "generated_at": "2026-11-18 20:42:53"
}
```
