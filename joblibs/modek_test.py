import pandas as pd
from autolab.evaluation.model_store import load_model, predict

payload = load_model(r"reports/<run_id>/model.joblib")
print(payload["meta"]["model_kind"], payload["meta"]["cv_mean"])

df = pd.read_csv("你的新数据.csv")
X = df.drop(columns=["income"])          # 列名/顺序要和训练时一致（目标列除外）
y_pred = predict(payload, X)             # 分类会自动映射回原始类别标签
print(y_pred[:10])