"""数据集：合成任务（离线兜底）+ 真实公开数据集准备。"""

from .loader import load_dataframe, describe_dataset
from .synthetic import make_t1

__all__ = ["load_dataframe", "describe_dataset", "make_t1"]
