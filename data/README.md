# 数据目录说明

- `data/raw/`：原始下载文件（git 忽略）；
- `data/processed/`：预处理后的 CSV/NPZ，供任务配置引用（git 忽略）。

## 离线演示（零下载）

`configs/tasks/t1_demo.json` 使用 `synthetic:t1`，由 `autolab/datasets/synthetic.py`
用固定种子现场生成，无需任何数据文件：

```bash
python -m autolab demo --no-recipe
```

## 真实数据集（W2 起）

```bash
python -m autolab prepare --task t1_adult        # UCI Adult（1000 子集）     → data/processed/t1_adult.csv
python -m autolab prepare --task t2_digits       # UCI Optdigits（2000 子集）  → data/processed/t2_digits.csv
python -m autolab prepare --task t3_timeseries   # UCI AirQuality（滑窗序列）  → data/processed/t3_timeseries.csv
python -m autolab run --task t1_adult
```

说明：`prepare` 需要联网；下载失败时检查网络/镜像，或手动放置文件后重跑。
数据来源与许可证见各下载函数注释（均为公开数据集，仅用于赛事演示）。
