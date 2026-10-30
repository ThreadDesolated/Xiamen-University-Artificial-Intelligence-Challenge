# AutoLab —— 科研实验自动化智能体

> 第六届厦门大学人工智能挑战赛 · 赛道二（AI 智能体创新赛）
> 队伍：**泥虾** ｜ 队长：陈传轩 ｜ 队员：张一鸣、陈景祥、张谋谆

**仓库**：[ThreadDesolated/Xiamen-University-Artificial-Intelligence-Challenge](https://github.com/ThreadDesolated/Xiamen-University-Artificial-Intelligence-Challenge)
克隆：`git clone https://github.com/ThreadDesolated/Xiamen-University-Artificial-Intelligence-Challenge.git`

AutoLab 让 Agent 像研究生一样做实验：给它一句话目标（例如"把这个数据集验证准确率做到 0.85 以上，预算 2 分钟"），
它会自主完成 **数据画像 → 实验方案规划 → 沙箱执行 → 指标评估 → 失败归因 → 重规划 → 配方沉淀与复现包交付**，
全过程写入可回放的调用链 trace。

## 核心特性

| 特性 | 说明 |
|---|---|
| 指标即奖励的闭环 | 用真实评测指标（K 折交叉验证均值±标准差）作为奖励信号；达标/继续迭代由代码规则与预算决定，而不是模型"自称完成" |
| 失败归因驱动重规划 | 把"实验失败"当一等公民：输出结构化失败原因和修复动作，再生成下一轮方案 |
| 跨任务配方迁移 | 成功方案以"任务画像 + 方案 JSON"沉淀到 SQLite 配方库，新任务先检索复用 |
| 受限动作空间 | 方案是一份 JSON，只允许白名单算子/模型；做 Schema 与白名单双重校验，非法方案回退模板方案 |
| 安全沙箱 | 子进程隔离、超时即杀、socket 级禁网、不依赖 Docker |
| 可复现交付 | 每次运行自动导出 trace、报告、配置、环境信息与一键复现脚本 zip |

## 快速开始（零额外依赖：只用 numpy + pandas）

```bash
pip install -r requirements-core.txt     # 如已装好 numpy/pandas 可跳过
python -m autolab demo
```

`demo` 使用内置合成任务（缺失值 + 类别失衡的高维表格分类），完整走一遍"弱 baseline 失败 → 归因 → 重规划 → 达标 → 导出复现包"：

```
[round 1] plan=template     metric=0.7xxx (<0.85) attribution=[missing_values_unhandled, class_imbalance, feature_scale]
[round 2] plan=attribution  metric=0.8xxx (>=0.85) OK
trace:  traces/<run_id>.jsonl
report: reports/<run_id>.zip
```

装齐完整依赖后：

```bash
python -m autolab prepare --task t1_adult        # 下载并准备固定子集（UCI Adult 1k 行）
python -m autolab run --task configs/tasks/t1_adult.json
python -m autolab serve --port 8000              # FastAPI + SSE 调用链面板后端（W2 起接前端）
python -m autolab replay traces/<run_id>.jsonl   # 回放一次运行（演示断网兜底用）
python -m unittest discover -s tests -v          # 跑测试
```

## 环境安装

- Python 3.10+（开发机 3.12）
- 最小运行：`pip install -r requirements-core.txt`（numpy、pandas）
- 完整开发：`pip install -r requirements.txt`（W2 起需要 scikit-learn / flaml / fastapi / langgraph）
- 大模型（可选）：`cp .env.example .env` 后填 `DEEPSEEK_API_KEY`。
  未配置 Key 时自动退化为 **模板规划器 + 规则归因**——这是现场断网/限流时的兜底路径，永远可用。

## 目录结构

```
.
├── autolab/
│   ├── schemas.py            # 任务 / 画像 / 方案 / 结果 / trace 数据结构（与 docs/接口契约.md 一一对应）
│   ├── profile.py            # 数据画像
│   ├── config.py             # 路径与运行配置
│   ├── cli.py                # 命令行入口（demo / run / prepare / serve / replay）
│   ├── agent/                # 编排层
│   │   ├── graph.py          #   闭环主链路：profile → plan → execute → evaluate → decide → attribute → replan
│   │   ├── planner.py        #   模板规划器（离线保底） + LLM 规划器（JSON 约束）
│   │   ├── attributor.py     #   失败归因规则 + 重规划
│   │   └── prompts.py        #   提示词
│   ├── sandbox/              # 执行层
│   │   ├── validate.py       #   Schema + 白名单校验
│   │   ├── operators.py      #   白名单算子/模型实现
│   │   ├── executor.py       #   子进程执行器（超时/禁网/隔离）
│   │   ├── runner.py         #   沙箱子进程入口（跑交叉验证、写 result.json）
│   │   └── netguard.py       #   socket 级禁网
│   ├── evaluation/           # 评测层
│   │   ├── metrics.py        #   accuracy / f1_macro / rmse / mae
│   │   └── protocol.py       #   固定种子 K 折交叉验证协议
│   ├── tracing/              # 可观测层
│   │   ├── logger.py         #   JSONL trace 写入 + 订阅（SSE 用）
│   │   └── replay.py         #   trace 回放
│   ├── store/db.py           # 配方库 + 归因知识库 + 运行记录（SQLite）
│   ├── report.py             # 实验报告 + 复现包导出
│   ├── datasets/             # 数据集准备与加载（合成/真实）
│   └── serving/server.py     # FastAPI + SSE 服务
├── configs/
│   ├── whitelist_ops.json    # 动作空间白名单（算子、模型、参数类型/范围）
│   └── tasks/*.json          # 任务配置（数据路径、目标指标、预算、轮数）
├── docs/                     # 设计方案、接口契约、演示剧本
├── tests/                    # unittest（无需 pytest）
├── traces/                   # 运行时 trace（git 忽略）
├── reports/                  # 运行报告与复现包（git 忽略）
└── data/                     # 数据集缓存（git 忽略内容）
```

## 任务配置

`configs/tasks/*.json` 的字段见 [docs/接口契约.md](docs/接口契约.md#1-taskspec任务配置)：

```json
{
  "task_id": "t1_adult",
  "name": "UCI Adult 收入二分类（固定子集）",
  "task_type": "binary_classification",
  "data_path": "data/processed/t1_adult.csv",
  "target": "income",
  "metric": "accuracy",
  "threshold": 0.85,
  "higher_is_better": true,
  "n_splits": 5,
  "seed": 42,
  "budget_seconds": 120,
  "max_rounds": 3
}
```

## 白名单动作空间

方案 JSON 只能引用 `configs/whitelist_ops.json` 中声明的算子、模型与参数范围，例如：

```json
{
  "ops": [
    {"op": "impute_median", "params": {}},
    {"op": "scale_standard", "params": {}},
    {"op": "onehot", "params": {"max_categories": 20}}
  ],
  "model": {"kind": "logistic_gd", "params": {"lr": 0.1, "epochs": 300, "class_weight": "balanced"}},
  "eval": {"n_splits": 5, "seed": 42}
}
```

非法方案（未知算子 / 参数越界 / 超长）会被 `sandbox/validate.py` 拦下，并自动回退到模板方案——保证现场演示不会因为模型幻觉直接崩掉。

## 开发计划（W1–W5，2026-11-02 → 12-05）

| 周 | 目标 | 验收 |
|---|---|---|
| W1 | 骨架跑通：`demo` 全链路可跑、trace 可回放、接口契约冻结 | `python -m autolab demo` 通过 + 测试全绿 |
| W2 | 表格域真数据：T1 自动达标（≤2 轮、<30s）、归因库/配方库落库 | `run --task t1_adult` 达标 |
| W3 | 图像域 + 时序域 + 配方迁移证据（T3 首轮命中） | 三任务全跑通 |
| W4 | 指标实测校准、断网回放演练、3 次彩排、视频/报告 | 断网也能完整演示 |
| W5 | 彩排与材料定稿，12-05 初赛 | 现场 3 分钟脚本零失误 |

## 团队分工

| 成员 | 角色 | 主责 |
|---|---|---|
| 陈传轩（队长） | 项目负责人 / Agent 与算法 | 架构与接口、编排与提示词、规划/归因策略、评测协议、语音入口、答辩主讲 |
| 张一鸣 | 工程 / 后端与执行 | FastAPI+SSE、沙箱执行器、SQLite 配方/归因库、trace 采集回放、部署测试 |
| 陈景祥 | 算法与实验 | 数据集与指标脚本、算子库、baseline 与预置失败案例、归因规则与回归 |
| 张谋谆 | 可视化与交付 | trace 面板前端、复现包/README、演示视频与 PPT、现场第二操作手 |

## 演示与兜底

- 现场演示 3 分钟剧本见 [docs/演示剧本.md](docs/演示剧本.md)
- **回放兜底**：所有演示任务提前跑通并把 trace 缓存下来；现场断网/API 限流时 `python -m autolab replay <trace.jsonl>` 一键切回放
- **2+1 任务边界**：表格 + 时序 2 个任务保底，图像域作加分项
- 演示全程录屏，赛前一周冻结功能

## 说明

- 作品为本届赛事新设计开发，未使用往届作品原样参赛，也不属于科研项目包装。
- 使用的开源库与公开数据集均遵循其许可证，来源、版本与引用见技术报告与 `requirements.txt`。
- 设计方案全文见 [docs/设计方案_泥虾队.docx](docs/设计方案_泥虾队.docx)。
