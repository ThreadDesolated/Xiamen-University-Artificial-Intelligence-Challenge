# API 接口调试文档

> 面向：前端（trace 面板）、联调同学、现场演示排查
> 服务端入口：[autolab/serving/server.py](../autolab/serving/server.py)
> 契约字段定义：[docs/接口契约.md](接口契约.md)

本文只讲「怎么连、怎么调、出错怎么查」。所有示例都可直接复制执行。

---

## 1. 启动服务

```powershell
# 仓库根目录
python -m autolab serve --port 8000
```

启动后：

| 地址 | 用途 |
|---|---|
| `http://127.0.0.1:8000/api/health` | 健康检查 |
| `http://127.0.0.1:8000/docs` | Swagger UI（可点着试） |
| `http://127.0.0.1:8000/openapi.json` | 机器可读的接口描述 |

> 依赖 Web 组件（fastapi / uvicorn / sse-starlette）：`pip install -r requirements.txt`。
> 没装也能用 CLI 跑实验，只是起不了服务。

---

## 2. 接口总表

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 存活 + 是否配置了 LLM Key |
| GET | `/api/tasks` | 列出 `configs/tasks/*.json` |
| GET | `/api/stats` | 配方库 / 归因库 / 运行记录行数 |
| GET | `/api/runs?limit=20` | 历史运行（SQLite，重启不丢） |
| POST | `/api/runs` | 发起一次运行（后台线程） |
| GET | `/api/runs/{run_id}` | 本次运行的内存态摘要 |
| GET | `/api/runs/{run_id}/trace?after=0` | 拉取 trace 事件（一次性） |
| GET | `/api/runs/{run_id}/report` | 取 `report.md` 原文 |
| GET | `/api/runs/{run_id}/stream` | **SSE 实时推流** |

`run_id` 由服务端生成，形如 `run_20261106_205014_0a402b`。

---

## 3. 逐个接口

### 3.1 健康检查

```powershell
curl.exe -sS http://127.0.0.1:8000/api/health
```

```json
{"ok": true, "time": 1793969398.56, "llm_enabled": true}
```

`llm_enabled` 为 `false` 说明 `.env` 里没读到 `DEEPSEEK_API_KEY`——此时**功能仍可用**，
规划器会自动退回模板方案。

### 3.2 发起运行

```powershell
curl.exe -sS -X POST http://127.0.0.1:8000/api/runs `
  -H "Content-Type: application/json" `
  -d "{\"task_id\":\"t1_demo\",\"use_llm\":false,\"recipe_lookup\":false}"
```

请求体字段：

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `task_id` | str | 必填 | `configs/tasks` 下的任务 id，如 `t1_demo` |
| `use_llm` | bool | `false` | 是否启用大模型规划器 |
| `max_rounds` | int \| null | `null` | 覆盖任务配置里的轮数上限 |
| `recipe_lookup` | bool | `true` | 是否检索历史配方（演示"失败→归因"要设 `false`） |

```json
{"run_id": "run_20261106_205014_0a402b", "status": "running", "trace": "/api/runs/run_20261106_205014_0a402b/trace"}
```

### 3.3 拉取 trace（一次性 / 轮询）

```powershell
curl.exe -sS "http://127.0.0.1:8000/api/runs/run_20261106_205014_0a402b/trace?after=5"
```

`after` 是**事件下标**（从 0 开始），返回 `events[after:]`。轮询式面板每次带上次长度即可。

### 3.4 报告

```powershell
curl.exe -sS "http://127.0.0.1:8000/api/runs/run_20261106_205014_0a402b/report"
```

返回 `text/markdown`；`404` 表示这一轮还没导出。

---

## 4. SSE 实时推流（重点）

### 4.1 管道格式

```
id: 6
data: {"run_id": "...", "seq": 6, "node": "replan", "tool": "attributor.replan", ...}

: ping
```

- 每个事件一帧：`id: <seq>` + `data: <JSON>`，以空行结束；
- 空闲超过 15 秒发一个 `: ping` 注释帧（EventSource 会忽略，代理不会掐线）；
- 收到 `"node": "done"` 后服务端主动结束连接。

### 4.2 命令行观察

```powershell
curl.exe -N "http://127.0.0.1:8000/api/runs/<run_id>/stream"
```

`-N` 关掉 curl 自己的缓冲，否则看起来像"卡住不动"。

只看事件类型：

```powershell
curl.exe -N "http://127.0.0.1:8000/api/runs/<run_id>/stream" | Select-String '^data: ' | ForEach-Object { ($_ -replace '^data: ','') | ConvertFrom-Json } | Select-Object seq,node,status
```

### 4.3 前端接入

```javascript
const es = new EventSource(`/api/runs/${runId}/stream`);
es.onmessage = (e) => {
  const event = JSON.parse(e.data);      // {seq, node, tool, input, output, duration_ms}
  console.log(event.seq, event.node);
  if (event.node === "done") es.close(); // 服务端也会关，双保险
};
es.onerror = () => { /* 浏览器会自动重连，并带上 Last-Event-ID */ };
```

### 4.4 断线续传

浏览器重连时自动带 `Last-Event-ID: 7`；也可手动指定：

```powershell
curl.exe -N -H "Last-Event-ID: 7" "http://127.0.0.1:8000/api/runs/<run_id>/stream"
curl.exe -N "http://127.0.0.1:8000/api/runs/<run_id>/stream?after=7"   # 等价
```

两种写法都已验证：从 `seq=8` 继续推，不重复也不漏。

### 4.5 稳定性约定（改动时别破坏）

| 行为 | 原因 |
|---|---|
| 响应头带 `Cache-Control: no-cache, no-transform`、`X-Accel-Buffering: no` | 防止 nginx / 代理把流缓冲成一次性返回 |
| 每 15s 一个 `: ping` | 防止空闲连接被中间设备掐断 |
| 增量读 trace（按字节偏移） | 事件攒到几百条也不会越来越慢 |
| `request: Request` 必须导入到**模块级** | 本文件有 `from __future__ import annotations`，endpoint 注解是字符串，FastAPI 只在模块命名空间解析；写成局部导入会被当成查询参数 → 前端一连就 422 |
| trace 未生成时返回 404 | 前端等 200ms 重试即可（`POST /api/runs` 返回后 trace 通常毫秒级就绪） |

---

## 5. 故障注入测试案例

「白名单拦截」「子进程超时强杀」这两条兜底，均有可复现的测试：

```powershell
python -m unittest tests.test_guard -v
```

| 案例 | 构造方式 | 期望结果 |
|---|---|---|
| 未知算子 | 方案里写 `{"op": "drop_all_tables"}` | `算子不在白名单` → 自动回退模板方案 → **回退后仍跑出指标** |
| 参数越界 | `{"kind": "logistic_gd", "params": {"epochs": 99999999}}` | `epochs: 超出范围 [50, 2000]` → 回退 |
| 未知模型 | `{"kind": "gpt_finetune_black_box"}` | `model.kind 不在白名单` |
| 超时强杀 | 40 折 × 2000 轮（正常约 5s），`AUTOLAB_SANDBOX_TIMEOUT=2` | `status="timeout"`，错误含"已强制终止"，**2 秒左右被掐断**，主流程继续归因 |
| 数据缺失 | 任务 `data_path` 指向不存在的文件 | 沙箱返回结构化 `status="error"`，不抛穿主流程 |

手工构造一次「非法方案」也很简单——绕开 Agent，直接把坏方案喂给执行器：

```powershell
python -c "from autolab.sandbox.validate import validate_plan; from autolab.schemas import ExperimentPlan; from autolab import config; from autolab.schemas import TaskSpec; t=TaskSpec.from_dict(config.load_task_config('t1_demo')).validate(); p=ExperimentPlan(ops=[{'op':'drop_all_tables','params':{}}], model={'kind':'nearest_centroid','params':{}}, eval={'n_splits':5,'seed':42}); print(validate_plan(p, t))"
```

超时预算可用环境变量临时收紧/放宽（也在测试里用到）：

```powershell
$env:AUTOLAB_SANDBOX_TIMEOUT = "5"     # 单位秒；不设则 max(30, budget_seconds + 20)
```

---

## 6. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `404 trace 尚未生成` | run 刚发起或 `run_id` 写错；等 200ms 重试 |
| `422 Unprocessable Content` | 请求体缺 `task_id`，或 endpoint 注解解析失败（见 4.5 最后两条） |
| 流"卡住"不动 | 客户端缓冲（curl 加 `-N`，前端别用 `fetch` 读全量）或代理缓冲（确认响应头） |
| `500` + `PydanticUserError` | 请求模型/`Request` 没定义在模块级 |
| `database is locked` | 历史版本连接未关闭已修复（`db.open_conn`），若复现请贴 trace |
| 端口占用 | `python -m autolab serve --port 8011` 换端口 |

---

## 7. 回归测试一览

```powershell
python -m unittest discover -s tests -v
```

| 文件 | 覆盖 |
|---|---|
| `tests/test_e2e.py` | 失败 → 归因 → 重规划 → 达标 → 配方复用 → 复现 |
| `tests/test_whitelist.py` | Schema + 白名单校验、回退 |
| `tests/test_guard.py` | 白名单拦截落地、超时强杀、异常结构化 |
| `tests/test_serving.py` | OpenAPI 可生成、请求模型绑定、**SSE 推流/续传/404** |
