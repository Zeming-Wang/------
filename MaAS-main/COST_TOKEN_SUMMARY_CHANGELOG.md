# 完整运行 Cost/Token 汇总修改记录

## 目标

在原版 MaAS 的每次完整训练或完整测试结束时，向数据集级统一 CSV 追加一行，记录主执行 graph 在该次运行中累计产生的：

- prompt tokens；
- completion tokens；
- total tokens；
- total cost。

汇总粒度是一次完整的 `BaseBenchmark.run_evaluation()` 调用，不是单题粒度。

## 计费边界

本次实现只读取 `graph.llm.get_costs()`，边界如下：

### 纳入统计

- 完整训练中的所有已加载题目；
- 训练参数 `sample` 指定的全部 repetition；
- 完整测试中的所有已加载题目；
- 主 graph 中共享 `graph.llm` 的 operator 调用；
- 已经由 provider 写入主 `CostManager` 的 token 和 cost。

### 明确不纳入统计

- TextGrad 独立创建的 `textgrad_llm`；
- embedding、Controller 前向/反向传播和本地代码执行；
- 未向 provider 返回 usage、因而未进入 `CostManager` 的请求；
- 单独运行、没有经过 `run_evaluation()` 的调试或 smoke 脚本。

没有加入 `textgrad_llm.cost_manager = graph.llm.cost_manager`。这是有意保留的边界，目的是让原版当前统计口径与 `maas_reproduction` 对齐；复现版当前没有执行 TextGrad。

## 输出位置和格式

统一文件位于：

```text
maas/ext/maas/scripts/optimized/<DATASET>/cost_token_summary.csv
```

例如 MATH：

```text
maas/ext/maas/scripts/optimized/MATH/cost_token_summary.csv
```

字段顺序：

```text
timestamp,mode,dataset,sample_count,problem_count,average_score,prompt_tokens,completion_tokens,total_tokens,total_cost,status
```

格式约定：

- `average_score` 保留 5 位小数；运行级失败时为空；
- `total_cost` 保留 8 位小数；
- `problem_count` 是本次实际加载的数据条数，不乘 repetition；
- `sample_count` 原样记录命令参数 `sample`，测试模式也保留；
- `total_tokens = prompt_tokens + completion_tokens`。

## success/failed 定义

- `success`：`run_evaluation()` 按原有语义正常返回。单题异常若已被原有 benchmark 转换为失败结果，不会把完整运行改判为失败。
- `failed`：有异常逃出 `run_evaluation()`。写入失败行后，原异常继续向上传播。

训练中保存 Controller 的异常原本会被捕获并只写日志。本次没有改变该行为，因此这类情况仍遵循原有运行语义并记录为 `success`。

如果汇总 CSV 自身写入失败，只记录错误日志，不覆盖正常返回值，也不替换正在传播的原始异常。

## 源码修改

### `maas/ext/maas/benchmark/benchmark.py`

1. 增加标准库 `csv` 导入。
2. 在 `BaseBenchmark` 中增加 `save_cost_summary()`：
   - 读取主 graph 的累计成本；
   - 创建数据集级汇总目录；
   - 在空文件或新文件中写表头；
   - 追加一行固定格式记录。
3. 在 `run_evaluation()` 外层增加运行级状态管理：
   - 默认状态为 `failed`；
   - 原有训练或测试路径正常完成后设为 `success`；
   - 在 `finally` 中恰好尝试一次汇总写入；
   - 不吞掉原有训练/测试异常。

没有修改：

- `MATHBenchmark.evaluate_problem()`；
- 逐题结果 CSV；
- graph 和 operator；
- 并发参数；
- 重试机制；
- CostManager；
- TextGrad 的创建和执行逻辑；
- controller loss/utility 计算。

## 回归测试

新增 `tests/test_benchmark_cost_summary.py`，覆盖：

1. 完整测试成功时只追加一行，字段、token 求和和数字精度正确；
2. 完整训练成功时记录 `train`，并确认 `problem_count` 不乘 repetition；
3. 完整运行失败时追加 `failed` 行；
4. 失败行保留已累计的主 graph token/cost；
5. 写完失败行后仍抛出原来的运行异常。
