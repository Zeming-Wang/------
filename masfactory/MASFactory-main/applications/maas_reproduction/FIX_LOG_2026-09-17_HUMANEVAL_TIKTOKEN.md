# HumanEval 与 tiktoken 修复日志（2026-09-17）

## 范围

本轮处理审查项 5（HumanEval 执行逻辑）和审查项 13（tiktoken 导入依赖）。目标是保持原始 MaAS 的评分思路，同时遵守 reproduction 的 MASFactory 数据接口：候选代码执行失败应成为当前样本的失败结果，不应让异常穿透 evaluator、graph 或 DatasetRunner。

## HumanEval 对照原始 MaAS 的修复

原始 `MaAS-main/maas/ext/maas/benchmark/humaneval.py` 的关键行为已恢复：

- 对候选代码执行 sanitize/提取，并保留 entry point 可达的定义；
- 注入 `decode_cyclic`、`decode_shift`、`find_zero` 所需的 helper；
- 使用受控全局对象提供 `math`、`hashlib`、`re` 及 typing 类型；
- 执行异常、语法错误、入口缺失和测试失败统一得到 `FAIL`；
- 候选为空或执行失败返回 score `0.0`，作为可统计的失败样本；
- 成功只在测试函数返回 `None` 时判定为 PASS，保持源实现的判断语义。

实现位于：

`applications/maas_reproduction/maas_reproduction/benchmarks/humaneval.py`

## MASFactory 执行边界

不在 graph 节点内部直接执行候选代码，也没有使用无法终止的 daemon thread。HumanEval scorer 复用了 reproduction 已有的 `CodeExecutor` adapter：

- 在独立 Python 子进程中执行候选代码和测试；
- 超时由 `subprocess.run(..., timeout=...)` 终止子进程；
- 默认超时保持源实现的 15 秒；
- 子进程使用受限 builtins 和 import allowlist，拒绝 `os`、`subprocess`、文件访问、动态执行等高风险入口；
- scorer 在 `BaseBenchmark.evaluate()` seam 上返回 `ScoreResult(0.0, True)`，所以失败样本不会被误分类为 evaluator infrastructure failure，也不会中断整个训练批次。

`BaseBenchmark`、`EvaluatorNode`、`OperatorResult` 和 MASFactory 公共 graph API 均未改动。

## tiktoken 依赖修复

- `masfactory/adapters/token_usage_tracker.py` 不再在模块导入阶段无条件要求 `tiktoken`。
- `tiktoken` 缺失时，MASFactory、fake model 和测试可以完成 collection；真正创建本地 token counter 时才抛出带安装命令的明确 `RuntimeError`。
- 项目原有正式依赖保持不变：`pyproject.toml` 声明 `tiktoken>=0.7.0`，`requirements.txt` 锁定 `tiktoken==0.12.0`。
- 已在 `mas_env` 安装/升级到 `tiktoken==0.12.0`。

## 测试

新增：

- `tests/unit/test_humaneval_execution.py`
  - 特殊 helper；
  - 执行异常变为可靠 0 分；
  - 有界超时；
  - 危险 import 被拒绝。
- `tests/unit/test_tiktoken_import_boundary.py`
  - 模拟缺少 tiktoken 时仍可导入 token tracker；
  - 真正创建计数器时返回明确安装错误。

验证结果：

- HumanEval 与 tiktoken 定向测试：`6 passed`；
- reproduction 全套测试：`96 passed, 3 failed`；
- 3 个失败均为既有 `tests/graphs/test_root_graphs.py` 使用旧的 `architecture_graph=` builder 参数，与本轮 HumanEval 和 tiktoken 修改无关；
- `compileall` 和已有定向回归保持通过。

## 未改变的范围

本轮没有处理 GSM8K 解析语义、`max_iterations`、Evaluator 详情日志和旧 `architecture_graph` 测试兼容性；没有修改 MASFactory 通用 RootGraph/Loop 实现。

