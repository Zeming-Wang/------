可以。下面给出一版“直接完成全部缺失部分”的完整实施计划，不再采用“先做最小闭环、再逐步扩展”的策略。但各阶段仍按依赖关系排列，避免出现模块已经写完却无法互相连接的问题。

同时，`LoopControllerMessage.should_continue` 的语义和 attributes 验证会在方案中明确冻结。

# 一、最终冻结的 Loop 设计

## 1. LoopControllerMessage 的职责

```python
@dataclass(frozen=True, slots=True)
class LoopControllerMessage:
    cursor: int
    iteration: int
    should_continue: bool
```

字段语义：

| 字段 | 含义 | 计算者 |
|---|---|---|
| `cursor` | 下一次待执行的 `RouteItem.sequence_index` | `OperatorDispatchLoop` 初始计算；之后由 `StateReducerNode` 更新 |
| `iteration` | 当前已经完成的循环轮数 | `OperatorDispatchLoop` 初始设为 `0`；之后由 `StateReducerNode` 加一 |
| `should_continue` | 是否允许 Loop 再执行一个 Operator | `OperatorDispatchLoop` 初始计算；之后由 `StateReducerNode` 计算 |

重要结论：

```text
LoopController 不计算 should_continue
LoopController 只消费 should_continue 并决定继续或终止
```

终止函数：

```python
def should_terminate(message, attributes):
    control = message["loop_control"]
    return not control.should_continue
```

因此：

```text
should_continue = True
    → Loop Controller 不终止
    → RouteCursor 执行下一个 Operator

should_continue = False
    → Loop Controller 终止
    → Loop 输出最终 DispatchState
```

## 2. 谁计算初始 should_continue

由 `OperatorDispatchLoop._forward()` 计算：

```python
should_continue = (
    not state.termination_requested
    and state.route_cursor < len(state.route_plan.items)
)
```

初始消息：

```python
LoopControllerMessage(
    cursor=state.route_cursor,
    iteration=0,
    should_continue=should_continue,
)
```

初始状态来源：

```text
NativeBootstrapGraph
    → DispatchState
    → OperatorDispatchLoop._forward()
```

## 3. 谁计算下一轮 should_continue

由 `StateReducerNode` 计算。

输入：

```text
operator_result
```

读取：

```python
previous = attributes["dispatch_state"]
```

生成：

```python
next_state = reduce_result(previous, operator_result)
```

然后：

```python
should_continue = (
    not next_state.termination_requested
    and next_state.route_cursor < len(next_state.route_plan.items)
)
```

生成下一条控制消息：

```python
LoopControllerMessage(
    cursor=next_state.route_cursor,
    iteration=current_iteration + 1,
    should_continue=should_continue,
)
```

## 4. DispatchState 的生命周期

```text
NativeBootstrapGraph 输出初始 DispatchState
        ↓
OperatorDispatchLoop 接收
        ↓
写入 Loop local attributes
        ↓
RouteCursor 从 local attributes 读取
        ↓
Operator 执行
        ↓
StateReducer 从 local attributes 读取旧状态
        ↓
生成新 DispatchState
        ↓
写回 Loop local attributes
        ↓
Loop 终止
        ↓
Loop exit 输出 final DispatchState
```

## 5. RootGraph attributes 验证要求

必须验证下面三个条件：

```text
RootGraph attributes
    ✗ 不存在 dispatch_state

Loop local attributes
    ✓ 存在 dispatch_state

Loop exit
    → 只输出 final dispatch_state
```

需要新增明确测试：

```python
output, root_attributes = root.invoke(sample)

assert "dispatch_state" not in root_attributes
assert "dispatch_state" not in root.attributes

architecture = root._nodes["architecture"]
dispatch_loop = architecture._nodes["operator_dispatch_loop"]

assert "dispatch_state" in dispatch_loop.attributes
assert output["sample_result"]["prediction"] is not None
```

同时检查：

```python
assert "dispatch_state" not in loop._controller.input_keys
assert "dispatch_state" not in loop._controller.output_keys
```

这可以验证 `dispatch_state` 没有由于：

```text
pull_keys=None
push_keys=None
```

被隐式传播到 RootGraph。

## 6. Loop 的最终 keys

Controller 边：

```python
LOOP_CONTROL_KEYS = {
    "loop_control": "LoopControllerMessage only."
}
```

Loop 输入：

```python
pull_keys={}
```

Loop 输出：

```python
push_keys={
    "dispatch_state": "Final dispatch state."
}
```

注意：

```text
Loop 的 dispatch_state 输出只传给 ArchitectureExecGraph 的 Finalize 节点，
不允许继续推送到 RootGraph attributes。
```

因此 ArchitectureExecGraph 也必须：

```python
pull_keys={}
push_keys={}
```

---

# 二、EarlyStop 的最终规则

## 1. 第一层 EarlyStop

保留原始 MaAS 机制：

```text
第一层采样到 EarlyStop
    → 替换为 Generate
    → policy_log_prob 加上 -1.5
```

形式：

```python
policy_log_prob = selected_log_prob - 1.5
```

这不是 reward 惩罚。

## 2. 后续层 EarlyStop

```text
后续层采样到 EarlyStop
    → 停止后续层采样
    → 不调用 EarlyStop operator
```

## 3. Utility

永久固定：

```python
utility = score - 3.0 * cost_delta
```

禁止：

```python
utility = score - 3.0 * cost_delta - 1.5
```

需要测试：

```python
assert signal.utility == score - 3.0 * cost_delta
assert -1.5 only appears in policy_log_prob path
```

---

# 三、完整实现计划

## 阶段 0：工程边界和导入路径统一

目标：

```text
pytest 可以从 MASFactory 根目录正常收集
所有模块使用同一种 import 方式
```

处理：

- 统一 `applications.maas_reproduction...` 导入
- 删除 `maas_reproduction` 和 `components` 的双路径兼容
- 删除 `_MaASRootGraph.invoke()` 的旧 sample 兼容逻辑
- 删除旧 `architecture_graph` callable 兼容入口
- 删除旧动态 Workflow 加载路径
- 删除 `METAGPT_PROJECT_ROOT` 依赖
- 删除重复的 `attribute_firewall.py`
- 更新 `SPEC_part.md` 中与最终实现冲突的部分

验收：

```text
python -m compileall applications/maas_reproduction
pytest applications/maas_reproduction/tests --collect-only
```

---

## 阶段 1：RuntimeSettings 和配置系统

实现：

```text
maas_reproduction/runtime/settings.py
```

定义：

```python
@dataclass(frozen=True)
class ModelSettings:
    provider: str
    model_name: str
    api_key_env: str
    base_url_env: str | None
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class RuntimeSettings:
    dataset: str
    split: str
    mode: str
    round_number: int
    sample: int
    batch_size: int
    epochs: int
    seed: int
    learning_rate: float
    embedding_model: str
    output_root: Path
    model: ModelSettings
```

实现：

```python
def load_settings(
    config_root: Path,
    *,
    mode_override: str | None = None,
    dataset_override: str | None = None,
    split_override: str | None = None,
) -> RuntimeSettings:
    ...
```

填充配置：

- [models.json](C:/Users/lenovo/Desktop/论文复现相关/masfactory/MASFactory-main/applications/maas_reproduction/assets/config/models.json)
- [experiments.json](C:/Users/lenovo/Desktop/论文复现相关/masfactory/MASFactory-main/applications/maas_reproduction/assets/config/experiments.json)
- [bootstrap.json](C:/Users/lenovo/Desktop/论文复现相关/masfactory/MASFactory-main/applications/maas_reproduction/assets/config/bootstrap.json)
- [operators.json](C:/Users/lenovo/Desktop/论文复现相关/masfactory/MASFactory-main/applications/maas_reproduction/assets/config/operators.json)
- [evaluation.json](C:/Users/lenovo/Desktop/论文复现相关/masfactory/MASFactory-main/applications/maas_reproduction/assets/config/evaluation.json)

配置禁止包含：

```text
API key
Model 实例
Optimizer 实例
Graph 实例
expected_answer
```

---

## 阶段 2：ModelFactory 和 FakeModel

新增：

```text
maas_reproduction/models/model_factory.py
maas_reproduction/models/fake_model.py
```

### ModelFactory

```python
def create_shared_model(settings: ModelSettings) -> Model:
    ...
```

职责：

```text
读取环境变量
校验 API key
读取 base URL
创建唯一 MASFactory OpenAIModel
返回 shared Model
```

### FakeModel

必须支持：

```text
固定响应
按 operator 区分响应
记录 messages
模拟失败
返回 token usage
```

所有 Operator 最终使用同一个 Model 实例：

```text
shared_model
    ├── Generate
    ├── GenerateCoT
    ├── SelfRefine
    ├── MultiGenerateCoT
    ├── ScEnsemble
    ├── Programmer
    └── HumanEval repair
```

不允许 Operator 自己创建 Model。

---

## 阶段 3：EmbeddingProvider

实现：

```text
maas_reproduction/models/embeddings.py
```

实现：

```python
class EmbeddingProvider:
    def encode(self, text: str) -> Tensor:
        ...

    def encode_many(self, texts: Sequence[str]) -> Tensor:
        ...
```

要求：

- 延迟加载
- 单次 runtime 只创建一次 encoder
- 默认使用 `all-MiniLM-L6-v2`
- 输出维度 384
- 参数冻结
- 不参与 optimizer
- 支持 FakeEmbeddingProvider
- operator 名称顺序和 embedding 行严格对应

---

## 阶段 4：MultiLayerController

实现：

```text
maas_reproduction/models/controller.py
```

从 olderone 迁移算法，不迁移旧 Workflow。

需要保留：

```text
四层 controller
query embedding
operator embedding
cosine similarity
threshold = 0.3
分层采样
Generate 优先
第一层 EarlyStop 替换
第一层 -1.5 log-prob adjustment
后续层 EarlyStop 截断
layer log-prob 聚合
```

接口固定：

```python
controller.forward(
    query: str,
    operator_embeddings: Tensor,
    operator_names: Sequence[str],
) -> tuple[list[Tensor], list[list[str]]]
```

禁止：

```text
调用 Model
访问 expected_answer
执行 Operator
修改 DispatchState
修改 attributes
计算 score/utility
```

---

## 阶段 5：复用现有 Operator，完成运行时接入

这一步不重写 Operator。

继续使用：

- [native_agent_operator_graph.py](C:/Users/lenovo/Desktop/论文复现相关/masfactory/MASFactory-main/applications/maas_reproduction/components/operators/native_agent_operator_graph.py)
- [multi_generate_cot_graph.py](C:/Users/lenovo/Desktop/论文复现相关/masfactory/MASFactory-main/applications/maas_reproduction/components/operators/multi_generate_cot_graph.py)
- [sc_ensemble_graph.py](C:/Users/lenovo/Desktop/论文复现相关/masfactory/MASFactory-main/applications/maas_reproduction/components/operators/sc_ensemble_graph.py)
- [programmer_graph/workflow.py](C:/Users/lenovo/Desktop/论文复现相关/masfactory/MASFactory-main/applications/maas_reproduction/components/operators/programmer_graph/workflow.py)
- [registry.py](C:/Users/lenovo/Desktop/论文复现相关/masfactory/MASFactory-main/applications/maas_reproduction/components/operators/registry.py)

需要完成：

```text
operators.json → registry config
shared Model → operator config
prompt template → operator instructions
cost tracker → model invocation
```

测试重点：

```text
所有 Operator 复用同一 Model
OperatorResult 结构统一
Operator 不接收 expected_answer
Operator 不修改 DispatchState
```

---

## 阶段 6：DatasetLoader 和数据文件

实现：

```text
maas_reproduction/adapters/dataset_loader.py
```

建议接口：

```python
@dataclass(frozen=True)
class DatasetSample:
    problem_index: int
    dataset: str
    problem: str
    expected_answer: object
    entry_point: str = ""
    test: str | None = None
    canonical_solution: str | None = None


class DatasetLoader:
    def load(
        self,
        dataset: str,
        split: str,
        indices: Sequence[int] | None = None,
    ) -> list[DatasetSample]:
        ...
```

支持：

```text
GSM8K
MATH
HumanEval
```

字段隔离：

```text
ArchitectureRequest:
    problem
    problem_index
    entry_point

EvaluationContext:
    problem
    problem_index
    expected_answer
    entry_point
    test
    canonical_solution
```

数据文件：

```text
assets/data/gsm8k_train.jsonl
assets/data/gsm8k_test.jsonl
assets/data/math_train.jsonl
assets/data/math_test.jsonl
assets/data/humaneval.jsonl
```

Loader 必须拒绝：

```text
负索引
越界索引
缺失 problem
缺失 HumanEval entry_point
格式错误 JSONL
```

---

## 阶段 7：RuntimeDependencies 和 Bootstrap

实现：

```text
maas_reproduction/runtime/dependencies.py
maas_reproduction/runtime/bootstrap.py
maas_reproduction/runtime/seed.py
```

建议 RuntimeContext：

```python
@dataclass
class RuntimeContext:
    settings: RuntimeSettings
    model: Model
    embedding_provider: EmbeddingProvider
    policy_controller: MultiLayerController
    operator_embeddings: Tensor
    operator_catalog: tuple[str, ...]
    operator_registry: Mapping[str, Mapping[str, Any]]
    scorer: BaseScorer
    cost_tracker: CostTracker
    checkpoint_manager: CheckpointManager
    optimizer: Optimizer | None
    batch_accumulator: BatchAccumulator | None
    root_graph: RootGraph
    dataset_runner: DatasetRunner
```

Bootstrap 负责：

```text
读取配置
→ 创建 Model
→ 创建 EmbeddingProvider
→ 创建 Controller
→ 加载 Operator Registry
→ 创建 scorer
→ 创建 CostTracker
→ train 创建 optimizer
→ train 创建 BatchAccumulator
→ test 不创建 optimizer
→ 创建 CheckpointManager
→ 创建 RootGraph
→ 创建 DatasetRunner
```

Bootstrap 不负责：

```text
执行具体 Operator
计算 score
计算 utility
保存 expected_answer
```

---

## 阶段 8：修正 RootGraph attributes 隔离

RootGraph 链路保持：

```text
InputSplit
    ├── architecture_request → ArchitectureExecGraph
    └── evaluation_context → Evaluator
```

ArchitectureExecGraph：

```python
pull_keys={}
push_keys={}
```

Evaluator：

```python
pull_keys={}
```

Loop：

```python
pull_keys={}
push_keys={"dispatch_state": "..."}
```

必须新增递归属性检查：

```python
def assert_no_dispatch_state_in_root_attributes(root):
    ...
```

验证：

```text
RootGraph.attributes 不包含 dispatch_state
RootGraph.invoke 返回的 attributes 不包含 dispatch_state
ArchitectureExecGraph.attributes 不包含 expected_answer
Loop.attributes 包含 dispatch_state
Loop controller input/output 不包含 dispatch_state
```

---

## 阶段 9：补齐 Completion Graph

实现：

```text
components/benchmark_completion_graph/
```

### GSM8K

```text
DispatchState
→ 最终答案整理
→ 可选 Programmer 验证
→ CompletionResult
```

### MATH

```text
候选为空 → 结构化失败
单候选 → 透传
多候选 → ScEnsemble
```

### HumanEval

```text
DispatchState
→ 代码提取
→ Test
→ 通过：完成
→ 失败：Repair
→ 再 Test
→ retry exhausted：结构化失败
```

CompletionGraph 不修改：

```text
RoutePlan
policy_log_prob
DispatchState.route_cursor
Controller 状态
```

---

## 阶段 10：CodeExecutor 和 HumanEval Test

实现：

```text
maas_reproduction/adapters/code_executor.py
components/operators/humaneval_test_graph.py
```

统一执行接口：

```python
@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool
    error: str | None
```

要求：

```text
子进程隔离
timeout
stdout/stderr 截断
危险 import 限制
文件访问限制
网络访问限制
资源限制
```

HumanEval 的 `test` 只能进入：

```text
CodeExecutor / HumanEval completion / Evaluator
```

不能进入：

```text
ArchitectureRequest
DispatchState
OperatorInvocation
Controller
```

---

## 阶段 11：DatasetRunner 和 Checkpoint 接入

现有基础实现保留：

- [dataset_runner.py](C:/Users/lenovo/Desktop/论文复现相关/masfactory/MASFactory-main/applications/maas_reproduction/maas_reproduction/runtime/dataset_runner.py)
- [checkpoint_manager.py](C:/Users/lenovo/Desktop/论文复现相关/masfactory/MASFactory-main/applications/maas_reproduction/maas_reproduction/runtime/checkpoint_manager.py)
- [batch_accumulator.py](C:/Users/lenovo/Desktop/论文复现相关/masfactory/MASFactory-main/applications/maas_reproduction/maas_reproduction/training/batch_accumulator.py)

补齐：

```text
epoch
round
cursor
resume
partial batch flush
checkpoint save
checkpoint load
metrics
```

Checkpoint 不保存：

```text
Model client
Graph instance
expected_answer
完整 prompt
未完成 autograd batch
```

DatasetRunner 的公开结果只允许：

```text
problem_index
prediction
score
cost
status
failure_source
route_summary
```

---

## 阶段 12：ArtifactStore

实现：

```text
maas_reproduction/adapters/artifact_store.py
```

保存：

```text
运行配置
公开 sample result
metrics
checkpoint metadata
版本信息
```

禁止写入：

```text
API key
expected_answer
完整 prompt
隐藏 route state
live Tensor
```

---

## 阶段 13：CLI 入口

实现：

- [main.py](C:/Users/lenovo/Desktop/论文复现相关/masfactory/MASFactory-main/applications/maas_reproduction/main.py)
- `scripts/run_train.py`
- `scripts/run_test.py`
- `scripts/run_smoke.py`

统一入口：

```bash
python -m applications.maas_reproduction.main \
  --mode train \
  --dataset GSM8K \
  --split train \
  --batch-size 4 \
  --round 1 \
  --sample 1 \
  --seed 42
```

CLI 只负责：

```text
解析参数
→ load_settings
→ build_runtime
→ DatasetRunner.run
→ 输出公开结果
```

CLI 不负责：

```text
创建 Model
创建 Controller
创建 Optimizer
执行 Operator
计算 utility
```

---

# 四、测试计划

## 1. Loop 状态测试

必须覆盖：

```text
should_continue=True → 执行下一 Operator
should_continue=False → Loop 退出
route_cursor 到末尾 → should_continue=False
termination_requested=True → should_continue=False
```

## 2. Attributes 防泄露测试

必须验证：

```python
assert "dispatch_state" not in root.attributes
assert "dispatch_state" not in returned_root_attributes
assert "dispatch_state" in loop.attributes
assert "dispatch_state" not in loop._controller.input_keys
assert "dispatch_state" not in loop._controller.output_keys
```

同时递归检查：

```text
ArchitectureRequest
RoutePlan
OperatorInvocation
OperatorResult.metadata
route_metadata
execution_metadata
```

不能出现：

```text
expected_answer
test
canonical_solution
```

## 3. EarlyStop 测试

验证：

```text
第一层 EarlyStop 被替换为 Generate
第一层 policy_log_prob 应用 -1.5
utility 不包含 -1.5
后续 EarlyStop 截断路线
EarlyStop 不实际调用
```

## 4. Controller 梯度测试

验证：

```text
policy_log_prob.requires_grad == True
loss.requires_grad == True
controller 参数存在 grad
optimizer.step() 后参数发生变化
```

## 5. Operator 集成测试

验证：

```text
Generate
GenerateCoT
SelfRefine
MultiGenerateCoT
ScEnsemble
Programmer
HumanEval Test
```

都能完成：

```text
Invocation → OperatorResult
```

## 6. 三类 Benchmark 测试

### GSM8K

```text
数据加载
答案抽取
数值比较
completion
evaluator
```

### MATH

```text
boxed 提取
数值等价
符号等价
多候选 ensemble
```

### HumanEval

```text
代码提取
沙箱执行
公共测试
失败修复
retry 上限
```

## 7. Checkpoint 测试

验证：

```text
controller 恢复
optimizer 恢复
cursor 恢复
epoch 恢复
RNG 恢复
operator catalog 校验
```

## 8. CLI 测试

验证：

```text
python -m ... --mode smoke
python -m ... --mode train
python -m ... --mode test
```

---

# 五、最终不重复实现清单

以下模块保留现有实现，只做接入和测试：

```text
schemas.py
contracts.py
ArchitectureExecGraph
NativeBootstrapGraph
OperatorDispatchLoop
RoutePlannerNode
RouteCursorNode
StateReducerNode
FinalizeArchitectureResultNode
GenerateGraph
GenerateCoTGraph
SelfRefineGraph
MultiGenerateCoTGraph
ScEnsembleGraph
ProgrammerGraph
EarlyStop
EvaluatorNode
TrainingSignal
BatchAccumulator
CheckpointManager
CostTracker
GSM8K scorer
MATH scorer
HumanEval scorer
DatasetRunner 基础逻辑
```

以下模块才是需要新增或实质补齐的：

```text
RuntimeSettings
ModelFactory
FakeModel
EmbeddingProvider
MultiLayerController
DatasetLoader
RuntimeDependencies
Runtime Bootstrap
Prompt Loader
真实配置
真实数据
BenchmarkCompletionGraph
HumanEvalTestGraph
CodeExecutor
ArtifactStore
CLI
```

# 六、最终验收标准

项目完成必须同时满足：

```text
1. pytest 可以从仓库根目录正常收集
2. 所有非测试空文件已实现或明确删除
3. FakeModel 可运行完整 train/test 流程
4. 真实 Model 可由环境变量创建
5. Controller 路线行为符合原始 MaAS
6. -1.5 只进入 policy log-prob，不进入 utility
7. utility 严格为 score - 3 * cost_delta
8. LoopControllerMessage 只包含三个控制字段
9. should_continue 的计算责任明确且有测试
10. Loop local attributes 存在 dispatch_state
11. RootGraph attributes 不存在 dispatch_state
12. Loop exit 只输出 final dispatch_state
13. expected_answer 不进入 Architecture/Loop/Operator/prompt/trace
14. 现有 Operator 不被重复重写
15. GSM8K/MATH/HumanEval completion graph 全部实现
16. Programmer 和 HumanEval 使用安全 CodeExecutor
17. DatasetRunner 支持 epoch/cursor/checkpoint/flush
18. Train 模式可 backward 和 optimizer.step
19. Test 模式不创建 optimizer
20. CLI 可以启动 train/test/smoke
21. 输出结果不包含 key、expected_answer、完整 prompt 或隐藏状态
```

最终采用的核心模型是：

```text
LoopControllerMessage
    = 控制消息

DispatchState
    = Loop 局部业务状态

should_continue
    = OperatorDispatchLoop 初始计算
      + StateReducer 每轮重新计算

LoopController
    = 只根据 should_continue 终止或继续

-1.5
    = Controller policy log-prob adjustment

3.0
    = utility cost coefficient
```

这版计划不会重复实现已经存在的普通 Operator，而是直接完成它们的 Runtime 接入、配置接入、Completion 接入和端到端运行能力。