# MASFactory 原生 MaAS 复现最终实施方案

本版以“严格复现 MaAS 训练目标”为最高原则，正式冻结以下规则：

> Failure 不产生额外 reward、advantage 或 penalty。  
> 只要能够形成合法结果、获得可信 `cost_delta`、并由 Evaluator 可靠评分，就使用原始目标：
>
> `utility = score - 3.0 × cost_delta`
>
> 无法形成合法结果、无法获得可信成本、无法可靠评分或没有有效 `policy_log_prob` 时，跳过该样本的 policy update。

`failure_source` 只负责控制、日志和统计，不直接参与 utility 公式。

本方案只实现 MASFactory 原生架构，不包含 Legacy、Compatibility、Hybrid 或旧 `ArchitectureExecNode`。

---

相关的apikey

api_key = sk-ud2hiSb4GqQYi2MuGr3yGVo4pYNtKw3huWY9K9hFerZJcE0c
base_url = https://api.csun.site/v1
model_name = gpt-4o-mini


## 一、核心设计原则

整个系统长期守住五个 seam：

```text
唯一策略决策：RoutePlannerNode
唯一执行循环：OperatorDispatchLoop
唯一状态更新：StateReducerNode
唯一训练资格判断：TrainingSignal
唯一跨样本生命周期：DatasetRunner
```

`ArchitectureExecGraph` 只负责组合这些模块，不承载具体业务逻辑。

---

## 二、最终总体架构

```text
DatasetRunner
│
├── 数据集顺序与 epoch
├── policy_controller
├── shared MASFactory Model
├── optimizer / BatchAccumulator
├── checkpoint / RNG
├── 全局指标与运行目录
│
└── 顺序调用 RootGraph.invoke(one_sample)
```

## TrainRootGraph

```text
ENTRY
  ↓
InputSplitNode
  ├── architecture_request ──→ ArchitectureExecGraph ──┐
  │                                                    ↓
  └── evaluation_context ───────────────────────→ EvaluatorNode
                                                       ↓
                                                LossUpdateNode
                                                       ↓
                                                SampleResultNode
                                                       ↓
                                                     EXIT
```

## TestRootGraph

```text
ENTRY
  ↓
InputSplitNode
  ├── architecture_request ──→ ArchitectureExecGraph ──┐
  │                                                    ↓
  └── evaluation_context ───────────────────────→ EvaluatorNode
                                                       ↓
                                                  MetricsNode
                                                       ↓
                                                SampleResultNode
                                                       ↓
                                                     EXIT
```

分别构建：

```python
build_train_root_graph(...)
build_test_root_graph(...)
```

Test Graph 中不创建：

- optimizer；
- LossUpdateNode；
- BatchAccumulator；
- 训练 checkpoint 保存逻辑。

---

## 三、ArchitectureExecGraph 内部结构

```text
ArchitectureExecGraph
ENTRY
  ↓
InitializeExecutionNode
  ↓
RoutePlannerNode
  ↓
NativeBootstrapGraph
  ↓
OperatorDispatchLoop
  │
  ├── dispatch_loop_controller
  ├── RouteCursorNode
  ├── LogicSwitch
  ├── Native OperatorGraph
  ├── InvalidOperatorNode
  └── StateReducerNode
          └── dispatch_state → dispatch_loop_controller
  ↓
FinalizeArchitectureResultNode
  ↓
EXIT
```

内部顺序冻结为：

```text
Initialize
→ RoutePlanner
→ NativeBootstrap
→ DispatchLoop
→ Finalize
```

不能改成：

```text
Bootstrap
→ RoutePlanner
```

因为 bootstrap 产生的 solution 不属于 `policy_controller` 的策略输入。

---

## 四、统一术语

| 名称 | 含义 |
|---|---|
| `policy_controller` | 可训练的 operator 路由策略模型 |
| `RoutePlannerNode` | 调用策略模型并生成 RoutePlan |
| `dispatch_loop_controller` | MASFactory Loop 内部控制器 |
| `raw_selected_layers` | policy controller 返回的分层选择 |
| `RoutePlan` | 策略决策的唯一产物 |
| `policy_log_prob` | 整条路线的聚合策略 log-probability |
| `DispatchState` | 已确定路线的执行状态 |
| `OperatorInvocation` | 当前 operator 输入快照 |
| `OperatorResult` | 单次 operator 的结构化结果 |
| `FailureSource` | 失败发生位置，仅用于控制和观测 |
| `TrainingSignal` | 是否更新及对应 utility |

---

## 五、RootGraph 消息契约

## InputSplitNode

```python
return {
    "architecture_request": {
        "problem": problem,
        "problem_index": problem_index,
        "entry_point": entry_point or "",
    },
    "evaluation_context": {
        "problem": problem,
        "problem_index": problem_index,
        "entry_point": entry_point or "",
        "expected_answer": expected_answer,
    },
}
```

连接：

```text
InputSplitNode → ArchitectureExecGraph
    architecture_request

InputSplitNode → EvaluatorNode
    evaluation_context
```

## ArchitectureExecGraph 输出

```python
{
    "architecture_result": {
        "prediction": prediction,
        "cost_delta": cost_delta,
        "policy_log_prob": policy_log_prob,
        "status": status,
        "failure_source": failure_source,
        "result_valid": result_valid,
        "cost_reliable": cost_reliable,
        "route_metadata": route_metadata,
        "execution_metadata": execution_metadata,
    }
}
```

其中：

```text
prediction       str | None
cost_delta       float | None
policy_log_prob  torch.Tensor | None
failure_source   FailureSource | None
result_valid     bool
cost_reliable    bool
```

`expected_answer` 永远不得进入：

- ArchitectureExecGraph；
- DispatchState；
- OperatorInvocation；
- Agent prompt；
- route metadata；
- execution trace。

---

## 六、RoutePlannerNode

## NativeBootstrapGraph 的唯一职责

```text
ArchitectureRequest
→ policy_controller.forward()
→ RoutePlan
```

负责：

1. 调用 `policy_controller.forward()`；
2. 获取 `log_probs_layers` 和 `raw_selected_layers`；
3. 归一化 EarlyStop；
4. 将多层路线线性展开；
5. 聚合 `policy_log_prob`；
6. 返回 RoutePlan。

构造依赖：

```python
RoutePlannerNode(
    policy_controller=policy_controller,
    operator_embeddings=operator_embeddings,
    operator_catalog=operator_catalog,
)
```

这些依赖不进入 Edge、attributes 或 Agent prompt。

## 明确禁止

RoutePlannerNode 不得：

- 执行 bootstrap；
- 调用 operator；
- 创建 candidates；
- 修改 current solution；
- 计算 score、cost 或 utility；
- 调用 optimizer；
- 保存 checkpoint；
- 管理 DispatchLoop。

如果规划失败：

```text
failure_source = PLANNING
policy_log_prob = None
result_valid = False
→ skip policy update
```

---

## 七、RoutePlan

```python
@dataclass(frozen=True)
class RouteItem:
    sequence_index: int
    layer_index: int
    position: int
    operator_name: str
    is_control_marker: bool = False


@dataclass(frozen=True)
class RoutePlan:
    items: tuple[RouteItem, ...]
    policy_log_prob: torch.Tensor
```

例如：

```text
layer 0: Generate, Programmer
layer 1: SelfRefine
layer 2: ScEnsemble
```

展开为：

```text
0 → layer 0 / position 0 → Generate
1 → layer 0 / position 1 → Programmer
2 → layer 1 / position 0 → SelfRefine
3 → layer 2 / position 0 → ScEnsemble
```

不变量：

```python
for index, item in enumerate(route_plan.items):
    assert item.sequence_index == index
```

DispatchLoop 只维护一个线性 `route_cursor`。

## policy_log_prob

定义为：

> 当前整条 RoutePlan 对应的聚合策略 log-probability Tensor。

operator 执行阶段不得：

- 重新计算；
- 修改；
- 拆分；
- detach；
- 转为 JSON。

日志值单独生成：

```python
policy_log_prob_value = float(
    route_plan.policy_log_prob.detach().cpu()
)
```

---

## 八、NativeBootstrapGraph

## 唯一职责

```text
ArchitectureRequest + RoutePlan
→ bootstrap operator execution
→ initial DispatchState
```

MATH 固定流程：

```text
Programmer
→ Generate refinement
→ initial current_solution/candidates
```

正常输出：

```python
{
    "dispatch_state": DispatchState(
        request=request,
        route_plan=route_plan,
        route_cursor=0,
        current_solution=bootstrap_solution,
        candidates=(bootstrap_solution,),
        termination_requested=False,
        error_state=None,
    )
}
```

Bootstrap 不得：

- 再次调用 policy controller；
- 修改 RoutePlan；
- 执行路线中的普通 operator；
- 访问 expected answer；
- 计算 utility；
- 保存 checkpoint。

## Bootstrap 失败

Bootstrap 内部可以 retry。

如果 retry 后仍能产生合法、可消费的结构化结果，可以继续执行。

如果最终无法初始化合法 DispatchState：

```text
failure_source = BOOTSTRAP
result_valid = False
→ Finalize 生成不可训练结果
→ skip policy update
```

Bootstrap 失败本身不产生额外 penalty。

---

## 九、精简 DispatchState

```python
@dataclass(frozen=True)
class DispatchState:
    request: ArchitectureRequest
    route_plan: RoutePlan
    route_cursor: int
    current_solution: str | None
    candidates: tuple[str, ...]
    termination_requested: bool
    error_state: dict | None
```

收录原则：

> 只有后续 operator、Loop 终止或最终 prediction 选择必须读取的字段，才能进入 DispatchState。

允许：

- request；
- route plan；
- route cursor；
- current solution；
- candidates；
- termination requested；
- error state。

禁止：

- generated code；
- execution output；
- Programmer 审计产物；
- raw prompt；
- cost；
- reward；
- score；
- utility；
- loss；
- metrics；
- retry count；
- hook context；
- Model；
- optimizer。

Programmer 的代码、stdout/stderr、exit code、工作目录和 retry 信息由 ProgrammerGraph、handler、Hook 和运行产物负责保存。

Reducer 更新时必须创建新对象，不得原地修改旧 state。

---

## 十、OperatorDispatchLoop

## 唯一职责

```text
DispatchState
→ 按 RoutePlan 顺序执行
→ DispatchState
```

Loop 只理解：

```text
state.route_plan.items
state.route_cursor
state.termination_requested
```

它不理解：

- 路线如何生成；
- policy controller；
- bootstrap；
- score；
- utility；
- optimizer；
- checkpoint；
- evaluation。

## 往返契约

```python
DISPATCH_LOOP_KEYS = {
    "dispatch_state": "Current complete operator dispatch state."
}
```

Controller 入口、反馈和 Loop 出口统一使用：

```python
{"dispatch_state": state}
```

---

## 十一、Loop 终止函数

MASFactory 的正式语义为：

```text
terminate_condition_function 返回 True
→ 终止

返回 False
→ 继续
```

框架无需修改。新代码必须使用：

```python
def should_terminate(
    message: dict,
    attributes: dict,
) -> bool:
    state = message["dispatch_state"]

    return (
        state.termination_requested
        or state.route_cursor >= len(state.route_plan.items)
    )
```

禁止把 `should_continue` 传给 `terminate_condition_function`。

测试真值表：

| 状态 | 返回值 |
| --- | ---: |
| cursor 未到末尾且未请求终止 | `False` |
| termination requested | `True` |
| cursor 等于 items 长度 | `True` |
| 空 RoutePlan | `True` |
| cursor 超过 items 长度 | `True` |

---

## 十二、Loop state hydration

## Controller → RouteCursorNode

```python
dispatch_loop.edge_from_controller(
    route_cursor_node,
    keys=DISPATCH_LOOP_KEYS,
)
```

## RouteCursorNode

唯一职责：

```text
DispatchState
→ 当前 RouteItem
→ OperatorInvocation
```

```python
def route_cursor_forward(message, attributes):
    state = message["dispatch_state"]
    item = state.route_plan.items[state.route_cursor]

    return {
        "dispatch_state": state,
        "operator_invocation": {
            "sequence_index": item.sequence_index,
            "layer_index": item.layer_index,
            "position": item.position,
            "operator_name": item.operator_name,
            "problem": state.request.problem,
            "entry_point": state.request.entry_point or "",
            "current_solution": state.current_solution or "",
            "candidates": list(state.candidates),
        },
    }
```

配置：

```python
pull_keys={}

push_keys={
    "dispatch_state": "Current dispatch state."
}
```

RouteCursor 不得：

- 推进 cursor；
- 修改 state；
- 修改 candidates；
- 调用 operator；
- 处理 failure；
- 计算 cost 或 utility。

## StateReducer → Controller

```python
dispatch_loop.edge_to_controller(
    state_reducer_node,
    keys=DISPATCH_LOOP_KEYS,
)
```

---

## 十三、Canonical state

执行阶段唯一 canonical state 是：

```text
dispatch_state
```

数据流固定为：

```text
DispatchState
→ RouteCursor 提取 OperatorInvocation
→ OperatorGraph 返回 OperatorResult
→ StateReducer 生成新 DispatchState
```

`OperatorInvocation` 是输入快照，`OperatorResult` 是业务结果，二者都不是持久状态。

不另外维护：

```text
attributes["current_solution"]
attributes["candidates"]
```

---

## 十四、LogicSwitch 与 OperatorRegistry

## LogicSwitch

唯一职责：

```text
operator_name
→ 对应 OperatorGraph
```

predicate 必须是纯函数：

```python
lambda message, attributes: (
    message["operator_invocation"]["operator_name"] == "Generate"
)
```

不得修改 cursor、state、attributes 或 candidates。

保留 `InvalidOperatorNode`。无效 operator 表示规划、catalog 或配置异常，不产生额外 utility penalty；通常无法形成合法结果，因此跳过训练。

## OperatorRegistry

只能保存类、模板或配置：

```python
operator_registry = {
    "Generate": {
        "node_class": NativeAgentOperatorGraph,
        "config": generate_config,
    },
    "Programmer": {
        "node_class": ProgrammerGraph,
        "config": programmer_config,
    },
}
```

不能保存预构建 Graph 实例。

所有 operator 必须由所属 Graph 的 `create_node()` 创建。

---

## 十五、OperatorGraph interface

```text
OperatorInvocation
→ OperatorResult
```

```python
{
    "operator_result": {
        "operator_name": "...",
        "status": "success",
        "solution": "...",
        "candidates": [],
        "code": None,
        "execution_output": None,
        "metadata": {},
    }
}
```

OperatorGraph 只负责：

- 构造当前 prompt；
- 调用 MASFactory Agent；
- 执行内部 retry；
- 解析和验证输出；
- 返回合法 OperatorResult。

不得：

- 修改 DispatchState；
- 增加 route cursor；
- 修改全局 candidates；
- 修改 RoutePlan；
- 访问 expected answer；
- 计算 utility；
- 调用 optimizer；
- 保存 checkpoint；
- 创建新的 Model；
- 调用 provider SDK。

---

## 十六、Agent 的 Edge 规则

业务输入来自 incoming Edge 时：

```python
pull_keys={}
```

业务输出通过 outgoing Edge 时：

```python
push_keys={}
```

规则：

```text
业务输入
→ incoming Edge keys

业务输出
→ outgoing Edge keys

共享 Loop 状态
→ pull_keys/push_keys

Model、policy controller、cost tracker
→ 构造注入
```

同一业务字段不得同时存在于 Edge 和 attributes。

---

## 十七、Native operator

## 共享深模块

以下 operator 共用 `NativeAgentOperatorGraph`：

- Generate；
- GenerateCoT；
- SelfRefine。

差异仅由 instructions、输入字段、输出 schema、formatter 和 validator 表达。

## 独立 Graph

以下 operator 有真实独立控制流：

- MultiGenerateCoT；
- ScEnsemble；
- Programmer。

EarlyStop 是确定性 control marker。

## ProgrammerGraph

```text
CodeGenerationAgent
→ CodeParseNode
→ ProgrammerExecutionNode
→ LogicSwitch
    ├── success
    ├── retry
    └── failure
```

Programmer timeout/retry 属于 ProgrammerGraph 内部控制流：

```text
attempt
  ↓
失败？
├── 是 → retry
│        ↓
│     retry exhausted
│        ↓
│     生成结构化 OperatorResult
└── 否 → 成功 OperatorResult
```

不能把“timeout”直接等同于 fatal failure。

### 可恢复 timeout

如果 retry 耗尽后仍可返回合法 OperatorResult，例如：

```python
{
    "operator_name": "Programmer",
    "status": "timeout",
    "execution_output": "timeout",
    "metadata": {...},
}
```

则 seam 仍然成立：

```text
OperatorGraph
→ OperatorResult
→ StateReducer
→ 继续或终止路线
```

只要最终能形成合法 ArchitectureResult、可信 cost 并可靠评分，仍按原 MaAS utility 训练。

### Fatal failure

只有最终无法形成符合 contract 的 OperatorResult，例如：

- Graph exception；
- provider 崩溃且无法结构化恢复；
- 执行器完全不可用；
- 输出对象缺失或无法解析为最小结果；

才升级为 fatal execution failure。

---

## 十八、StateReducerNode

唯一职责：

```text
DispatchState + OperatorResult
→ 新 DispatchState
```

负责：

- 更新 current solution；
- 添加或收敛 candidates；
- 更新 error state；
- 设置 termination；
- 推进 cursor。

```python
def reduce_operator_result(message, attributes):
    previous = attributes["dispatch_state"]
    result = message["operator_result"]

    updated = reduce_result(previous, result)

    return {
        "dispatch_state": updated
    }
```

Reducer 不决定：

- 是否参与训练；
- utility；
- score；
- cost 是否可信；
- failure penalty。

它只负责执行状态转换。

---

## 十九、FailureSource 的定位

```python
class FailureSource(str, Enum):
    PLANNING = "planning"
    BOOTSTRAP = "bootstrap"
    ROUTE_EXECUTION = "route_execution"
    EVALUATION = "evaluation"
    INFRASTRUCTURE = "infrastructure"
```

`FailureSource` 只用于：

- 描述失败发生在哪里；
- 决定执行是否能够继续；
- 日志与统计；
- 判断是否还能形成合法结果；
- 帮助定位不可训练样本的原因。

它不参与：

```python
utility = score - 3.0 * cost_delta
```

禁止出现：

```python
if failure_source == FailureSource.ROUTE_EXECUTION:
    utility -= failure_penalty
```

也不再定义：

- failure reward；
- failure advantage；
- route penalty；
- fixed failure penalty。

---

## 二十、合法结果与 fatal failure

必须明确区分两类结果。

## 1. 可形成合法 ArchitectureResult

例如：

- Programmer timeout，但返回结构化 OperatorResult；
- 某个 operator 产生空解，但 StateReducer 能明确处理；
- ScEnsemble fallback；
- EarlyStop；
- operator 返回可评分的失败文本；
- 路线提前结束但已有候选结果。

只要满足：

```text
ArchitectureResult contract 成立
cost_delta 可信
Evaluator 能可靠评分
policy_log_prob 存在
```

就正常训练：

```python
utility = score - 3.0 * cost_delta
```

FailureSource 不改变公式。

## 2. 无法形成合法 ArchitectureResult

例如：

- API/provider 崩溃；
- Graph exception；
- OperatorResult 根本不存在；
- 状态损坏；
- cost 无法可靠获得；
- 无法构造任何可评分 prediction；
- Evaluator 无法可靠评分。

处理：

```text
记录 failure
记录已经掌握的诊断信息
skip policy update
```

不得：

- 把 cost 默认为 0；
- 把 score 默认为 0 后训练；
- 人工添加 penalty；
- 伪造合法 ArchitectureResult。

---

## 二十一、FinalizeArchitectureResultNode

唯一职责：

```text
最终 DispatchState
→ ArchitectureResult
```

负责：

- 选择 prediction；
- 判断结果是否合法；
- 处理空 candidates；
- 生成 status；
- 计算或读取单题 cost delta；
- 判断成本是否可信；
- 保留 policy logprob；
- 生成 metadata。

示意：

```python
{
    "architecture_result": {
        "prediction": prediction,
        "cost_delta": cost_delta,
        "policy_log_prob": state.route_plan.policy_log_prob,
        "status": status,
        "failure_source": failure_source,
        "result_valid": result_valid,
        "cost_reliable": cost_reliable,
        "route_metadata": route_metadata,
        "execution_metadata": execution_metadata,
    }
}
```

如果成本获取失败：

```python
cost_delta = None
cost_reliable = False
```

禁止静默改成：

```python
cost_delta = 0.0
```

Finalize 不计算 score 或 utility，也不 detach policy Tensor。

---

## 二十二、成本规则

单题开始：

```python
cost_before = cost_tracker.snapshot()
```

单题完成：

```python
cost_after = cost_tracker.snapshot()
cost_delta = cost_tracker.delta(cost_before, cost_after)
```

可信成本至少满足：

```text
snapshot 成功
delta 可计算
delta 为有限数值
delta >= 0
执行期间 tracker 未被 reset
retry 调用已计入
```

建议提供明确结果：

```python
@dataclass(frozen=True)
class CostResult:
    value: float | None
    reliable: bool
    error: str | None = None
```

若：

```text
cost_reliable=False
```

则：

```text
不计算 utility
skip policy update
记录 cost failure
```

不能把缺失成本当成零成本，否则会改变原 MaAS 目标。

---

## 二十三、EvaluatorNode

唯一职责：

```text
ArchitectureResult + EvaluationContext
→ EvaluationResult
```

Evaluator 首先检查 `result_valid`。

如果结果合法，可以评分：

```python
{
    "evaluation_result": {
        "problem_index": problem_index,
        "prediction": prediction,
        "score": score,
        "cost_delta": cost_delta,
        "policy_log_prob": policy_log_prob,
        "status": status,
        "failure_source": failure_source,
        "result_valid": True,
        "cost_reliable": cost_reliable,
        "evaluation_reliable": True,
    }
}
```

如果 scorer 异常：

```text
failure_source = EVALUATION
evaluation_reliable = False
→ skip policy update
```

成本不可信时，可以选择记录诊断性 score，但不得进入训练 utility。正式训练资格仍由 TrainingSignal 统一判断。

Evaluator 不负责：

- backward；
- optimizer；
- batch；
- checkpoint；
- failure penalty；
- route 修改。

---

## 二十四、TrainingSignal

TrainingSignal 精简为：

```python
@dataclass(frozen=True)
class TrainingSignal:
    should_update: bool
    utility: float | None
    skip_reason: str | None
```

不包含：

- advantage；
- failure reward；
- failure penalty；
- route penalty；
- failure weight。

构造规则：

```python
def build_training_signal(
    evaluation_result: dict,
) -> TrainingSignal:
    if not evaluation_result["result_valid"]:
        return TrainingSignal(
            should_update=False,
            utility=None,
            skip_reason="invalid_architecture_result",
        )

    if not evaluation_result["cost_reliable"]:
        return TrainingSignal(
            should_update=False,
            utility=None,
            skip_reason="unreliable_cost",
        )

    if not evaluation_result["evaluation_reliable"]:
        return TrainingSignal(
            should_update=False,
            utility=None,
            skip_reason="unreliable_evaluation",
        )

    if evaluation_result["policy_log_prob"] is None:
        return TrainingSignal(
            should_update=False,
            utility=None,
            skip_reason="missing_policy_log_prob",
        )

    utility = (
        evaluation_result["score"]
        - 3.0 * evaluation_result["cost_delta"]
    )

    return TrainingSignal(
        should_update=True,
        utility=utility,
        skip_reason=None,
    )
```

注意：

> `failure_source` 不直接决定是否更新。

即使存在 `failure_source=ROUTE_EXECUTION`，只要最终结果合法、成本可信且可可靠评分，仍然按相同公式训练。

---

## 二十五、唯一训练目标

严格复现目标：

```python
utility = score - 3.0 * cost_delta
```

Policy loss 使用原项目一致的符号、batch reduction 和归一化方式。概念上：

```python
loss = -policy_log_prob * utility
```

但实现时应保持原 MaAS 的具体 batch 计算口径，不额外增加：

- failure penalty；
- entropy bonus；
- route penalty；
- timeout penalty；
- invalid output penalty；
- 人工 advantage；
- reward clipping。

如果后续希望实验这些策略，必须作为独立消融实验，不能混入复现模式。

---

## 二十六、LossUpdateNode 与 BatchAccumulator

## LossUpdateNode

唯一职责：

```text
EvaluationResult
→ TrainingSignal
→ BatchAccumulator.add()
```

```python
signal = build_training_signal(evaluation_result)

if not signal.should_update:
    return {
        "update_result": {
            "update_performed": False,
            "skip_reason": signal.skip_reason,
        }
    }

return batch_accumulator.add(
    policy_log_prob=evaluation_result["policy_log_prob"],
    utility=signal.utility,
)
```

LossUpdateNode 不解释 operator 内部失败，也不修改 utility 公式。

## BatchAccumulator

唯一职责：

```text
policy_log_prob + utility
→ batch loss
→ backward
→ optimizer.step
```

负责：

- 暂存 batch 内的 policy Tensor；
- 按原 MaAS 公式计算 loss；
- backward；
- optimizer.step；
- zero_grad；
- 清除已使用 Tensor；
- flush partial batch。

不负责：

- failure attribution；
- cost 可信度判断；
- score；
- expected answer；
- utility 业务规则。

---

## 二十七、Tensor 生命周期

内部：

```text
RoutePlan.policy_log_prob
→ DispatchState
→ ArchitectureResult
→ EvaluationResult
→ LossUpdate
→ BatchAccumulator
→ backward
```

在 backward 完成前不能 detach。

公开结果由 SampleResultNode 转换：

```python
policy_log_prob_value = (
    float(policy_log_prob.detach().cpu())
    if policy_log_prob is not None
    else None
)
```

公开 `sample_result`：

```python
{
    "sample_result": {
        "problem_index": problem_index,
        "prediction": prediction,
        "score": score,
        "cost_delta": cost_delta,
        "status": status,
        "failure_source": failure_source,
        "result_valid": result_valid,
        "cost_reliable": cost_reliable,
        "evaluation_reliable": evaluation_reliable,
        "policy_log_prob_value": policy_log_prob_value,
        "utility": utility,
        "update_performed": update_performed,
        "skip_reason": skip_reason,
        "loss_value": loss_value,
    }
}
```

DatasetRunner 只能长期保存公开结果。

---

## 二十八、checkpoint

checkpoint 只能在 batch 边界保存。

保存前：

```python
assert batch_accumulator.pending_count == 0
assert not batch_accumulator.has_live_tensors
```

保存：

- policy controller state；
- optimizer state；
- dataset cursor；
- epoch；
- Python、NumPy、PyTorch、CUDA RNG；
- operator catalog；
- 配置。

禁止保存：

- pending policy logprob；
- pending loss；
- autograd graph；
- DispatchState；
- ArchitectureResult 内部 Tensor。

如果在 batch 中途请求保存：

```text
延迟请求
→ 完成当前 batch
→ backward/step
→ 清空 Tensor
→ 保存 checkpoint
```

第一版不支持恢复 live autograd graph。

---

## 二十九、DatasetRunner

唯一职责：

```text
dataset
→ RootGraph.invoke()
→ 跨样本生命周期
```

负责：

- 顺序迭代；
- cursor；
- epoch；
- checkpoint 调度；
- RNG 恢复；
- partial batch flush；
- 全局指标；
- 输出文件；
- execution/system failure 统计。

跳过 policy update 的样本仍必须记录：

- problem index；
- failure source；
- skip reason；
- 已知 cost；
- exception 类型；
- provider/executor 状态；
- 是否形成合法结果。

跳过训练不等于从实验统计中消失。

---

## 三十、ArchitectureExecGraph 防止膨胀

`ArchitectureExecGraph.build()` 只允许：

- 创建节点；
- 创建边；
- 注入依赖；
- 声明消息 keys；
- 组合子图。

禁止：

- 加载配置；
- 创建 Model；
- 遍历数据集；
- 直接计算 route；
- 执行 operator 业务；
- 计算 score；
- 计算 utility；
- backward；
- optimizer.step；
- 保存 checkpoint；
- 写实验结果。

本版按深模块原则，将复杂性分别收敛到 RoutePlanner、DispatchLoop、StateReducer 和 TrainingSignal 的 interface 后面，避免把所有判断重新堆回 ArchitectureExecGraph。

---

## 三十一、模块责任矩阵

| 模块 | 输入 | 输出 | 允许依赖 | 明确禁止 |
|---|---|---|---|---|
| RoutePlannerNode | ArchitectureRequest | RoutePlan | policy controller、embeddings | operator、utility、optimizer |
| NativeBootstrapGraph | Request、RoutePlan | DispatchState | Programmer、Generate | 重新规划路线 |
| DispatchLoop | DispatchState | DispatchState | operator catalog | policy controller、utility |
| RouteCursorNode | DispatchState | OperatorInvocation | 无 | 修改 state |
| LogicSwitch | OperatorInvocation | 控制流 | operator names | 副作用 |
| OperatorGraph | OperatorInvocation | OperatorResult | shared Model | 修改 DispatchState |
| StateReducer | State、Result | 新 State | 纯 reducer | Model、score、utility |
| Finalize | DispatchState | ArchitectureResult | cost tracker | Evaluator、backward |
| Evaluator | Result、Context | EvaluationResult | scorer | optimizer、route |
| TrainingSignal | EvaluationResult | update/utility/skip | utility coefficient | failure penalty |
| LossUpdateNode | EvaluationResult | update result | TrainingSignal、Accumulator | operator 逻辑 |
| BatchAccumulator | Tensor、utility | optimizer update | optimizer | score和失败规则 |
| DatasetRunner | dataset | 全局结果 | RootGraph、checkpoint manager | operator 内部实现 |

---

## 三十二、项目结构

```text
applications/maas_reproduction/
├── assets/
│   ├── config/
│   │   ├── models.json
│   │   ├── bootstrap.json
│   │   ├── operators.json
│   │   ├── evaluation.json
│   │   └── prompts/
│   └── output/
│       └── <run_id>/
│
├── components/
│   ├── architecture_exec_graph.py
│   ├── route_planner_node.py
│   ├── bootstrap_graph.py
│   ├── dispatch_loop.py
│   ├── route_cursor_node.py
│   ├── state_reducer_node.py
│   ├── finalize_architecture_result_node.py
│   ├── operator_graph.py
│   └── programmer_graph.py
│
├── maas_reproduction/
│   ├── schemas.py
│   ├── state.py
│   ├── reducers.py
│   ├── training/
│   │   ├── training_signal.py
│   │   └── batch_accumulator.py
│   └── runtime/
│       ├── dataset_runner.py
│       ├── checkpoint_manager.py
│       └── seed.py
│
└── workflow/
    ├── main.py
    ├── handlers.py
    ├── utils.py
    └── main.aml
```

---

## 三十三、实施任务

## Task 0：冻结复现公式与契约

明确：

- operator catalog 和顺序；
- RoutePlan 展开规则；
- bootstrap 顺序；
- EarlyStop；
- policy logprob 聚合；
- 合法 OperatorResult；
- 合法 ArchitectureResult；
- 可信 cost 标准；
- Evaluator 可靠性标准；
- `utility = score - 3 × cost_delta`；
- policy loss 的原始 batch 计算方式。

## Task 1：建立 schemas

实现：

- FailureSource；
- ArchitectureRequest；
- RouteItem；
- RoutePlan；
- DispatchState；
- OperatorResult；
- ArchitectureResult；
- EvaluationResult；
- TrainingSignal；
- SampleResult。

## Task 2：DatasetRunner 与 RootGraph

实现：

- Train/Test builders；
- envelope；
- expected answer 隔离；
- 顺序 invocation；
- detached SampleResult；
- 状态隔离测试。

## Task 3：RoutePlannerNode

实现：

- policy controller 调用；
- 多层选择展开；
- sequence index；
- EarlyStop；
- policy logprob 聚合；
- replay 测试模式。

## Task 4：NativeBootstrapGraph

实现：

- Native Programmer；
- Native Generate；
- 初始 candidates；
- retry；
- fatal bootstrap failure；
- DispatchState 初始化。

## Task 5：DispatchLoop

实现：

- state hydration；
- `should_terminate`；
- RouteCursor；
- LogicSwitch；
- InvalidOperator；
- StateReducer；
- Loop 反馈。

## Task 6：Native operator

依次实现：

1. Generate；
2. GenerateCoT；
3. SelfRefine；
4. MultiGenerateCoT；
5. ScEnsemble；
6. Programmer。

重点验证 operator 即使失败，也尽可能通过合法 OperatorResult 返回，而不是直接击穿 Graph。

## Task 7：Finalize 与成本

实现：

- prediction；
- result validity；
- cost snapshot/delta；
- cost reliability；
- failure source；
- metadata；
- policy Tensor 保留。

## Task 8：Evaluator 与 TrainingSignal

实现：

- scorer；
- evaluation reliability；
- eligibility 判断；
- 原 MaAS utility；
- skip reason；
- 零额外 failure penalty。

## Task 9：训练生命周期

实现：

- BatchAccumulator；
- 原始 policy loss；
- partial batch flush；
- Tensor 清理；
- optimizer step。

## Task 10：checkpoint

实现：

- batch 边界保存；
- controller/optimizer；
- cursor/epoch；
- RNG；
- operator catalog；
- 禁止保存 live Tensor。

## Task 11：端到端复现

依次运行：

1. fake Model；
2. 单题真实 Test；
3. 10/50 条 Test；
4. 小 batch Train；
5. checkpoint reload；
6. 完整训练；
7. 完整测试；
8. accuracy、cost、utility、skip 和 failure 统计。

---

## 三十四、最终 Gate

## G0 — Specification

必须明确：

- RoutePlan；
- bootstrap；
- EarlyStop；
- 合法结果；
- 可信成本；
- 可靠评分；
- 原始 utility；
- policy loss 口径。

## G1 — Structural

必须满足：

- RootGraph 无普通环；
- Graph owner 正确；
- Loop keys 统一；
- state hydration 正确；
- `should_terminate` 真值表正确；
- Registry 不保存 Graph 实例；
- Agent Edge 输入使用 `pull_keys={}`；
- StateReducer 是唯一状态写入口；
- ArchitectureExecGraph 只负责编排。

## G2 — Execution Semantics

必须满足：

- RoutePlan 顺序正确；
- Planner 位于 Bootstrap 前；
- Loop 不访问 policy controller；
- operator 只返回 OperatorResult；
- Programmer retry 属于内部控制流；
- 可恢复失败产生合法 OperatorResult；
- fatal failure 不伪造结果或 cost；
- expected answer 不泄漏；
- DispatchState 保持精简。

## G3 — Training Correctness

必须满足：

- 唯一 utility 为 `score - 3 × cost_delta`；
- FailureSource 不修改 utility；
- 合法结果、可信 cost、可靠评分时正常更新；
- cost 缺失时不默认为零；
- invalid result、unreliable cost、unreliable evaluation 时跳过；
- policy logprob 保留梯度；
- BatchAccumulator 正确；
- public result 不含 autograd Tensor；
- checkpoint 不保存 live Tensor；
- Test 不更新 Controller。

---

## 三十五、必须固定的测试矩阵

| 场景 | 合法结果 | 可信 cost | 可评分 | Policy update |
|---|---:|---:|---:|---:|
| 正常成功 | 是 | 是 | 是 | 是 |
| Programmer timeout，返回结构化结果 | 是 | 是 | 是 | 是 |
| Operator 返回可评分失败结果 | 是 | 是 | 是 | 是 |
| EarlyStop 后已有候选 | 是 | 是 | 是 | 是 |
| OperatorResult 完全缺失 | 否 | 视情况 | 否 | 否 |
| Provider 崩溃 | 否 | 通常否 | 否 | 否 |
| cost snapshot 失败 | 可能 | 否 | 可选诊断 | 否 |
| Evaluator 异常 | 是 | 是 | 否 | 否 |
| policy logprob 缺失 | 可能 | 是 | 是 | 否 |
| 数据或 expected answer 损坏 | 不相关 | 可能 | 否 | 否 |

再增加一个关键不变量测试：

```text
相同 score + 相同 cost_delta
即使 failure_source 不同
→ utility 必须完全相同
```

这可以直接防止后续实现偷偷加入 failure penalty。

---

最终训练数据流冻结为：

```text
Execution
  ↓
能否形成合法 ArchitectureResult？
  ├── 否 → 记录失败 → skip policy update
  └── 是
       ↓
cost_delta 是否可信？
  ├── 否 → 记录成本异常 → skip policy update
  └── 是
       ↓
Evaluator 是否可靠评分？
  ├── 否 → 记录评估异常 → skip policy update
  └── 是
       ↓
policy_log_prob 是否存在？
  ├── 否 → skip policy update
  └── 是
       ↓
utility = score - 3.0 × cost_delta
       ↓
policy loss
       ↓
backward / optimizer.step
```

`failure_source` 始终位于这条主链旁边，负责日志、统计和执行控制，但永远不直接改变 utility。