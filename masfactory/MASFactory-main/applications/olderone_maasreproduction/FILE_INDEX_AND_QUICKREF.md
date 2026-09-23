# MaAS Reproduction 文件索引与快速查找

**用途**: 快速定位文件、理解职责分工、跟踪数据流

---

## 📑 完整文件索引

### 顶层目录（5个文件）

```
maas_reproduction/
├── main.py                    [CLI入口]
├── workflow.py                [图定义]
├── visual_workflow.py         [可视化路径]
├── runtime_graph_preview.py   [运行时预览]
└── __init__.py                [包初始化]
```

| 文件 | 大小 | 行数 | 职责 | 关键函数 |
|------|------|------|------|---------|
| `main.py` | ~1.5KB | 60+ | CLI参数解析、工作流调用 | `main()`, `run()`, `_parse_args()` |
| `workflow.py` | ~1.5KB | 70+ | RootGraph定义 | `build_maas_reproduction_graph()` |
| `visual_workflow.py` | ~2KB | 70+ | 可视化图构建（仅拓扑） | `build_maas_reproduction_visual_graph()` |
| `runtime_graph_preview.py` | ~0.5KB | 20+ | 运行时图预览 | `build_runtime_preview_graph()` |
| `__init__.py` | ~0.1KB | 5- | 包初始化 | (empty) |

---

### config/ 目录（4个文件）

```
config/
├── settings.py            [数据类定义]
├── experiments.py         [实验配置]
├── model_config.py        [模型配置解析]
└── __init__.py            [导出]
```

| 文件 | 内容 | 关键类/函数 | 输入 | 输出 |
|------|------|-----------|------|------|
| **settings.py** | 路径管理、配置容器 | `MaASPaths`, `OptimizerSettings`, `MaASRuntimeSettings` | 应用根目录、环境变量 | 规范化的配置对象 |
| **experiments.py** | 搜索空间定义 | `EXPERIMENT_CONFIGS: dict`, `ExperimentConfig` | (静态) | 数据集→操作符映射 |
| **model_config.py** | LLM配置解析 | `resolve_model_configs()` | opt_model_name, exec_model_name | (opt_llm_config, exec_llm_config) |
| **__init__.py** | 公共导出 | `__all__` | - | 导出所有配置类型 |

**数据类依赖关系**:
```
CLI input (dict)
    ↓
OptimizerSettings + resolve_model_configs()
    ↓
MaASRuntimeSettings.from_experiment()
    ↓
config_forward() 返回值
    ↓
build_runtime_attributes() 消费
```

---

### nodes/ 目录（8个文件）

```
nodes/
├── config_node.py             [参数→配置]
├── architecture_exec_node.py  [执行工作流]
├── evaluator_node.py          [评分]
├── loss_update_node.py        [梯度更新]
├── training_controller.py     [循环控制]
├── artifact_writer.py         [结果写入]
├── result_node.py             [最终输出]
└── __init__.py                [导出]
```

| 节点 | 输入数据 | 处理 | 输出数据 | 特殊属性写入 |
|------|---------|------|---------|------------|
| **ConfigNode** | application_root, dataset, ... | 参数验证、配置构建 | settings, dataset, batch_size, ... | - |
| **ArchitectureExecNode** | problem, entry_point, expected_answer, problem_index | 调用workflow()、超时重试 | prediction, cost, logprob, problem_index | - |
| **EvaluatorNode** | prediction, expected_answer, entry_point | 数据集评分器 | score, cost, logprob, problem_index | `sample_results[]`, `result_columns` |
| **LossUpdateNode** | score, cost, logprob, problem_index | 批次累积、优化 | result_score, result_loss, result_update_performed | `batch_logprobs[]`, `batch_scores[]`, `batch_costs[]` |
| **TrainingController** | (循环条件函数) | 问题迭代、轮次管理 | result_score, result_cost, ... | `problem_index`, `repetition`, `all_scores[]` |
| **ArtifactWriter** | 无（工具函数集） | 日志写入、CSV生成、检查点保存 | Path对象 | 文件系统 |
| **ResultNode** | average_score, checkpoint_path, ... | 结果包装 | average_score, round, checkpoint_path, ... | - |

**关键节点函数签名**:
```python
config_forward(input_data: dict, attributes: dict) → dict
architecture_exec_forward(input_data: dict, attributes: dict) → dict
evaluator_forward(input_data: dict, attributes: dict) → dict
loss_update_forward(input_data: dict, attributes: dict) → dict
training_controller(input_data: dict, attributes: dict) → bool  # 循环条件
result_forward(input_data: dict, attributes: dict) → dict
```

---

### runtime/ 目录（4个文件）

```
runtime/
├── initializer.py         [对象构建]
├── data_loader.py         [数据集加载]
├── async_runner.py        [异步桥接]
└── __init__.py            [导出]
```

| 文件 | 主要函数 | 输入 | 输出 | 依赖 |
|------|---------|------|------|------|
| **initializer.py** | `build_runtime_attributes(settings, specific_indices, workflow_class)` | MaASRuntimeSettings | attributes: dict | torch, MultiLayerController, load_workflow_class |
| **data_loader.py** | `load_problems(settings, specific_indices)` | settings.dataset_file | problems: list[dict] | json, Path |
| **async_runner.py** | `run_async_once(coro)` | 协程 | T(返回值) | asyncio |

**运行时属性结构**:
```python
{
    # 核心对象
    "settings": MaASRuntimeSettings,
    "controller": MultiLayerController,
    "optimizer": torch.optim.Adam,
    "operator_embeddings": torch.Tensor,
    "architecture_workflow": Workflow(MaAS),
    
    # 数据
    "problems": list[dict],
    
    # 循环状态
    "problem_index": int,
    "repetition": int,
    
    # 批次缓冲
    "batch_logprobs": list[torch.Tensor],
    "batch_scores": list[float],
    "batch_costs": list[float],
    
    # 统计
    "all_scores": list[float],
    "current_repetition_scores": list[float],
    "sample_results": list[list],
    
    # 配置
    "batch_size": int,
    "device": torch.device,
    "run_directory": Path,
}
```

---

### benchmarks/ 目录（4个文件）

```
benchmarks/
├── gsm8k.py       [GSM8K评分器]
├── math.py        [MATH评分器]
├── humaneval.py   [HumanEval评分器]
└── __init__.py    [导出]
```

| 数据集 | 文件 | 评分器类 | 输入 | 处理方法 | 分数 |
|--------|------|---------|------|---------|------|
| **GSM8K** | gsm8k.py | `GSM8KScorer` | expected_number, prediction | `extract_number()` + 数值比较 | 0或1 |
| **MATH** | math.py | `MATHScorer` | expected_output, prediction | `extract_model_answer()` + 符号比较 | 0或1 |
| **HumanEval** | humaneval.py | `HumanEvalScorer` | solution, test, entry_point | `exec()` + 测试执行（15s超时） | 0或1 |

**评分流程**:
```
GSM8K:      问题答案 →extract_number→ 数字 →isclose(±1e-6)→ 分数
MATH:       问题答案 →extract_model_answer→ 表达式 →sympy/数值→ 分数
HumanEval:  解决方案+测试 →exec()→ check(fn) →PASS/FAIL→ 分数
```

---

### models/ 目录（4个文件）

```
models/
├── controller.py  [神经网络控制器]
├── utils.py       [嵌入与采样]
└── __init__.py    [导出]
```

| 文件 | 类/函数 | 职责 | 输入 | 输出 |
|------|--------|------|------|------|
| **controller.py** | `OperatorSelector` | 单层注意力+softmax选择 | query_embed, operators_embed | log_probs, probs |
| **controller.py** | `MultiLayerController` | 4层级联选择+特殊规则 | query (str), operators_embed, names | 最终选择的操作符列表 |
| **utils.py** | `SentenceEncoder` | 冻结BERT编码 | 文本 | 384维张量 |
| **utils.py** | `sample_operators(probs, threshold)` | 多项采样直到累积≥25% | 概率分布 | 选中索引 |

**架构选择链**:
```
问题文本
    ↓ sentence_encoder (all-MiniLM-L6-v2)
384维嵌入
    ↓ MultiLayerController [4层]
    第1层: OperatorSelector → Generate (forced)
    第2层: OperatorSelector → [Generate, SelfRefine, ...]
    第3层: OperatorSelector → [GenerateCoT, ...]
    第4层: OperatorSelector → [Generate, ...]
    ↓
最终操作符序列
```

---

### graphs/ 目录（2个文件）

```
graphs/
├── training_loop.py   [TrainingLoop内部图]
└── __init__.py        [导出]
```

| 文件 | 函数 | 职责 | 修改 |
|------|------|------|------|
| **training_loop.py** | `attach_training_loop_body(loop)` | 在Loop中创建3个节点并连接 | Loop对象 |

**TrainingLoop内部图**:
```
TrainingLoop(terminate_condition=training_controller, max_iterations=100000)
    ├─→ ArchitectureExecNode
    ├─→ EvaluatorNode
    └─→ LossUpdateNode
    
数据流:
controller → [problem, entry_point, ...] → ArchitectureExecNode
    ↓
[prediction, cost, logprob] → EvaluatorNode
    ↓
[score] → LossUpdateNode
    ↓
[result_score, result_loss, result_update_performed] → controller
```

---

## 🔍 快速查找表

### 按功能查找

#### 我想要...添加一个新的数据集

```
1. config/experiments.py - 添加ExperimentConfig
2. benchmarks/new_dataset.py - 实现Scorer类
3. benchmarks/__init__.py - 添加导入
4. nodes/evaluator_node.py - 添加elif分支
```

#### 我想要...修改控制器架构

```
models/controller.py
    - OperatorSelector: 改变线性层维度
    - MultiLayerController: 改变层数或添加特殊规则
```

#### 我想要...改变优化器

```
nodes/loss_update_node.py
    - _compute_loss(): 改变效用公式
    - loss_update_forward(): 改变优化器逻辑
```

#### 我想要...修改评分策略

```
benchmarks/{gsm8k,math,humaneval}.py
    - Scorer.calculate_score(): 改变分数计算
```

#### 我想要...添加新的CLI参数

```
1. main.py - _parse_args()中添加参数
2. nodes/config_node.py - config_forward()中处理
3. config/settings.py - 可能需要扩展配置类
```

#### 我想要...跟踪值从何而来

```
loss_update_node.py:
    logprob ← architecture_exec_node.py:logprob ← workflow.__call__()
    cost ← architecture_exec_node.py:cost ← workflow.__call__()
    score ← evaluator_node.py:score ← Scorer.calculate_score()
```

---

### 按执行阶段查找

#### 阶段1: 参数解析
```
main.py → _parse_args()
    ↓
main.py → run(input_data)
    ↓
nodes/config_node.py → config_forward()
```

#### 阶段2: 运行时初始化
```
nodes/config_node.py → return settings
    ↓
main.py → build_runtime_attributes()
    ↓
runtime/initializer.py → build_runtime_attributes()
    ↓
return attributes (controller, optimizer, problems, etc.)
```

#### 阶段3: 图执行
```
main.py → graph.invoke(input_data, attributes)
    ↓
RootGraph(
    ConfigNode → TrainingLoop(100000次迭代)
        每次迭代:
        training_controller() → 返回False继续，True停止
        ArchitectureExecNode → prediction, cost, logprob
        EvaluatorNode → score
        LossUpdateNode → 累积、优化
    → ResultNode
)
```

#### 阶段4: 结果输出
```
training_controller → _write_final_result()
    ↓
artifact_writer.py:
    - write_results_csv()
    - append_round_summary()
    - torch.save(checkpoint)
    ↓
ResultNode → return to exit
```

---

### 按数据流查找

#### 输入数据流向
```
CLI input (dict)
    ↓ ConfigNode
MaASRuntimeSettings
    ↓ build_runtime_attributes()
attributes: dict
    ↓ TrainingLoop
problem_index → 查询problems数组
    ↓ ArchitectureExecNode
prediction, cost, logprob
    ↓ EvaluatorNode
score
    ↓ LossUpdateNode
result_score, result_loss
    ↓ training_controller
all_scores[], sample_results[]
    ↓ ResultNode
final output (dict)
```

#### 属性流向（通过Loop attributes）
```
build_runtime_attributes()
    ↓
attributes = {
    settings, controller, optimizer,
    problems, batch_logprobs, batch_scores, all_scores, ...
}
    ↓
ArchitectureExecNode: 读settings, problems, workflow
EvaluatorNode: 读settings, 写sample_results, result_columns
LossUpdateNode: 读settings, optimizer, batch_size
              写batch_logprobs, batch_scores, batch_costs, all_scores
training_controller: 读问题索引, 重复计数
                    写problem_index, repetition
    ↓
最终attributes包含所有统计信息
```

---

## 📊 关键变量与数据结构

### input_data (Dict)

**ConfigNode输入**:
```python
{
    "application_root": Path,
    "dataset": "GSM8K"|"MATH"|"HumanEval",
    "mode": "Graph"|"Test",
    "sample": int,
    "round_number": int,
    "batch_size": int,
    "learning_rate": float,
    "opt_model_name": str,
    "exec_model_name": str,
    "is_textgrad": bool,
    "indices": list[int] | None,
}
```

**TrainingLoop输入（每次迭代）**:
```python
{
    "problem": str,              # 当前问题文本
    "entry_point": str,          # HumanEval only
    "expected_answer": str,      # 答案/测试代码
    "problem_index": int,        # 问题序号
    
    # result_*键（下一轮重用）
    "result_score": float,
    "result_cost": float,
    "result_logprob": float,
    "result_loss": float | None,
    "result_update_performed": bool,
    "result_problem_index": int,
}
```

### attributes (Dict)

**关键属性**:
```python
{
    # 核心对象
    "settings": MaASRuntimeSettings,
    "controller": torch.nn.Module,
    "optimizer": torch.optim.Optimizer,
    "operator_embeddings": torch.Tensor,
    "architecture_workflow": Workflow,
    "problems": list[dict],
    
    # 循环状态
    "problem_index": int,      # [0, len(problems))
    "repetition": int,         # [1, sample]
    "batch_size": int,
    
    # 批次缓冲（当len >= batch_size时更新）
    "batch_logprobs": list[torch.Tensor],
    "batch_scores": list[float],
    "batch_costs": list[float],
    
    # 累积统计
    "all_scores": list[float],              # 所有问题得分
    "current_repetition_scores": list[float],  # 当前轮次
    "sample_results": list[list],           # CSV行数据
    
    # 配置与路径
    "device": torch.device,
    "run_directory": Path,
    "result_columns": list[str],
    "previous_cost": float,
    
    # 可选
    "textgrad_events": list,
}
```

---

## 🔗 关键依赖关系

### 导入关系

```
workflow.py
    ├─ graphs/training_loop.py
    ├─ nodes/{config,result}_node.py
    └─ nodes/training_controller.py

runtime/initializer.py
    ├─ models/controller.py
    ├─ runtime/data_loader.py
    └─ 动态导入: assets.optimized.{dataset}.{split}.graph

nodes/evaluator_node.py
    ├─ benchmarks/{gsm8k,math,humaneval}.py
    └─ nodes/artifact_writer.py

nodes/loss_update_node.py
    └─ torch (张量操作)

models/controller.py
    ├─ models/utils.py (SentenceEncoder)
    └─ torch (神经网络)
```

### 配置依赖

```
CLI args
    ↓
config/settings.py (MaASRuntimeSettings)
    ↓
config/model_config.py (resolve_model_configs)
config/experiments.py (EXPERIMENT_CONFIGS)
    ↓
nodes/config_node.py (config_forward)
    ↓
runtime/initializer.py (build_runtime_attributes)
    ├─ models/controller.py (MultiLayerController)
    ├─ runtime/data_loader.py (load_problems)
    └─ 动态加载: assets.optimized
```

---

## 🎯 常见操作查找

### 查找：如何读取问题数据？
```python
# 在runtime/initializer.py
problems = load_problems(settings, specific_indices=specific_indices)

# 在nodes/training_controller.py
problem = problems[problem_index]
_write_problem_message(input_data, dataset, problem, problem_index)

# 访问问题字段（取决于数据集）
if dataset == "GSM8K":
    input_data["problem"] = problem["question"]
    input_data["expected_answer"] = problem["answer"]
elif dataset == "HumanEval":
    input_data["problem"] = problem["prompt"]
    input_data["entry_point"] = problem["entry_point"]
    input_data["expected_answer"] = problem["test"]
```

### 查找：如何调用MaAS工作流？
```python
# 在nodes/architecture_exec_node.py
workflow = attributes["architecture_workflow"]

if settings.dataset == "HumanEval":
    result = _run_workflow_with_retry(workflow, problem, entry_point, run_directory)
else:
    result = _run_workflow_with_retry(workflow, problem)

prediction, cost, logprob = result
```

### 查找：如何评分？
```python
# 在nodes/evaluator_node.py
if dataset == "GSM8K":
    scorer = GSM8KScorer()
    expected_number = scorer.extract_number(expected_answer)
    score, extracted_output = scorer.calculate_score(expected_number, prediction)
elif dataset == "MATH":
    scorer = MATHScorer()
    score, extracted_output = scorer.calculate_score(expected_answer, prediction)
elif dataset == "HumanEval":
    scorer = HumanEvalScorer(log_path=settings.run_directory)
    result = scorer.check_solution(prediction, expected_answer, entry_point)
    score = 1.0 if result[0] == scorer.PASS else 0.0
```

### 查找：如何访问控制器参数？
```python
# 在任何节点中
controller = attributes["controller"]  # MultiLayerController

# 保存
torch.save(controller.state_dict(), checkpoint_path)

# 加载
controller.load_state_dict(torch.load(checkpoint_path, map_location=device))

# 获取参数
for param in controller.parameters():
    # param是torch.nn.Parameter
```

### 查找：如何修改损失函数？
```python
# 在nodes/loss_update_node.py的_compute_loss()
def _compute_loss(attributes):
    logprobs = torch.stack(attributes["batch_logprobs"])
    scores = torch.tensor(attributes["batch_scores"], dtype=torch.float32)
    costs = torch.tensor(attributes["batch_costs"], dtype=torch.float32)
    
    # 当前: utility = score - 3*cost
    # 修改这行来改变公式
    utility = scores - 3 * costs  # ← 修改这里
    
    loss = -(logprobs * utility).mean()
    return loss
```

---

**文档完成**: 2026-08-13
**维护**: MaAS迁移团队
