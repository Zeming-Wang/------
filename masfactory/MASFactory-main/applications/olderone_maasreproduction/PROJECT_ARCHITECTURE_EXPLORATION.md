# MaAS Reproduction 应用架构详细探索

**项目路径**: `applications/maas_reproduction/`  
**探索日期**: 2026-08-13  
**主要目标**: 将 MaAS（Multi-Attempt Architecture Search）迁移到 MASFactory 框架中

---

## 📋 项目概览

maas_reproduction 是一个完整的机器学习模型搜索框架，用于在多个基准数据集上进行自适应架构搜索和优化。该应用利用 MASFactory 的 Graph/Loop 组件来编排整个训练流程。

**支持的数据集**:
- **GSM8K**: 数学推理问题（提取最后一个数字作为答案）
- **MATH**: 数学问题（符号化答案匹配）
- **HumanEval**: 代码生成评估

---

## 🏗️ 整体架构

### 执行流程（RootGraph）

```
RootGraph(MaASReproduction)
    ↓
[ConfigNode] 
    ↓ (settings)
[TrainingLoop] ← 核心优化循环（最多100000次迭代）
    │
    ├─→ [ArchitectureExecNode]  ← 执行MaAS架构
    │       ↓
    ├─→ [EvaluatorNode]         ← 评估结果（三种评分器）
    │       ↓
    └─→ [LossUpdateNode]        ← 控制器梯度更新
    ↓ (average_score, checkpoint_path, etc.)
[ResultNode]
    ↓ (exit)
```

### 关键执行模式

```
重复循环 (sample = 4次):
  对每个问题 (problem_index):
    1. 训练控制器采样架构操作符
    2. 执行架构对当前问题求解
    3. 评估预测结果
    4. 累积批次，达到batch_size后更新控制器
  问题完成后:
    刷新剩余批次、计算轮次统计
  所有样本完成后:
    保存检查点、生成CSV结果
    可选: TextGrad优化
```

---

## 📂 项目目录结构

### 顶级文件

| 文件 | 功能 | 关键内容 |
|------|------|--------|
| `main.py` | CLI入口点 | 参数解析、工作流构建、图执行 |
| `workflow.py` | 图定义 | build_maas_reproduction_graph() - RootGraph定义 |
| `visual_workflow.py` | 可视化路径 | MASFactory Visualizer展示（静态） |
| `runtime_graph_preview.py` | 运行时预览 | 使用真实图构建器，不初始化LLM |
| `__init__.py` | 包初始化 | 空或最小化 |

### maas_reproduction/ 子目录

#### 1. **config/** - 配置管理

| 文件 | 职责 |
|------|------|
| `settings.py` | 数据类定义：MaASPaths, MaASRuntimeSettings, OptimizerSettings |
| `experiments.py` | 数据集的MaAS搜索空间定义（操作符元组） |
| `model_config.py` | LLM模型配置解析（从ModelsConfig） |
| `__init__.py` | 导出所有配置类型 |

**关键数据结构**:
```python
# OptimizerSettings - 优化参数
- sample: 采样轮数 (default=4)
- round_number: MaAS优化轮数 (default=1)
- batch_size: 梯度更新批次大小 (default=4)
- learning_rate: Adam优化器学习率 (default=0.01)
- is_textgrad: 是否启用TextGrad优化
- opt_model_name: 优化模型（default=gpt-4o-mini）
- exec_model_name: 执行模型（default=gpt-4o-mini）

# MaASRuntimeSettings - 运行时配置
- dataset: 数据集名称
- mode: "Graph"或"Test"
- paths: MaASPaths对象
- optimizer: OptimizerSettings对象
- opt_llm_config/exec_llm_config: LLM配置对象
```

**支持的实验配置**:
```python
EXPERIMENT_CONFIGS = {
    "MATH": 操作符=("Generate", "GenerateCoT", "MultiGenerateCoT", 
                    "ScEnsemble", "Programmer", "SelfRefine", "EarlyStop"),
    "GSM8K": 相同的操作符集
    "HumanEval": 相同但"Programmer"→"Test"
}
```

---

#### 2. **nodes/** - 计算节点（CustomNode实现）

| 节点名 | 输入 | 处理 | 输出 |
|--------|------|------|------|
| **ConfigNode** | CLI参数 | 解析参数、创建MaASRuntimeSettings | settings、dataset、batch_size等 |
| **ArchitectureExecNode** | problem, entry_point, expected_answer | 调用MaAS Workflow、超时重试 | prediction, cost, logprob |
| **EvaluatorNode** | prediction, expected_answer | 三种评分器之一 | score (0或1), result_row |
| **LossUpdateNode** | score, cost, logprob | 累积批次、计算效用函数、优化步 | result_score, result_loss, result_update_performed |
| **ResultNode** | average_score, checkpoint_path | 包装最终结果 | 返回到exit |

**关键节点文件**:

##### **config_node.py**
```python
def config_forward(input_data, attributes):
    # 1. 验证CLI参数
    # 2. 创建OptimizerSettings
    # 3. 调用resolve_model_configs()获取LLM配置
    # 4. 创建MaASRuntimeSettings.from_experiment()
    # 返回: settings对象 + 展开字段
```

##### **architecture_exec_node.py**
```python
def architecture_exec_forward(input_data, attributes):
    # 1. 获取workflow对象（来自attributes）
    # 2. 调用_run_workflow_with_retry()
    #    - 支持超时（200秒）和重试（2次）
    #    - 使用AsyncRunnerContextError处理
    # 3. 对于HumanEval: workflow(problem, entry_point, run_directory)
    #    对于数学: workflow(problem)
    # 返回: prediction, cost, logprob + 原始问题字段
```

##### **evaluator_node.py**
```python
def evaluator_forward(input_data, attributes):
    dataset = settings.dataset
    if dataset == "GSM8K":
        scorer = GSM8KScorer()
        score = 0或1 (数字精确匹配)
    elif dataset == "MATH":
        scorer = MATHScorer()
        score = 0或1 (符号化匹配)
    elif dataset == "HumanEval":
        scorer = HumanEvalScorer()
        score = 1或0 (测试通过/失败)
    
    # 如果失败: append_mismatch_log()
    返回: score, result_row (用于CSV写入)
```

##### **loss_update_node.py**
```python
def loss_update_forward(input_data, attributes):
    # 累积批次数据
    attributes["batch_logprobs"].append(logprob_tensor)
    attributes["batch_scores"].append(score)
    attributes["batch_costs"].append(cost_delta)
    
    # 当batch满时:
    if len(batch_logprobs) >= batch_size:
        loss = -(logprobs * (scores - 3*costs)).mean()
        if mode == "Graph":
            loss.backward()
            optimizer.step()
            update_performed = True
    
    返回: result_score, result_loss, result_update_performed
```

##### **training_controller.py**
```python
# 循环控制函数（loop.terminate_condition_function）
def training_controller(input_data, attributes):
    # 1. 清除result_*前缀的键（下一轮重用）
    # 2. 获取当前问题和重复计数
    
    if problem_index >= len(problems):
        # 轮次结束:
        flush_remaining_batch()  # 最后批次优化
        _finish_repetition()    # 计算轮次统计
        
        if repetition >= max_repetitions:
            _write_final_result()  # 保存检查点和CSV
            return True  # 停止循环
        
        # 可选TextGrad优化
        _maybe_run_textgrad()
        repetition += 1
        problem_index = 0
        return False  # 继续下一轮
    
    # 设置当前问题到input_data
    problem = problems[problem_index]
    input_data["problem"] = problem["question/prompt"]
    input_data["entry_point"] = problem["entry_point"]  # HumanEval only
    input_data["expected_answer"] = problem["answer/test"]
    
    return False  # 继续处理该问题
```

##### **artifact_writer.py**
```python
def author_round_directory(settings):
    # 返回round_${round_number}目录路径
    
def append_mismatch_log(log_directory, problem, expected, prediction):
    # 写入JSON失败日志: question, right_answer, model_output等
    
def write_results_csv(log_directory, columns, rows, average_score):
    # 写入CSV: ${average_score}_${timestamp}.csv
    # 排除cost列（保留其他字段）
    
def append_round_summary(results_file, round_number, score, avg_cost):
    # 追加轮次统计到JSON文件
```

---

#### 3. **runtime/** - 运行时初始化与执行

| 文件 | 职责 |
|------|------|
| `initializer.py` | 构建runtime attributes: controller, optimizer, workflow, problems |
| `data_loader.py` | 从JSONL文件加载数据集 |
| `async_runner.py` | 同步→异步桥接 |

**initializer.py** - build_runtime_attributes()
```python
def build_runtime_attributes(settings, specific_indices=None):
    device = cuda or cpu
    controller = MultiLayerController().to(device)  # 4层OperatorSelector
    optimizer = torch.optim.Adam(controller.params, lr=learning_rate)
    operator_embeddings = _load_operator_embeddings()  # 从assets加载
    
    # 如果mode="Test": 从checkpoint加载controller.state_dict()
    
    workflow_type = load_workflow_class(settings)  # 动态导入assets模块
    architecture_workflow = workflow_type(
        name=dataset,
        llm_config=exec_llm_config,
        controller=controller,
        operator_embeddings=operator_embeddings
    )
    
    problems = load_problems()  # 从dataset_file加载
    
    返回attributes字典:
        settings, controller, optimizer, workflow,
        problems, problem_index=0, repetition=1,
        batch_logprobs=[], batch_scores=[], all_scores=[],
        run_directory, device, ...
```

**async_runner.py**
```python
def run_async_once(coro):
    """同步节点调用异步workflow的桥接"""
    try:
        asyncio.get_running_loop()  # 检查是否已在异步上下文
    except RuntimeError:
        return asyncio.run(coro)  # 创建新event loop
    
    raise AsyncRunnerContextError("必须从同步上下文调用")
```

---

#### 4. **benchmarks/** - 评分器（数据集特定）

| 评分器 | 算法 |
|--------|------|
| **GSM8KScorer** | `extract_number()`: 用正则表达式提取最后一个数字<br/>`calculate_score()`: 数值误差≤1e-6→1分，否则0分 |
| **MATHScorer** | `extract_model_answer()`: 从 `\boxed{}` 提取或从最后一句提取<br/>`symbolic_equal()`: 使用sympy进行符号化比较，必要时数值比较 |
| **HumanEvalScorer** | `check_solution()`: 执行代码+测试用例<br/>15秒超时，线程式实现<br/>返回: (PASS/FAIL, 消息) |

**关键特性**:
- **GSM8K**: 提取数字，精确比较
- **MATH**: 符号化数学等价性（sympy），支持分数/百分比
- **HumanEval**: 代码执行，超时保护

---

#### 5. **models/** - 搜索空间与采样

**OperatorSelector** (torch.nn.Module)
```python
class OperatorSelector(torch.nn.Module):
    def __init__(self, input_dim=384, hidden_dim=32, is_first_layer=False):
        # query_encoder: Linear(384 → 32)
        # operator_encoder: 
        #   - 首层: Linear(384 → 32)
        #   - 后层: Linear(384*2 → 32)  # 与前层选择连接
    
    def forward(query_embed, operators_embed, prev_operators_embed=None):
        # 1. 归一化query和operators
        # 2. 计算点积相似度: scores = query @ operators^T
        # 3. softmax → probs, log_softmax → log_probs
        # 返回: (log_probs, probs)
```

**MultiLayerController** (torch.nn.Module)
```python
class MultiLayerController(torch.nn.Module):
    def __init__(self, num_layers=4, input_dim=384, hidden_dim=32):
        self.layers = ModuleList([
            OperatorSelector(..., is_first_layer=(i==0))
            for i in range(4)
        ])
    
    def forward(query, operators_embedding, selection_operator_names):
        query_embedding = sentence_encoder(query)  # all-MiniLM-L6-v2: 384维
        
        for layer_idx, layer in enumerate(layers):
            log_probs, probs = layer(query_emb, ops_emb, prev_ops)
            
            selected_indices = sample_operators(probs_1d, threshold=0.3)
            selected_names = [ops[idx] for idx in selected_indices]
            
            # 特殊规则:
            # - 首层若选EarlyStop → 强制选Generate
            # - 首层若未选任何generate → 强制选Generate
            # - 后层: 维持前层至少一个generate
            
            prev_ops = operators_embedding[selected_indices]
        
        返回: selected_names (最后一层的选择)
```

**Utility函数**:
```python
class SentenceEncoder(torch.nn.Module):
    def forward(sentence):
        embeddings = sentence_transformers.encode(sentence)
        返回: torch.Tensor(384维)

def sample_operators(probs, threshold=0.25):
    """样本操作符直到累积概率≥threshold"""
    selected = []
    cumulative = 0
    for sampled_idx in multinomial_sample(probs):
        if probs[sampled_idx] < cumulative:
            break
        selected.append(sampled_idx)
        cumulative += probs[sampled_idx]
    
    if selected为空: selected = [argmax(probs)]
    返回: selected_indices

def get_sentence_embedding(sentence):
    # 一次性方式（非Module）
    return torch.tensor(encode(sentence))
```

---

#### 6. **graphs/** - 图构建器

**training_loop.py**
```python
CONTROLLER_TO_ARCHITECTURE_KEYS = {
    "problem", "entry_point", "expected_answer", "problem_index"
}

ARCHITECTURE_TO_EVALUATOR_KEYS = {
    "problem", "entry_point", "expected_answer", 
    "prediction", "cost", "logprob", "problem_index"
}

EVALUATOR_TO_LOSS_KEYS = {
    "score", "cost", "logprob", "problem_index"
}

LOSS_TO_CONTROLLER_KEYS = {
    "result_score", "result_cost", "result_logprob",
    "result_loss", "result_update_performed", "result_problem_index"
}

TRAINING_LOOP_PUSH_KEYS = {
    "average_score", "round", "checkpoint_path", 
    "result_path", "runtime_metadata"
}

def attach_training_loop_body(loop):
    # 在Loop中创建3个节点
    architecture_node = loop.create_node(
        CustomNode, "ArchitectureExecNode", forward=architecture_exec_forward)
    evaluator_node = loop.create_node(
        CustomNode, "EvaluatorNode", forward=evaluator_forward)
    loss_update_node = loop.create_node(
        CustomNode, "LossUpdateNode", forward=loss_update_forward)
    
    # 连接
    loop.edge_from_controller(architecture_node, CONTROLLER_TO_ARCHITECTURE_KEYS)
    loop.create_edge(architecture_node, evaluator_node, ...)
    loop.create_edge(evaluator_node, loss_update_node, ...)
    loop.edge_to_controller(loss_update_node, LOSS_TO_CONTROLLER_KEYS)
```

---

## 🔑 核心数据流

### 数据路径示例（GSM8K）

```
CLI Input:
{
  "application_root": "/path/to/app",
  "dataset": "GSM8K",
  "sample": 4,
  "batch_size": 4,
  "learning_rate": 0.01,
  "opt_model_name": "gpt-4o-mini",
  "exec_model_name": "gpt-4o-mini"
}
    ↓
ConfigNode:
{
  "settings": MaASRuntimeSettings(...),
  "dataset": "GSM8K",
  "batch_size": 4,
  "model_config": {...},
  "paths": {...}
}
    ↓
build_runtime_attributes():
{
  "controller": MultiLayerController(),
  "optimizer": Adam(...),
  "problems": [{"question": "...", "answer": "42"}, ...],
  "architecture_workflow": Workflow(...),
  ...
}
    ↓
TrainingLoop (重复 4 次样本):
    第1问题:
    problem="1+1=?"
    → ArchitectureExecNode → prediction="2", logprob=-0.1
    → EvaluatorNode → score=1.0
    → LossUpdateNode → batch=[1], logprobs=[-0.1]
    
    第4问题:
    → batch满 → optimizer.step() → 权重更新
    
    所有问题完成:
    → 保存checkpoint.pth
    → 写入results_${avg_score}_${timestamp}.csv
    ↓
ResultNode:
{
  "average_score": 0.95,
  "checkpoint_path": ".../round_1/checkpoint.pth",
  "result_path": ".../runs/GSM8K/Graph/..."
}
```

---

## 🔄 特殊流程

### 1. TextGrad优化（可选）

```python
_maybe_run_textgrad(settings, attributes, current_repetition_score)
    if settings.optimizer.is_textgrad:
        # 使用previous_repetition_score和current_repetition_score
        # 进行优化模型提示词的梯度更新
```

### 2. 批次刷新策略

```python
_finish_repetition():
    # 在重复轮次边界处
    flush_remaining_batch(attributes)  # 强制优化最后部分批次
    
    # 计算轮次统计
    current_repetition_score = average(current_repetition_scores)
    append_round_summary(..., current_repetition_score)
```

### 3. 检查点管理

```python
checkpoint_path = settings.paths.controller_checkpoint(
    dataset="GSM8K",
    round_number=1,
    sample=4
)
# 返回: ${checkpoint_root}/GSM8K/round_1_sample_4/controller.pth

# 模式 == "Test" 时: 从checkpoint加载
torch.load(checkpoint_path, map_location=device)
```

---

## 📊 关键参数与默认值

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `dataset` | 无 | {GSM8K, MATH, HumanEval} | 必需 |
| `mode` | "Graph" | {Graph, Test} | Graph: 训练, Test: 评估 |
| `sample` | 4 | > 0 | 采样轮数（重复优化次数） |
| `round` | 1 | > 0 | MaAS优化轮数 |
| `batch_size` | 4 | > 0 | 控制器梯度更新批次 |
| `learning_rate` | 0.01 | > 0 | Adam优化器学习率 |
| `opt_model_name` | gpt-4o-mini | - | 用于生成优化提示词 |
| `exec_model_name` | gpt-4o-mini | - | 用于执行架构 |
| `is_textgrad` | False | {True, False} | 启用TextGrad元优化 |

---

## 📁 完整文件清单

### 顶级文件（5个）
```
main.py                           ← CLI入口
workflow.py                       ← 图定义
visual_workflow.py               ← 可视化路径（无LLM）
runtime_graph_preview.py         ← 运行时图预览
__init__.py                      ← 包初始化
```

### maas_reproduction/config/ （4个）
```
settings.py                      ← 数据类: MaASPaths, MaASRuntimeSettings
experiments.py                   ← EXPERIMENT_CONFIGS (操作符定义)
model_config.py                  ← resolve_model_configs()
__init__.py                      ← __all__导出
```

### maas_reproduction/nodes/ （8个）
```
config_node.py                   ← config_forward()
architecture_exec_node.py        ← architecture_exec_forward()
evaluator_node.py                ← evaluator_forward()
loss_update_node.py              ← loss_update_forward()
training_controller.py           ← training_controller() 循环条件
artifact_writer.py               ← 结果写入
result_node.py                   ← result_forward()
__init__.py                      ← __all__导出
```

### maas_reproduction/runtime/ （4个）
```
initializer.py                   ← build_runtime_attributes()
data_loader.py                   ← load_problems(), load_jsonl_data()
async_runner.py                  ← run_async_once()
__init__.py                      ← 空或导出
```

### maas_reproduction/benchmarks/ （4个）
```
gsm8k.py                         ← GSM8KScorer (数字提取+比较)
math.py                          ← MATHScorer (符号化匹配)
humaneval.py                     ← HumanEvalScorer (代码执行)
__init__.py                      ← __all__导出
```

### maas_reproduction/models/ （4个）
```
controller.py                    ← MultiLayerController, OperatorSelector
utils.py                         ← SentenceEncoder, sample_operators()
__init__.py                      ← __all__导出
```

### maas_reproduction/graphs/ （2个）
```
training_loop.py                 ← attach_training_loop_body()
__init__.py                      ← 空或导出
```

---

## 🎯 关键代码流程

### 执行流程（从main.py开始）

```python
# 1. 参数解析
args = _parse_args()

# 2. 调用run()函数
output = run({
    "application_root": ...,
    "dataset": args.dataset,
    "sample": args.sample,
    ...
})

# 3. run()函数内部:
#    a) ConfigNode处理CLI参数
config = config_forward(input_data)

#    b) 初始化运行时对象
runtime_attributes = build_runtime_attributes(config["settings"])

#    c) 构建并执行图
graph = build_maas_reproduction_graph()
graph.build()
output, _attributes = graph.invoke(input_data, attributes=runtime_attributes)

# 4. 打印JSON结果
print(json.dumps(output, indent=2))
```

### 损失函数公式

```python
# 在loss_update_node.py中
utility = score - 3 * cost_delta

loss = -(logprobs * utility).mean()  # 负最大似然估计
# 符号: 向上优化log_prob与效用的乘积
```

---

## 🔧 环境与依赖

**Python版本**: 3.10+

**关键依赖**:
- `torch>=2.1.0`: 张量计算、优化器
- `sentence-transformers`: all-MiniLM-L6-v2 嵌入（384维）
- `tenacity`: 重试机制
- `tiktoken`: token计数
- `sympy`: 符号数学（MATH数据集）
- `pandas`: 数据处理
- `tqdm`: 进度条

**GPU**: 可选（默认cuda/cpu自动选择）

---

## 🎓 核心设计原则

1. **不修改MASFactory核心**: 使用现有的RootGraph、Loop、CustomNode
2. **保留MaAS行为**: 
   - 搜索空间（操作符）保持不变
   - 控制器架构保持不变
   - 损失函数公式保持一致
3. **属性传递**: 运行时对象通过Loop attributes传递，不通过Edge消息
4. **异步桥接**: MaAS Workflow是异步的，通过AsyncRunnerContextError处理
5. **分阶段实现**: Phase 1不嵌套operator-level循环，所有操符在单个ArchitectureExecNode中调用

---

## 📈 扩展与改进方向

1. **性能优化**:
   - 并行化多问题处理
   - 缓存operator_embeddings
   - 批量评分

2. **新数据集**:
   - 添加BENCHMARK_SCORERS字典
   - 实现新的Scorer类

3. **高级优化**:
   - TextGrad集成
   - 多卡分布式
   - 模型蒸馏

4. **监控与调试**:
   - wandb/tensorboard集成
   - 更详细的日志
   - 可视化训练曲线

---

**本文档生成**: 2026-08-13  
**维护者**: MaAS框架迁移团队
