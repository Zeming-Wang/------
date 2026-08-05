# 基于 MASFactory 的 Sparse Multi-Agent Debate 论文复现说明

## 1. 论文方法概述

复现论文为 **Improving Multi-Agent Debate with Sparse Communication Topology**。论文关注 Multi-Agent Debate, MAD 中的通信成本问题：传统 MAD 在辩论阶段通常让每个 Agent 看到其他所有 Agent 的上一轮回答，这种全连接通信可以提升推理质量，但会随着 Agent 数量和辩论轮数增加带来很高的输入 token 成本。

论文提出将 Agent 间通信建模为图：

```text
G = (V, E)
```

其中 `V` 是 Agent 集合，`E` 是通信边。如果两个 Agent 之间存在边，则它们可以在辩论阶段参考对方上一轮回答。图密度定义为：

```text
D = 2|E| / (|V|(|V| - 1))
```

`D = 1` 表示全连接图。论文核心思想是：使用稀疏通信拓扑，让 Agent 只读取邻居回答，在尽量保持任务表现的同时降低输入上下文长度和成本。

## 2. 论文标准流程

论文中的 MAD 流程可抽象为三步：

1. **Individual Response Generation**：第一轮所有 Agent 独立回答同一问题，不互相通信。
2. **Multi-Agent Debate**：从第二轮开始，每个 Agent 根据通信拓扑读取允许可见的上一轮回答，并更新自己的答案。
3. **Reaching Consensus**：最后收集所有 Agent 的最终答案，通过多数投票得到系统输出。

全连接 MAD 和 Sparse MAD 的差异不在于 Agent 数量，而在于辩论阶段每个 Agent 可见的邻居集合。

## 3. MASFactory Node / Edge / Graph 抽象

本项目将论文方法拆成两个层次的 MASFactory 图。

### 3.1 实验 SOP 图

文件：`sparse_mad/visual_workflow.py`

核心函数：

```python
build_visual_comparison_graph(client=...)
```

该图展示完整复现实验流程：

```text
entry
  -> PrepareDataset
  -> BuildFullTopology_D_equals_1
  -> RunFullyConnectedMAD
  -> BuildNeighborTopology
  -> RunNeighborConnectedMAD
  -> ComputeAccuracyAndCostSaving
  -> FormatExperimentReport
  -> exit
```

这些 MASFactory `Node` 表示实验步骤，MASFactory `Edge` 负责传递数据，例如 `dataset`、`num_agents`、`max_rounds`、两种拓扑的评测结果、`accuracy`、`token_cost` 和 `cost_saving`。

### 3.2 Agent 通信拓扑图

文件：`sparse_mad/agent_topology_hex_visual.py`

核心函数：

```python
build_hex_topology_visual_graph()
```

该图直接展示论文通信结构：

```text
Full_Agent_1..6: 全连接拓扑 K6
Neighbor_Agent_1..6: 环形邻居拓扑 C6
```

在这个视角中：

```text
Agent = Node
Agent 间通信关系 = Edge
```

MASFactory 的 `RootGraph` 是有向无环图，因此可视化文件用有向边展示论文中的无向通信关系。箭头方向只服务于 Visualizer 展示，不表示论文方法只能单向通信。

## 4. 代码模块说明

- `sparse_mad/topology.py`：构建 `fully_connected` 和 `ring / neighbor_connected` 拓扑，输出 `agents`、`edges`、`neighbors` 和 `density`。
- `sparse_mad/llm_runner.py`：执行真实 LLM 多智能体辩论。第一轮独立回答，后续轮次只读取 `topology.neighbors[agent]` 中允许可见的回答。
- `sparse_mad/dry_runner.py`：无 API 的确定性模拟，用于快速验证图结构、轮次和投票流程。
- `sparse_mad/voting.py`：多数投票，对应论文中的 consensus 阶段。
- `sparse_mad/metrics.py`：答案归一化、正确率判断和输入 token 成本估算。
- `sparse_mad/experiment.py`：对同一数据集分别运行全连接拓扑和邻居拓扑，并计算 `cost_saving = 1 - sparse_cost / full_cost`。
- `sparse_mad/workflow.py`：将可执行实验封装成 MASFactory `RootGraph`。
- `sparse_mad/hf_datasets.py`：接入 GSM8K、Hendrycks MATH 和 DeepMind Mathematics Dataset。

论文 sparse communication 的关键实现落在 `llm_runner.py`：

```python
neighbor_responses={
    neighbor: previous[neighbor]
    for neighbor in topology.neighbors[agent]
}
```

这保证每个 Agent 在辩论阶段只能看到拓扑允许的邻居回答。

## 5. 实验验证

工程测试覆盖：拓扑构造、多数投票、答案解析与归一化、Hugging Face 数据集字段映射、DeepMind Mathematics 数据读取、MASFactory Graph build / invoke、Visualizer 专用 Graph build、全连接与邻居拓扑比较逻辑。

当前验证命令：

```powershell
python -m pytest tests
```

最近一次验证结果：

```text
32 passed
```

可运行的数据集入口包括：

```powershell
python examples\compare_topologies.py --dataset mini_math --limit 20 --num-agents 4 --max-rounds 2
python examples\compare_topologies.py --hf-dataset gsm8k --limit 50 --num-agents 6 --max-rounds 5
python examples\compare_topologies.py --hf-dataset math --math-subset algebra --limit 50 --num-agents 6 --max-rounds 5
python examples\compare_topologies.py --hf-dataset deepmind_math --dm-math-category algebra__linear_1d_composed --limit 100 --num-agents 6 --max-rounds 5
```

其中 `deepmind_math` 默认优先读取 Hugging Face 转换后的 parquet 快照；如果不可用，则回退到 DeepMind 原始 `mathematics_dataset-v1.0.tar.gz` 数据。

## 6. 运行方式

安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

配置 OpenAI-compatible 模型：

```powershell
$env:OPENAI_API_KEY="your_api_key"
$env:OPENAI_MODEL_NAME="gpt-4o-mini"
# 可选：仅当使用第三方 OpenAI-compatible 网关时设置
$env:OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
```

运行无 API demo：

```powershell
python examples\run_dry_demo.py
```

运行真实 LLM 对比实验：

```powershell
python examples\compare_topologies.py --hf-dataset deepmind_math --limit 50 --num-agents 6 --max-rounds 5
```

## 7. 复现边界

本项目已经完成论文方法在 MASFactory 框架下的可运行复现，并能验证稀疏通信降低输入上下文成本这一核心现象。但它不声称已经严格复现论文全部数值结果。

若要达到论文级完整复现，还需要进一步对齐：原论文模型、采样策略、Agent 数量、辩论轮数、更多图密度、多次重复实验、MathVista / Anthropic-HH 等非文本或偏好任务，以及官方 token usage 统计方式。

因此，本项目更适合作为 MASFactory 的研究复现示例：展示如何把论文 SOP 抽象为 Node / Edge / Graph，并用 MASFactory 执行和可视化多智能体算法。

