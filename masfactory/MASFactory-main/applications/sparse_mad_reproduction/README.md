# 基于 MASFactory 的 Sparse MAD 论文复现示例

本示例基于 MASFactory 的图工作流抽象，复现论文 **Improving Multi-Agent Debate with Sparse Communication Topology** 的核心方法。

本项目的目标不是完全复现论文中的所有数值结果，而是展示如何将一篇多智能体算法论文抽象为 MASFactory 的 Node / Edge / Graph 工作流，并运行全连接多智能体辩论与稀疏通信多智能体辩论的对比实验，观察 accuracy 与估算输入 token 成本之间的关系。

## 示例内容

论文研究的是 Multi-Agent Debate, MAD 中的通信拓扑问题。传统 MAD 通常采用全连接通信方式：在后续辩论轮次中，每个 Agent 都能看到其他所有 Agent 的上一轮回答。这样做可能提升推理质量，但会显著增加输入上下文长度和 token 成本。

Sparse MAD 则限制每个 Agent 只能看到部分邻居 Agent 的回答，从而降低通信成本。

本项目将论文方法实现为如下流程：

```text
1. 构建通信拓扑 G = (V, E)
2. 第 1 轮：每个 Agent 独立回答问题
3. 第 2..N 轮：每个 Agent 只读取拓扑允许可见的上一轮回答
4. 对所有 Agent 的最终答案进行多数投票
5. 比较 accuracy 与估算输入 token 成本
```

## MASFactory 抽象方式

本项目提供两个互补的图视角。

### 实验 SOP 图

在 VS Code 中使用 MASFactory Visualizer 打开：

```text
sparse_mad/visual_workflow.py
```

查看函数：

```python
build_visual_comparison_graph(client=...)
```

该图将论文复现实验过程展示为 MASFactory 节点：

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

这些 MASFactory Edge 负责在节点间传递实验状态，例如 dataset、拓扑运行结果、accuracy、token cost 和 cost saving。

### Agent 通信拓扑图

打开：

```text
sparse_mad/agent_topology_hex_visual.py
```

查看函数：

```python
build_hex_topology_visual_graph()
```

该图直接展示论文中的通信拓扑：

```text
Full_Agent_1..6: 全连接图 K6
Neighbor_Agent_1..6: 环形邻居图 C6
```

在这个视角中：

```text
Agent = Node
Agent 之间的通信关系 = Edge
```

MASFactory 的 RootGraph 是有向无环图，因此可视化文件中使用有向边展示论文中的无向通信关系。箭头方向仅用于 Visualizer 展示，不表示论文方法只能单向通信。

## 项目结构

```text
sparse_mad/
  topology.py                  # 构建全连接拓扑和邻居/环形拓扑
  llm_runner.py                # 真实 LLM 多智能体辩论执行器
  dry_runner.py                # 无 API 的确定性模拟执行器，用于测试和 demo
  voting.py                    # 多数投票
  metrics.py                   # 答案归一化、accuracy 判断、token 成本估算
  experiment.py                # 全连接拓扑 vs 稀疏拓扑实验对比
  workflow.py                  # 可执行 MASFactory RootGraph 封装
  visual_workflow.py           # SOP 层级的可视化图
  agent_topology_hex_visual.py # 静态 6-Agent 通信拓扑可视化
  hf_datasets.py               # GSM8K、Hendrycks MATH、DeepMind Mathematics 数据加载
examples/
  compare_topologies.py        # 主实验命令行入口
  run_dry_demo.py              # 无 API 快速 demo
  run_llm_demo.py              # 单次真实 LLM 辩论 demo
  run_visual_workflow.py       # 运行可视化 SOP 工作流
tests/                         # 单元测试与工作流测试
```

## 安装依赖

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS / Linux：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 配置模型 API

真实 LLM 示例使用 OpenAI-compatible SDK。请通过环境变量配置密钥，不要把 API key 写入代码。

Windows PowerShell：

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
$env:OPENAI_MODEL_NAME="gpt-4o-mini"
# 可选：仅当使用第三方 OpenAI-compatible 网关时设置
$env:OPENAI_BASE_URL="https://your-compatible-endpoint/v1"
```

为了兼容早期本地脚本，如果没有设置 `OPENAI_MODEL_NAME`，程序也会读取 `MODEL_NAME`。

## 运行无 API Demo

```powershell
python examples\run_dry_demo.py
```

该命令不会调用外部模型，用于快速验证拓扑构造、稀疏邻居可见性、辩论轮次和多数投票流程。

## 对比全连接 MAD 与稀疏 MAD

运行内置 toy 数据集：

```powershell
python examples\compare_topologies.py --dataset mini_math --limit 20 --num-agents 4 --max-rounds 2
```

运行 GSM8K：

```powershell
python examples\compare_topologies.py --hf-dataset gsm8k --limit 50 --num-agents 6 --max-rounds 5
```

运行 Hendrycks MATH：

```powershell
python examples\compare_topologies.py --hf-dataset math --math-subset algebra --limit 50 --num-agents 6 --max-rounds 5
```

当简单数据集上两种拓扑都达到 100% accuracy 时，推荐运行更难的 Hendrycks MATH Level 5 压测：

```powershell
python examples\compare_topologies.py --hf-dataset math --math-subset intermediate_algebra --math-level 5 --shuffle --seed 42 --limit 50 --num-agents 6 --max-rounds 5
```

运行 DeepMind Mathematics Dataset，该数据集更接近论文 text reasoning 方向中的数学推理设置：

```powershell
python examples\compare_topologies.py --hf-dataset deepmind_math --dm-math-category algebra__linear_1d_composed --limit 100 --num-agents 6 --max-rounds 5
```

脚本会输出全连接拓扑的 accuracy / cost、稀疏邻居拓扑的 accuracy / cost，以及相对 cost saving。

常用数据集参数：

```text
--math-level 5       将 Hendrycks MATH 过滤到最难等级
--shuffle --seed 42  在应用 --limit 之前随机打乱样本，并保证可复现
--limit N            控制实验规模和 API 成本
```

建议先用 `--limit 10` 或 `--limit 30` 快速试跑。对于 100 个样本、6 个 Agent、5 轮辩论的实验，总模型调用次数约为：

```text
2 * 100 * 6 * 5 = 6000
```

因此，`--limit 100` 更适合作为最终实验，而不是第一次 smoke test。

## 实验结果解读

主要输出格式如下：

```text
Fully-connected accuracy=..., cost=...
Neighbor-connected accuracy=..., cost=...
Sparse cost saving: ...
```

其中，`accuracy` 表示多数投票后的最终答案是否与数据集标准答案一致；`cost` 是估算输入 token 数，用于比较不同通信拓扑的相对成本；`Sparse cost saving` 计算方式为：

```text
1 - sparse_cost / fully_connected_cost
```

以下是在 DeepMind Mathematics `algebra__linear_1d_composed` 上运行 50 个样本、6 个 Agent、5 轮辩论得到的示例结果：

```text
Fully-connected MAD (D=1)
  accuracy: 1.000 (50/50)
  estimated input tokens: 1246626

Neighbor-connected MAD
  accuracy: 1.000 (50/50)
  estimated input tokens: 675438

Sparse cost saving: 45.8%
```

该结果说明，本项目中的 MASFactory 工作流能够正确完成全连接通信拓扑与稀疏邻居通信拓扑的对比实验；同时，稀疏邻居拓扑能够明显降低估算输入 token 成本。在该实验中，稀疏通信节省了 45.8% 的估算输入 token。

但是，该数据集子任务对当前测试模型来说过于简单，两种拓扑都达到了 100% accuracy。因此，这组实验存在明显的 ceiling effect，尚不能完全验证稀疏多智能体辩论在困难推理任务上的效能保持或提升能力。它主要验证了复现流程、MASFactory 图工作流实现，以及稀疏通信的 cost-saving 行为。

若要更严格地评估 sparse debate 的有效性，建议使用更困难的 Hendrycks MATH Level 5 设置，并开启随机采样，例如 `intermediate_algebra`、`counting_and_probability` 或 `number_theory`。

## 运行可视化工作流

```powershell
python examples\run_visual_workflow.py --hf-dataset deepmind_math --limit 10 --num-agents 6 --max-rounds 5
```

该命令可配合 MASFactory Visualizer 查看论文 SOP 图。

## 运行测试

```powershell
python -m pytest tests
```

测试覆盖拓扑构造、多数投票、答案解析与归一化、数据集映射、dry runner、fake LLM runner、MASFactory workflow invoke，以及 Visualizer 专用图构造。

## 关于复现结论的说明

本示例是论文方法在 MASFactory 框架下的可运行复现，并不声称已经严格复现论文中的全部实验数值。若要达到论文级完整复现，还需要对齐原论文模型、数据采样方式、Agent 数量、辩论轮数、图密度、多次重复实验以及官方 token usage 统计方式。

本项目中的实验主要用于展示论文方法、MASFactory 抽象方式，以及全连接通信与稀疏通信之间的 accuracy / cost trade-off。
