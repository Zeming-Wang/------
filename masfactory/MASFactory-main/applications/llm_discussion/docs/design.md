# LLM Discussion 的 MASFactory 复现设计

本文档记录本应用的核心设计取舍

## 复现目标

本应用复现的是 LLM Discussion 主方法在四类创造力任务上的自动评估流程：

- `AUT`
- `Scientific`
- `Instances`
- `Similarities`

复现目标不是逐 token 一致，也不是完整复刻论文中的所有 baseline、消融实验和人工评估。我们关注的是方法级复现：

```text
相同任务输入
相同 4-agent role-play 配置
相同 discussion rounds
相同输出 JSON 结构
相同自动评估脚本
```

在这个定义下，只要 MASFactory 版本能够完整跑通主讨论流程，并把结果送入原评估链路，就可以和原脚本实现进行趋势对比。

## 保留的外壳

原项目的 `llm_discussion.py` 和 `discussion.py` 已经承担了很多实验外壳职责：

- 读取任务数据集
- 选择 `AUT / Scientific / Instances / Similarities` 分支
- 组织输出文件名
- 保存 `chat_log / init / Output`
- 调用 `Evaluation/auto_grade_final.py`

这部分不是 MASFactory 复现的重点。如果贸然重写，反而容易破坏输出命名和评估兼容性。因此本应用保留这层实验外壳，只替换最核心的一段：

```text
单个 sample 内部的 4-agent 多轮讨论流程
```

换句话说，dataset-level runner 仍然由 Python 负责，sample-level discussion 交给 MASFactory graph。

## 图式设计

核心图位于 `Experiments/multi_agent/sample_graph.py`，节点语义位于 `Experiments/multi_agent/graph_nodes.py`。

我们采用一个显式的 `RootGraph + Loop` 结构：

```text
RootGraph
  sample_state_init
  discussion_loop
    assembler_A -> agent_call_A
    assembler_B -> agent_call_B
    assembler_C -> agent_call_C
    assembler_D -> agent_call_D
    round_collector
  final_extract
```

其中：

- `sample_state_init` 初始化当前 sample 的状态、agent 私有历史和结果容器。
- `assembler_*` 根据当前轮次、角色设定和其他 agent 的上一轮输出组装 prompt。
- `agent_call_*` 调用模型，得到该 agent 当前轮的回复。
- `round_collector` 汇总四个 agent 的回复，更新当前 sample 的状态。
- `final_extract` 导出首轮结果、末轮结果和完整讨论记录。

这个结构有意保持显式，方便在 MASFactory graph 中观察每个 agent 的输入、输出和每轮状态变化。

## sample-level graph设计

原脚本是按数据集循环，每个问题独立跑完整讨论。我们延续这个语义，让每个 sample 拥有独立 graph invocation。

这样做有三个好处：

- 避免不同 sample 的 `chat_history` 或中间状态互相污染。
- 保持输出条目数等于 `sample_count x agent_count`。
- 让 graph 聚焦在多智能体讨论本身，而不是承担数据集调度、落盘和评估职责。

这也是本复现里最重要的边界：MASFactory graph 负责表达讨论流程，Python runner 负责实验编排。

## Prompt 与消息流对齐

复现时最容易偏离原方法的是消息流。原脚本并不是一个普通共享聊天室，而是每个 agent 保留自己的私有 `chat_history`，并在下一轮看到其他 agent 的上一轮回答。

因此本应用在 `round_collector` 中只收集当前轮各 agent 的输出，再在下一轮由 `assembler_*` 组装为：

```text
These are the solutions to the problem from other agents:
One agent solution: `...`
...
```

最后一轮会额外加入最终答案格式约束，要求输出 `1. ... 2. ... 3. ...`。这一步非常关键，因为评估前的结果抽取依赖编号列表。

## 输出与评估兼容

本应用保留原项目的结果目录：

```text
Results/<Task>/chat_log/
Results/<Task>/init/
Results/<Task>/Output/multi_agent/
Results/<Task>/Eval_Result/multi_agent/
Results/LeaderBoard/
```

最终输出仍然使用原评估脚本可识别的文件名结构，尤其保留 `multi_agent` 目录和文件名中的 `multi` 字段。这样 `auto_grade_final.py` 可以直接定位输入文件，并继续写入 `Eval_Result` 和 `LeaderBoard`。

## 复现边界

当前应用没有复现以下内容：

- `Single Agent`
- `Brainstorm then Select`
- `LLM Debate`
- prompt 消融
- agent 数量消融
- round 数量消融
- human evaluation

这些内容可以作为后续扩展，但不属于本应用的主复现目标。当前版本的核心结论是：LLM Discussion 主方法可以用 MASFactory 的图式编排表达，并在相同数据、模型和评估脚本下得到可对比的自动评估结果。
