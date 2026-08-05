# DDO MASFactory 复现应用设计文档

## 1. 复现范围

本应用复现论文 **DDO: Dual-Decision Optimization for LLM-Based Medical Consultation via Multi-Agent Collaboration** 的核心多智能体问诊流程。

由于当前实验环境不部署论文中的本地 Qwen2.5 模型与 BTP adapter，本应用采用 API 模型 `gpt-4o-mini` 进行复现。公平对照对象是“作者源代码 API 改造版”，而不是论文中所有原始 baseline。

因此，本应用的主要目标是验证：

```text
在相同数据、相同本地 policy 资产、相同最大问诊轮数、相同 API 模型设置下，
MASFactory 实现是否能够达到作者源代码 API 改造版的行为与性能。
```

## 2. 总体架构

应用采用“Python 实验层 + MASFactory 单病例图”的结构：

```text
Python main loop:
    读取配置与数据集
    初始化 app_runtime
    for case in dataset:
        调用 g_consultation
        保存单病例 record
    计算 Acc_init / Acc / Avg_n

MASFactory g_consultation:
    初始诊断置信度估计
    多轮问诊循环:
        policy 生成候选问诊动作
        inquiry agent 选择症状
        patient agent / record 模拟回答
        更新症状状态
        重新估计诊断置信度
        保存本轮 interaction
```

病例级调度、进度条、结果落盘和汇总指标计算由 Python 实验层负责。单个病例内部的多轮 DDO 问诊流程由 MASFactory graph 表达。

## 3. MASFactory 图设计

核心图为 `ddo/graphs/g_consultation.py` 中的 `g_consultation`。

它包含三个主要阶段：

1. 初始诊断置信度估计：调用 `g_confidence_estimation`，基于 self-report 得到 initial confidence。
2. 多轮问诊循环：每轮先生成候选动作并选择症状，再模拟患者回答，然后重新估计诊断置信度。
3. 记录结果：每轮 response 之后保存 interaction snapshot，最终输出 case-level record。

子图职责如下：

```text
g_confidence_estimation.py
    对候选疾病计算 raw confidence，并归一化为诊断置信度。

g_actions_generation_and_selection.py
    使用原 DDO policy 接口采样候选症状动作，并由 inquiry agent 从候选症状中选择一个。

g_response_simulation.py
    优先使用病例中已记录的症状状态；若没有记录，则根据疾病知识和 patient agent 模拟回答。
```

整体流程保持接近作者源码的 `reset + step` 风格：

```text
reset:
    初始化病例状态
    基于 self-report 计算 initial confidence

step:
    生成候选动作
    选择症状
    获取患者回答
    更新 positive / negative symptoms
    重新计算 confidence
```

## 4. 状态与运行时

单病例状态使用 dataclass，定义在 `ddo/state_model/state.py`：

```text
ConsultationState
├── CaseState
├── ControlState
├── DiagnosisState
├── InquiryState
└── ResponseState
```

图中主要通过 `attrs["state"]` 保存当前病例状态，通过 dataclass 属性访问业务字段，例如：

```python
state.diagnosis_state.positive_symptoms
state.inquiry_state.selected_symptom
state.response_state.symptom_status
state.control_state.should_terminate
```

跨病例共享的静态运行依赖由 `ddo/runtime/app_runtime.py` 中的 `app_runtime` 保存，包括：

```text
cfg
dataset_ctx
client
model
sampler
base_agent
token_usage
```

当前病例状态、交互记录、实验汇总结果不放入 `app_runtime`，避免跨病例污染。

## 5. RL Policy 与本地资源文件

DDO 的候选问诊动作生成依赖作者原版 RL policy。应用保留原 policy 接口与 observation 语义，主要包括：

```text
symptom vector: -1 / 0 / 1
disease confidence vector
termination action index: symptom_num
已询问症状 masking
candidate action sampling
```

policy 资源文件未提交。实际运行前，用户需要从原 DDO 资产中获取并本地放置：

```text
policy/<DATASET>/(custom_rl_training|curstom_rl_training)/
├── best_settings.json
└── policy.pth
```

其中 `curstom_rl_training` 是原作者的 policy 文件中的错误拼写，无需修改即可正常运行，两种拼写均可正常识别。

其中 `best_settings.json` 保存 `max_turns`、`top_k`、`window_size`、`num_samples` 等复现实验参数；`policy.pth` 是候选症状采样所需的 RL policy checkpoint。二者都是运行时资源。

`configs/config.yaml` 中的 `general.device` 只影响本地 policy 网络推理，不影响 API LLM。默认使用 `cpu`，以保证在无 GPU 环境中也能运行；只有在 CUDA、PyTorch 与 GPU 都可用时才建议改为 `cuda`。

## 6. Prompt 组织

本应用包含三个主要 prompt：

```text
Diagnosis / BTP confidence estimation prompt
Inquiry / symptom selection prompt
Patient / response simulation prompt
```

为了减少复现实验中的 prompt 漂移风险，当前 prompt 保留在其直接使用位置：

```text
ddo/services/api_btp.py
ddo/graphs/g_actions_generation_and_selection.py
ddo/graphs/g_response_simulation.py
```

本 PR 不额外提交独立的 `prompts/` 目录，以保持 application 结构简洁。后续如果集中迁移 prompt，应保持机械搬迁，不修改语义、占位符或输出约束，并在实验记录中说明 prompt 文件迁移不改变 prompt 内容。

## 7. 评测记录与正确性规则

每个病例输出 record，包含：

```text
case_id
disease_label
initial_symptoms_status
initial_diagnostic_confidence
final_diagnostic_confidence
interactions
```

主要指标包括：

```text
Acc_init / Acc_wo_iq: self-report 后初始诊断准确率
Acc: 多轮问诊后最终诊断准确率
Avg_n: 平均有效问诊轮数
```

正确性判断沿用作者源代码中的“并列第一”规则：

```text
如果真实疾病标签处于最高置信度并列集合中，则 Top-1 视为正确。
```

该规则用于 `Acc_init`、最终 `Acc`、每轮 Acc、分病种 Acc 以及病例级 correct/wrong 统计。
