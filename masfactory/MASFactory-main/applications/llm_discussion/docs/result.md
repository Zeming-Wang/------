# 复现结果观察

本文档用于解释 README 中的结果表

## 结果来源

本应用使用相同的自动评估流程比较两条实现：

```text
LLM_D source implementation
vs
MASFactory reproduction
```

两边都使用 `gpt-4o-mini` 作为生成模型和自动评估模型。README 中的模型列统一按实际运行模型展示为 `gpt-4o-mini`。

结果来自四类任务：

- `AUT`
- `Scientific`
- `Instances`
- `Similarities`

以及三个样本规模：

- `1`
- `3`
- `30`

其中 `AUT / 3` 没有对应结果，因为当前 3-sample 数据集只覆盖 `Scientific / Instances / Similarities`。

## 指标含义

评估脚本对每个最终回答计算四项指标：

- `Fluency`
- `Flexibility`
- `Originality`
- `Elaboration`

README 表格中的每个指标都保留 `Mean` 和 `Std`。这些数值来自 `Evaluation/automation_csv.py` 对完整 evaluation JSON 的汇总，而不是手动从 chat log 估算。

## 主要观察

在 `30` 样本规模上，MASFactory 版本与源代码版本整体接近：

| Task         | Fluency Delta | Flexibility Delta | Originality Delta | Elaboration Delta |
| ------------ | ------------: | ----------------: | ----------------: | ----------------: |
| AUT          |        +0.417 |            -0.047 |            +0.044 |            +0.058 |
| Scientific   |        -0.153 |            -0.136 |            +0.032 |            +0.088 |
| Instances    |        +1.972 |            +0.920 |            -0.038 |            -0.012 |
| Similarities |        +0.091 |            +0.131 |            -0.009 |            +0.001 |

这里的 delta 按 `MASFactory - source` 计算。正值表示 MASFactory 版本在该指标上更高，负值表示源代码版本更高。

整体来看，四类任务的差异都处于方法级复现可以接受的范围内。考虑到 LLM 调用存在随机性，并且本复现目标不是逐 token 对齐，这组结果说明 MASFactory 图式实现基本复现了原方法的行为趋势。

## 小样本结果的用途

`1` 和 `3` 样本规模主要用于 sanity check：

- 验证四类任务的数据解析是否正确。
- 验证 graph 能真正执行，而不只是能画出来。
- 验证输出结构能接入原评估脚本。
- 快速观察不同任务上的结果趋势。

这些小样本结果不应被视为稳定 benchmark 结论。正式对比更应参考 `30` 样本规模。

## 阅读结果时的注意点

本应用复现的是 LLM Discussion 主方法，不是论文完整实验表中的所有 baseline。因此 README 表格只比较源代码主方法和 MASFactory 主方法。

自动评估本身也依赖 LLM judge。即使使用相同模型，API 调用仍可能因为采样、服务端实现和时间差异产生轻微波动。因此结果判断应关注整体趋势，而不是单个指标的小数点级别变化。

我们保留原评估脚本和结果目录，为了让结果具备可追溯性：审阅者可以从 `Output/multi_agent` 追到 `Eval_Result/multi_agent`，再追到 `LeaderBoard` 汇总表。
