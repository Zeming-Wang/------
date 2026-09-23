# MaAS 复现审查记录（2026-09-17）

## 审查范围

- 原始源码：`MaAS-main/maas/ext/maas`
- 当前复现：`masfactory/MASFactory-main/applications/maas_reproduction`
- 排除目录：`olderone_maasreproduction`
- 检查范围：graph、operator、benchmark、runtime、training、schema、checkpoint、测试代码

## 审查问题记录

### 1. Controller 算子采样策略不等价（高优先级）

- 当前复现按 `p >= 0.3` 直接选择算子，空选择时随机采 1 个。
- 原 MaAS 使用概率无放回采样，直到累计概率达到 `0.3`；空选择时使用 `argmax`。
- 当前差异会改变 route、算子数量/顺序、policy log probability 和梯度估计。
- 后续处理：详细复核并恢复源算法语义。

### 2. 首层 Generate 重排、大小写和缺失 catalog 容错不等价（高优先级）

- 当前复现只在 Generate 缺失时强制替换，没有实现 Generate 已选中但不在首位时的重排。
- 当前精确使用大小写敏感的 `"Generate"` 查找；原实现使用大小写不敏感判断，并在缺失时退化到 index `0`。
- 当前差异会改变首层顺序、后续 previous-operator embedding、log probability，并可能导致 planning failure。
- 后续处理：详细复核 controller 与 route planner 两个相关 seam，避免单点修复后被 planner 校验重新拦截。

### 3. Controller 默认设备从 CUDA 改为 CPU

- 当前复现默认 `torch.device("cpu")`。
- 原实现默认使用 `cuda`（可用时），否则回退到 CPU。
- 影响 GPU 训练吞吐、张量搬运及实验设备一致性。
- 后续处理：将训练默认 policy/controller 路径改为 CUDA 优先，并确保 policy layers、query embedding、operator embedding 在同一 device。

### 4. GSM8K 答案解析失败改变训练资格

- 当前复现解析失败会产生不可靠 evaluation，进而跳过 policy update。
- 原实现经过重试后将该样本作为 score `0.0`、cost `0.0`、zero logprob 的失败样本返回。
- 这会改变有效 batch 样本数、loss reduction、policy 更新次数和统计值。
- 本轮处理：暂不修改。

### 5. HumanEval 执行缺少安全隔离、超时和特殊题 helper

- 当前复现直接使用完整 `__builtins__` 执行预测代码，没有 sanitize、超时、进程隔离或特殊 helper。
- 原实现具备受控 globals、危险结构清理、15 秒线程超时、异常转 FAIL 及特殊入口处理。
- 可能导致系统能力暴露、死循环永久阻塞以及官方题目误判。
- 本轮处理：暂不修改；修复前不建议运行不可信 HumanEval 代码。

### 6. MATH 符号等价判断能力削弱

- 当前复现只尝试 `parse_expr` 加 `simplify`。
- 原实现依次尝试 LaTeX 解析、普通表达式解析、符号化简和数值近似比较。
- LaTeX、分数、根式及解析器无法直接解析但数值等价的答案可能被误判为 0 分。
- 后续处理：恢复源实现的多级解析和数值 fallback，保持现有 `BaseBenchmark` 接口。

### 7. `MultiGenerateCoT` 可能超过源实现的候选数量

- 当前复现每次调用若得到 `candidates`，会将整个列表展开；三次调用可能产生超过 3 个候选。
- 原实现固定调用三次，每次返回一个 response，最多 3 个独立结果。
- 这会改变 `ScEnsemble` 输入规模、投票分布、token/cost 和 self-consistency 语义。
- 后续处理：保持三次调用，并将每次调用归一化为至多一个 response。

### 8. `ScEnsemble` 对非法输出静默 fallback

- 当前复现遇到非法 letter 或其他异常时选择 `candidates[0]` 并返回 `fallback`。
- 原实现非法映射进入异常处理，不会把第一个候选伪装成模型选择结果。
- 当前差异可能将 operator failure 转换为可评估 prediction，改变 score、utility 和训练资格。
- 后续处理：通过现有 `OperatorResult` seam 返回无 solution 的结构化失败，保留错误类型和信息，不让异常穿透整个 RootGraph。

### 9. `InputSplitNode` 违反属性隔离约定

- 当前 `InputSplitNode` 未显式声明 `pull_keys`/`push_keys`，会继承 MASFactory 的隐式属性语义。
- MASFactory 中 `pull_keys=None` 表示拉取外部全部属性，`push_keys=None` 可能回写属性；RootGraph 默认也不会自动清理旧 attributes。
- 连续 invocation 可能残留上一条样本的 execution/evaluation/failure 属性，造成缺字段时读旧值。
- 后续处理：仅在复现目录内为输入分割节点声明 `pull_keys={}`、`push_keys={}`，并在 MaAS 专用 RootGraph invocation seam 清理 invocation-local state。

### 10. `OperatorDispatchLoop` 的 `dispatch_state` 可能跨 invocation 残留

- 当前 Loop 在 `_attributes_store` 中写入 `dispatch_state`；`Loop.reset_gate()` 只重置 gate/controller，不清理属性。
- 当新输入没有 `dispatch_state` 时，当前实现返回 `None`，但没有删除旧的 loop-local `dispatch_state`。
- 同一个 Loop 实例复用时可能沿用旧 route cursor 或 early-stop 状态。
- 后续处理：在 `OperatorDispatchLoop._forward` 的 invocation 起点清理旧 `dispatch_state`，再写入当前 state；不修改 MASFactory 通用 Loop。

### 11. `max_iterations` 参数校验不完整

- 当前只排除 bool 和非正值，没有排除 float/string 等非 int 输入。
- 错误可能延迟到 MASFactory 循环内部才暴露。
- 本轮处理：用户当前指定范围未包含该项，暂不修改。

### 12. Evaluator 异常被吞掉，诊断信息丢失

- 当前 `EvaluatorNode` 捕获异常后只设置 `failure_source=EVALUATION`、`status=evaluation_error`。
- 异常类型和错误文本没有进入 `execution_metadata` 或 `failure_detail`。
- 这会妨碍区分 scorer 崩溃、格式错误、依赖缺失和超时。
- 本轮处理：用户当前指定范围未包含该项，暂不修改。

### 13. 测试环境无法统一完成完整回归验证

- 用户报告：`masfactory/adapters/token_usage_tracker.py` 无条件导入 `tiktoken`，缺失时导致 10 个测试模块在 collection 阶段失败。
- 本机系统 Python `C:\Python314\python.exe` 当前更早缺少 `torch`，无法作为复现测试解释器。
- 项目 `mas_env` 当前检测到 `torch 2.1.0+cu118` 和 `tiktoken`，但 `torch.cuda.is_available()` 为 `False`；使用正确的 `PYTHONPATH` 可收集全部 82 项测试。
- 结论：测试依赖问题仍需在目标离线/CI 环境验证；本轮不修改 MASFactory 通用 tokenizer 依赖链，除非后续明确纳入修复范围。

## 本轮请求的处理边界

- 需要详细复核/修复：1、2、3、6、7、8、9、10。
- 暂不处理：4、5、11、12、13。
- 原则：优先复用 MASFactory 现有 `Node`/`CustomNode` 的显式属性策略、`Graph`/`Loop` 的生命周期和当前 `OperatorResult` 结构化失败接口；不修改无关通用框架代码。

## 当前验证记录

```text
系统 Python：torch 不可导入；不能代表项目运行环境。
mas_env：torch 2.1.0+cu118；CUDA available=False；tiktoken 可导入。
mas_env + 正确 PYTHONPATH：82 tests collected。
```

## 当前实现最小行为探针

在 `mas_env`、正确的 `PYTHONPATH` 下直接调用相关 seam，得到：

```text
InputSplitNode policies: pull_keys=None, push_keys=None
OperatorDispatchLoop after dispatch_state=None: old dispatch_state remains "OLD"
MultiGenerateCoT: adapter returns 2 candidates per call -> 6 accumulated candidates
ScEnsemble: invalid solution_letter -> status="fallback", solution=first candidate
```

这些结果与第 7、8、9、10 条审查描述一致，后续回归测试应分别锁定上述四个症状。
