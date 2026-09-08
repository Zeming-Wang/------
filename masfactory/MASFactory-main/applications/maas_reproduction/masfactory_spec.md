# MASFactory 多智能体系统工程设计与开发规范

## 1. 规范目标

本规范用于指导基于 MASFactory 构建多智能体系统（Multi-Agent System，MAS）。目标不是把所有业务都拆成节点，而是建立清晰、可验证、可复用的工作流结构：

- 用 Graph 表达系统结构；
- 用 Agent 承担模型推理；
- 用 CustomNode 承担确定性计算和副作用；
- 用 Edge 传递业务消息；
- 用 attributes 管理跨层级共享状态；
- 用 Loop 和 Switch 表达控制流；
- 用 Model、Memory、Retrieval、Tool 和 Skill 作为可替换的适配器；
- 将复杂实现隐藏在少数具有深度的模块之后。

这里的“模块”包括函数、类、子图和整个应用层。每个模块都应拥有清晰的 interface、实现和 seam。interface 不只是函数参数，还包括字段契约、执行顺序、终止条件、错误行为和副作用。

## 2. 基本心智模型

MASFactory 工作流由三类关系组成：

```text
数据关系：Edge keys
状态关系：attributes / pull_keys / push_keys
控制关系：Graph / Loop / Switch / Gate
```

一个节点的执行过程通常是：

```text
检查入边是否就绪
 → 聚合输入消息
 → 拉取本地所需状态
 → 执行 _forward
 → 分发输出消息
 → 回写允许更新的状态
 → 更新节点和边状态
```

因此，设计工作流时必须同时回答三个问题：

1. 当前节点需要什么业务数据？
2. 当前节点需要读取或更新什么共享状态？
3. 当前节点为什么在此时执行，下一步由谁触发？

如果一个字段同时被当作 Edge 消息、attributes 状态和控制标记使用，应重新划分其职责。

## 3. 推荐的分层结构

一个可维护的 MASFactory 应用通常分为以下层次：

```text
application/
├── workflow/       # RootGraph 和顶层流程装配
├── graphs/         # 可复用 Graph、Loop、复合节点
├── nodes/          # Agent、CustomNode 及其 forward 实现
├── domain/         # 领域状态、结果、规则和纯计算
├── adapters/       # 模型、工具、检索器、外部系统适配
├── config/         # 可序列化配置和实验参数
├── runtime/        # 运行时初始化、依赖组装、执行入口
├── assets/         # prompt、skill、数据和静态资源
└── tests/          # 单元测试、图构建测试、集成测试
```

职责要求：

- `workflow` 只负责装配 Graph、节点和边；不承载复杂业务算法。
- `graphs` 负责把一组节点封装成具有稳定 interface 的可复用模块。
- `nodes` 负责单步计算、Agent 推理或确定性动作。
- `domain` 保存与 MASFactory 无关的业务规则，优先设计成纯函数或小型深模块。
- `adapters` 隔离外部模型、数据库、文件系统、网络和旧系统。
- `runtime` 负责依赖注入，不应成为隐藏的全局状态中心。
- `config` 负责声明参数，不应混入运行期对象。

顶层 Graph 应尽量简洁。理想状态是从主图可以直接看出业务流程，而不需要理解每个节点的实现细节。

## 4. RootGraph、Graph 和 Node 的使用规则

### 4.1 RootGraph

`RootGraph` 是应用的顶层可执行图：

- 一个应用通常只有一个主要 RootGraph；
- RootGraph 持有全局 attributes 和顶层输入输出契约；
- RootGraph 负责连接各个阶段；
- 在 `invoke` 前必须调用 `build()`；
- RootGraph 不应直接实现大量业务逻辑。

### 4.2 Graph

`Graph` 是可嵌套、可复用的子工作流。适合封装：

- 一组稳定的 Agent 协作；
- 一次完整的业务阶段；
- 一个领域操作流程；
- 一个需要独立测试和独立终止条件的子系统。

一个 Graph 应有单一的职责和较小的对外 interface。内部节点、内部边和内部状态应尽可能隐藏。

### 4.3 Node

节点是计算单元。新增节点前先判断：

- 如果逻辑是模型推理，用 `Agent`；
- 如果逻辑是确定性计算或副作用，用 `CustomNode`；
- 如果逻辑包含多个节点和内部控制流，用 `Graph`；
- 如果逻辑是反复执行直到满足条件，用 `Loop`；
- 如果逻辑是路由决策，用 `LogicSwitch` 或 `AgentSwitch`。

不要为了“可视化上更细”而把一个不可分割的算法强行拆成大量节点。节点数量增加会同步增加消息契约、状态同步和调试成本。

## 5. Graph 构建和边的硬性要求

### 5.1 构建生命周期

标准生命周期是：

```python
graph = RootGraph(...)
graph.create_node(...)
graph.create_edge(...)
graph.build()
output, attributes = graph.invoke(input, attributes=...)
```

`build()` 是装配阶段，不应与执行阶段混淆。需要在构建时生成内部节点的复合 Graph，应通过 build hook 或明确的 build 方法完成。

### 5.2 入口和出口

连接图入口和出口时必须使用：

```python
graph.edge_from_entry(receiver, keys)
graph.edge_to_exit(sender, keys)
```

不要用普通 `create_edge` 连接保留的入口、出口节点。

### 5.3 普通边

普通边同时承担：

- 执行依赖；
- 消息传递；
- 字段过滤；
- 可选的格式化。

边的 `keys` 是下游节点的字段契约。每个字段都应有稳定名称、清晰描述和明确的数据类型。`keys={}` 应只用于明确的控制流边。

### 5.4 图约束

- 普通 Graph 不得出现非法环路；
- 需要循环时使用 Loop；
- 不要创建悬空节点或无入口、无出口路径的节点；
- 边的两端必须属于当前 Graph；
- 不要依赖节点创建顺序隐式表达业务顺序；
- 分支必须保证每个合法输入都有可达路径；
- 合并节点必须明确是否需要等待所有入边。

## 6. Loop 和控制流规范

Loop 是控制流组件，不是普通的无限循环容器。每个 Loop 必须明确：

- 最大迭代次数；
- 每轮开始时由 Controller 传入的字段；
- 每轮结束时回传 Controller 的字段；
- 终止条件；
- 是否允许中途退出；
- 循环状态的作用域。

连接 Controller 的边必须使用 Loop 专用接口：

```python
loop.edge_from_controller(node, keys)
loop.edge_to_controller(node, keys)
loop.edge_to_terminate_node(node, keys)
```

按照 MASFactory 的 Loop 约束，Controller 相关边以及 Loop 对外输入输出的 `keys` 应保持一致。若业务输入和业务结果结构不同，不应直接把它们混成 Controller 往返消息；应将循环控制字段统一化，业务状态放入 attributes 或由内部节点转换。

终止函数必须：

- 可重复执行；
- 不依赖未声明的隐式状态；
- 在状态缺失时有安全默认值；
- 能被单独测试；
- 不承担大量结果写入、文件保存或模型调用。

不要使用极大的 `max_iterations` 掩盖缺失的终止条件，也不要仅依赖模型自行停止。

## 7. Edge 消息与 attributes 状态

### 7.1 水平消息：Edge

适合放入 Edge 的数据：

- 当前任务输入；
- 当前节点产生的结果；
- 评分、预测、错误摘要；
- 下游节点立即需要的结构化业务 payload。

### 7.2 垂直状态：attributes

适合放入 attributes 的数据：

- 跨阶段共享的配置；
- 迭代计数、重试次数和阶段标记；
- 控制器、优化器、Memory 等运行时对象；
- 跨多个节点积累的统计数据；
- 长生命周期的资源引用。

### 7.3 pull_keys 和 push_keys

不要依赖默认值。对跨层级状态应显式声明：

- `pull_keys=None`：继承上层全部状态；只应在确实需要完整环境时使用；
- `pull_keys={}`：不继承上层状态；
- 非空 `pull_keys`：只读取列出的字段；
- `push_keys={}`：不回写任何状态；
- 非空 `push_keys`：只回写列出的字段；
- `attributes`：节点自身初始化的本地状态。

推荐对每个 Graph 维护一张状态表：

| 字段 | 类型 | 来源 | 读取者 | 写入者 | 作用域 |
|---|---|---|---|---|---|
| task | str | 外部输入 | 多个阶段 | 顶层 | 全局 |
| current_iteration | int | Loop | Controller | Controller | 当前循环 |
| result | dict | 节点输出 | 下游节点 | 当前节点 | 单轮 |

不要把几十个相互关联的平级字段塞进 attributes。可以将它们封装为一个 `RuntimeState`、`BatchState` 或 `SessionState`，并通过一个清晰的 interface 管理。

尤其禁止依赖“在节点回调中直接修改 attributes，所以它一定会自动传播”。状态是否传播必须由 `pull_keys/push_keys` 或明确的共享对象语义保证。

## 8. Agent 设计规范

Agent 的职责是观察上下文、调用模型、执行工具并返回结构化结果。一个 Agent 应明确：

- `name` 和 `role_name`；
- `instructions`；
- `prompt_template`；
- 输入字段；
- 输出字段；
- formatter；
- tools；
- memories/retrievers；
- model settings；
- pull/push 状态字段。

要求：

- 指令和业务代码分离；
- prompt 中引用的每个字段都必须来自 Edge keys 或 attributes；
- 结构化输出必须使用明确的 formatter 和输出字段契约；
- Agent 的输出字段不应依赖自然语言中的模糊约定；
- 工具函数必须有类型注解和参数 docstring；
- Tool 返回值应是可序列化数据；
- 一个 Agent 最多挂载一个 HistoryProvider 类型的 memory；
- 有状态 memory 必须明确是共享还是隔离。

Agent 不应直接负责复杂文件管理、实验统计、训练更新或外部系统编排。需要这些能力时，应通过 Tool、CustomNode 或 Adapter 提供。

## 9. Model、Memory、Retrieval、Tool 和 Skill

### Model

模型调用必须通过 `Model` 适配器。适配器负责 provider 差异，包括：

- 请求格式转换；
- 响应归一化；
- Tool call 处理；
- 超时、重试和退避；
- 限流和错误分类；
- token/cost 统计；
- 日志脱敏。

业务 Graph 不应直接依赖某个模型供应商 SDK。

### Memory 和 Retrieval

- HistoryMemory 表示有顺序的对话历史；
- VectorMemory、RAG、MCP 等表示可检索上下文；
- passive provider 用于自动注入上下文；
- active provider 用于通过工具按需检索；
- 同一份 memory 的共享范围必须显式设计。

不要把“上下文资料”伪装成业务状态，也不要把“可执行动作”伪装成 Memory。

### Skill

Skill 应采用独立目录，至少包含 `SKILL.md`。Skill 是可复用指令包，不是节点，不应替代 Graph、Agent 或 NodeTemplate。

### NodeTemplate

NodeTemplate 用于复用节点配置，不是节点实例。适合集中管理：

- model；
- formatter；
- tools；
- memory；
- retriever；
- 通用 instructions。

对可变对象必须明确生命周期：

- `Shared(obj)`：共享同一实例；
- `Factory(fn)`：每次物化创建新实例。

模板的 `defaults` 和 `overrides` 只在物化阶段生效，相关作用域必须包裹 `build()`。

## 10. CustomNode 和副作用

CustomNode 适合处理：

- 确定性转换；
- 数据清洗；
- 文件读写；
- 日志和指标；
- 外部系统调用；
- 阶段前后处理。

推荐回调形式：

```python
def forward(input_data: dict, attributes: dict) -> dict:
    ...
```

要求：

- 输入输出结构稳定；
- 副作用集中在少数模块；
- 依赖通过参数传入，不在函数内部隐式创建；
- 错误要么结构化返回，要么明确抛出；
- 不要在普通业务节点中随意删除目录、覆盖用户文件或执行 shell 命令；
- 写文件、写日志和更新外部状态应可测试、可追踪、可重试。

如果自定义 Node 需要使用 Forward Hook，应在 `_forward` 上显式使用 `@masf_hook(Node.Hook.FORWARD)`。

## 11. 配置、运行时和依赖注入

配置应分为三类：

```text
静态配置：模型名、角色、prompt、数据路径
实验配置：采样数、迭代数、学习率、开关
运行时对象：模型实例、优化器、缓存、连接和数据集
```

静态配置和实验配置可以序列化；运行时对象不应直接写入 JSON/YAML。

推荐流程：

```text
读取配置
 → 校验配置
 → 构造运行时对象
 → 注入 RootGraph attributes
 → build
 → invoke
```

配置初始化只能有一个权威入口。不要在 CLI、Graph 节点和 runtime initializer 中重复创建同一份 settings。

依赖要求：

- 应用必须声明自己的必需依赖；
- 测试环境必须能从仓库根目录启动；
- 不要依赖未声明的工作目录、环境变量或 sys.path 修改；
- 环境变量名称必须统一并有默认行为；
- 可选依赖应在使用点给出清晰错误。

## 12. 可观测性、错误和结果

推荐使用 Hooks 进行横切观测：

- Node Execute：节点开始和结束；
- Forward：输入输出字段；
- Edge Send/Receive：消息流向；
- Error：异常、节点和上下文。

Hook 是同步、进程内机制。Hook 回调不能抛出未经处理的异常，否则可能中断主流程。日志必须避免泄露 API key、完整敏感输入和不必要的 prompt 内容。

错误分类应至少区分：

```text
模型调用失败
工具执行失败
业务计算失败
输入数据非法
工作流结构错误
外部资源不可用
```

不要把异常字符串伪装成正常业务结果。失败应通过结构化错误字段、日志和明确的执行状态表达。

每次运行应有独立的结果目录，并保存：

- 输入任务；
- 使用的配置；
- 模型信息；
- 运行日志；
- 中间结果；
- 最终结果；
- 必要的 checkpoint；
- 版本或提交信息。

## 13. 声明式、命令式和复合模块的选择

### 声明式

适合静态、清晰、可视化优先的结构。节点和边集中声明，便于检查、导出和批量覆写。

### 命令式

适合：

- 动态节点数量；
- 依赖运行时数据；
- 复杂条件和循环；
- 需要逐步构建的复合模块。

### 推荐组合方式

复杂子流程先用命令式代码实现并测试，再封装为 Graph；主图使用声明式或简洁的命令式装配引用这些复合模块。

不要在同一模块中无理由混用多种装配方式，也不要为每个节点重复写同一组 model、formatter、memory 和 prompt 配置。

## 14. 设计中的禁区

以下做法容易造成死锁、状态丢失、不可复现或难以维护，应视为禁止或高风险做法：

1. 用普通 Graph 边构造环路，而不使用 Loop。
2. 使用普通 `create_edge` 连接入口、出口或 Loop Controller。
3. 让 Loop 的 Controller 输入输出字段不一致，却没有转换层或统一控制消息。
4. 依赖 `pull_keys=None` 和 `push_keys={}` 的隐式组合传递业务状态。
5. 在多个位置重复初始化同一份配置和运行时对象。
6. 让 attributes 变成未定义结构的全局变量仓库。
7. 在 Agent、Graph 和 CustomNode 中交叉复制同一段业务逻辑。
8. 把复杂算法强行拆成大量节点，只为了让图看起来更细。
9. 把模型异常、工具异常和业务错误都转成普通文本结果。
10. 在 import 阶段执行构图、创建文件、连接外部服务或运行任务。
11. 在业务代码中直接执行不可控的 shell 命令或递归删除目录。
12. 使用未声明的依赖、隐式工作目录或临时修改 sys.path 作为正常运行条件。
13. 让有状态 Memory、模型客户端或可变模板对象被无意共享。
14. 通过模型自然语言约定字段，却没有 formatter、keys 或 schema 契约。
15. 只测试单个 forward 函数，不测试 `build()`、消息路由、Loop 终止和完整 `invoke()`。

## 15. 测试规范

测试应覆盖四个层次：

### 模块测试

测试纯函数、状态转换、评分规则、终止函数和错误分类。

### 节点测试

使用假的 Model、Memory、Tool 和外部资源，验证输入输出及 attributes 读写。

### 图结构测试

至少验证：

- 图能够 build；
- 入口和出口可达；
- Loop 的 Controller 边正确；
- Switch 的每条路由可达；
- 节点名称和边字段符合预期。

### 端到端测试

使用最小输入和假的外部依赖验证：

- `invoke()` 可以结束；
- 终止条件能够生效；
- 状态能够跨节点传播；
- 结果和错误可以被记录；
- 重复运行不会污染上一次结果。

## 16. 开发前检查清单

### 结构

- [ ] 顶层是否只有一个明确的 RootGraph？
- [ ] 复杂流程是否被封装为可复用 Graph？
- [ ] 主图是否只表达流程，而没有大量业务细节？
- [ ] 是否存在重复的构图入口？

### 消息和状态

- [ ] 每条 Edge 的 keys 是否有明确含义？
- [ ] 业务数据和控制状态是否分离？
- [ ] pull_keys 和 push_keys 是否显式声明？
- [ ] 每个状态字段是否有唯一的主要写入者？
- [ ] 是否验证了状态跨 Graph/Loop 的传播？

### 控制流

- [ ] 所有循环是否使用 Loop？
- [ ] Loop Controller 相关 keys 是否一致？
- [ ] 是否存在明确、可测试的终止条件？
- [ ] 分支是否都有合法路径？

### Agent 和适配器

- [ ] Agent 输出是否结构化？
- [ ] Model、Memory、Tool 是否通过适配器隔离？
- [ ] 有状态依赖的生命周期是否明确？
- [ ] Tool 是否有类型注解和 docstring？

### 运行和验证

- [ ] 配置是否只有一个权威入口？
- [ ] 依赖和环境变量是否已声明？
- [ ] 是否测试 build 和 invoke？
- [ ] 是否有节点、边和错误日志？
- [ ] 每次运行是否使用隔离的结果目录？

## 17. 最终设计原则

一个合格的 MASFactory 多智能体系统，应满足：

> 用 Graph 表达结构，用 Edge 表达业务消息，用 attributes 表达共享状态，用 Loop/Switch 表达控制流，用 Agent 表达推理，用 CustomNode 表达确定性动作，用 Adapter 隔离外部依赖，用深模块隐藏复杂实现。

最终评价标准不是节点越少越好，也不是图越复杂越强，而是：调用者需要理解的 interface 足够小，复杂行为集中在正确的 seam 后面，修改和测试能够保持良好的 locality。
