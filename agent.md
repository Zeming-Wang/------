# AGENT.md

# 项目开发规范（MaAS → MASFactory 重构）

## 项目目标

本项目旨在将 MaAS（Multi-agent Architecture Search）的实现迁移至 MASFactory 框架。

本项目属于**框架重构（Framework Migration）**，而非算法重写。

开发过程中必须遵循以下原则：

- 保持 MaAS 原有算法逻辑一致；
- 保持模块职责一致；
- 保持执行流程一致；
- 优先适配 MASFactory 的架构设计；
- 不引入新的搜索算法；
- 不主动优化论文中的实现逻辑；
- 所有修改均应保证后续能够接入 MASFactory 可视化插件进行展示。

---

# 第一原则

## 优先遵循 MASFactory 架构

所有新增代码必须优先符合 MASFactory 的组织方式，而不是保持 MaAS 原有目录。

允许：

MaAS
↓

MASFactory Agent

↓

MASFactory Graph

↓

Workflow

↓

Plugin Visualization

禁止：

为了保持 MaAS 原有目录而绕过 MASFactory 的设计。

---

# 模块职责

每个模块必须只负责一项职责。

禁止：

一个类同时负责：

- Prompt
- Memory
- LLM
- Tool
- Scheduling
- Evaluation

如果出现多个职责，应立即拆分。

---

# 代码修改原则

修改已有代码时：

优先修改已有模块。

禁止：

为了实现一个功能，新建多个功能重复的新模块。

如果已有模块可以扩展：

必须扩展已有模块。

不要复制已有代码。

---

# 保持接口稳定

除非必要：

不得修改已有公共接口。

如果接口必须修改：

必须同步更新：

- 调用方
- Graph
- Workflow
- 配置文件
- 插件注册入口

不得留下失效调用。

---

# 不允许兜底逻辑

禁止：

```python
try:
    ...
except:
    pass
```

禁止：

```python
except Exception:
    return None
```

禁止：

```python
if xxx is None:
    return ""
```

禁止：

静默失败。

禁止：

默认吞掉异常。

所有异常必须明确暴露。

例如：

```python
raise RuntimeError(...)
```

或者：

```python
logger.exception(...)
raise
```

---

# 日志规范

所有重要流程必须记录日志。

包括：

- Graph 构建
- Agent 创建
- Workflow 执行
- Architecture Sampling
- Optimizer 更新
- Tool 调用
- Prompt 执行
- Plugin 数据生成

日志至少包含：

模块名称

输入

输出

耗时

异常信息

推荐：

```python
logger.info(...)
logger.warning(...)
logger.error(...)
logger.exception(...)
```

不要使用：

```python
print(...)
```

作为正式日志。

---

# 不允许隐藏问题

任何异常：

必须尽早暴露。

禁止：

自动修复。

禁止：

自动重试。

禁止：

自动生成默认值。

例如：

错误：

```python
if config is None:
    config={}
```

正确：

```python
raise ValueError(...)
```

---

# 配置规范

所有可配置内容：

统一放入配置。

禁止：

魔法数字。

禁止：

硬编码路径。

禁止：

硬编码模型名称。

例如：

Agent 数量

Prompt

模型

API

搜索参数

Graph 参数

全部配置化。

---

# Prompt 管理

Prompt 必须独立管理。

禁止：

Prompt 写在业务逻辑内部。

例如：

错误：

```python
prompt = "You are..."
```

正确：

```
prompts/

controller.md

solver.md

critic.md
```

---

# Graph 规范

所有 Agent 的连接关系：

必须能够映射为 MASFactory Graph。

要求：

每个节点：

具有唯一职责。

每条边：

具有明确数据流。

禁止：

隐式调用。

禁止：

跨模块共享状态。

---

# 状态管理

禁止：

全局变量。

禁止：

单例缓存。

推荐：

Context

State

Message

Event

进行状态传递。

所有状态必须能够追踪来源。

---

# 数据流

保证：

Input

↓

Controller

↓

Architecture

↓

Graph

↓

Agents

↓

Result

↓

Evaluation

↓

Feedback

↓

Optimizer

数据流必须保持单向。

避免循环依赖。

---

# 类型规范

新增代码：

必须使用：

Python Type Hint。

例如：

```python
def execute(query: str) -> AgentResult:
```

禁止：

无类型函数。

---

# 文档规范

新增模块必须包含：

模块说明

输入

输出

依赖

用途

复杂逻辑必须补充注释。

不要解释 Python 语法。

重点解释：

为什么这样设计。

---

# 文件修改规范

修改文件时：

保持：

Import 顺序。

命名风格。

代码风格。

不要一次修改整个文件。

仅修改必要内容。

---

# 删除代码

删除代码前：

确认：

没有引用。

没有 Graph 使用。

没有插件调用。

没有配置依赖。

禁止：

遗留死代码。

---

# 插件兼容性

最终所有 Workflow 必须能够：

注册到 MASFactory。

能够生成 Graph。

能够被插件识别。

能够展示：

Agent

Edge

Workflow

Execution Flow

Runtime State

因此：

所有新增组件必须符合 MASFactory 的插件要求。

不要绕开 Graph。

不要绕开 Workflow。

不要直接调用底层 Agent。

---

# 可维护性

任何新增模块：

必须易于：

替换

扩展

测试

组合

不要为了当前需求牺牲后续维护。

---

# 输出要求

AI 每次修改代码时：

必须说明：

1. 修改原因；

2. 修改文件；

3. 是否影响其它模块；

4. 是否需要同步修改；

5. 是否影响 Graph；

6. 是否影响插件展示；

7. 是否影响配置；

8. 是否需要迁移已有代码。

不要输出无意义解释。

保持修改最小化。

优先保证系统一致性。

---

# 最终目标

最终完成后的项目应满足：

✓ 保持 MaAS 算法逻辑一致；

✓ 完全运行于 MASFactory；

✓ Workflow 可正常执行；

✓ Graph 可正确构建；

✓ Plugin 能正确识别所有节点；

✓ Plugin 能完整展示执行流程；

✓ 所有异常可定位；

✓ 所有模块职责清晰；

✓ 无兜底逻辑；

✓ 无静默失败；

✓ 易于继续维护与扩展。