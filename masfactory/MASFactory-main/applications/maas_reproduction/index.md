# MASFactory 原生 MaAS 复现目录与职责


## 顶层执行结构

```text
DatasetRunner
└── RootGraph.invoke(one_sample)
    ├── InputSplitNode
    ├── ArchitectureExecGraph
    │   ├── InitializeExecutionNode
    │   ├── RoutePlannerNode
    │   ├── NativeBootstrapGraph
    │   ├── OperatorDispatchLoop
    │   ├── BenchmarkCompletionGraph
    │   └── FinalizeArchitectureResultNode
    ├── EvaluatorNode
    ├── LossUpdateNode / MetricsNode
    └── SampleResultNode
```

`BenchmarkCompletionGraph` 只承接原 MaAS 各数据集的确定性后置路径。它不修改策略生成的 `RoutePlan` 或 `policy_log_prob`，也不把 operator 执行塞入 Finalize。

## 完整目录树与职责

```text
applications/maas_reproduction/                         # MaAS 原生复现应用根目录；隔离实验代码与 MASFactory 框架代码
├── __init__.py                                         # 将应用根目录声明为 Python 包；不在导入阶段构图或启动实验
├── main.py                                             # 统一 CLI 入口；解析 train/test/smoke 参数并调用 runtime，不实现业务规则
├── workflow.py                                         # 构建 TrainRootGraph 和 TestRootGraph；只装配顶层 Graph、Node 和 Edge
├── README.md                                           # 面向使用者的安装、配置、训练、测试和产物说明
├── .env.example                                        # 环境变量名称模板；不得保存真实 API key 或其他凭据
├── pytest.ini                                          # 隔离本应用测试发现路径，避免与外层同名目录冲突
├── agent.md                                            # 当前目录的工程开发约束
├── index.md                                            # 本目录树、模块职责、消息契约和实现索引
├── maas_reproduction_spec.md                           # MASFactory 原生 MaAS 最终实施方案
├── maas_source.md                                      # 原 MaAS 源码事实与等价映射
├── masfactory_spec.md                                  # MASFactory 多智能体系统工程规范
│
├── components/                                        # MASFactory 可执行 Graph、Loop、Agent 和 CustomNode；表达工作流结构
│   ├── __init__.py                                     # 集中导出稳定的顶层构建器和节点类型
│   ├── input_split_node.py                             # 将单题输入拆成 architecture_request 与 evaluation_context；隔离 expected_answer
│   ├── evaluator_node.py                               # 用 benchmark scorer 将 ArchitectureResult 与 EvaluationContext 转为 EvaluationResult
│   ├── loss_update_node.py                             # 构造 TrainingSignal，并把合格 Tensor/utility 交给 BatchAccumulator
│   ├── metrics_node.py                                 # Test 模式累计准确率、成本、跳过原因和失败统计；禁止更新 controller
│   ├── sample_result_node.py                           # 生成可序列化公开结果；在训练使用完成后 detach policy Tensor
│   │
│   ├── architecture_exec_graph/                       # 单题架构执行深模块；interface 为 ArchitectureRequest → ArchitectureResult
│   │   ├── __init__.py                                 # 导出 ArchitectureExecGraph 构建器
│   │   ├── workflow.py                                 # 固定装配 Initialize→Planner→Bootstrap→Dispatch→Completion→Finalize
│   │   └── components/                                # ArchitectureExecGraph 私有节点，不作为应用顶层入口
│   │       ├── __init__.py                             # 导出本子图内部节点
│   │       ├── initialize_execution_node.py            # 校验请求、记录单题成本快照并建立最小执行上下文
│   │       ├── route_planner_node.py                   # 唯一策略决策点；调用 policy_controller 并生成 RoutePlan/policy_log_prob
│   │       └── finalize_architecture_result_node.py    # 选择 prediction、结算 cost_delta、判断可靠性并生成 ArchitectureResult
│   │
│   ├── native_bootstrap_graph/                        # MATH 固定前置初始化深模块；复用已有 operator Graph
│   │   ├── __init__.py                                 # 导出 NativeBootstrapGraph 构建器
│   │   ├── workflow.py                                 # 组合 ProgrammerGraph→Generate refinement→BootstrapResultNode
│   │   └── components/                                # Bootstrap 专属确定性节点
│   │       ├── __init__.py                             # 导出 Bootstrap 内部节点
│   │       └── bootstrap_result_node.py                # 将复用 operator 的结果汇总为初始 DispatchState
│   │
│   ├── operator_dispatch_loop/                        # 唯一动态路线执行循环；按 RoutePlan 顺序消费 operator
│   │   ├── __init__.py                                 # 导出 OperatorDispatchLoop 构建器和统一 keys
│   │   ├── workflow.py                                 # 创建 MASFactory Loop、Controller 边、终止函数和 operator 分支
│   │   └── components/                                # Dispatch Loop 的游标、路由和状态转换节点
│   │       ├── __init__.py                             # 导出循环内部节点
│   │       ├── route_cursor_node.py                    # 从 DispatchState 当前游标构造只读 OperatorInvocation 快照
│   │       ├── logic_switch.py                         # 用纯 predicate 将 operator_name 路由到对应 OperatorGraph
│   │       ├── state_reducer_node.py                   # 唯一动态 DispatchState 写入口；生成新状态并推进 route_cursor
│   │       └── invalid_operator_node.py                # 将未知 operator 转为结构化执行错误，不添加 utility penalty
│   │
│   ├── benchmark_completion_graph/                    # 数据集固定后置执行深模块；DispatchState → CompletionResult
│   │   ├── __init__.py                                 # 导出 BenchmarkCompletionGraph 构建器
│   │   ├── workflow.py                                 # 按 dataset/mode 路由后置 Graph，并合并统一 CompletionResult
│   │   ├── gsm8k_completion_graph.py                   # GSM8K 最终 ensemble 与训练态 Programmer 验证；复用 operator Graph
│   │   ├── math_completion_graph.py                    # MATH 候选多于一个时执行最终 ScEnsemble，否则透传当前解
│   │   ├── humaneval_completion_graph/                # HumanEval 固定 Test→失败修复控制流
│   │   │   ├── __init__.py                             # 导出 HumanEvalCompletionGraph 构建器
│   │   │   ├── workflow.py                             # 装配公共测试、结果分支、修复和统一输出
│   │   │   └── components/                            # HumanEval 后置流程私有节点
│   │   │       ├── __init__.py                         # 导出 HumanEval completion 内部节点
│   │   │       ├── humaneval_test_node.py              # 执行可靠、限时的公共测试并返回结构化 TestResult
│   │   │       ├── test_result_switch.py               # 纯路由：测试通过直接结束，失败进入 repair
│   │   │       └── humaneval_repair_node.py            # 使用共享 Model 按测试反馈生成修复代码；不直接调用 provider SDK
│   │   └── components/                                # Completion Graph 的共享输入、路由和归一化节点
│   │       ├── __init__.py                             # 导出 completion 共享节点
│   │       ├── dataset_completion_switch.py            # 纯路由：选择 GSM8K、MATH、HumanEval 或无需后置流程分支
│   │       ├── completion_input_node.py                 # 从 DispatchState 构造不含 policy Tensor 变更的 CompletionInvocation
│   │       ├── completion_result_node.py                # 将各数据集分支归一化为统一 CompletionResult
│   │       └── no_completion_node.py                    # 对无需固定后置步骤的场景生成透传 CompletionResult
│   │
│   └── operators/                                     # 可复用原生 operator Graph；统一 interface 为 Invocation → OperatorResult
│       ├── __init__.py                                 # 导出 operator Graph 构建器与 registry
│       ├── registry.py                                 # 保存 operator 类/模板/配置；禁止保存预构建 Graph 实例
│       ├── native_agent_operator_graph.py              # 共享 Generate、GenerateCoT、SelfRefine 的单 Agent 实现
│       ├── multi_generate_cot_graph.py                 # 固定生成多个独立候选并返回结构化 OperatorResult
│       ├── sc_ensemble_graph.py                        # 标记候选、选择合法候选并提供可验证 fallback
│       ├── humaneval_test_graph.py                     # 可选动态 HumanEval Test operator；与固定后置测试复用测试 seam
│       ├── early_stop.py                               # 确定性 control marker；不调用模型、不产生人工惩罚
│       └── programmer_graph/                           # 代码生成、解析、沙箱执行和有限重试深模块
│           ├── __init__.py                             # 导出 ProgrammerGraph 构建器
│           ├── workflow.py                             # 装配 ProgrammerRetryLoop 与最终结果出口
│           ├── programmer_retry_loop.py                # 创建真实 MASFactory Loop；声明最大次数、终止函数和往返 keys
│           └── components/                            # Programmer 内部 Agent 与确定性执行节点
│               ├── __init__.py                         # 导出 Programmer 私有节点
│               ├── code_generation_agent.py            # 根据问题、当前解和上次错误生成 solve/code；只通过 MASFactory Model
│               ├── code_parse_node.py                  # 清理并验证模型代码输出，拒绝无法执行的结构
│               ├── programmer_execution_node.py        # 通过注入的 CodeExecutor 限时执行并捕获 stdout/stderr/exit 状态
│               ├── retry_decision_node.py              # 纯计算是否成功、可重试或已耗尽；不代替 Loop
│               └── programmer_result_node.py            # 将成功、timeout 或耗尽结果归一化为 OperatorResult
│
├── maas_reproduction/                                 # 与具体图结构解耦的领域、策略、训练和运行实现
│   ├── __init__.py                                     # 导出应用稳定 interface；不触发运行时副作用
│   ├── contracts.py                                    # 冻结 operator 顺序、bootstrap、EarlyStop、logprob、utility 与 policy loss 口径
│   ├── schemas.py                                      # ArchitectureRequest/Result、OperatorInvocation/Result、EvaluationResult 等不可变契约
│   ├── state.py                                        # 兼容导出 RouteItem、RoutePlan、DispatchState；实现统一位于 schemas.py
│   ├── reducers.py                                     # DispatchState 的纯 reducer；集中候选、错误、终止和 cursor 规则
│   ├── models/                                        # 可训练 policy controller 及 operator embedding
│   │   ├── __init__.py                                 # 导出 controller 和 embedding 构建 interface
│   │   ├── controller.py                               # 复现 OperatorSelector/MultiLayerController 和分层采样规则
│   │   └── embeddings.py                               # operator 描述编码、共享 encoder/cache 和顺序一致性校验
│   ├── benchmarks/                                    # 三类数据集加载后可调用的评分规则
│   │   ├── __init__.py                                 # 按数据集名称选择 scorer
│   │   ├── base.py                                     # scorer 共同 interface 与可靠性结果，不管理 optimizer
│   │   ├── gsm8k.py                                    # 最后数值抽取与 1e-6 容差评分
│   │   ├── math.py                                     # boxed/末句抽取、数值与符号等价评分
│   │   └── humaneval.py                                # 代码清理、check(candidate) 执行与二值评分
│   ├── training/                                      # 训练资格与梯度生命周期
│   │   ├── __init__.py                                 # 导出 TrainingSignal 和 BatchAccumulator
│   │   ├── training_signal.py                          # 唯一训练资格判断；utility 固定为 score - 3×cost_delta
│   │   └── batch_accumulator.py                        # 聚合有效 Tensor、计算原始 batch loss、backward/step/flush
│   ├── adapters/                                      # 外部依赖 seam 的具体适配器
│   │   ├── __init__.py                                 # 导出可注入 adapter
│   │   ├── cost_tracker.py                             # 对共享 Model usage 做单题 snapshot/delta，并显式报告可靠性
│   │   ├── dataset_loader.py                           # 加载、校验并规范化 GSM8K/MATH/HumanEval 样本
│   │   ├── code_executor.py                            # 隔离进程、timeout、危险能力限制和结构化执行结果
│   │   └── artifact_store.py                           # 写入配置、日志、样本结果、指标和版本信息
│   └── runtime/                                       # 运行时对象构造与跨样本生命周期
│       ├── __init__.py                                 # 导出 settings/bootstrap/runner interface
│       ├── settings.py                                 # 唯一配置读取与校验入口；区分静态、实验和敏感配置
│       ├── bootstrap.py                                # 创建共享 Model、controller、optimizer、adapters 和 RootGraph
│       ├── dataset_runner.py                           # 唯一跨样本生命周期；维护 epoch/cursor、调用 RootGraph、汇总结果
│       ├── checkpoint_manager.py                       # 仅在 batch 边界保存/恢复 controller、optimizer、cursor 和 RNG
│       └── seed.py                                     # 设置和捕获 Python、NumPy、PyTorch、CUDA RNG 状态
│
├── assets/                                            # 可序列化配置、prompt 和运行产物；不保存运行时对象
│   ├── config/                                        # 唯一受版本控制的实验配置目录
│   │   ├── models.json                                 # Model provider、名称和环境变量引用；不保存真实密钥
│   │   ├── experiments.json                            # dataset、epoch、batch、学习率、模式和输出设置
│   │   ├── bootstrap.json                              # 数据集固定前置流程及 retry 参数
│   │   ├── operators.json                              # operator 顺序、描述、可用数据集和控制标记
│   │   └── evaluation.json                             # scorer、timeout、可靠性和成本系数配置
│   ├── prompts/                                       # 与 Python 实现分离的 Agent 指令和 prompt 模板
│   │   ├── shared/                                     # Generate、GenerateCoT、SelfRefine、ensemble 等共享模板
│   │   ├── gsm8k/                                      # GSM8K 求解、refine 和 Programmer 验证模板
│   │   ├── math/                                       # MATH 详细解答、代码输出整理和 ensemble 模板
│   │   └── humaneval/                                  # 代码生成、公共测试反思和修复模板
│   └── output/                                        # 每次运行独立的 run_id 目录；保存结果、日志和 checkpoint
│
├── scripts/                                           # 面向人工调用的薄入口；所有实现委托给 main/runtime
│   ├── run_smoke.py                                    # fake Model 或极小数据集的构建/invoke 冒烟入口
│   ├── run_train.py                                    # 训练模式便捷入口
│   └── run_test.py                                     # checkpoint 测试模式便捷入口
│
└── tests/                                             # 通过正式 interface 验证领域规则、节点、图结构和端到端行为
    ├── __init__.py                                     # 将应用测试声明为包
    ├── unit/                                          # 不调用真实网络的纯函数、节点和 adapter 测试
    │   ├── test_contracts.py                           # operator/bootstrap 顺序及 logprob、utility、policy loss 冻结值
    │   ├── test_schemas.py                             # 类型约束、不可变性、序列化和非法字段
    │   ├── test_route_planner.py                       # 分层展开、sequence_index、EarlyStop 和 logprob 聚合
    │   ├── test_dispatch_termination.py                # cursor/termination_requested 的 Loop 终止真值表
    │   ├── test_reducers.py                            # 状态不原地修改、候选归约、cursor 推进和错误处理
    │   ├── test_training_signal.py                     # 更新资格、skip reason 和零额外 failure penalty
    │   ├── test_batch_accumulator.py                   # batch loss、partial flush、optimizer step 和 Tensor 清理
    │   ├── test_cost_tracker.py                        # 正常/缺失/负数/非有限成本及 tracker reset
    │   └── test_benchmarks.py                          # GSM8K、MATH、HumanEval 的成功与评分异常场景
    ├── graphs/                                        # MASFactory build、Edge、Loop、Switch 和可达性测试
    │   ├── test_root_graphs.py                         # Train/Test RootGraph 构建、分支和 Test 无 optimizer
    │   ├── test_architecture_exec_graph.py             # 固定阶段顺序、契约和 expected_answer 隔离
    │   ├── test_native_bootstrap_graph.py              # 复用 operator、MATH 初始化和 fatal bootstrap failure
    │   ├── test_operator_dispatch_loop.py              # Controller 往返 keys、Switch 路由、InvalidOperator 和 hydration
    │   ├── test_programmer_retry_loop.py               # 最大重试、timeout 恢复、耗尽和结构化 OperatorResult
    │   ├── test_benchmark_completion_graph.py          # 三数据集固定后置路径及策略 logprob 不变
    │   └── test_operator_graphs.py                     # 所有 operator 的 Invocation→Result interface
    └── integration/                                   # 使用 fake dependencies 的完整 invoke/训练生命周期测试
        ├── test_fake_model_pipeline.py                 # fake Model 下的单题 Train/Test 完整链路
        ├── test_gradient_chain.py                      # policy_log_prob 从 RoutePlan 保持到 backward
        ├── test_checkpoint_resume.py                   # batch 边界保存及 controller/optimizer/cursor/RNG 恢复
        ├── test_expected_answer_isolation.py           # expected_answer 不进入执行图、prompt、状态或 trace
        └── test_fixed_completion_paths.py              # GSM8K ensemble/验证、MATH ensemble、HumanEval Test/Repair
```
