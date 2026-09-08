# MaAS-main 源码全量介绍（源码事实版）

> 这是一份只根据 MaAS-main 当前源码整理的工程导读。文档的目标是让另一个 AI 能够从代码结构、调用关系、数据契约和运行入口上理解这个项目实际做了什么。
>
> 本文不把论文中的定义、理想流程或实验描述当成代码事实。凡是论文可能与源码不同的地方，均以源码中的实际调用和返回值为准。
>
> 源码根目录：C:/Users/lenovo/Desktop/论文复现相关/MaAS-main
>
> 复现目录：masfactory/MASFactory-main/applications/maas_reproduction

## 1. 项目整体定位

MaAS-main 是一个以 MetaGPT 风格组织的 Python 多智能体工程。仓库同时包含两部分：

1. 通用 agent framework：负责配置、上下文、消息、LLM 调用、结构化 Action、工具、RAG、文件/代码处理和若干策略组件。
2. MaAS 扩展：位于 maas.ext.maas，负责数据集上的动态算子选择、工作流执行、控制器训练、评测和可选 prompt 改写。

从源码看，项目最明确、最完整的实验入口是：

~~~
python -m examples.maas.optimize --dataset GSM8K
~~~

这个入口并不是一个通用的多智能体聊天应用，而是一个“控制器选择工作流算子并依据任务得分更新控制器”的实验程序。

项目的关键依赖关系可以概括为：

~~~
命令行入口
  -> ModelsConfig / ExperimentConfig
  -> Optimizer
  -> Dataset Evaluator
  -> optimized/{dataset}/{train|test}/graph.py
  -> Workflow.__call__
  -> Controller.forward
  -> operator implementations
  -> LLM Provider / ActionNode
  -> dataset score and cost
  -> controller loss and checkpoint
~~~

通用框架的主要依赖关系为：

~~~
Action
  -> ActionNode
  -> Pydantic output model
  -> BaseLLM.aask
  -> concrete provider

ToolRegistry
  -> tool schema extraction
  -> tool recommendation
  -> prompt injection

SimpleEngine
  -> document/object loading
  -> node transformation
  -> index
  -> retriever
  -> ranker
  -> response synthesizer
~~~

## 2. 源码目录地图

### 2.1 顶层目录

~~~text
MaAS-main/
├── maas/
│   ├── actions/       Action、ActionNode、具体业务动作
│   ├── configs/       LLM、模型、搜索、浏览器、RAG 等配置模型
│   ├── ext/maas/      MaAS 动态工作流搜索和训练实现
│   ├── prompts/       通用任务 prompt
│   ├── provider/      LLM provider 抽象和多 provider 适配
│   ├── rag/           RAG schema、factory、retriever、ranker、engine
│   ├── strategy/      Planner、Tree-of-Thought、solver 抽象
│   ├── tools/         工具注册、工具 schema、搜索、浏览器、转换工具
│   ├── utils/         文件、Git、解析、序列化、成本、日志等基础设施
│   ├── context.py     全局运行上下文
│   ├── document.py    文档处理相关类型
│   ├── llm.py         LLM 工厂入口
│   ├── schema.py      Message、Task、Plan、Document 等核心数据类型
│   └── config2.py     全局 Config 和配置加载
├── examples/maas/     MaAS 训练和 smoke test 入口
├── config/            config2.example.yaml
├── assets/             MaAS 图片和框架图片
├── setup.py           安装元数据和可选依赖
└── requirements.txt   依赖锁定列表
~~~

### 2.2 MaAS 扩展目录

~~~text
maas/ext/maas/
├── benchmark/
│   ├── benchmark.py        BaseBenchmark 和异步 batch 训练
│   ├── gsm8k.py            GSM8K 评分
│   ├── math.py             MATH 评分
│   ├── humaneval.py        HumanEval 代码执行评分
│   ├── experiment_configs.py
│   └── utils.py
├── data/                   gsm8k、math 数据文件
├── models/
│   ├── controller.py       OperatorSelector、MultiLayerController
│   └── utils.py            SentenceTransformer 和 operator sampling
└── scripts/
    ├── optimizer.py        主训练/测试 orchestration
    ├── evaluator.py        根据数据集加载 benchmark
    ├── optimizer_utils/    结果、图、经验和收敛辅助类
    ├── textgrad/           prompt 改写和独立文本梯度脚本
    └── optimized/
        ├── GSM8K/
        ├── MATH/
        └── HumanEval/
            ├── train/
            └── test/
                ├── graph.py
                └── template/
                    ├── operator.py
                    ├── operator_an.py
                    ├── operator.json
                    ├── operator_registry.py
                    ├── op_prompt.py
                    └── prompt.py
~~~

### 2.3 包初始化行为

maas/__init__.py 会导入 maas._compat。

maas/actions/__init__.py 会集中导出大量 Action，并定义 ActionType 枚举。导出的动作包括需求、PRD、设计、代码、测试、运行、调试、搜索和研究等。

maas/tools/__init__.py 会导入 maas.tools.libs。这个导入具有副作用：工具库中的工具通过装饰器注册到全局 TOOL_REGISTRY。

因此，不能只导入某个工具类就假定所有工具已经注册；工具注册是否发生取决于相应包是否被导入。

## 3. 配置、路径和运行上下文

### 3.1 配置文件加载

maas.config2.Config 是全局配置模型，字段包括：

- llm：默认 LLM 配置；
- embedding：RAG embedding 配置；
- omniparse：文档解析服务配置；
- search：搜索引擎配置；
- browser：浏览器配置；
- mermaid：图渲染配置；
- s3、redis：外部存储；
- workspace：工作区；
- prompt_schema：json、markdown 或 raw；
- repair_llm_output：是否尝试修复非法 LLM 输出；
- enable_longterm_memory、code_review_k_times 等其他运行开关。

Config.default() 的合并顺序是：

~~~text
环境变量
  -> METAGPT_ROOT/config/config2.yaml
  -> ~/.maas/config2.yaml
~~~

后面的配置覆盖前面的配置。

maas.configs.models_config.ModelsConfig 单独管理 models 字段，可按模型名或 api_type 查找 LLMConfig。MaAS 优化器通过 ModelsConfig.default().get(model_name) 获取执行模型。

### 3.2 根目录和路径常量

maas.const.py 定义：

- CONFIG_ROOT：用户目录下的 .maas；
- METAGPT_ROOT：由 METAGPT_PROJECT_ROOT 或包路径推导；
- DEFAULT_WORKSPACE_ROOT：workspace；
- DATA_PATH：data；
- SOURCE_ROOT：maas；
- PROMPT_PATH：maas/prompts；
- TOOL_SCHEMA_PATH：maas/tools/schemas；
- TOOL_LIBS_PATH：maas/tools/libs；
- 多种 docs、resources、tests、storage 路径；
- LLM/API timeout 和消息路由常量。

这里保留了大量 MetaGPT 命名，例如 METAGPT_ROOT、metagpt_tti_url 和 setup.py 中的 package name metagpt，但当前源码主体目录实际名为 maas。复现时应以实际 import 路径为准。

### 3.3 Context

maas.context.Context 是运行时环境对象，包含：

- kwargs：可扩展的 AttrDict；
- config：全局 Config；
- repo：ProjectRepo；
- git_repo：GitRepository；
- src_workspace；
- cost_manager；
- 缓存的 _llm。

Context.llm() 根据 config.llm 创建 provider，并在需要时选择 CostManager：

- FIREWORKS -> FireworksCostManager；
- OPEN_LLM -> TokenCostManager；
- 其他类型 -> Context 自己的 CostManager。

Context.llm_with_cost_manager_from_llm_config() 用于用指定 LLMConfig 创建独立 LLM。

Context.serialize() 只序列化工作目录、kwargs 和成本管理器；deserialize() 会尝试恢复 GitRepository、ProjectRepo 和成本状态。

## 4. 核心数据模型

### 4.1 SerializationMixin

schema.py 中的 SerializationMixin 让 Pydantic 模型能保存自己的模块名和类名：

~~~text
序列化：
  普通字段 + __module_class_name

反序列化：
  读取 __module_class_name
  -> 在子类注册表中查找真实类
  -> 使用真实类构造对象
~~~

它被 Action 等多态对象使用，目的是让不同子类在持久化后还能恢复具体类型。

### 4.2 Document 和 Documents

Document 只有三个核心字段：

- root_path；
- filename；
- content。

Document.root_relative_path 将 root_path 和 filename 拼接。Documents 以 filename 为 key 管理多个 Document，并能转成 ActionOutput。

### 4.3 Message

Message 是通用消息类型，字段包括：

- id；
- content；
- instruct_content；
- role：system、user、assistant；
- cause_by：消息来源 Action；
- sent_from；
- send_to。

Message 会把 cause_by、sent_from 转成字符串，把 send_to 转成字符串集合。instruct_content 可以是 Pydantic 模型，也可以是 ActionNode 动态创建的模型。

Message.to_dict() 只给 LLM 调用返回 role 和 content。

### 4.4 Task、Plan 和 TaskResult

schema.py 还定义了任务规划相关模型：

- Task：task_id、依赖 task id、instruction、task_type、code、result、is_finished、is_success 等；
- Plan：goal、context、tasks、current_task_id 等；
- TaskResult：任务执行结果和成功状态。

Planner 通过 Plan.current_task 管理当前任务，通过 finish_current_task 推进任务。

### 4.5 其他上下文模型

schema.py 还定义 CodingContext、TestingContext、RunCodeContext、RunCodeResult、CodeSummarizeContext、BugFixContext、CodePlanAndChangeContext，以及 UMLClassView 等结构，用于把代码工程任务、测试、运行和图视图信息传给 Action。

## 5. LLM 抽象和 Provider

### 5.1 统一入口

maas.llm.LLM(llm_config=None, context=None) 是通用入口：

1. 没有传入 config 时创建 Context 并调用 Context.llm()；
2. 传入 LLMConfig 时通过 Context.llm_with_cost_manager_from_llm_config() 创建指定 provider；
3. 返回 BaseLLM 子类实例。

### 5.2 LLMConfig

LLMConfig 包含：

- api_key、api_type、base_url、api_version；
- model、pricing_plan；
- access_key、secret_key、session_token、endpoint；
- app_id、api_secret、domain；
- max_token、temperature、top_p、top_k；
- repetition_penalty、stop、presence_penalty、frequency_penalty；
- stream、seed、logprobs、timeout、context_length；
- proxy、calc_usage、use_system_prompt。

api_key 和 timeout 有校验。api_key 未配置时会提示在 .maas/config2.yaml 或项目 config2.yaml 中配置。

### 5.3 Provider 注册

provider/llm_provider_registry.py 提供：

- LLMProviderRegistry；
- register_provider(keys) 装饰器；
- create_llm_instance(config)；
- 全局 LLM_REGISTRY。

各具体 provider 模块通过装饰器注册到 LLM_REGISTRY。源码中包含 OpenAI、Azure OpenAI、Anthropic、Gemini、Ollama、Qianfan、DashScope、Spark、ZhipuAI、Bedrock、Ark、Metagpt、Human 等适配文件。

create_llm_instance() 根据 config.api_type 取 provider 类，并处理 use_system_prompt。

### 5.4 BaseLLM 调用链

BaseLLM 统一处理消息和成本，具体网络请求交给子类：

~~~text
BaseLLM.aask()
  -> 选择 system messages
  -> 追加 format_msgs 或 user message
  -> 处理图片 URL/base64
  -> 根据 config.stream 决定是否流式
  -> acompletion_text()
  -> _achat_completion() 或 _achat_completion_stream()
  -> get_choice_text()
~~~

BaseLLM 支持：

- str、Message、dict、消息列表；
- system prompt；
- 图片输入；
- batch 对话；
- function/tool call 结果提取；
- API 超时；
- ConnectionError 重试；
- usage token 成本更新；
- stream 和非 stream。

BaseLLM.aask_code() 是抽象接口，默认未实现。

## 6. Action 和 ActionNode

### 6.1 Action

maas.actions.action.Action 是 Pydantic BaseModel，同时继承 SerializationMixin 和 ContextMixin。

Action 的主要字段：

- name；
- i_context；
- prefix；
- desc；
- node；
- llm_name_or_type。

Action 初始化时，如果传入 llm_name_or_type，会从 ModelsConfig 中取对应模型并替换内部 LLM，同时复用已有 cost_manager。

Action 支持：

- set_prefix()：设置系统 prompt；
- repo、prompt_schema、project_name、project_path 属性；
- _aask()；
- _run_action_node()；
- run()。

如果 Action 有 node，run() 会把历史消息整理成 History Messages 后调用 node.fill()；没有 node 的 Action 需要自己实现 run()。

### 6.2 ActionNode 的数据结构

ActionNode 是一个节点树，既可以表示一个输出字段，也可以表示多个嵌套字段。关键字段包括：

- key；
- expected_type；
- instruction；
- example；
- content；
- children；
- prevs、nexts；
- llm、context；
- instruct_content。

节点关系有两种：

1. children：用于构造嵌套输出 schema；
2. prevs/nexts：用于 ActionGraph 的依赖边。

### 6.3 Pydantic 动态输出模型

ActionNode.create_model_class() 根据节点映射动态创建 Pydantic 模型：

~~~text
ActionNode
  -> get_mapping()
  -> mapping: field -> (type, Field)
  -> create_model_class()
  -> instruct_content
~~~

创建模型时会检查必需字段是否缺失，并对未识别字段记录 warning。

ActionNode.from_pydantic(Model) 会把 Pydantic 模型的字段转成 ActionNode children。MaAS 的 operator.py 正是通过这种方式把 GenerateOp、CodeGenerateOp、ScEnsembleOp 等 Pydantic 模型变成结构化 LLM 输出节点。

### 6.4 Prompt 编译

ActionNode 可以把节点结构编译成：

- raw；
- json；
- markdown。

默认 SIMPLE_TEMPLATE 包含：

- context；
- format example；
- nodes 字段、类型和 instruction；
- 语言约束；
- CONTENT 包裹约束；
- 最后的 action instruction。

compile_instruction() 输出字段类型和要求，compile_example() 输出示例值，compile() 把两者注入模板。

### 6.5 三种特殊填充模式

ActionNode.fill() 支持三种 MaAS 代码中实际使用的 mode。

#### single_fill

直接把 context 交给 LLM，返回第一个输出字段：

~~~text
LLM 原始文本
  -> {field_name: raw_text}
  -> Pydantic instruct_content
~~~

#### xml_fill

先根据字段名追加 XML 输出格式，然后从 LLM 输出中提取：

~~~text
<field_a>...</field_a>
<field_b>...</field_b>
~~~

支持 str、int、bool、list、dict 的简单转换。list 和 dict 使用 eval(raw_value)，这是源码实际行为，也意味着复现时要注意安全边界。

#### code_fill

直接请求 LLM 返回代码，再调用 sanitize(code, entrypoint) 清理输出，最后包装为第一个字段。

HumanEval 的代码生成和数学 Programmer 都使用 code_fill。

### 6.6 普通填充、review 和 revise

如果 mode 不是上述三种，ActionNode.fill() 根据 schema、mode、strgy 执行 simple_fill 或 complex fill。

- simple：一次性填充当前节点/子节点；
- complex：逐个子节点填充再合并。

ActionNode 还提供：

- human_review；
- auto_review；
- simple_review；
- human_revise；
- auto_revise；
- simple_revise。

它们通过 REVIEW_TEMPLATE 和 REVISE_TEMPLATE 让 LLM 对结构化输出做检查或修订。

## 7. ActionGraph 和 Solver

### 7.1 ActionGraph

ActionGraph 是一个轻量有向图：

- nodes：key -> ActionNode；
- edges：from key -> next keys；
- execution_order：拓扑排序结果。

add_edge() 同时更新 ActionNode 的 nexts 和 prevs。topological_sort() 用 DFS 将节点放入 execution_order。

### 7.2 Solver

strategy/solver.py 提供 BaseSolver 及多种 solver 类型：

- NaiveSolver：实际实现，先拓扑排序，再顺序对 graph 节点调用 fill；
- TOTSolver：未实现；
- DataInterpreterSolver：未实现；
- ReActSolver：未实现；
- IOSolver：未实现；
- COTSolver：未实现。

所以这些 solver 名称是接口/占位能力；当前源码里真正可运行的是 NaiveSolver。

## 8. Actions 业务层

actions 目录包含大量业务动作，它们大多继承 Action，并通过 ActionNode 定义结构化输出。

主要类别如下：

| 类别 | 主要源码文件 | 作用 |
|---|---|---|
| 需求 | add_requirement.py | 把用户需求包装成 UserRequirement |
| PRD | write_prd.py、write_prd_review.py、write_prd_an.py | 生成和评审产品需求文档 |
| 系统设计 | design_api.py、design_api_an.py、design_api_review.py | 生成 API/系统设计 |
| 项目管理 | project_management.py、project_management_an.py | 生成任务和项目管理内容 |
| 代码 | write_code.py、write_code_an_draft.py、write_code_plan_and_change_an.py | 生成代码、计划和修改 |
| 代码评审 | write_code_review.py | 审查代码 |
| 测试 | write_test.py | 生成测试内容 |
| 运行 | run_code.py | 执行代码相关流程 |
| 调试 | debug_error.py、fix_bug.py | 处理错误和修复 |
| 文档 | summarize_code.py、prepare_documents.py、write_docstring.py | 代码摘要、文档准备和 docstring |
| 研究 | research.py、search_and_summarize.py、prepare_interview.py | 搜索、研究、摘要和面试准备 |
| 教学 | write_tutorial.py、write_teaching_plan.py、generate_questions.py | 教程、教案、问题生成 |
| 图视图 | rebuild_class_view.py、rebuild_sequence_view.py | 重新构造 class/sequence 视图 |
| DI | actions/di/* | notebook、分析代码和计划相关动作 |

actions/__init__.py 中的 ActionType 枚举用于集中索引这些 Action。

这些通用 Actions 与 MaAS 动态 operator 不是同一套东西：

- Action 是框架级业务动作；
- MaAS operator 是 benchmark 工作流内的可选算子；
- MaAS operator 主要位于 optimized/{dataset}/{train|test}/template/operator.py。

## 9. Tool 子系统

### 9.1 Tool 数据结构

tool_data_type.py 定义：

- ToolSchema：目前主要要求 description；
- Tool：name、path、schemas、code、tags。

### 9.2 ToolRegistry

ToolRegistry 是全局工具注册表，维护：

- tools：name -> Tool；
- tools_by_tags：tag -> tools。

注册工具的方式：

1. 使用 register_tool() 装饰器；
2. 从 Python 文件读取 AST，批量注册；
3. 从目录递归注册；
4. 直接调用 register_tool() 传入 source object。

注册时会：

- 读取 source code；
- 从 docstring 和 signature 生成 schema；
- 将 schema 写入 maas/tools/schemas；
- 保存 tool path、code、tags。

### 9.3 Schema 生成

tool_convert.py 提供两条路径：

- inspect 路径：读取函数/类、docstring、signature；
- AST 路径：解析 Python 源码并提取函数、类、方法和代码片段。

生成的函数 schema 包含：

- type：function 或 async_function；
- description；
- signature；
- parameters。

### 9.4 Tool 名称校验和推荐

validate_tool_names() 接受：

- 已注册工具名；
- tag；
- 文件路径；
- 目录路径。

ToolRecommender 有两个阶段：

~~~text
recall
  -> 从工具池召回候选
rank
  -> 让 LLM 选择最终工具
~~~

当前实现：

- TypeMatchToolRecommender：按 task_type 和 tag 精确匹配；
- BM25ToolRecommender：对 tool name、tags、description 做 BM25 召回；
- EmbeddingToolRecommender：只有接口，recall_tools() 未实现；
- ToolRecommender.rank_tools()：让 LLM 输出 JSON 工具名列表，再校验工具名。

如果 force=True，或没有 context/plan，推荐器会直接返回全部指定工具。

### 9.5 工具类别

tools 目录包含：

- 搜索引擎：Bing、DDG、Google API、SerpAPI、Serper、Meilisearch；
- 浏览器：Playwright、Selenium、可扩展浏览器基类；
- 代码/单元测试：ut_writer；
- 翻译；
- 图像生成；
- 文本 embedding；
- 文本转语音；
- OpenAPI 示例；
- moderation；
- Prompt writer；
- 数据处理和 feature engineering；
- schemas 目录下的 YAML 工具定义。

## 10. RAG 子系统

### 10.1 RAG 的总体实现

RAG 依赖 LlamaIndex。当前代码把 RAG 拆成：

~~~text
embedding factory
  -> index factory
  -> retriever factory
  -> ranker factory
  -> engine
~~~

### 10.2 RAG schema

rag/schema.py 定义了配置类：

Retriever：

- BaseRetrieverConfig；
- IndexRetrieverConfig；
- FAISSRetrieverConfig；
- BM25RetrieverConfig；
- MilvusRetrieverConfig；
- ChromaRetrieverConfig；
- ElasticsearchRetrieverConfig；
- ElasticsearchKeywordRetrieverConfig。

Ranker：

- BaseRankerConfig；
- LLMRankerConfig；
- ColbertRerankConfig；
- CohereRerankConfig；
- BGERerankConfig；
- ObjectRankerConfig。

Index：

- BaseIndexConfig；
- VectorIndexConfig；
- FAISSIndexConfig；
- ChromaIndexConfig；
- MilvusIndexConfig；
- BM25IndexConfig；
- ElasticsearchIndexConfig；
- ElasticsearchKeywordIndexConfig。

### 10.3 ObjectNode

ObjectNode 允许把实现了 RAGObject 协议的 Pydantic 对象放进 RAG：

1. 用 obj.rag_key() 作为文本；
2. 保存 obj_json、obj_cls_name、obj_mod_name；
3. 检索后由 SimpleEngine 动态 import class 并重建对象；
4. 放到 node.metadata["obj"]。

### 10.4 SimpleEngine

SimpleEngine 支持三类构建方式：

~~~text
SimpleEngine.from_docs(...)
SimpleEngine.from_objs(...)
SimpleEngine.from_index(...)
~~~

from_docs()：

1. SimpleDirectoryReader 读取文件；
2. 修正文档 metadata；
3. 默认 SentenceSplitter；
4. 运行 transformations；
5. 创建 embedding、index、retriever、ranker 和 response synthesizer。

from_objs()：

1. 将对象转成 ObjectNode；
2. 采用同样的 index/retriever/ranker 流程。

SimpleEngine 还支持：

- asearch()/aretrieve()；
- retrieve()；
- add_docs()；
- add_objs()；
- persist()；
- 从持久化 index 恢复；
- 多 retriever 组成 SimpleHybridRetriever；
- PDF 通过 OmniParse 配置使用外部解析服务。

如果 retriever 不支持可修改或持久化，add/persist 会抛出类型错误。

### 10.5 RAG 的可选后端

源码包含：

- BM25；
- FAISS；
- Chroma；
- Milvus；
- Elasticsearch；
- 向量检索；
- 关键词检索；
- hybrid retriever；
- LLM/ColBERT/Cohere/BGE/object ranker。

部分后端依赖 setup.py 的 rag extra，不能假定基础安装后全部可用。

## 11. Strategy 子系统

### 11.1 ThoughtTree

strategy/base.py 定义 ThoughtNode 和 ThoughtTree：

- ThoughtNode 保存 name、value、id、valid_status；
- ThoughtTree 基于 anytree；
- update_node() 根据 LLM 输出的 node_id 和 node_state_instruction 添加子节点；
- parse_node_path() 返回从根到当前节点的路径；
- show() 输出树。

### 11.2 Planner

Planner 持有：

- Plan；
- working_memory；
- auto_run。

update_plan() 的源码流程：

1. 如果有新 goal，创建新 Plan；
2. 从 useful memories 构造上下文；
3. 调用 WritePlan；
4. 用 precheck_update_plan_from_rsp() 检查结果；
5. 调用 AskReview；
6. review 通过后更新 Plan；
7. 清空工作记忆。

process_task_result() 会：

- 请求任务结果评审；
- 通过后更新当前 Task 并推进；
- 用户要求 redo 时保留当前任务；
- 用户改变需求时重新 update_plan()。

get_useful_memories() 会把用户需求、上下文、任务列表和当前任务拼接成 Message，减少无关历史。

### 11.3 Solver

BaseSolver 需要一个 ActionGraph、SearchSpace、BaseLLM 和 Context。

当前只有 NaiveSolver 真正实现执行逻辑；它对 ActionGraph 做拓扑排序，然后逐节点调用 fill。

TOTSolver、DataInterpreterSolver、ReActSolver、IOSolver、COTSolver 都是未实现的扩展点。

### 11.4 Tree-of-Thought 求解器

strategy/tot.py 提供 ThoughtSolverBase、BFSSolver 和 DFSSolver 相关实现。

ThoughtSolverBase：

- 使用 parser.propose() 生成候选 thoughts；
- 让 LLM 输出 JSON/list；
- CodeParser.parse_code() 提取结果；
- ThoughtTree.update_node() 创建子节点；
- evaluator 对每个 node 打分；
- status_verify() 判断节点有效性；
- 通过 value 累计父节点得分。

BFSSolver：

- 从根节点开始；
- 每步并行扩展当前节点；
- 并行评估子节点；
- 按配置选择节点；
- 重复 max_steps；
- 取最高 value 节点路径。

部分 strategy 配置和 sample selection 仍是未实现或占位状态，不能把所有名称理解为已经接通的算法。

## 12. 基础工具和基础设施

utils 目录不属于一个单独算法，而是支撑整个工程的公共层。

### 12.1 文件和项目

包括：

- file.py、file_repository.py：文件读写和文件仓库；
- git_repository.py：Git 工作区；
- project_repo.py：项目级文件/代码访问；
- dependency_file.py：依赖文件；
- save_code.py：代码保存；
- repo_to_markdown.py：仓库转 Markdown；
- read_document.py：文档读取；
- json_to_markdown.py：JSON 转 Markdown。

### 12.2 解析

包括：

- pycst.py：Python CST/代码结构；
- parse_docstring.py：Google docstring；
- parse_html.py：HTML；
- omniparse_client.py：外部文档解析 client；
- custom_decoder.py；
- repair_llm_raw_output.py；
- sanitize.py；
- highlight.py；
- text.py；
- common.py 中的通用解析器。

### 12.3 图和可视化

包括：

- graph_repository.py；
- di_graph_repository.py；
- visual_graph_repo.py；
- tree.py；
- mermaid.py；
- mmdc_ink.py；
- mmdc_playwright.py；
- mmdc_pyppeteer.py。

### 12.4 网络和异步

包括：

- ahttp_client.py；
- async_helper.py；
- stream_pipe.py；
- redis.py；
- s3.py；
- recovery_util.py；
- human_interaction.py。

### 12.5 成本和 token

cost_manager.py 定义：

- Costs；
- CostManager；
- TokenCostManager；
- FireworksCostManager。

CostManager 根据 prompt_tokens、completion_tokens 和 model 的价格表累计 total_cost。BaseLLM 在拿到 API usage 后调用 update_cost()。

token_counter.py 提供不同模型的 token 价格表及 token 统计相关能力。

## 13. MaAS 动态工作流扩展

这一节只描述 maas.ext.maas 中当前代码实际执行的内容。

### 13.1 训练入口

examples/maas/optimize.py：

1. 解析 dataset、sample、round、batch_size、两个模型名、lr、is_test、is_textgrad；
2. 从 EXPERIMENT_CONFIGS 获取数据集配置；
3. 从 ModelsConfig 获取 opt_llm_config 和 exec_llm_config；
4. 创建 Optimizer；
5. is_test 为真时调用 optimizer.optimize("Test")；
6. 否则调用 optimizer.optimize("Graph")。

run_gsm8k_smoke.py 是一个更明确的 smoke test：

1. 固定 GSM8K；
2. 只取前 10 个样本；
3. 重复 2 次；
4. batch_size=4；
5. 直接创建 Workflow 和 GSM8KBenchmark；
6. 运行 evaluate_all_problems()；
7. 保存 CSV、results.json 和 checkpoint。

### 13.2 数据集配置

experiment_configs.py 中实际配置：

GSM8K：

~~~text
question_type = math
Generate
GenerateCoT
MultiGenerateCoT
ScEnsemble
Programmer
SelfRefine
EarlyStop
~~~

MATH：

~~~text
question_type = math
Generate
GenerateCoT
MultiGenerateCoT
ScEnsemble
Programmer
SelfRefine
EarlyStop
~~~

HumanEval：

~~~text
question_type = code
Generate
GenerateCoT
MultiGenerateCoT
ScEnsemble
Test
SelfRefine
EarlyStop
~~~

### 13.3 Optimizer

Optimizer.__init__() 创建：

- graph_utils；
- data_utils；
- experience_utils；
- evaluation_utils；
- device；
- MultiLayerController；
- Adam(controller.parameters())。

Optimizer 只把 controller 参数交给 Adam。算子内部的 LLM、prompt 和代码不进入 PyTorch optimizer。

optimize("Graph") 会：

1. 建立 asyncio event loop；
2. 调用 _optimize_graph_maas()；
3. 失败时进行一次 retry；
4. 输出 score。

_optimize_graph_maas() 会：

1. 从 train/results.json 读取历史结果；
2. 从 train/template/operator.json 加载 operator description/interface；
3. 调用 get_sentence_embedding() 得到 operator embeddings；
4. 创建 round 目录；
5. 动态导入 train.graph.Workflow；
6. 调用 EvaluationUtils.evaluate_graph_maas()。

opt_llm_config 在 Optimizer 中被保存，但当前主训练路径没有直接使用它。textgrad 分支会在 benchmark.py 中重新从 ModelsConfig.default() 取 gpt-4o-mini。

### 13.4 Controller

models/controller.py 的 OperatorSelector：

1. query embedding 经过 Linear(384, 32)；
2. operator embedding 经过 Linear(384, 32)；
3. 非第一层会把当前 operator embedding 和上一层 operator embedding 拼接后经过 Linear(768, 32)；
4. 两边 L2 normalize；
5. 点积生成 scores；
6. softmax 得到 probs；
7. log_softmax 得到 log_probs。

MultiLayerController 默认 4 层。每层调用 sample_operators(probs, threshold=0.3)。

sample_operators()：

1. 按当前 probs 做 multinomial；
2. 不放回；
3. 累计选中概率；
4. 达到 0.3 结束；
5. 没选中时回退 argmax。

每层保存所选算子的 log probability 之和。

第一层还有硬编码规则：

- 第一层出现 EarlyStop 时改成 Generate，并增加 -1.5 的 log probability 惩罚；
- 第一层没有 Generate 时强制改成 Generate；
- Generate 不是第一个时重排到第一个；
- 后续层出现 EarlyStop 时停止继续采样。

这里的 operator embedding 使用 SentenceTransformer all-MiniLM-L6-v2。SentenceEncoder 冻结模型参数。get_sentence_embedding() 每次都会重新加载 SentenceTransformer，是源码行为但效率较低。

### 13.5 GraphUtils 和动态 Workflow

GraphUtils.load_graph_maas() 把路径转换成 Python module path，动态 import graph 模块，并取其中的 Workflow 类。

Workflow 初始化参数统一为：

~~~text
name
llm_config
dataset
controller
operator_embeddings
~~~

Workflow 内部：

1. 用 create_llm_instance() 创建 LLM；
2. 创建 CostManager；
3. 创建需要的固定辅助 operator；
4. 创建 operator_mapping 中所有可选择 operator 的实例；
5. 保存 operator_names。

Workflow.__call__()：

1. controller.forward(problem, operator_embeddings, operator_names)；
2. 得到 log_probs_layers 和 selected_names_layers；
3. 初始化 current_solution 和 solutions；
4. 按 layer、按 operator name 分派；
5. 更新 current_solution、solutions；
6. 累加 log probability；
7. 执行最终合并/验证；
8. 返回 prediction、total_cost、sum_log_prob。

### 13.6 数学 operator

数学 template/operator.py 中的 Operator 基类只有 llm 和 name，并通过 _fill_node() 调用：

~~~text
ActionNode.from_pydantic(op_class).fill(...)
~~~

具体算子：

Generate：

- instruction + input；
- single_fill；
- 返回 GenerateOp.response。

GenerateCoT：

- 使用 GENERATE_COT_PROMPT；
- single_fill；
- 返回 GenerateOp.response。

MultiGenerateCoT：

- 相同 prompt 连续调用 3 次；
- 返回 response 列表。

ScEnsemble：

- 给 solutions 标记 A、B、C；
- 调用 ScEnsembleOp；
- 根据 solution_letter 返回候选列表中的一个；
- 如果 letter 非法，源码可能抛出索引错误。

SelfRefine：

- 用问题和当前 solution 构造 SELFREFINE_PROMPT；
- 返回改写后的 response。

Programmer：

1. 让 LLM 生成 solve()；
2. 在 ProcessPoolExecutor 中执行；
3. 禁止一组高风险 import；
4. 失败时把错误反馈给下一轮；
5. 最多循环 3 次；
6. 外层带 tenacity retry。

EarlyStop 类本身只返回 NotImplementedError。实际停止由 Controller 根据 operator name 判断，并没有真正调用 EarlyStop.__call__()。

### 13.7 GSM8K Workflow

GSM8K train/test graph.py 的主要分派：

~~~text
Generate / GenerateCoT
  -> 使用 MATH_SOLVE_PROMPT
  -> 产生文本 solution

SelfRefine
  -> 输入 problem 和 current_solution

Programmer
  -> 输入 problem 和 current_solution
  -> 代码执行
  -> 使用 Generate + REFINE_ANSWER_PROMPT 整理答案

ScEnsemble
  -> 输入 problem 和 solutions
  -> 清空并重建 solutions

MultiGenerateCoT
  -> 生成三个候选并加入 solutions
~~~

完成选中 operator 后：

- solutions 数量大于 1 时执行一次最终 ScEnsemble；
- 训练版用 Programmer 验证最终解；
- Programmer 没有生成代码时返回 final_solution；
- 测试版的空值判断与训练版略有区别。

### 13.8 MATH Workflow

MATH train/test graph.py 在 Controller 采样执行之前固定执行：

~~~text
Programmer(problem)
  -> code_solution
Generate(problem + Code output, REFINE_ANSWER_PROMPT)
  -> refined_solution
  -> solutions
~~~

之后再执行 controller 选择的 operator。

MATH 的 Generate 类算子使用 DETAILED_SOLUTION_PROMPT 或 GENERATE_SOLUTION_PROMPT。Programmer 结果也可能被加入 solutions。最后如果候选多于一个，执行 ScEnsemble，否则返回 current_solution。

注意：MATH 的 Programmer 前置路径不是 controller 动态选择出来的，而是 Workflow.__call__() 中的固定代码。

### 13.9 HumanEval Workflow

HumanEval 的 Workflow.__call__(problem, entry_point, log_path) 使用：

~~~text
Generate / GenerateCoT
  -> code_fill

SelfRefine
  -> code_fill

Test
  -> 执行公共测试
  -> 失败时让 LLM 反思并改写

MultiGenerateCoT
  -> 连续生成三个代码候选

ScEnsemble
  -> 选择一个代码候选
~~~

所有动态 operator 完成后，Workflow 还会固定调用 Test：

- Test 通过：返回测试后的 solution；
- Test 失败：调用 CustomCodeGenerate 和 IMPROVE_CODE_PROMPT 生成新代码。

Test.exec_code() 会从 humaneval_public_test.jsonl 或硬编码用例中读取测试。当前部分硬编码 entry point 的测试内容是空字符串，因此可能出现不执行实际测试却返回 no error 的行为。

### 13.10 Benchmark 训练器

BaseBenchmark 是三个数据集 benchmark 的公共基类，负责：

- 异步加载 JSON/JSONL；
- 限制并发；
- 调用 evaluate_problem()；
- 保存结果 CSV；
- 收集 score、cost、logprob；
- 计算 loss；
- 更新 optimizer；
- 保存 controller checkpoint；
- 记录错误样本。

训练时默认：

- max_concurrent_tasks=30；
- 每个问题 graph 调用 timeout=1500 秒；
- 单问题异常最多重试 20 次；
- batch_size 由 Optimizer 传入；
- repetitions 使用 CLI 的 sample。

每个样本返回：

~~~text
(input, prediction, expected_output, score, cost, logprob)
~~~

batch 内收集后，当前源码执行：

~~~text
cost_delta = cost - previous_cost
utility = score - 3 * cost_delta
loss = -(logprob * utility).mean()
loss.backward()
optimizer.step()
optimizer.zero_grad()
~~~

训练 graph 返回的 logprob 保留 Tensor 计算图；测试 graph 把 logprob 转成 float。

### 13.11 Benchmark 评测

Evaluator 根据 dataset 选择：

- GSM8KBenchmark；
- MATHBenchmark；
- HumanEvalBenchmark。

Evaluator._configure_graph() 把以下对象传入 Workflow：

- name；
- llm_config；
- dataset；
- controller；
- operator_embeddings。

训练调用 benchmark.run_evaluation(..., is_test=False)，测试调用 is_test=True。

### 13.12 数据集评分

GSM8K：

- 从 answer 抽取最后一个数字；
- 从 prediction 抽取最后一个数字；
- 误差不超过 1e-6 得分 1，否则 0。

MATH：

- 优先抽取最后一个 boxed 内容；
- 否则取最后一句；
- 支持字符串相等；
- 支持数字误差 1e-3；
- 支持 sympy/LaTeX 解析和符号等价。

HumanEval：

- sanitize 代码；
- exec solution；
- 找到 entry_point；
- 执行 test 中的 check(candidate)；
- 单个测试 run_with_timeout 15 秒；
- 通过得分 1，其他情况得分 0。

### 13.13 结果文件

训练目录下会产生：

~~~text
optimized/{dataset}/train/
├── results.json
├── round_{round}/
│   ├── {dataset}_controller_sample{sample}.pth
│   ├── <score>_<timestamp>.csv
│   └── log.json
└── template/
~~~

测试 checkpoint 从 train/round_{round} 读取，测试结果写入 test 目录。

DataUtils 负责：

- load_results；
- load_log；
- create_result_data；
- save_results；
- _load_scores。

ExperienceUtils 能读取 round_x/experience.json 并生成 processed_experience.json，但当前 Optimizer 主训练流程没有调用它。

ConvergenceUtils 能按 round 计算平均分和标准差，也能检查收敛，但当前 Optimizer 主循环没有接入自动收敛停止。

### 13.14 TextGrad 源码路径

benchmark.py 中有一个内嵌 prompt 改写逻辑：

1. textgrad 初始为 False；
2. 如果当前 repetition 平均分低于上一 repetition，textgrad 变为 True；
3. 下一 repetition 开始时随机读取 op_prompt.py 中一个 prompt；
4. 调用 gpt-4o-mini 生成新的 prompt；
5. 检查新旧占位符集合是否一致；
6. 更新 prompt 文件；
7. 将 is_textgrad 置 False。

scripts/textgrad/ 下还有更完整的独立脚本和 prompt archive，用于让 LLM 提议新 operator，但当前 Optimizer 主路径没有把它接成完整的自动改代码流程。

## 14. MaAS 的完整源码运行流程

### 14.1 训练模式

~~~text
optimize.py
  -> 解析 dataset/model/sample/batch_size/lr
  -> 加载 ModelsConfig
  -> 创建 Optimizer
  -> 初始化 Controller + Adam
  -> 加载 train graph
  -> 加载 operator.json
  -> 计算 operator embeddings
  -> 创建 Workflow
  -> BaseBenchmark.load_data()
  -> repetition loop
  -> batch loop
  -> 并发 evaluate_problem()
  -> Workflow.__call__()
  -> Controller.forward()
  -> operator LLM calls / code execution / tests
  -> dataset evaluator score
  -> collect cost and logprob
  -> compute utility and loss
  -> controller backward/update
  -> save CSV and checkpoint
~~~

### 14.2 测试模式

~~~text
optimize.py --is_test True
  -> 创建新 Controller
  -> 加载 test Workflow
  -> 从 train/round_x 加载 state_dict
  -> controller.eval()
  -> test data evaluate
  -> 不执行 backward/update
  -> 保存 test CSV 和 results.json
~~~

### 14.3 通用 Action 模式

通用框架中的 Action 调用流程是：

~~~text
Action.run()
  -> ActionNode.fill()
  -> compile prompt
  -> LLM.aask()
  -> parse raw/json/XML/code
  -> Pydantic output
  -> ActionOutput 或 Message
~~~

这个流程与 MaAS benchmark 中的 operator 调用相互复用：MaAS operator 不是直接调用 provider，而是使用 ActionNode.from_pydantic() 统一获得结构化输出。

## 15. 当前源码的实际边界和问题

下面内容不是论文推断，而是从当前源码调用关系可以直接看到的事实。

### 15.1 包名和入口不完全一致

setup.py 的项目名、描述和 console entry point 仍然使用 metagpt 命名，但源码 import 使用 maas。某些文件还从 maas.memory 导入 Memory，而当前目录列表中没有 maas/memory.py。环境中若没有额外包或文件，这些通用 Planner 路径可能无法直接导入。

### 15.2 不是所有策略都实现

NaiveSolver 可以执行；TOTSolver、DataInterpreterSolver、ReActSolver、IOSolver、COTSolver 是未实现接口。

EmbeddingToolRecommender.recall_tools() 未实现。

部分 RAG backend 和 ranker 需要额外依赖。

### 15.3 Controller 的 operator embedding 有性能问题

get_sentence_embedding() 每次调用都新建 SentenceTransformer。主训练初始化会对多个 operator 重复加载模型。语义上是对 operator 做编码，工程上应改成共享实例或缓存，但如果目标是严格复现源码行为，需要记录这个差异。

### 15.4 训练成本差分可能不代表真实成本

BaseBenchmark 使用跨 batch、跨 repetition 的 previous_cost：

~~~text
cost_delta = current_sample_cost - previous_cost
~~~

但每个 Workflow 有自己的 CostManager，异步结果的顺序也不表示全局调用顺序。因此 cost_delta 可能出现负值或不代表真正增量成本。

### 15.5 论文参数和代码参数不是一一对应

CLI 有 opt_model_name，但 Optimizer 的主训练路径不直接使用 opt_llm_config。

主 loss 中成本系数是硬编码 3，不是从 CLI 读取的独立 lambda。

question_type 会传入 Optimizer，但在主路径中没有参与控制器计算。

round 在当前一次 optimize 调用中不会形成完整的自动递增训练循环。

### 15.6 Experience 和 convergence 没有接入主训练

ExperienceUtils 和 ConvergenceUtils 被创建或单独实现，但当前主训练流程没有用经验来限制修改，也不会根据 convergence 自动停训。

### 15.7 HumanEval 的部分测试数据问题

scripts/utils.py 对若干 entry point 使用空字符串作为硬编码 test case。Test.exec_code() 迭代空字符串时可能不执行任何断言。

### 15.8 ActionNode XML 解析不适合不可信输出

xml_fill() 对 list 和 dict 使用 eval。源码能完成简单结构化解析，但在复现或生产化时必须考虑恶意输出和执行风险。

### 15.9 动态模块导入依赖当前工作目录

GraphUtils.load_graph_maas() 把相对路径转换为 module path 后直接 __import__。如果当前工作目录、PYTHONPATH 或 optimized 目录结构改变，Workflow 动态导入会失败。

## 16. 迁移到 MASFactory 时的源码等价映射

### 16.1 通用框架映射

| MaAS-main 源码 | MASFactory 复现职责 |
|---|---|
| Context | RuntimeContext / dependency container |
| Message | Edge message / state message |
| Action | 可调用的业务节点模板 |
| ActionNode | 结构化 LLM Node |
| ActionGraph | MASFactory Graph |
| BaseLLM | Model adapter interface |
| LLMProviderRegistry | Model provider registry |
| ToolRegistry | Tool registry |
| SimpleEngine | Retrieval subgraph |
| Planner | Planner/loop state |

### 16.2 MaAS 映射

| MaAS-main 源码 | MASFactory 复现职责 |
|---|---|
| Optimizer | Train/Test runner |
| MultiLayerController | Controller node |
| sample_operators | Sampling policy |
| Workflow.__call__ | Architecture execution graph |
| operator_mapping | Operator registry |
| current_solution | 当前解状态 |
| solutions | 候选解集合 |
| CostManager | Cost accounting |
| BaseBenchmark | Dataset evaluator |
| score | Environment reward |
| logprob | Policy gradient path |
| results.json/CSV/pth | Artifact/checkpoint store |

### 16.3 推荐的最小 MASFactory 状态

~~~python
state = {
    "query": str,
    "dataset": str,
    "entry_point": str | None,
    "selected_names_layers": list[list[str]],
    "layer_log_probs": list,
    "current_solution": str,
    "solutions": list[str],
    "prediction": str,
    "score": float,
    "cost": float,
    "architecture_logprob": object,
    "failure": dict | None,
}
~~~

状态约束：

1. Controller 只负责选择，不直接隐藏执行 LLM；
2. Operator 只返回自己的输出和错误；
3. architecture_logprob 在训练模式必须保持计算图；
4. score、cost 在 evaluator 产生；
5. EarlyStop 只结束当前样本的 layer loop；
6. train/test Workflow 需要分别保留；
7. MATH 固定的 Programmer 前置步骤不能被误删；
8. GSM8K 最后的 Programmer 验证不能被误认为普通采样算子；
9. HumanEval 的 Test 和最终修复路径必须单独保留。

## 17. 另一个 AI 的推荐阅读顺序

如果需要理解整个项目而不是只改一个函数，推荐按以下顺序读取：

1. README.md：项目概览和 MaAS 入口；
2. setup.py、requirements.txt：安装名、包发现、依赖和可选能力；
3. config2.py、configs/llm_config.py、configs/models_config.py：配置来源；
4. context.py、schema.py：运行上下文和消息/任务契约；
5. provider/base_llm.py、provider/llm_provider_registry.py：LLM 调用边界；
6. actions/action.py、actions/action_node.py：结构化 Action 执行；
7. tools/tool_registry.py、tool_recommend.py：工具注册和推荐；
8. rag/schema.py、rag/factories/*、rag/engines/simple.py：RAG；
9. strategy/planner.py、strategy/solver.py、strategy/tot.py：策略层；
10. examples/maas/optimize.py：MaAS 实验入口；
11. ext/maas/scripts/optimizer.py：MaAS orchestration；
12. ext/maas/models/controller.py、models/utils.py：选择器；
13. ext/maas/benchmark/benchmark.py：训练更新；
14. ext/maas/benchmark/gsm8k.py、math.py、humaneval.py：任务评分；
15. ext/maas/scripts/optimized/*/train/graph.py：三类工作流；
16. ext/maas/scripts/optimized/*/template/operator.py：实际算子；
17. ext/maas/scripts/textgrad/*：可选 prompt 演化；
18. utils/*：遇到具体基础设施问题时按需深入。

## 18. 源码事实版总结

MaAS-main 当前源码本质上由一个通用 LLM agent 框架和一个 MaAS benchmark 扩展组成。

通用框架通过 Context 管理配置、LLM、项目仓库和成本；通过 Message、Task、Plan 管理任务状态；通过 Action 和 ActionNode 把自然语言任务转成结构化的 LLM 输出；通过 Provider Registry 接入多个 LLM；通过 ToolRegistry 和 RAG Engine 扩展工具与知识检索；通过 Strategy 提供 Planner、Tree-of-Thought 和 Solver 接口。

MaAS 扩展在这个框架上增加了一个查询条件的 controller。它把问题和 operator 描述编码成向量，按层随机选择 operator，然后让对应 Workflow 执行 LLM、代码、测试和 ensemble。Benchmark 根据最终输出计算离散分数，Workflow 返回的 log probability 被用于更新 controller；LLM 本身不被 PyTorch optimizer 直接训练。

当前源码不是一个完全闭环、所有论文能力都实现的系统。已接通的主路径是：Controller sampling、三类数据集 Workflow、异步 benchmark、基于 score/cost/logprob 的 controller 更新、CSV/checkpoint 保存。Prompt textgrad、经验管理、收敛停止、自动新增/合并 operator、若干通用 Solver 和部分工具/RAG 扩展仍是可选、未接通或部分实现状态。

因此 MASFactory 复现时，第一优先级应是保持上述实际调用链、返回值和状态传递不变；第二优先级才是把未接通的扩展能力补齐。
