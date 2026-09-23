# MaAS Reproduction 关键代码片段参考

**用途**: 快速查找和理解关键函数的实现细节

---

## 📄 Table of Contents

1. [配置与初始化](#配置与初始化)
2. [节点实现](#节点实现)
3. [评分器](#评分器)
4. [控制器与采样](#控制器与采样)
5. [运行时管理](#运行时管理)

---

## 配置与初始化

### 1. workflow.py - RootGraph 定义

```python
def build_maas_reproduction_graph(name: str = "MaASReproduction") -> RootGraph:
    """构建MaAS再现应用的RootGraph"""
    graph = RootGraph(name=name)
    
    # 节点1: 配置
    config_node = graph.create_node(
        CustomNode,
        "ConfigNode",
        forward=config_forward,
        push_keys={"settings": "MaAS runtime settings."},
    )
    
    # 节点2: 训练循环（核心优化）
    training_loop = graph.create_node(
        Loop,
        "TrainingLoop",
        max_iterations=100000,  # 最多100k次问题处理
        terminate_condition_function=training_controller,
        pull_keys=None,
        push_keys=TRAINING_LOOP_PUSH_KEYS,
    )
    
    # 在训练循环内附加3个节点
    attach_training_loop_body(training_loop)
    
    # 节点3: 结果汇总
    result_node = graph.create_node(
        CustomNode,
        "ResultNode",
        forward=result_forward,
    )

    # 连接关系
    graph.edge_from_entry(config_node, ENTRY_TO_CONFIG_KEYS)
    graph.create_edge(config_node, training_loop, CONFIG_TO_TRAINING_KEYS)
    graph.create_edge(training_loop, result_node, TRAINING_LOOP_PUSH_KEYS)
    graph.edge_to_exit(result_node, TRAINING_LOOP_PUSH_KEYS)

    logger.info("Built MaAS reproduction RootGraph wiring")
    return graph
```

### 2. config/settings.py - 数据类定义

```python
@dataclass(frozen=True)
class OptimizerSettings:
    """控制器优化参数"""
    sample: int  # 采样轮数
    round_number: int  # MaAS优化轮数
    batch_size: int  # 梯度更新批次
    learning_rate: float  # Adam学习率
    is_textgrad: bool  # TextGrad启用标志
    opt_model_name: str  # 优化模型（生成提示词）
    exec_model_name: str  # 执行模型（架构调用）

@dataclass(frozen=True)
class MaASPaths:
    """文件系统位置"""
    application_root: Path
    data_root: Path  # 数据集目录
    optimized_root: Path  # 优化架构根目录
    checkpoint_root: Path  # 检查点目录
    runs_root: Path  # 运行输出目录
    
    @classmethod
    def from_environment(cls, application_root: Path) -> "MaASPaths":
        """从环境变量解析路径覆盖"""
        defaults = cls.from_application_root(application_root)
        return cls(
            application_root=defaults.application_root,
            data_root=_resolve_path_override("MAAS_DATA_ROOT", defaults.data_root),
            optimized_root=_resolve_path_override("MAAS_OPTIMIZED_ROOT", defaults.optimized_root),
            checkpoint_root=_resolve_path_override("MAAS_CHECKPOINT_ROOT", defaults.checkpoint_root),
            runs_root=_resolve_path_override("MAAS_RUNS_ROOT", defaults.runs_root),
        )
    
    def dataset_file(self, dataset: DatasetName, split: DatasetSplit) -> Path:
        """返回数据集文件路径"""
        jsonl_path = self.data_root / f"{dataset.lower()}_{split}.jsonl"
        if jsonl_path.exists():
            return jsonl_path
        json_path = self.data_root / f"{dataset.lower()}_{split}.json"
        return json_path
    
    def controller_checkpoint(self, dataset: str, round_num: int, sample: int) -> Path:
        """返回控制器检查点路径"""
        return self.checkpoint_root / dataset / f"round_{round_num}_sample_{sample}" / "controller.pth"

@dataclass(frozen=True)
class MaASRuntimeSettings:
    """运行时配置集合"""
    dataset: DatasetName
    mode: ExecutionMode  # "Graph" or "Test"
    paths: MaASPaths
    optimizer: OptimizerSettings
    opt_llm_config: object  # LLM配置对象
    exec_llm_config: object
    
    @classmethod
    def from_experiment(cls, dataset, mode, paths, optimizer, opt_llm_config, exec_llm_config):
        """从实验配置创建运行时设置"""
        return cls(
            dataset=dataset,
            mode=mode,
            paths=paths,
            optimizer=optimizer,
            opt_llm_config=opt_llm_config,
            exec_llm_config=exec_llm_config,
        )
    
    @property
    def dataset_file(self) -> Path:
        """当前数据集的文件路径"""
        split = "test" if self.mode == "Test" else "train"
        return self.paths.dataset_file(self.dataset, split)
    
    @property
    def run_directory(self) -> Path:
        """运行输出目录"""
        return self.paths.run_root(self.dataset, self.mode)
```

### 3. config/experiments.py - 实验定义

```python
EXPERIMENT_CONFIGS: dict[str, ExperimentConfig] = {
    "MATH": ExperimentConfig(
        dataset="MATH",
        question_type="math",
        operators=(
            "Generate",          # 基础生成
            "GenerateCoT",        # 思维链生成
            "MultiGenerateCoT",   # 多个思维链
            "ScEnsemble",         # 自一致性集成
            "Programmer",         # 代码编写辅助
            "SelfRefine",         # 自我改进
            "EarlyStop",          # 提前停止（会被覆盖）
        ),
    ),
    "GSM8K": ExperimentConfig(
        dataset="GSM8K",
        question_type="math",
        operators=(
            "Generate", "GenerateCoT", "MultiGenerateCoT",
            "ScEnsemble", "Programmer", "SelfRefine", "EarlyStop",
        ),
    ),
    "HumanEval": ExperimentConfig(
        dataset="HumanEval",
        question_type="code",
        operators=(
            "Generate", "GenerateCoT", "MultiGenerateCoT",
            "ScEnsemble", "Test", "SelfRefine", "EarlyStop",
        ),
    ),
}
```

---

## 节点实现

### 4. nodes/config_node.py - 参数解析与配置

```python
def config_forward(input_data: dict[str, object], attributes: dict[str, object]) -> dict[str, object]:
    """ConfigNode的前向函数：将CLI参数转换为MaAS配置"""
    
    # 解析CLI输入
    application_root = Path(input_data["application_root"]).expanduser().resolve()
    dataset = str(input_data["dataset"])
    mode = str(input_data.get("mode", "Graph"))
    opt_model_name = str(input_data.get("opt_model_name", "gpt-4o-mini"))
    exec_model_name = str(input_data.get("exec_model_name", "gpt-4o-mini"))

    # 创建优化器设置
    optimizer = OptimizerSettings(
        sample=int(input_data.get("sample", 4)),
        round_number=int(input_data.get("round_number", 1)),
        batch_size=int(input_data.get("batch_size", 4)),
        learning_rate=float(input_data.get("learning_rate", 0.01)),
        is_textgrad=bool(input_data.get("is_textgrad", False)),
        opt_model_name=opt_model_name,
        exec_model_name=exec_model_name,
    )
    
    # 解析LLM配置
    opt_llm_config, exec_llm_config = resolve_model_configs(opt_model_name, exec_model_name)
    
    # 创建运行时设置
    settings = MaASRuntimeSettings.from_experiment(
        dataset=dataset,
        mode=mode,
        paths=MaASPaths.from_environment(application_root),
        optimizer=optimizer,
        opt_llm_config=opt_llm_config,
        exec_llm_config=exec_llm_config,
    )

    logger.info(
        "Configured MaAS reproduction: dataset=%s mode=%s round=%s sample=%s batch_size=%s",
        settings.dataset,
        settings.mode,
        settings.optimizer.round_number,
        settings.optimizer.sample,
        settings.optimizer.batch_size,
    )
    
    return {
        "settings": settings,
        "dataset": settings.dataset,
        "mode": settings.mode,
        "sample": settings.optimizer.sample,
        "batch_size": settings.optimizer.batch_size,
        "round": settings.optimizer.round_number,
        "model_config": {
            "opt_model_name": settings.optimizer.opt_model_name,
            "exec_model_name": settings.optimizer.exec_model_name,
        },
        "paths": {
            "dataset_file": str(settings.dataset_file),
            "run_directory": str(settings.run_directory),
            "checkpoint_path": str(
                settings.paths.controller_checkpoint(
                    settings.dataset,
                    settings.optimizer.round_number,
                    settings.optimizer.sample,
                )
            ),
        },
    }
```

### 5. nodes/architecture_exec_node.py - MaAS工作流执行

```python
_MAAS_WORKFLOW_TIMEOUT_SECONDS = 200
_MAAS_WORKFLOW_MAX_RETRIES = 2

async def _call_workflow(workflow, *args):
    """使用原始基准测试的超时调用MaAS工作流"""
    return await asyncio.wait_for(workflow(*args), timeout=_MAAS_WORKFLOW_TIMEOUT_SECONDS)

@retry(
    stop=stop_after_attempt(_MAAS_WORKFLOW_MAX_RETRIES),
    wait=wait_fixed(1),
    retry=retry_if_exception(lambda exc: not isinstance(exc, AsyncRunnerContextError)),
    reraise=True,
)
def _run_workflow_with_retry(workflow, *args):
    """使用原始基准测试的重试策略运行MaAS工作流"""
    return run_async_once(_call_workflow(workflow, *args))

def architecture_exec_forward(
    input_data: dict[str, object],
    attributes: dict[str, object],
) -> dict[str, object]:
    """执行当前问题通过活跃的MaAS架构工作流"""
    settings = attributes["settings"]
    workflow = attributes["architecture_workflow"]
    problem = str(input_data["problem"])
    entry_point = str(input_data["entry_point"])

    try:
        # HumanEval需要额外参数：entry_point和run_directory
        if settings.dataset == "HumanEval":
            result = _run_workflow_with_retry(
                workflow,
                problem,
                entry_point,
                str(settings.run_directory),
            )
        else:
            # GSM8K和MATH只需要problem
            result = _run_workflow_with_retry(workflow, problem)
        
        # 解包结果元组: (prediction, cost, logprob)
        prediction, cost, logprob = result
    
    except AsyncRunnerContextError:
        raise  # 同步上下文错误直接传播
    except Exception as exc:
        logger.exception(
            "Architecture execution failed for problem index %s",
            input_data["problem_index"],
        )
        # 异常处理：返回错误消息作为预测
        prediction, cost, logprob = str(exc), 0.0, 0.0

    return {
        "problem": problem,
        "entry_point": entry_point,
        "expected_answer": input_data["expected_answer"],
        "prediction": prediction,
        "cost": cost,
        "logprob": logprob,
        "problem_index": input_data["problem_index"],
    }
```

### 6. nodes/evaluator_node.py - 结果评分

```python
def evaluator_forward(
    input_data: dict[str, object],
    attributes: dict[str, object],
) -> dict[str, object]:
    """评估MaAS输出与预期答案"""
    settings = attributes["settings"]
    dataset = settings.dataset
    prediction = str(input_data["prediction"])
    expected_answer = str(input_data["expected_answer"])

    # 选择数据集特定的评分器
    if dataset == "GSM8K":
        scorer = GSM8KScorer()
        expected_number = scorer.extract_number(expected_answer)
        score, extracted_output = scorer.calculate_score(expected_number, prediction)
        expected_output = expected_number
        columns = ["question", "prediction", "expected_output", "score", "cost", "logprob"]
    
    elif dataset == "MATH":
        scorer = MATHScorer()
        score, extracted_output = scorer.calculate_score(expected_answer, prediction)
        expected_output = expected_answer
        columns = ["question", "prediction", "expected_output", "score", "cost", "logprob"]
    
    elif dataset == "HumanEval":
        scorer = HumanEvalScorer(log_path=settings.run_directory)
        result = scorer.check_solution(prediction, expected_answer, str(input_data["entry_point"]))
        score = 1.0 if result[0] == scorer.PASS else 0.0
        extracted_output = score
        expected_output = result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else expected_answer
        columns = ["inputs", "prediction", "expected_output", "score", "cost", "logprob"]
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # 记录失败案例
    if score == 0:
        append_mismatch_log(
            author_round_directory(settings),
            str(input_data["problem"]),
            expected_output,
            prediction,
            extracted_output,
            extract_answer_code=_extract_answer_code(dataset),
        )

    # 累积结果行（用于后续CSV写入）
    result_row = [
        str(input_data["problem"]),
        prediction,
        expected_output,
        score,
        input_data["cost"],
        input_data["logprob"],
    ]
    attributes.setdefault("result_columns", columns)
    attributes.setdefault("sample_results", []).append(result_row)

    logger.info("MaAS evaluator scored problem %s on %s as %s", 
                input_data["problem_index"], dataset, score)
    
    return {
        "score": score,
        "cost": input_data["cost"],
        "logprob": input_data["logprob"],
        "problem_index": input_data["problem_index"],
        "result_columns": columns,
        "result_row": result_row,
    }
```

### 7. nodes/loss_update_node.py - 控制器更新

```python
def loss_update_forward(
    input_data: dict[str, object],
    attributes: dict[str, object],
) -> dict[str, object]:
    """累积损失并在批次满时更新控制器"""
    
    score = float(input_data["score"])
    cost = float(input_data["cost"])
    logprob = input_data["logprob"]
    
    # 计算成本增量
    previous_cost = float(attributes.get("previous_cost", 0.0))
    cost_delta = cost - previous_cost
    attributes["previous_cost"] = cost

    # 累积批次数据
    attributes.setdefault("batch_logprobs", []).append(_as_logprob_tensor(logprob, attributes))
    attributes.setdefault("batch_scores", []).append(score)
    attributes.setdefault("batch_costs", []).append(cost_delta)
    attributes.setdefault("all_scores", []).append(score)
    attributes.setdefault("current_repetition_scores", []).append(score)

    loss_value = None
    update_performed = False
    batch_size = int(attributes["batch_size"])
    settings = attributes["settings"]

    # 当批次满时执行梯度更新
    if len(attributes["batch_logprobs"]) >= batch_size:
        if getattr(settings, "mode", "Graph") == "Graph":
            # 计算损失：-(logprob * utility).mean()
            # 其中 utility = score - 3 * cost
            loss = _compute_loss(attributes)
            loss_value = float(loss.detach().cpu().item())
            
            if loss.requires_grad:
                # 标准的梯度更新流程
                loss.backward()
                attributes["optimizer"].step()
                attributes["optimizer"].zero_grad()
                update_performed = True
                logger.info("MaAS controller updated at problem %s with loss %.6f", 
                           input_data["problem_index"], loss_value)
            else:
                logger.info("MaAS batch loss at problem %s has no gradient and update was skipped", 
                           input_data["problem_index"])

        # 清空批次缓冲
        attributes["batch_logprobs"].clear()
        attributes["batch_scores"].clear()
        attributes["batch_costs"].clear()

    return {
        "result_score": score,
        "result_cost": cost,
        "result_logprob": logprob,
        "result_loss": loss_value,
        "result_update_performed": update_performed,
        "result_problem_index": input_data["problem_index"],
    }

def _compute_loss(attributes: dict[str, object]) -> torch.Tensor:
    """计算强化学习损失函数"""
    device = _resolve_device(attributes)
    logprobs = torch.stack(attributes["batch_logprobs"]).to(device)
    scores = torch.tensor(attributes["batch_scores"], dtype=torch.float32, device=device)
    costs = torch.tensor(attributes["batch_costs"], dtype=torch.float32, device=device)
    
    # 效用 = 分数 - 成本权重*成本
    utility = scores - 3 * costs
    
    # 负最大似然估计
    loss = -(logprobs * utility).mean()
    return loss
```

### 8. nodes/training_controller.py - 循环控制

```python
def training_controller(input_data: dict[str, object], attributes: dict[str, object]) -> bool:
    """Loop循环的终止条件函数（每次迭代调用）"""
    
    # 清除上一轮的result_*键，为下一轮重用做准备
    for key in list(input_data.keys()):
        if key.startswith("result_"):
            del input_data[key]

    settings = attributes["settings"]
    problems = attributes["problems"]
    problem_index = int(attributes.get("problem_index", 0))
    repetition = int(attributes.get("repetition", 1))
    max_repetitions = int(settings.optimizer.sample)

    # 检查是否已处理完所有问题
    if problem_index >= len(problems):
        # 轮次结束处理
        from maas_reproduction.nodes.loss_update_node import flush_remaining_batch
        
        flush_remaining_batch(attributes)  # 强制优化最后部分批次
        current_repetition_score = _finish_repetition(attributes)
        
        # 检查是否已完成所有采样轮次
        if repetition >= max_repetitions:
            _write_final_result(input_data, attributes)  # 保存检查点和结果
            logger.info("MaAS training loop finished with average score %.5f", 
                       input_data["average_score"])
            return True  # 停止循环
        
        # 可选：TextGrad元优化
        _maybe_run_textgrad(settings, attributes, current_repetition_score)
        
        # 开始下一个采样轮次
        repetition += 1
        problem_index = 0
        attributes["repetition"] = repetition
        logger.info("MaAS training loop starting repetition %s/%s", repetition, max_repetitions)

    # 为当前问题设置输入数据
    problem = problems[problem_index]
    _write_problem_message(input_data, settings.dataset, problem, problem_index)
    attributes["problem_index"] = problem_index + 1
    attributes["repetition"] = repetition
    
    return False  # 继续处理下一个问题

def _write_problem_message(
    input_data: dict[str, object],
    dataset: str,
    problem: dict[str, object],
    problem_index: int,
) -> None:
    """从数据集问题对象中提取输入字段"""
    if dataset == "GSM8K":
        input_data["problem"] = problem["question"]
        input_data["entry_point"] = ""
        input_data["expected_answer"] = problem["answer"]
    elif dataset == "MATH":
        input_data["problem"] = problem["problem"]
        input_data["entry_point"] = ""
        input_data["expected_answer"] = problem["solution"]
    elif dataset == "HumanEval":
        input_data["problem"] = problem["prompt"]
        input_data["entry_point"] = problem["entry_point"]
        input_data["expected_answer"] = problem["test"]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    input_data["problem_index"] = problem_index

def _finish_repetition(attributes: dict[str, object]) -> float:
    """完成一个采样轮次：计算统计信息"""
    current_repetition_scores = attributes.get("current_repetition_scores", [])
    current_repetition_score = sum(current_repetition_scores) / len(current_repetition_scores) \
                                if current_repetition_scores else 0.0
    
    # 追加轮次统计到JSON文件
    append_round_summary(...)
    
    attributes["current_repetition_scores"] = []  # 重置
    return current_repetition_score

def _write_final_result(input_data: dict[str, object], attributes: dict[str, object]) -> None:
    """写入最终结果：检查点和CSV"""
    settings = attributes["settings"]
    all_scores = attributes.get("all_scores", [])
    sample_results = list(attributes.get("sample_results", []))
    
    average_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    # 保存控制器检查点
    checkpoint_path = settings.paths.controller_checkpoint(
        settings.dataset,
        settings.optimizer.round_number,
        settings.optimizer.sample,
    )
    if getattr(settings, "mode", "Graph") == "Graph":
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(attributes["controller"].state_dict(), checkpoint_path)
        logger.info("Saved MaAS controller parameters to %s", checkpoint_path)

    # 写入CSV结果
    artifact_directory = author_round_directory(settings)
    csv_path = write_results_csv(
        artifact_directory,
        attributes.get("result_columns"),
        sample_results,
        average_score,
    )
    
    # 更新input_data以传递到结果节点
    input_data["average_score"] = average_score
    input_data["checkpoint_path"] = str(checkpoint_path)
    input_data["result_path"] = str(artifact_directory)
    input_data["runtime_metadata"] = {
        "total_problems": len(sample_results),
        "average_score": average_score,
        "dataset": settings.dataset,
    }
```

---

## 评分器

### 9. benchmarks/gsm8k.py - 数字提取评分

```python
class GSM8KScorer:
    """GSM8K数学推理的评分器"""
    
    def extract_number(self, text: str) -> float | None:
        """从文本中提取最后一个数字"""
        matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
        if matches:
            last_number = matches[-1].replace(",", "")
            try:
                return float(last_number)
            except ValueError:
                return None
        return None

    def calculate_score(self, expected_output: float, prediction: str | float | None) -> tuple[float, float | None]:
        """计算分数：精确数值匹配"""
        predicted_number = self.extract_number(str(prediction)) if prediction is not None else None
        if predicted_number is None:
            return 0.0, predicted_number
        # 数值相差≤1e-6则认为相同
        return 1.0 if abs(expected_output - predicted_number) <= 1e-6 else 0.0, predicted_number
```

### 10. benchmarks/math.py - 符号化匹配评分

```python
class MATHScorer:
    """MATH数据集的评分器（符号化数学等价）"""
    
    def extract_model_answer(self, text: str) -> str:
        """从文本中提取答案"""
        # 优先从 \boxed{} 中提取
        pattern = r"\\boxed{((?:[^{}]|{[^{}]*})*)\}"
        boxed_matches = re.findall(pattern, text, re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1].strip()

        # 其次从最后一句提取
        sentence_end_pattern = r"(?<!\\d)[.!?]\\s+"
        sentences = re.split(sentence_end_pattern, text)
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        return sentences[-1] if sentences else ""

    def calculate_score(self, expected_output: str, prediction: str) -> tuple[int, str]:
        """计算分数：符号化等价性"""
        expected_answer = self.extract_model_answer(expected_output)
        predicted_answer = self.extract_model_answer(prediction)

        if self.math_equal(predicted_answer, expected_answer):
            return 1, predicted_answer
        return 0, predicted_answer

    def math_equal(self, prediction, reference) -> bool:
        """检查数学等价性：字符串→数值→符号"""
        # 1. 字符串相等
        if str(prediction) == str(reference):
            return True

        # 2. 数值比较
        try:
            if self.is_digit(prediction) and self.is_digit(reference):
                prediction = self.parse_digits(prediction)
                reference = self.parse_digits(reference)
                return isclose(prediction, reference, abs_tol=1e-3)
        except Exception:
            pass

        # 3. 符号化比较（使用sympy）
        try:
            return self.symbolic_equal(prediction, reference)
        except Exception:
            pass

        return False

    def symbolic_equal(self, a, b) -> bool:
        """使用sympy进行符号化等价性检查"""
        from sympy import N, simplify
        from sympy.parsing.latex import parse_latex
        from sympy.parsing.sympy_parser import parse_expr

        def _parse(s):
            for parser in [parse_latex, parse_expr]:
                try:
                    return parser(s)
                except Exception:
                    pass
            return s

        a = _parse(a)
        b = _parse(b)

        # 简化后相等
        try:
            if simplify(a - b) == 0:
                return True
        except Exception:
            pass

        # 数值评估后接近
        try:
            if isclose(N(a), N(b), abs_tol=1e-3):
                return True
        except Exception:
            pass
        
        return False
```

### 11. benchmarks/humaneval.py - 代码执行评分

```python
class HumanEvalScorer:
    PASS = "PASS"
    FAIL = "FAIL"

    def check_solution(self, solution, test, entry_point) -> tuple[str, str]:
        """执行代码解决方案并检查测试"""
        solution = self._with_special_case_helpers(solution, entry_point)
        try:
            global_dict = {
                "math": __import__("math"),
                "hashlib": __import__("hashlib"),
                "re": __import__("re"),
                "List": List,
                "Dict": Dict,
                "Tuple": Tuple,
                "Optional": Optional,
                "Any": Any,
            }

            # 执行解决方案代码
            exec(solution, global_dict)

            if entry_point not in global_dict:
                raise ValueError(f"Function {entry_point} is not defined in the solution.")

            # 执行测试代码
            exec(test, global_dict)

            # 调用check函数验证解决方案
            check = global_dict["check"]
            result = self.run_with_timeout(check, (global_dict[entry_point],), 15)

            if result is None:
                result = (self.PASS, "The solution passed all test cases.")

        except self.TimeoutError:
            result = (
                self.FAIL,
                "Execution timed out. Please check if your solution contains infinite loops or overly time-consuming operations.",
            )
        except Exception as exc:
            error_message = f"Error: {str(exc)}.\\n Solution: {solution}.\\n Test: {test}"
            result = (self.FAIL, error_message)

        return result

    def run_with_timeout(self, func, args, timeout) -> object:
        """在超时内运行函数（线程式）"""
        result = []
        stop_event = threading.Event()

        def target():
            try:
                result.append(func(*args))
            except Exception as exc:
                result.append(exc)
            finally:
                stop_event.set()

        thread = threading.Thread(target=target)
        thread.start()
        is_timeout = not stop_event.wait(timeout)

        if is_timeout:
            raise self.TimeoutError("Function execution timed out")

        if not result:
            return None
        if isinstance(result[0], Exception):
            raise result[0]
        return result[0]
```

---

## 控制器与采样

### 12. models/controller.py - 多层架构选择器

```python
class OperatorSelector(torch.nn.Module):
    """单层操作符选择器"""
    
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 32,
        device=None,
        is_first_layer: bool = False,
    ):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_first_layer = is_first_layer
        
        # 基于层位置选择编码器结构
        if self.is_first_layer:
            # 首层：只处理问题和操作符嵌入
            self.operator_encoder = torch.nn.Linear(input_dim, hidden_dim)
        else:
            # 后层：将问题和前一层选择的操作符连接
            self.operator_encoder = torch.nn.Linear(input_dim * 2, hidden_dim)
        
        # 问题编码器（所有层相同）
        self.query_encoder = torch.nn.Linear(input_dim, hidden_dim)

    def forward(
        self,
        query_embed: torch.Tensor,        # 问题嵌入 (batch_size, 384)
        operators_embed: torch.Tensor,    # 所有操作符嵌入 (num_ops, 384)
        prev_operators_embed: torch.Tensor = None,  # 前一层选择 (selected_ops, 384)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回: (log_probs, probs)"""
        
        # 确保query_embed是2D
        if query_embed.dim() == 1:
            query_embed = query_embed.unsqueeze(0)

        # 编码问题
        query_embed = self.query_encoder(query_embed)  # (batch_size, hidden_dim)
        query_embed = F.normalize(query_embed, p=2, dim=1)

        # 编码操作符
        if prev_operators_embed is not None and self.is_first_layer is False:
            # 后层：与前一层连接
            prev_operator = prev_operators_embed[0].unsqueeze(0)
            prev_expanded = prev_operator.expand(operators_embed.size(0), -1)
            concat_embed = torch.cat([operators_embed, prev_expanded], dim=1)
            all_operators_embed = self.operator_encoder(concat_embed)
        else:
            # 首层或无前一层
            all_operators_embed = self.operator_encoder(operators_embed)

        all_operators_embed = F.normalize(all_operators_embed, p=2, dim=1)

        # 计算相似度
        scores = torch.matmul(query_embed, all_operators_embed.T)

        # 转换为概率
        probs = F.softmax(scores, dim=1)
        log_probs = F.log_softmax(scores, dim=1)

        return log_probs, probs


class MultiLayerController(torch.nn.Module):
    """4层操作符选择控制器"""
    
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 32,
        num_layers: int = 4,
        device=None,
    ):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 创建4层
        self.layers = torch.nn.ModuleList(
            [
                OperatorSelector(input_dim, hidden_dim, device=self.device, is_first_layer=(i == 0))
                for i in range(num_layers)
            ]
        )

    def forward(
        self,
        query: str,                      # 问题文本
        operators_embedding: torch.Tensor,  # 所有操作符嵌入
        selection_operator_names: list[str],  # 操作符名称列表
        log_path=None,
    ) -> list[str]:
        """
        返回: 最后一层选择的操作符名称列表
        """
        # 编码问题文本
        query_embedding = sentence_encoder(query).to(self.device)  # (384,)
        operators_embedding = operators_embedding.to(self.device)
        
        prev_operators = None

        for layer_idx, layer in enumerate(self.layers):
            # 前向传播
            if layer_idx == 0:
                log_probs, probs = layer(query_embedding, operators_embedding)
            else:
                log_probs, probs = layer(query_embedding, operators_embedding, prev_operators)

            probs_1d = probs.squeeze(0)
            
            # 采样操作符
            selected_indices = sample_operators(probs_1d, threshold=0.3)
            selected_names = [selection_operator_names[idx] for idx in selected_indices.cpu().tolist()]

            # === 特殊规则（首层） ===
            if layer_idx == 0:
                # 规则1: 若选了EarlyStop，强制改为Generate
                if any(name.lower() == "earlystop" for name in selected_names):
                    try:
                        generate_idx = selection_operator_names.index("Generate")
                    except ValueError:
                        generate_idx = 0
                    selected_indices = torch.tensor([generate_idx], device=self.device)
                    selected_names = ["Generate"]
                
                # 规则2: 若未选任何generate，强制选Generate
                elif not any("generate" in name.lower() for name in selected_names):
                    try:
                        generate_idx = selection_operator_names.index("Generate")
                    except ValueError:
                        generate_idx = 0
                    selected_indices = torch.tensor([generate_idx], device=self.device)
                    selected_names = ["Generate"]
            
            # === 记录后层选择 ===
            prev_operators = operators_embedding[selected_indices]

        return selected_names
```

### 13. models/utils.py - 嵌入与采样

```python
class SentenceEncoder(torch.nn.Module):
    """使用sentence-transformers进行文本嵌编码"""
    
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # 冻结权重（仅用于特征提取）
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, sentence: str) -> torch.Tensor:
        """编码单个句子为384维张量"""
        embeddings = self.model.encode(sentence)
        return torch.tensor(embeddings)

def sample_operators(probs: torch.Tensor, threshold: float = 0.25) -> torch.Tensor:
    """
    采样操作符直到累积概率≥阈值
    
    Args:
        probs: 操作符概率分布 (num_ops,)
        threshold: 累积概率阈值 (default=0.25 → ~25%的高概率操作符)
    
    Returns:
        selected_indices: 选择的操作符索引张量
    """
    device = probs.device
    probs = probs.detach()

    num_ops = probs.size(0)
    if num_ops == 0:
        return torch.tensor([], dtype=torch.long, device=device)

    selected = torch.tensor([], dtype=torch.long, device=device)
    cumulative = 0.0
    remaining = torch.arange(num_ops, device=device)

    # 循环采样直到达到阈值
    while cumulative < threshold and remaining.numel() > 0:
        # 从剩余操作符中多项采样一个
        sampled = torch.multinomial(probs[remaining], num_samples=1)
        idx = remaining[sampled].squeeze()

        # 检查重复
        if not torch.any(selected == idx):
            selected = torch.cat([selected, idx.unsqueeze(0)])
            cumulative += probs[idx].item()

        # 移除已采样的操作符
        mask = torch.ones_like(remaining, dtype=torch.bool)
        mask[sampled] = False
        remaining = remaining[mask]

    # 若未选任何操作符，选最高概率的
    if selected.numel() == 0:
        selected = probs.argmax().unsqueeze(0)

    return selected

def get_sentence_embedding(sentence: str) -> torch.Tensor:
    """一次性编码句子（非Module方式）"""
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(sentence)
    return torch.tensor(embeddings)
```

---

## 运行时管理

### 14. runtime/initializer.py - 运行时初始化

```python
def build_runtime_attributes(
    settings,
    specific_indices: Iterable[int] | None = None,
    workflow_class=None,
) -> dict[str, object]:
    """构建TrainingLoop消耗的所有状态对象"""
    
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建控制器和优化器
    controller = MultiLayerController(device=device).to(device)
    optimizer = torch.optim.Adam(controller.parameters(), lr=settings.optimizer.learning_rate)
    
    # 加载操作符嵌入
    operator_embeddings = _load_operator_embeddings(settings, device)

    # 如果是测试模式，加载检查点
    if settings.mode == "Test":
        checkpoint_path = settings.paths.controller_checkpoint(
            settings.dataset,
            settings.optimizer.round_number,
            settings.optimizer.sample,
        )
        controller.load_state_dict(torch.load(checkpoint_path, map_location=device))
        controller.eval()

    # 动态导入MaAS工作流类
    workflow_type = workflow_class or load_workflow_class(settings)
    
    # 实例化工作流
    architecture_workflow = workflow_type(
        name=settings.dataset,
        llm_config=settings.exec_llm_config,
        dataset=settings.dataset,
        controller=controller,
        operator_embeddings=operator_embeddings,
    )
    
    # 加载数据集
    settings.run_directory.mkdir(parents=True, exist_ok=True)
    problems = load_problems(settings, specific_indices=specific_indices)

    logger.info(
        "Initialized MaAS runtime objects: dataset=%s mode=%s problems=%s",
        settings.dataset,
        settings.mode,
        len(problems),
    )
    
    return {
        "settings": settings,
        "controller": controller,
        "optimizer": optimizer,
        "operator_embeddings": operator_embeddings,
        "architecture_workflow": architecture_workflow,
        "problems": problems,
        
        # 循环状态
        "problem_index": 0,
        "repetition": 1,
        "batch_index": 0,
        
        # 批次累积
        "batch_logprobs": [],
        "batch_scores": [],
        "batch_costs": [],
        
        # 统计
        "all_scores": [],
        "current_repetition_scores": [],
        "sample_results": [],
        
        # 配置
        "result_columns": _default_result_columns(settings.dataset),
        "previous_cost": 0.0,
        "previous_repetition_score": None,
        "batch_size": settings.optimizer.batch_size,
        "device": device,
        "run_directory": settings.run_directory,
    }

def load_workflow_class(settings):
    """动态导入并返回MaAS Workflow类"""
    maas_project_root = os.getenv("METAGPT_PROJECT_ROOT")
    if maas_project_root:
        if maas_project_root in sys.path:
            sys.path.remove(maas_project_root)
        sys.path.insert(0, maas_project_root)
    
    application_root = str(settings.paths.application_root)
    if application_root in sys.path:
        sys.path.remove(application_root)
    sys.path.insert(0, application_root)
    
    split = "test" if settings.mode == "Test" else "train"
    module_name = f"assets.optimized.{settings.dataset}.{split}.graph"
    
    _clear_assets_modules()
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    return module.Workflow
```

### 15. runtime/async_runner.py - 异步桥接

```python
class AsyncRunnerContextError(RuntimeError):
    """当从异步上下文调用同步节点时抛出"""

def run_async_once(coro: Coroutine[Any, Any, T]) -> T:
    """
    从同步MASFactory节点运行MaAS异步工作流
    
    MASFactory节点是同步的，但MaAS Workflow是异步的。这个函数
    允许同步节点调用一个完整的MaAS异步调用，同时拒绝嵌套调用
    （即从已运行的event loop内调用）。
    
    Args:
        coro: 要执行的协程
    
    Returns:
        协程的返回值
    
    Raises:
        AsyncRunnerContextError: 如果从异步上下文调用
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 没有运行的event loop，创建一个新的
        return asyncio.run(coro)

    # 已在异步上下文中
    coro.close()
    raise AsyncRunnerContextError(
        "MaAS MASFactory workflow must be invoked from a synchronous context."
    )
```

### 16. runtime/data_loader.py - 数据集加载

```python
def load_jsonl_data(
    file_path: str | Path,
    specific_indices: Iterable[int] | None = None
) -> list[dict]:
    """加载JSONL文件中的所有对象"""
    path = Path(file_path)
    data = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))

    if specific_indices is not None:
        return [data[index] for index in specific_indices if index < len(data)]
    
    return data

def load_problems(settings, specific_indices: Iterable[int] | None = None) -> list[dict]:
    """加载数据集的问题"""
    return load_jsonl_data(settings.dataset_file, specific_indices=specific_indices)
```

---

**End of Code Snippets Reference**
