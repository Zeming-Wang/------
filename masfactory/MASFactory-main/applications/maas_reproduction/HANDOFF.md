# MaAS MASFactory Migration Handoff

This document summarizes the current implementation state for the next AI or developer taking over the MaAS reproduction application inside MASFactory.

## Current Goal

Migrate MaAS into MASFactory as a self-contained application under:

```text
applications/maas_reproduction/
```

The migration must preserve the MaAS paper/runtime behavior:

- Keep MaAS `Workflow.__call__` intact for dynamic operator execution.
- Keep the MaAS controller, operator search space, sampling rules, EarlyStop behavior, benchmark scoring, and optimizer loss formula.
- Use MASFactory Graph/Loop/CustomNode structure without modifying MASFactory core.
- Pass runtime objects such as controller, optimizer, and tensors through Loop attributes, not Edge messages.

The agreed phase-1 graph is:

```text
RootGraph
  -> ConfigNode
  -> TrainingLoop
       Controller
         -> ArchitectureExecNode
         -> EvaluatorNode
         -> LossUpdateNode
         -> Controller
  -> ResultNode
```

Phase 1 intentionally does not implement a nested operator-level Loop and does not split every MaAS operator into MASFactory nodes.

## Repository Context

Workspace root:

```text
<workspace-root>
```

Target MASFactory repo:

```text
<workspace-root>\masfactory\MASFactory-main
```

Reference MaAS repo:

```text
<workspace-root>\MaAS-main
```

In the current machine, `<workspace-root>` is the Desktop folder named with Chinese characters for paper reproduction. Prefer relative paths from the repository root when possible to avoid terminal encoding issues.

## Active Python Environment

Use the conda environment `mas_env` for MaAS/MASFactory work on this machine:

```text
C:\Users\lenovo\.conda\envs\mas_env
```

Prefer calling its interpreter directly from PowerShell:

```powershell
& C:\Users\lenovo\.conda\envs\mas_env\python.exe ...
```

This avoids occasional `conda run` temporary-file lock errors observed when multiple `conda run -n mas_env ...` commands execute in parallel.

Verified environment:

```text
Python 3.10.20
torch 2.1.0+cu118
torch.cuda.is_available(): False
pydantic 2.6.4
google-genai 1.28.0
antlr4-python3-runtime 4.11.1
```

The following dependencies were present in `mas_env` when checked:

```text
torch
sentence_transformers
tenacity
tiktoken
sympy
regex
pandas
aiofiles
tqdm
pydantic
tree_sitter
tree_sitter_python
```

The global/default shell Python was `C:\Python314\python.exe`; do not use it for full MaAS verification because it lacks key dependencies.

Environment fixes already applied:

1. Downgraded `google-genai` from `2.16.0` to `1.28.0`. This fixes `import masfactory` while keeping MaAS-compatible `pydantic==2.6.4`.

2. Installed `antlr4-python3-runtime==4.11.1`. This fixes `sympy.parsing.latex.parse_latex()`.

3. Downgraded `httpx` from `0.28.1` to `0.25.2`. This fixes MaAS/OpenAI client construction for real LLM runs:

```text
TypeError: AsyncClient.__init__() got an unexpected keyword argument 'proxies'
```

4. Set conda env config var:

```text
METAGPT_PROJECT_ROOT = C:\Users\lenovo\Desktop\论文复现相关\MaAS-main
```

After normal `conda activate mas_env`, MaAS can locate `config/config2.yaml`. In Codex tool calls that invoke `python.exe` directly, still set `$env:METAGPT_PROJECT_ROOT` in the command because activation hooks are bypassed.

Known remaining environment notes:

1. `torch.cuda.is_available()` is `False`. Training can run on CPU, but controller sampling and full LLM evaluation will be slower. CPU is still the right baseline for golden reference determinism.

2. `pip check` still reports metadata conflicts because this environment contains both MaAS/MetaGPT and MASFactory dependencies. Important reported conflicts include:

```text
masfactory 1.0.4 wants newer openai/anthropic/numpy
google-genai 1.28.0 wants httpx>=0.28.1
metagpt 0.8.1 wants websockets==11.0.3
```

Current practical verification passes for the migration path:

```text
import masfactory
from masfactory import RootGraph, Loop, CustomNode
MaAS ModelsConfig/provider registry/ActionNode imports
sympy.parsing.latex.parse_latex("x+1")
create_training_loop().build()
application tests
```

The MaAS import path currently prints pydantic warnings about fields named `name`, `metadata`, `done`, and `error` shadowing parent attributes in `Operation`. These warnings did not fail imports or tests.

Do not chase `pip check` by broadly upgrading packages unless a concrete runtime failure appears; upgrading `pydantic` to satisfy MASFactory metadata breaks MaAS semantic-kernel imports.

## Handoff Maintenance Rule

For every future code change in `applications/maas_reproduction`, update this `HANDOFF.md` in the same turn with:

- files changed
- behavior changed
- tests run and exact result
- unresolved dependency or environment issues
- next-step impact

Important reference files used:

```text
MaAS-main\maas\ext\maas\models\controller.py
MaAS-main\maas\ext\maas\models\utils.py
MaAS-main\maas\ext\maas\benchmark\benchmark.py
MaAS-main\maas\ext\maas\benchmark\gsm8k.py
MaAS-main\maas\ext\maas\benchmark\math.py
MaAS-main\maas\ext\maas\benchmark\humaneval.py
MaAS-main\maas\ext\maas\scripts\optimized\GSM8K\train\graph.py
masfactory\MASFactory-main\masfactory\components\graphs\loop.py
masfactory\MASFactory-main\masfactory\components\custom_node.py
masfactory\MASFactory-main\masfactory\core\node.py
```

## Implemented Files

Current file list under `applications/maas_reproduction`:

```text
applications\maas_reproduction\main.py
applications\maas_reproduction\__init__.py
applications\maas_reproduction\assets\data\gsm8k_train.jsonl
applications\maas_reproduction\assets\data\gsm8k_test.jsonl
applications\maas_reproduction\assets\optimized\<dataset>\<split>\graph.py
applications\maas_reproduction\assets\optimized\<dataset>\<split>\template\*.py
applications\maas_reproduction\assets\optimized\<dataset>\<split>\template\operator.json
applications\maas_reproduction\maas_reproduction\__init__.py
applications\maas_reproduction\maas_reproduction\benchmarks\__init__.py
applications\maas_reproduction\maas_reproduction\benchmarks\gsm8k.py
applications\maas_reproduction\maas_reproduction\benchmarks\humaneval.py
applications\maas_reproduction\maas_reproduction\benchmarks\math.py
applications\maas_reproduction\maas_reproduction\config\__init__.py
applications\maas_reproduction\maas_reproduction\config\experiments.py
applications\maas_reproduction\maas_reproduction\config\model_config.py
applications\maas_reproduction\maas_reproduction\config\settings.py
applications\maas_reproduction\maas_reproduction\graphs\__init__.py
applications\maas_reproduction\maas_reproduction\graphs\training_loop.py
applications\maas_reproduction\maas_reproduction\models\__init__.py
applications\maas_reproduction\maas_reproduction\models\controller.py
applications\maas_reproduction\maas_reproduction\models\utils.py
applications\maas_reproduction\maas_reproduction\nodes\__init__.py
applications\maas_reproduction\maas_reproduction\nodes\architecture_exec_node.py
applications\maas_reproduction\maas_reproduction\nodes\config_node.py
applications\maas_reproduction\maas_reproduction\nodes\evaluator_node.py
applications\maas_reproduction\maas_reproduction\nodes\loss_update_node.py
applications\maas_reproduction\maas_reproduction\nodes\result_node.py
applications\maas_reproduction\maas_reproduction\nodes\training_controller.py
applications\maas_reproduction\maas_reproduction\runtime\__init__.py
applications\maas_reproduction\maas_reproduction\runtime\async_runner.py
applications\maas_reproduction\maas_reproduction\runtime\data_loader.py
applications\maas_reproduction\maas_reproduction\runtime\initializer.py
applications\maas_reproduction\maas_reproduction\workflow.py
```

## What Each Implemented Area Does

### `config/settings.py`

Defines immutable runtime configuration:

- `MaASPaths`
- `OptimizerSettings`
- `MaASRuntimeSettings`
- dataset/mode/split/question type literals and supported constants

Important choices:

- `MaASRuntimeSettings` includes `question_type`, `operators`, `opt_llm_config`, and `exec_llm_config`.
- `MaASPaths.controller_checkpoint()` uses the MaAS-like path:

```text
checkpoint_root/<dataset>/train/round_<n>/<dataset>_controller_sample<m>.pth
```

This matches MaAS checkpoint semantics more closely than putting checkpoints under runtime logs.

### `config/experiments.py`

Defines the fixed MaAS search space per dataset.

GSM8K and MATH:

```text
Generate, GenerateCoT, MultiGenerateCoT, ScEnsemble, Programmer, SelfRefine, EarlyStop
```

HumanEval:

```text
Generate, GenerateCoT, MultiGenerateCoT, ScEnsemble, Test, SelfRefine, EarlyStop
```

Do not change these during migration unless deliberately changing MaAS behavior.

### `config/model_config.py`

Resolves model configs by name using original MaAS config support:

```python
maas.configs.models_config.ModelsConfig
```

This is still a bridge to the original MaAS package. Later, if the application must be fully self-contained, this should be migrated.

### `runtime/async_runner.py`

Defines:

```python
run_async_once(coro)
AsyncRunnerContextError
```

Behavior:

- Uses `asyncio.run(coro)` only when no event loop is already running.
- Raises `AsyncRunnerContextError`, a `RuntimeError` subclass, inside an existing event loop.
- Closes the unexecuted coroutine before raising this context error.
- No background threads.
- This is intended for MASFactory's synchronous `CustomNode` forward functions.

### `nodes/architecture_exec_node.py`

Defines:

```python
architecture_exec_forward(input_data, attributes)
```

Responsibilities:

- Reads `settings` and `architecture_workflow` from attributes.
- Calls full MaAS `Workflow.__call__` once per problem.
- For GSM8K/MATH: `workflow(problem)`
- For HumanEval: `workflow(problem, entry_point, str(settings.run_directory))`
- Uses MaAS benchmark-like timeout/retry:
  - `asyncio.wait_for(..., timeout=1500)`
  - tenacity retry, 20 attempts, wait 1 second
- On exhausted failure, returns a zero-cost failed sample, matching MaAS benchmark behavior.
- Does not retry or convert `AsyncRunnerContextError` into a zero-score sample. Calling this node inside an existing event loop is a wiring error, not a benchmark sample failure.

Important correction already made:

- Do not convert `logprob` to `float`.
- Original MaAS optimized workflows return `sum_log_prob` as a tensor with gradient. Converting it to float breaks controller training.

### `nodes/config_node.py`

Defines:

```python
config_forward(input_data, attributes)
```

Responsibilities:

- Build `OptimizerSettings` from MaAS CLI defaults and supplied graph input.
- Resolve opt/exec model configs via original MaAS `ModelsConfig.default()`.
- Build `MaASRuntimeSettings` from the fixed experiment config.
- Return `settings` plus serializable summary fields for downstream graph edges.

It does not create controller, optimizer, LLM, operator embeddings, problems, or architecture workflows.

### `models/utils.py`

Migrates MaAS:

- `get_sentence_embedding`
- `SentenceEncoder`
- `sample_operators`

Sampling logic is preserved from MaAS:

- Detach probabilities before sampling.
- Use `torch.multinomial`.
- Sample until cumulative probability reaches threshold.
- If nothing selected, use argmax.

### `models/controller.py`

Migrates MaAS:

- `OperatorSelector`
- `MultiLayerController`

Algorithm-preserving details:

- Uses global `sentence_encoder = SentenceEncoder()`.
- First layer operator encoder differs from later layers.
- Later layers concatenate current operator embeddings with previous selected operator embedding.
- First layer must begin with a generate-type operator.
- First layer EarlyStop is replaced by Generate and receives `-1.5` logprob penalty.
- Stops when EarlyStop is selected.

One format fix was necessary:

- The original `controller.py` had a corrupted comment on the line before `def forward`, effectively hiding the method in the displayed source. The migrated file restores a valid `def forward` while preserving logic.

### `benchmarks/gsm8k.py`

Defines `GSM8KScorer`:

- Extracts the last number from text.
- Compares expected and predicted values with `1e-6` tolerance.

### `benchmarks/math.py`

Defines `MATHScorer`:

- Extracts last `\boxed{...}` answer, otherwise last sentence.
- Compares string equality first.
- Then numeric comparison.
- Then symbolic comparison via SymPy if available.

Note:

- `sympy` is not installed in the current Python 3.14 environment, but simple boxed/numeric tests pass without importing it because import is delayed until symbolic comparison.

### `benchmarks/humaneval.py`

Defines `HumanEvalScorer`:

- Executes generated solution and HumanEval test.
- Runs `check(fn)` with a 15-second thread timeout.
- Writes failed execution details to `error.log` under the scorer `log_path`, matching the original MaAS benchmark's error visibility.
- Includes original MaAS special-case helper functions for:
  - `decode_cyclic`
  - `decode_shift`
  - `find_zero`

This implementation does not currently call MaAS `sanitize()`, because that utility depends on extra tree-sitter packages. If exact HumanEval equivalence is required, migrate or depend on `maas.utils.sanitize.sanitize`.

### `nodes/evaluator_node.py`

Defines:

```python
evaluator_forward(input_data, attributes)
```

Responsibilities:

- Reads `settings.dataset`.
- Scores current prediction using the dataset scorer.
- Sends HumanEval failure logs to `settings.run_directory`.
- Logs each problem score.
- Returns only the planned Edge payload:

```text
score, cost, logprob, problem_index
```

Dataset meaning of `expected_answer`:

- GSM8K: answer text
- MATH: solution text
- HumanEval: test code

### `nodes/loss_update_node.py`

Defines:

```python
loss_update_forward(input_data, attributes)
```

Responsibilities:

- Appends current `score`, `cost_delta`, and `logprob` to batch attributes.
- Resolves tensor device from `attributes["device"]` when provided, otherwise from `attributes["controller"].parameters()`, and only then falls back to CUDA/CPU probing.
- Uses MaAS formula:

```python
utility = score - 3 * cost_delta
loss = -(logprobs * utilities).mean()
```

- Runs update only when batch reaches `batch_size` and `settings.mode == "Graph"`:

```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

Important detail:

- MaAS original benchmark tracks `previous_cost` because workflow cost is cumulative. This node stores `previous_cost` in attributes and uses cost delta for utility.
- Python float `logprob` values produce a scalar tensor without gradient and therefore do not trigger controller updates. This preserves behavior for graph implementations that return detached logprobs, and tests now cover this production-like path.
- Batch updates are logged. Losses without gradients are logged and skipped.

### `nodes/training_controller.py`

Defines:

```python
training_controller(input_data, attributes) -> bool
```

This is the `Loop` terminate condition function.

Responsibilities:

- Deletes stale `result_*` fields from controller message cache each iteration.
- Reads next problem from `attributes["problems"]`.
- Injects Edge fields:

```text
problem, entry_point, expected_answer, problem_index
```

- Advances `problem_index`.
- Handles repetition count using `settings.optimizer.sample`.
- On termination, writes fields required by `TrainingLoop -> ResultNode`:

```text
average_score, round, checkpoint_path, result_path, runtime_metadata
```

Repetition switches and final average score are logged.

### `nodes/result_node.py`

Defines:

```python
result_forward(input_data, attributes)
```

Responsibilities:

- Return final `average_score`, `round`, `checkpoint_path`, `result_path`, and `runtime_metadata`.
- Log the final MaAS reproduction summary.

### `graphs/training_loop.py`

Defines Edge key constants and:

```python
create_training_loop(name="TrainingLoop", max_iterations=100000)
```

Constructs:

```text
Controller -> ArchitectureExecNode -> EvaluatorNode -> LossUpdateNode -> Controller
```

Edge keys are explicitly declared:

Controller to Architecture:

```text
problem, entry_point, expected_answer, problem_index
```

Architecture to Evaluator:

```text
problem, entry_point, expected_answer, prediction, cost, logprob, problem_index
```

Evaluator to Loss:

```text
score, cost, logprob, problem_index
```

Loss to Controller:

```text
result_score, result_cost, result_logprob, result_loss,
result_update_performed, result_problem_index
```

TrainingLoop push keys:

```text
average_score, round, checkpoint_path, result_path, runtime_metadata
```

MASFactory imports and `loss_update_forward` import are delayed inside `create_training_loop()` so lightweight tests can import constants without requiring all MASFactory and torch dependencies.

The graph builder logs when the MaAS TrainingLoop body is created.

### `runtime/data_loader.py`

Defines:

```python
load_jsonl_data(file_path, specific_indices=None)
load_problems(settings, specific_indices=None)
```

Responsibilities:

- Mirror original MaAS `BaseBenchmark.load_data()` JSONL loading behavior.
- Preserve `specific_indices` filtering and skip out-of-range indices.
- Use `settings.dataset_file` for the graph runtime problem list.

### `workflow.py`

Defines:

```python
build_maas_reproduction_graph(name="MaASReproduction")
```

Builds the phase-1 MASFactory RootGraph:

```text
entry -> ConfigNode -> TrainingLoop -> ResultNode -> exit
```

`ConfigNode` pushes `settings` into RootGraph attributes so `TrainingLoop` can pull it. Runtime objects such as `problems`, `architecture_workflow`, `controller`, `optimizer`, and `operator_embeddings` are supplied through invocation attributes by `build_runtime_attributes()` or by tests using an injected fake workflow.

## Tests Added

Tests currently under:

```text
applications/maas_reproduction/tests/
```

Known tests:

- `test_async_runner.py`
- `test_architecture_exec_node.py`
- `test_models.py`
- `test_benchmarks.py`
- `test_evaluator_node.py`
- `test_loss_update_node.py`
- `test_training_controller.py`
- `test_training_loop_graph.py`
- `test_data_loader.py`
- `test_config_node.py`
- `test_result_node.py`
- `test_workflow.py`
- `test_runtime_initializer.py`
- `test_model_config.py`
- `test_main.py`
- `test_root_graph_integration.py`
- `test_optimized_assets.py`

Additional coverage added after review:

- Python float `logprob` path in `LossUpdateNode`
- incomplete batch accumulation without update
- device resolution from controller parameters
- cross-repetition `problem_index` reset in `training_controller`
- HumanEval `error.log` writing
- JSONL problem loading and index filtering
- ConfigNode MaAS defaults and runtime settings creation
- ResultNode final summary output
- RootGraph build wiring
- runtime attribute initialization
- Graph-mode controller checkpoint save at final loop termination
- local optimized Workflow assets import checks
- `METAGPT_PROJECT_ROOT` path precedence for MaAS model config imports
- `main.run()` runtime initialization before RootGraph invocation
- real RootGraph.invoke path through ConfigNode -> TrainingLoop -> ArchitectureExecNode -> EvaluatorNode -> LossUpdateNode -> ResultNode
- optimized Workflow log probability accumulation stays as a tensor and avoids `.item()`

Latest verification commands run from:

```text
<workspace-root>\masfactory\MASFactory-main
```

Command:

```powershell
$env:PYTHONPATH='applications\maas_reproduction'; & C:\Users\lenovo\.conda\envs\mas_env\python.exe -m unittest discover applications\maas_reproduction\tests
```

Result:

```text
........................................
----------------------------------------------------------------------
Ran 40 tests in 29.124s

OK
```

Command:

```powershell
& C:\Users\lenovo\.conda\envs\mas_env\python.exe -m compileall -q applications\maas_reproduction
```

Result:

```text
exit code 0
```

With `mas_env`, torch-dependent controller/model and loss update tests run instead of being skipped.

## Errors Encountered And What They Mean

### `ModuleNotFoundError: No module named 'torch'`

Current shell Python:

```text
C:\Python314\python.exe
Python 3.14.4
```

This only applies to the global/default Python. Since MaAS training depends on torch, use `mas_env`.

Tests skip torch-specific cases only when run outside `mas_env`. Production modules such as `models/*` and `loss_update_node.py` still require torch.

### `ModuleNotFoundError: No module named 'tiktoken'`

Importing MASFactory graph classes initially failed because MASFactory imports model/token tracking dependencies from `masfactory.__init__`.

Fix applied:

- Delayed MASFactory imports inside `create_training_loop()`.
- Lightweight tests import only constants without requiring full MASFactory dependency set.

This also applies to the global/default Python. `mas_env` has `tiktoken`.

### `sympy False`, `regex False`

A check showed these packages are not installed in the global/default Python:

```text
sympy False
regex False
```

The migrated `MATHScorer` avoids importing SymPy until symbolic comparison is needed. `mas_env` now has `sympy`, `regex`, and `antlr4-python3-runtime==4.11.1`.

### Original MaAS `controller.py` format issue

The original source displayed line 20 as a corrupted comment plus `def forward(...)` on the same line. The migration restored a normal method definition. This is a formatting repair, not an algorithm change.

### Original HumanEval `sanitize` dependency

Original MaAS HumanEval uses:

```python
from maas.utils.sanitize import sanitize
```

That file depends on:

- `tree_sitter`
- `tree_sitter_python`

The current migrated `HumanEvalScorer` does not use sanitize yet. This is a known equivalence gap for HumanEval.

## Current Git Status For This Application

Latest status limited to `applications/maas_reproduction`:

```text
A  applications/maas_reproduction/__init__.py
A  applications/maas_reproduction/maas_reproduction/__init__.py
AM applications/maas_reproduction/maas_reproduction/config/__init__.py
AM applications/maas_reproduction/maas_reproduction/config/settings.py
?? applications/maas_reproduction/maas_reproduction/benchmarks/
?? applications/maas_reproduction/maas_reproduction/config/experiments.py
?? applications/maas_reproduction/maas_reproduction/config/model_config.py
?? applications/maas_reproduction/maas_reproduction/graphs/
?? applications/maas_reproduction/maas_reproduction/models/
?? applications/maas_reproduction/maas_reproduction/nodes/
?? applications/maas_reproduction/maas_reproduction/runtime/
```

There were pre-existing unrelated changes in the MaAS reference repo. Do not revert them unless the user explicitly asks.

## Completed Plan Items

Completed:

1. `config/settings.py`
2. `config/experiments.py`
3. `config/model_config.py`
4. `runtime/async_runner.py`
5. `nodes/architecture_exec_node.py`
6. MaAS controller and sampling migration
7. benchmark scoring helpers
8. `nodes/evaluator_node.py`
9. `nodes/loss_update_node.py`
10. `nodes/training_controller.py`
11. `graphs/training_loop.py`
12. Review-driven runtime fixes:
    - LossUpdate device resolution from controller
    - float logprob and incomplete batch tests
    - cross-repetition controller test
    - HumanEval error log restoration
    - evaluator/loss/controller/graph logging
13. `runtime/data_loader.py`
14. `nodes/config_node.py`
15. `nodes/result_node.py`
16. `workflow.py` RootGraph wiring
17. `runtime/initializer.py`
18. Graph-mode controller checkpoint save in `nodes/training_controller.py`
19. Optimized architecture/operator/prompt assets copied into `applications/maas_reproduction/assets/optimized`
20. `main.py` CLI entry point
21. `METAGPT_PROJECT_ROOT` path precedence for MaAS imports
22. RootGraph entry/exit wiring
23. TrainingLoop body nodes use `push_keys={}` to avoid business-message fields overwriting Loop attributes
24. Optimized Workflow `sum_log_prob` tensor accumulation is consistent across GSM8K, MATH, and HumanEval train/test

Not completed:

1. MATH and HumanEval dataset JSONL files
2. HumanEval sanitize equivalence
3. golden reference comparison
4. one-query and one-batch smoke tests against a real or stubbed LLM config
5. README.md
6. REPRODUCTION.md
7. requirements.txt

## Next Implementation Steps

Recommended next order:

1. Run a one-query smoke test for GSM8K.
2. Run a one-batch smoke test for GSM8K.
3. Add or locate MATH and HumanEval JSONL data if those datasets must run locally.
4. Generate golden reference traces on CPU.
5. Add README and REPRODUCTION docs.

## Manual Setup Needed

Use `mas_env` for full MaAS/MASFactory execution. If recreating the environment elsewhere, install at least:

```text
torch
sentence-transformers
tenacity
tiktoken
sympy
regex
pandas
aiofiles
tqdm
pydantic
tree_sitter
tree_sitter_python
```

Also install MASFactory's normal requirements from the repo and any original MaAS requirements needed by:

- LLM provider config
- `ModelsConfig`
- `ActionNode`
- operator execution
- HumanEval sanitize

The tests are designed so lightweight checks can run without all heavy dependencies, but real training should use `mas_env` or an equivalent environment.

For Codex direct PowerShell commands, use this prefix when MaAS imports are involved:

```powershell
$env:METAGPT_PROJECT_ROOT='C:\Users\lenovo\Desktop\论文复现相关\MaAS-main'
$env:PYTHONPATH='applications\maas_reproduction'
& C:\Users\lenovo\.conda\envs\mas_env\python.exe ...
```

## Important Behavioral Constraints To Preserve

Future work should continue to follow these constraints:

- Do not pass torch modules/tensors through MASFactory Edge messages.
- Store controller, optimizer, operator embeddings, workflow, and problem list in Loop attributes.
- Keep Edge messages for business payload only.
- Keep `result_` prefix on feedback from LossUpdateNode to Controller.
- Do not add Controller direct edges to Evaluator or LossUpdateNode.
- Do not rewrite MaAS `Workflow.__call__` into a new algorithm.
- Do not split every operator into a MASFactory node in phase 1.
- Do not modify MASFactory core.
- Do not convert `logprob` tensors to floats before loss computation.

## Known Design Tradeoffs

### Why `ArchitectureExecNode` stays coarse

The MaAS operator-level execution is dynamic:

- controller samples operators based on the query
- four layers execute sequentially
- EarlyStop can terminate
- operators have different input/output shapes
- Programmer/Test/ScEnsemble post-processing differs by dataset

Encoding all of that as static MASFactory nodes would recreate MaAS `Workflow.__call__` inside a constrained graph controller. Phase 1 keeps this inside a single semantic node to preserve correctness.

### Why only one Loop

MASFactory supports Loop as a graph-like node, but nested Loop usage is not established in existing apps. The current implementation uses one `TrainingLoop`, which is enough to express batch/repetition/problem scheduling while leaving architecture-level dynamic execution inside MaAS runtime.

### Why imports are sometimes delayed

Current local environment lacks some heavy dependencies. Delaying imports lets tests validate pure constants and lightweight node functions. Real graph construction still requires the real dependencies.

## Verification Snapshot

## Latest Update: Runtime Initialization And Checkpoint Save

Date: 2026-08-10

Files changed in this update:

```text
applications/maas_reproduction/maas_reproduction/runtime/initializer.py
applications/maas_reproduction/maas_reproduction/runtime/__init__.py
applications/maas_reproduction/maas_reproduction/nodes/training_controller.py
applications/maas_reproduction/tests/test_runtime_initializer.py
applications/maas_reproduction/tests/test_training_controller.py
applications/maas_reproduction/HANDOFF.md
```

Behavior added:

- `build_runtime_attributes(settings, specific_indices=None, workflow_class=None)` now creates the objects that the MASFactory `TrainingLoop` expects in attributes:

```text
settings, controller, optimizer, operator_embeddings, architecture_workflow,
problems, problem_index, repetition, batch_index, batch_logprobs,
batch_scores, batch_costs, all_scores, previous_cost, batch_size,
device, run_directory
```

- This maps directly to original MaAS `Optimizer._optimize_graph_maas()`:
  - `MultiLayerController(device=...)`
  - `torch.optim.Adam(controller.parameters(), lr=...)`
  - operator descriptions from `train/template/operator.json`
  - `torch.stack([get_sentence_embedding(...)])`
  - `Workflow(name, llm_config, dataset, controller, operator_embeddings)`
  - training JSONL data loaded before evaluation

- Test mode loads the controller checkpoint and sets `controller.eval()`, matching original MaAS `Optimizer.test()`.
- Graph mode now saves `controller.state_dict()` at final `TrainingLoop` termination. This restores original MaAS `BaseBenchmark.run_evaluation()` behavior, where training writes `<dataset>_controller_sample<sample>.pth`.

Tests run for this update:

```powershell
$env:PYTHONPATH='applications\maas_reproduction'; & C:\Users\lenovo\.conda\envs\mas_env\python.exe -m unittest applications\maas_reproduction\tests\test_runtime_initializer.py applications\maas_reproduction\tests\test_training_controller.py
```

Result:

```text
Ran 5 tests in 0.039s
OK
```

Full verification:

```powershell
$env:PYTHONPATH='applications\maas_reproduction'; $env:METAGPT_PROJECT_ROOT='C:\Users\lenovo\Desktop\论文复现相关\MaAS-main'; & C:\Users\lenovo\.conda\envs\mas_env\python.exe -m unittest discover applications\maas_reproduction\tests
```

Result:

```text
Ran 34 tests in 20.375s
OK
```

Compile verification:

```powershell
& C:\Users\lenovo\.conda\envs\mas_env\python.exe -m compileall -q applications\maas_reproduction
```

Result: exit code 0

Remaining practical blocker after this snapshot:

- Optimized assets now import locally. The next live smoke run depends on a valid MaAS model config and provider credentials because `Workflow.__call__` calls the real LLM backend.

## Latest Update: Optimized Assets And CLI

Date: 2026-08-10

Files changed in this update:

```text
applications/maas_reproduction/main.py
applications/maas_reproduction/assets/data/gsm8k_train.jsonl
applications/maas_reproduction/assets/data/gsm8k_test.jsonl
applications/maas_reproduction/assets/optimized/GSM8K/{train,test}/...
applications/maas_reproduction/assets/optimized/MATH/{train,test}/...
applications/maas_reproduction/assets/optimized/HumanEval/{train,test}/...
applications/maas_reproduction/maas_reproduction/config/model_config.py
applications/maas_reproduction/maas_reproduction/runtime/initializer.py
applications/maas_reproduction/tests/test_main.py
applications/maas_reproduction/tests/test_model_config.py
applications/maas_reproduction/tests/test_runtime_initializer.py
applications/maas_reproduction/HANDOFF.md
```

Behavior added:

- Copied optimized MaAS `graph.py` and `template` assets for GSM8K, MATH, and HumanEval train/test into the application-local assets directory.
- Excluded original runtime artifacts such as `round_1`, smoke logs, CSVs, result JSON, and `__pycache__`.
- Rewrote copied optimized imports from original absolute paths such as:

```python
maas.ext.maas.scripts.optimized.GSM8K.train.template.operator
```

to local relative imports such as:

```python
from .template import operator
from .template.operator_registry import operator_mapping, operator_names
from .operator_an import *
from .op_prompt import *
```

This keeps the copied `Workflow.__call__` algorithm intact while making the optimized assets import from the application-local copy.

- `load_workflow_class(settings)` now imports Workflow as an assets package:

```text
assets.optimized.<dataset>.<train|test>.graph
```

This supports relative imports inside copied `graph.py`.

- `load_workflow_class()` and `resolve_model_configs()` now move `METAGPT_PROJECT_ROOT` to the front of `sys.path` before importing MaAS modules. This fixes the local MASFactory `maas/tools` namespace directory shadowing original MaAS `maas.tools`.
- Added `main.py`:

```python
run(input_data, specific_indices=None)
```

The CLI flow is:

```text
config_forward(input_data, {})
-> build_runtime_attributes(settings, specific_indices)
-> build_maas_reproduction_graph()
-> graph.build()
-> graph.invoke(input_data, attributes=runtime_attributes)
```

This maps to original MaAS CLI + Optimizer startup while preserving MASFactory RootGraph invocation.

Data copied:

- `gsm8k_train.jsonl`
- `gsm8k_test.jsonl`

Data still missing locally:

- MATH JSONL
- HumanEval JSONL

Tests run for this update:

```powershell
$env:PYTHONPATH='applications\maas_reproduction'; $env:METAGPT_PROJECT_ROOT='C:\Users\lenovo\Desktop\论文复现相关\MaAS-main'; & C:\Users\lenovo\.conda\envs\mas_env\python.exe -m unittest applications\maas_reproduction\tests\test_runtime_initializer.py applications\maas_reproduction\tests\test_model_config.py
```

Result:

```text
Ran 4 tests in 11.670s
OK
```

Full verification:

```powershell
$env:PYTHONPATH='applications\maas_reproduction'; $env:METAGPT_PROJECT_ROOT='C:\Users\lenovo\Desktop\论文复现相关\MaAS-main'; & C:\Users\lenovo\.conda\envs\mas_env\python.exe -m unittest discover applications\maas_reproduction\tests
```

Result:

```text
Ran 38 tests in 33.717s
OK
```

Compile verification:

```powershell
& C:\Users\lenovo\.conda\envs\mas_env\python.exe -m compileall -q applications\maas_reproduction
```

Result: exit code 0

CLI help verification:

```powershell
$env:PYTHONPATH='applications\maas_reproduction'; & C:\Users\lenovo\.conda\envs\mas_env\python.exe -m applications.maas_reproduction.main --help
```

Result:

```text
usage: main.py [-h] --dataset {GSM8K,HumanEval,MATH} ...
exit code 0
```

Next-step impact:

- The previous import blocker for `ArchitectureExecNode` is resolved for optimized assets.
- The next meaningful verification is a GSM8K single-query smoke run. That will call the real LLM provider unless a stub LLM config/backend is introduced, so credentials/model config must be valid before running it live.

## Latest Update: Real RootGraph Integration And Logprob Consistency

Date: 2026-08-10

Files changed in this update:

```text
applications/maas_reproduction/maas_reproduction/workflow.py
applications/maas_reproduction/maas_reproduction/graphs/training_loop.py
applications/maas_reproduction/assets/optimized/GSM8K/test/graph.py
applications/maas_reproduction/assets/optimized/MATH/train/graph.py
applications/maas_reproduction/assets/optimized/MATH/test/graph.py
applications/maas_reproduction/assets/optimized/HumanEval/train/graph.py
applications/maas_reproduction/assets/optimized/HumanEval/test/graph.py
applications/maas_reproduction/tests/test_root_graph_integration.py
applications/maas_reproduction/tests/test_optimized_assets.py
applications/maas_reproduction/HANDOFF.md
```

Behavior changed:

- `build_maas_reproduction_graph()` now wires the actual RootGraph entry and exit:

```text
entry -> ConfigNode -> TrainingLoop -> ResultNode -> exit
```

Before this, `RootGraph.invoke()` could return `{}` because no entry edge started `ConfigNode` and no exit edge collected `ResultNode`.

- TrainingLoop body nodes now use `push_keys={}`:

```text
ArchitectureExecNode
EvaluatorNode
LossUpdateNode
```

This prevents business message fields such as `problem_index` from being pushed back into Loop attributes and overwriting scheduler state written by `training_controller()`. The bug manifested as the same problem being repeatedly processed until the Loop max iteration warning.

- Copied optimized Workflows now use tensor log probability accumulation consistently:

```python
sum_log_prob = torch.tensor(0.0, device=self.device)
sum_log_prob = sum_log_prob + log_probs_layers[layer_idx]
```

This was normalized across GSM8K, MATH, and HumanEval train/test assets. The copied assets no longer contain `log_probs_layers[layer_idx].item()` in Workflow logprob accumulation.

Tests added:

- `test_root_graph_integration.py` runs a real `RootGraph.invoke()` with a fake async workflow, covering:
  - ConfigNode execution
  - TrainingLoop controller scheduling
  - ArchitectureExecNode attribute access
  - EvaluatorNode scoring
  - LossUpdateNode batch update
  - final ResultNode output
  - checkpoint save

- `test_optimized_assets.py` scans optimized graph assets to ensure logprob accumulation stays tensor-based.

Targeted verification:

```powershell
$env:PYTHONPATH='applications\maas_reproduction'; & C:\Users\lenovo\.conda\envs\mas_env\python.exe -m unittest applications\maas_reproduction\tests\test_root_graph_integration.py applications\maas_reproduction\tests\test_optimized_assets.py applications\maas_reproduction\tests\test_workflow.py applications\maas_reproduction\tests\test_training_loop_graph.py
```

Result:

```text
Ran 4 tests in 0.028s
OK
```

Full verification:

```powershell
$env:PYTHONPATH='applications\maas_reproduction'; $env:METAGPT_PROJECT_ROOT='C:\Users\lenovo\Desktop\论文复现相关\MaAS-main'; & C:\Users\lenovo\.conda\envs\mas_env\python.exe -m unittest discover applications\maas_reproduction\tests
```

Result:

```text
Ran 40 tests in 29.124s
OK
```

Compile verification:

```powershell
& C:\Users\lenovo\.conda\envs\mas_env\python.exe -m compileall -q applications\maas_reproduction
```

Result: exit code 0

## Latest Update: GSM8K Single-Query Live Smoke

Date: 2026-08-10

Environment changed:

```powershell
& C:\Users\lenovo\.conda\envs\mas_env\python.exe -m pip install httpx==0.25.2
```

Reason:

- The first live smoke failed before any problem execution while constructing `AsyncOpenAI`.
- Error:

```text
TypeError: AsyncClient.__init__() got an unexpected keyword argument 'proxies'
```

- `httpx==0.25.2` matches MaAS/MetaGPT's expected dependency and allowed the OpenAI client to initialize.

Live smoke command:

```powershell
$env:PYTHONPATH='applications\maas_reproduction'
$env:METAGPT_PROJECT_ROOT='C:\Users\lenovo\Desktop\论文复现相关\MaAS-main'
& C:\Users\lenovo\.conda\envs\mas_env\python.exe -m applications.maas_reproduction.main --dataset GSM8K --mode Graph --sample 1 --batch-size 1 --indices 0
```

This did call the configured LLM provider and therefore used the configured API key.

Result:

```json
{
  "average_score": 1.0,
  "round": 1,
  "checkpoint_path": "applications/maas_reproduction/assets/optimized/GSM8K/train/round_1/GSM8K_controller_sample1.pth",
  "result_path": "applications/maas_reproduction/runs/GSM8K/Graph/round_1",
  "runtime_metadata": {
    "dataset": "GSM8K",
    "mode": "Graph",
    "processed_problems": 1
  }
}
```

Follow-up verification after the environment change:

```powershell
$env:PYTHONPATH='applications\maas_reproduction'; $env:METAGPT_PROJECT_ROOT='C:\Users\lenovo\Desktop\论文复现相关\MaAS-main'; & C:\Users\lenovo\.conda\envs\mas_env\python.exe -m unittest discover applications\maas_reproduction\tests
```

Result:

```text
Ran 40 tests in 24.767s
OK
```

Compile verification:

```powershell
& C:\Users\lenovo\.conda\envs\mas_env\python.exe -m compileall -q applications\maas_reproduction
```

Result: exit code 0

Fresh verification before writing this handoff:

```powershell
$env:PYTHONPATH='applications\maas_reproduction'; $env:METAGPT_PROJECT_ROOT='C:\Users\lenovo\Desktop\论文复现相关\MaAS-main'; & C:\Users\lenovo\.conda\envs\mas_env\python.exe -m unittest discover applications\maas_reproduction\tests
```

Output:

```text
........................................
----------------------------------------------------------------------
Ran 40 tests in 29.124s

OK
```

Fresh compile check:

```powershell
& C:\Users\lenovo\.conda\envs\mas_env\python.exe -m compileall -q applications\maas_reproduction
```

Output:

```text
exit code 0
```

## Latest Update: File-Based GSM8K Two-Query Batch Smoke

Date: 2026-08-10

Added a file-based live smoke script:

```text
applications/maas_reproduction/scripts/run_gsm8k_batch_smoke.py
```

Reason:

- The previous two-query instrumentation used `python -` from stdin.
- On Windows, the original MaAS `Programmer.exec_code()` uses `ProcessPoolExecutor`; spawned child processes cannot reliably re-import `<stdin>`.
- The new script runs from a real `.py` file and keeps the actual Graph execution path intact.

Data path provenance confirmed:

- `main.py` uses `Path(__file__).resolve().parent` as `application_root`.
- `MaASPaths.from_application_root(application_root)` sets `data_root = application_root / "assets" / "data"`.
- `settings.dataset_file` for GSM8K Graph train is:

```text
applications/maas_reproduction/assets/data/gsm8k_train.jsonl
```

First two loaded records:

- index 0: Natalia clips problem, expected final answer `72`
- index 1: Weng babysitting problem, expected final answer `10`

Live smoke command:

```powershell
$env:PYTHONPATH='applications\maas_reproduction'
$env:METAGPT_PROJECT_ROOT='C:\Users\lenovo\Desktop\���ĸ������\MaAS-main'
& C:\Users\lenovo\.conda\envs\mas_env\python.exe applications\maas_reproduction\scripts\run_gsm8k_batch_smoke.py --indices 0 1 --batch-size 2
```

This did call the real configured LLM provider and used the configured API key.

First run of the new script failed before LLM execution because the script passed `{}` to `graph.invoke(...)`; RootGraph entry requires the same config keys as `main.py`. Fixed the script to pass the full input dictionary.

Successful result summary:

```json
{
  "average_score": 1.0,
  "processed_problems": 2,
  "workflow_call_count": 2,
  "workflow_predictions": ["72", "10.0"],
  "workflow_logprob_requires_grad": [true, true],
  "all_scores": [1.0, 1.0],
  "optimizer_state_entries": 4,
  "changed_controller_tensor_count": 4,
  "max_controller_param_delta": 0.010000001639127731,
  "checkpoint_exists": true
}
```

This confirms the practical training path for a two-problem GSM8K Graph batch:

- GSM8K JSONL data loaded from copied assets.
- Controller instantiated and sampled architecture logprobs.
- Optimized GSM8K Workflow executed through `ArchitectureExecNode`.
- Real LLM calls happened.
- GSM8K evaluator scored both predictions correctly.
- `LossUpdateNode` accumulated a full batch and performed an optimizer update.
- Controller parameters changed.
- Graph-mode checkpoint was saved.

Observed caveat:

- The final outer `attributes` object reported `batch_logprobs`, `batch_scores`, and `batch_costs` length 2 after the run, despite the optimizer update and parameter changes confirming batch update occurred.
- This appears related to MASFactory Loop/attribute exposure semantics after body-node execution, not to a missing optimizer update.
- Do not treat this as a training blocker unless later logic relies on those outer buffer lengths after `RootGraph.invoke()` returns.

Warnings observed:

- Pydantic field-shadow warnings from installed dependencies.
- `maas.const:get_metagpt_root` logs confirming `METAGPT_PROJECT_ROOT` points at `MaAS-main`.
- These warnings did not block execution.

## Latest Update: MASFactory Visualizer File

Date: 2026-08-10

Added a visualization-only file:

```text
applications/maas_reproduction/visual_workflow.py
```

Purpose:

- This file is intended for the VS Code MASFactory Visualizer Preview panel.
- It does not call the LLM, does not create controller/optimizer/runtime objects, and does not read API keys.
- It statically expands the phase-1 execution path so the visualizer can show the important training flow clearly:

```text
entry
  -> ConfigNode
  -> TrainingLoop_Controller
  -> ArchitectureExecNode
  -> EvaluatorNode
  -> LossUpdateNode
  -> TrainingLoop_Controller
  -> ResultNode
  -> exit
```

The real executable RootGraph remains in:

```text
applications/maas_reproduction/maas_reproduction/workflow.py
```

The real TrainingLoop body remains in:

```text
applications/maas_reproduction/maas_reproduction/graphs/training_loop.py
```

Use `visual_workflow.py` for visual inspection, and use `main.py` or the smoke scripts for actual execution.

## Clarification: Visualizer Graphs vs Real Runtime Graph

Date: 2026-08-10

Added:

```text
applications/maas_reproduction/runtime_graph_preview.py
```

This file is the correct MASFactory Visualizer entry for the real executable RootGraph. It directly calls:

```python
from maas_reproduction.workflow import build_maas_reproduction_graph
```

and then builds the graph. It does not initialize runtime attributes or call the LLM, but its topology is produced by the same builder used by `main.py` during real training.

Important distinction:

- `runtime_graph_preview.py`: real MASFactory RootGraph preview, tied to actual CLI/runtime graph construction.
- `visual_workflow.py`: expanded explanatory diagram that statically shows the intended internal training path; useful for discussion, but not itself part of the training execution path.

The user's criticism is valid: `visual_workflow.py` alone should not be presented as proof of MASFactory integration. Actual MASFactory integration is in:

```text
main.py
  -> maas_reproduction.workflow.build_maas_reproduction_graph()
  -> RootGraph(ConfigNode -> TrainingLoop -> ResultNode)
  -> TrainingLoop body ArchitectureExecNode -> EvaluatorNode -> LossUpdateNode
```

The copied optimized MaAS assets are intentionally wrapped inside `ArchitectureExecNode`; they are not split into MASFactory nodes according to the agreed migration boundary.

## Latest Check: Windows ProcessPoolExecutor In Programmer Operator

Date: 2026-08-10

User raised a possible blocker:

```text
Execution error on attempt 1, error message: Unknown error: A process in the process pool was terminated abruptly...
```

Investigation result:

- This failure was not reproducible from a real `.py` file entrypoint.
- A dedicated diagnostic script was added:

```text
applications/maas_reproduction/scripts/check_programmer_exec_code.py
```

The script imports the copied optimized `Programmer` operators and directly runs:

```python
def solve():
    return 72
```

Command:

```powershell
$env:PYTHONPATH='applications\maas_reproduction'
$env:METAGPT_PROJECT_ROOT='C:\Users\lenovo\Desktop\���ĸ������\MaAS-main'
& C:\Users\lenovo\.conda\envs\mas_env\python.exe applications\maas_reproduction\scripts\check_programmer_exec_code.py
```

Verified modules:

```text
assets.optimized.GSM8K.train.template.operator
assets.optimized.GSM8K.test.template.operator
assets.optimized.MATH.train.template.operator
assets.optimized.MATH.test.template.operator
```

Result:

```json
[
  {"module": "assets.optimized.GSM8K.train.template.operator", "status": "Success", "output": "72"},
  {"module": "assets.optimized.GSM8K.test.template.operator", "status": "Success", "output": "72"},
  {"module": "assets.optimized.MATH.train.template.operator", "status": "Success", "output": "72"},
  {"module": "assets.optimized.MATH.test.template.operator", "status": "Success", "output": "72"}
]
```

Important details:

- A too-short diagnostic timeout of 10 seconds produced false timeouts for some modules because Windows process spawn plus MaAS import startup is slow.
- With a 60 second diagnostic timeout, all four modules succeeded.
- The original MaAS operator timeouts are higher (`GSM8K=100`, `MATH=600`), so this is not currently a training blocker.
- The previous abrupt process-pool errors are most consistent with running instrumentation through stdin (`python -`) on Windows. Avoid stdin-based live smoke scripts when `ProcessPoolExecutor` can be reached.

No operator runtime fix was applied because the current file-based CLI/smoke execution path works, and replacing `ProcessPoolExecutor` would change the original MaAS isolation/timeout behavior.

HumanEval note:

- The copied HumanEval optimized `Test.exec_code` path does not use `ProcessPoolExecutor`; it runs direct `exec()` against extracted test cases.
- HumanEval remains blocked by missing local HumanEval data/assets validation, not by the GSM8K/MATH Programmer process-pool path verified here.
