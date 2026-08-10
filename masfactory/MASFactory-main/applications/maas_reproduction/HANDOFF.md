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

3. Set conda env config var:

```text
METAGPT_PROJECT_ROOT = C:\Users\lenovo\Desktop\论文复现相关\MaAS-main
```

After normal `conda activate mas_env`, MaAS can locate `config/config2.yaml`. In Codex tool calls that invoke `python.exe` directly, still set `$env:METAGPT_PROJECT_ROOT` in the command because activation hooks are bypassed.

Known remaining environment notes:

1. `torch.cuda.is_available()` is `False`. Training can run on CPU, but controller sampling and full LLM evaluation will be slower. CPU is still the right baseline for golden reference determinism.

2. `pip check` still reports metadata conflicts because this environment contains both MaAS/MetaGPT and MASFactory dependencies. Important reported conflicts include:

```text
masfactory 1.0.4 wants newer openai/anthropic/numpy
metagpt 0.8.1 wants httpx==0.25.2 and websockets==11.0.3
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
applications\maas_reproduction\__init__.py
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
ConfigNode -> TrainingLoop -> ResultNode
```

`ConfigNode` pushes `settings` into RootGraph attributes so `TrainingLoop` can pull it. Runtime objects such as `problems`, `architecture_workflow`, `controller`, `optimizer`, and `operator_embeddings` still need to be supplied by the next runtime initialization stage.

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
..................................
----------------------------------------------------------------------
Ran 34 tests in 20.375s

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

Not completed:

1. CLI entry point
2. optimized architecture/operator/prompt migration
3. HumanEval sanitize equivalence
4. golden reference comparison
5. README.md
6. REPRODUCTION.md
7. requirements.txt

## Next Implementation Steps

Recommended next order:

1. Migrate optimized architecture files and operator/prompt assets needed by `ArchitectureExecNode`.
2. Implement `main.py` CLI that calls `build_runtime_attributes()` before invoking the MASFactory graph.
4. Run a one-query smoke test.
5. Run a one-batch smoke test.
6. Generate golden reference traces on CPU.
7. Add README and REPRODUCTION docs.

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

Fresh verification before writing this handoff:

```powershell
$env:PYTHONPATH='applications\maas_reproduction'; $env:METAGPT_PROJECT_ROOT='C:\Users\lenovo\Desktop\论文复现相关\MaAS-main'; & C:\Users\lenovo\.conda\envs\mas_env\python.exe -m unittest discover applications\maas_reproduction\tests
```

Output:

```text
..................................
----------------------------------------------------------------------
Ran 34 tests in 20.375s

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
