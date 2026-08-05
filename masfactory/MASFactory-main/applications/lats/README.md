# LATS (Language Agent Tree Search) – HumanEval on MASFactory

This directory is a [MASFactory](https://github.com/BUPT-GAMMA/MASFactory) application that reproduces **LATS** (Language Agent Tree Search) on the **HumanEval** (programming) benchmark.

- **Paper**: [Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models](https://arxiv.org/abs/2310.04406) (ICML 2024)
- **Upstream reference**: [LanguageAgentTreeSearch](https://github.com/andyz245/LanguageAgentTreeSearch) (programming / HumanEval)

## Layout

```
applications/lats/
├── main.py                 # Entry: argparse, load dataset, build graph, run loop, tee to log
├── README.md
├── assets/
│   └── config/             # Config (default dataset path, etc.); datasets not in repo
│       └── defaults.json
├── workflows/              # Graph and controller
│   ├── graph.py            # RootGraph, LATSTemplate (Loop with LLM_Agent, Executor, Reflection), run_one_problem
│   └── controller.py     # lats_controller_logic (MCTS select / expand / backprop / terminate)
├── components/
│   ├── agents.py           # run_humaneval_forward, set_print_code_attempts (no custom Agent/CustomNode classes)
│   └── tree.py             # LATSNode, TreeManager, gather_context_from_tree
├── humaneval/              # HumanEval data and execution
│   ├── load.py             # load_humaneval_jsonl, parse_internal_tests_from_test, extract_python_code
│   ├── executor.py         # run_internal_tests, full_evaluate, verify_evaluation
│   └── timeout_utils.py   # function_with_timeout
└── utils/
    └── tee.py              # Tee output to terminal and optional log file
```

## Design

- **No custom Agent or CustomNode subclasses.** The graph uses only MASFactory’s built-in **Agent** and **CustomNode** via **NodeTemplate**:
  - **LLM_Agent**: `NodeTemplate(Agent, model=..., instructions=..., formatters=[ParagraphMessageFormatter(), TwinsFieldTextFormatter()])`
  - **Executor**: `NodeTemplate(CustomNode, forward=run_humaneval_forward, pull_keys=..., push_keys=...)`
  - **Reflection**: `NodeTemplate(Agent, model=..., instructions=..., formatters=[...])`
- **Messaging** is done via **edges** and Loop **pull_keys** (no Agent/MessageFormatter overloading for pass-through). Controller sends `reflexion_prompt` to LLM; controller sends `problem` and `internal_tests` to Executor; LLM sends `content` to Executor; Executor sends `action`/`observation` to Reflection and `action`/`observation`/`reward`/`full_passed` to controller; Reflection sends only `reflection` to controller.
- **Executor** and **Reflection** are separate nodes (executor logic in `run_humaneval_forward`; reflection is a plain Agent). NodeTemplates are wrapped in **Shared(...)** so the Loop config can be cloned without deepcopying them.

## Context and memory in this port

In the LATS paper, **context** and **memory** are conceptual. In this app they are not separate nodes:

- **Context**: Built inside the controller and sent to the LLM as `reflexion_prompt` (previous attempts, test results, reflections). See `workflows/controller.py` and `gather_context_from_tree` in `components/tree.py`.
- **Memory**: The search tree (`LATSNode`, `TreeManager` in `components/tree.py`) held by the controller; it is updated each loop (selection, backprop, new children).

## Setup

From the MASFactory repo root:

```bash
pip install masfactory openai
```

Optional: set default dataset in `assets/config/defaults.json` (`"dataset_path": "path/to/HumanEval.jsonl.gz"`).

Environment variables:

| Variable | Description |
|----------|-------------|
| **OPENAI_API_KEY** | Required |
| **OPENAI_API_BASE** | Optional (proxy/custom endpoint) |
| **LATS_MODEL** | Optional, default `gpt-4` |
| **LATS_MAX_ITERS** | Optional, default `8` |
| **NUMBER_OF_TESTS** | Optional, default `2` |
| **MASFACTORY_VISUALIZER_PORT** | Optional, for runtime view |

Example (PowerShell):

```powershell
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_API_BASE="https://your-endpoint/v1/"
$env:LATS_MODEL="你的模型"
...
```

## Run

From the **MASFactory repo root**:

```bash
python applications/lats/main.py --dataset "path/to/HumanEval.jsonl.gz" --log logs/lats.log
```

Or from **applications/lats**:

```bash
python main.py --dataset "path/to/HumanEval.jsonl.gz" --log logs/lats.log
```

Examples:

```bash
# Limit to 5 problems
python applications/lats/main.py --dataset "path/to/HumanEval.jsonl.gz" --limit 5 --log logs/lats.log

# Paper-aligned: max_iters=8, number_of_tests=2 (defaults)
python applications/lats/main.py --dataset "path/to/HumanEval.jsonl.gz" --log logs/lats.log
```

Output is printed to the terminal and, when `--log` is set, appended to the given file.

## Metrics

- **Pass@1**: fraction of problems for which the best solution passes the full HumanEval test.
- Defaults match the upstream GPT-4 setup: `max_iters=8`, `number_of_tests=2`.
