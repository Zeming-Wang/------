# DDO MASFactory Application

This directory is a MASFactory application for reproducing **DDO: Dual-Decision
Optimization for LLM-Based Medical Consultation via Multi-Agent Collaboration**.

The current reproduction target is not to rerun every baseline in the paper. Under
limited compute resources, this application compares:

```text
API-adapted original DDO source code
vs
MASFactory DDO application
```

Both sides use the same API model, currently `gpt-4o-mini`, so the main question
is whether the MASFactory implementation reaches the behavior and performance of
the API-adapted source implementation.

## Application Layout

```text
applications/ddo/
├── README.md
├── .env.example
├── main.py
├── pyproject.toml
├── configs/
│   └── config.yaml
├── ddo/
│   ├── graphs/
│   ├── runtime/
│   ├── services/
│   └── state_model/
└── docs/
    ├── design.md
    └── results_analysis.pdf
```

## Runtime Assets And Setup

This application has its own `pyproject.toml` because DDO needs extra runtime
dependencies such as `gymnasium` and `stable-baselines3`. It uses the MASFactory
repo root as an editable dependency:

```toml
masfactory = { path = "../..", editable = true }
```

From the MASFactory repo root:

```bash
cd applications/ddo
uv sync
```

Create a local `.env` from `.env.example` and fill in the API key:

```bash
OPENAI_API_KEY="your_api_key"
OPENAI_BASE_URL="https://api.openai.com/v1"
```

Experiments require the DXY, GMD, or CMD files under:

```text
data/
├── <DATASET>/
│   ├── test.json
│   ├── disease_corpurs.txt
│   ├── symptom_corpurs.txt
│   └── empirical_knowledge.json
└── turn_data/
    ├── DXY_train_turn.json
    ├── GMD_train_turn.json
    └── CMD_train_turn.json
```

Download the datasets from the original DDO repository:

- DDO repository: <https://github.com/zh-jia/DDO>
- Dataset directory: <https://github.com/zh-jia/DDO/tree/main/data>

The DDO workflow depends on the original RL policy assets for candidate symptom
generation. These assets are not included in this PR. Before running real
experiments, place both `best_settings.json` and `policy.pth` locally with the
original DDO layout:

```text
policy/<DATASET>/(custom_rl_training|curstom_rl_training)/
├── best_settings.json
└── policy.pth
```

The misspelled `curstom_rl_training` directory is supported for compatibility
with the source repository.

Download the policy checkpoints from the original DDO release assets:

- DDO README checkpoint table: <https://github.com/zh-jia/DDO>
- RL policy directory on Hugging Face: <https://huggingface.co/zhjia/DDO/tree/main/policy>

The default config is `configs/config.yaml`. Change `dataset.name` to `DXY`,
`GMD`, or `CMD` after placing the matching local data and policy assets.

The `general.device` option only controls the local RL policy network used for
candidate symptom generation. It does not affect API-based LLM calls. The default
is `cpu` for portability; use `cuda` only with a compatible CUDA environment.

The confidence-estimation agent requests token logprobs from the selected API
model. Choose a provider/model endpoint that supports `logprobs` and
`top_logprobs`; otherwise confidence estimation will frequently fall back to
default values and the reproduction quality may degrade.

## Run

Run the application:

```bash
cd applications/ddo
uv run python main.py
```

## Current Result Summary

With `gpt-4o-mini` as the shared API model, the MASFactory implementation reaches
the API-adapted source implementation on the main reproduction comparison. 

| Method | LLM | DXY Acc_init | DXY Acc | DXY Avg.n | GMD Acc_init | GMD Acc | GMD Avg.n | CMD Acc_init | CMD Acc | CMD Avg.n |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Paper DDO | Qwen2.5-14B-Instruct | 66.30 | 94.20 | 10.00 | 67.80 | 80.30 | 9.80 | 65.30 | 68.60 | 10.00 |
| Paper DDO | Qwen2.5-7B-Instruct | 66.30 | 87.50 | 9.90 | 69.50 | 79.50 | 9.60 | 60.60 | 63.60 | 10.00 |
| Paper DDO | Qwen2.5-7B-Instruct w/o adapter | 66.50 | 86.50 | -- | 63.60 | 78.70 | -- | 54.20 | 54.20 | -- |
| DDO source API | gpt-4o-mini | 65.00 | 93.20 | 10.00 | 62.80 | 79.10 | 9.80 | 62.90 | 61.10 | 10.00 |
| DDO_MASFactory | gpt-4o-mini | 64.10 | 98.10 | 10.00 | 63.60 | 78.70 | 9.80 | 62.90 | 63.00 | 10.00 |

The intended conclusion is conservative: under the same API-model setting, the
MASFactory application basically reproduces the core DDO consultation workflow
and reaches the API-adapted source implementation.
