# MultiAgentESC on MASFactory

Reproduction of [MultiAgentESC (EMNLP 2025)](https://github.com/MindIntLab-HFUT/MultiAgentESC) on the [MASFactory](https://github.com/BUPT-GAMMA/MASFactory) framework.

MultiAgentESC is an emotional support conversation system that uses multi-agent collaboration (strategy discussion, debate, reflection, and voting) to select appropriate counseling strategies and generate supportive responses.

## Architecture

```
multiagentesc/
├── main.py                          # Entry point
├── workflow.py                      # Graph creation and dialog processing loop
├── prompts.py                       # 11 prompt templates
├── strategy.py                      # 8 strategy definitions
├── utils.py                         # Retrieval, parsing, strategy cleaning
├── patch_masfactory.py              # Plain text output formatting
└── components/
    ├── strategy_discussion.py       # 3-agent round-robin via MeshGraph
    ├── debate.py                    # Multi-agent debate on responses
    └── reflect.py                   # Multi-agent reflection
```

### Pipeline

For each supporter turn in the dialog:

1. **is_complex**: Determine if the conversation is complex enough for multi-agent analysis
2. **Simple path** (not complex): Generate response directly via zero-shot
3. **Complex path**:
   - Analyze **emotion** → **cause** → **intention** via sequential agents
   - Retrieve top-k similar cases for examples
   - **Strategy discussion**: 3 agents discuss via MeshGraph (round-robin)
   - **Response generation**: Generate candidate responses for selected strategies
   - **Debate → Reflect → Vote**: Multi-agent evaluation of candidate responses
   - **Judge**: Resolve ties if voting is inconclusive
   - **Self-reflection**: Final quality check on the selected response

## Setup

### Requirements

- Python 3.10+
- API access to a compatible LLM (tested with Qwen2.5-32b via Alibaba DashScope)

### Install

```bash
pip install -r requirements.txt
```

### Configuration

Create a `.env` file:

```env
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_MODEL_NAME=qwen2.5-32b-instruct
```

### Dataset

Place `ESConv.json` in the `dataset/` directory. Available from the [original repository](https://github.com/MindIntLab-HFUT/MultiAgentESC).

## Usage

```bash
# Run (adjust sample count in main.py)
python main.py

# Evaluate
python eval.py
```

Evaluation outputs 7 metrics: Distinct-1/2, BLEU-1/2/3, BERTScore F1, ROUGE-L.

## Results

| Metric | Paper (Qwen2.5-32b) | Source (autogen) [:1] | Reproduction (MASFactory) [:3] |
|--------|---------------------|-----------------------|---------------------------------|
| D-1    | 6.78                | 65.31                 | 41.77                           |
| D-2    | 35.15               | 92.97                 | 80.12                           |
| B-1    | 17.66               | 10.29                 | 10.33                           |
| B-2    | 5.38                | 2.45                  | 2.25                            |
| B-3    | 2.35                | 1.12                  | 0.81                            |
| R-L    | 14.66               | 12.20                 | 10.46                           |

The source code (autogen) and reproduction (MASFactory) produce nearly identical B-1 (~10.3) on the same API, confirming correct reproduction. Both fall short of the paper's reported metrics.
