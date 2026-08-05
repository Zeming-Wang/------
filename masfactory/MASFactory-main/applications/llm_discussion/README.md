# LLM Discussion MASFactory Application

This directory is a MASFactory application for reproducing **LLM Discussion: Enhancing the Creativity of Large Language Models via Discussion Framework and Role-Play**.

## Overview

We keep the original four creativity tasks:

- `AUT`
- `Scientific`
- `Instances`
- `Similarities`

The reproduction rewrites the multi-agent discussion process with MASFactory while preserving the original role-play setting, multi-round discussion flow, output format, and evaluation pipeline.

## Application Layout

```text
.
├─ README.md
├─ .env.example
├─ requirements.txt
├─ Datasets/
├─ Evaluation/
├─ Experiments/
│  └─ multi_agent/
│     ├─ llm_discussion.py
│     ├─ discussion.py
│     ├─ graph_nodes.py
│     ├─ sample_graph.py
│     └─ config_role_gpt4o_mini.json
└─ Results/
```

The default setting uses `4` agents, `5` discussion rounds, and `gpt-4o-mini`.

## Environment

From the application root:

```bash
pip install -r requirements.txt
```

Create `.env` in the same directory as this README. A minimal configuration is:

```bash
OPENAI_API_KEY=your_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
```

You can start from `.env.example`.

Both `Experiments/multi_agent/llm_discussion.py` and `Evaluation/auto_grade_final.py` automatically search upward for the nearest `.env`, so the same setup remains valid after this application is moved under `MASFactory/applications/`.

## Documentation

- [design](docs/design.md): design choices, MASFactory graph mapping, and reproduction boundaries.
- [result](docs/result.md): how to read the reproduced metrics and compare them with the source implementation.

## Data

Datasets are stored in:

- `Datasets/AUT`
- `Datasets/Scientific`
- `Datasets/Instances`
- `Datasets/Similarities`

Each task provides files such as:

- `*_example.json`
- `*_3.json`
- `*_10.json`
- `*_30_test.json`

## Run

Run experiments from the application root.

Quick smoke run:

```bash
python Experiments/multi_agent/llm_discussion.py -c Experiments/multi_agent/config_role_gpt4o_mini.json -d Datasets/Scientific/scientific_example.json -r 1 -t Scientific -p 1 -e
```

Standard example run:

```bash
python Experiments/multi_agent/llm_discussion.py -c Experiments/multi_agent/config_role_gpt4o_mini.json -d Datasets/Scientific/scientific_example.json -r 5 -t Scientific -p 1 -e
```

30-sample test:

```bash
python Experiments/multi_agent/llm_discussion.py -c Experiments/multi_agent/config_role_gpt4o_mini.json -d Datasets/AUT/aut_30_test.json -r 5 -t AUT -p 1 -e
python Experiments/multi_agent/llm_discussion.py -c Experiments/multi_agent/config_role_gpt4o_mini.json -d Datasets/Scientific/scientific_30_test.json -r 5 -t Scientific -p 1 -e
python Experiments/multi_agent/llm_discussion.py -c Experiments/multi_agent/config_role_gpt4o_mini.json -d Datasets/Instances/instances_30_test.json -r 5 -t Instances -p 1 -e
python Experiments/multi_agent/llm_discussion.py -c Experiments/multi_agent/config_role_gpt4o_mini.json -d Datasets/Similarities/similarities_30_test.json -r 5 -t Similarities -p 1 -e
```

Main arguments:

- `-c`: agent configuration file
- `-d`: dataset path
- `-r`: discussion rounds
- `-t`: task type
- `-p`: prompt version
- `-e`: run evaluation after generation

## Reproduction Results

The following tables summarize our reproduced results for sample sizes `1`, `3`, and `30`. README only keeps the compact comparison needed for the reproduction PR.

### Data Size = 1

| Task         | Project | Model       | Fluency Mean | Fluency Std | Flexibility Mean | Flexibility Std | Originality Mean | Originality Std | Elaboration Mean | Elaboration Std | Timestamp           |
| ------------ | ------- | ----------- | -----------: | ----------: | ---------------: | --------------: | ---------------: | --------------: | ---------------: | --------------: | ------------------- |
| AUT          | llm_d   | gpt-4o-mini |       37.500 |       4.330 |            9.250 |           0.363 |            3.577 |           0.049 |            2.638 |           0.185 | 2026-06-01 10:34:38 |
| AUT          | masf    | gpt-4o-mini |       39.083 |       7.057 |            9.333 |           0.527 |            3.539 |           0.097 |            2.556 |           0.467 | 2026-06-01 10:24:49 |
| Scientific   | llm_d   | gpt-4o-mini |       20.250 |       1.090 |            8.417 |           0.595 |            3.479 |           0.109 |            2.000 |           0.061 | 2026-06-01 10:47:50 |
| Scientific   | masf    | gpt-4o-mini |       21.333 |       2.483 |           11.417 |           3.003 |            3.260 |           0.161 |            1.868 |           0.073 | 2026-06-01 10:54:05 |
| Instances    | llm_d   | gpt-4o-mini |       33.917 |       3.244 |           10.167 |           0.726 |            2.579 |           0.130 |            1.772 |           0.110 | 2026-06-01 11:01:22 |
| Instances    | masf    | gpt-4o-mini |       40.000 |       1.650 |           10.750 |           1.115 |            2.647 |           0.072 |            1.611 |           0.069 | 2026-06-01 11:09:14 |
| Similarities | llm_d   | gpt-4o-mini |       20.000 |       0.000 |           19.167 |           1.443 |            2.137 |           0.054 |            1.900 |           0.035 | 2026-06-01 11:20:03 |
| Similarities | masf    | gpt-4o-mini |       22.000 |       1.414 |           22.000 |           1.414 |            2.022 |           0.050 |            1.734 |           0.084 | 2026-06-01 11:32:43 |

### Data Size = 3

| Task         | Project | Model       | Fluency Mean | Fluency Std | Flexibility Mean | Flexibility Std | Originality Mean | Originality Std | Elaboration Mean | Elaboration Std | Timestamp                      |
| ------------ | ------- | ----------- | -----------: | ----------: | ---------------: | --------------: | ---------------: | --------------: | ---------------: | --------------: | ------------------------------ |
| AUT          | llm_d   | -           |            - |           - |                - |               - |                - |               - |                - |               - | No result for this sample size |
| AUT          | masf    | -           |            - |           - |                - |               - |                - |               - |                - |               - | No result for this sample size |
| Scientific   | llm_d   | gpt-4o-mini |       26.083 |       3.499 |            9.750 |           1.428 |            3.164 |           0.166 |            2.035 |           0.083 | 2026-06-01 14:47:36            |
| Scientific   | masf    | gpt-4o-mini |       24.139 |       3.096 |            9.500 |           1.782 |            3.107 |           0.174 |            2.014 |           0.144 | 2026-06-01 15:29:56            |
| Instances    | llm_d   | gpt-4o-mini |       26.583 |      13.128 |           10.861 |           1.134 |            2.358 |           0.287 |            1.456 |           0.352 | 2026-06-01 15:39:46            |
| Instances    | masf    | gpt-4o-mini |       39.194 |       6.990 |           10.222 |           1.343 |            2.465 |           0.133 |            1.614 |           0.178 | 2026-06-01 16:21:50            |
| Similarities | llm_d   | gpt-4o-mini |       20.833 |       1.772 |           20.361 |           1.424 |            2.117 |           0.138 |            1.888 |           0.098 | 2026-06-01 16:32:47            |
| Similarities | masf    | gpt-4o-mini |       21.417 |       1.891 |           21.306 |           1.761 |            2.097 |           0.145 |            1.877 |           0.122 | 2026-06-01 17:48:37            |

### Data Size = 30

| Task         | Project | Model       | Fluency Mean | Fluency Std | Flexibility Mean | Flexibility Std | Originality Mean | Originality Std | Elaboration Mean | Elaboration Std | Timestamp           |
| ------------ | ------- | ----------- | -----------: | ----------: | ---------------: | --------------: | ---------------: | --------------: | ---------------: | --------------: | ------------------- |
| AUT          | llm_d   | gpt-4o-mini |       36.036 |       6.677 |            9.675 |           1.119 |            3.718 |           0.330 |            2.728 |           0.376 | 2026-06-02 10:24:47 |
| AUT          | masf    | gpt-4o-mini |       36.453 |       7.328 |            9.628 |           1.220 |            3.762 |           0.325 |            2.786 |           0.392 | 2026-06-02 10:24:50 |
| Scientific   | llm_d   | gpt-4o-mini |       18.828 |       7.400 |           12.628 |           4.247 |            2.628 |           0.613 |            1.680 |           0.446 | 2026-06-02 11:02:12 |
| Scientific   | masf    | gpt-4o-mini |       18.675 |       6.912 |           12.492 |           4.158 |            2.660 |           0.716 |            1.768 |           0.439 | 2026-06-02 11:16:43 |
| Instances    | llm_d   | gpt-4o-mini |       32.189 |      14.940 |           13.308 |           5.316 |            2.395 |           0.414 |            1.459 |           0.409 | 2026-06-02 11:28:52 |
| Instances    | masf    | gpt-4o-mini |       34.161 |      14.010 |           14.228 |           6.202 |            2.357 |           0.334 |            1.447 |           0.357 | 2026-06-02 11:43:02 |
| Similarities | llm_d   | gpt-4o-mini |       20.428 |       2.261 |           19.833 |           2.202 |            2.109 |           0.212 |            1.813 |           0.136 | 2026-06-02 12:01:24 |
| Similarities | masf    | gpt-4o-mini |       20.519 |       2.483 |           19.964 |           2.472 |            2.100 |           0.193 |            1.814 |           0.151 | 2026-06-02 12:15:52 |

## Output

Results are written to `Results/`:

- `Results/<Task>/chat_log/`: discussion logs
- `Results/<Task>/init/`: initial responses
- `Results/<Task>/Output/multi_agent/`: final generated outputs
- `Results/<Task>/Eval_Result/multi_agent/`: evaluation JSON files
- `Results/LeaderBoard/LeaderBoard-<Task>.csv`: final metric summary

## References

- Original paper: [LLM Discussion](https://arxiv.org/abs/2405.06373)
- Framework: [MASFactory](https://github.com/BUPT-GAMMA/MASFactory)
