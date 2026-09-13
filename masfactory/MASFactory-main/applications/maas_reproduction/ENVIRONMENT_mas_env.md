# mas_env 运行环境记录

记录日期：2026-09-13

## Conda

```text
环境名：mas_env
Python：3.10.20
环境路径：C:\Users\lenovo\.conda\envs\mas_env
```

## 关键依赖

```text
torch==2.1.0+cu118
numpy==1.26.4
openai==1.39.0
sentence-transformers==2.2.2
pytest==8.0.2
```

CUDA 在当前机器上不可用，PyTorch 使用 CPU：

```text
torch.cuda.is_available() = False
```

## 修复记录

`conda-meta/state` 原先包含以非 UTF-8 编码保存的旧中文路径和废弃的
`METAGPT_PROJECT_ROOT` 环境变量，导致 `conda run -n mas_env` 激活失败。
该文件已改为 UTF-8 编码的空 JSON 对象 `{}`。项目运行不再依赖
`METAGPT_PROJECT_ROOT`。

## 推荐运行方式

在 MASFactory 根目录执行：

```powershell
$env:PYTHONPATH = (Get-Location).Path
conda run -n mas_env python -m pytest -q --import-mode=importlib applications/maas_reproduction/tests
```

## 本次验证

使用该环境的绝对解释器执行 `compileall` 成功；FakeModel 的 GSM8K
`smoke` 和 `train` 入口均可运行。修复 Loop 内 ProgrammerLogicSwitch 的
隐式 attribute 继承后，原有测试为 `62 passed, 3 failed`；失败的 3 个旧测试
仍调用已删除的 `architecture_graph` 兼容入口，属于与冻结规范冲突的旧测试，
需要改写为原生 ArchitectureExecGraph 测试。

FakeModel 训练还验证了 batch 边界 checkpoint，运行产物位于
`assets/output/checkpoints/checkpoint.pt`；生产代码不会把 API key 或
`expected_answer` 写入 checkpoint。
