# mas_env 运行环境记录

记录日期：2026-09-22（根据本机环境重新核对）

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
openai==1.97.0
sentence-transformers==2.2.2
pytest==8.0.2
masfactory==1.0.4
metagpt==0.8.1
tiktoken==0.12.0
```

完整的第三方依赖锁定快照见
requirements_mas_env.lock.txt。其中 masfactory 和 metagpt 两个本地源码包未写入
锁定文件，服务器上应从当前仓库以 editable 方式安装。

## 服务器安装

在仓库根目录执行：

```bash
conda create -n mas_env python=3.10.20 -y
conda activate mas_env
pip install -r masfactory/MASFactory-main/applications/maas_reproduction/requirements_mas_env.lock.txt
pip install -e masfactory/MASFactory-main --no-deps
```

如果还要运行旧版 MaAS 源码，再执行：

```bash
pip install -e MaAS-main --no-deps
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

注意：不要使用 maas_legacy 代替 maas_legacy2。maas_legacy 的 Python 是 3.10.21，
而目标复现环境固定为 Python 3.10.20；maas_legacy 也没有完整的复现依赖。

Pydantic 方面，当前 mas_env 实际使用 pydantic==2.6.4、pydantic-core==2.16.3；
服务器复现应使用本目录的 requirements_mas_env.lock.txt。MASFactory 根目录 requirements.txt
属于较新的依赖线（pydantic==2.11.7、pydantic-core==2.33.2），不要在锁定环境上再次叠加安装。
