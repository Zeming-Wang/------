# maas_legacy2 运行环境记录

记录日期：2026-09-22

## 环境名称确认

目标环境名称是 maas_legacy2。依据如下：

- MaAS-main/original_prompt_question11_20260921_221614.log 使用了
  C:/Users/lenovo/.conda/envs/maas_legacy2。
- 本机 Conda 记录显示该环境是在 2026-09-19 由 mas_env clone 创建的。
- maas_legacy 虽然也存在，但只有 Python 3.10 和少量基础包，不是当前复现所用环境。

## 当前快照

```text
环境名：maas_legacy2
Python：3.10.20
环境路径：C:/Users/lenovo/.conda/envs/maas_legacy2
torch==2.1.0+cu118
numpy==1.26.4
openai==1.97.0
sentence-transformers==2.2.2
pytest==8.0.2
masfactory==1.0.4
metagpt==0.8.1
tiktoken==0.12.0
```

该环境与 mas_env 的已安装发行包逐项核对一致，共 314 个发行包。
对应的 requirements 文件是 requirements_maas_legacy2.lock.txt，它引用同目录下的
requirements_mas_env.lock.txt，避免维护两份完全相同的锁定清单。

## 服务器安装

在仓库根目录执行：

```bash
conda create -n maas_legacy2 python=3.10.20 -y
conda activate maas_legacy2
pip install -r masfactory/MASFactory-main/applications/maas_reproduction/requirements_maas_legacy2.lock.txt
pip install -e MaAS-main --no-deps
```

如果需要同时导入 MASFactory 原生复现代码，再执行：

```bash
pip install -e masfactory/MASFactory-main --no-deps
```

## 运行边界

- mas_env：当前 MASFactory 原生 MaAS 复现，入口位于
  masfactory/MASFactory-main/applications/maas_reproduction。
- maas_legacy2：旧版 MaAS-main 代码及其原始训练/测试入口。
- 两个环境虽然依赖快照相同，但建议按上述用途分开创建，避免 editable 安装的源码入口互相覆盖。

## 与其他环境的差异核对

maas_legacy2 与 mas_env 当前逐项一致：Python 都是 3.10.20，pip 都是 26.1.2，
packaging 都是 24.2，核心依赖也完全一致。

容易混淆的 maas_legacy 并不是这次 clone 出来的目标环境。它是后来单独创建的基础环境，
其中 Python 升到了 3.10.21，pip 升到了 26.2.1，packaging 升到了 26.3，且没有本项目所需的
torch、numpy、openai 等依赖。

## Pydantic 版本说明

MaAS-main/requirements.txt 显式锁定 pydantic==2.6.4 和 pydantic_core==2.16.3；
MASFactory 根目录的 requirements.txt 则是另一条较新的依赖线，锁定 pydantic==2.11.7
和 pydantic-core==2.33.2。pydantic_core 与 pydantic-core 只是 pip 的命名规范化差异。

本次两个实际环境都使用 pydantic==2.6.4、pydantic-core==2.16.3；服务器要复现当前结果，
以 requirements_mas_env.lock.txt 为准，不要再叠加安装 MASFactory 根目录 requirements.txt。
