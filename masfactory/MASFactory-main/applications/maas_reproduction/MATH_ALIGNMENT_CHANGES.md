# MATH 判分与提示词对齐说明

## 目标与边界

本次修改将 `maas_reproduction` 的 MATH 最终答案格式约定与原版 `MaAS-main` 对齐，并在原版判分逻辑失败后增加保守的等价答案回退判定。

修改遵循以下原则：

- 先完整执行现有/原版 MaAS 判分路径，只在其返回不等时才使用新回退逻辑。
- 不改变任何已经判对的 `maas_reproduction` 结果。
- 不自动覆盖旧的 JSON/CSV 运行结果。
- 不修改 `.env`、`config2.yaml`、Controller 权重、bundle 或数据集。
- 不将任意推理文本中的数字当作答案；只接受 `Answer:` / `Final answer:` 等显式最终答案标记。

## 发现的不同步点

1. 原版 MaAS 的 MATH 提示词强调最终答案使用 `\boxed{}`；`maas_reproduction` 的 MATH 专用提示词目录原先不存在，运行时回退到较弱的 shared/default 提示词。
2. `BootstrapGenerate` 原先使用函数内硬编码指令，没有经过 MATH 提示词加载器。
3. `MultiGenerateCoT` 原先复用 `GenerateCoT` 的 adapter，因此其自身的提示词无法生效。
4. 原版与复现版的基础判分器都偏向从 `\boxed{}` 提取答案，对已确认等价的输出格式会产生漏判。

## 修改文件

### `maas_reproduction/benchmarks/math.py`

- 将现有比较路径抽取为 `_source_equal`，并保持原有顺序：字符串相等、数值近似、SymPy 符号等价、SymPy 数值近似。
- 只在上述路径失败后，处理以下已确认的等价格式：
  - `Final answer:` / `Answer:` 显式标签及其 Markdown 强调符。
  - `x=...` 等单个变量赋值前缀。
  - `\dfrac` 与 `\frac`。
  - `\sqrt3` 与 `\sqrt{3}` 类根式，避免 SymPy 对缺失花括号的根式静默误解析。
  - 度数后缀 `^\circ`。
  - 答案末尾的 `\text{...}` / `\mathrm{...}` 单位。
  - 仅在题目或参考答案明确包含 percent/% 语义时，处理 `20%` 与“20 percent”这类百分点表达。

### `assets/prompts/math/*.txt`

新增 MATH 专用提示词：

- `Generate.txt`
- `GenerateCoT.txt`
- `MultiGenerateCoT.txt`
- `SelfRefine.txt`
- `BootstrapGenerate.txt`

这些提示词要求最终答案严格放在一个 `\boxed{}` 中，box 内不放变量赋值、单位、解释或 Markdown，根式使用完整花括号。

### `runtime/bootstrap.py` 和 `runtime/prompt_loader.py`

- `BootstrapGenerate` 改为通过 `PromptLoader` 读取数据集专用提示词。
- `MultiGenerateCoT` 改为使用它自己的 adapter 和提示词。
- 为 `BootstrapGenerate` 增加默认提示词，保证其他数据集没有专用文件时仍可运行。

### `tests/unit/test_task8_evaluator_training.py`

- 增加本次真实运行中 10 个漏判样例的回归测试。
- 增加 3 个真实错误答案仍必须判错的负例测试。
- 增加 5 类 MATH 提示词必须包含 `\boxed` 格式约定的测试。

## 真实错误回放结果

回放来源：

`applications/maas_reproduction/assets/output/run_20260929_024628_725209/results/`

未修改任何原始结果文件，仅将其 prediction 和原始 MATH test 参考答案送入修改后的判分器：

| 回放类别 | 数量 | problem index |
| --- | ---: | --- |
| 原本 `score=1`，修改后仍为 1 | 14 | 0, 1, 4, 9, 10, 14, 15, 16, 23, 26, 27, 28, 29, 32 |
| 原本 `score=1`，修改后回退 | 0 | 无 |
| 已确认漏判，修改后由 0 变为 1 | 10 | 2, 3, 5, 6, 8, 12, 25, 30, 33, 34 |
| 真实答错，修改后仍为 0 | 4 | 7, 11, 13, 17 |
| 原运行无效，不参与判分 | 7 | 18, 19, 20, 21, 22, 24, 31 |

如果只对已有 28 条有效结果重新判分，正确数从 14 变为 24；这不会自动改写原运行统计。

## 测试状态

- 目标回归测试：`22 passed`。
- 服务器 `mas_env` 完成三个修改模块的 `py_compile` 检查。
- 服务器 `mas_env` 未安装 `pytest`，因此服务器端使用等价断言脚本完成真实结果回放，没有为测试改动环境。
- 本地全部 `applications/maas_reproduction/tests` 结果为 `100 passed`；另有 3 个与本次修改无关的旧 `_build_root` 接口断言失败，以及 24 个 Windows pytest 临时目录权限错误。

## 后续全量测试

新的提示词只会对修改后新启动的运行生效。已停止的 `run_20260929_024628_725209` 不应继续作为新旧逻辑混合的全量结果，建议使用原来的 bundle 新建一次运行。
