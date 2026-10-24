# Git 状态检查与分阶段提交记录

> 检查时间：2026-10-24（Asia/Shanghai）
> 分支：`main`；检查时 HEAD：`3e61b08 fix: clarify A-C1-HARD path in train-v3`
> 性质：只读检查 + 提交前整理；不改写历史、不 force push。

## 1. 检查时的仓库状态

| 指标 | 数值 |
| --- | ---: |
| 已跟踪文件 | 79 |
| 已修改（未暂存） | 10 |
| 未跟踪文件 | 129 |
| `git status --porcelain` 总条目 | 139 |

已修改文件（10）：

- `.env.example`
- `README.md`
- `docs/benchmark.md`
- `docs/evidence.md`
- `scripts/constraint_experiment.py`
- `src/agent/core/llm.py`
- `src/agent/tools/base.py`
- `src/agent/tools/registry.py`
- `tests/test_llm.py`
- `tests/test_registry.py`

未跟踪文件集中在 `docs/evidence/`、`docs/plan/`、`scripts/`、`tests/`、`training/`，均为已完成的评测、审计、训练管线与 Kaggle 执行代码/文档，不含数据文件。

## 2. 大文件检查

| 检查项 | 结果 |
| --- | --- |
| 已跟踪文件 >10 MB | 0 |
| 已跟踪文件最大体积 | `scripts/benchmark.py`，66,829 bytes |
| 未跟踪文件 >1 MB | 0 |
| `private/` 下训练数据 / adapter / Kaggle 输出 | 已被 `.gitignore` 覆盖，未进入暂存区 |

结论：当前没有大文件误入 Git；训练数据、adapter、trace、Kaggle 拉取输出都在 `private/`，只以 manifest + SHA-256 记录版本。

## 3. `.gitignore` 覆盖确认

| 类型 | 规则 | 检查前 | 本次处理 |
| --- | --- | --- | --- |
| 私有数据 / 输出 | `private/` | 已覆盖 | 保持不变 |
| JSONL（rollout / 训练数据） | `*.jsonl` + `!docs/evidence/**/*.jsonl` | 未覆盖 | 新增；证据表显式放行 |
| safetensors adapter | `*.safetensors` | 未覆盖 | 新增 |
| PyTorch 权重 | `*.pt` / `*.pth` / `*.bin` | 未覆盖 | 新增 |
| Python wheel | `*.whl` | 未覆盖 | 新增 |
| trace 目录 | `traces/` | 未覆盖 | 新增 |
| Kaggle 拉取目录 | `kaggle_pull_*/` | 未覆盖 | 新增 |
| 临时目录 | `tmp_*/` | 未覆盖 | 新增 |
| wheel 解包目录 | `**/wheel/` | 未覆盖 | 新增 |
| 环境变量 / 密钥 | `.env`、`.env.*`、`!.env.example` | 已覆盖 | 保持不变 |

验证方式：`git check-ignore -v` 对上述模式逐项命中；`docs/evidence/benchmark_runs.jsonl` 等小体积证据表仍可入库。

## 4. 密钥与绝对路径扫描

- 密钥模式：未发现真实 API key / token / 密码。`tests/test_trace_utils.py` 中的 `sk-...`、`password=...` 是脱敏测试的假数据，不是凭据。
- 本机绝对路径：检查前命中 4 个文件，已脱敏：
  - `docs/evidence.md`：仓库绝对路径 → `<repo>`；
  - `docs/evidence/CTX_HARD_token审计.md`：HF 缓存绝对路径 → `$env:HF_HOME\hub\...`；
  - `docs/evidence/补数据决策.md`：复现命令中的 HF 缓存绝对路径 → `$env:HF_HOME\hub\...`；
  - `docs/evidence/ctx_hard_token_audit.json`：`tokenizer_path` → `<HF_HOME>\hub\...`。
- `tests/test_trace_utils.py` 保留一个合成的 Windows 绝对路径作为“路径脱敏”测试输入，不是本机真实路径。
- `docs/evidence/benchmark修正回归.md` 中原先用于描述扫描模式的本机路径片段已改为 `<repo>` / `<home>`。

## 5. 提交拆分与 tag

由于 0.0 之前的评测证据、训练管线和采样元数据一直未入库，本次先补一个历史归档 commit，再提交 0.0 与 0.0a 两个阶段；从 0.0 起每个阶段验收后单独 commit + tag。

| 顺序 | commit message | 内容 | tag |
| --- | --- | --- | --- |
| 1 | `stage-0.0-pre: 归档 0.0 之前的评测证据、训练管线与采样元数据` | 历史欠账：benchmark / 冲突 / 统一数据集 / 7B / SFT 证据、`training/`、`scripts/`、`tests/`、采样与结构化错误元数据 | 无 |
| 2 | `stage-0.0: 数据多样性审计完成，判定补数据` | `scripts/audit_data_diversity.py`、审计测试与四维分布文档、`docs/plan/训练侧执行计划.md`、本报告 | `stage-0.0-audit` |
| 3 | `stage-0.0a: 修复 CTX-HARD fixture 超长并完成 train-v5 SFT 基线` | `training/build_train_v5.py`、CTX token 审计、补数据决策、训练管线验证 10.7、train-v5 SHA 与 SFT 结果 | `stage-0.0a-train-v5` |

## 6. 提交后校验

- 提交前：`pytest -q -k "not mcp"`、`ruff check .`、`ruff format --check .` 全绿（见下方回填）。
- tag 校验：`git tag --list` 与 `git show --stat <tag>`（见下方回填）。
- 工作区预期：仅剩 `private/` 等被忽略文件；`git status` 干净。

## 7. 后续规则

已写入 `docs/plan/训练侧执行计划.md` 第 6 节：

1. 每阶段验收后立即 commit + tag，不攒到最后；
2. commit message 写“做了什么 + 验收结论”；
3. 大文件不进 Git，用 manifest + SHA-256 记录；
4. 关键结果文档必须进 Git；
5. 提交前测试与 lint 全绿，不提交密钥与本机绝对路径；
6. 不 force push、不改写已推送历史。
